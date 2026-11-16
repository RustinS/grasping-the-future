"""GTF training: LightningModule and LightningDataModule.

GTFLitModule runs one step as
    batch -> observation mask -> present-camera re-anchoring -> global SE(3) augmentation -> centering
    -> delta-codec encoding -> flow noise -> forward -> losses -> logging.
GTFDataModule wraps GTFDataset for the train and val splits.

Usage:
    from gtf.training.trainer import GTFLitModule, GTFDataModule
    lit = GTFLitModule(cfg)
    dm = GTFDataModule(cfg)
    pl.Trainer(max_steps=40000, gradient_clip_val=1.0, gradient_clip_algorithm="norm").fit(lit, dm)
"""

import dataclasses
import logging

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import lightning.pytorch as pl

from gtf.paths import MANO_DIR
from gtf.config import GTFConfig
from gtf.data.augmentation import apply_global_se3_augmentation
from gtf.data.compute_stats import load_stats, validate_stats_provenance
from gtf.eval.checkpoint import file_digest
from gtf.data.confidence import neutralize_confidence_leak
from gtf.data.curriculum import PiecewiseSchedule, curriculum_mass_report
from gtf.data.dataset import GTFDataset, load_observed_scene_features, gtf_collate
from gtf.data.sampler import StepScheduledDistributedWeightedSampler, compute_window_weights
from gtf.data.uni3d_cache import runtime_uni3d_dataset_kwargs
from gtf.model.flow.losses import combined_loss, ddp_window_mean, loss_attachment, loss_energy_score, loss_fm_per_sample
from gtf.model.flow.masking import sample_mask_batched
from gtf.model.flow.matching import TOKEN_TYPES, apply_flow_noise, estimate_x0, sample_timestep
from gtf.model.gtf_model import GTFModel
from gtf.model.repr.delta_codec import STATE_SLOTS, DeltaCodec, _run_starts, build_anchor_feats
from gtf.model.repr.hand_anatomy import anatomy_violation_terms, excess_over_reference, mano_rest_bone_axes
from gtf.model.repr.mano_pose_prior import MANONaturalPosePrior
from gtf.model.repr.present_anchor import reanchor_tokens_to_present_camera
from gtf.model.repr.relation import _fingertip_surface_distances
from gtf.model.repr.rotation import rotation_6d_to_matrix
from gtf.model.repr.timing import physical_finite_differences
from gtf.model.repr.tokenizer import (
    HAND_CAM_T_SLICE,
    HAND_ORIENT_SLICE,
    HAND_POSE_SLICE,
    OBJ_ROTATION_SLICE,
    OBJ_TRANSLATION_SLICE,
    center_tokens_at_present,
    center_translations,
)
from gtf.training.loss_schedule import LossRampConfig, LossSchedule


def build_lr_scheduler(optimizer, tc):
    """Linear warmup, then cosine annealing to lr * lr_min_ratio at max_steps. Stepped every optimizer step."""
    warmup_steps = int(tc.warmup_steps)
    max_steps = int(tc.max_steps)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps - warmup_steps, eta_min=tc.lr * float(tc.lr_min_ratio))
    return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


def rotation_error_6d(x_hat: Tensor, x_clean: Tensor) -> Tensor:
    """Per-element rotation error theta^2 / 2 between two 6D rotations.

    theta = atan2(|skew(R_rel)|, cos theta): the gradient does not saturate as theta approaches pi, and the value
    matches 1 - cos(theta) at small angles. At theta = pi exactly the axis is undefined and the gradient is zero.
    """
    R_rel = rotation_6d_to_matrix(x_hat).transpose(-1, -2) @ rotation_6d_to_matrix(x_clean)
    cos = ((R_rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    v = (
        torch.stack(
            (R_rel[..., 2, 1] - R_rel[..., 1, 2], R_rel[..., 0, 2] - R_rel[..., 2, 0], R_rel[..., 1, 0] - R_rel[..., 0, 1]),
            dim=-1,
        )
        * 0.5
    )
    theta = torch.atan2(v.norm(dim=-1), cos)
    return 0.5 * theta.pow(2)


def _decoded_contact_triangle_distances(
    records: list[tuple[Tensor, Tensor, str, Tensor]],
    obj_canonical_verts: Tensor,
    obj_scale: Tensor,
    obj_faces: Tensor,
    mano_helper,
) -> list[Tensor]:
    """Decoded fingertip-to-object-surface distances (relation dims 9:14) for one window.

    ``records`` holds ``(hand_state, betas, side, obj_state)`` tuples; results come back in the same order.
    Each record runs its own MANO forward pass; the fingertips of all records, expressed in the object frame,
    share one point-to-triangle query against the window's static mesh.
    """
    if not records:
        return []

    tips_canon: list[Tensor] = []
    frame_counts: list[int] = []
    for hand_state, betas, side, obj_state in records:
        n_frames = int(hand_state.shape[0])
        betas_expanded = betas.unsqueeze(0).expand(n_frames, -1)
        _, _, fingertips = mano_helper(
            hand_state[:, 0:6],
            hand_state[:, 9:99],
            betas_expanded,
            hand_state[:, 6:9],
            side=side,
        )
        obj_R_inv = rotation_6d_to_matrix(obj_state[:, 0:6]).transpose(-1, -2)
        tips_canon.append(
            torch.einsum(
                "tij,tfj->tfi",
                obj_R_inv,
                fingertips - obj_state[:, 6:9].unsqueeze(-2),
            )
        )
        frame_counts.append(n_frames)

    distances = _fingertip_surface_distances(
        torch.cat(tips_canon, dim=0),
        obj_canonical_verts * obj_scale,
        obj_faces,
    )
    return list(torch.split(distances, frame_counts, dim=0))


class GTFLitModule(pl.LightningModule):
    """PyTorch Lightning module for GTF training.

    Args:
        cfg: Full GTFConfig (dataclass, DictConfig or plain dict).
        mano_root: Path to MANO model files.
        stats_path: Override for cfg.data.stats_path, or None to use the config.
        stats: The statistics themselves (an exported checkpoint carries them); overrides both paths.
    """

    def __init__(
        self,
        cfg: GTFConfig,
        mano_root: str = MANO_DIR,
        stats_path: str | None = None,
        stats: dict | None = None,
    ) -> None:
        super().__init__()
        # save_hyperparameters() stores a plain container: torch.load(weights_only=True) rejects OmegaConf and
        # dataclass objects.
        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=True)
        elif dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
            cfg = dataclasses.asdict(cfg)
        self.save_hyperparameters(ignore=["stats"])
        self.cfg = OmegaConf.create(cfg)

        self.model = GTFModel(self.cfg, mano_root=mano_root)

        # Normalization stats (CPU here, moved to the training device on first use). The digest of the statistics
        # file is written into every checkpoint: evaluation and resuming refuse a different file.
        if stats is None:
            stats_file = str(stats_path or self.cfg.data.stats_path)
            self.stats = load_stats(stats_file)
            self._stats_content_digest = file_digest(stats_file)
        else:
            self.stats = stats
            self._stats_content_digest = None
        self._stats_on_device = False
        validate_stats_provenance(self.stats, strict=True)

        # Masking regime probabilities, in config order.
        self.regime_probs = {str(name): float(p) for name, p in self.cfg.masking.regimes.items()}

        lc = self.cfg.loss
        self.lambda_aux_contact = float(lc.lambda_aux_contact)
        self.fm_component_weights = {str(k): float(v) for k, v in dict(lc.fm_component_weights).items()}
        self.run_seed = int(self.cfg.training.seed)

        # Energy score on sampled rollouts, gated per step by a seeded Bernoulli draw (identical on every rank).
        self.lambda_energy_score = float(lc.lambda_energy_score)
        self.es_samples = int(lc.es_samples)
        self.es_steps = int(lc.es_steps)
        self.es_p = float(lc.es_p)
        self.es_rel_weight = float(lc.es_rel_weight)
        self.es_path_points = int(lc.es_path_points)
        self.es_dense_weight = float(lc.es_dense_weight)
        self.es_rot_weight = float(lc.es_rot_weight)

        # Support-surface penetration penalty on the energy-score samples (per-source gated).
        self.lambda_support_pen = float(lc.lambda_support_pen)
        self.support_pen_margin_mm = float(lc.support_pen_margin_mm)
        self.support_pen_band_mm = float(lc.support_pen_band_mm)
        self.support_pen_topk = int(lc.support_pen_topk)
        self.support_pen_sources = {str(s) for s in (lc.support_pen_sources or [])}

        # Losses on the decoded (codec-integrated) x0 estimate.
        self.lambda_x0_pos = float(lc.lambda_x0_pos)
        self.lambda_x0_rot = float(lc.lambda_x0_rot)
        self.lambda_x0_artic = float(lc.lambda_x0_artic)
        self.lambda_smooth_decoded = float(lc.lambda_smooth_decoded)
        self.lambda_anatomy = float(lc.lambda_anatomy)
        self.anatomy_twist_mcp_deg = float(lc.anatomy_twist_mcp_deg)
        self.anatomy_twist_pip_deg = float(lc.anatomy_twist_pip_deg)
        self.anatomy_swing_deg = float(lc.anatomy_swing_deg)
        self.lambda_mano_prior = float(lc.lambda_mano_prior)
        self.mano_prior_reference_floor = float(lc.mano_prior_reference_floor)
        self.mano_pose_prior = None
        if self.lambda_mano_prior > 0.0:
            # Reuses the model's MANO asset: registers small constant buffers, no second FK model.
            self.mano_pose_prior = MANONaturalPosePrior.from_mano_helper(self.model.mano_helper)
        self.lambda_dcontact = float(lc.lambda_dcontact)
        self.dcontact_max_frames = int(lc.dcontact_max_frames)
        self.dcontact_clamp_m = float(lc.dcontact_clamp_m)
        self.lambda_attach_pos = float(lc.lambda_attach_pos)
        self.lambda_attach_rot = float(lc.lambda_attach_rot)
        self.lambda_attach_vel = float(lc.lambda_attach_vel)
        self.attach_contact_thr = float(lc.attach_contact_thr)
        self.attach_max_frames = int(lc.attach_max_frames)
        self.attach_min_run = int(lc.attach_min_run)

        # Delta codec at the flow boundary: run-start absolutes plus per-second derivatives.
        self.delta_codec = DeltaCodec(self.stats)

        # Per-loss linear ramps (loss_schedule) over the weights passed to combined_loss.
        static_lambdas = {"lambda_aux_contact": self.lambda_aux_contact}
        ramps: dict[str, LossRampConfig] = {}
        for name, ramp in dict(self.cfg.loss_schedule or {}).items():
            if str(name) not in static_lambdas:
                raise ValueError(f"loss_schedule entry {name!r} is not a scheduled weight (supported: {sorted(static_lambdas)})")
            ramp_dict = dict(ramp)
            ramps[str(name)] = LossRampConfig(
                start_step=int(ramp_dict.get("start_step", 0)),
                end_step=int(ramp_dict.get("end_step", 0)),
                start_val=float(ramp_dict.get("start_val", 0.0)),
                end_val=float(ramp_dict.get("end_val", 0.0)),
            )
        self.loss_schedule_fn = LossSchedule(ramps, static_lambdas)

        # Global SE(3) augmentation: one yaw about gravity plus one translation per window, applied to hands,
        # object, scene and support plane alike (contacts and contact labels stay exact).
        tc = self.cfg.training
        self.aug_global_prob = float(tc.aug_global_prob)
        self.aug_global_yaw_sigma = float(tc.aug_global_yaw_sigma)
        self.aug_global_trans_sigma = float(tc.aug_global_trans_sigma)

    def _seed_global_rng(self, step: int) -> None:
        """Seed the global torch RNG per (seed, rank, step): ranks draw independent masks, flow times, noise,
        augmentations and dropout, and a resume continues on a fresh deterministic stream."""
        rank = int(getattr(self.trainer, "global_rank", 0) or 0)
        seed = self.run_seed * 9176 + rank * 7919 + step * 257 + 1
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["stats_content_digest"] = self._stats_content_digest

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        saved = checkpoint.get("stats_content_digest")
        if saved is not None and saved != self._stats_content_digest:
            raise RuntimeError(f"normalization statistics differ from the checkpoint's (sha256 {self._stats_content_digest} vs {saved}): resume with the run's own statistics")

    def on_fit_start(self) -> None:
        self._move_stats_to_device()
        # Model init stays common to all ranks (DDP broadcasts rank-0 weights); re-key afterwards.
        self._seed_global_rng(step=0)

    def on_train_start(self) -> None:
        # Runs after Lightning restores the loop state; checkpoints carry no torch RNG state, so re-key with the
        # restored step (step 0 reproduces the on_fit_start seed).
        self._seed_global_rng(step=int(getattr(self.trainer, "global_step", 0) or 0))

    def _move_stats_to_device(self) -> None:
        if self._stats_on_device:
            return
        device = self.device
        for top_key in ("mean", "std", "velocity_std", "acceleration_std"):
            if top_key in self.stats:
                self.stats[top_key] = {k: v.to(device) for k, v in self.stats[top_key].items()}
        self._stats_on_device = True

    def _shared_step(self, batch: dict) -> tuple[Tensor, dict[str, Tensor]]:
        """Run the training pipeline on a batch.

        Returns:
            (total_loss, loss_dict) where loss_dict maps component names to detached scalars. In eval mode the
            per-source validation macro inputs are left in ``_val_fm_per_sample`` and ``_val_aux_parts``.
        """
        self._move_stats_to_device()
        presence_mask = batch["presence_mask"]
        valid_frames = batch["valid_frames"]
        tokens_dict = {name: batch[name] for name in TOKEN_TYPES}
        B = presence_mask.shape[0]

        # 1. Observation masks (one masking regime per window).
        obs_mask = sample_mask_batched(presence_mask, valid_frames, self.regime_probs)

        # 2. Re-anchor moving-camera samples to the present frame. Scene content comes from the last fully observed
        # frame, re-expressed in the present frame.
        tokens_dict, present_frames = reanchor_tokens_to_present_camera(tokens_dict, batch, obs_mask, presence_mask, valid_frames)
        batch["scene_features"] = load_observed_scene_features(batch, present_frames, obs_mask, presence_mask, valid_frames)

        # Confidence features 0/1 are computed from the GT trajectory around each frame (and across the hand-object
        # pair), so they are set to neutral wherever their receptive field touches a masked slot.
        if batch.get("confidence_features") is not None:
            batch["confidence_features"] = neutralize_confidence_leak(
                batch["confidence_features"],
                obs_mask,
                presence_mask=presence_mask,
                valid_frames=valid_frames,
                frame_times_s=batch.get("frame_times_s"),
            )

        # 3. Global SE(3) augmentation on uncentered tokens. Left-hand cam_t lives in the mirrored MANO frame, so
        # geometry formed after centering would displace the left hand by 2 * origin_x; uncentered, the transform is
        # an exact isometry. Runs in fp32: bf16 autocast would perturb the recomputed fingertip distances by millimeters.
        if self.training and self.aug_global_prob > 0.0:
            _, pivot = center_tokens_at_present(tokens_dict, obs_mask, presence_mask, present_frames)
            with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu", enabled=False):
                tokens_dict, aug_scene, aug_support_y = apply_global_se3_augmentation(
                    tokens_dict,
                    presence_mask=presence_mask,
                    valid_frames=valid_frames,
                    betas_left=batch["betas_left"],
                    betas_right=batch["betas_right"],
                    obj_canonical_verts_list=batch["obj_canonical_verts"],
                    obj_scale=batch["obj_scale"],
                    mano_helper=self.model.mano_helper,
                    scene_features=batch.get("scene_features"),
                    centering_origin=pivot,
                    support_plane_y=batch.get("support_plane_y"),
                    has_support_plane=batch.get("has_support_plane"),
                    prob=self.aug_global_prob,
                    yaw_sigma=self.aug_global_yaw_sigma,
                    trans_sigma=self.aug_global_trans_sigma,
                    obj_faces_list=batch["obj_faces"],
                )
            if aug_scene is not None:
                batch["scene_features"] = aug_scene
            if aug_support_y is not None:
                batch["support_plane_y"] = aug_support_y

        # 4. Center at the present frame (one shared physical frame; the left hand subtracts M * origin).
        centered_tokens, centering_origin = center_tokens_at_present(tokens_dict, obs_mask, presence_mask, present_frames)

        # 5. Encode into the delta-codec space with the exact per-transition durations. The observation mask is
        # transformed alongside (an interior derivative slot is observed iff both of its frames are).
        obs_mask_sem = obs_mask  # semantic (pre-codec) observation mask
        physical_dt = batch["edge_dt_s"].to(device=obs_mask.device, dtype=torch.float32)
        if physical_dt.shape != (B, obs_mask.shape[1] - 1):
            raise RuntimeError(f"edge_dt_s must be (B, T-1)={(B, obs_mask.shape[1] - 1)}, got {tuple(physical_dt.shape)}")
        tokens_norm = self.delta_codec.encode(centered_tokens, presence_mask, physical_dt)
        obs_mask = self.delta_codec.transform_masks(obs_mask, presence_mask)

        # 6. Flow time and noise.
        t = sample_timestep(B, device=presence_mask.device)
        x_t, v_target = apply_flow_noise(tokens_norm, obs_mask, presence_mask, t)

        # 7. Forward pass.
        conditioning = {
            "obj_embedding_bf16": batch.get("obj_embedding_bf16"),
            "obj_embedding_fp32": batch.get("obj_embedding_fp32"),
            "betas_left": batch["betas_left"],
            "betas_right": batch["betas_right"],
            "obj_scale": batch["obj_scale"],
            "confidence_features": batch.get("confidence_features"),
            # list[dict | None] of per-segment scene features; pack_scene_batch moves each tensor to the device.
            "scene_features": batch.get("scene_features"),
        }
        dt = physical_dt.to(device=t.device, dtype=t.dtype)
        # Observed absolute positions (at semantically observed frames) as an input side channel.
        anchor_feats = build_anchor_feats(centered_tokens, obs_mask_sem, self.stats)
        output = self.model(x_t, t, obs_mask, presence_mask, conditioning, origin=centering_origin, dt=dt, anchor_feats=anchor_feats)

        # 8. Loss mask (masked, present, real frame) and x0 estimate.
        loss_mask = ~obs_mask & presence_mask & valid_frames.unsqueeze(-1)
        x0_hat = {name: estimate_x0(x_t[name], output["v_pred"][name], t) for name in TOKEN_TYPES}

        # 9. Energy score on M sampled rollouts (training only).
        es_loss = None
        support_pen_loss = None
        if self.training and self.lambda_energy_score > 0.0:
            gen_es = torch.Generator(device="cpu")
            gen_es.manual_seed(int(self.global_step) + self.run_seed + 7919)
            if bool(torch.rand((), generator=gen_es).item() < self.es_p):
                support_ctx = None
                if self.lambda_support_pen > 0.0:
                    support_ctx = {
                        "scene_list": batch.get("scene_features"),
                        "obj_verts_list": batch["obj_canonical_verts"],
                        "obj_scale": batch["obj_scale"],
                        "source_list": batch.get("source_dataset"),
                        "allow_sources": self.support_pen_sources,
                        "margin_m": self.support_pen_margin_mm / 1000.0,
                        "band_r": self.support_pen_band_mm / 1000.0,
                        "topk": self.support_pen_topk,
                    }
                es_out = loss_energy_score(
                    self.model,
                    tokens_norm,
                    obs_mask,
                    presence_mask,
                    valid_frames,
                    conditioning,
                    present_frames=present_frames,
                    M=self.es_samples,
                    K=self.es_steps,
                    origin=centering_origin,
                    phys_dt=dt,
                    codec=self.delta_codec,
                    codec_dt=physical_dt,
                    es_rel_weight=self.es_rel_weight,
                    es_path_points=self.es_path_points,
                    es_dense_weight=self.es_dense_weight,
                    es_rot_weight=self.es_rot_weight,
                    anchor_feats=anchor_feats,
                    support_pen=support_ctx,
                )
                if support_ctx is not None:
                    es_loss, support_pen_loss = es_out
                else:
                    es_loss = es_out

        # 10. Flow-matching loss and the auxiliary contact head (scheduled weight).
        lambdas = self.loss_schedule_fn(int(self.global_step))
        total_loss, loss_dict = combined_loss(
            v_pred=output["v_pred"],
            v_target=v_target,
            loss_mask=loss_mask,
            presence_mask=presence_mask,
            valid_frames=valid_frames,
            contact_labels=batch["contact_labels"],
            contact_logits=output["contact_logits"],
            phase_logits=output["phase_logits"],
            lambda_aux_contact=lambdas["lambda_aux_contact"],
            fm_weights=self.fm_component_weights,
        )

        # 11. Losses on the decoded trajectory. The FM loss is a per-frame velocity MSE and is blind to integrated
        # drift, so x0_hat (observed slots held clean) is integrated through the codec and compared with the clean
        # centered states. Each per-window term is reduced as a DDP-exact mean over windows with mass.
        loss_x0_pos = None
        loss_x0_rot = None
        loss_x0_artic = None
        loss_anatomy = None
        loss_mano_prior = None
        loss_smooth_dec = None
        loss_dcontact = None
        loss_attach_terms: dict[str, Tensor] = {}
        # Validation macro: lambda-weighted per-window component values with their applicability masks.
        macro_parts: dict[str, tuple[Tensor, Tensor]] = {}
        want_macro = not self.training

        def _macro_add(comp: str, lam: float, l_vec: Tensor, n_vec: Tensor) -> None:
            if want_macro:
                with torch.no_grad():
                    macro_parts[comp] = (float(lam) * (l_vec / n_vec.clamp(min=1.0)), n_vec > 0)

        def _aux_reduce(l_vec: Tensor, n_vec: Tensor) -> Tensor:
            return ddp_window_mean(l_vec / n_vec.clamp(min=1.0), n_vec > 0)

        need_decoded = any(
            lam > 0.0
            for lam in (
                self.lambda_x0_pos,
                self.lambda_x0_rot,
                self.lambda_x0_artic,
                self.lambda_anatomy,
                self.lambda_mano_prior,
                self.lambda_smooth_decoded,
                self.lambda_dcontact,
                self.lambda_attach_pos,
                self.lambda_attach_rot,
                self.lambda_attach_vel,
            )
        )
        if need_decoded:
            x0_int = {name: torch.where(obs_mask[:, :, k].unsqueeze(-1), tokens_norm[name], x0_hat[name]) for name, k in STATE_SLOTS.items()}
            dec_x0 = self.delta_codec.decode_state_fast(x0_int, presence_mask, physical_dt)
            amp_dev = "cuda" if obs_mask.device.type == "cuda" else "cpu"

            # Supervised frames per entity: within each presence run, every frame at or after the first
            # codec-masked slot (its decoded value depends on a predicted derivative, including the observed suffix
            # after a gap). Fully observed prefixes decode exactly and are excluded.
            x0_gate: dict[str, Tensor] = {}
            with torch.no_grad():
                for name, k in STATE_SLOTS.items():
                    pres_k = presence_mask[:, :, k]
                    cm = (~obs_mask[:, :, k]) & pres_k
                    rs_k = _run_starts(pres_k)
                    dep = torch.zeros_like(cm)
                    prev = torch.zeros_like(cm[:, 0])
                    for _t in range(cm.shape[1]):
                        prev = cm[:, _t] | (prev & ~rs_k[:, _t])
                        dep[:, _t] = prev
                    x0_gate[name] = dep & pres_k & valid_frames

            if self.lambda_x0_pos > 0.0:
                l_x0 = torch.zeros(B, device=obs_mask.device, dtype=torch.float32)
                n_x0 = torch.zeros(B, device=obs_mask.device, dtype=torch.float32)
                for name, k in STATE_SLOTS.items():
                    sl = OBJ_TRANSLATION_SLICE if name == "obj_state" else HAND_CAM_T_SLICE
                    m = x0_gate[name].unsqueeze(-1).float()
                    err2 = (dec_x0[name][:, :, sl] - centered_tokens[name][:, :, sl].detach()).pow(2) * m
                    l_x0 = l_x0 + err2.sum(dim=(1, 2))
                    n_x0 = n_x0 + m.sum(dim=(1, 2)) * 3.0
                loss_x0_pos = _aux_reduce(l_x0, n_x0)
                _macro_add("x0_pos", self.lambda_x0_pos, l_x0, n_x0)

            # Geodesic rotation error of the integrated wrist and object rotations (fp32: under bf16 the
            # orthonormalization loses the small-angle signal).
            if self.lambda_x0_rot > 0.0:
                with torch.amp.autocast(amp_dev, enabled=False):
                    l_rot = torch.zeros(B, device=obs_mask.device, dtype=torch.float32)
                    n_rot = torch.zeros(B, device=obs_mask.device, dtype=torch.float32)
                    for name, k in STATE_SLOTS.items():
                        sl = OBJ_ROTATION_SLICE if name == "obj_state" else HAND_ORIENT_SLICE
                        err = rotation_error_6d(dec_x0[name][:, :, sl].float(), centered_tokens[name][:, :, sl].detach().float())
                        m = x0_gate[name].float()
                        l_rot = l_rot + (err * m).sum(dim=1)
                        n_rot = n_rot + m.sum(dim=1)
                    loss_x0_rot = _aux_reduce(l_rot, n_rot)
                    _macro_add("x0_rot", self.lambda_x0_rot, l_rot, n_rot)

            # Same geodesic error per finger joint (15 x 6D per hand).
            if self.lambda_x0_artic > 0.0:
                with torch.amp.autocast(amp_dev, enabled=False):
                    l_art = torch.zeros(B, device=obs_mask.device, dtype=torch.float32)
                    n_art = torch.zeros(B, device=obs_mask.device, dtype=torch.float32)
                    for name in ("left_state", "right_state"):
                        p_hat = dec_x0[name][:, :, HAND_POSE_SLICE].float()
                        p_clean = centered_tokens[name][:, :, HAND_POSE_SLICE].detach().float()
                        Bh, Th = p_hat.shape[:2]
                        err = rotation_error_6d(p_hat.reshape(Bh, Th, 15, 6), p_clean.reshape(Bh, Th, 15, 6))
                        m = x0_gate[name].float().unsqueeze(-1)
                        l_art = l_art + (err * m).sum(dim=(1, 2))
                        n_art = n_art + m.sum(dim=(1, 2)) * 15.0
                    loss_x0_artic = _aux_reduce(l_art, n_art)
                    _macro_add("x0_artic", self.lambda_x0_artic, l_art, n_art)

            # Joint-limit safety: penalize only swing/twist excess above the clean trajectory, so GT stays a
            # stationary point even for fitted sources that cross the heuristic limits.
            if self.lambda_anatomy > 0.0:
                with torch.amp.autocast(amp_dev, enabled=False):
                    l_anat = torch.zeros(B, device=obs_mask.device, dtype=torch.float32)
                    n_anat = torch.zeros(B, device=obs_mask.device, dtype=torch.float32)
                    for name, beta_key in (("left_state", "betas_left"), ("right_state", "betas_right")):
                        p_hat = dec_x0[name][:, :, HAND_POSE_SLICE].float().reshape(B, -1, 15, 6)
                        p_clean = centered_tokens[name][:, :, HAND_POSE_SLICE].detach().float().reshape(B, -1, 15, 6)
                        axes = mano_rest_bone_axes(self.model.mano_helper.mano_layer, batch[beta_key].float()).unsqueeze(1)
                        pred_terms = anatomy_violation_terms(
                            rotation_6d_to_matrix(p_hat),
                            axes,
                            twist_mcp_deg=self.anatomy_twist_mcp_deg,
                            twist_pip_deg=self.anatomy_twist_pip_deg,
                            swing_deg=self.anatomy_swing_deg,
                        )
                        clean_terms = anatomy_violation_terms(
                            rotation_6d_to_matrix(p_clean),
                            axes,
                            twist_mcp_deg=self.anatomy_twist_mcp_deg,
                            twist_pip_deg=self.anatomy_twist_pip_deg,
                            swing_deg=self.anatomy_swing_deg,
                        )
                        excess = excess_over_reference(pred_terms["energy"], clean_terms["energy"])
                        m = x0_gate[name].float().unsqueeze(-1)
                        l_anat = l_anat + (excess * m).sum(dim=(1, 2))
                        n_anat = n_anat + m.sum(dim=(1, 2)) * 15.0
                    loss_anatomy = _aux_reduce(l_anat, n_anat)
                    _macro_add("anatomy", self.lambda_anatomy, l_anat, n_anat)

            # MANO natural-pose prior: standardized PCA energy introduced above the clean pose, per component.
            if self.lambda_mano_prior > 0.0:
                l_prior = torch.zeros(B, device=obs_mask.device, dtype=torch.float32)
                n_prior = torch.zeros(B, device=obs_mask.device, dtype=torch.float32)
                for name, side in (("left_state", "left"), ("right_state", "right")):
                    p_hat = dec_x0[name][:, :, HAND_POSE_SLICE]
                    p_clean = centered_tokens[name][:, :, HAND_POSE_SLICE].detach()
                    excess = self.mano_pose_prior.excess(
                        p_hat,
                        p_clean,
                        representation="rotation_6d",
                        side=side,
                        reduction="mean",
                        reference_floor=self.mano_prior_reference_floor,
                    )
                    m = x0_gate[name].to(dtype=excess.dtype)
                    l_prior = l_prior + (excess * m).sum(dim=1)
                    n_prior = n_prior + m.sum(dim=1)
                loss_mano_prior = _aux_reduce(l_prior, n_prior)
                _macro_add("mano_prior", self.lambda_mano_prior, l_prior, n_prior)

            # Decoded-geometry contact: FK fingertips of the decoded (uncentered) states against the posed object
            # mesh, L1 to the clean relation dims 9:14 on fingers whose GT distance is within dcontact_clamp_m.
            # Frames: the relation slot is masked, or the hand or object frame is decode-dependent.
            if self.lambda_dcontact > 0.0:
                dec_unc = center_translations({name: dec_x0[name] for name in STATE_SLOTS}, -centering_origin)
                with torch.amp.autocast(amp_dev, enabled=False):
                    l_dc = torch.zeros(B, device=obs_mask.device, dtype=torch.float32)
                    n_dc = torch.zeros(B, device=obs_mask.device, dtype=torch.float32)
                    for b in range(B):
                        records = []
                        targets = []
                        for side, k in (("left", 0), ("right", 1)):
                            pair_ok = presence_mask[b, :, k] & presence_mask[b, :, 2] & valid_frames[b]
                            m_b = ~obs_mask_sem[b, :, 3 + k] & pair_ok
                            m_b = m_b | ((x0_gate[f"{side}_state"][b] | x0_gate["obj_state"][b]) & pair_ok)
                            fr = m_b.nonzero(as_tuple=True)[0]
                            if fr.numel() == 0:
                                continue
                            if self.dcontact_max_frames > 0 and fr.numel() > self.dcontact_max_frames:
                                # Always keep the last eligible frame (the goal/suffix anchor); draw the rest uniformly.
                                last = fr[-1:]
                                rest = fr[:-1]
                                rest = rest[torch.randperm(rest.numel(), device=fr.device)[: self.dcontact_max_frames - 1]]
                                fr, _ = torch.sort(torch.cat([rest, last]))
                            # Clamp by selection, not by value: clamping the prediction would zero the gradient in the
                            # pred-too-far regime.
                            d_gt = centered_tokens[f"{side}_rel"][b, fr, 9:14].detach().float()
                            near = d_gt <= self.dcontact_clamp_m
                            n_near = torch.count_nonzero(near)
                            if not bool(n_near):
                                continue
                            st = dec_unc[f"{side}_state"][b, fr].float()
                            ob = dec_unc["obj_state"][b, fr].float()
                            records.append((st, batch[f"betas_{side}"][b].float(), side, ob))
                            targets.append((d_gt, near, n_near))
                        if records:
                            decoded_distances = _decoded_contact_triangle_distances(
                                records,
                                batch["obj_canonical_verts"][b].float(),
                                batch["obj_scale"][b].float(),
                                batch["obj_faces"][b],
                                self.model.mano_helper,
                            )
                            for d_dec, (d_gt, near, n_near) in zip(decoded_distances, targets, strict=True):
                                l_dc[b] = l_dc[b] + ((d_dec - d_gt).abs() * near.float()).sum()
                                n_dc[b] = n_dc[b] + n_near.to(dtype=n_dc.dtype)
                    loss_dcontact = _aux_reduce(l_dc, n_dc)
                    _macro_add("dcontact", self.lambda_dcontact, l_dc, n_dc)

            # Decoded object-frame attachment over contiguous GT-contact runs: wrist/fingertip position, rotation
            # and velocity of the hand expressed in the object frame. Clean states go through the same inverse
            # centering (MANO mirrors the left FK output, so mixed conventions would add a 2 * origin_x offset).
            if self.lambda_attach_pos > 0.0 or self.lambda_attach_rot > 0.0 or self.lambda_attach_vel > 0.0:
                dec_unc_a = center_translations({name: dec_x0[name] for name in STATE_SLOTS}, -centering_origin)
                clean_unc_a = center_translations({name: centered_tokens[name] for name in STATE_SLOTS}, -centering_origin)
                with torch.amp.autocast(amp_dev, enabled=False):
                    attach = loss_attachment(
                        dec_states={k: v.float() for k, v in dec_unc_a.items()},
                        clean_states={k: clean_unc_a[k].detach().float() for k in STATE_SLOTS},
                        clean_rel={s: centered_tokens[f"{s}_rel"].detach().float() for s in ("left", "right")},
                        sup_gate=x0_gate,
                        presence_mask=presence_mask,
                        valid_frames=valid_frames,
                        betas={"left": batch["betas_left"], "right": batch["betas_right"]},
                        mano_helper=self.model.mano_helper,
                        edge_dt=physical_dt,
                        contact_thr=self.attach_contact_thr,
                        max_frames=self.attach_max_frames,
                        min_run=self.attach_min_run,
                    )
                for comp, lam in (("attach_pos", self.lambda_attach_pos), ("attach_rot", self.lambda_attach_rot), ("attach_vel", self.lambda_attach_vel)):
                    if lam <= 0.0:
                        continue
                    l_vec, n_vec = attach[comp]
                    loss_attach_terms[comp] = _aux_reduce(l_vec, n_vec)
                    _macro_add(comp, lam, l_vec, n_vec)

            # Smoothness of the decoded trajectory in SI units: acceleration^2 + jerk^2 / 30 on fully masked stencils
            # (boundary drift is owned by the x0 losses), counted only where it exceeds the clean trajectory's own
            # value per stencil. The detached > 0 mask gives exactly zero gradient at pred == clean.
            if self.lambda_smooth_decoded > 0.0:
                l_acc_v = torch.zeros(B, device=obs_mask.device, dtype=torch.float32)
                n_acc_v = torch.zeros(B, device=obs_mask.device, dtype=torch.float32)
                l_jrk_v = torch.zeros(B, device=obs_mask.device, dtype=torch.float32)
                n_jrk_v = torch.zeros(B, device=obs_mask.device, dtype=torch.float32)
                for name, k in STATE_SLOTS.items():
                    sl = OBJ_TRANSLATION_SLICE if name == "obj_state" else HAND_CAM_T_SLICE
                    obs_sem = obs_mask_sem[:, :, k].unsqueeze(-1)
                    clean_pos = centered_tokens[name][:, :, sl].detach()
                    pos = torch.where(obs_sem, clean_pos, dec_x0[name][:, :, sl])  # (B, T, 3) meters
                    ok = presence_mask[:, :, k] & valid_frames
                    msk = ~obs_mask_sem[:, :, k]
                    _vel, acc, jerk = physical_finite_differences(pos, physical_dt, validate=False)
                    ok2 = ok[:, 1:] & ok[:, :-1]
                    m2 = msk[:, 1:] & msk[:, :-1]
                    acc_gate = m2[:, 1:] & m2[:, :-1]
                    acc_m = (ok2[:, 1:] & ok2[:, :-1] & acc_gate).float()
                    jerk_m = (acc_m[:, 1:].bool() & acc_m[:, :-1].bool()).float()
                    with torch.no_grad():
                        _, acc_c, jerk_c = physical_finite_differences(clean_pos, physical_dt, validate=False)
                    acc_ex = acc.pow(2).mean(-1) - acc_c.pow(2).mean(-1)
                    acc_term = acc_ex * (acc_ex.detach() > 0).to(acc_ex.dtype)
                    jrk_ex = jerk.pow(2).mean(-1) - jerk_c.pow(2).mean(-1)
                    jrk_term = jrk_ex * (jrk_ex.detach() > 0).to(jrk_ex.dtype)
                    l_acc_v = l_acc_v + (acc_term * acc_m).sum(dim=1)
                    n_acc_v = n_acc_v + acc_m.sum(dim=1)
                    l_jrk_v = l_jrk_v + (jrk_term * jerk_m).sum(dim=1)
                    n_jrk_v = n_jrk_v + jerk_m.sum(dim=1)
                # Separate normalizers (m^2/s^4 vs m^2/s^6); 1/30 is the jerk term's relative weight.
                sm_vec = l_acc_v / n_acc_v.clamp(min=1.0) + (1.0 / 30.0) * (l_jrk_v / n_jrk_v.clamp(min=1.0))
                has_sm = (n_acc_v + n_jrk_v) > 0
                loss_smooth_dec = ddp_window_mean(sm_vec, has_sm)
                if want_macro:
                    with torch.no_grad():
                        macro_parts["smooth_dec"] = (self.lambda_smooth_decoded * sm_vec, has_sm)

        # 12. Total objective.
        if es_loss is not None:
            total_loss = total_loss + self.lambda_energy_score * es_loss
            loss_dict["loss_energy_score"] = es_loss.detach()
        if support_pen_loss is not None:
            total_loss = total_loss + self.lambda_support_pen * support_pen_loss
            loss_dict["loss_support_pen"] = support_pen_loss.detach()
        if loss_x0_pos is not None:
            total_loss = total_loss + self.lambda_x0_pos * loss_x0_pos
            loss_dict["loss_x0_pos"] = loss_x0_pos.detach()
        if loss_x0_rot is not None:
            total_loss = total_loss + self.lambda_x0_rot * loss_x0_rot
            loss_dict["loss_x0_rot"] = loss_x0_rot.detach()
        if loss_x0_artic is not None:
            total_loss = total_loss + self.lambda_x0_artic * loss_x0_artic
            loss_dict["loss_x0_artic"] = loss_x0_artic.detach()
        if loss_anatomy is not None:
            total_loss = total_loss + self.lambda_anatomy * loss_anatomy
            loss_dict["loss_anatomy"] = loss_anatomy.detach()
        if loss_mano_prior is not None:
            total_loss = total_loss + self.lambda_mano_prior * loss_mano_prior
            loss_dict["loss_mano_prior"] = loss_mano_prior.detach()
        if loss_smooth_dec is not None:
            total_loss = total_loss + self.lambda_smooth_decoded * loss_smooth_dec
            loss_dict["loss_smooth_decoded"] = loss_smooth_dec.detach()
        if loss_dcontact is not None:
            total_loss = total_loss + self.lambda_dcontact * loss_dcontact
            loss_dict["loss_dcontact"] = loss_dcontact.detach()
        for comp, lam in (("attach_pos", self.lambda_attach_pos), ("attach_rot", self.lambda_attach_rot), ("attach_vel", self.lambda_attach_vel)):
            value = loss_attach_terms.get(comp)
            if value is not None:
                total_loss = total_loss + lam * value
                loss_dict[f"loss_{comp}"] = value.detach()
        loss_dict["loss_total"] = total_loss.detach()

        # 13. Validation macro inputs: per-window weighted FM loss and the per-component (value, applicable) pairs.
        # The energy score and support penalty are rollout-sampled training-only terms and are not part of it.
        if want_macro:
            with torch.no_grad():
                self._val_fm_per_sample = loss_fm_per_sample(output["v_pred"], v_target, loss_mask, fm_weights=self.fm_component_weights)
                if self.lambda_aux_contact > 0:
                    contact_logits = output["contact_logits"]
                    l_c = torch.zeros(contact_logits.shape[0], device=contact_logits.device, dtype=torch.float32)
                    n_c = torch.zeros(contact_logits.shape[0], device=contact_logits.device, dtype=torch.float32)
                    for h in range(2):
                        mk = (loss_mask[:, :, 3 + h] & presence_mask[:, :, h] & presence_mask[:, :, 2] & valid_frames).unsqueeze(-1).float()
                        bce = F.binary_cross_entropy_with_logits(contact_logits[:, :, h, :].float(), batch["contact_labels"][:, :, h, :].float(), reduction="none")
                        l_c = l_c + (bce * mk).sum(dim=(1, 2))
                        n_c = n_c + mk.sum(dim=(1, 2)) * 5.0
                    # Weighted at the final scheduled value so the monitor is the same objective throughout the ramp.
                    _macro_add("aux_contact", float(self.loss_schedule_fn(10**9)["lambda_aux_contact"]), l_c, n_c)
                self._val_aux_parts = macro_parts

        return total_loss, loss_dict

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        total_loss, loss_dict = self._shared_step(batch)
        B = batch["left_state"].shape[0]
        self.log("train/loss", total_loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=B)
        log_every = int(self.cfg.trainer.log_every_n_steps)
        if log_every <= 0 or int(self.global_step) % log_every == 0:
            # sync_dist: the DDP-exact per-window means are per-rank partial values that only average to the
            # global mean. The every-step train/loss above stays rank-local.
            self.log_dict({f"train/{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=False, prog_bar=False, batch_size=B, sync_dist=True)
            self.log("train/loss_global", total_loss, on_step=True, on_epoch=False, prog_bar=False, batch_size=B, sync_dist=True)
        if log_every > 0 and int(self.global_step) % log_every == 0 and self.loss_schedule_fn.ramps:
            lambdas = self.loss_schedule_fn(int(self.global_step))
            self.log_dict({f"schedule/{k}": float(v) for k, v in lambdas.items()}, on_step=True, on_epoch=False, batch_size=B)
        return total_loss

    # Fixed source vocabulary for the macro accumulators: a fixed length keeps the epoch-end all_gather
    # shape-consistent across ranks. Sources outside the list fall into the trailing bucket.
    _MACRO_SOURCES = ("dexycb", "hot3d", "oakink2", "taco", "show3d", "ho3d", "h2o", "hocap", "action100m", "other")
    # Components of val/loss_macro besides the FM loss, each averaged over its applicable windows per source.
    _MACRO_COMPONENTS = (
        "x0_pos",
        "x0_rot",
        "x0_artic",
        "anatomy",
        "mano_prior",
        "dcontact",
        "attach_pos",
        "attach_rot",
        "attach_vel",
        "smooth_dec",
        "aux_contact",
    )

    def validation_step(self, batch: dict, batch_idx: int) -> Tensor:
        self._move_stats_to_device()
        # Deterministic validation: masks, flow times and noise depend only on (seed, batch, rank), and the
        # training RNG stream is left untouched.
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            seed = int(self.run_seed) * 100003 + int(batch_idx) + int(getattr(self.trainer, "global_rank", 0) or 0) * 1000003
            torch.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed_all(seed)
            total_loss, loss_dict = self._shared_step(batch)
        B = batch["left_state"].shape[0]
        self.log_dict({f"val/{k}": v for k, v in loss_dict.items()}, on_step=False, on_epoch=True, sync_dist=True, batch_size=B)
        self.log("val/loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=B)

        # Per-source accumulation for val/fm_macro and val/loss_macro.
        fm = getattr(self, "_val_fm_per_sample", None)
        parts = getattr(self, "_val_aux_parts", None) or {}
        srcs = batch.get("source_dataset")
        if fm is not None and srcs is not None:
            if not hasattr(self, "_val_macro_sums") or self._val_macro_sums.device != fm.device:
                S = len(self._MACRO_SOURCES)
                C = len(self._MACRO_COMPONENTS)
                self._val_macro_sums = torch.zeros(S, device=fm.device)
                self._val_macro_counts = torch.zeros(S, device=fm.device)
                self._val_aux_sums = torch.zeros(C, S, device=fm.device)
                self._val_aux_counts = torch.zeros(C, S, device=fm.device)
            idx_of = {s: i for i, s in enumerate(self._MACRO_SOURCES)}
            cidx_of = {c: i for i, c in enumerate(self._MACRO_COMPONENTS)}
            for b in range(min(B, fm.shape[0])):
                i = idx_of.get(str(srcs[b]), len(self._MACRO_SOURCES) - 1)
                # Windows without any FM target are NaN-encoded by loss_fm_per_sample and skipped.
                if not bool(torch.isnan(fm[b])):
                    self._val_macro_sums[i] += fm[b]
                    self._val_macro_counts[i] += 1.0
                for comp, (v, has) in parts.items():
                    ci = cidx_of.get(comp)
                    if ci is not None and b < v.shape[0] and bool(has[b]):
                        self._val_aux_sums[ci, i] += v[b].float()
                        self._val_aux_counts[ci, i] += 1.0
        self._val_fm_per_sample = None
        self._val_aux_parts = None
        return total_loss

    def on_validation_epoch_start(self) -> None:
        if hasattr(self, "_val_macro_sums"):
            self._val_macro_sums.zero_()
            self._val_macro_counts.zero_()
        if hasattr(self, "_val_aux_sums"):
            self._val_aux_sums.zero_()
            self._val_aux_counts.zero_()

    def on_validation_epoch_end(self) -> None:
        if not hasattr(self, "_val_macro_sums"):
            return
        # all_gather is a collective with a fixed-shape tensor on every rank; the gathered values are
        # rank-consistent, so logging needs no sync_dist.
        sums = self.all_gather(self._val_macro_sums).reshape(-1, len(self._MACRO_SOURCES)).sum(dim=0)
        counts = self.all_gather(self._val_macro_counts).reshape(-1, len(self._MACRO_SOURCES)).sum(dim=0)
        per_src = sums / counts.clamp(min=1.0)
        active = counts > 0
        if active.any():
            self.log("val/fm_macro", per_src[active].mean(), on_epoch=True, sync_dist=False)
            for i, name in enumerate(self._MACRO_SOURCES):
                if active[i]:
                    self.log(f"val/fm_{name}", per_src[i], on_epoch=True, sync_dist=False)
        # val/loss_macro (the checkpoint monitor): equal-weight source mean of FM plus every decodable auxiliary
        # component. The energy score and support penalty are training-only and not monitored.
        C, S = len(self._MACRO_COMPONENTS), len(self._MACRO_SOURCES)
        asums = self.all_gather(self._val_aux_sums).reshape(-1, C, S).sum(dim=0)
        acounts = self.all_gather(self._val_aux_counts).reshape(-1, C, S).sum(dim=0)
        per_src_total = per_src + (asums / acounts.clamp(min=1.0)).sum(dim=0)
        if active.any():
            self.log("val/loss_macro", per_src_total[active].mean(), on_epoch=True, sync_dist=False)

    def configure_optimizers(self):
        tc = self.cfg.training
        # Weight decay only on >= 2-D weights outside Embedding/Norm modules; biases, norms, embeddings and 1-D
        # gains are decay-free, as are the learned 2-D token arrays and query banks named below.
        no_decay_owner = (torch.nn.Embedding, torch.nn.LayerNorm, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.GroupNorm)
        owned_no_decay: set[int] = set()
        for m in self.modules():
            if isinstance(m, no_decay_owner):
                for p in m.parameters(recurse=False):
                    owned_no_decay.add(id(p))
        no_decay_name_suffixes = ("scene_global_tokens", "null_cond_tokens", "queries")
        for pname, p in self.named_parameters():
            if pname.split(".")[-1] in no_decay_name_suffixes:
                owned_no_decay.add(id(p))
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (no_decay if (p.ndim <= 1 or id(p) in owned_no_decay) else decay).append(p)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": tc.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=tc.lr,
        )
        scheduler = build_lr_scheduler(optimizer, tc)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}


class GTFDataModule(pl.LightningDataModule):
    """Lightning DataModule wrapping GTFDataset for train/val.

    Args:
        cfg: Full GTFConfig.
        mano_root: Path to MANO model files.
        num_workers: DataLoader workers.
        pin_memory: Pin memory for faster GPU transfer.
        persistent_workers: Keep workers alive between epochs (off when num_workers=0).
        segment_ids: If given, restrict both splits to these segment IDs (smoke tests).
        segment_cache_max_bytes: Override for cfg.data.segment_cache_max_bytes, or None to use the config.
    """

    def __init__(
        self,
        cfg: GTFConfig,
        mano_root: str = MANO_DIR,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        segment_ids: list[str] | None = None,
        segment_cache_max_bytes: int | None = None,
    ) -> None:
        super().__init__()
        if isinstance(cfg, DictConfig):
            self.cfg = cfg
        elif isinstance(cfg, dict):
            self.cfg = OmegaConf.create(cfg)
        elif dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
            self.cfg = OmegaConf.structured(cfg)
        else:
            self.cfg = cfg
        dc, tc = self.cfg.data, self.cfg.training
        self.mano_root = mano_root
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.segment_ids = segment_ids
        self.segment_cache_max_bytes = dc.segment_cache_max_bytes if segment_cache_max_bytes is None else segment_cache_max_bytes
        # Window-balanced source sampler with tier boost, hand balance and a step-scheduled Action100M mass.
        self.dataset_alpha = float(dc.dataset_alpha)
        self.tier_boost: dict[str, float] = {str(k): float(v) for k, v in dict(dc.tier_boost).items()}
        self.hand_balance_target = float(dc.hand_balance_target)
        self.hand_balance_max_boost = float(dc.hand_balance_max_boost)
        self.sampler_action100m_mass_knots = [(int(pair[0]), float(pair[1])) for pair in (dc.sampler_action100m_mass_knots or [])]
        if not self.sampler_action100m_mass_knots:
            raise ValueError("data.sampler_action100m_mass_knots must be non-empty")
        # Temporal stride augmentation (train split only).
        self.aug_stride_prob = float(tc.aug_stride_prob)
        self.aug_stride_target_hz = float(tc.aug_stride_target_hz)
        self.aug_stride_max = int(tc.aug_stride_max)
        # Seconds-defined windows (both splits).
        self.window_seconds = float(tc.window_seconds)
        self.stride_seconds = float(tc.stride_seconds)
        self._uni3d_dataset_kwargs = runtime_uni3d_dataset_kwargs(dc)
        self._train_dataset: GTFDataset | None = None
        self._val_dataset: GTFDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        dc, tc = self.cfg.data, self.cfg.training
        if stage in ("fit", None) and self._train_dataset is None:
            self._train_dataset = GTFDataset(
                manifest_path=dc.manifest_path,
                window_size=tc.window_size,
                split="train",
                mano_root=self.mano_root,
                segment_ids=self.segment_ids,
                segment_cache_max_bytes=self.segment_cache_max_bytes,
                aug_stride_prob=self.aug_stride_prob,
                aug_stride_target_hz=self.aug_stride_target_hz,
                aug_stride_max=self.aug_stride_max,
                window_seconds=self.window_seconds,
                stride_seconds=self.stride_seconds,
                **self._uni3d_dataset_kwargs,
            )
        if stage in ("fit", "validate", None) and self._val_dataset is None:
            self._val_dataset = GTFDataset(
                manifest_path=dc.manifest_path,
                window_size=tc.window_size,
                split="val",
                mano_root=self.mano_root,
                segment_ids=self.segment_ids,
                segment_cache_max_bytes=self.segment_cache_max_bytes,
                window_seconds=self.window_seconds,
                stride_seconds=self.stride_seconds,
                **self._uni3d_dataset_kwargs,
            )

    def _build_weighted_sampler(self) -> StepScheduledDistributedWeightedSampler:
        """Window-balanced source sampler whose Action100M mass follows the step curriculum."""
        weights = compute_window_weights(
            self._train_dataset,
            alpha=self.dataset_alpha,
            tier_boost=self.tier_boost,
            hand_balance_target=self.hand_balance_target,
            hand_balance_max_boost=self.hand_balance_max_boost,
        )
        num_replicas = 1
        rank = 0
        if self.trainer is not None:
            num_replicas = max(1, int(getattr(self.trainer, "world_size", 1)))
            rank = int(getattr(self.trainer, "global_rank", 0))
        num_samples = max(1, len(self._train_dataset) // num_replicas)
        # The sampler redraws each epoch from its seed, so a resume folds the restored global_step into the seed:
        # the resumed stream is fresh and deterministic instead of replaying the consumed prefix. This runs at
        # train_dataloader() time, after Lightning restores the loop state and before the iterator is created.
        start_step = int(getattr(self.trainer, "global_step", 0) or 0) if self.trainer is not None else 0
        start_epoch = int(getattr(self.trainer, "current_epoch", 0) or 0) if self.trainer is not None else 0
        off = start_step * 1000003 if start_step > 0 else 0
        schedule = PiecewiseSchedule.from_pairs(self.sampler_action100m_mass_knots)
        is_action100m = torch.tensor(
            [str(self._train_dataset.segments[seg_idx].get("source_dataset", "unknown")).lower() == "action100m" for seg_idx, _start, _end in self._train_dataset.windows],
            dtype=torch.bool,
        )
        accumulate = int(self.cfg.trainer.accumulate_grad_batches)
        samples_per_rank_step = int(self.cfg.training.batch_size) * accumulate
        num_samples = (num_samples // samples_per_rank_step) * samples_per_rank_step
        if num_samples <= 0:
            raise ValueError(
                "curriculum dataset is smaller than one complete per-rank optimizer-step block: "
                f"local_windows={len(self._train_dataset) // num_replicas}, required={samples_per_rank_step}"
            )
        sampler = StepScheduledDistributedWeightedSampler(
            base_weights=weights,
            selected=is_action100m,
            schedule=schedule,
            num_samples=num_samples,
            num_replicas=num_replicas,
            rank=rank,
            batch_size=int(self.cfg.training.batch_size),
            accumulate_grad_batches=accumulate,
            start_global_step=start_step,
            start_epoch=start_epoch,
            global_step_getter=((lambda: int(getattr(self.trainer, "global_step", 0) or 0)) if self.trainer is not None else None),
            exposure_steps=int(self.cfg.training.max_steps),
            seed=42 + off,
        )
        masses = curriculum_mass_report(sampler.weights, is_extracted=is_action100m)
        logging.getLogger("sampler").info(
            "step curriculum: action100m knots=%s start_step=%d whole-run expected mass=%.4f curated=%.4f",
            self.sampler_action100m_mass_knots,
            start_step,
            masses["extracted"],
            masses["curated"],
        )
        return sampler

    def _train_worker_init_fn(self):
        """Seed each DataLoader worker on (seed, rank, restored global_step, worker_id), so a resume does not replay
        the worker-side augmentation draws (worker base seeds are drawn before the step-keyed re-seed)."""
        base = int(self.cfg.training.seed)
        rank = int(getattr(self.trainer, "global_rank", 0) or 0) if self.trainer is not None else 0
        step = int(getattr(self.trainer, "global_step", 0) or 0) if self.trainer is not None else 0

        def _init(worker_id: int, _s=base * 9176 + rank * 7919 + step * 257):
            import random as _random

            import numpy as _np

            s = (_s + worker_id * 6151) % (2**31 - 1)
            torch.manual_seed(s)
            _np.random.seed(s)
            _random.seed(s)

        return _init

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self.cfg.training.batch_size,
            sampler=self._build_weighted_sampler(),
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=gtf_collate,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,
            worker_init_fn=self._train_worker_init_fn(),
        )

    def val_dataloader(self) -> DataLoader:
        # Lightning's automatic distributed sampler is off (the train sampler is custom), so validation is sharded
        # here; sync_dist then averages over the per-rank subsets.
        sampler = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = DistributedSampler(self._val_dataset, shuffle=False, drop_last=False)
        return DataLoader(
            self._val_dataset,
            batch_size=self.cfg.training.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=gtf_collate,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )
