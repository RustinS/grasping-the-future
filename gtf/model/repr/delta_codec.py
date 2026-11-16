"""Temporal-difference state codec ("Δ-codec") at the flow boundary.

Reparameterizes the STATE tokens (left_state, right_state, obj_state) that the flow
model operates on: within each contiguous PRESENCE RUN of an entity, the run-start
frame carries the (centered) ABSOLUTE state and every interior frame carries the
per-second temporal DERIVATIVE (x[t] − x[t−1]) / dt_phys. Relations (left_rel,
right_rel) stay ABSOLUTE — they carry the per-frame hand↔object grounding that
pure derivatives would destroy.

Derivative targets are near-zero-mean and horizon-stationary: rest is the zero point
of the output space, and dividing by the physical transition time and normalizing by
the per-second velocity std makes the representation cadence-invariant across the
mixed-fps corpus.

Mask semantics: a codec slot is observed iff its inputs are observed —
run-start: obs[t]; interior: obs[t] & obs[t−1]. Presence is unchanged (every
present frame has a defined codec value). Absent slots are zero-filled.

Decode integrates derivatives within each run, anchored at the run-start absolute,
then restores originally-observed frames exactly. All ops are differentiable
(used inside the energy-score loss). No learnable parameters.

Timing: ``dt_phys`` is ``(B, T-1)`` seconds, column ``t`` spanning frame ``t → t+1``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from gtf.model.repr.tokenizer import (
    HAND_CAM_T_SLICE,
    HAND_ORIENT_SLICE,
    HAND_POSE_SLICE,
    OBJ_ROTATION_SLICE,
    OBJ_TRANSLATION_SLICE,
)
from gtf.model.repr.timing import frame_times_from_edge_dt, normalize_edge_dt

STATE_TOKENS = ("left_state", "right_state", "obj_state")
STATE_SLOTS = {"left_state": 0, "right_state": 1, "obj_state": 2}
REL_TOKENS = ("left_rel", "right_rel")

# Per-token map: which velocity_std group covers which channel slice.
_VEL_GROUPS = {
    "left_state": [("hand_root_rotation", HAND_ORIENT_SLICE), ("hand_root_translation", HAND_CAM_T_SLICE), ("hand_articulation", HAND_POSE_SLICE)],
    "right_state": [("hand_root_rotation", HAND_ORIENT_SLICE), ("hand_root_translation", HAND_CAM_T_SLICE), ("hand_articulation", HAND_POSE_SLICE)],
    "obj_state": [("obj_rotation", OBJ_ROTATION_SLICE), ("obj_translation", OBJ_TRANSLATION_SLICE)],
}

_STATE_DIMS = {"left_state": 99, "right_state": 99, "obj_state": 9}


def _run_starts(presence_slot: Tensor) -> Tensor:
    """(B, T) bool → (B, T) bool: True where a presence run begins."""
    prev = torch.zeros_like(presence_slot)
    prev[:, 1:] = presence_slot[:, :-1]
    return presence_slot & ~prev


def _codec_edge_dt(
    dt_phys: Tensor,
    presence_mask: Tensor,
    ref: Tensor,
) -> Tensor:
    """Return the ``(B,T-1)`` transition durations on ``ref``'s device and dtype.

    Finiteness/positivity of active edges is validated at the dataset boundary
    rather than with a CUDA host synchronization in this per-step hot path.
    """
    B, T = presence_mask.shape[:2]
    return normalize_edge_dt(dt_phys, n_frames=T, batch_size=B, device=ref.device, dtype=ref.dtype, validate=False)


def _frame_aligned_dt(edge_dt: Tensor) -> Tensor:
    """``(B,T-1)`` transition durations → slot-aligned ``(B,T,1)``.

    Slot ``t`` uses the preceding edge ``t-1→t``. The frame-0 dummy is never
    consumed because the first present frame of every run is absolute.
    """
    return torch.cat([edge_dt.new_ones(edge_dt.shape[0], 1), edge_dt], dim=1).unsqueeze(-1)


class DeltaCodec:
    """Stats-driven, parameter-free temporal-difference codec for state tokens.

    ``stats["velocity_std"]`` is stored in normalized-position units per second
    (compute_stats divides by the positional std); the codec multiplies it by the
    positional std slice to get the physical per-second velocity std, so codec
    derivative channels have approximately unit variance."""

    def __init__(self, stats: dict):
        # Stored on CPU; _dev() moves per consumer call (callers pass either CPU
        # stats or stats already moved to the GPU).
        self.pos_mean: dict[str, Tensor] = {}
        self.pos_std: dict[str, Tensor] = {}
        self.vel_std: dict[str, Tensor] = {}
        for name in STATE_TOKENS + REL_TOKENS:
            self.pos_mean[name] = stats["mean"][name].detach().float().cpu().clone()
            self.pos_std[name] = stats["std"][name].detach().float().cpu().clamp(min=1e-6).clone()
        for name in STATE_TOKENS:
            v = torch.ones(_STATE_DIMS[name])
            for group, sl in _VEL_GROUPS[name]:
                v[sl] = stats["velocity_std"][group].detach().float().cpu()
                v[sl] = v[sl] * self.pos_std[name][sl]
            self.vel_std[name] = v.clamp(min=1e-6)

    def _dev(self, name: str, ref: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
        d = ref.device
        vstd = self.vel_std[name].to(d) if name in self.vel_std else None
        return self.pos_mean[name].to(d), self.pos_std[name].to(d), vstd

    # ── mask transform ───────────────────────────────────────────────────────
    def transform_masks(self, obs_mask: Tensor, presence_mask: Tensor) -> Tensor:
        """(B,T,5) obs → codec obs. Interior slots need obs[t] & obs[t−1];
        run-start slots need only obs[t]. Relations (slots 3,4) unchanged."""
        obs_c = obs_mask.clone()
        for name, k in STATE_SLOTS.items():
            pres = presence_mask[:, :, k]
            rs = _run_starts(pres)
            prev_obs = torch.zeros_like(obs_mask[:, :, k])
            prev_obs[:, 1:] = obs_mask[:, :-1, k]
            obs_c[:, :, k] = torch.where(rs, obs_mask[:, :, k], obs_mask[:, :, k] & prev_obs)
        return obs_c

    # ── encode: centered absolute → normalized codec space ─────────────────
    def encode(
        self,
        centered_tokens: dict[str, Tensor],
        presence_mask: Tensor,
        dt_phys: Tensor,
    ) -> dict[str, Tensor]:
        """centered_tokens: dict of (B,T,D) UNNORMALIZED centered tokens.
        dt_phys: ``(B,T-1)`` transition durations in seconds.
        Returns normalized codec tokens (state = run-start absolute-normalized /
        interior velocity-normalized, relations = standard-normalized). Zero at
        absent slots."""
        out: dict[str, Tensor] = {}
        edge_dt = _codec_edge_dt(dt_phys, presence_mask, centered_tokens["obj_state"])
        # Slot t stores the derivative over transition (t-1)→t. Slot 0 is always a
        # presence-run absolute, so its dummy duration is unused.
        dt = _frame_aligned_dt(edge_dt)
        for name in STATE_TOKENS:
            x = centered_tokens[name]
            mean, std, vstd = self._dev(name, x)
            pres = presence_mask[:, :, STATE_SLOTS[name]].unsqueeze(-1)
            rs = _run_starts(presence_mask[:, :, STATE_SLOTS[name]]).unsqueeze(-1)
            prev = torch.zeros_like(x)
            prev[:, 1:] = x[:, :-1]
            vel = (x - prev) / dt.clamp(min=1e-6)
            enc = torch.where(rs, (x - mean) / std, vel / vstd)
            out[name] = enc * pres.to(enc.dtype)
        for name in REL_TOKENS:
            if name in centered_tokens:
                x = centered_tokens[name]
                mean, std, _ = self._dev(name, x)
                pres = presence_mask[:, :, 3 if name == "left_rel" else 4].unsqueeze(-1)
                out[name] = ((x - mean) / std) * pres.to(x.dtype)
        return out

    # ── decode: normalized codec space → centered absolute (denormalized) ──
    def decode_state(
        self,
        x_norm: dict[str, Tensor],
        presence_mask: Tensor,
        dt_phys: Tensor,
    ) -> dict[str, Tensor]:
        """Integrate codec-space state tokens back to centered absolute values
        with a sequential per-frame loop (the decode of decode_with_observed).
        Relations are denormalized standard. Absent slots come back zero."""
        out: dict[str, Tensor] = {}
        edge_dt = _codec_edge_dt(dt_phys, presence_mask, x_norm["obj_state"])
        for name in STATE_TOKENS:
            xc = x_norm[name]
            B, T, D = xc.shape
            mean, std, vstd = self._dev(name, xc)
            pres = presence_mask[:, :, STATE_SLOTS[name]]
            rs = _run_starts(pres)
            abs_val = xc * std + mean  # interpretation at run-start slots
            vel_val = xc * vstd  # interpretation at interior slots (per-second)
            pos = torch.zeros_like(xc)
            prev = torch.zeros(B, D, device=xc.device, dtype=xc.dtype)
            for t in range(T):
                p = pres[:, t].unsqueeze(-1).to(xc.dtype)
                r = rs[:, t].unsqueeze(-1).to(xc.dtype)
                dt_t = edge_dt[:, t - 1].unsqueeze(-1) if t > 0 else xc.new_ones(B, 1)
                cur = r * abs_val[:, t] + (1.0 - r) * (prev + vel_val[:, t] * dt_t)
                cur = cur * p  # absent → zero (and does not advance the chain)
                pos[:, t] = cur
                prev = torch.where(pres[:, t].unsqueeze(-1), cur, prev)
            out[name] = pos
        for name in REL_TOKENS:
            if name in x_norm:
                mean, std, _ = self._dev(name, x_norm[name])
                pres = presence_mask[:, :, 3 if name == "left_rel" else 4].unsqueeze(-1)
                out[name] = (x_norm[name] * std + mean) * pres.to(x_norm[name].dtype)
        return out

    def decode_state_fast(
        self,
        x_norm: dict[str, Tensor],
        presence_mask: Tensor,
        dt_phys: Tensor,
    ) -> dict[str, Tensor]:
        """Vectorized decode_state for the 3 STATE tokens (segment-wise cumsum):
        the same integration with the additions reassociated. Used by the training
        losses (energy score and decoded x0 terms). Relations are not decoded."""
        out: dict[str, Tensor] = {}
        edge_dt = _codec_edge_dt(dt_phys, presence_mask, x_norm["obj_state"])
        dt = _frame_aligned_dt(edge_dt)
        for name in STATE_TOKENS:
            xc = x_norm[name]
            B, T, D = xc.shape
            mean, std, vstd = self._dev(name, xc)
            pres = presence_mask[:, :, STATE_SLOTS[name]]
            rs = _run_starts(pres)
            p = pres.unsqueeze(-1).to(xc.dtype)
            r = rs.unsqueeze(-1).to(xc.dtype)
            abs_val = xc * std + mean
            vel_val = xc * vstd
            contrib = (r * abs_val + (1.0 - r) * vel_val * dt) * p
            cs = torch.cumsum(contrib, dim=1)
            cs_prev = torch.cat([torch.zeros_like(cs[:, :1]), cs[:, :-1]], dim=1)
            # forward-fill each run's pre-run cumulative through the run, then subtract.
            t_idx = torch.arange(T, device=xc.device).unsqueeze(0).expand(B, T)
            start_idx = torch.where(rs, t_idx, torch.zeros_like(t_idx)).cummax(dim=1).values  # (B, T)
            base = torch.gather(cs_prev, 1, start_idx.unsqueeze(-1).expand(B, T, D))
            out[name] = (cs - base) * p
        return out

    def decode_with_observed(
        self,
        x_norm: dict[str, Tensor],
        clean_centered: dict[str, Tensor],
        obs_mask_orig: Tensor,
        presence_mask: Tensor,
        dt_phys: Tensor,
    ) -> dict[str, Tensor]:
        """decode_state + restore originally-observed frames exactly from the
        clean (unnormalized, centered) tokens — the generate() convention.

        Anchored decode: the forward integration carries no information from FUTURE
        observed anchors (only run-start absolutes and derivatives exist in codec
        space), so masked gaps drift and the trajectory would jump at the gap→anchor
        boundary when observed frames are restored. Therefore each masked run
        bounded by observed frames on both sides gets its integration
        mismatch distributed across the gap in proportion to elapsed time:
        corrected[t] = dec[t] − o − (m − o)·w(t), where o/m are the integration
        offsets at the past/future anchors and w the elapsed-time fraction — exact
        convergence to BOTH anchors by construction. Leading gaps (future anchor
        only) get a constant back-anchor shift; trailing gaps (future prediction) a
        constant forward re-anchor. Runs with no observed anchor at all are
        untouched. Rotation channels use an SO(3)-geodesic closure (see
        _close_masked_gaps). Differentiable."""
        dec = self.decode_state(x_norm, presence_mask, dt_phys)
        B, T = presence_mask.shape[:2]
        edge_dt = normalize_edge_dt(
            dt_phys,
            n_frames=T,
            batch_size=B,
            device=x_norm["obj_state"].device,
            dtype=torch.float32,
            validate=False,
        )
        closure_times = frame_times_from_edge_dt(edge_dt)
        out = {}
        for name in STATE_TOKENS:
            k = STATE_SLOTS[name]
            obs = obs_mask_orig[:, :, k].unsqueeze(-1)
            rot_blocks = [0] if name == "obj_state" else [0] + list(range(9, 99, 6))
            val = _close_masked_gaps(
                dec[name],
                clean_centered[name].to(dec[name].dtype),
                obs_mask_orig[:, :, k],
                presence_mask[:, :, k],
                rot_blocks=rot_blocks,
                frame_times=closure_times,
            )
            out[name] = torch.where(obs, clean_centered[name].to(val.dtype), val)
        for name in REL_TOKENS:
            if name in dec:
                k = 3 if name == "left_rel" else 4
                obs = obs_mask_orig[:, :, k].unsqueeze(-1)
                out[name] = torch.where(obs, clean_centered[name].to(dec[name].dtype), dec[name])
        return out


def build_anchor_feats(centered_tokens: dict[str, Tensor], obs_mask_orig: Tensor, stats: dict) -> dict[str, Tensor]:
    """Anchor side-channel inputs: per STATE token, [absolute-normalized state at
    OBSERVED frames (zeros at masked), observed-indicator bit] → (B,T,D+1).

    Gives the transformer direct access to every observed anchor's ABSOLUTE
    position — in Δ-codec space an observed future frame otherwise contributes
    only derivative slots, so the anchor's absolute position would be invisible.
    Consumed by InputEmbedding via a zero-init additive projection.
    """
    out: dict[str, Tensor] = {}
    for name, k in STATE_SLOTS.items():
        x = centered_tokens[name]
        mean = stats["mean"][name].to(device=x.device, dtype=x.dtype)
        std = stats["std"][name].to(device=x.device, dtype=x.dtype).clamp(min=1e-6)
        obs = obs_mask_orig[:, :, k].unsqueeze(-1)
        absn = ((x - mean) / std) * obs.to(x.dtype)
        out[name] = torch.cat([absn, obs.to(x.dtype)], dim=-1)
    return out


def _masked_runs_1d(masked: Tensor) -> list[tuple[int, int]]:
    """(T,) bool → maximal contiguous True runs as (start, end) inclusive."""
    runs, start = [], None
    T = masked.shape[0]
    for t in range(T):
        if masked[t] and start is None:
            start = t
        elif not masked[t] and start is not None:
            runs.append((start, t - 1))
            start = None
    if start is not None:
        runs.append((start, T - 1))
    return runs


def _close_masked_gaps(
    dec: Tensor,
    clean: Tensor,
    obs_slot: Tensor,
    pres_slot: Tensor,
    rot_blocks: list[int],
    frame_times: Tensor,
) -> Tensor:
    """Anchored-decode gap closure (see decode_with_observed docstring).

    dec/clean: (B,T,D) centered absolute; obs_slot/pres_slot: (B,T) bool.
    Returns a corrected copy; observed-frame values are NOT written here (the
    caller's torch.where restore handles them) — only masked frames move.

    rot_blocks: start indices of 6D rotation blocks within D. Those dims get an
    SO(3)-GEODESIC closure instead of the linear one (a linear 6D correction
    degenerates near 180° offsets): the anchor offsets Δ_p = R_clean(p)·R_dec(p)ᵀ
    and Δ_f (the rotations taking the decoded chain onto the clean anchors) are
    interpolated along their geodesic Δ(t) = Δ_p · exp(w(t)·log(Δ_pᵀΔ_f)) and
    left-applied to R_dec(t) — exact at both anchors by construction.
    Leading/trailing gaps use the constant single-anchor offset.

    ``frame_times`` is the ``(B,T)`` physical time: the interpolation weight is
    the elapsed-time fraction.
    """
    out = dec.clone()
    B, T, _ = dec.shape
    idx_flat = torch.tensor([s + i for s in rot_blocks for i in range(6)], device=dec.device, dtype=torch.long)  # (K*6,)
    for b in range(B):
        masked_b = pres_slot[b] & ~obs_slot[b]
        for g0, g1 in _masked_runs_1d(masked_b):
            # Anchors must be PRESENT + OBSERVED and temporally adjacent (an adjacent
            # present frame is observed by construction — else it would be in the run).
            p = g0 - 1 if g0 > 0 and bool(pres_slot[b, g0 - 1]) else None
            f = g1 + 1 if g1 < T - 1 and bool(pres_slot[b, g1 + 1]) else None
            o = (dec[b, p] - clean[b, p]) if p is not None else None
            m = (dec[b, f] - clean[b, f]) if f is not None else None
            w = None
            if p is not None and f is not None:
                times_b = frame_times[b].to(device=dec.device, dtype=torch.float32)
                denom = (times_b[f] - times_b[p]).clamp_min(torch.finfo(torch.float32).eps)
                w = (times_b[g0 : g1 + 1] - times_b[p]) / denom
                out[b, g0 : g1 + 1] = dec[b, g0 : g1 + 1] - o.unsqueeze(0) - (m - o).unsqueeze(0) * w.unsqueeze(-1)
            elif f is not None:  # leading gap: back-anchor (constant shift onto the future anchor)
                out[b, g0 : g1 + 1] = dec[b, g0 : g1 + 1] - m.unsqueeze(0)
            elif p is not None:  # trailing gap (future prediction): forward re-anchor to the last observed frame
                out[b, g0 : g1 + 1] = dec[b, g0 : g1 + 1] - o.unsqueeze(0)
            else:
                continue  # no anchors at all — untouched (rot dims included)

            from gtf.model.repr.rotation import axis_angle_to_matrix, matrix_to_rotation_6d, rotation_6d_to_axis_angle, rotation_6d_to_matrix

            K = len(rot_blocks)
            G = g1 - g0 + 1

            def _R_at(x: Tensor, t: int) -> Tensor:
                return rotation_6d_to_matrix(x[b, t][idx_flat].view(K, 6).float())  # (K,3,3)

            Dp = _R_at(clean, p) @ _R_at(dec, p).transpose(-1, -2) if p is not None else None
            Df = _R_at(clean, f) @ _R_at(dec, f).transpose(-1, -2) if f is not None else None
            if Dp is None:
                # Leading gap: constant future offset (rel = identity).
                Dp = Df
                rel_aa = torch.zeros(K, 3, device=dec.device, dtype=torch.float32)
            elif Df is None:
                rel_aa = torch.zeros(K, 3, device=dec.device, dtype=torch.float32)
            else:
                rel_aa = rotation_6d_to_axis_angle(matrix_to_rotation_6d(Dp.transpose(-1, -2) @ Df))  # (K,3) principal log
            w_g = w.float() if w is not None else torch.zeros(G, device=dec.device, dtype=torch.float32)
            aa = w_g.view(G, 1, 1) * rel_aa.view(1, K, 3)  # (G,K,3)
            Dt = Dp.view(1, K, 3, 3) @ axis_angle_to_matrix(aa.reshape(-1, 3)).view(G, K, 3, 3)
            Rdec_g = rotation_6d_to_matrix(dec[b, g0 : g1 + 1][:, idx_flat].reshape(-1, 6).float()).view(G, K, 3, 3)
            out6 = matrix_to_rotation_6d((Dt @ Rdec_g).reshape(-1, 3, 3)).view(G, K * 6).to(dec.dtype)
            out[b, g0 : g1 + 1, idx_flat] = out6
    return out
