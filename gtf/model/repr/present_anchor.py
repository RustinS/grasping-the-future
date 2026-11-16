"""Present-frame anchoring for moving-camera canonical data.

Canonical files may store world-frame states plus per-frame world->camera
extrinsics. The observation mask is sampled at training/eval time, so the
correct camera anchor cannot be fixed during ingest. This module converts world
states into the camera/gravity frame of the latest observed state frame.
"""

from __future__ import annotations

import torch
from torch import Tensor

from gtf.model.repr.canonicalize import _LEFT_MIRROR_DIAG, _mano_root_joint_right, compute_gravity_alignment, mano_root_joint_right_beta
from gtf.model.repr.rotation import matrix_to_rotation_6d, rotation_6d_to_matrix
from gtf.model.repr.tokenizer import HAND_CAM_T_SLICE, HAND_ORIENT_SLICE, HAND_POSE_SLICE, OBJ_ROTATION_SLICE, OBJ_TRANSLATION_SLICE


def select_present_frames(obs_mask: Tensor, presence_mask: Tensor, valid_frames: Tensor | None = None) -> Tensor:
    """Return the local frame index whose camera should anchor each sample.

    Present is the latest valid frame where at least one state slot
    (left/right/object) is both present and observed. That matches the last
    real-world frame the model is allowed to see across all mask regimes.
    Full-mask samples fall back to the first valid frame.
    """
    if obs_mask.ndim != 3 or presence_mask.ndim != 3:
        raise ValueError("obs_mask and presence_mask must be (B,T,5)")
    if obs_mask.shape != presence_mask.shape:
        raise ValueError(f"obs_mask shape {tuple(obs_mask.shape)} != presence_mask shape {tuple(presence_mask.shape)}")

    B, T, _ = obs_mask.shape
    device = obs_mask.device
    valid = valid_frames if valid_frames is not None else presence_mask.any(dim=-1)
    valid = valid.to(device=device, dtype=torch.bool)

    observed_state = (obs_mask[:, :, :3] & presence_mask[:, :, :3]).any(dim=-1) & valid
    idx = torch.arange(T, device=device).view(1, T).expand(B, T)
    latest = torch.where(observed_state, idx, torch.zeros_like(idx)).amax(dim=1)

    has_observed = observed_state.any(dim=1)
    first_valid = torch.where(valid, idx, torch.full_like(idx, T)).amin(dim=1).clamp(max=T - 1)
    return torch.where(has_observed, latest, first_valid)


def select_fully_observed_frames(obs_mask: Tensor, presence_mask: Tensor, valid_frames: Tensor | None = None) -> Tensor:
    """Latest valid frame where EVERY present state entity is observed.

    Selects the scene row without seeing the targets: the per-frame scene row carries
    d_obj/d_hand distance fields and an interaction-centered crop for its own frame,
    so conditioning on a frame where some state entity is masked would leak that
    entity's position there (hand_to_object's present frame is the final frame with
    the object masked, where d_obj encodes the answer). On future_prediction,
    temporal_infilling and all-entity goal_infilling this coincides with
    select_present_frames (their present frame is fully observed). On
    hand_to_object, object_to_hand and the hands-only / object-only goal_infilling
    variants it returns frame 0 (always fully visible).
    Falls back to select_present_frames when no fully-observed frame exists (fully
    masked windows).
    """
    if obs_mask.ndim != 3 or presence_mask.ndim != 3:
        raise ValueError("obs_mask and presence_mask must be (B,T,5)")
    B, T, _ = obs_mask.shape
    device = obs_mask.device
    valid = valid_frames if valid_frames is not None else presence_mask.any(dim=-1)
    valid = valid.to(device=device, dtype=torch.bool)

    pres = presence_mask[:, :, :3]
    fully = (obs_mask[:, :, :3] | ~pres).all(dim=-1) & pres.any(dim=-1) & valid  # (B, T)
    idx = torch.arange(T, device=device).view(1, T).expand(B, T)
    latest = torch.where(fully, idx, torch.zeros_like(idx)).amax(dim=1)
    has = fully.any(dim=1)
    fallback = select_present_frames(obs_mask, presence_mask, valid_frames)
    return torch.where(has, latest, fallback)


def _scene_frame_transforms_impl(batch: dict, frames: Tensor) -> tuple[Tensor, Tensor] | None:
    """(R_tot, t_tot) mapping WORLD → the gravity-aligned camera frame at `frames`,
    per sample — the exact convention reanchor_tokens_to_present_camera applies to
    the state tokens and gtf.datasets.scene.preprocess applies to per-frame scene rows. Returns
    None when the batch carries no world-anchor metadata (static-camera sources —
    every frame shares one camera frame, no transform needed).
    """
    cam = batch.get("camera_extrinsics")
    if cam is None or batch.get("has_present_anchor") is None:
        return None
    cam = cam.float()
    sel = _gather_present_extrinsics(cam, frames)
    R_cw = sel[:, :3, :3]
    t_cw = sel[:, :3, 3]
    gravity_world = batch.get("gravity_world")
    if gravity_world is not None:
        g = gravity_world.to(device=R_cw.device, dtype=R_cw.dtype)
        use_g = g.norm(dim=-1) > 1e-6
        g_cam = torch.einsum("bij,bj->bi", R_cw, g)
        R_align = _batched_gravity_alignment(g_cam)
        R_tot = torch.einsum("bij,bjk->bik", R_align, R_cw)
        t_tot = torch.einsum("bij,bj->bi", R_align, t_cw)
        R_tot = torch.where(use_g.view(-1, 1, 1), R_tot, R_cw)
        t_tot = torch.where(use_g.view(-1, 1), t_tot, t_cw)
    else:
        R_tot, t_tot = R_cw, t_cw
    return R_tot, t_tot


def scene_frame_transforms(batch: dict, frames: Tensor) -> tuple[Tensor, Tensor] | None:
    """Exact scene-frame twin of the shaped state re-anchor.

    Scene content is loaded at a fully-observed frame and re-expressed in the
    present frame. Under an outer bf16 autocast the small einsums would round the
    fp32 transform tensors, putting state and scene coordinates in slightly
    different frames, so this rigid-frame calculation runs with autocast off and
    IEEE fp32 matmuls, restoring the caller's state.
    """
    cam = batch.get("camera_extrinsics")
    if cam is None:
        return _scene_frame_transforms_impl(batch, frames)
    device_type = cam.device.type if isinstance(cam, Tensor) else "cpu"
    previous_matmul_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        with torch.amp.autocast(device_type, enabled=False):
            return _scene_frame_transforms_impl(batch, frames)
    finally:
        torch.set_float32_matmul_precision(previous_matmul_precision)


def transform_scene_row(scene: dict, R_from: Tensor, t_from: Tensor, R_to: Tensor, t_to: Tensor) -> dict:
    """Re-express one per-frame scene row from frame FROM's gravity-aligned camera
    frame into frame TO's: X_to = R_rel (X_from) + t_rel with R_rel = R_to·R_fromᵀ,
    t_rel = t_to − R_rel·t_from (observed-frame scene content, present-frame
    coordinates).

    xyz (P_scene + the scene_geo_feats[:, 0:3] copy) map as points; normals
    (geo[:, 3:6] + standalone scene_normals) rotate. The remaining channels
    (d_obj, d_hand, h_support, θ_grav, curvature, log_density, DINOv3 sem) are
    INVARIANT: both frames are gravity-aligned so R_rel is a yaw about the
    gravity axis (heights and |n_y| preserved; on no-gravity samples those
    channels are already degenerate), and the distance fields are isometric.
    """
    R_rel = R_to @ R_from.transpose(-1, -2)
    t_rel = t_to - R_rel @ t_from
    out = dict(scene)
    P = scene["P_scene"]
    Rr = R_rel.to(device=P.device, dtype=P.dtype)
    tr = t_rel.to(device=P.device, dtype=P.dtype)
    P_new = P @ Rr.T + tr
    out["P_scene"] = P_new
    geo = scene.get("scene_geo_feats")
    if geo is not None:
        geo = geo.clone()
        geo[:, 0:3] = P_new
        geo[:, 3:6] = geo[:, 3:6] @ Rr.T
        out["scene_geo_feats"] = geo
    sn = scene.get("scene_normals")
    if sn is not None:
        out["scene_normals"] = sn @ Rr.T
    return out


def _batched_gravity_alignment(gravity_direction: Tensor) -> Tensor:
    mats = []
    for g in gravity_direction:
        norm = g.norm()
        if norm < 1e-6:
            mats.append(torch.eye(3, device=g.device, dtype=g.dtype))
        else:
            mats.append(compute_gravity_alignment(g / norm))
    return torch.stack(mats, dim=0)


def _gather_present_extrinsics(camera_extrinsics: Tensor, present_frames: Tensor) -> Tensor:
    B = camera_extrinsics.shape[0]
    return camera_extrinsics[torch.arange(B, device=camera_extrinsics.device), present_frames]


def _transform_hand_state(state_world: Tensor, R: Tensor, t: Tensor, valid: Tensor, side: str, betas: Tensor | None = None) -> Tensor:
    B, T, _ = state_world.shape
    R_eff = R
    t_eff = t
    if side == "left":
        m = torch.tensor(_LEFT_MIRROR_DIAG, device=R.device, dtype=R.dtype)
        R_eff = m.view(1, 3, 1) * R * m.view(1, 1, 3)
        t_eff = m.view(1, 3) * t

    orient = state_world[:, :, HAND_ORIENT_SLICE]
    R_hand = rotation_6d_to_matrix(orient)
    R_out = torch.einsum("bij,btjk->btik", R_eff, R_hand)
    orient_out = matrix_to_rotation_6d(R_out)

    cam_t = state_world[:, :, HAND_CAM_T_SLICE]
    if betas is not None:
        # Exact per-sample shaped J0 pivot (keeps the physical relations frame-invariant).
        j0 = mano_root_joint_right_beta(betas, cam_t.device, cam_t.dtype).view(B, 1, 3)  # (B, 1, 3)
    else:
        j0 = _mano_root_joint_right(cam_t.device, cam_t.dtype).view(1, 1, 3)
    trans_out = torch.einsum("btj,bij->bti", cam_t + j0, R_eff) + t_eff.view(B, 1, 3) - j0

    out = state_world.clone()
    out[:, :, HAND_ORIENT_SLICE] = orient_out
    out[:, :, HAND_CAM_T_SLICE] = trans_out
    out[:, :, HAND_POSE_SLICE] = state_world[:, :, HAND_POSE_SLICE]
    out = torch.where(valid.view(B, T, 1), out, torch.zeros_like(out))
    return out


def _transform_obj_state(state_world: Tensor, R: Tensor, t: Tensor, valid: Tensor) -> Tensor:
    B, T, _ = state_world.shape
    R_obj = rotation_6d_to_matrix(state_world[:, :, OBJ_ROTATION_SLICE])
    R_out = torch.einsum("bij,btjk->btik", R, R_obj)
    trans = state_world[:, :, OBJ_TRANSLATION_SLICE]
    trans_out = torch.einsum("btj,bij->bti", trans, R) + t.view(B, 1, 3)

    out = state_world.clone()
    out[:, :, OBJ_ROTATION_SLICE] = matrix_to_rotation_6d(R_out)
    out[:, :, OBJ_TRANSLATION_SLICE] = trans_out
    out = torch.where(valid.view(B, T, 1), out, torch.zeros_like(out))
    return out


def _reanchor_tokens_to_present_camera_impl(
    tokens: dict[str, Tensor],
    batch: dict,
    obs_mask: Tensor,
    presence_mask: Tensor,
    valid_frames: Tensor | None = None,
) -> tuple[dict[str, Tensor], Tensor]:
    """Convert samples with world anchor metadata to the present camera frame.

    Samples without `has_present_anchor` are returned unchanged. Static-camera
    samples are safe on this path: a constant/identity extrinsic applies the
    same rigid transform for every possible present frame. The hand wrist pivot
    uses the per-sample shaped root J0(β) from the batch betas.
    """
    present_frames = select_present_frames(obs_mask, presence_mask, valid_frames)
    has_anchor = batch.get("has_present_anchor")
    if has_anchor is None:
        return tokens, present_frames
    if not isinstance(has_anchor, Tensor):
        has_anchor = torch.as_tensor(has_anchor, device=obs_mask.device, dtype=torch.bool)
    else:
        has_anchor = has_anchor.to(device=obs_mask.device, dtype=torch.bool)
    if not bool(has_anchor.any().item()):
        return tokens, present_frames

    required = ("left_state_world", "right_state_world", "obj_state_world", "camera_extrinsics")
    if any(batch.get(k) is None for k in required):
        return tokens, present_frames

    cam = batch["camera_extrinsics"].to(device=obs_mask.device, dtype=tokens["obj_state"].dtype)
    present_cam = _gather_present_extrinsics(cam, present_frames)
    R_cw = present_cam[:, :3, :3]
    t_cw = present_cam[:, :3, 3]

    gravity_world = batch.get("gravity_world")
    if gravity_world is not None:
        gravity_world = gravity_world.to(device=obs_mask.device, dtype=tokens["obj_state"].dtype)
        use_gravity = gravity_world.norm(dim=-1) > 1e-6
        g_cam = torch.einsum("bij,bj->bi", R_cw, gravity_world)
        R_align = _batched_gravity_alignment(g_cam)
        R_total = torch.einsum("bij,bjk->bik", R_align, R_cw)
        t_total = torch.einsum("bij,bj->bi", R_align, t_cw)
        R_total = torch.where(use_gravity.view(-1, 1, 1), R_total, R_cw)
        t_total = torch.where(use_gravity.view(-1, 1), t_total, t_cw)
    else:
        R_total = R_cw
        t_total = t_cw

    out = dict(tokens)
    _bl = batch.get("betas_left")
    _br = batch.get("betas_right")
    left = _transform_hand_state(
        batch["left_state_world"].to(device=obs_mask.device, dtype=tokens["left_state"].dtype),
        R_total,
        t_total,
        presence_mask[:, :, 0],
        "left",
        betas=(_bl.to(device=obs_mask.device) if _bl is not None else None),
    )
    right = _transform_hand_state(
        batch["right_state_world"].to(device=obs_mask.device, dtype=tokens["right_state"].dtype),
        R_total,
        t_total,
        presence_mask[:, :, 1],
        "right",
        betas=(_br.to(device=obs_mask.device) if _br is not None else None),
    )
    obj = _transform_obj_state(
        batch["obj_state_world"].to(device=obs_mask.device, dtype=tokens["obj_state"].dtype),
        R_total,
        t_total,
        presence_mask[:, :, 2],
    )

    mask_hand = has_anchor.view(-1, 1, 1)
    out["left_state"] = torch.where(mask_hand, left, tokens["left_state"])
    out["right_state"] = torch.where(mask_hand, right, tokens["right_state"])
    out["obj_state"] = torch.where(has_anchor.view(-1, 1, 1), obj, tokens["obj_state"])
    return out, present_frames


def reanchor_tokens_to_present_camera(
    tokens: dict[str, Tensor],
    batch: dict,
    obs_mask: Tensor,
    presence_mask: Tensor,
    valid_frames: Tensor | None = None,
) -> tuple[dict[str, Tensor], Tensor]:
    """Re-anchor states to the present camera frame with exact fp32 transforms.

    The physical relation features are invariant to this frame change only if the
    state transform is itself an accurate isometry. Under an outer bf16 autocast the
    einsums would run in bf16 (and TF32 under the ``high`` matmul setting), with the
    rounded values silently assigned into fp32 clones. The re-anchor therefore runs
    with autocast off and IEEE fp32 matmuls, restoring the caller's autocast nesting
    and matmul setting on every exit.

    Returns:
        (tokens, present_frames): re-anchored tokens and the (B,) present frame indices.
    """
    previous_matmul_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        with torch.amp.autocast(obs_mask.device.type, enabled=False):
            return _reanchor_tokens_to_present_camera_impl(tokens, batch, obs_mask, presence_mask, valid_frames)
    finally:
        torch.set_float32_matmul_precision(previous_matmul_precision)
