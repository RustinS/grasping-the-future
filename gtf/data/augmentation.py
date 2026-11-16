"""Global rigid (SE(3)) augmentation for GTF training.

One rigid transform per window — a yaw about the gravity axis plus a
translation — applied identically to both hands, the object, the scene point
cloud and the support plane. A shared isometry preserves contacts, support
geometry, contact labels and confidence features exactly while varying the
absolute frame.

Integration: called in ``GTFLitModule._shared_step`` on uncentered tokens
(after the present-camera re-anchor, before centering and normalization), under
``autocast(enabled=False)``.
"""

import torch
from torch import Tensor

from gtf.model.repr.mano import MANOHelper
from gtf.model.repr.relation import compute_relation_features
from gtf.model.repr.rotation import matrix_to_rotation_6d, rotation_6d_to_matrix
from gtf.model.repr.tokenizer import (
    HAND_CAM_T_SLICE,
    HAND_ORIENT_SLICE,
    HAND_POSE_SLICE,
    OBJ_ROTATION_SLICE,
    OBJ_TRANSLATION_SLICE,
)


def _transform_hand_tokens_rigid(orient_6d: Tensor, cam_t: Tensor, valid: Tensor, side: str, R: Tensor, t: Tensor, betas: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Apply a rigid (R, t) to one hand's 6D orient + cam_t tokens (one sample, (T, ·)).

    Left-hand convention: the left hand is M @ FK_right(...) with M = diag(-1, 1, 1),
    so the transform is M-conjugated (R_eff = M R M, t_eff = M t) and the wrist
    pivots about the RIGHT-model root joint J0. With ``betas`` the pivot is the
    shaped root J0(β); a zero-beta pivot would leave a (R−I)(J0(β)−J0(0))
    residual for shaped hands. Invalid frames are zeroed (mask-zero invariant —
    the J0 pivot is non-zero on zero input).
    """
    from gtf.model.repr.canonicalize import _LEFT_MIRROR_DIAG, _mano_root_joint_right, mano_root_joint_right_beta

    if side == "left":
        m = torch.diag(torch.tensor(_LEFT_MIRROR_DIAG, device=R.device, dtype=R.dtype))
        R_eff = m @ R @ m
        t_eff = torch.tensor(_LEFT_MIRROR_DIAG, device=R.device, dtype=R.dtype) * t
    else:
        R_eff, t_eff = R, t

    R_new = torch.matmul(R_eff, rotation_6d_to_matrix(orient_6d))  # (T, 3, 3)
    orient_out = matrix_to_rotation_6d(R_new)
    if betas is not None and bool(betas.abs().sum() > 1e-8):
        j0 = mano_root_joint_right_beta(betas.float().view(1, 10), cam_t.device, cam_t.dtype).view(3)
    else:
        j0 = _mano_root_joint_right(cam_t.device, cam_t.dtype)
    cam_t_out = (cam_t + j0) @ R_eff.transpose(-1, -2) - j0 + t_eff

    v = valid.to(torch.bool).unsqueeze(-1)
    return torch.where(v, orient_out, torch.zeros_like(orient_out)), torch.where(v, cam_t_out, torch.zeros_like(cam_t_out))


def apply_global_se3_augmentation(
    state_tokens: dict[str, Tensor],
    *,
    presence_mask: Tensor,
    valid_frames: Tensor,
    betas_left: Tensor,
    betas_right: Tensor,
    obj_canonical_verts_list: list[Tensor],
    obj_scale: Tensor,
    mano_helper: MANOHelper,
    scene_features: list[dict | None] | None = None,
    centering_origin: Tensor | None = None,
    support_plane_y: Tensor | None = None,
    has_support_plane: Tensor | None = None,
    prob: float = 0.5,
    yaw_sigma: float = 0.35,
    trans_sigma: float = 0.0,
    generator: torch.Generator | None = None,
    obj_faces_list: list[Tensor] | None = None,
) -> tuple[dict[str, Tensor], list[dict | None] | None, Tensor | None]:
    """One shared rigid transform per window (applied with probability ``prob``).

    ``state_tokens`` are the uncentered state/relation tokens. States pivot about the
    world point ``centering_origin`` (the present-frame centering origin)
    (t_state = org − R·org + t), the same pivot the scene uses:
    P' = R (P − origin) + origin + t. Yaw preserves every y-coordinate, so the
    h_support / theta_grav / d_obj / d_hand / curvature / density / semantic
    scene channels are invariant; only P_scene, the xyz copy in
    scene_geo_feats[:, 0:3], the normals in [:, 3:6] and the standalone
    scene_normals change. support_plane_y shifts by t_y.

    The physical relation tokens (FK root joint + point-to-triangle fingertip
    distances) are invariant to a shared isometry, so for fp32 state tokens the
    input relation values are retained instead of recomputing MANO FK and
    triangle queries. That state transform runs with IEEE fp32 matmuls
    (``highest`` precision): TF32 would move the geometry enough to invalidate the
    retained values near the 5 mm contact boundary. Non-fp32 state tokens take
    the recompute path. Contact labels and confidence are not touched.

    Returns (tokens, scene_features, support_plane_y) — lists/tensors are copied
    where mutated (batch references may be shared with the segment cache).
    """
    B, T = presence_mask.shape[:2]
    device = presence_mask.device
    dtype = state_tokens["obj_state"].dtype

    fire = torch.rand(B, device=device, generator=generator) < prob
    tokens = {k: v.clone() for k, v in state_tokens.items()}
    scene_features = list(scene_features) if scene_features is not None else None
    if support_plane_y is not None:
        support_plane_y = support_plane_y.clone()

    if not fire.any():
        return tokens, scene_features, support_plane_y

    if centering_origin is None:
        raise ValueError("apply_global_se3_augmentation requires centering_origin")
    retain_invariant_relations = all(state_tokens[key].dtype == torch.float32 for key in ("left_state", "right_state", "obj_state"))

    # The translation draw is made even when trans_sigma == 0 so the RNG stream
    # does not depend on its value.
    theta = torch.randn(B, device=device, generator=generator) * yaw_sigma
    t_all = torch.randn(B, 3, device=device, generator=generator) * trans_sigma

    origin_cpu = centering_origin.detach().to("cpu", torch.float32)

    def _transform_state_sample(b: int) -> tuple[Tensor, Tensor]:
        c, s = torch.cos(theta[b]), torch.sin(theta[b])
        R = torch.stack(
            [
                torch.stack([c, torch.zeros_like(c), s]),
                torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype),
                torch.stack([-s, torch.zeros_like(c), c]),
            ]
        ).to(dtype)
        t = t_all[b].to(dtype)
        # States pivot about the world point centering_origin — the same pivot the
        # scene transform uses (R(P − org) + org + t == R·P + t_state).
        org_b = centering_origin[b].to(device=device, dtype=dtype)
        t_state = org_b - R @ org_b + t

        # Object state.
        obj = tokens["obj_state"][b]
        obj_valid = presence_mask[b, :, 2] & valid_frames[b]
        rot_new = matrix_to_rotation_6d(torch.matmul(R, rotation_6d_to_matrix(obj[:, OBJ_ROTATION_SLICE])))
        trans_new = obj[:, OBJ_TRANSLATION_SLICE] @ R.transpose(-1, -2) + t_state
        obj_new = torch.cat([rot_new, trans_new], dim=-1)
        tokens["obj_state"][b] = torch.where(obj_valid.unsqueeze(-1), obj_new, obj)

        # Hand states (M-conjugated left, shaped J0(β) wrist pivot).
        for side, slot in (("left", 0), ("right", 1)):
            key = f"{side}_state"
            st = tokens[key][b]
            hv = presence_mask[b, :, slot] & valid_frames[b]
            if not hv.any():
                continue
            _betas_b = (betas_left if side == "left" else betas_right)[b]
            o_new, t_new = _transform_hand_tokens_rigid(st[:, HAND_ORIENT_SLICE], st[:, HAND_CAM_T_SLICE], hv, side, R, t_state, betas=_betas_b)
            st_new = st.clone()
            st_new[:, HAND_ORIENT_SLICE] = o_new
            st_new[:, HAND_CAM_T_SLICE] = t_new
            # finger pose is body-relative — invariant under a global rigid transform
            tokens[key][b] = torch.where(hv.unsqueeze(-1), st_new, st)

        return R, t

    def _transform_scene_support_sample(b: int, R: Tensor, t: Tensor) -> None:
        # Scene: rotate about the world-frame centering origin, then translate.
        if scene_features is not None and scene_features[b] is not None:
            org = origin_cpu[b].to(dtype)
            sf = dict(scene_features[b])
            R_cpu = R.to("cpu", sf["P_scene"].dtype)
            t_cpu = t.to("cpu", sf["P_scene"].dtype)
            org_cpu = org.to("cpu", sf["P_scene"].dtype)
            P = (sf["P_scene"] - org_cpu) @ R_cpu.transpose(-1, -2) + org_cpu + t_cpu
            sf["P_scene"] = P
            geo = sf["scene_geo_feats"].clone()
            geo[:, 0:3] = P  # xyz copy — must track P_scene exactly
            geo[:, 3:6] = geo[:, 3:6] @ R_cpu.transpose(-1, -2)  # normals rotate (unit-preserved)
            sf["scene_geo_feats"] = geo
            sn = sf.get("scene_normals")
            if sn is not None:
                sf["scene_normals"] = sn @ R_cpu.transpose(-1, -2)
            scene_features[b] = sf

        # Support plane height shifts by t_y (yaw preserves heights).
        if support_plane_y is not None:
            hsp = bool(has_support_plane[b]) if has_support_plane is not None else True
            if hsp:
                support_plane_y[b] = support_plane_y[b] + t[1].to(support_plane_y.dtype)

    if retain_invariant_relations:
        # Training sets matmul precision "high" globally (TF32 on CUDA). Scope IEEE
        # fp32 matmuls to the small state transform and restore the caller's
        # setting even if the transform raises. Autocast is disabled here as well:
        # fp32 inputs alone do not stop autocast matmuls returning bf16 values.
        previous_matmul_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        transforms: list[tuple[Tensor, Tensor] | None] = [None] * B
        try:
            with torch.amp.autocast(device.type, enabled=False):
                for b in range(B):
                    if fire[b]:
                        transforms[b] = _transform_state_sample(b)
        finally:
            torch.set_float32_matmul_precision(previous_matmul_precision)
        # The CPU scene transform and support bookkeeping run outside the
        # precision override, with the exact R/t used for the states.
        for b, transform in enumerate(transforms):
            if transform is not None:
                _transform_scene_support_sample(b, *transform)
        # ``tokens`` cloned every input tensor above, so the relation tensors are
        # retained without aliasing the caller's tensors.
        return tokens, scene_features, support_plane_y

    for b in range(B):
        if not fire[b]:
            continue
        R, t = _transform_state_sample(b)
        _transform_scene_support_sample(b, R, t)

        # Recompute relation tokens from the transformed states.
        obj_rot_6d_b = tokens["obj_state"][b, :, OBJ_ROTATION_SLICE]
        obj_trans_b = tokens["obj_state"][b, :, OBJ_TRANSLATION_SLICE]
        canon_verts_b = obj_canonical_verts_list[b].to(device=device, dtype=dtype)
        for side, slot, rel_key in (("left", 0, "left_rel"), ("right", 1, "right_rel")):
            hand_present = presence_mask[b, :, slot]
            if not hand_present.any():
                continue
            st = tokens[f"{side}_state"][b]
            betas_b = betas_left[b] if side == "left" else betas_right[b]
            with torch.no_grad():
                rel = compute_relation_features(
                    st[:, HAND_ORIENT_SLICE],
                    st[:, HAND_CAM_T_SLICE],
                    st[:, HAND_POSE_SLICE],
                    betas_b,
                    side,
                    obj_rot_6d_b,
                    obj_trans_b,
                    canon_verts_b,
                    obj_scale[b],
                    mano_helper,
                    obj_faces=(obj_faces_list[b].to(device) if obj_faces_list is not None else None),
                )
            rel[~hand_present] = 0.0
            tokens[rel_key][b] = rel

    return tokens, scene_features, support_plane_y
