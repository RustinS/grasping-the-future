"""Gravity alignment of canonical.npz fields, and the MANO wrist-pivot helpers it shares.

gravity_align rotates the axis-angle fields so gravity points along +Y; the dataset calls it
before tokenization. Runtime centering at the present frame acts on tokens
(tokenizer.center_tokens_at_present).

Pipeline order:
  canonical.npz (axis-angle) → gravity_align → tokenize (→ 6D) → ... → centering
"""

import torch
from torch import Tensor

from gtf.model.repr.rotation import axis_angle_to_matrix, matrix_to_rotation_6d, rotation_6d_to_axis_angle

# canonical.npz per-frame vertex fields, rotated along with the poses when present
_VERTEX_FIELDS = ["hand_left_verts", "hand_right_verts", "obj_verts"]


def compute_gravity_alignment(gravity_direction: Tensor) -> Tensor:
    """Compute rotation matrix that aligns gravity_direction to +Y axis.

    Convention: Y = gravity direction = downward.

    Args:
        gravity_direction: (3,) unit vector pointing in gravity direction (downward).

    Returns:
        R_align: (3, 3) rotation matrix such that R_align @ gravity_direction ≈ [0, 1, 0].
    """
    device = gravity_direction.device
    dtype = gravity_direction.dtype
    target = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)

    # Rodrigues formula between two unit vectors
    v = torch.linalg.cross(gravity_direction, target)
    s = v.norm()  # sin(angle)
    c = gravity_direction.dot(target)  # cos(angle)

    if s < 1e-6:
        # Nearly aligned or opposite
        if c > 0:
            return torch.eye(3, device=device, dtype=dtype)
        else:
            # 180-degree rotation: find a perpendicular axis
            perp = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
            if gravity_direction[0].abs() > 0.9:
                perp = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
            axis = torch.linalg.cross(gravity_direction, perp)
            axis = axis / axis.norm()
            # R = 2 * outer(axis, axis) - I  (180-degree rotation around axis)
            return 2.0 * axis.unsqueeze(-1) @ axis.unsqueeze(0) - torch.eye(3, device=device, dtype=dtype)

    # Skew-symmetric matrix of v
    vx = torch.zeros(3, 3, device=device, dtype=dtype)
    vx[0, 1] = -v[2]
    vx[0, 2] = v[1]
    vx[1, 0] = v[2]
    vx[1, 2] = -v[0]
    vx[2, 0] = -v[1]
    vx[2, 1] = v[0]

    R_align = torch.eye(3, device=device, dtype=dtype) + vx + vx @ vx * ((1.0 - c) / (s * s))
    return R_align


def _rotate_axis_angle(R_align: Tensor, axis_angle: Tensor) -> Tensor:
    """Apply R_align to axis-angle rotations: R_new = R_align @ R_old, returned as axis-angle.

    Args:
        R_align: (3, 3) alignment rotation.
        axis_angle: (T, 3) axis-angle vectors.

    Returns:
        (T, 3) rotated axis-angle vectors.
    """
    R_old = axis_angle_to_matrix(axis_angle)  # (T, 3, 3)
    R_new = R_align @ R_old  # (T, 3, 3)
    # Round-trip: matrix → 6D → axis-angle
    r6d = matrix_to_rotation_6d(R_new)  # (T, 6)
    return rotation_6d_to_axis_angle(r6d)  # (T, 3)


# MANO left-hand reflection M = diag(-1, 1, 1): mano_fk_verts(..., "left") runs the
# right MANO model and mirrors the output verts about x=0, so the left hand is
# M @ FK_right(...). A world rotation applied to the left hand's params must
# therefore be M-conjugated (R -> M R M); see gtf/utils/hand_rigid.py.
_LEFT_MIRROR_DIAG = (-1.0, 1.0, 1.0)
_J0_RIGHT_CACHE: list[Tensor] = []


def _mano_root_joint_right(device: torch.device, dtype: torch.dtype) -> Tensor:
    """Right-MANO rest root joint J0 (meters) at zero betas, cached once per process.

    Sourced from the model's own MANO (gtf.model.repr.mano.MANOHelper) at zero
    pose/betas — identical to the ingestion-side
    gtf.utils.hand_rigid._get_mano_root_joint("right"), so the wrist pivot stays
    consistent with both the model's FK and transform_hand_rigid. Computed
    in the model package so that gtf.model stays free of the extraction helpers
    (gtf.utils) and their heavy imports.
    """
    if not _J0_RIGHT_CACHE:
        from gtf.model.repr.mano import MANOHelper
        from gtf.model.repr.rotation import axis_angle_to_6d

        mano = MANOHelper()
        with torch.no_grad():
            orient_6d = axis_angle_to_6d(torch.zeros(1, 3))
            pose_6d = axis_angle_to_6d(torch.zeros(1, 15, 3)).reshape(1, 90)
            _, joints, _ = mano(orient_6d, pose_6d, torch.zeros(1, 10), torch.zeros(1, 3), "right")
        _J0_RIGHT_CACHE.append(joints[0, 0].detach().float())
    return _J0_RIGHT_CACHE[0].to(device=device, dtype=dtype)


_J0_RIGHT_BETA_BASIS_CACHE: list = []


def mano_root_joint_right_beta(betas: Tensor, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Right-MANO rest root joint J0(betas): (..., 10) → (..., 3).

    MANO joint locations regress from the SHAPED rest template, so J0 is exactly
    linear in betas: j0(β) = j0₀ + S·β. The (3, 10) basis S is extracted once per
    process by unit-beta FK differences (exact for a linear map). Rigid transforms
    of hand parameters pivot about this shaped root, which keeps the physical
    relation features invariant under the transform.
    """
    if not _J0_RIGHT_BETA_BASIS_CACHE:
        from gtf.model.repr.mano import MANOHelper
        from gtf.model.repr.rotation import axis_angle_to_6d

        mano = MANOHelper()
        with torch.no_grad():
            orient_6d = axis_angle_to_6d(torch.zeros(11, 3))
            pose_6d = axis_angle_to_6d(torch.zeros(11, 15, 3)).reshape(11, 90)
            b = torch.zeros(11, 10)
            b[1:] = torch.eye(10)
            _, joints, _ = mano(orient_6d, pose_6d, b, torch.zeros(11, 3), "right")
        j0_0 = joints[0, 0].detach().float()  # (3,)
        S = (joints[1:, 0].detach().float() - j0_0.unsqueeze(0)).T.contiguous()  # (3, 10)
        _J0_RIGHT_BETA_BASIS_CACHE.extend([j0_0, S])
    j0_0, S = _J0_RIGHT_BETA_BASIS_CACHE
    j0_0 = j0_0.to(device=device, dtype=dtype)
    S = S.to(device=device, dtype=dtype)
    return j0_0 + betas.to(device=device, dtype=dtype) @ S.T


def _rotate_hand_params(R_align: Tensor, orient_aa: Tensor, cam_t: Tensor, valid: Tensor | None, side: str, betas: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Apply a world rotation R_align to one hand's global orient + wrist cam_t.

    A raw rotation of the params is wrong for a MANO hand on two counts:
      * Wrist pivot — the global orient rotates the hand about the wrist joint J0,
        so the wrist translation must pivot: cam_t -> R_eff @ (cam_t + J0) - J0.
      * Left reflection — the left hand is M @ FK_right(...), so the rotation must
        be M-conjugated: R_eff = M R_align M (right hand: R_eff = R_align).
    This is transform_hand_rigid(orient, cam_t, valid, side, R=R_align)
    (gtf/utils/hand_rigid.py) specialised to a pure rotation (t=0).
    Invalid frames are zeroed to preserve the mask-zero invariant (the pivot term
    R_eff @ J0 - J0 is non-zero on a zero input).

    With ``betas`` the pivot is the SHAPED root J0(β) (mano_root_joint_right_beta),
    which makes the transform exactly rigid for shaped hands; None uses the
    zero-beta root.
    """
    if side == "left":
        m = torch.diag(torch.tensor(_LEFT_MIRROR_DIAG, device=R_align.device, dtype=R_align.dtype))
        R_eff = m @ R_align @ m  # M R M
    else:
        R_eff = R_align

    # Orientation: R_new = R_eff @ R_old (matrix -> 6D -> axis-angle, as _rotate_axis_angle).
    R_new = torch.matmul(R_eff, axis_angle_to_matrix(orient_aa))  # (T, 3, 3)
    orient_out = rotation_6d_to_axis_angle(matrix_to_rotation_6d(R_new))  # (T, 3)

    # Wrist position: rotate about the (right) root joint J0, then place.
    # Row-vector form: x @ R_eff^T == R_eff @ x.
    if betas is not None:
        j0 = mano_root_joint_right_beta(betas.view(1, 10), cam_t.device, cam_t.dtype).view(3)
    else:
        j0 = _mano_root_joint_right(cam_t.device, cam_t.dtype)
    cam_t_out = (cam_t + j0) @ R_eff.transpose(-1, -2) - j0  # (T, 3)

    if valid is None:
        valid = (orient_aa.abs().sum(-1) + cam_t.abs().sum(-1)) > 0
    v = valid.to(torch.bool).unsqueeze(-1)
    orient_out = torch.where(v, orient_out, torch.zeros_like(orient_out))
    cam_t_out = torch.where(v, cam_t_out, torch.zeros_like(cam_t_out))
    return orient_out, cam_t_out


def gravity_align(data_dict: dict[str, Tensor], gravity_direction: Tensor | None) -> dict[str, Tensor]:
    """Apply gravity alignment rotation to canonical.npz fields.

    Operates on axis-angle rotation fields (BEFORE tokenization to 6D).
    hand_pose fields (parent-relative joint angles) are NEVER touched.

    Args:
        data_dict: Dict with canonical.npz field names. Modified fields:
            - Rotation fields (axis-angle): hand_{side}_orient, obj_rotation
            - Translation fields: hand_{side}_cam_t, obj_translation
            - Vertex fields (if present): hand_{side}_verts, obj_verts
        gravity_direction: (3,) unit gravity vector, or None to skip.

    The hand wrist pivot uses the shaped root J0(β) from ``hand_{side}_betas``
    (zero or missing betas use the zero-beta root).

    Returns:
        New dict with aligned fields. Unmodified fields are shared (shallow copy).
    """
    if gravity_direction is None:
        return data_dict

    R_align = compute_gravity_alignment(gravity_direction)
    result = dict(data_dict)  # shallow copy

    # Object: rigid rotation about its own origin — raw R_align is exact for both
    # the axis-angle rotation (R_new = R_align @ R_old) and the translation.
    if result.get("obj_rotation") is not None:
        result["obj_rotation"] = _rotate_axis_angle(R_align, result["obj_rotation"])
    if result.get("obj_translation") is not None:
        result["obj_translation"] = (R_align @ result["obj_translation"].unsqueeze(-1)).squeeze(-1)

    # Hands: wrist-pivot + left-mirror aware (NOT a raw rotation of orient/cam_t),
    # because the tokenizer/model/eval consume the PARAMS, not the verts.
    # See _rotate_hand_params.
    for side in ("left", "right"):
        if result.get(f"hand_{side}_orient") is not None:
            _betas = result.get(f"hand_{side}_betas")
            if _betas is not None and _betas.abs().sum() < 1e-8:
                _betas = None  # zero betas ≡ the constant; skip the linear map
            result[f"hand_{side}_orient"], result[f"hand_{side}_cam_t"] = _rotate_hand_params(
                R_align,
                result[f"hand_{side}_orient"],
                result[f"hand_{side}_cam_t"],
                result.get(f"hand_{side}_valid"),
                side,
                betas=(_betas.float().view(-1)[:10] if _betas is not None else None),
            )

    # Rotate vertex fields if present (for verification/visualization)
    for key in _VERTEX_FIELDS:
        if key in result and result[key] is not None:
            # verts: (T, V, 3) → (T, V, 3)
            result[key] = (R_align @ result[key].transpose(-1, -2)).transpose(-1, -2)

    return result
