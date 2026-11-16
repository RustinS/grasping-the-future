"""Rigid transforms of MANO hand parameters (global orient + wrist translation).

`transform_hand_rigid` applies a rigid (R, t) to one hand's parameters so that the hand the FK produces moves rigidly:
the wrist-pivot (root joint J0) correction and, for the left hand, the M-conjugation its mirrored right-model
convention needs. Used by the MoCap ingestion loaders and `gtf/datasets/scene/preprocess.py`.
The model has its own in-package copy of the pivot (`gtf/model/repr/canonicalize.py`).
"""

import numpy as np
from scipy.spatial.transform import Rotation

from gtf.paths import MANO_DIR

_MANO_ROOT_JOINT: dict[str, np.ndarray] = {}


def _get_mano_root_joint(side: str = "right") -> np.ndarray:
    """Return the MANO root-joint offset (meters) for one side, default betas.

    MANO FK rotates vertices around the root joint (wrist), not the origin.
    When transforming MANO params between coordinate frames, the root-joint
    offset must be accounted for to preserve hand-object distances. Left-hand
    params live in right-model (mirrored) coordinates, so transform_hand_rigid
    always asks for the right side. One value is cached per side.
    """
    if side in _MANO_ROOT_JOINT:
        return _MANO_ROOT_JOINT[side]

    try:
        import torch

        from gtf.utils.canonical import _import_manolayer

        ManoLayer = _import_manolayer()
        mano_layer = ManoLayer(side=side, use_pca=False, ncomps=45, flat_hand_mean=True, mano_root=MANO_DIR)
        with torch.no_grad():
            _, joints = mano_layer(torch.zeros(1, 48), torch.zeros(1, 10), torch.zeros(1, 3))
        _MANO_ROOT_JOINT[side] = (joints[0, 0].numpy() / 1000.0).astype(np.float32)  # mm → m
    except Exception:
        # Fallback when the MANO layer cannot be built: the right model's J0 for
        # zero betas, flat_hand_mean=True. Left-hand params live in right-model
        # coordinates, so the same offset serves both sides.
        _MANO_ROOT_JOINT[side] = np.array([0.09567, 0.00638, 0.00619], dtype=np.float32)

    return _MANO_ROOT_JOINT[side]


_MANO_ROOT_JOINT_BETA: dict[bytes, np.ndarray] = {}


def _get_mano_root_joint_beta(betas: np.ndarray) -> np.ndarray:
    """The shaped root joint J0(β): right ManoLayer FK at zero pose with the given
    betas (J0 is linear in β, so this matches the model's
    `gtf.model.repr.canonicalize.mano_root_joint_right_beta` up to float32 rounding).
    Cached per betas value. If the FK fails, zero betas fall back to the zero-beta
    constant and any other betas raise."""
    key = np.asarray(betas, dtype=np.float32).tobytes()
    if key in _MANO_ROOT_JOINT_BETA:
        return _MANO_ROOT_JOINT_BETA[key]
    try:
        import torch

        from gtf.utils.canonical import _import_manolayer

        ManoLayer = _import_manolayer()
        mano_layer = ManoLayer(side="right", use_pca=False, ncomps=45, flat_hand_mean=True, mano_root=MANO_DIR)
        with torch.no_grad():
            _, joints = mano_layer(torch.zeros(1, 48), torch.as_tensor(betas, dtype=torch.float32).view(1, 10), torch.zeros(1, 3))
        _MANO_ROOT_JOINT_BETA[key] = (joints[0, 0].numpy() / 1000.0).astype(np.float32)
    except Exception as e:
        # With non-zero betas the zero-beta pivot would be wrong, not just approximate.
        if float(np.abs(np.asarray(betas, dtype=np.float64)).sum()) > 1e-8:
            raise RuntimeError(f"_get_mano_root_joint_beta: shaped-J0 FK failed ({e!r}); check the torch and MANO installation") from e
        _MANO_ROOT_JOINT_BETA[key] = _get_mano_root_joint("right")
    return _MANO_ROOT_JOINT_BETA[key]


# MANO left-hand reflection. `gtf.utils.canonical.mano_fk_verts(..., "left")` runs the
# RIGHT MANO model and mirrors the output verts about x=0 (M = diag(-1, 1, 1)).
_LEFT_MIRROR = np.array([-1.0, 1.0, 1.0], dtype=np.float32)


def transform_hand_rigid(
    orient: np.ndarray,
    cam_t: np.ndarray,
    valid: np.ndarray,
    side: str,
    R: np.ndarray | None = None,
    t: np.ndarray | None = None,
    betas: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a world-frame rigid transform (R, t) to one hand side's global
    ``orient`` (axis-angle) + wrist ``cam_t``. Mask-zero invariant preserved.

    **Left hand.** The left hand is reconstructed as ``M @ FK_right(...)``:
    ``mano_fk_verts(..., "left")`` runs the right MANO model and mirrors the output
    by ``M = diag(-1, 1, 1)``. A reflection does not commute with a rotation, and
    the FK's translation term is mirrored too, so a rigid (R, t) that moves the
    visible left hand rigidly must be M-conjugated on the params: ``R -> M R M``
    and ``t -> M t``. The right hand takes (R, t) as is.

    The wrist-pivot (J0) correction always uses the **right** root joint because
    the underlying FK is the right model (``J0_left`` would double-count the
    mirror). ``R=None`` means identity (pure translation); ``t=None`` means zero.

    ``betas``: pivot about the shaped root joint J0(β), the exact pivot (MANO
    rotates about the betas-dependent root joint; the zero-beta constant leaves a
    (R−I)(J0(β)−J0(0)) error). None = the zero-beta pivot. The loaders that write
    world-frame fields and the scene preprocessing pass betas, so their fields and
    the present-camera re-anchoring of the model are exact inverses.
    """
    orient = np.asarray(orient, dtype=np.float32)
    cam_t = np.asarray(cam_t, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    orient_out = np.zeros_like(orient)
    cam_t_out = np.zeros_like(cam_t)
    if not valid.any():
        return orient_out, cam_t_out

    R_eff = None if R is None else np.asarray(R, dtype=np.float32)
    t_eff = np.zeros(3, dtype=np.float32) if t is None else np.asarray(t, dtype=np.float32)
    if side == "left":
        m = _LEFT_MIRROR
        if R_eff is not None:
            R_eff = (m[:, None] * R_eff * m[None, :]).astype(np.float32)  # M R M
        t_eff = (m * t_eff).astype(np.float32)  # M t

    # Orientation: compose R_eff with the global orient (unchanged if R is None).
    if R_eff is None:
        orient_out[valid] = orient[valid]
    else:
        for i in range(orient.shape[0]):
            if not valid[i]:
                continue
            R_hand = Rotation.from_rotvec(orient[i]).as_matrix().astype(np.float32)
            orient_out[i] = Rotation.from_matrix(R_eff @ R_hand).as_rotvec().astype(np.float32)

    # Wrist position: rotate about the (right) root joint J0, then translate.
    if betas is not None and np.abs(np.asarray(betas, dtype=np.float32)).sum() > 1e-8:
        J0 = _get_mano_root_joint_beta(np.asarray(betas, dtype=np.float32))
    else:
        J0 = _get_mano_root_joint("right")
    if R_eff is None:
        shifted = cam_t + t_eff
    else:
        shifted = (cam_t + J0) @ R_eff.T + t_eff - J0
    cam_t_out[valid] = shifted[valid].astype(np.float32)
    return orient_out, cam_t_out
