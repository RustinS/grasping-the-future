"""Compute 14D hand-object relation features.

Per hand, per frame: wrist_in_obj_position(3) + wrist_in_obj_rotation(6) + fingertip_distances(5) = 14D.

Calling convention: per-sample only (V varies per segment). The caller loops over the batch
and calls this function with leading dim T.

Used in dataset preprocessing (after tokenization to 6D), augmentation-time recomputation,
statistics, and recomputation from decoded geometry.
"""

import torch
from torch import Tensor

from gtf.model.repr.mano import MANOHelper
from gtf.model.repr.rotation import matrix_to_rotation_6d, rotation_6d_to_matrix


def _fingertip_surface_distances(fingertips_canon: Tensor, scaled_verts: Tensor, obj_faces: Tensor) -> Tensor:
    """True point-to-TRIANGLE distances from canonical-frame fingertips to the static
    scaled canonical mesh. One call per hand per segment (the caller back-transforms
    the fingertips by the per-frame object pose — a rigid map, so distances to the
    posed mesh equal distances to the canonical mesh).

    Args:
        fingertips_canon: (T, 5, 3) fingertips expressed in the canonical mesh frame.
        scaled_verts: (V, 3) canonical verts x scale (the metric mesh).
        obj_faces: (F, 3) long.

    Returns:
        (T, 5) unsigned surface distances.

    CUDA tensors use Kaolin's triangle-distance kernel on GPUs with compute capability below 10; CPU tensors and
    newer GPUs use libigl on the CPU. Both are exact point-to-triangle distances; the result keeps the input's
    device and dtype.
    """
    T = fingertips_canon.shape[0]
    out_dtype = fingertips_canon.dtype
    pts = fingertips_canon.reshape(1, T * 5, 3)
    if fingertips_canon.is_cuda and torch.cuda.get_device_capability(fingertips_canon.device)[0] < 10:
        from kaolin.metrics.trianglemesh import point_to_mesh_distance
        from kaolin.ops.mesh import index_vertices_by_faces

        # fp32 + autocast-off: kaolin's triangle-distance CUDA kernel has no bf16
        # implementation, and the aug-time recompute runs under Lightning's
        # bf16-mixed autocast.
        with torch.amp.autocast("cuda", enabled=False):
            fv = index_vertices_by_faces(scaled_verts.float().unsqueeze(0), obj_faces.long())  # (1, F, 3, 3)
            sq, _, _ = point_to_mesh_distance(pts.float().contiguous(), fv)
        # sqrt at exactly 0 has a NaN gradient (a fingertip exactly on a triangle);
        # the 1e-12 floor (<=1e-3 mm distance perturbation) keeps it finite.
        return sq.clamp(min=1e-12).sqrt().reshape(T, 5).to(out_dtype)
    import igl
    import numpy as np

    sq, _, _ = igl.point_mesh_squared_distance(
        pts.squeeze(0).detach().float().cpu().numpy().astype(np.float64),
        scaled_verts.detach().float().cpu().numpy().astype(np.float64),
        obj_faces.detach().cpu().numpy().astype(np.int64),
    )
    d = torch.from_numpy(np.sqrt(np.maximum(sq, 1e-12))).to(dtype=out_dtype, device=fingertips_canon.device)
    return d.reshape(T, 5)


def compute_relation_features(
    hand_orient_6d: Tensor,
    hand_cam_t: Tensor,
    hand_pose_6d: Tensor,
    hand_betas: Tensor,
    hand_side: str,
    obj_rotation_6d: Tensor,
    obj_translation: Tensor,
    obj_canonical_verts: Tensor,
    obj_scale: Tensor,
    mano_helper: MANOHelper,
    obj_faces: Tensor | None = None,
) -> Tensor:
    """Compute 14D relation features between one hand and the object.

    All inputs are in meters, post-tokenization (6D rotations).

    Dims 0:9 come from the PHYSICAL wrist (FK root joint, mirrored for the left
    hand) and the M-conjugated left rotation, so they are invariant under any rigid
    camera change. The inputs must describe an UNCENTERED configuration (centering
    shifts the mirrored left cam_t differently from the object, which corrupts the
    FK-based dims). Dims 9:14 are true point-to-triangle fingertip distances to the
    object surface (kaolin on CUDA, igl on CPU); contact labels and confidence
    derive from them.

    Args:
        hand_orient_6d: (T, 6) global wrist orientation in 6D.
        hand_cam_t: (T, 3) wrist translation in meters.
        hand_pose_6d: (T, 90) finger pose in 6D (15 joints x 6).
        hand_betas: (10,) MANO shape params, constant per segment.
        hand_side: "left" or "right".
        obj_rotation_6d: (T, 6) object rotation in 6D.
        obj_translation: (T, 3) object translation in meters.
        obj_canonical_verts: (V, 3) rest-pose mesh vertices, meters.
        obj_scale: (3,) per-axis scale, constant per segment.
        mano_helper: MANOHelper instance for forward kinematics.
        obj_faces: (F, 3) long object faces (required).

    Returns:
        (T, 14) relation features: [wrist_in_obj(3) | wrist_rot_in_obj(6) | fingertip_dist(5)].
    """
    if obj_faces is None or obj_faces.numel() < 3:
        raise ValueError("compute_relation_features requires obj_faces")
    T = hand_orient_6d.shape[0]

    # --- 0. MANO FK ---
    betas_expanded = hand_betas.unsqueeze(0).expand(T, -1)  # (T, 10)
    _, joints, fingertips = mano_helper(hand_orient_6d, hand_pose_6d, betas_expanded, hand_cam_t, side=hand_side)
    # joints: (T, 21, 3) PHYSICAL positions (left already x-mirrored); fingertips: (T, 5, 3)

    obj_R = rotation_6d_to_matrix(obj_rotation_6d)  # (T, 3, 3)
    obj_R_inv = obj_R.transpose(-1, -2)  # (T, 3, 3) — inverse for proper rotations
    hand_R = rotation_6d_to_matrix(hand_orient_6d)  # (T, 3, 3)

    # --- 1. Physical wrist (FK root joint) in object frame ---
    delta = joints[:, 0] - obj_translation  # (T, 3)
    # --- 2. Physical hand rotation in object frame — the left param rotation is
    # M-conjugated to the proper physical rotation (det(MRM)=+1), which transforms
    # like a point rotation under any rigid change.
    if hand_side == "left":
        m = torch.tensor([-1.0, 1.0, 1.0], device=hand_R.device, dtype=hand_R.dtype)
        hand_R = m.view(1, 3, 1) * hand_R * m.view(1, 1, 3)

    wrist_in_obj = (obj_R_inv @ delta.unsqueeze(-1)).squeeze(-1)  # (T, 3)
    wrist_rot_in_obj_mat = obj_R_inv @ hand_R  # (T, 3, 3)
    wrist_rot_in_obj = matrix_to_rotation_6d(wrist_rot_in_obj_mat)  # (T, 6)

    # --- 3. Fingertip distances (5D) ---
    scaled_verts = obj_canonical_verts * obj_scale  # (V, 3) broadcast
    # Pose-invariance: dist(tip, R·mesh + t) == dist(Rᵀ(tip − t), mesh) — one
    # static-mesh query for the whole trajectory instead of T posed meshes.
    tips_canon = torch.einsum("tij,tfj->tfi", obj_R_inv, fingertips - obj_translation.unsqueeze(-2))  # (T, 5, 3)
    fingertip_dists = _fingertip_surface_distances(tips_canon, scaled_verts, obj_faces)  # (T, 5)

    # --- 4. Concatenate ---
    return torch.cat([wrist_in_obj, wrist_rot_in_obj, fingertip_dists], dim=-1)  # (T, 14)
