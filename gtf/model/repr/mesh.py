"""Shared mesh-topology validation for model-side geometry consumers."""

from __future__ import annotations

import torch
from torch import Tensor


def sanitize_mesh_faces(
    vertices: Tensor,
    faces: Tensor,
    *,
    context: str = "object mesh",
) -> Tensor:
    """Return an ordered face tensor with repeated-index triangles removed.

    Point-to-triangle CUDA kernels assume every indexed primitive is a genuine
    triangle. A face that references the same vertex more than once has no
    surface measure but can produce NaN distances and gradients in Kaolin, so it
    is dropped; the represented surface and the order of the remaining faces are
    preserved.

    Distinct-index collinear (zero-area) triangles are kept: MoCap meshes contain
    them, and Kaolin treats some as line-like primitives that contribute to the
    fingertip distances.

    Meant for per-segment preprocessing. The topology is validated fail-closed,
    so malformed indices never reach asynchronous CUDA kernels.
    """
    if vertices.ndim != 2 or vertices.shape[-1] != 3:
        raise ValueError(f"{context}: vertices must have shape (V, 3), got {tuple(vertices.shape)}")
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError(f"{context}: faces must have shape (F, 3), got {tuple(faces.shape)}")
    if vertices.shape[0] == 0:
        raise ValueError(f"{context}: mesh has no vertices")
    if faces.shape[0] == 0:
        raise ValueError(f"{context}: mesh has no faces")
    if not vertices.is_floating_point():
        raise TypeError(f"{context}: vertices must be floating point, got {vertices.dtype}")
    if faces.is_floating_point() or faces.is_complex() or faces.dtype == torch.bool:
        raise TypeError(f"{context}: faces must have an integer dtype, got {faces.dtype}")
    if not bool(torch.isfinite(vertices).all()):
        raise ValueError(f"{context}: vertices contain NaN or Inf")

    faces_long = faces.long()
    face_min = int(faces_long.min())
    face_max = int(faces_long.max())
    if face_min < 0 or face_max >= vertices.shape[0]:
        raise IndexError(f"{context}: face indices out of range [0, {vertices.shape[0] - 1}] (observed [{face_min}, {face_max}])")

    distinct = (faces_long[:, 0] != faces_long[:, 1]) & (faces_long[:, 1] != faces_long[:, 2]) & (faces_long[:, 0] != faces_long[:, 2])
    clean = faces_long[distinct].contiguous()
    if clean.shape[0] == 0:
        raise ValueError(f"{context}: all {faces_long.shape[0]} faces are degenerate")
    return clean
