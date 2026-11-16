"""Signed distance field of an object mesh on a regular grid, with differentiable queries.

`compute_object_sdf_grid` builds the grid from a triangle mesh (Open3D ray casting, sign cross-checked with the
generalized winding number); `query_sdf` interpolates it trilinearly with `grid_sample`, so distances are
differentiable in the query points. Used by `contact_projection.py`.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def mesh_to_sdf_grid(
    mesh_verts: np.ndarray,
    mesh_faces: np.ndarray,
    grid_resolution: int = 128,
    padding: float = 0.05,
) -> tuple[torch.Tensor, dict]:
    """Compute signed distance field on a regular grid from a triangle mesh.

    Uses Open3D's raycasting scene: `compute_signed_distance` for the distance, its sign
    checked against the generalized winding number (`compute_occupancy`), which keeps the
    sign right on non-watertight or inconsistently oriented meshes.

    Args:
        mesh_verts: (V, 3) mesh vertices.
        mesh_faces: (F, 3) triangle face indices.
        grid_resolution: Number of grid cells per axis.
        padding: Meters of padding around mesh bounding box.

    Returns:
        sdf_grid: (1, 1, R, R, R) float32 tensor (ready for grid_sample).
        grid_info: dict with 'origin', 'spacing', 'shape', 'bbox_min', 'bbox_max'.
    """
    import open3d as o3d

    mesh_t = o3d.t.geometry.TriangleMesh()
    mesh_t.vertex.positions = o3d.core.Tensor(mesh_verts.astype(np.float32))
    mesh_t.triangle.indices = o3d.core.Tensor(mesh_faces.astype(np.int32))

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)

    # Compute bounding box with padding
    bbox_min = mesh_verts.min(axis=0) - padding
    bbox_max = mesh_verts.max(axis=0) + padding

    # Create regular grid
    R = grid_resolution
    x = np.linspace(bbox_min[0], bbox_max[0], R)
    y = np.linspace(bbox_min[1], bbox_max[1], R)
    z = np.linspace(bbox_min[2], bbox_max[2], R)

    # Grid points (R^3, 3)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    grid_points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.float32)

    query = o3d.core.Tensor(grid_points, dtype=o3d.core.Dtype.Float32)
    # Closest-point distance with a ray-casting sign: negative inside, positive outside.
    sdf_values = scene.compute_signed_distance(query).numpy()

    # The ray-casting sign is unreliable on meshes with inverted faces: where the
    # generalized-winding-number occupancy disagrees with it, the occupancy decides
    # (the magnitude is kept).
    occ = scene.compute_occupancy(query).numpy()
    inside_gwn = occ > 0.5
    disagreement = inside_gwn & (sdf_values > 0)
    sdf_values[disagreement] = -sdf_values[disagreement]
    disagreement_out = (~inside_gwn) & (sdf_values < 0)
    sdf_values[disagreement_out] = -sdf_values[disagreement_out]

    sdf_values = sdf_values.reshape(R, R, R).astype(np.float32)

    # Convert to tensor in grid_sample format: (1, 1, D, H, W)
    sdf_grid = torch.from_numpy(sdf_values).unsqueeze(0).unsqueeze(0)

    spacing = np.array([(bbox_max[i] - bbox_min[i]) / (R - 1) for i in range(3)])

    grid_info = {
        "origin": bbox_min.tolist(),
        "spacing": spacing.tolist(),
        "shape": [R, R, R],
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
    }

    return sdf_grid, grid_info


def query_sdf(
    points: Tensor,
    sdf_grid: Tensor,
    grid_info: dict,
) -> Tensor:
    """Query SDF at arbitrary 3D points via trilinear interpolation (`grid_sample`, differentiable).

    The grid tensor is `(1, 1, D, H, W)` with D = x, H = y, W = z (from
    `meshgrid(x, y, z, indexing='ij')`), and `grid_sample` reads the last-dim
    3-vector as `(w, h, d)`, so the normalized query is passed as `(z, y, x)`.

    Args:
        points: (N, 3) query points in world coordinates, `(x, y, z)`.
        sdf_grid: (1, 1, D, H, W) SDF grid tensor.
        grid_info: dict with 'bbox_min' (3,) and 'bbox_max' (3,).

    Returns:
        (N,) signed distances. Negative = inside mesh (penetrating).
    """
    device = points.device
    sdf_grid = sdf_grid.to(device)

    bbox_min = torch.tensor(grid_info["bbox_min"], device=device, dtype=points.dtype)
    bbox_max = torch.tensor(grid_info["bbox_max"], device=device, dtype=points.dtype)

    # Normalize (x, y, z) each to [-1, 1]
    normalized = 2.0 * (points - bbox_min) / (bbox_max - bbox_min) - 1.0
    normalized = normalized.clamp(-1.0, 1.0)

    reversed_xyz = normalized.flip(-1)  # (z, y, x): grid_sample's (w, h, d)
    grid = reversed_xyz.reshape(1, 1, 1, -1, 3)

    result = F.grid_sample(sdf_grid, grid, mode="bilinear", padding_mode="border", align_corners=True)

    return result.reshape(-1)  # (N,)


def compute_object_sdf_grid(
    obj_canonical_verts: np.ndarray,
    obj_faces: np.ndarray,
    obj_scale: np.ndarray,
    grid_resolution: int = 64,
    padding: float = 0.02,
) -> tuple[torch.Tensor, dict]:
    """Compute SDF for an object in its canonical pose (for hand-object penetration).

    Args:
        obj_canonical_verts: (V, 3) canonical mesh vertices.
        obj_faces: (F, 3) face indices.
        obj_scale: (3,) scale factors.
        grid_resolution: SDF grid resolution.
        padding: padding around object bbox.

    Returns:
        sdf_grid, grid_info (same format as mesh_to_sdf_grid).
    """
    scaled_verts = obj_canonical_verts * obj_scale
    return mesh_to_sdf_grid(scaled_verts, obj_faces, grid_resolution, padding)
