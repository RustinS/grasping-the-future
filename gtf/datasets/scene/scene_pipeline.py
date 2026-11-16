"""Scene feature pipeline: one present-frame scene point cloud per canonical frame.

Given a frame's depth + RGB + hand/object masks and the gravity-aligned canonical
data, ``prepare_scene_cloud`` unprojects, aligns and crops the scene;
``batch_fps_indices`` selects the points; ``finish_scene_features`` computes the
per-point features (CPU only, safe in a spawn worker).

Invariants:
- All points are rotated by the same ``R_world_from_cam`` that ``gravity_align``
  applied to the canonical hand/object verts, so scene points and canonical verts
  share the ``+Y = down`` gravity frame.
- Hand and object masks are combined into one invalid-pixel mask before the depth
  validity mask is built.
- The DINOv3 semantic channel uses one global PCA basis for the whole corpus
  (``gtf/datasets/scene/fit_pca.py``), so channel meanings agree across segments.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree

from gtf.model.repr.mesh import sanitize_mesh_faces

logger = logging.getLogger(__name__)

# A frame needs at least this many scene points after cropping.
MIN_POINTS_AFTER_CROP = 500
MAX_POINTS_BEFORE_NORMALS = 30000
_PIXEL_GRID_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
_SCENE_THREADPOOL_LIMITER: Any | None = None


@dataclass
class PreparedSceneCloud:
    """Pickle-safe CPU work item for scene-feature construction.

    Holds no CUDA objects, so it can cross a ``spawn`` process boundary. It keeps
    the full point cloud: Open3D's normal signs depend on the full-cloud graph,
    while every other feature is evaluated at the FPS indices only.
    """

    pts_world: np.ndarray
    pix_uv: np.ndarray
    obj_verts_world: np.ndarray
    obj_faces: np.ndarray | None
    hand_verts_world: np.ndarray
    hand_faces: np.ndarray | None
    support_plane_y: float | None


@dataclass
class SceneFeatureFinishTask:
    """One pickle-safe CPU-tail task for a persistent spawn worker.

    ``task_id`` is deliberately opaque to the worker. The caller should use a
    source frame index (or another stable ordinal) so asynchronously completed
    results can be restored to their original order before writing.
    """

    task_id: int
    prepared: PreparedSceneCloud
    fps_indices: np.ndarray
    normal_k: int = 30
    pca_basis: dict[str, np.ndarray] | None = None
    dino_patch_features: np.ndarray | None = None
    dino_patch_size: int = 16
    dino_patch_features_projected: bool = False


@dataclass
class SceneFeatureFinishResult:
    """A completed CPU-tail payload tagged with its caller-owned ordinal."""

    task_id: int
    payload: dict[str, Any]


def _sanitized_scene_obj_faces(
    canonical_aligned: dict[str, Any],
    obj_verts_world: np.ndarray,
    obj_faces: Any,
) -> np.ndarray | None:
    """Sanitize object topology against the stored canonical geometry.

    Scene distances use posed vertices, but repeated-index topology is a
    canonical-mesh property and is invariant to the pose.
    """
    if obj_faces is None:
        return None
    topology_verts = canonical_aligned.get("obj_canonical_verts")
    if topology_verts is None:
        topology_verts = obj_verts_world
    topology_verts_t = torch.as_tensor(topology_verts).detach().cpu().float()
    faces_t = torch.as_tensor(obj_faces).detach().cpu()
    return (
        sanitize_mesh_faces(
            topology_verts_t,
            faces_t,
            context="scene object mesh",
        )
        .numpy()
        .astype(np.int32, copy=False)
    )


def _pixel_grids(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    key = (int(height), int(width))
    grids = _PIXEL_GRID_CACHE.get(key)
    if grids is None:
        u_grid, v_grid = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        grids = (u_grid, v_grid)
        _PIXEL_GRID_CACHE[key] = grids
    return grids


def _uniform_subset_indices(n: int, max_n: int) -> np.ndarray:
    if n <= max_n:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, int(max_n), dtype=np.int64)


def _compute_midpoint_aligned(canonical_aligned: dict[str, Any], anchor_idx: int) -> np.ndarray:
    """Crop center at the anchor frame in the gravity-aligned frame.

    midpoint(mean(present hands), object), the model's centering convention. The left
    cam_t is a mirrored-frame parameter (M = diag(-1,1,1) reflects the whole left FK
    output), so its physical position is M·cam_t.
    """
    hands = []
    for side in ("left", "right"):
        valid_key = f"hand_{side}_valid"
        if valid_key in canonical_aligned and bool(canonical_aligned[valid_key][anchor_idx]):
            t = canonical_aligned[f"hand_{side}_cam_t"][anchor_idx]
            if hasattr(t, "numpy"):
                t = t.numpy()
            t = np.asarray(t, dtype=np.float32)
            if side == "left":
                t = t * np.array([-1.0, 1.0, 1.0], dtype=np.float32)  # mirrored-frame → physical
            hands.append(t)
    obj_t = canonical_aligned["obj_translation"][anchor_idx]
    if hasattr(obj_t, "numpy"):
        obj_t = obj_t.numpy()
    obj_t = np.asarray(obj_t, dtype=np.float32)
    if not hands:
        return obj_t.astype(np.float32)
    hand_mean = np.mean(np.stack(hands, axis=0), axis=0)
    return (0.5 * (hand_mean + obj_t)).astype(np.float32)


def _gather_anchor_hand_verts(canonical_aligned: dict[str, Any], anchor_idx: int) -> np.ndarray:
    parts = []
    for side in ("left", "right"):
        valid_key = f"hand_{side}_valid"
        verts_key = f"hand_{side}_verts"
        if valid_key in canonical_aligned and verts_key in canonical_aligned and bool(canonical_aligned[valid_key][anchor_idx]):
            v = canonical_aligned[verts_key][anchor_idx]
            if hasattr(v, "numpy"):
                v = v.numpy()
            parts.append(np.asarray(v, dtype=np.float32))
    if not parts:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(parts, axis=0)


_PYTORCH3D_CUDA_FPS_UNSUPPORTED = False


def _open3d_fps_indices(pts: np.ndarray, n_points: int) -> np.ndarray:
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd_fps = pcd.farthest_point_down_sample(n_points)
    fps_np = np.asarray(pcd_fps.points, dtype=np.float32)
    kd = cKDTree(pts)
    _, idx = kd.query(fps_np, k=1)
    return idx.astype(np.int64)


def _is_unsupported_cuda_kernel(error: BaseException) -> bool:
    message = str(error).lower()
    return "no kernel image is available" in message or "cudaerrornokernelimage" in message


def _padded_indices(n: int, n_points: int) -> np.ndarray:
    """All ``n <= n_points`` points, padded to ``n_points`` by resampling with ``RandomState(0)``."""
    idx = np.arange(n)
    if n < n_points:
        extra = np.random.RandomState(0).choice(n, n_points - n, replace=True)
        idx = np.concatenate([idx, extra])
    return idx


def batch_fps_indices(
    point_clouds: list[np.ndarray],
    n_points: int,
    *,
    device: torch.device | str | None = None,
) -> list[np.ndarray]:
    """Farthest-point sampling for a ragged batch of point clouds.

    Clouds with at most ``n_points`` points are padded by resampling (each cloud gets
    its own ``RandomState(0)``). Longer clouds are padded only for transport to
    PyTorch3D; ``lengths`` excludes the padding, so the indices equal one
    ``sample_farthest_points`` call per cloud.

    On a PyTorch3D failure the Open3D FPS is used instead. A missing CUDA kernel
    (PyTorch3D on SM100 GPUs) is remembered per process so later batches skip the
    failing launch.
    """
    if not point_clouds:
        return []
    n_points = int(n_points)
    if n_points <= 0:
        raise ValueError(f"n_points must be positive, got {n_points}")

    result: list[np.ndarray | None] = [None] * len(point_clouds)
    long_positions: list[int] = []
    long_clouds: list[np.ndarray] = []
    for pos, points in enumerate(point_clouds):
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"point cloud {pos} must have shape (N, 3), got {points.shape}")
        if points.shape[0] == 0:
            raise ValueError(f"point cloud {pos} is empty")
        if points.shape[0] <= n_points:
            result[pos] = _padded_indices(points.shape[0], n_points)
        else:
            long_positions.append(pos)
            long_clouds.append(points)

    if long_clouds:
        global _PYTORCH3D_CUDA_FPS_UNSUPPORTED
        target_device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            if target_device.type == "cuda" and _PYTORCH3D_CUDA_FPS_UNSUPPORTED:
                raise RuntimeError("PyTorch3D CUDA FPS kernel is unsupported on this GPU")
            from pytorch3d.ops import sample_farthest_points

            lengths = torch.tensor(
                [points.shape[0] for points in long_clouds],
                dtype=torch.int64,
                device=target_device,
            )
            max_len = int(lengths.max().item())
            padded = torch.zeros(
                (len(long_clouds), max_len, 3),
                dtype=torch.float32,
                device=target_device,
            )
            for row, points in enumerate(long_clouds):
                value = torch.from_numpy(np.asarray(points)).to(device=target_device, dtype=torch.float32)
                padded[row, : value.shape[0]].copy_(value)
            _, indices = sample_farthest_points(padded, lengths=lengths, K=n_points)
            indices_np = indices.cpu().numpy().astype(np.int64)
            for row, pos in enumerate(long_positions):
                result[pos] = indices_np[row]
        except Exception as e:
            if target_device.type == "cuda" and _is_unsupported_cuda_kernel(e):
                _PYTORCH3D_CUDA_FPS_UNSUPPORTED = True
            if "unsupported on this GPU" not in str(e):
                logger.warning(f"batched pytorch3d FPS failed ({type(e).__name__}: {e}); falling back to Open3D FPS")
            for pos, points in zip(long_positions, long_clouds):
                result[pos] = _open3d_fps_indices(points, n_points)

    if any(indices is None for indices in result):  # pragma: no cover - defensive invariant
        raise RuntimeError("internal error: batched FPS left an unresolved point cloud")
    return [np.asarray(indices, dtype=np.int64) for indices in result]


def prepare_scene_cloud(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    hand_mask: np.ndarray | None,
    obj_mask: np.ndarray | None,
    *,
    canonical_aligned: dict[str, Any],
    R_world_from_cam: np.ndarray,
    anchor_canonical_idx: int,
    crop_radius_m: float = 0.5,
    hand_faces: np.ndarray | None = None,
) -> PreparedSceneCloud | None:
    """Unproject, align, crop, and package one frame. Returns None if too few points survive.

    Model-free and CUDA-free: a parent prepares many frames, runs
    :func:`batch_fps_indices`, then finishes each cloud (possibly in spawn workers)
    with :func:`finish_scene_features`.
    """
    H, W = depth.shape
    if intrinsics.shape != (3, 3):
        raise ValueError(f"intrinsics must be (3, 3), got {intrinsics.shape}")
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    invalid_rgb = np.zeros((H, W), dtype=bool)
    if hand_mask is not None:
        invalid_rgb |= hand_mask.astype(bool)
    if obj_mask is not None:
        invalid_rgb |= obj_mask.astype(bool)
    valid = (depth > 0.1) & (depth < 10.0) & np.isfinite(depth) & ~invalid_rgb
    if int(valid.sum()) < MIN_POINTS_AFTER_CROP:
        return None

    u_grid, v_grid = _pixel_grids(H, W)
    z = depth[valid]
    x = (u_grid[valid] - cx) * z / fx
    y = (v_grid[valid] - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=-1).astype(np.float32)
    pix_uv = np.stack([u_grid[valid], v_grid[valid]], axis=-1).astype(np.int32)

    R = np.asarray(R_world_from_cam, dtype=np.float32)
    pts_world = pts_cam @ R.T
    midpoint_world = _compute_midpoint_aligned(canonical_aligned, anchor_canonical_idx)
    rel = pts_world - midpoint_world[None, :]
    dist_mid = np.linalg.norm(rel, axis=-1)
    in_sphere = dist_mid < float(crop_radius_m)
    if int(in_sphere.sum()) < MIN_POINTS_AFTER_CROP:
        for radius in (
            max(float(crop_radius_m) * 2.0, 1.0),
            max(float(crop_radius_m) * 4.0, 2.0),
        ):
            in_sphere = dist_mid < radius
            if int(in_sphere.sum()) >= MIN_POINTS_AFTER_CROP:
                break
    pts_world = pts_world[in_sphere]
    pix_uv = pix_uv[in_sphere]
    if pts_world.shape[0] < MIN_POINTS_AFTER_CROP:
        return None
    if pts_world.shape[0] > MAX_POINTS_BEFORE_NORMALS:
        keep = _uniform_subset_indices(pts_world.shape[0], MAX_POINTS_BEFORE_NORMALS)
        pts_world = pts_world[keep]
        pix_uv = pix_uv[keep]

    a = int(anchor_canonical_idx)
    obj_verts_world = canonical_aligned["obj_verts"][a]
    if hasattr(obj_verts_world, "numpy"):
        obj_verts_world = obj_verts_world.numpy()
    obj_verts_world = np.asarray(obj_verts_world, dtype=np.float32)
    obj_faces = canonical_aligned.get("obj_faces")
    if obj_faces is not None and hasattr(obj_faces, "numpy"):
        obj_faces = obj_faces.numpy()
    obj_faces_np = _sanitized_scene_obj_faces(canonical_aligned, obj_verts_world, obj_faces)
    hand_verts_world = _gather_anchor_hand_verts(canonical_aligned, a)
    hand_faces_np = None if hand_faces is None else np.asarray(hand_faces)

    sp = canonical_aligned.get("support_plane_height")
    support_plane_y = None if sp is None else -float(sp)
    return PreparedSceneCloud(
        pts_world=pts_world,
        pix_uv=pix_uv,
        obj_verts_world=obj_verts_world,
        obj_faces=obj_faces_np,
        hand_verts_world=hand_verts_world,
        hand_faces=hand_faces_np,
        support_plane_y=support_plane_y,
    )


def compute_oriented_normals(pts_world: np.ndarray, normal_k: int) -> np.ndarray:
    """Open3D normals (k-NN), consistently oriented over the full cloud."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pts_world).astype(np.float64))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=int(normal_k)))
    try:
        pcd.orient_normals_consistent_tangent_plane(k=int(normal_k))
    except Exception:
        pass  # an orientation failure keeps the unoriented normals
    return np.asarray(pcd.normals, dtype=np.float32).copy()


def _local_curvature_selected(pts: np.ndarray, query_idx: np.ndarray, k: int) -> np.ndarray:
    """Curvature λ_min / (λ_0 + λ_1 + λ_2) of the k-NN neighborhood in the full cloud, at selected points."""
    M = pts.shape[0]
    if M == 0 or query_idx.size == 0:
        return np.zeros(query_idx.size, dtype=np.float32)
    k_eff = min(int(k), M)
    kd = cKDTree(pts)
    _, nn_idx = kd.query(pts[query_idx], k=k_eff)
    if nn_idx.ndim == 1:
        nn_idx = nn_idx[:, None]
    neigh = pts[nn_idx]
    centered = neigh - neigh.mean(axis=1, keepdims=True)
    cov = np.einsum("mkd,mke->mde", centered, centered) / max(k_eff - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    total = eigvals.sum(axis=-1).clip(min=1e-12)
    return (eigvals[:, 0] / total).astype(np.float32)


def _log_density_selected(pts: np.ndarray, query_idx: np.ndarray) -> np.ndarray:
    """log(10-NN density) in the full cloud, at selected points."""
    k_dens = min(10, pts.shape[0])
    nn_dists, _ = cKDTree(pts).query(pts[query_idx], k=k_dens)
    if nn_dists.ndim == 1:
        nn_dists = nn_dists[:, None]
    density = float(k_dens) / (nn_dists[:, -1] ** 3 + 1e-6)
    return np.log(density + 1e-6).astype(np.float32)


def _selected_surface_distances(prepared: PreparedSceneCloud, selected_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Point-to-triangle distances from the selected points to the posed object and hand meshes."""
    import igl

    hand_verts_world = prepared.hand_verts_world
    sqd_o, _, _ = igl.point_mesh_squared_distance(
        selected_points.astype(np.float64),
        prepared.obj_verts_world.astype(np.float64),
        np.asarray(prepared.obj_faces, dtype=np.int32),
    )
    d_obj = np.sqrt(np.maximum(np.asarray(sqd_o, dtype=np.float64), 0.0)).astype(np.float32)
    if hand_verts_world.shape[0] == 0:
        return d_obj, np.full(selected_points.shape[0], 1.0, dtype=np.float32)
    n_hands = 0
    if prepared.hand_faces is not None:
        hf0 = np.asarray(prepared.hand_faces, dtype=np.int64)
        per = int(hf0.max()) + 1  # MANO verts per hand (778)
        n_hands = hand_verts_world.shape[0] // per if per > 0 and hand_verts_world.shape[0] % per == 0 else 0
    if n_hands >= 1:
        hf = np.concatenate([hf0 + i * per for i in range(n_hands)], axis=0) if n_hands > 1 else hf0
        sqd_h, _, _ = igl.point_mesh_squared_distance(
            selected_points.astype(np.float64),
            hand_verts_world.astype(np.float64),
            hf.astype(np.int32),
        )
        d_hand = np.sqrt(np.maximum(np.asarray(sqd_h, dtype=np.float64), 0.0)).astype(np.float32)
    else:
        d_hand = np.asarray(cKDTree(hand_verts_world).query(selected_points, k=1)[0], dtype=np.float32)
    return d_obj, d_hand


def _selected_semantic_features(
    prepared: PreparedSceneCloud,
    query_idx: np.ndarray,
    *,
    pca_basis: dict[str, np.ndarray] | None,
    dino_patch_features: np.ndarray | None,
    dino_patch_size: int,
    dino_patch_features_projected: bool,
) -> np.ndarray | None:
    if pca_basis is None or dino_patch_features is None:
        return None
    from gtf.datasets.scene.dinov3_semantic import pixel_to_patch_index

    patch_feats = dino_patch_features
    H_p, W_p, C = patch_feats.shape
    if dino_patch_features_projected:
        # Patches were projected on the GPU already; gather at the selected points.
        pix_uv = prepared.pix_uv[query_idx]
        patch_v, patch_u = pixel_to_patch_index(pix_uv[:, 0], pix_uv[:, 1], H_p, W_p, int(dino_patch_size))
        raw_sem = patch_feats[patch_v, patch_u]
        return raw_sem.astype(np.float32)

    # Project every row, then gather, so a point's features do not depend on how many points
    # are selected (BLAS may use another kernel and summation order for a smaller product).
    pix_uv = prepared.pix_uv
    patch_v, patch_u = pixel_to_patch_index(pix_uv[:, 0], pix_uv[:, 1], H_p, W_p, int(dino_patch_size))
    raw_sem = patch_feats[patch_v, patch_u]
    components = np.asarray(pca_basis["components"], dtype=np.float32)
    mean = np.asarray(pca_basis["mean"], dtype=np.float32)
    if components.shape[1] != C:
        raise ValueError(f"PCA basis dim mismatch: components {components.shape} vs DINOv3 feat dim {C}")
    centered = raw_sem - mean[None, :]
    semantic_full = (centered @ components.T).astype(np.float32)
    return semantic_full[query_idx]


def finish_scene_features(
    prepared: PreparedSceneCloud,
    fps_indices: np.ndarray,
    *,
    normal_k: int = 30,
    pca_basis: dict[str, np.ndarray] | None = None,
    dino_patch_features: np.ndarray | None = None,
    dino_patch_size: int = 16,
    dino_patch_features_projected: bool = False,
) -> dict[str, Any]:
    """Compute the per-point features at the FPS points (CPU only; runs in a spawn worker).

    Normals are estimated and oriented on the full cloud; distances, curvature,
    density and semantics are pointwise or use full-cloud neighborhoods.
    Geometric features (12): xyz, normal, d_obj, d_hand, h_support, theta_grav,
    curvature, log_density.
    """
    idx = np.asarray(fps_indices, dtype=np.int64)
    if idx.ndim != 1:
        raise ValueError(f"fps_indices must be 1-D, got {idx.shape}")
    if idx.size == 0 or int(idx.min()) < 0 or int(idx.max()) >= prepared.pts_world.shape[0]:
        raise ValueError(f"fps_indices out of range for {prepared.pts_world.shape[0]} points")

    normals_full = compute_oriented_normals(prepared.pts_world, normal_k)

    P_scene = prepared.pts_world[idx].astype(np.float32)
    scene_normals = normals_full[idx]
    d_obj, d_hand = _selected_surface_distances(prepared, P_scene)
    if prepared.support_plane_y is None:
        h_support = np.zeros(P_scene.shape[0], dtype=np.float32)
    else:
        h_support = (prepared.support_plane_y - P_scene[:, 1]).astype(np.float32)
    theta_grav = np.arccos(np.clip(np.abs(scene_normals[:, 1]), 0.0, 1.0)).astype(np.float32)
    curvature = _local_curvature_selected(prepared.pts_world, idx, k=int(normal_k))
    log_density = _log_density_selected(prepared.pts_world, idx)
    scene_geo_feats = np.concatenate(
        [
            P_scene,
            scene_normals,
            d_obj[:, None],
            d_hand[:, None],
            h_support[:, None],
            theta_grav[:, None],
            curvature[:, None],
            log_density[:, None],
        ],
        axis=-1,
    ).astype(np.float32)
    scene_sem_feats = _selected_semantic_features(
        prepared,
        idx,
        pca_basis=pca_basis,
        dino_patch_features=dino_patch_features,
        dino_patch_size=int(dino_patch_size),
        dino_patch_features_projected=bool(dino_patch_features_projected),
    )
    return {
        "P_scene": P_scene,
        "scene_normals": scene_normals,
        "scene_geo_feats": scene_geo_feats,
        "scene_sem_feats": scene_sem_feats,
        "support_plane_y": prepared.support_plane_y,
        "n_points_actual": int(np.unique(idx).size),
    }


def warm_scene_cpu_worker() -> None:
    """Initialize the CPU geometry libraries (no CUDA) before the first real task."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    compute_oriented_normals(points, normal_k=4)
    cKDTree(points).query(points[:1], k=4)
    import igl  # noqa: F401


def initialize_scene_cpu_worker() -> None:
    """Initialize a persistent spawn worker with one CPU thread per runtime.

    This function is suitable as ``ProcessPoolExecutor(initializer=...)``. It
    makes no CUDA calls. Setting the environment variables protects libraries
    imported lazily after initialization, while ``threadpoolctl`` constrains
    BLAS/OpenMP runtimes already loaded during module import.
    """
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"

    global _SCENE_THREADPOOL_LIMITER
    try:
        from threadpoolctl import threadpool_limits

        _SCENE_THREADPOOL_LIMITER = threadpool_limits(limits=1)
    except ImportError:
        _SCENE_THREADPOOL_LIMITER = None

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting inter-op threads only before parallel work;
        # a reused worker may already have initialized that pool.
        pass
    warm_scene_cpu_worker()


def finish_scene_features_task(
    task: SceneFeatureFinishTask,
) -> SceneFeatureFinishResult:
    """Top-level spawn-worker entry point; exceptions intentionally propagate."""
    payload = finish_scene_features(
        task.prepared,
        task.fps_indices,
        normal_k=int(task.normal_k),
        pca_basis=task.pca_basis,
        dino_patch_features=task.dino_patch_features,
        dino_patch_size=int(task.dino_patch_size),
        dino_patch_features_projected=bool(task.dino_patch_features_projected),
    )
    return SceneFeatureFinishResult(task_id=int(task.task_id), payload=payload)
