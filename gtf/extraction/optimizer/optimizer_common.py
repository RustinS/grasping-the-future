"""Shared pieces of the hand / object optimizer: errors, MANO forward, temporal basis, winding sign, mask targets, camera poses."""

import os
import pickle

import cv2
import numpy as np
import torch
import torch.nn as nn
import warp as wp
from manopth.manolayer import ManoLayer

from gtf.paths import MANO_DIR

SILHOUETTE_N_DIRS = 16
MIN_HAND_CAMERA_DEPTH_M = 0.05


class OptimizerGeometryError(RuntimeError):
    """A deterministic camera-space geometry problem that may use pose fallback."""


class OptimizerTerminalReject(RuntimeError):
    """An intrinsic source-quality failure that cannot produce a usable window."""


def _validate_corrected_hand_depth(hand_cz, scale_z_vals, correction, side):
    """Return the nearest post-correction hand depth or reject the pose chain."""

    projected_cz = hand_cz - correction * scale_z_vals
    min_projected_cz = float(projected_cz.min().item())
    if min_projected_cz < MIN_HAND_CAMERA_DEPTH_M:
        raise OptimizerGeometryError(
            f"unsafe {side} depth correction would place a valid hand at z={min_projected_cz:.4f}m < {MIN_HAND_CAMERA_DEPTH_M:.4f}m (camera correction={correction:.4f}m)"
        )
    return min_projected_cz


_sil_angles = np.linspace(0.0, 2.0 * np.pi, SILHOUETTE_N_DIRS, endpoint=False, dtype=np.float32)
SILHOUETTE_DIRS = np.stack([np.cos(_sil_angles), np.sin(_sil_angles)], axis=1).astype(np.float32)


@wp.kernel(enable_backward=False)
def _warp_query_closest_winding(
    mesh: wp.uint64,
    points: wp.array[wp.vec3],
    faces: wp.array[int],
    bary_u: wp.array[float],
    bary_v: wp.array[float],
    signs: wp.array[float],
):
    index = wp.tid()
    query = wp.mesh_query_point_sign_winding_number(mesh, points[index], 1.0e6)
    if query.result:
        faces[index] = query.face
        bary_u[index] = query.u
        bary_v[index] = query.v
        signs[index] = query.sign
    else:
        faces[index] = -1
        signs[index] = 1.0


def _is_warp_kernel_build_error(error):
    """Identify Warp/NVRTC build failures that are safe to fall back from."""

    detail = str(error).lower()
    return any(
        fragment in detail
        for fragment in (
            "nvrtc",
            "cuda kernel build failed",
            "kernel compilation failed",
            "temporary files is too long",
        )
    )


def _sanitize_object_faces(vertices, faces):
    """Drop triangles that make Warp/Kaolin distance queries undefined."""

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 3:
        raise ValueError(f"object vertices must have shape (V, 3), got {vertices.shape}")
    if not np.isfinite(vertices).all():
        raise OptimizerGeometryError("object vertices contain non-finite coordinates")
    if faces.ndim != 2 or faces.shape[1:] != (3,) or len(faces) == 0:
        raise ValueError(f"object faces must have shape (F, 3), got {faces.shape}")
    face_min = int(faces.min())
    face_max = int(faces.max())
    if face_min < 0 or face_max >= len(vertices):
        raise ValueError(f"object face indices are outside the object vertex roster: range=[{face_min}, {face_max}], vertices={len(vertices)}")

    distinct = (faces[:, 0] != faces[:, 1]) & (faces[:, 0] != faces[:, 2]) & (faces[:, 1] != faces[:, 2])
    triangles = vertices[faces]
    doubled_area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    object_diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    area_floor = max(object_diagonal * object_diagonal * 1e-12, 1e-24)
    keep = distinct & np.isfinite(doubled_area) & (doubled_area > area_floor)
    cleaned = np.ascontiguousarray(faces[keep])
    if len(cleaned) == 0:
        raise ValueError("object mesh has no non-degenerate triangles")
    return cleaned, int((~keep).sum())


@torch.no_grad()
def winding_sign(points, verts, faces, frame_chunk=4):
    """Inside/outside sign for penetration losses: +1 outside, −1 inside, by exact
    generalized winding number (van Oosterom–Strackee signed solid angles, summed over
    all faces). Robust on non-watertight / self-intersecting SAM 3D meshes and inside
    cavities (bowl or cup interiors), where a nearest-face-normal sign flips.

    points (T,P,3), verts (T,V,3), faces (F,3) long. Returns (T,P) of ±1. The sign is a
    constant factor on the differentiable distance, so autograd is unaffected.
    """
    T, P = points.shape[:2]
    out = torch.ones(T, P, device=points.device, dtype=points.dtype)
    for lo in range(0, T, frame_chunk):
        hi = min(T, lo + frame_chunk)
        tri = verts[lo:hi][:, faces]  # (t, F, 3, 3)
        a = tri[:, None, :, 0] - points[lo:hi, :, None]  # (t, P, F, 3)
        b = tri[:, None, :, 1] - points[lo:hi, :, None]
        c = tri[:, None, :, 2] - points[lo:hi, :, None]
        la, lb, lc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
        det = (a * torch.cross(b, c, dim=-1)).sum(-1)
        denom = la * lb * lc + (a * b).sum(-1) * lc + (b * c).sum(-1) * la + (c * a).sum(-1) * lb
        w = torch.atan2(det, denom).sum(-1) / (2 * torch.pi)  # (t, P) winding number
        out[lo:hi] = torch.where(w > 0.5, -torch.ones_like(w), torch.ones_like(w))
    return out


class TemporalBasis(nn.Module):
    """Low-frequency DCT basis for structurally smooth trajectories."""

    def __init__(self, T, dim, n_basis=None, device=None):
        super().__init__()
        if n_basis is None:
            n_basis = max(2, T // 3)
        n_basis = min(n_basis, T)
        self.coeffs = nn.Parameter(torch.zeros(n_basis, dim, device=device))
        t = torch.arange(T, dtype=torch.float32)
        basis = torch.stack([torch.cos(torch.pi * k * (2 * t + 1) / (2 * T)) for k in range(n_basis)])
        self.register_buffer("basis", basis.to(device) if device else basis)

    def forward(self):
        return self.basis.T @ self.coeffs  # (T, dim)


class MANOForward(nn.Module):
    """Differentiable MANO FK in the packed hand-stage convention (the WiLoR output in hands/).

    Both hands are posed with the right-hand MANO model (from gtf.paths.MANO_DIR) and the
    left-hand result is mirrored in camera space (x -> -x; its faces are re-wound to keep
    outward normals). Reconstructing the same way here makes the optimizer start from the
    exact saved geometry.
    """

    def __init__(self, side="right", device="cuda"):
        super().__init__()
        self.side = side
        self.mano = ManoLayer(
            mano_root=MANO_DIR,
            side="right",
            use_pca=False,
            ncomps=45,
            flat_hand_mean=True,
        ).to(device)
        hand_faces = self.mano.th_faces.long()
        if side == "left":
            hand_faces = hand_faces[:, [0, 2, 1]]
        self.register_buffer("hand_faces", hand_faces)

    def forward(self, global_orient, hand_pose, betas, cam_t):
        """MANO FK: params → (B, 778, 3) verts + (B, 21, 3) joints in camera-space meters."""
        full_pose = torch.cat([global_orient, hand_pose], dim=-1)
        verts, joints = self.mano(full_pose, th_betas=betas)
        verts = verts / 1000.0 + cam_t.unsqueeze(1)
        joints = joints / 1000.0 + cam_t.unsqueeze(1)
        if self.side == "left":
            verts[..., 0] = -verts[..., 0]
            joints[..., 0] = -joints[..., 0]
        return verts, joints


def load_hand_mano(segment_dir, side):
    """Load the hand stage's MANO parameters for one side (the WiLoR output, packed in
    hands/{side}.pt|.npz; valid frames from hands/det.pkl). Returns dict or None."""
    packed_dir = os.path.join(segment_dir, "hands")
    pt_path = os.path.join(packed_dir, f"{side}.pt")
    npz_path = os.path.join(packed_dir, f"{side}.npz")
    if not os.path.exists(pt_path) or not os.path.exists(npz_path):
        return None

    pt = torch.load(pt_path, map_location="cpu", weights_only=False)
    frame_names = np.load(npz_path, allow_pickle=False)["frame_names"]
    frame_to_idx = {}
    for i, fn in enumerate(frame_names):
        frame_to_idx.setdefault(int(fn), i)

    det_valid = set(frame_to_idx.keys())
    det_path = os.path.join(segment_dir, "hands", "det.pkl")
    if os.path.exists(det_path):
        with open(det_path, "rb") as f:
            det_data = pickle.load(f)
        for entry in det_data:
            if entry.get("seq", "").endswith(f"_{side}"):
                valid_arr, frame_idx_arr = entry.get("valid"), entry.get("frame_idx")
                if valid_arr is not None and frame_idx_arr is not None:
                    det_valid = {int(fi) for fi, v in zip(frame_idx_arr, valid_arr) if v}

    return {k: pt[k].numpy() for k in ("hand_pose", "global_orient", "betas", "cam_t", "intr", "img_size")} | {
        "verts": pt["cHands"].numpy(),
        "joints": pt["cJoints"].numpy(),
        "frame_to_idx": frame_to_idx,
        "valid_frames": det_valid,
    }


def _build_mask_target(mask, sdf_size=(180, 320)):
    """Prepare lightweight silhouette targets from a binary object mask."""
    target = {
        "valid": False,
        "centroid": np.zeros(2, dtype=np.float32),
        "bbox": np.zeros(4, dtype=np.float32),
        "sdf": np.zeros(sdf_size, dtype=np.float32),
        "support": np.zeros(SILHOUETTE_N_DIRS, dtype=np.float32),
    }
    if mask is None:
        return target

    # Upstream masks legitimately arrive as either {0, 1} or {0, 255}.  The
    # distance-transform complement below requires a true binary array;
    # uint8(1 - 255) wraps to 2 and contains no zero background.
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    if mask_u8.ndim != 2 or mask_u8.sum() == 0:
        return target

    H, W = mask_u8.shape
    ys, xs = np.nonzero(mask_u8)
    norm = np.array([max(W - 1, 1), max(H - 1, 1)], dtype=np.float32)
    target["valid"] = True
    target["centroid"] = np.array([xs.mean(), ys.mean()], dtype=np.float32) / norm
    target["bbox"] = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32) / np.tile(norm, 2)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        contour_xy = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    else:
        contour_xy = np.stack([xs, ys], axis=1).astype(np.float32)
    target["support"] = (contour_xy / norm @ SILHOUETTE_DIRS.T).max(axis=0).astype(np.float32)

    mask_small = cv2.resize(mask_u8, (sdf_size[1], sdf_size[0]), interpolation=cv2.INTER_NEAREST)
    # A very small source mask can disappear under nearest-neighbor SDF
    # downsampling. Its complement is then all ones, and OpenCV returns a
    # ~FLT_MAX distance sentinel rather than a usable field. Do not treat that
    # frame as silhouette evidence; the original-resolution quality gates still
    # adjudicate the undersized mask.
    if not mask_small.any():
        target["valid"] = False
        target["sdf"].fill(0)
        return target

    # A one-pixel zero border is the correct image-domain boundary condition
    # for a full-frame mask and keeps its inside transform finite as well.
    padded = np.pad(mask_small, 1, mode="constant", constant_values=0)
    dist_out = cv2.distanceTransform(1 - padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
    dist_in = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
    sdf = (dist_out - dist_in) * (max(H / sdf_size[0], W / sdf_size[1]) / max(H, W))
    if not np.isfinite(sdf).all():
        raise ValueError("object silhouette SDF contains non-finite values")
    target["sdf"] = np.asarray(sdf, dtype=np.float32)
    return target


def _robust_segment_beta(betas):
    """Collapse noisy per-frame MANO shape into one robust per-segment identity."""
    if betas is None or len(betas) == 0:
        return np.zeros(10, dtype=np.float32)

    betas = np.asarray(betas, dtype=np.float32)
    center = np.median(betas, axis=0)
    mad = np.median(np.abs(betas - center), axis=0)
    tol = np.maximum(3.0 * mad, 0.05)
    inlier = np.all(np.abs(betas - center) <= tol, axis=1)
    if inlier.any():
        return np.median(betas[inlier], axis=0).astype(np.float32)
    return center.astype(np.float32)


def _uniform_subset(indices, max_keep):
    """Keep a roughly uniform subset of integer indices."""
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) <= max_keep:
        return indices
    sel = np.linspace(0, len(indices) - 1, max_keep, dtype=np.int64)
    return indices[sel]


def _load_ba_camera_poses(segment_dir, max_median_reproj_px=12.0):
    """Camera-frame object rotations + centroid translations from the BA texture tracks.

    run_object_pose's BA produces a temporally coherent object track (CoTracker
    points + depth, jointly optimized), much smoother than the independent
    per-keyframe SAM 3D pose estimates. BA poses are object-local -> world; they are
    composed with world-to-camera so the result lives in the MoGe camera frame used here.

    Returns (R_cam (T_video,3,3), t_cam (T_video,3)) or None if unavailable or
    the BA solution is unreliable (high reprojection error).
    """
    object_pose_dir = os.path.join(segment_dir, "object_pose")
    if not os.path.exists(os.path.join(object_pose_dir, ".object_pose_done")):
        return None
    obj_poses_path = os.path.join(object_pose_dir, "object_poses.npz")
    if not os.path.exists(obj_poses_path):
        return None
    d = np.load(obj_poses_path)
    R_world = d["rotations"].astype(np.float64)  # (T_video, 3, 3), local->world
    t_world = d["translations"].astype(np.float64)  # (T_video, 3) — BA point-cloud centroid in world
    reproj = d.get("reprojection_errors")
    if reproj is not None:
        # NaN entries are zero-evidence frames; a chain with NO evidence
        # frames at all is pure smoothness interpolation — reject it like a bad median.
        if not np.isfinite(reproj).any():
            print("  BA poses rejected: no evidence frames (all-NaN reprojection errors)")
            return None
        if float(np.nanmedian(reproj)) > max_median_reproj_px:
            print(f"  BA poses rejected: median reproj {float(np.nanmedian(reproj)):.1f}px > {max_median_reproj_px}px")
            return None

    cam_path = os.path.join(object_pose_dir, "camera_poses.npz")
    if os.path.exists(cam_path):
        c2w = np.load(cam_path)["cam_to_world"].astype(np.float64)  # (T_video, 4, 4)
        T = min(len(R_world), len(c2w))
        w2c_R = np.transpose(c2w[:T, :3, :3], (0, 2, 1))
        w2c_t = -np.einsum("tij,tj->ti", w2c_R, c2w[:T, :3, 3])
        R_cam = np.einsum("tij,tjk->tik", w2c_R, R_world[:T])
        t_cam = np.einsum("tij,tj->ti", w2c_R, t_world[:T]) + w2c_t
        return R_cam, t_cam
    return R_world, t_world
