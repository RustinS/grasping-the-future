"""Geometry helpers of the quality-gate metrics: rotations, silhouettes, hand scale, object-RGB consistency, window scoping."""

import json
import os
import numpy as np
from scipy.spatial.transform import Rotation
from gtf.paths import MANO_DIR
from gtf.utils import load_mano_faces, load_moge_intrinsics, opencv_to_opengl, render_scene_pyrender


# ── Rotation helpers ────────────────────────────────────────────────────────


def _procrustes_rotations(obj_verts):
    """Per-frame rotation matrices via Procrustes alignment to frame 0."""
    ref = obj_verts[0] - obj_verts[0].mean(axis=0)
    rotations = []
    for t in range(len(obj_verts)):
        centered = obj_verts[t] - obj_verts[t].mean(axis=0)
        H = ref.T @ centered
        U, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        R = Vt.T @ np.diag([1, 1, d]) @ U.T
        rotations.append(R)
    return rotations


def _continuous_quaternions(rotation_matrices):
    """Rotation matrices → sign-corrected continuous quaternions [x,y,z,w]."""
    quats = []
    for R in rotation_matrices:
        q = Rotation.from_matrix(R).as_quat()
        if quats and np.dot(quats[-1], q) < 0:
            q = -q
        quats.append(q)
    return np.array(quats)


def _angular_velocity(quats):
    """Angular displacement in degrees between consecutive quaternions."""
    avs = []
    for i in range(len(quats) - 1):
        r_diff = Rotation.from_quat(quats[i]).inv() * Rotation.from_quat(quats[i + 1])
        avs.append(np.degrees(r_diff.magnitude()))
    return np.array(avs)


# ── Hand metric scale ───────────────────────────────────────────────────────

_MANO_EDGE_CACHE = {}


def _hand_scale_metrics(d):
    """Pose-invariant hand metric scale vs canonical MANO (mean mesh-edge-length ratio).

    Returns (scale_med, scale_drift) for the worst side (largest |log scale|), or
    (None, None) when no side has enough valid frames. scale_drift = within-segment
    p90/p10 ratio (catches a hand whose size changes from frame to frame)."""
    import torch
    from manopth.manolayer import ManoLayer

    if "layer" not in _MANO_EDGE_CACHE:
        _MANO_EDGE_CACHE["layer"] = ManoLayer(
            mano_root=MANO_DIR,
            side="right",
            use_pca=False,
            ncomps=45,
            flat_hand_mean=True,
        )
        faces_r, _ = load_mano_faces()
        e = np.sort(
            np.concatenate([faces_r[:, [0, 1]], faces_r[:, [1, 2]], faces_r[:, [2, 0]]]),
            axis=1,
        )
        _MANO_EDGE_CACHE["edges"] = np.unique(e, axis=0)
    layer, edges = _MANO_EDGE_CACHE["layer"], _MANO_EDGE_CACHE["edges"]

    def mean_edge(v):
        return np.linalg.norm(v[..., edges[:, 0], :] - v[..., edges[:, 1], :], axis=-1).mean(-1)

    worst = None
    for side in ("left", "right"):
        vk, mk, bk = f"hand_{side}_verts", f"hand_{side}_valid", f"hand_{side}_betas"
        if vk not in d or mk not in d:
            continue
        valid = d[mk].astype(bool)
        if valid.sum() < 3:
            continue
        betas = torch.tensor(np.asarray(d[bk])[None], dtype=torch.float32) if bk in d else torch.zeros(1, 10)
        with torch.no_grad():
            v0, _ = layer(torch.zeros(1, 48), th_betas=betas)
        sc = mean_edge(d[vk][valid]) / max(float(mean_edge(v0[0].numpy() / 1000.0)), 1e-9)
        med = float(np.median(sc))
        drift = float(np.percentile(sc, 90) / max(np.percentile(sc, 10), 1e-9))
        if worst is None or abs(np.log(max(med, 1e-9))) > abs(np.log(max(worst[0], 1e-9))):
            worst = (med, drift)
    return worst if worst else (None, None)


def _interaction_metrics(d):
    """(min hand-object vert distance over all frames/sides [m], object centroid path
    diagonal [m]): the no-interaction detector (e.g. both hands far from an idle object
    for the whole clip)."""
    ov = d["obj_verts"][:, ::10]
    min_dist = None
    for side in ("left", "right"):
        vk, mk = f"hand_{side}_verts", f"hand_{side}_valid"
        if vk not in d or mk not in d:
            continue
        rows = np.where(d[mk].astype(bool))[0]
        if len(rows) == 0:
            continue
        hv = d[vk][:, ::6]
        for k in rows:
            dist = float(np.linalg.norm(hv[k][:, None] - ov[k][None], axis=-1).min())
            min_dist = dist if min_dist is None else min(min_dist, dist)
    centroids = d["obj_verts"].mean(axis=1)
    obj_motion = float(np.linalg.norm(centroids.max(axis=0) - centroids.min(axis=0)))
    return min_dist, obj_motion


# ── Silhouette rendering ───────────────────────────────────────────────────


def _render_silhouette(verts_cv, faces, K, H, W):
    """Render binary silhouette from vertices in OpenCV camera coords."""
    silhouette, _ = _render_silhouette_depth(verts_cv, faces, K, H, W)
    return silhouette


def _render_silhouette_depth(verts_cv, faces, K, H, W):
    """Render a silhouette and its z-buffer from OpenCV camera coordinates."""
    verts_gl = opencv_to_opengl(verts_cv)
    color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    rgba, depth = render_scene_pyrender(
        [(verts_gl, faces, color)],
        K[0, 0],
        K[1, 1],
        K[0, 2],
        K[1, 2],
        W,
        H,
    )
    return rgba[:, :, 3] > 0.5, np.asarray(depth)


def _depth_aware_area_ratios(
    seg_dir,
    d,
    obj_verts,
    obj_faces,
    frame_indices,
    sam3_masks,
    mano_faces_map,
    K_default,
    H_img,
    W_img,
):
    """Area-ratio samples whose hand occlusion is resolved by the z-buffer.

    A fallback only: callers use the projected-overlap value whenever any frame
    qualifies under it, and this z-buffer version only when none does (e.g. a hand
    behind the object overlaps its projection on every frame), so a measurable area
    ratio is not reported as ``None``.
    """

    ratios = []
    for t_idx, fi_value in enumerate(frame_indices):
        fi = int(fi_value)
        if fi < 0 or fi >= len(sam3_masks):
            continue
        K_frame = load_moge_intrinsics(seg_dir, fi)
        if K_frame is None:
            K_frame = K_default
        obj_sil, obj_depth = _render_silhouette_depth(obj_verts[t_idx], obj_faces, K_frame, H_img, W_img)
        obj_total = int(obj_sil.sum())
        if obj_total <= 100:
            continue

        hand_meshes = []
        for side in ("left", "right"):
            hv_key, valid_key = f"hand_{side}_verts", f"hand_{side}_valid"
            if hv_key not in d or valid_key not in d or not d[valid_key][t_idx]:
                continue
            hand_meshes.append(
                (
                    opencv_to_opengl(d[hv_key][t_idx]),
                    mano_faces_map[side],
                    np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                )
            )
        if hand_meshes:
            rgba, hand_depth = render_scene_pyrender(
                hand_meshes,
                K_frame[0, 0],
                K_frame[1, 1],
                K_frame[0, 2],
                K_frame[1, 2],
                W_img,
                H_img,
            )
            hand_sil = rgba[:, :, 3] > 0.5
            # A hand only hides the object where it is physically in front.
            # The tiny metric epsilon prevents equal-depth/z-fighting pixels
            # at true contact from being classified as reliable occlusion.
            hand_in_front = obj_sil & hand_sil & (obj_depth > 0) & (hand_depth > 0) & (hand_depth < obj_depth - 1e-4)
            occ_ratio = float(hand_in_front.sum() / obj_total)
        else:
            occ_ratio = 0.0
        if occ_ratio < 0.2:
            ratios.append(float(sam3_masks[fi].astype(bool).sum() / obj_total))
    return ratios


def _resolve_area_ratio(area_ratios, depth_aware_fallback):
    """Mean of the projected-overlap ratios; when there are none, of the lazily computed depth-aware ratios; else None."""

    ratios = area_ratios
    if not ratios:
        ratios = depth_aware_fallback()
    return float(np.mean(ratios)) if ratios else None


def _camera_rotation_speed(seg_dir, video_path):
    """Median camera rotation speed (deg/s) from the BA camera chain
    (object_pose/camera_poses.npz, per-video-frame c2w), as refine_trajectory.camera_rotation_speed
    computes it. Observability only, not a gate."""
    cp = os.path.join(seg_dir, "object_pose", "camera_poses.npz")
    if not os.path.exists(cp):
        return None
    c2w = np.load(cp, allow_pickle=True)["cam_to_world"]
    if len(c2w) < 2:
        return None
    fps = 30.0
    if os.path.exists(video_path):
        import cv2

        cap = cv2.VideoCapture(video_path)
        v_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if v_fps and v_fps > 0:
            fps = float(v_fps)
    Rrel = np.einsum("tij,tik->tjk", c2w[:-1, :3, :3], c2w[1:, :3, :3])
    tr = np.clip((np.trace(Rrel, axis1=1, axis2=2) - 1) / 2, -1, 1)
    return float(np.median(np.degrees(np.arccos(tr)) * fps))


def _project_visible(verts, faces, K, H, W, depth_tol=0.015, occluders=()):
    """Project (V,3) OpenCV-camera verts; return (u, v, visible) where visible
    requires in-bounds, z>0, and agreement with the depth buffer rendered from the
    mesh plus any occluders (self- and hand-occlusion test)."""
    white = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    meshes = [(opencv_to_opengl(verts), faces, white)] + [(opencv_to_opengl(ov), of, white) for ov, of in occluders]
    _, depth = render_scene_pyrender(meshes, K[0, 0], K[1, 1], K[0, 2], K[1, 2], W, H)
    z = verts[:, 2]
    u = np.round(K[0, 0] * verts[:, 0] / np.maximum(z, 1e-6) + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * verts[:, 1] / np.maximum(z, 1e-6) + K[1, 2]).astype(int)
    ok = (z > 0.01) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    vis = ok.copy()
    vis[ok] &= np.abs(depth[v[ok], u[ok]] - z[ok]) < depth_tol
    return u, v, vis


def _obj_rgb_consistency(
    seg_dir,
    obj_verts,
    obj_faces,
    frame_indices,
    anchor_idx,
    K_default,
    H_img,
    W_img,
    n_eval=8,
    hands=(),
):
    """Self-baked appearance consistency of the object reconstruction.

    Bake: each vertex visible at the anchor keyframe samples its video color there.
    Evaluate: at keyframes far from the anchor, Pearson-correlate baked colors of
    visible baked verts against the video colors at their projected pixels. A correct
    shape+pose keeps the correspondence; a wrong reconstruction scrambles it.
    `hands` = [(verts (T,V,3), faces, valid (T,))...] rendered as depth occluders so
    hand-covered vertices don't sample hand pixels (wrap grasps read false-low
    otherwise). Returns median correlation over eval keyframes, or None when
    uninformative (no video, anchor barely visible, or texture-less baked colors)."""
    import cv2

    video_path = os.path.join(seg_dir, "video.mp4")
    if not os.path.exists(video_path):
        return None
    fi_arr = np.asarray(frame_indices)
    T = len(fi_arr)
    anchor_pos = int(np.argmin(np.abs(fi_arr - anchor_idx)))
    eval_pos = [t for t in np.linspace(0, T - 1, min(T, n_eval + 2)).astype(int).tolist() if abs(t - anchor_pos) > max(2, T // 10)][:n_eval]
    if not eval_pos:
        return None

    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi_arr[anchor_pos]))
        ok_read, frame_bgr = cap.read()
        if not ok_read:
            return None
        K_a = _frame_K(seg_dir, fi_arr[anchor_pos], K_default)

        def occluders_at(t):
            return [(hv[t], hf) for hv, hf, hvalid in hands if hvalid[t]]

        u, v, vis = _project_visible(
            obj_verts[anchor_pos],
            obj_faces,
            K_a,
            H_img,
            W_img,
            occluders=occluders_at(anchor_pos),
        )
        if vis.sum() < 100:
            return None
        baked = np.full((len(obj_verts[anchor_pos]), 3), np.nan, dtype=np.float32)
        baked[vis] = frame_bgr[v[vis], u[vis]][:, ::-1].astype(np.float32)
        if np.nanstd(baked) < 8.0:
            return None  # texture-less object: correlation is undefined noise

        corrs = []
        for t in eval_pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi_arr[t]))
            ok_read, frame_bgr = cap.read()
            if not ok_read:
                continue
            K_t = _frame_K(seg_dir, fi_arr[t], K_default)
            u, v, vis = _project_visible(obj_verts[t], obj_faces, K_t, H_img, W_img, occluders=occluders_at(t))
            vis &= np.isfinite(baked[:, 0])
            if vis.sum() < 100:
                continue
            observed = frame_bgr[v[vis], u[vis]][:, ::-1].astype(np.float32)
            b = baked[vis]
            if b.std() > 1e-6 and observed.std() > 1e-6:
                corrs.append(float(np.corrcoef(b.ravel(), observed.ravel())[0, 1]))
    finally:
        cap.release()
    return float(np.median(corrs)) if corrs else None


def _frame_K(seg_dir, fi, K_default):
    try:
        K = load_moge_intrinsics(seg_dir, int(fi))
    except Exception:
        K = None
    return K if K is not None else K_default


# ── Window scoping ─────────────────────────────────────────────────────────

# per-keyframe arrays of optimized.npz: the exact list export_canonical's crop slices
_WINDOW_PER_FRAME_KEYS = ["frame_indices", "obj_verts", "obj_scale_init"] + [f"hand_{s}_{f}" for s in ("left", "right") for f in ("verts", "valid", "pose")]


def _exported_window(seg_dir):
    """Video-frame bounds (lo, hi) of the segment's exported interaction window, or None.

    Every gate must measure the window the canonical trains on (the export may crop; a
    multi-window {seg}__iwN child is one window), not the full extraction chain, whose
    out-of-window frames were deliberately excluded. Source of truth: the objref sidecar's
    frame_indices (written by export from the post-crop chain); fallback: the canonical
    metadata's interaction_crop record (sidecar missing)."""
    for sub in ("optimized_stage2", "optimized_mano"):
        p = os.path.join(seg_dir, sub, "objref_chain.npz")
        if os.path.exists(p):
            try:
                fi = np.asarray(np.load(p)["frame_indices"], np.int64)
                if len(fi):
                    return int(fi.min()), int(fi.max())
            except Exception:  # noqa: BLE001
                pass
    mp = os.path.join(seg_dir, "canonical", "metadata.json")
    if os.path.exists(mp):
        try:
            ic = json.load(open(mp)).get("interaction_crop")
            if ic and ic.get("applied"):
                return int(ic["first_frame"]), int(ic["last_frame"])
        except (json.JSONDecodeError, OSError, KeyError, TypeError):
            pass
    return None


def _scope_sam3d_frames(frames, window, window_scoped):
    """The SAM 3D keyframe rows inside the exported window (all rows when the gate is not window-scoped)."""

    if window_scoped:
        return [frame for frame in frames if window[0] <= frame.get("frame_index", -1) <= window[1]]
    return frames


def _subset_chain_to_window(d, lo, hi):
    """Restrict a loaded optimized-chain dict to keyframes with lo <= frame_index <= hi.
    Returns (d, applied): applied=False when the window covers the whole chain (no-op)
    or would leave <3 keyframes (never let scoping erase a segment)."""
    fi = np.asarray(d["frame_indices"], np.int64)
    sel = (fi >= lo) & (fi <= hi)
    if sel.all() or int(sel.sum()) < 3:
        return d, False
    T0 = len(fi)
    for k in _WINDOW_PER_FRAME_KEYS:
        if k in d and getattr(d[k], "ndim", 0) >= 1 and len(d[k]) == T0:
            d[k] = d[k][sel]
    return d, True
