"""Camera poses of a segment: MegaSaM with the object masked out, and the PnP fallback."""

import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from gtf.utils import rprint as print
from gtf.extraction.object_pose.pose_cache import ROOT, _sha256_file
from gtf.extraction.object_pose.point_tracking import _cotracker_checkpoint_sha256, get_cotracker_predictor


MEGASAM_DEFAULT_BUFFER_FRAMES = 1024
MEGASAM_TRAJECTORY_FILL_BATCH_FRAMES = 16
_MEGASAM_CHECKPOINT_DIGEST = None


def _camera_model_checkpoint_sha256(camera_method):
    """Content identity for the model used by the selected camera method."""

    if not str(camera_method).startswith("megasam"):
        return _cotracker_checkpoint_sha256()
    global _MEGASAM_CHECKPOINT_DIGEST  # noqa: PLW0603

    if _MEGASAM_CHECKPOINT_DIGEST is not None:
        return _MEGASAM_CHECKPOINT_DIGEST
    path = Path(ROOT) / "third-party/mega-sam/checkpoints/megasam_final.pth"
    if not path.is_file() or path.is_symlink():
        return "megasam_checkpoint_not_resolved"
    _MEGASAM_CHECKPOINT_DIGEST = _sha256_file(path)
    return _MEGASAM_CHECKPOINT_DIGEST


def _megasam_buffer_frames(n_frames):
    """Size DROID state for the full trajectory plus its temporary fill batch."""

    if n_frames < 1:
        raise ValueError(f"MegaSaM requires at least one frame, got {n_frames}")
    # PoseTrajectoryFiller temporarily appends one 16-frame batch after the
    # retained keyframes before resetting the counter, then writes the complete
    # trajectory into the same DepthVideo. The buffer must therefore hold every
    # frame plus one fill batch.
    return max(
        MEGASAM_DEFAULT_BUFFER_FRAMES,
        n_frames + MEGASAM_TRAJECTORY_FILL_BATCH_FRAMES,
    )


def _try_import_megasam():
    """Check if MegaSaM CUDA extensions are available."""
    try:
        import droid_backends  # noqa: F401
        import lietorch  # noqa: F401

        return True
    except ImportError:
        return False


def _build_megasam_background_masks(masks, depth_results, out_h, out_w):
    """Build per-frame valid background masks for MegaSaM."""
    dilate_px = max(3, int(round(0.01 * max(out_h, out_w))))
    if dilate_px % 2 == 0:
        dilate_px += 1
    kernel = np.ones((dilate_px, dilate_px), dtype=np.uint8)

    bg_masks = []
    relaxed_frames = 0
    min_pixels = 0.05 * float(out_h * out_w)

    for mask_np, depth_data in zip(masks, depth_results):
        object_mask = cv2.resize(
            (mask_np > 0).astype(np.uint8),
            (out_w, out_h),
            interpolation=cv2.INTER_NEAREST,
        )
        dilated_object = cv2.dilate(object_mask, kernel, iterations=1) > 0

        depth_valid = np.isfinite(depth_data["depth"]) & (depth_data["depth"] > 0.01)
        if "mask" in depth_data:
            depth_valid &= depth_data["mask"] > 0
        depth_valid = (
            cv2.resize(
                depth_valid.astype(np.uint8),
                (out_w, out_h),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )

        bg_mask = depth_valid & (~dilated_object)
        if bg_mask.sum() < min_pixels:
            relaxed_frames += 1
            bg_mask = depth_valid & (object_mask == 0)
        if bg_mask.sum() == 0:
            bg_mask = depth_valid
        if bg_mask.sum() == 0:
            bg_mask = np.ones((out_h, out_w), dtype=bool)

        bg_masks.append(torch.from_numpy(bg_mask.astype(np.float32)))

    if relaxed_frames > 0:
        print(f"  MegaSaM: relaxed background mask on {relaxed_frames}/{len(bg_masks)} frames")
    return bg_masks


def run_camera_poses_megasam(frames, masks, depth_results, fps):
    """Estimate camera poses using MegaSaM (DROID-SLAM).

    Returns:
        cam_to_world: (T, 4, 4) float32
    """
    sys.path.insert(0, os.path.join(ROOT, "third-party", "mega-sam", "base", "droid_slam"))
    from droid import Droid
    from lietorch import SE3

    T, H, W, _ = frames.shape

    target_area = 384 * 512
    scale = (target_area / (H * W)) ** 0.5
    h1 = max((int(H * scale) // 8) * 8, 8)
    w1 = max((int(W * scale) // 8) * 8, 8)

    bg_masks = _build_megasam_background_masks(masks, depth_results, h1, w1)

    K_px = depth_results[0]["intrinsics_px"]
    intrinsics = torch.tensor([K_px[0, 0], K_px[1, 1], K_px[0, 2], K_px[1, 2]], dtype=torch.float32)
    intrinsics[0::2] *= w1 / W
    intrinsics[1::2] *= h1 / H

    class DroidArgs:
        weights = os.path.join(ROOT, "third-party", "mega-sam", "checkpoints", "megasam_final.pth")
        buffer = _megasam_buffer_frames(T)
        image_size = [h1, w1]
        stereo = False
        depth = True
        upsample = False
        beta = 0.3
        filter_thresh = 2.0
        warmup = 8
        keyframe_thresh = 2.0
        frontend_thresh = 12.0
        frontend_window = 25
        frontend_radius = 2
        frontend_nms = 1
        backend_thresh = 16.0
        backend_radius = 2
        backend_nms = 3
        disable_vis = True

    droid = Droid(DroidArgs())

    def _prepare_frame(t):
        frame_bgr = cv2.cvtColor(cv2.resize(frames[t], (w1, h1)), cv2.COLOR_RGB2BGR)
        image = torch.from_numpy(frame_bgr).permute(2, 0, 1).unsqueeze(0)
        depth_resized = cv2.resize(depth_results[t]["depth"], (w1, h1), interpolation=cv2.INTER_NEAREST)
        return image, torch.from_numpy(depth_resized).float()

    for t in range(T):
        image, depth_tensor = _prepare_frame(t)
        if t < T - 1:
            droid.track(t, image, depth_tensor, intrinsics=intrinsics, mask=bg_masks[t])
        else:
            droid.track_final(t, image, depth_tensor, intrinsics=intrinsics, mask=bg_masks[t])

    def image_stream():
        for t in range(T):
            image, depth_tensor = _prepare_frame(t)
            yield t, image, depth_tensor, intrinsics, bg_masks[t]

    traj_est, _ = droid.terminate(image_stream(), _opt_intr=False, full_ba=False)

    poses_th = torch.as_tensor(traj_est).float()
    cam_to_world = SE3(poses_th).inv().matrix().cpu().numpy()

    del droid
    torch.cuda.empty_cache()

    print(f"  MegaSaM: {T} camera poses estimated")
    return cam_to_world.astype(np.float32)


def run_camera_poses_pnp_fallback(frames, masks, depth_results):
    """Estimate camera poses using PnP on background points.

    Tracks background points via CoTracker3, lifts to 3D from frame 0,
    then solves PnP per frame.

    Returns:
        cam_to_world: (T, 4, 4) float32
    """
    T, H, W, _ = frames.shape

    bg_mask = masks[0] == 0
    ys, xs = np.where(bg_mask)
    if len(ys) < 50:
        print("  WARNING: Very few background pixels, using identity camera poses")
        return np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))

    n_bg = min(400, len(ys))
    indices = np.random.choice(len(ys), n_bg, replace=False)
    bg_queries = np.array([[0, xs[i], ys[i]] for i in indices], dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    queries = torch.tensor(bg_queries, dtype=torch.float32, device=device).unsqueeze(0)

    video = torch.tensor(frames, dtype=torch.float32, device=device).permute(0, 3, 1, 2).unsqueeze(0)

    predictor = get_cotracker_predictor(device)
    with torch.inference_mode():
        bg_tracks, bg_vis = predictor(video, queries=queries, backward_tracking=True)

    del video
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if bg_tracks is None:
        print("  WARNING: No background tracks, using identity camera poses")
        return np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))

    bg_tracks = bg_tracks[0].cpu().numpy()
    bg_vis = bg_vis[0].cpu().numpy()

    K0 = depth_results[0]["intrinsics_px"]
    depth0 = depth_results[0]["depth"]
    fx, fy = K0[0, 0], K0[1, 1]
    cx, cy = K0[0, 2], K0[1, 2]

    pts_3d = []
    valid_bg = []
    for i in range(bg_tracks.shape[1]):
        x, y = bg_tracks[0, i]
        iy = np.clip(int(round(y)), 0, H - 1)
        ix = np.clip(int(round(x)), 0, W - 1)
        d = depth0[iy, ix]
        if d > 0.01 and np.isfinite(d):
            pts_3d.append([(x - cx) * d / fx, (y - cy) * d / fy, d])
            valid_bg.append(i)

    if len(pts_3d) < 6:
        print("  WARNING: Too few valid bg points for PnP, using identity poses")
        return np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))

    pts_3d = np.array(pts_3d, dtype=np.float64)
    valid_bg = np.array(valid_bg)

    camera_matrix = K0[:3, :3].astype(np.float64)
    dist_coeffs = np.zeros(4, dtype=np.float64)

    cam_to_world = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
    prev_rvec = np.zeros(3, dtype=np.float64)
    prev_tvec = np.zeros(3, dtype=np.float64)

    for t in range(1, T):
        vis_at_t = bg_vis[t, valid_bg] > 0
        pts_2d_t = bg_tracks[t, valid_bg[vis_at_t]].astype(np.float64)
        pts_3d_t = pts_3d[vis_at_t]

        if len(pts_2d_t) < 6:
            cam_to_world[t] = cam_to_world[t - 1]
            continue

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d_t,
            pts_2d_t,
            camera_matrix,
            dist_coeffs,
            rvec=prev_rvec.copy(),
            tvec=prev_tvec.copy(),
            useExtrinsicGuess=True,
            iterationsCount=1000,
            reprojectionError=5.0,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success or inliers is None or len(inliers) < 6:
            cam_to_world[t] = cam_to_world[t - 1]
            continue

        prev_rvec = rvec.ravel()
        prev_tvec = tvec.ravel()

        R, _ = cv2.Rodrigues(rvec)
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ tvec.ravel()
        cam_to_world[t] = c2w.astype(np.float32)

    print(f"  PnP fallback: {T} camera poses estimated")
    return cam_to_world


def run_camera_poses(frames, masks, depth_results, fps, use_megasam=True):
    """Estimate per-frame camera-to-world poses.

    Returns:
        cam_to_world: (T, 4, 4) float32
    """
    if use_megasam and _try_import_megasam():
        print("  Using MegaSaM for camera pose estimation")
        return run_camera_poses_megasam(frames, masks, depth_results, fps)
    if use_megasam:
        print("  MegaSaM not available, falling back to PnP")
    return run_camera_poses_pnp_fallback(frames, masks, depth_results)
