"""The object-pose stage for one segment: depth, tracks, cameras, bundle adjustment, outputs."""

import json
import os
import time
import cv2
import numpy as np
import torch
from decord import VideoReader, cpu
from pathlib import Path
from gtf.utils import rprint as print
from gtf.extraction.object_pose.pose_cache import (
    OBJECT_POSE_PIPELINE_VERSION,
    ObjectPoseTerminalReject,
    _binding_cache_fields,
    _cache_binding_matches,
    _load_previous_stage_npz,
    _npz_scalar,
    _object_pose_data_and_stat_binding,
    _object_pose_done_matches,
    _object_pose_stage_binding,
    _output_artifact_binding,
    _publish_object_pose_done,
    _savez_compressed_atomic,
    _sha256_file,
    _write_json_atomic,
)
from gtf.extraction.object_pose.object_depth import _load_depth_cache, _load_matching_depth_cache_contract, publish_depth_cache_contract, run_depth_estimation
from gtf.extraction.object_pose.point_tracking import _cotracker_checkpoint_sha256, offload_cotracker_cache, run_cotracker, select_reference_frame
from gtf.extraction.object_pose.camera_poses import _camera_model_checkpoint_sha256, _try_import_megasam, run_camera_poses
from gtf.extraction.object_pose.bundle_adjustment import ObjectBundleAdjuster


DEFAULT_TRACKING_MODE = "hybrid"


def render_visualization(
    frames,
    tracks_data,
    object_poses,
    object_points,
    cam_to_world,
    intrinsics_list,
    reproj_errors,
    out_path,
    fps,
):
    """Render 4-panel visualization video: Original | Tracks | Pose Overlay | 3D Cloud."""
    import imageio.v3 as iio

    T, H, W, _ = frames.shape
    tracks = tracks_data["tracks"]
    visibility = tracks_data["visibility"]
    N = tracks.shape[1]
    rotations = object_poses["rotations"]
    translations = object_poses["translations"]

    np.random.seed(42)
    colors = np.random.randint(50, 255, (N, 3), dtype=np.uint8)

    pH = H // 2 + (H // 2) % 2
    pW = W // 2 + (W // 2) % 2

    pts = object_points
    N_pts = pts.shape[0]
    bb_min, bb_max = pts.min(axis=0), pts.max(axis=0)
    bb_center = (bb_min + bb_max) / 2
    bb_corners = np.array(
        [
            [bb_min[0], bb_min[1], bb_min[2]],
            [bb_max[0], bb_min[1], bb_min[2]],
            [bb_max[0], bb_max[1], bb_min[2]],
            [bb_min[0], bb_max[1], bb_min[2]],
            [bb_min[0], bb_min[1], bb_max[2]],
            [bb_max[0], bb_min[1], bb_max[2]],
            [bb_max[0], bb_max[1], bb_max[2]],
            [bb_min[0], bb_max[1], bb_max[2]],
        ]
    )
    bb_edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    axis_len = np.linalg.norm(bb_max - bb_min) * 0.2

    vis_frames = []
    for t in range(T):
        frame_small = cv2.resize(frames[t], (pW, pH))
        scale_x, scale_y = pW / W, pH / H

        # Panel 1: Original with frame counter
        p1 = frame_small.copy()
        cv2.putText(
            p1,
            f"Frame {t}/{T}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Panel 2: Tracks
        p2 = frame_small.copy()
        for i in range(N):
            if not visibility[t, i]:
                continue
            x, y = int(tracks[t, i, 0] * scale_x), int(tracks[t, i, 1] * scale_y)
            color = tuple(int(c) for c in colors[i])
            cv2.circle(p2, (x, y), 2, color, -1)
            for tt in range(max(0, t - 10), t):
                if visibility[tt, i] and visibility[min(tt + 1, t), i]:
                    x0 = int(tracks[tt, i, 0] * scale_x)
                    y0 = int(tracks[tt, i, 1] * scale_y)
                    x1 = int(tracks[min(tt + 1, t), i, 0] * scale_x)
                    y1 = int(tracks[min(tt + 1, t), i, 1] * scale_y)
                    cv2.line(p2, (x0, y0), (x1, y1), color, 1)

        # Panel 3: Pose overlay
        p3 = frame_small.copy()
        R_obj, t_obj = rotations[t], translations[t]
        K = intrinsics_list[t]
        w2c = np.linalg.inv(cam_to_world[t])

        def project_point(pt_local):
            pt_cam = w2c[:3, :3] @ (R_obj @ pt_local + t_obj) + w2c[:3, 3]
            if pt_cam[2] < 0.01:
                return None
            return int((K[0, 0] * pt_cam[0] / pt_cam[2] + K[0, 2]) * scale_x), int((K[1, 1] * pt_cam[1] / pt_cam[2] + K[1, 2]) * scale_y)

        for i0, i1 in bb_edges:
            p0, p1_pt = project_point(bb_corners[i0]), project_point(bb_corners[i1])
            if p0 is not None and p1_pt is not None:
                cv2.line(p3, p0, p1_pt, (0, 255, 255), 1)

        origin = project_point(bb_center)
        for axis_dir, color in [
            (np.array([axis_len, 0, 0]), (255, 0, 0)),
            (np.array([0, axis_len, 0]), (0, 255, 0)),
            (np.array([0, 0, axis_len]), (0, 0, 255)),
        ]:
            tip = project_point(bb_center + axis_dir)
            if origin is not None and tip is not None:
                cv2.line(p3, origin, tip, color, 2)

        cv2.putText(
            p3,
            f"Reproj: {reproj_errors[t]:.1f}px",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 0),
            1,
        )

        # Panel 4: 3D cloud (orbiting view)
        p4 = np.zeros((pH, pW, 3), dtype=np.uint8)
        angle = 2 * np.pi * t / max(T, 1)
        cam_dist = np.linalg.norm(bb_max - bb_min) * 2.5
        cam_pos = np.array([cam_dist * np.cos(angle), cam_dist * 0.3, cam_dist * np.sin(angle)])
        forward = -cam_pos / np.linalg.norm(cam_pos)
        right = np.cross(forward, np.array([0, 1, 0]))
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1, 0, 0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        virtual_f = pW * 0.8
        for i in range(N_pts):
            rel = pts[i] - bb_center
            sx = int(pW / 2 + (rel @ right) * virtual_f / cam_dist)
            sy = int(pH / 2 - (rel @ up) * virtual_f / cam_dist)
            if 0 <= sx < pW and 0 <= sy < pH:
                cv2.circle(p4, (sx, sy), 2, (0, 200, 0), -1)

        origin_3d = (pW // 2, pH // 2)
        for axis_dir, color in [
            (np.array([axis_len, 0, 0]), (255, 0, 0)),
            (np.array([0, axis_len, 0]), (0, 255, 0)),
            (np.array([0, 0, axis_len]), (0, 0, 255)),
        ]:
            tip = (
                int(pW / 2 + (axis_dir @ right) * virtual_f / cam_dist),
                int(pH / 2 - (axis_dir @ up) * virtual_f / cam_dist),
            )
            cv2.line(p4, origin_3d, tip, color, 2)

        cv2.putText(p4, "3D View", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        top = np.concatenate([p1, p2], axis=1)
        bottom = np.concatenate([p3, p4], axis=1)
        vis_frames.append(np.concatenate([top, bottom], axis=0))

    # Ensure even dimensions
    h_out, w_out = vis_frames[0].shape[:2]
    if h_out % 2 != 0 or w_out % 2 != 0:
        pad_h, pad_w = h_out % 2, w_out % 2
        vis_frames = [np.pad(f, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant") for f in vis_frames]

    iio.imwrite(out_path, np.stack(vis_frames), fps=fps, codec="libx264")
    print(f"  Visualization saved: {out_path}")


def process_segment(
    depth_estimator,
    name,
    seg_dir,
    pose_dir,
    force=False,
    use_megasam=True,
    cotracker_grid_size=30,
    tracking_mode=DEFAULT_TRACKING_MODE,
    visualize_only=False,
    visualize=False,
    require_megasam=False,
    defer_bundle_adjustment=False,
    object_ba=None,
):
    """Run the full 6D object-pose pipeline on one segment. Returns True on success.

    object_ba is the chain's chains.ObjectBA (None or the default ObjectBA(): plain BA; otherwise the
    depth-relief BA of object_pose_relief.py). With defer_bundle_adjustment the stage stops after the
    cameras and publishes the pre-BA receipt (pre_ba_metadata.json + .pre_ba_done) for a separate BA worker.
    """
    video_path = os.path.join(seg_dir, "video.mp4")
    sam3_dir = os.path.join(seg_dir, "SAM3")
    depth_dir = os.path.join(pose_dir, "depth")
    sam3d_dir = os.path.join(seg_dir, "sam3d")
    metadata_path = Path(pose_dir) / "metadata.json"
    previous_metadata = {}
    complete_marker = Path(pose_dir) / ".object_pose_done"
    try:
        if _object_pose_done_matches(complete_marker, metadata_path):
            previous_metadata = json.loads(metadata_path.read_text())
            if not isinstance(previous_metadata, dict):
                previous_metadata = {}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        previous_metadata = {}
    if not previous_metadata:
        # An asynchronous BA worker resumes from this content-bound receipt.
        # It is deliberately separate from the final completion marker: a
        # crash may reuse exact tracks/camera bytes, but can never acknowledge
        # the segment until BA and final metadata have been atomically written.
        pre_ba_metadata_path = Path(pose_dir) / "pre_ba_metadata.json"
        pre_ba_marker_path = Path(pose_dir) / ".pre_ba_done"
        try:
            candidate = json.loads(pre_ba_metadata_path.read_text())
            if (
                _object_pose_done_matches(pre_ba_marker_path, pre_ba_metadata_path)
                and isinstance(candidate, dict)
                and candidate.get("format") == "object-pose-pre-ba-cache-v1"
                and candidate.get("pipeline_version") == OBJECT_POSE_PIPELINE_VERSION
                and candidate.get("segment") == name
            ):
                previous_metadata = candidate
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    if not os.path.exists(os.path.join(sam3_dir, ".sam3_done")):
        print("  SKIP: no SAM3 .sam3_done marker")
        return False
    masks_path = os.path.join(sam3_dir, "masks.npz")
    if not os.path.exists(masks_path):
        print("  SKIP: no SAM3 masks.npz")
        return False

    megasam_available = _try_import_megasam() if use_megasam else False
    if require_megasam and (not use_megasam or not megasam_available):
        raise RuntimeError("MegaSaM was required for this extraction run but its CUDA extensions are unavailable")

    if not visualize_only:
        os.makedirs(pose_dir, exist_ok=True)
        Path(pose_dir, ".object_pose_done").unlink(missing_ok=True)

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
    fps = vr.get_avg_fps()
    T = len(vr)
    _batch = vr[:]
    frames = _batch.asnumpy() if hasattr(_batch, "asnumpy") else np.asarray(_batch)
    H, W = frames.shape[1], frames.shape[2]
    del vr

    masks = np.load(masks_path)["masks"]
    if masks.ndim != 3 or masks.shape[0] != T:
        raise ValueError(
            f"object pose requires one SAM3 mask per decoded video frame (rerun SAM3 on the current video): masks={masks.shape}, video_frames={T}"
        )
    if masks.shape[1] != H or masks.shape[2] != W:
        masks_resized = np.zeros((T, H, W), dtype=np.uint8)
        for t in range(T):
            masks_resized[t] = cv2.resize(masks[t], (W, H), interpolation=cv2.INTER_NEAREST)
        masks = masks_resized

    obj_frames = sum(1 for t in range(T) if masks[t].any())
    if obj_frames < T * 0.3:
        raise ObjectPoseTerminalReject(f"object_pose_low_visibility: object visible in only {obj_frames}/{T} frames (<30%)")

    os.makedirs(pose_dir, exist_ok=True)
    timings = dict(previous_metadata.get("timings", {}))
    camera_method = "megasam" if megasam_available else "pnp"
    tracks_recomputed = False
    camera_recomputed = False

    # ---- Stage 1: Depth ----
    depth_done = os.path.join(pose_dir, ".depth_done")
    depth_contract = None
    if not force:
        depth_contract = _load_matching_depth_cache_contract(
            video_path=video_path,
            depth_dir=depth_dir,
            pose_dir=pose_dir,
            sam3d_dir=sam3d_dir,
            n_frames=T,
            decoded_frame_shape=frames.shape,
        )
    need_depth_compute = not visualize_only and (force or depth_contract is None)
    if need_depth_compute:
        # CoTracker is cached across segments by default. It must leave the
        # device before MoGe is made resident for a new depth roster.
        offload_cotracker_cache()
        print(f"  Stage 1: MoGe depth estimation ({T} frames)...")
        if depth_estimator is None:
            raise RuntimeError("metric-depth recomputation requires a DepthEstimator")
        Path(depth_done).unlink(missing_ok=True)
        t0 = time.time()
        depth_write_bindings = {}
        depth_write_set_binding = {}
        depth_results, depth_reused = run_depth_estimation(
            depth_estimator,
            frames,
            depth_dir,
            # A missing contract means the existing bytes are not known to
            # belong to this exact RGB/model/source recipe, so the complete
            # roster is recomputed; filling only the missing member could
            # retain stale depth from an earlier video.
            sam3d_dir=sam3d_dir,
            output_bindings=depth_write_bindings,
            output_set_binding=depth_write_set_binding,
        )
        depth_contract = publish_depth_cache_contract(
            video_path=video_path,
            depth_dir=depth_dir,
            pose_dir=pose_dir,
            sam3d_dir=sam3d_dir,
            n_frames=T,
            decoded_frame_shape=frames.shape,
            sam3d_depth_frames_reused=depth_reused,
            produced_bindings=depth_write_bindings,
            produced_set_sha256=depth_write_set_binding.get("sha256"),
            validated_results=depth_results,
        )
        timings["depth_s"] = time.time() - t0
        print(f"  Stage 1 done in {timings['depth_s']:.1f}s")
        torch.cuda.empty_cache()
    else:
        print("  Stage 1: Loading cached depth")
        depth_results = _load_depth_cache(depth_dir, T, frame_shape=(H, W))
        if depth_results is None:
            print("  SKIP: missing or malformed depth cache")
            return False
        if depth_contract is None:
            # visualize-only may inspect a cache without a depth contract, but
            # normal extraction never consumes an unattested cache.
            depth_reused = int(previous_metadata.get("sam3d_depth_frames_reused", 0))
        else:
            depth_reused = int(depth_contract["sam3d_depth_frames_reused"])

    # MoGe and CoTracker resident together leave too little memory for
    # CoTracker's attention. Only one inference model needs to be on the GPU
    # at a time; the depth arrays are already on CPU at this point.
    if depth_estimator is not None:
        depth_estimator.offload()

    shared_input_binding, shared_input_stat_binding = _object_pose_data_and_stat_binding(
        video_path=video_path,
        masks_path=masks_path,
        depth_dir=depth_dir,
        n_frames=T,
        decoded_frame_shape=frames.shape,
        verified_depth_output_binding=(depth_contract.get("output_binding") if isinstance(depth_contract, dict) else None),
    )

    # ---- Stage 2: CoTracker3 ----
    tracks_path = os.path.join(pose_dir, "tracks.npz")
    tracks_done = os.path.join(pose_dir, ".tracks_done")
    selected_ref_frame_idx = select_reference_frame(masks, depth_results)
    normalized_tracking_mode = str(tracking_mode).lower()
    tracks_input_binding = _object_pose_stage_binding(
        shared_input_binding,
        "tracks",
        tracking_mode=normalized_tracking_mode,
        cotracker_grid_size=int(cotracker_grid_size),
        reference_frame_index=int(selected_ref_frame_idx),
        cotracker_checkpoint_sha256=_cotracker_checkpoint_sha256(),
    )
    tracks_binding_fields = _binding_cache_fields("tracks", tracks_input_binding)

    cached_tracks = _load_previous_stage_npz(previous_metadata, "tracks", tracks_path) if not force and os.path.exists(tracks_done) and os.path.exists(tracks_path) else None
    if cached_tracks is not None:
        cached_mode = _npz_scalar(cached_tracks.get("tracking_mode"), "")
        cached_version = _npz_scalar(cached_tracks.get("pipeline_version"), -1)
        if (
            cached_mode == normalized_tracking_mode
            and int(cached_version) == OBJECT_POSE_PIPELINE_VERSION
            and _cache_binding_matches(cached_tracks, "tracks", tracks_input_binding)
        ):
            print("  Stage 2: Loading cached tracks")
            tracks_data = cached_tracks
        else:
            print(f"  Stage 2: Recomputing tracks (cache mode={cached_mode!r}, version={cached_version}, content_binding=stale_or_absent)")
            Path(tracks_done).unlink(missing_ok=True)
            t0 = time.time()
            tracks_data = run_cotracker(
                frames,
                masks,
                selected_ref_frame_idx,
                grid_size=cotracker_grid_size,
                tracking_mode=normalized_tracking_mode,
            )
            tracks_data.update(tracks_binding_fields)
            _savez_compressed_atomic(tracks_path, **tracks_data)
            Path(tracks_done).write_text(tracks_input_binding["input_digest_sha256"] + "\n")
            timings["tracking_s"] = time.time() - t0
            print(f"  Stage 2 done in {timings['tracking_s']:.1f}s")
            tracks_recomputed = True
    elif not visualize_only:
        print("  Stage 2: CoTracker3 dense tracking...")
        Path(tracks_done).unlink(missing_ok=True)
        t0 = time.time()
        tracks_data = run_cotracker(
            frames,
            masks,
            selected_ref_frame_idx,
            grid_size=cotracker_grid_size,
            tracking_mode=normalized_tracking_mode,
        )
        tracks_data.update(tracks_binding_fields)
        _savez_compressed_atomic(tracks_path, **tracks_data)
        timings["tracking_s"] = time.time() - t0
        print(f"  Stage 2 done in {timings['tracking_s']:.1f}s")
        tracks_recomputed = True
        Path(tracks_done).write_text(tracks_input_binding["input_digest_sha256"] + "\n")
        torch.cuda.empty_cache()
    else:
        if not os.path.exists(tracks_path):
            print("  SKIP: missing tracks cache for visualize-only")
            return False
        tracks_data = dict(np.load(tracks_path))

    # Save reference frame visualization
    ref_idx = int(tracks_data["ref_frame_idx"])
    ref_vis_path = os.path.join(pose_dir, "ref_frame_points.png")
    ref_img = frames[ref_idx].copy()
    tracks_at_ref = tracks_data["tracks"][ref_idx]
    vis_at_ref = tracks_data["visibility"][ref_idx]
    np.random.seed(42)
    pt_colors = np.random.randint(50, 255, (tracks_at_ref.shape[0], 3), dtype=np.uint8)
    for i in range(tracks_at_ref.shape[0]):
        if vis_at_ref[i]:
            x, y = int(round(tracks_at_ref[i, 0])), int(round(tracks_at_ref[i, 1]))
            color = tuple(int(c) for c in pt_colors[i])
            cv2.circle(ref_img, (x, y), 3, color, -1)
            cv2.circle(ref_img, (x, y), 3, (255, 255, 255), 1)
    contours, _ = cv2.findContours(
        (masks[ref_idx] > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(ref_img, contours, -1, (0, 255, 0), 1)
    cv2.putText(
        ref_img,
        f"Ref frame {ref_idx} | {int(vis_at_ref.sum())} tracks",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.imwrite(ref_vis_path, cv2.cvtColor(ref_img, cv2.COLOR_RGB2BGR))
    print(f"  Saved ref frame visualization: {ref_vis_path}")

    # ---- Stage 3: Camera poses ----
    cam_poses_path = os.path.join(pose_dir, "camera_poses.npz")
    cam_done = os.path.join(pose_dir, ".cam_poses_done")
    camera_input_binding = _object_pose_stage_binding(
        shared_input_binding,
        "camera",
        camera_method=camera_method,
        camera_model_checkpoint_sha256=(_camera_model_checkpoint_sha256(camera_method)),
        fps=float(fps),
    )
    camera_binding_fields = _binding_cache_fields("camera", camera_input_binding)

    if (
        cam_data := (
            _load_previous_stage_npz(previous_metadata, "camera_poses", cam_poses_path) if not force and os.path.exists(cam_done) and os.path.exists(cam_poses_path) else None
        )
    ) is not None:
        cached_camera_method = _npz_scalar(cam_data.get("camera_method"), "")
        cached_version = _npz_scalar(cam_data.get("pipeline_version"), -1)
        if (
            cached_camera_method == camera_method
            and int(cached_version) == OBJECT_POSE_PIPELINE_VERSION
            and _cache_binding_matches(cam_data, "camera", camera_input_binding)
        ):
            print("  Stage 3: Loading cached camera poses")
            cam_to_world = cam_data["cam_to_world"]
        else:
            print(f"  Stage 3: Recomputing camera poses (cache method={cached_camera_method!r}, version={cached_version})")
            Path(cam_done).unlink(missing_ok=True)
            t0 = time.time()
            cam_to_world = run_camera_poses(frames, masks, depth_results, fps, use_megasam=use_megasam)
            _savez_compressed_atomic(
                cam_poses_path,
                cam_to_world=cam_to_world,
                camera_method=np.array(camera_method),
                pipeline_version=np.int32(OBJECT_POSE_PIPELINE_VERSION),
                **camera_binding_fields,
            )
            Path(cam_done).write_text(camera_input_binding["input_digest_sha256"] + "\n")
            timings["camera_poses_s"] = time.time() - t0
            print(f"  Stage 3 done in {timings['camera_poses_s']:.1f}s")
            camera_recomputed = True
    elif not visualize_only:
        print("  Stage 3: Camera pose estimation...")
        Path(cam_done).unlink(missing_ok=True)
        t0 = time.time()
        cam_to_world = run_camera_poses(frames, masks, depth_results, fps, use_megasam=use_megasam)
        _savez_compressed_atomic(
            cam_poses_path,
            cam_to_world=cam_to_world,
            camera_method=np.array(camera_method),
            pipeline_version=np.int32(OBJECT_POSE_PIPELINE_VERSION),
            **camera_binding_fields,
        )
        timings["camera_poses_s"] = time.time() - t0
        print(f"  Stage 3 done in {timings['camera_poses_s']:.1f}s")
        camera_recomputed = True
        Path(cam_done).write_text(camera_input_binding["input_digest_sha256"] + "\n")
        torch.cuda.empty_cache()
    else:
        if not os.path.exists(cam_poses_path):
            print("  SKIP: missing camera poses cache for visualize-only")
            return False
        cam_to_world = np.load(cam_poses_path)["cam_to_world"]

    if defer_bundle_adjustment:
        if visualize_only:
            raise ValueError("cannot defer bundle adjustment in visualize-only mode")
        # Publish only the exact reusable prefix. This is not a completion
        # marker, and callers must keep their queue claim running until a BA
        # worker publishes .object_pose_done.
        pre_ba_metadata_path = Path(pose_dir) / "pre_ba_metadata.json"
        pre_ba_marker_path = Path(pose_dir) / ".pre_ba_done"
        pre_ba_metadata = {
            "format": "object-pose-pre-ba-cache-v1",
            "pipeline_version": OBJECT_POSE_PIPELINE_VERSION,
            "segment": name,
            "shared_input_binding": shared_input_binding,
            "tracks_input_binding": tracks_input_binding,
            "camera_input_binding": camera_input_binding,
            "timings": timings,
            "output_artifact_bindings": {
                "tracks": _output_artifact_binding(tracks_path),
                "camera_poses": _output_artifact_binding(cam_poses_path),
            },
        }
        _write_json_atomic(pre_ba_metadata_path, pre_ba_metadata)
        _publish_object_pose_done(pre_ba_marker_path, pre_ba_metadata_path)
        return True

    # ---- Stage 4: Bundle adjustment ----
    obj_poses_path = os.path.join(pose_dir, "object_poses.npz")
    obj_points_path = os.path.join(pose_dir, "object_points.npz")
    ba_done = os.path.join(pose_dir, ".ba_done")
    ba_cache_valid = False
    # A non-default chains.ObjectBA adds the depth-relief term (object_pose_relief.py).
    ba_relief_spec = object_ba.spec() if object_ba is not None else None
    ba_input_binding = _object_pose_stage_binding(
        shared_input_binding,
        "bundle-adjustment",
        tracks_sha256=_sha256_file(tracks_path),
        tracks_input_digest_sha256=tracks_input_binding["input_digest_sha256"],
        camera_poses_sha256=_sha256_file(cam_poses_path),
        camera_input_digest_sha256=camera_input_binding["input_digest_sha256"],
        tracking_mode=normalized_tracking_mode,
        camera_method=camera_method,
        **({"ba_relief_spec": json.dumps(ba_relief_spec, sort_keys=True)} if ba_relief_spec is not None else {}),
    )
    ba_binding_fields = _binding_cache_fields("ba", ba_input_binding)

    cached_object_poses = (
        _load_previous_stage_npz(previous_metadata, "object_poses", obj_poses_path)
        if not force and not tracks_recomputed and not camera_recomputed and os.path.exists(ba_done) and os.path.exists(obj_poses_path) and os.path.exists(obj_points_path)
        else None
    )
    cached_object_points = _load_previous_stage_npz(previous_metadata, "object_points", obj_points_path) if cached_object_poses is not None else None
    if (
        not force
        and not tracks_recomputed
        and not camera_recomputed
        and os.path.exists(ba_done)
        and os.path.exists(obj_poses_path)
        and os.path.exists(obj_points_path)
        and cached_object_poses is not None
        and cached_object_points is not None
    ):
        obj_data = cached_object_poses
        cached_mode = _npz_scalar(obj_data.get("tracking_mode"), "")
        cached_camera_method = _npz_scalar(obj_data.get("camera_method"), "")
        cached_version = _npz_scalar(obj_data.get("pipeline_version"), -1)
        ba_cache_valid = (
            cached_mode == normalized_tracking_mode
            and cached_camera_method == camera_method
            and int(cached_version) == OBJECT_POSE_PIPELINE_VERSION
            and _cache_binding_matches(obj_data, "ba", ba_input_binding)
        )
        if ba_cache_valid:
            print("  Stage 4: Loading cached object poses")
            rotations = obj_data["rotations"]
            translations = obj_data["translations"]
            reproj_errors = obj_data["reprojection_errors"]
            object_points = cached_object_points["object_points"]
        else:
            print(f"  Stage 4: Recomputing object poses (cache mode={cached_mode!r}, camera={cached_camera_method!r}, version={cached_version})")

    if not ba_cache_valid:
        if not visualize_only:
            print("  Stage 4: Object bundle adjustment...")
            Path(ba_done).unlink(missing_ok=True)
            t0 = time.time()
            ref_idx = int(tracks_data["ref_frame_idx"])
            if ba_relief_spec is not None:
                from gtf.extraction.object_pose.object_pose_relief import ReliefBundleAdjuster

                ba = ReliefBundleAdjuster(
                    tracks_data["tracks"],
                    tracks_data["visibility"],
                    depth_results,
                    cam_to_world,
                    ref_idx,
                    boundary_dists=tracks_data.get("boundary_dists"),
                    object_masks_all=masks,
                    relief_weight=object_ba.relief_weight,
                    relief_delta=object_ba.relief_delta,
                    rotation_smooth=object_ba.rotation_smooth,
                )
                reproj_errors = ba.optimize()
            else:
                ba = ObjectBundleAdjuster(
                    tracks_data["tracks"],
                    tracks_data["visibility"],
                    depth_results,
                    cam_to_world,
                    ref_idx,
                    boundary_dists=tracks_data.get("boundary_dists"),
                )
                reproj_errors = ba.optimize()
            rotations, translations, object_points = ba.get_results()
            timings["bundle_adjustment_s"] = time.time() - t0
            print(f"  Stage 4 done in {timings['bundle_adjustment_s']:.1f}s")

            _savez_compressed_atomic(
                obj_poses_path,
                rotations=rotations,
                translations=translations,
                reprojection_errors=reproj_errors,
                tracking_mode=np.array(normalized_tracking_mode),
                camera_method=np.array(camera_method),
                pipeline_version=np.int32(OBJECT_POSE_PIPELINE_VERSION),
                **ba_binding_fields,
            )
            _savez_compressed_atomic(obj_points_path, object_points=object_points)
            Path(ba_done).write_text(ba_input_binding["input_digest_sha256"] + "\n")
            del ba
            torch.cuda.empty_cache()

            # BA health marker: an evidence-frame median reprojection error above 30 px
            # marks a diverged solve.
            suspect_path = os.path.join(pose_dir, ".ba_suspect")
            ev = reproj_errors[np.isfinite(reproj_errors)]
            ev_med = float(np.median(ev)) if len(ev) else None
            if ev_med is None or ev_med > 30.0:
                print(f"  WARNING: suspect BA solve — evidence-frame median {ev_med if ev_med is not None else 'None (no evidence frames)'}px; writing .ba_suspect")
                open(suspect_path, "w").write(f"{ev_med}\n")
            elif os.path.exists(suspect_path):
                os.remove(suspect_path)
        else:
            if not os.path.exists(obj_poses_path):
                print("  SKIP: missing object poses cache for visualize-only")
                return False
            obj_data = np.load(obj_poses_path)
            rotations = obj_data["rotations"]
            translations = obj_data["translations"]
            reproj_errors = obj_data["reprojection_errors"]
            object_points = np.load(obj_points_path)["object_points"]

    # ---- Stage 5: Optional diagnostic visualization ----
    vis_path = os.path.join(pose_dir, "vis_pose.mp4")
    if visualize or visualize_only:
        print("  Stage 5: Rendering visualization...")
        t0 = time.time()
        render_visualization(
            frames,
            tracks_data,
            {"rotations": rotations, "translations": translations},
            object_points,
            cam_to_world,
            [dr["intrinsics_px"] for dr in depth_results],
            reproj_errors,
            vis_path,
            fps,
        )
        timings["visualization_s"] = time.time() - t0

    metadata = {
        "segment": name,
        "pipeline_version": OBJECT_POSE_PIPELINE_VERSION,
        "num_frames": T,
        "resolution": [H, W],
        "fps": float(fps),
        "num_tracks": int(tracks_data["tracks"].shape[1]),
        "ref_frame_idx": int(tracks_data["ref_frame_idx"]),
        "tracking_mode": normalized_tracking_mode,
        "cotracker_grid_size": int(cotracker_grid_size),
        "tracking_switch_frame": int(_npz_scalar(tracks_data.get("switch_frame"), -1)),
        "camera_method": camera_method,
        "shared_input_binding": shared_input_binding,
        "shared_input_stat_binding": shared_input_stat_binding,
        "tracks_input_binding": tracks_input_binding,
        "camera_input_binding": camera_input_binding,
        "ba_input_binding": ba_input_binding,
        "depth_input_binding": (depth_contract.get("input_binding") if depth_contract is not None else None),
        "depth_output_binding": (depth_contract.get("output_binding") if depth_contract is not None else None),
        "sam3d_depth_frames_reused": int(depth_reused),
        # frames without evidence are NaN and stay out of the mean / median
        "mean_reproj_error_px": float(np.nanmean(reproj_errors)) if np.isfinite(reproj_errors).any() else None,
        "median_reproj_error_px": float(np.nanmedian(reproj_errors)) if np.isfinite(reproj_errors).any() else None,
        "n_evidence_frames": int(np.isfinite(reproj_errors).sum()),
        "timings": timings,
    }
    metadata["output_artifact_bindings"] = {
        "tracks": _output_artifact_binding(tracks_path),
        "camera_poses": _output_artifact_binding(cam_poses_path),
        "object_poses": _output_artifact_binding(obj_poses_path),
        "object_points": _output_artifact_binding(obj_points_path),
    }
    _write_json_atomic(metadata_path, metadata)
    _publish_object_pose_done(Path(pose_dir) / ".object_pose_done", metadata_path)
    return True
