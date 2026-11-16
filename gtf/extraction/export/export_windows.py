"""Camera poses of an export and the interaction-window search behind its crop and its `__iwN` window children."""

import os
import numpy as np
import torch
from gtf.utils import rprint as print
from gtf.utils.npz import savez_compressed_level


# ---------------------------------------------------------------------------
# Camera pose estimation
# ---------------------------------------------------------------------------


def _estimate_camera_poses(segment_dir):
    """Load the cached object_pose/camera_poses.npz, or estimate the poses with MegaSaM and cache them.

    Returns cam_to_world (T_video, 4, 4), or None when no poses can be had (the export then assumes a static camera).
    """
    cam_path = os.path.join(segment_dir, "object_pose", "camera_poses.npz")
    if os.path.exists(cam_path):
        return np.load(cam_path)["cam_to_world"]

    # Prerequisites for estimation
    video_path = os.path.join(segment_dir, "video.mp4")
    masks_path = os.path.join(segment_dir, "SAM3", "masks.npz")
    depth_dir = os.path.join(segment_dir, "object_pose", "depth")

    if not os.path.exists(video_path) or not os.path.exists(masks_path):
        return None
    if not os.path.isdir(depth_dir) or len(os.listdir(depth_dir)) == 0:
        return None
    if not torch.cuda.is_available():
        # MegaSaM/DROID needs CUDA (its weights are CUDA-pickled). On a CPU worker (CUDA masked, e.g. the
        # export pool of the GPU/CPU split driver), take the static-camera fallback instead of crashing.
        print("  Camera-pose estimation skipped (no CUDA) — static-camera fallback")
        return None

    print("  Estimating camera poses (no cached camera_poses.npz)...")
    from decord import VideoReader, cpu as decord_cpu

    vr = VideoReader(video_path, ctx=decord_cpu(0), num_threads=2)
    _batch = vr[:]
    frames = _batch.numpy() if hasattr(_batch, "numpy") else _batch.asnumpy()  # (T_video, H, W, 3)
    T_video = len(frames)
    fps = vr.get_avg_fps()

    masks_data = np.load(masks_path)
    masks = masks_data["masks"]  # (T_video, H, W)

    # Load depth results
    depth_results = []
    for t in range(T_video):
        dp = os.path.join(depth_dir, f"{t:05d}.npz")
        if os.path.exists(dp):
            dd = np.load(dp)
            depth_results.append(
                {
                    "depth": dd["depth"],
                    "intrinsics_px": dd["intrinsics_px"],
                    "mask": dd.get("mask", np.ones_like(dd["depth"], dtype=np.uint8)),
                }
            )
        else:
            depth_results.append(None)

    # Fill gaps with nearest valid
    valid_idxs = [i for i, dr in enumerate(depth_results) if dr is not None]
    if not valid_idxs:
        return None
    for i in range(T_video):
        if depth_results[i] is None:
            nearest = min(valid_idxs, key=lambda x: abs(x - i))
            depth_results[i] = depth_results[nearest]

    from gtf.extraction.object_pose.run_object_pose import run_camera_poses

    cam_to_world = run_camera_poses(frames, masks, depth_results, fps, use_megasam=True)

    os.makedirs(os.path.join(segment_dir, "object_pose"), exist_ok=True)
    savez_compressed_level(
        cam_path,
        compresslevel=6,
        allow_pickle=False,
        cam_to_world=cam_to_world,
        camera_method=np.array("export_canonical"),
        pipeline_version=np.int32(0),
    )
    print(f"  Camera poses estimated and cached ({T_video} frames)")

    return cam_to_world


# ---------------------------------------------------------------------------
# Interaction windows
# ---------------------------------------------------------------------------


def _quality_hard_bad(d, acc_max=0.1):
    """Per-keyframe chain-corruption mask for the quality windows.

    A keyframe is bad when the object centroid acceleration exceeds acc_max (m per keyframe step squared) or its
    rotation step is an outlier (> max(10 deg, 3x the segment's median step)). Such frames fail the whole-span temporal
    gates (rot_jerk / ang_acc / acc_max) of segments whose clean sub-trajectories would pass; cutting the windows around
    them lets those sub-trajectories export. Hand validity, mask visibility and contact are not part of this mask: they
    keep their core / pad roles in `_interaction_windows` (a no-hand approach frame is valid pad context)."""
    obj = np.asarray(d["obj_verts"], np.float64)
    T = len(obj)
    cent = obj.mean(axis=1)
    acc = np.zeros(T)
    if T >= 3:
        acc[1:-1] = np.linalg.norm(np.diff(cent, axis=0, n=2), axis=1)
    step_deg = np.zeros(T)
    for t in range(1, T):
        a = obj[t - 1] - obj[t - 1].mean(0)
        b = obj[t] - obj[t].mean(0)
        sa = np.sqrt((a**2).sum(-1).mean())
        sb = np.sqrt((b**2).sum(-1).mean())
        if sa < 1e-9 or sb < 1e-9:
            continue
        U, _, Wt = np.linalg.svd((a / sa).T @ (b / sb))
        R = U @ Wt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Wt
        step_deg[t] = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    step_cap = max(10.0, 3.0 * np.median(step_deg[step_deg > 0]) if (step_deg > 0).any() else 10.0)
    return (acc > acc_max) | (step_deg > step_cap)


def _interaction_windows(
    d,
    masks,
    fps,
    contact_m=0.05,
    pad_s=0.75,
    min_keep_s=2.5,
    min_trim_kf=8,
    merge_gap_s=2.0,
    min_core_kf=8,
    quality_bad=None,
):
    """All disjoint keyframe windows [i0, i1] of live interaction: object visibly tracked (SAM3
    mask with sqrt-area >= 25% of its median; entering/exiting slivers don't count) and some
    valid hand within contact_m of the object. Core runs are merged across gaps <= merge_gap_s
    (re-grips / brief mask dropouts stay one trajectory; longer no-contact gaps split: a clip can
    hold multiple disconnected manipulations, each a separate training trajectory). Windows with
    fewer than min_core_kf contact keyframes are dropped (a spurious near-miss blip must not
    become its own window, it could wrongly clear the interaction_sustain gate). Each surviving
    run is padded by pad_s (approach/release context), clamped to the mask-visible extent and to
    the midpoint of the gap toward its neighbors (windows stay disjoint), and must keep
    min_keep_s. Returns a list of dicts sorted by position: {"i0", "i1", "core_kf"}. Empty list =
    no crop warranted (the gates decide).

    min_trim_kf applies only to the single-window case: if the sole window trims fewer than
    min_trim_kf keyframes off the segment, cropping is churn without value -> empty list.

    quality_bad (see _quality_hard_bad): optional per-keyframe bool mask of chain corruption.
    Three hooks, each a no-op when None / all-False:
    (1) core excludes quality-bad frames; (2) gaps containing a quality-bad frame are
    unmergeable (a spike is a separator; a mask dropout / re-grip stays mergeable);
    (3) pads never cross a quality-bad frame (approach/release context is wanted, chain
    corruption is not)."""
    fi = np.asarray(d["frame_indices"], np.int64)
    T = len(fi)
    if quality_bad is None:
        quality_bad = np.zeros(T, bool)
    quality_bad = np.asarray(quality_bad, bool)
    obj_v = np.asarray(d["obj_verts"], np.float64)
    sizes = np.zeros(T)
    for i, f in enumerate(fi):
        if 0 <= int(f) < len(masks):
            a = float(masks[int(f)].sum())
            sizes[i] = np.sqrt(a) if a >= 30 else 0.0
    if (sizes > 0).sum() < 4:
        return []
    visible = sizes >= max(np.sqrt(30.0), 0.25 * float(np.median(sizes[sizes > 0])))
    contact = np.zeros(T, bool)
    ostride = max(1, obj_v.shape[1] // 300)
    for side in ("left", "right"):
        vk, hk = f"hand_{side}_valid", f"hand_{side}_verts"
        if vk not in d or hk not in d:
            continue
        val = np.asarray(d[vk]).astype(bool)
        hv = np.asarray(d[hk], np.float64)
        for t in range(T):
            if val[t] and not contact[t]:
                dist = np.linalg.norm(hv[t][::15][:, None] - obj_v[t][::ostride][None], axis=2).min()
                contact[t] = bool(dist < contact_m)
    core = visible & contact & ~quality_bad
    if not core.any():
        return []
    stride = max(float(np.median(np.diff(fi))), 1.0) if T > 1 else 1.0
    kf_per_s = fps / stride
    pad_kf = int(round(pad_s * kf_per_s))
    merge_gap_kf = int(round(merge_gap_s * kf_per_s))
    min_keep_kf = max(int(round(min_keep_s * kf_per_s)), 8)
    v_idx = np.where(visible)[0]

    # core runs -> merge across small gaps -> drop low-core runs
    runs = []
    start = None
    for i in range(T + 1):
        on = i < T and core[i]
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append([start, i - 1])
            start = None
    merged = [runs[0]]
    for r0, r1 in runs[1:]:
        gap_clean = not quality_bad[merged[-1][1] + 1 : r0].any()
        if r0 - merged[-1][1] - 1 <= merge_gap_kf and gap_clean:
            merged[-1][1] = r1
        else:
            merged.append([r0, r1])
    merged = [r for r in merged if int(core[r[0] : r[1] + 1].sum()) >= min_core_kf]
    if not merged:
        return []

    windows = []
    for j, (r0, r1) in enumerate(merged):
        i0 = max(r0 - pad_kf, int(v_idx[0]), 0)
        i1 = min(r1 + pad_kf, int(v_idx[-1]), T - 1)
        if j > 0:  # keep disjoint: pads meet at the gap midpoint
            i0 = max(i0, (merged[j - 1][1] + r0 + 1) // 2)
        if j < len(merged) - 1:
            i1 = min(i1, (r1 + merged[j + 1][0]) // 2)
        bb = np.where(quality_bad[i0 : r0 + 1])[0]  # pads never cross a quality-bad frame
        if len(bb):
            i0 = i0 + int(bb[-1]) + 1
        bb = np.where(quality_bad[r1 : i1 + 1])[0]
        if len(bb):
            i1 = r1 + int(bb[0]) - 1
        if (i1 - i0 + 1) >= min_keep_kf:
            windows.append({"i0": int(i0), "i1": int(i1), "core_kf": int(core[r0 : r1 + 1].sum())})
    if len(windows) == 1 and (T - (windows[0]["i1"] - windows[0]["i0"] + 1)) < min_trim_kf:
        return []
    return windows
