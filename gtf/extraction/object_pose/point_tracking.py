"""Dense point tracks on the object (CoTracker): reference frame, seeding, temporal windows, crop stabilization."""

import gc
import cv2
import numpy as np
import torch
from pathlib import Path
from gtf.utils import rprint as print
from gtf.extraction.object_pose.pose_cache import OBJECT_POSE_PIPELINE_VERSION, ObjectPoseTerminalReject, _sha256_file


COTRACKER_MAX_TEMPORAL_FRAMES = 512
_COTRACKER_CACHE = {}
_COTRACKER_CHECKPOINT_DIGEST = None


def get_cotracker_predictor(device):
    """Load CoTracker once per process/device and reuse its stateless eval forward."""
    key = str(torch.device(device))
    predictor = _COTRACKER_CACHE.get(key)
    if predictor is None:
        predictor = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
        predictor = predictor.to(device).eval()
        _COTRACKER_CACHE[key] = predictor
    else:
        # Cached predictors rest on CPU between segments so MoGe can use the
        # device alone. Move the same model back without reloading weights.
        predictor = predictor.to(device).eval()
    return predictor


def offload_cotracker_cache():
    """Keep cached CoTracker weights, but release their GPU residency."""

    predictors = list(_COTRACKER_CACHE.values())
    for predictor in predictors:
        # Keeping MoGe and CoTracker resident together leaves too little room
        # for attention on longer clips, even on an 80 GiB GPU. Explicitly
        # offload before dropping the last cache reference so allocator cleanup
        # is deterministic after a caught per-segment OOM.
        try:
            predictor.to("cpu")
        except (AttributeError, RuntimeError):
            pass
    del predictors
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def clear_cotracker_cache():
    """Release resident predictors after the object-pose stage has drained."""

    offload_cotracker_cache()
    _COTRACKER_CACHE.clear()
    gc.collect()


def _cotracker_checkpoint_sha256():
    """Return the locally resolved CoTracker weight identity, if present.

    On a first-ever run Torch Hub may not have downloaded the file yet; the
    sentinel makes that first cache conservative. After download the binding
    changes once and the cache is republished with the exact weight hash.
    """

    global _COTRACKER_CHECKPOINT_DIGEST  # noqa: PLW0603

    if _COTRACKER_CHECKPOINT_DIGEST is not None:
        return _COTRACKER_CHECKPOINT_DIGEST
    path = Path(torch.hub.get_dir()) / "checkpoints/scaled_offline.pth"
    if not path.is_file() or path.is_symlink():
        return "torch_hub_checkpoint_not_yet_resolved"
    _COTRACKER_CHECKPOINT_DIGEST = _sha256_file(path)
    return _COTRACKER_CHECKPOINT_DIGEST


def select_reference_frame(masks, depth_results):
    """Select reference frame with jitter rejection, fragmentation filtering,
    depth-coverage gating, and edge penalty.

    Returns:
        int: frame index
    """
    T = len(masks)

    areas = np.zeros(T, dtype=np.float64)
    coverages = np.zeros(T, dtype=np.float64)
    for t in range(T):
        mask = masks[t]
        mask_area = mask.sum()
        areas[t] = mask_area
        if mask_area < 100:
            continue
        depth_mask = depth_results[t]["mask"]
        if depth_mask.shape != mask.shape:
            depth_mask = cv2.resize(
                depth_mask,
                (mask.shape[1], mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        coverages[t] = ((depth_mask > 0) & (mask > 0)).sum() / max(mask_area, 1)

    hard_ok = (areas >= 100) & (coverages >= 0.3)

    # Rolling median smoothing (window=15)
    half_w = 7
    smoothed = np.array([np.median(areas[max(0, t - half_w) : min(T, t + half_w + 1)]) for t in range(T)])

    jitter_ok = (smoothed == 0) | (areas <= 1.5 * smoothed)

    # Reject fragmented masks and compute solidity
    frag_ok = np.ones(T, dtype=bool)
    solidities = np.zeros(T, dtype=np.float64)
    for t in range(T):
        if areas[t] < 100:
            frag_ok[t] = False
            continue
        mask_u8 = (masks[t] > 0).astype(np.uint8)
        n_labels, _ = cv2.connectedComponents(mask_u8)
        if n_labels > 3:  # bg + >2 foreground components
            frag_ok[t] = False
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            h_area = cv2.contourArea(cv2.convexHull(largest))
            solidities[t] = cv2.contourArea(largest) / h_area if h_area > 0 else 1.0
        else:
            solidities[t] = 1.0

    valid_solidities = solidities[areas >= 100]
    median_solidity = np.median(valid_solidities) if len(valid_solidities) > 0 else 1.0
    shape_ok = solidities >= 0.8 * median_solidity

    # Edge penalty — penalize first/last 15%
    edge_zone = max(1, int(T * 0.15))
    dist_from_edge = np.minimum(np.arange(T), np.arange(T - 1, -1, -1))
    edge_bonus = np.where(dist_from_edge < edge_zone, 0.5 + 0.5 * (dist_from_edge / edge_zone), 1.0)

    candidate_mask = hard_ok & jitter_ok & frag_ok & shape_ok
    n_rejected_jitter = int((~jitter_ok).sum())
    n_rejected_frag = int((~frag_ok & (areas >= 100)).sum())
    n_rejected_shape = int((~shape_ok & (areas >= 100)).sum())

    if candidate_mask.any():
        scores = areas * edge_bonus
        scores[~candidate_mask] = -1
        best_idx = int(np.argmax(scores))
    elif hard_ok.any():
        scores = smoothed * edge_bonus
        scores[~hard_ok] = -1
        best_idx = int(np.argmax(scores))
        print("  Ref frame: all candidates rejected by jitter/frag/shape, using smoothed fallback")
    else:
        best_idx = int(np.argmax(areas))
        print(f"  WARNING: no frames pass hard gates, using max-area frame {best_idx}")

    if n_rejected_jitter > 0 or n_rejected_frag > 0 or n_rejected_shape > 0:
        print(f"  Ref frame selection: rejected {n_rejected_jitter} jitter, {n_rejected_frag} fragmented, {n_rejected_shape} shape-irregular frames")
    print(
        f"  Selected reference frame {best_idx}/{T} "
        f"(area={areas[best_idx]:.0f}, coverage={coverages[best_idx]:.2f}, "
        f"solidity={solidities[best_idx]:.3f}, edge_bonus={edge_bonus[best_idx]:.2f})"
    )
    return best_idx


def _seed_reference_points(masks, ref_frame_idx, grid_size=30):
    """Seed query points inside the reference-frame mask.

    Returns:
        seed_points_xy: (N, 2) float32
        boundary_dists: (N,) float32
    """
    from scipy.ndimage import distance_transform_edt

    _, H, W = masks.shape
    ref_mask = masks[ref_frame_idx] > 0
    ref_mask_original = ref_mask.copy()

    # Erode mask to avoid seeding near boundary
    ys_raw, xs_raw = np.where(ref_mask)
    if len(ys_raw) < 10:
        raise ValueError(f"Reference frame {ref_frame_idx} has only {len(ys_raw)} mask pixels")
    bbox_diag = np.sqrt((ys_raw.max() - ys_raw.min()) ** 2 + (xs_raw.max() - xs_raw.min()) ** 2)
    erode_px = max(3, min(int(bbox_diag * 0.03), 15))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
    ref_mask_eroded = cv2.erode(ref_mask.astype(np.uint8), kernel).astype(bool)
    original_area = ref_mask.sum()
    eroded_area = ref_mask_eroded.sum()
    if eroded_area >= 100 and eroded_area >= original_area * 0.5:
        ref_mask = ref_mask_eroded
        print(f"  Eroded ref mask by {erode_px}px ({original_area} -> {eroded_area} pixels)")
    else:
        print(f"  Erosion would leave {eroded_area}px (<100 or <50%), using original mask")

    ys, xs = np.where(ref_mask)
    if len(ys) < 10:
        raise ValueError(f"Reference frame {ref_frame_idx} has only {len(ys)} mask pixels after erosion")

    # Grid within mask bounding box, filter to mask interior
    grid_y = np.linspace(ys.min(), ys.max(), grid_size)
    grid_x = np.linspace(xs.min(), xs.max(), grid_size)
    gx, gy = np.meshgrid(grid_x, grid_y)
    candidates_x, candidates_y = gx.ravel(), gy.ravel()

    iy = np.clip(np.round(candidates_y).astype(int), 0, H - 1)
    ix = np.clip(np.round(candidates_x).astype(int), 0, W - 1)
    inside = ref_mask[iy, ix]

    if inside.sum() < 10:
        raise ObjectPoseTerminalReject(f"object_pose_insufficient_mask_support: only {int(inside.sum())} valid grid points in mask")

    seed_points = np.column_stack([candidates_x[inside], candidates_y[inside]]).astype(np.float32)

    # Boundary distance from original non-eroded mask
    dist_map = distance_transform_edt(ref_mask_original.astype(np.float32))
    vy = np.clip(np.round(seed_points[:, 1]).astype(int), 0, H - 1)
    vx = np.clip(np.round(seed_points[:, 0]).astype(int), 0, W - 1)
    boundary_dists = dist_map[vy, vx].astype(np.float32)

    return seed_points, boundary_dists


def _run_cotracker_predictor(predictor, frames, queries_np, device, backward_tracking=True):
    """Run CoTracker on a video and explicit queries."""
    queries = torch.tensor(queries_np, dtype=torch.float32, device=device).unsqueeze(0)
    video = torch.tensor(frames, dtype=torch.float32, device=device).permute(0, 3, 1, 2).unsqueeze(0)
    with torch.inference_mode():
        tracks, visibility = predictor(video, queries=queries, backward_tracking=backward_tracking)

    if tracks is None:
        raise ValueError("CoTracker returned no tracks")

    tracks_np = tracks[0].detach().cpu().numpy().astype(np.float32)
    vis_np = visibility[0].detach().cpu().numpy().astype(np.bool_)

    del video, queries, tracks, visibility
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return tracks_np, vis_np


def _cotracker_temporal_indices(n_frames, ref_frame_idx, max_frames=COTRACKER_MAX_TEMPORAL_FRAMES):
    """Bound CoTracker memory while retaining endpoints and the exact query frame."""

    if n_frames < 1 or not 0 <= ref_frame_idx < n_frames:
        raise ValueError(f"invalid CoTracker temporal roster: frames={n_frames}, ref={ref_frame_idx}")
    if max_frames < 3:
        raise ValueError("CoTracker temporal cap must be at least 3")
    if n_frames <= max_frames:
        return np.arange(n_frames, dtype=np.int64)
    base = np.rint(np.linspace(0, n_frames - 1, max_frames - 1, dtype=np.float64)).astype(np.int64)
    return np.unique(np.concatenate([base, np.array([ref_frame_idx])]))


def _expand_cotracker_temporal_result(tracks, visibility, source_mode, crop_sides, sample_indices, n_frames):
    """Interpolate sampled tracks and use nearest-neighbor discrete evidence."""

    sample_indices = np.asarray(sample_indices, dtype=np.int64)
    if (
        sample_indices.ndim != 1
        or len(sample_indices) != len(tracks)
        or len(visibility) != len(tracks)
        or sample_indices[0] != 0
        or sample_indices[-1] != n_frames - 1
        or np.any(np.diff(sample_indices) <= 0)
    ):
        raise ValueError("invalid sampled CoTracker result roster")
    target = np.arange(n_frames, dtype=np.int64)
    right = np.searchsorted(sample_indices, target, side="left")
    right = np.clip(right, 0, len(sample_indices) - 1)
    left = np.maximum(right - 1, 0)
    denominator = sample_indices[right] - sample_indices[left]
    alpha = np.divide(
        target - sample_indices[left],
        denominator,
        out=np.zeros(n_frames, dtype=np.float32),
        where=denominator != 0,
    ).astype(np.float32)
    tracks_full = tracks[left] * (1.0 - alpha[:, None, None]) + tracks[right] * alpha[:, None, None]
    nearest = np.where(
        target - sample_indices[left] <= sample_indices[right] - target,
        left,
        right,
    )
    visibility_full = visibility[nearest]
    source_mode_full = source_mode[nearest]
    crop_sides_full = crop_sides[left] * (1.0 - alpha) + crop_sides[right] * alpha
    return tracks_full, visibility_full, source_mode_full, crop_sides_full


def _filter_tracks_by_min_visibility(tracks_np, vis_np, boundary_dists, min_visible_frames=3):
    """Keep tracks visible in enough frames for downstream pose estimation."""
    vis_counts = vis_np.sum(axis=0)
    keep = vis_counts >= min_visible_frames
    if keep.sum() < 10:
        raise ObjectPoseTerminalReject(f"object_pose_insufficient_track_support: only {int(keep.sum())} tracks visible in >={min_visible_frames} frames")
    return tracks_np[:, keep], vis_np[:, keep], boundary_dists[keep]


def _interpolate_invalid_mask_stats(centers, sides, valid_mask):
    """Fill missing crop centers/sizes from nearby valid frames."""
    T = len(valid_mask)
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        raise ValueError("Cannot build crop transforms without any valid masks")
    if len(valid_idx) == 1:
        centers[:] = centers[valid_idx[0]]
        sides[:] = sides[valid_idx[0]]
        return centers, sides

    t = np.arange(T)
    centers[:, 0] = np.interp(t, valid_idx, centers[valid_idx, 0])
    centers[:, 1] = np.interp(t, valid_idx, centers[valid_idx, 1])
    sides[:] = np.interp(t, valid_idx, sides[valid_idx])
    return centers, sides


def _compute_crop_transforms(masks, out_size=512, margin=3.0, min_side=224, max_side=None, smooth_window=9):
    """Build per-frame crop transforms centered on the object mask."""
    T, H, W = masks.shape
    if max_side is None:
        max_side = max(H, W)

    centers = np.zeros((T, 2), dtype=np.float32)
    sides = np.zeros(T, dtype=np.float32)
    valid = np.zeros(T, dtype=bool)

    for t in range(T):
        ys, xs = np.where(masks[t] > 0)
        if len(xs) == 0:
            continue
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        centers[t] = [(x0 + x1) / 2.0, (y0 + y1) / 2.0]
        sides[t] = np.clip(max(x1 - x0 + 1, y1 - y0 + 1) * margin, min_side, max_side)
        valid[t] = True

    centers, sides = _interpolate_invalid_mask_stats(centers, sides, valid)

    pad = smooth_window // 2
    centers_s = centers.copy()
    sides_s = sides.copy()
    for t in range(T):
        lo, hi = max(0, t - pad), min(T, t + pad + 1)
        centers_s[t] = centers[lo:hi].mean(axis=0)
        sides_s[t] = sides[lo:hi].max()

    transforms = np.zeros((T, 2, 3), dtype=np.float32)
    inv_transforms = np.zeros((T, 2, 3), dtype=np.float32)
    for t in range(T):
        cx, cy = centers_s[t]
        scale = out_size / max(sides_s[t], 1.0)
        transform = np.array(
            [
                [scale, 0.0, out_size / 2.0 - scale * cx],
                [0.0, scale, out_size / 2.0 - scale * cy],
            ],
            dtype=np.float32,
        )
        transforms[t] = transform
        inv_transforms[t] = cv2.invertAffineTransform(transform)

    return transforms, inv_transforms, sides_s.astype(np.float32)


def _warp_frames(frames, transforms, out_size):
    """Warp RGB frames to a stabilized crop space."""
    T = len(frames)
    warped = np.empty((T, out_size, out_size, 3), dtype=frames.dtype)
    for t in range(T):
        warped[t] = cv2.warpAffine(
            frames[t],
            transforms[t],
            (out_size, out_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
    return warped


def _run_cotracker_crop_stabilized(
    predictor,
    frames,
    masks,
    ref_frame_idx,
    seed_points,
    device,
    out_size=512,
    margin=3.0,
    min_side=224,
    smooth_window=9,
):
    """Track on a stabilized crop, then map tracks back to full-frame pixels."""
    transforms, inv_transforms, crop_sides = _compute_crop_transforms(
        masks,
        out_size=out_size,
        margin=margin,
        min_side=min_side,
        max_side=max(frames.shape[1], frames.shape[2]),
        smooth_window=smooth_window,
    )
    warped_frames = _warp_frames(frames, transforms, out_size)
    seed_points_crop = cv2.transform(seed_points[None], transforms[ref_frame_idx])[0]
    queries_np = np.column_stack(
        [
            np.full(len(seed_points), ref_frame_idx, dtype=np.float32),
            seed_points_crop.astype(np.float32),
        ]
    )
    tracks_crop, vis_np = _run_cotracker_predictor(predictor, warped_frames, queries_np, device, backward_tracking=True)

    tracks_full = np.empty_like(tracks_crop)
    for t in range(len(tracks_crop)):
        tracks_full[t] = cv2.transform(tracks_crop[t][None], inv_transforms[t])[0]
    return tracks_full, vis_np, crop_sides


def _select_hybrid_switch(full_vis, crop_vis):
    """Choose a single crop->full switch that maximizes visible points."""
    full_counts = full_vis.sum(axis=1).astype(np.int64)
    crop_counts = crop_vis.sum(axis=1).astype(np.int64)
    crop_prefix = np.cumsum(crop_counts)
    full_suffix = np.cumsum(full_counts[::-1])[::-1]

    # Vectorized: total[s] = crop_prefix[s] + full_suffix[s+1]
    totals = crop_prefix.copy()
    totals[:-1] += full_suffix[1:]

    all_full_total = int(full_counts.sum())
    best_idx = int(np.argmax(totals))
    best_switch = best_idx if totals[best_idx] > all_full_total else -1

    return best_switch, full_counts, crop_counts


def run_cotracker(frames, masks, ref_frame_idx, grid_size=30, tracking_mode="hybrid"):
    """Run CoTracker3 on the object and return full-frame pixel tracks.

    Modes:
        full_bidir: full-frame bidirectional tracking
        crop_bidir: crop-stabilized bidirectional tracking, mapped back to full frame
        hybrid: crop-stabilized early, full-frame later, single switch chosen automatically
    """
    original_n_frames = len(frames)
    original_ref_frame_idx = int(ref_frame_idx)
    temporal_indices = _cotracker_temporal_indices(original_n_frames, original_ref_frame_idx)
    temporally_sampled = len(temporal_indices) < original_n_frames
    if temporally_sampled:
        frames = frames[temporal_indices]
        masks = masks[temporal_indices]
        ref_frame_idx = int(np.flatnonzero(temporal_indices == original_ref_frame_idx)[0])
        print(f"  CoTracker temporal cap: {original_n_frames} -> {len(temporal_indices)} frames (reference {original_ref_frame_idx} -> {ref_frame_idx})")

    seed_points, boundary_dists = _seed_reference_points(masks, ref_frame_idx, grid_size=grid_size)
    queries_np = np.column_stack(
        [
            np.full(len(seed_points), ref_frame_idx, dtype=np.float32),
            seed_points.astype(np.float32),
        ]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = get_cotracker_predictor(device)

    mode = tracking_mode.lower()
    if mode == "full_bidir":
        tracks_np, vis_np = _run_cotracker_predictor(predictor, frames, queries_np, device, backward_tracking=True)
        switch_frame = -1
        source_mode = np.zeros(len(frames), dtype=np.uint8)
        crop_sides = np.zeros(len(frames), dtype=np.float32)
    elif mode == "crop_bidir":
        tracks_np, vis_np, crop_sides = _run_cotracker_crop_stabilized(
            predictor,
            frames,
            masks,
            ref_frame_idx,
            seed_points,
            device,
        )
        switch_frame = len(frames) - 1
        source_mode = np.ones(len(frames), dtype=np.uint8)
    elif mode == "hybrid":
        full_tracks, full_vis = _run_cotracker_predictor(predictor, frames, queries_np, device, backward_tracking=True)
        crop_tracks, crop_vis, crop_sides = _run_cotracker_crop_stabilized(
            predictor,
            frames,
            masks,
            ref_frame_idx,
            seed_points,
            device,
        )
        switch_frame, full_counts, crop_counts = _select_hybrid_switch(full_vis, crop_vis)
        tracks_np = full_tracks.copy()
        vis_np = full_vis.copy()
        source_mode = np.zeros(len(frames), dtype=np.uint8)
        if switch_frame >= 0:
            tracks_np[: switch_frame + 1] = crop_tracks[: switch_frame + 1]
            vis_np[: switch_frame + 1] = crop_vis[: switch_frame + 1]
            source_mode[: switch_frame + 1] = 1
        print(f"  Hybrid switch frame: {switch_frame} (full mean={full_counts.mean():.1f}, crop mean={crop_counts.mean():.1f})")
        del full_tracks, full_vis, crop_tracks, crop_vis
    else:
        raise ValueError(f"Unknown tracking mode '{tracking_mode}'. Expected one of: full_bidir, crop_bidir, hybrid")

    del predictor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tracks_np, vis_np, boundary_dists = _filter_tracks_by_min_visibility(tracks_np, vis_np, boundary_dists)

    if temporally_sampled:
        tracks_np, vis_np, source_mode, crop_sides = _expand_cotracker_temporal_result(
            tracks_np,
            vis_np,
            source_mode,
            crop_sides,
            temporal_indices,
            original_n_frames,
        )
        if switch_frame >= 0:
            switch_frame = int(temporal_indices[switch_frame])
        ref_frame_idx = original_ref_frame_idx

    print(f"  CoTracker ({mode}): {tracks_np.shape[1]} tracks, ref frame {ref_frame_idx}")
    return {
        "tracks": tracks_np.astype(np.float32),
        "visibility": vis_np.astype(np.bool_),
        "ref_frame_idx": np.int32(ref_frame_idx),
        "boundary_dists": boundary_dists.astype(np.float32),
        "tracking_mode": np.array(mode),
        "switch_frame": np.int32(switch_frame),
        "source_mode": source_mode.astype(np.uint8),
        "crop_window_sides": crop_sides.astype(np.float32),
        "tracking_sample_count": np.int32(len(temporal_indices)),
        "tracking_original_frame_count": np.int32(original_n_frames),
        "pipeline_version": np.int32(OBJECT_POSE_PIPELINE_VERSION),
    }
