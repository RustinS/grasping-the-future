"""Which frames SAM 3D runs on: usable masks, keyframe selection, anchor candidates."""

import cv2
import numpy as np
from gtf.utils import rprint as print


MIN_MASK_PIXELS = 100


def _sam3d_mask_geometry_usable(mask):
    """Match SAM3D's strict bounding-box precondition without loading the model."""

    y, x = np.nonzero(mask)
    return bool(y.size >= MIN_MASK_PIXELS and int(y.max()) - int(y.min()) >= 2 and int(x.max()) - int(x.min()) >= 2)


def _usable_keyframe_indices(masks, total_frames=None):
    mask_frames = int(masks.shape[0])
    upper = mask_frames if total_frames is None else min(mask_frames, int(total_frames))
    usable = []
    below_pixels = 0
    bad_bbox = 0
    for index in range(upper):
        mask = np.asarray(masks[index])
        pixels = int(np.count_nonzero(mask))
        if pixels < MIN_MASK_PIXELS:
            below_pixels += 1
        elif _sam3d_mask_geometry_usable(mask):
            usable.append(index)
        else:
            bad_bbox += 1
    return np.asarray(usable, dtype=np.int64), {
        "mask_frames_total": mask_frames,
        "video_frames_at_selection": (mask_frames if total_frames is None else int(total_frames)),
        "mask_frames_outside_video": max(0, mask_frames - upper),
        "mask_frames_below_min_pixels": below_pixels,
        "mask_frames_bad_sam3d_bbox": bad_bbox,
    }


def select_keyframes(
    masks,
    video_fps,
    select_fps=1.0,
    *,
    total_frames=None,
    return_stats=False,
):
    """Select target-rate frames whose masks satisfy SAM3D input geometry."""

    valid, stats = _usable_keyframe_indices(masks, total_frames=total_frames)
    T = min(int(masks.shape[0]), int(total_frames or masks.shape[0]))

    if len(valid) == 0:
        selected = []
        return (selected, stats) if return_stats else selected

    step = max(1, round(video_fps / select_fps))
    num_frames = max(1, T // step)

    if len(valid) <= num_frames:
        selected = valid.tolist()
        return (selected, stats) if return_stats else selected

    indices = np.linspace(0, len(valid) - 1, num_frames, dtype=int)
    selected = sorted(set(valid[indices].tolist()))
    return (selected, stats) if return_stats else selected


def rank_anchor_candidates(masks, keyframe_indices):
    """Rank keyframes best-first for anchor selection (jitter, fragmentation, solidity, edge penalty)."""
    T = masks.shape[0]
    areas = masks.reshape(T, -1).astype(np.float64).sum(axis=1)

    half_w = 7
    smoothed = np.array([np.median(areas[max(0, t - half_w) : t + half_w + 1]) for t in range(T)])
    jitter_ok = ~((smoothed > 0) & (areas > 1.5 * smoothed))

    frag_ok = {}
    solidities = {}
    for t in keyframe_indices:
        if areas[t] < MIN_MASK_PIXELS:
            frag_ok[t] = False
            solidities[t] = 0.0
            continue
        mask_u8 = (masks[t] > 0).astype(np.uint8)
        n_labels, _ = cv2.connectedComponents(mask_u8)
        frag_ok[t] = n_labels <= 3  # bg + <=2 foreground components
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            c_area = cv2.contourArea(largest)
            hull = cv2.convexHull(largest)
            h_area = cv2.contourArea(hull)
            solidities[t] = c_area / h_area if h_area > 0 else 1.0
        else:
            solidities[t] = 1.0

    valid_sols = [solidities[t] for t in keyframe_indices if areas[t] >= MIN_MASK_PIXELS]
    median_solidity = np.median(valid_sols) if valid_sols else 1.0
    shape_ok = {t: solidities.get(t, 0) >= 0.8 * median_solidity for t in keyframe_indices}

    edge_zone = max(1, int(T * 0.15))
    dist = np.minimum(np.arange(T), np.arange(T - 1, -1, -1))
    edge_bonus = np.where(dist < edge_zone, 0.5 + 0.5 * (dist / edge_zone), 1.0)

    scored = []
    n_rejected_jitter = 0
    n_rejected_frag = 0
    n_rejected_shape = 0
    for t in keyframe_indices:
        passes = True
        if areas[t] < MIN_MASK_PIXELS:
            passes = False
        if not jitter_ok[t]:
            n_rejected_jitter += 1
            passes = False
        if not frag_ok.get(t, True):
            n_rejected_frag += 1
            passes = False
        if not shape_ok.get(t, True):
            n_rejected_shape += 1
            passes = False
        score = areas[t] * edge_bonus[t] if passes else -1.0
        scored.append((t, score, passes))

    if n_rejected_jitter or n_rejected_frag or n_rejected_shape:
        print(f"  Anchor ranking: rejected {n_rejected_jitter} jitter, {n_rejected_frag} fragmented, {n_rejected_shape} shape-irregular")

    passing = [(t, s) for t, s, p in scored if p]
    rejected = [(t, areas[t]) for t, s, p in scored if not p]
    passing.sort(key=lambda x: x[1], reverse=True)
    rejected.sort(key=lambda x: x[1], reverse=True)

    result = [t for t, _ in passing] + [t for t, _ in rejected]

    if result:
        best = result[0]
        print(f"  Anchor candidate: frame {best} (area={areas[best]:.0f}, solidity={solidities.get(best, 0):.3f}, edge_bonus={edge_bonus[best]:.2f})")

    return result
