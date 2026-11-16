"""What the image says about the hands, as used by the upgraded chain's optimizer (optimize_mano.load_segment).

All functions are pure numpy / OpenCV on WiLoR's per-frame hand (camera-space vertices + its intrinsics) and the video's
masks (SAM 3 object mask, SAM 3 "hand" mask from run_hand_masks.py):

    image_gap           how far WiLoR's hand is from the object mask in the image (0 = touching)   -> contact evidence
    contact_weight      per-keyframe gate in [0, 1] from those gaps
    comotion_weight     per-keyframe-pair gate: does the hand move with the object in the image?   -> carry evidence
    mask_veto           drop WiLoR detections with no RGB hand under them
    image_extent_pct    how large the hand is in the image
    size_scene_scales   hand depth scale: from its apparent size, with the scene depth per frame where the hand is small
"""

import cv2
import numpy as np


def project_hand(verts, K, img_size, mask_shape):
    """WiLoR camera-space verts -> pixel coords at mask resolution (N,2) and an in-image flag."""
    P = np.asarray(verts, np.float64)
    K = np.asarray(K, np.float64)
    if (P[:, 2] <= 1e-6).any():
        return None, None
    u = P[:, :2] / P[:, 2:3] @ K[:2, :2].T + K[:2, 2]
    Hm, Wm = mask_shape
    W_i, H_i = float(img_size[0]), float(img_size[1])
    u = u * np.array([Wm / W_i, Hm / H_i])
    ok = (u[:, 0] >= 0) & (u[:, 0] <= Wm - 1) & (u[:, 1] >= 0) & (u[:, 1] <= Hm - 1)
    return u, ok


def image_gap(verts, K, img_size, obj_mask):
    """Image evidence for hand-object contact: pixel distance from WiLoR's projected hand to the SAM 3 object mask, in
    units of the hand's projected size (0 = touching / overlapping). None when there is no usable evidence."""
    if obj_mask is None or int(np.count_nonzero(obj_mask)) < 30:
        return None
    u, ok = project_hand(verts, K, img_size, obj_mask.shape)
    if u is None or int(ok.sum()) < 50:
        return None
    dt = cv2.distanceTransform((~np.asarray(obj_mask, bool)).astype(np.uint8), cv2.DIST_L2, 5)
    ui = np.round(u[ok]).astype(np.int64)
    gap = float(dt[ui[:, 1], ui[:, 0]].min())
    size = float(np.linalg.norm(u.max(0) - u.min(0)))
    return gap / max(size, 1.0)


def contact_weight(gaps, touch=0.02, release=0.15):
    """Per-keyframe gate in [0, 1] from the image gaps (None = no evidence -> 1): 1 up to `touch`, 0 beyond `release`,
    linear between, then widened by one keyframe (max over t-1..t+1) so a single-frame mask miss does not drop contact."""
    w = np.array([1.0 if g is None else float(np.clip((release - g) / (release - touch), 0.0, 1.0)) for g in gaps], np.float64)
    if len(w) > 1:
        w = np.maximum.reduce([w, np.r_[w[1:], w[-1]], np.r_[w[0], w[:-1]]])
    return w


def comotion_weight(hand_c, obj_c, still=0.005, keep=0.4, drop=0.8):
    """Per keyframe pair (T-1,) weight in [0, 1]: does the hand move with the object in the image? hand_c / obj_c: per
    keyframe 2D centroids (fractions of the image width; None = no evidence -> 1). rel = |d_obj - d_hand| / max(|d_obj|,
    |d_hand|, still): 1 up to `keep`, 0 beyond `drop`, then a 3-pair median (mask centroids are noisy)."""
    T = len(hand_c)
    w = np.ones(max(T - 1, 0))
    for t in range(T - 1):
        if any(x is None for x in (hand_c[t], hand_c[t + 1], obj_c[t], obj_c[t + 1])):
            continue
        dh = np.asarray(hand_c[t + 1]) - np.asarray(hand_c[t])
        do = np.asarray(obj_c[t + 1]) - np.asarray(obj_c[t])
        rel = float(np.linalg.norm(do - dh) / max(np.linalg.norm(do), np.linalg.norm(dh), still))
        w[t] = float(np.clip((drop - rel) / (drop - keep), 0.0, 1.0))
    if len(w) >= 3:
        w = np.array([np.median(w[max(0, t - 1) : t + 2]) for t in range(len(w))])
    return w


def mask_veto(hand_data, hand_masks, min_precision=0.15, min_clip_hand_frac=0.5, dilate_frac=0.02):
    """Drop WiLoR detections with no RGB hand under them (hand_masks: {frame: (H,W) bool}, SAM 3 "hand"). A detection is
    vetoed when < min_precision of its in-image projected vertices land on the (dilated) hand mask. Applied only when
    SAM 3 sees hands in >= min_clip_hand_frac of the masked keyframes (a clip where SAM 3 misses hands, e.g. gloves,
    vetoes nothing) and only on frames where SAM 3 found a hand somewhere (a frame-level miss vetoes nothing).
    Mutates hand_data[side]["valid_frames"]; returns {side: [vetoed frames]}."""
    if not hand_masks:
        return {}
    has = [bool(m.any()) for m in hand_masks.values()]
    if np.mean(has) < min_clip_hand_frac:
        return {}
    dil = {}
    vetoed = {}
    for side, hd in hand_data.items():
        out = []
        keep = set(hd["valid_frames"])
        for f in sorted(keep):
            m = hand_masks.get(int(f))
            if m is None or f not in hd["frame_to_idx"] or not m.any():  # no RGB hand anywhere in this frame: no evidence
                continue
            ai = hd["frame_to_idx"][f]
            u, ok = project_hand(hd["verts"][ai], hd["intr"][ai], hd["img_size"][ai], m.shape)
            if u is None or int(ok.sum()) < 100:
                continue
            if f not in dil:
                r = max(1, int(round(dilate_frac * m.shape[1])))
                dil[f] = cv2.dilate(m.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
            ui = np.round(u[ok]).astype(np.int64)
            if float(dil[f][ui[:, 1], ui[:, 0]].mean()) < min_precision:
                out.append(int(f))
        if out:
            hd["valid_frames"] = keep - set(out)
        vetoed[side] = out
    return vetoed


def image_extent_pct(verts, K_h, img_size_h):
    """Largest image extent (x or y) of WiLoR's hand mesh, as % of the image width; None if behind the camera."""
    v = np.asarray(verts, dtype=np.float64)
    if v.ndim != 2 or (v[:, 2] <= 1e-6).any():
        return None
    u = v[:, 0] / v[:, 2] * float(K_h[0, 0]) + float(K_h[0, 2])
    w = v[:, 1] / v[:, 2] * float(K_h[1, 1]) + float(K_h[1, 2])
    return float(max(np.ptp(u), np.ptp(w)) / max(float(img_size_h[0]), 1.0) * 100.0)


def size_scene_scales(size_scales, scene_scales, hand_extent, extent_range):
    """hand_depth_source="size_scene": per-frame depth scale = size^w * (k * scene)^(1-w) (see optimize_mano.load_segment).

    size_scales / scene_scales: {frame: scale or None} (the "size" and "moge" hand depth scales); hand_extent: per-frame hand
    extent in the image (% of width, None allowed). Returns ({frame: scale or None}, info string). w = 1 keeps "size"
    exactly; with fewer than 5 frames carrying both scales the size scales are returned unchanged.
    """
    lo, hi = float(extent_range[0]), float(extent_range[1])
    ext = [e for e in hand_extent if e is not None and np.isfinite(e)]
    both = [f for f, s in size_scales.items() if s is not None and scene_scales.get(f) is not None and s > 0 and scene_scales[f] > 0]
    if not ext or len(both) < 5:
        return dict(size_scales), f"kept size ({len(both)} frames with scene depth, {len(ext)} with a hand extent)"
    med = float(np.median(ext))
    w = float(np.clip((med - lo) / max(hi - lo, 1e-6), 0.0, 1.0))
    if w >= 1.0:
        return dict(size_scales), f"hand {med:.1f}% of the width >= {hi:g}: size depth kept"
    k = float(np.exp(np.median([np.log(size_scales[f] / scene_scales[f]) for f in both])))
    out = {}
    for f, s in size_scales.items():
        sc = scene_scales.get(f)
        if s is None or sc is None or sc <= 0:
            out[f] = None  # no scene depth at this hand in this frame: interpolated by smooth_scales, as in "moge"
            continue
        out[f] = float(np.exp(w * np.log(s) + (1.0 - w) * np.log(k * sc)))
    return out, f"hand {med:.1f}% of the width: size weight {w:.2f}, scene depth x {k:.3f} (median over {len(both)} frames)"
