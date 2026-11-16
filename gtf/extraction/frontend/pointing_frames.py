#!/usr/bin/env python3
"""Pointing-frame selection and point-set validation for Molmo2 -> SAM3.

A single fixed pointing frame (the middle of the longest validated action range, else the
video midpoint) can have no hand, be motion-blurred or show the object occluded. This module
provides two CPU-only, model-free pieces:

  select_candidate_frames - rank probe frames by (hand present at the frame, inside the
      trust region around the default frame, sharpness x in-valid-range bonus), then
      temporal NMS down to n_candidates well-separated moments. Hand presence comes from
      the WiLoR output (hands/{side}.npz frame_names, kept sides only); the hand stage
      always runs before pointing. The default frame is always a candidate, so the
      selection never does worse than pointing on that frame alone.

  filter_points / accept_points - validate a collected point set: drop cluster outliers
      (likely another object instance or the background), then check that the cluster
      medoid lies near a projected hand joint. HAND_PROX_FRAC and the outlier radius
      max(0.12 diag, 3x median) sit just above the medoid-to-hand distances and cluster
      spreads of segments that pass the quality gates.

Acceptance: >= 2 validated points, or 1 validated point whose hand-proximity check
actually passed (not just skipped); a failed hand-proximity check rejects the candidate.
"""

from pathlib import Path

import cv2
import numpy as np

N_PROBE = 14
N_CANDIDATES = 3
HAND_FRAME_TOL = 2  # a probe counts hand-present if a kept-side hand frame is within +/- this
IN_RANGE_BONUS = 1.15  # mild sharpness multiplier for probes inside the validated action range
TRUST_FRAC = 0.2  # temporal trust region around the default mid-action frame (fraction of T)
HAND_PROX_FRAC = 0.40  # medoid -> nearest projected hand joint, fraction of image diagonal
CLUSTER_BASE_FRAC = 0.12  # outlier radius floor, fraction of image diagonal
CLUSTER_MED_MULT = 3.0  # ... or 3x the median medoid distance, whichever is larger


def hand_present_frames(seg_dir):
    """Frame indices with a kept-side tracked hand (WiLoR output in hands/)."""
    frames = set()
    for side in ("left", "right"):
        p = Path(seg_dir) / "hands" / f"{side}.npz"
        if not p.exists():
            continue
        try:
            frames.update(int(f) for f in np.load(p)["frame_names"])
        except Exception:
            continue
    return frames


def hand_joints_2d(seg_dir, frame_indices):
    """{frame: (N,2) projected hand-joint pixels} for the requested frames, both kept sides.

    A frame missing its exact tracked entry borrows the nearest one within
    +/- HAND_FRAME_TOL (the hand-presence tolerance of candidate selection): a hand
    barely moves in 2 frames, and a skipped proximity check on a hand-present frame
    would turn a valid 1-point acceptance into a rejection. Frames with no tracked hand
    nearby are absent from the result. Empty dict when the packed hand data is missing
    (the proximity check is then skipped).
    """
    from gtf.utils import load_hand_data

    want = {int(f) for f in frame_indices}
    out = {}
    try:
        hands = load_hand_data(str(seg_dir))
    except Exception:
        return out
    for h in hands.values():
        for f in want:
            idx = None
            for d in sorted(range(-HAND_FRAME_TOL, HAND_FRAME_TOL + 1), key=abs):
                idx = h["frame_to_idx"].get(f + d)
                if idx is not None:
                    break
            if idx is None:
                continue
            J = h["joints"][idx]
            K = h["intr"][idx] if h["intr"].ndim == 3 else h["intr"]
            z = np.clip(J[:, 2], 1e-6, None)
            uv = np.stack([K[0, 0] * J[:, 0] / z + K[0, 2], K[1, 1] * J[:, 1] / z + K[1, 2]], axis=1)
            out[f] = np.concatenate([out[f], uv]) if f in out else uv
    return out


def frame_sharpness(frame_rgb):
    """Laplacian variance on a half-res grayscale: the motion-blur score."""
    h, w = frame_rgb.shape[:2]
    small = cv2.resize(frame_rgb, (max(1, w // 2), max(1, h // 2)))
    return float(cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())


def _in_range(frame_idx, valid_range, fps):
    if not valid_range or not fps:
        return False
    t = frame_idx / fps
    return valid_range.get("start", 0.0) <= t <= valid_range.get("end", float("inf"))


def select_candidate_frames(seg_dir, video_path=None, valid_range=None, fps=None, default_idx=None, n_candidates=N_CANDIDATES, n_probe=N_PROBE):
    """Rank pointing-frame candidates for one segment.

    default_idx is the default frame (middle of the longest validated action range).
    Returns a list of dicts {frame, hand_present, in_trust, sharpness, in_range}, best
    first, temporally NMS'd so retries see different moments. Ordering is
    (hand_present, in_trust, sharpness x range-bonus): the trust region around the
    default frame keeps the temporal prior (a sharp hand frame far from it can show the
    primary object already set down, binding pointing to a different in-hand object),
    while far hand-present frames still beat near handless ones (interactions that
    happen away from the middle). Never empty for a readable video; falls back to
    sharpness-only ranking when no hand data exists.
    """
    from decord import VideoReader, cpu

    seg_dir = Path(seg_dir)
    video_path = Path(video_path) if video_path else seg_dir / "video.mp4"
    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=2)
    T = len(vr)
    fps = fps or vr.get_avg_fps()

    probes = set(np.linspace(0, T - 1, min(n_probe, T)).astype(int).tolist())
    if default_idx is not None:
        probes.add(int(min(max(default_idx, 0), T - 1)))
    hand_frames = hand_present_frames(seg_dir)
    trust_center = int(min(max(default_idx, 0), T - 1)) if default_idx is not None else T // 2
    trust_radius = max(TRUST_FRAC * T, 15)

    rows = []
    for p in sorted(probes):
        f = vr[p]
        arr = f.numpy() if hasattr(f, "numpy") else f.asnumpy()
        hp = any((p + d) in hand_frames for d in range(-HAND_FRAME_TOL, HAND_FRAME_TOL + 1))
        rows.append(
            {
                "frame": int(p),
                "hand_present": bool(hp),
                "in_trust": bool(abs(p - trust_center) <= trust_radius),
                "sharpness": round(frame_sharpness(arr), 1),
                "in_range": bool(_in_range(p, valid_range, fps)),
            }
        )
    del vr

    ranked = sorted(rows, key=lambda r: (r["hand_present"], r["in_trust"], r["sharpness"] * (IN_RANGE_BONUS if r["in_range"] else 1.0)), reverse=True)
    gap = max(4, T // 8)
    out = []
    for r in ranked:
        if len(out) >= n_candidates:
            break
        if all(abs(r["frame"] - o["frame"]) >= gap for o in out):
            out.append(r)
    for r in ranked:  # backfill if NMS starved the list (short videos)
        if len(out) >= n_candidates:
            break
        if r not in out:
            out.append(r)
    # the default frame is always a candidate (last, when outranked): a moment can be
    # the only pointable one (object actually in hand there) even when it ranks below
    # the top-N on hand/trust/sharpness
    if default_idx is not None and not any(c["frame"] == int(min(max(default_idx, 0), T - 1)) for c in out):
        li = int(min(max(default_idx, 0), T - 1))
        out.append(next(r for r in rows if r["frame"] == li))
    return out


def filter_points(points_px, img_wh, hand_pts=None):
    """Validate a point set: cluster-outlier filter + medoid hand-proximity check.

    points_px: sequence of (x, y) pixels. hand_pts: (N,2) projected hand joints at
    the same frame, or None when no hand is tracked there (the check is then skipped,
    not failed). Returns (keep_indices, info) where info carries the decision evidence.
    """
    pts = np.asarray(list(points_px), dtype=float).reshape(-1, 2)
    W, H = img_wh
    diag = float(np.hypot(W, H))
    info = {"n_raw": int(len(pts)), "n_valid": 0, "spread_frac": None, "hand_dist_frac": None, "hand_prox": None}
    if len(pts) == 0:
        return [], info

    dmat = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    med_i = int(dmat.sum(axis=1).argmin())
    d_med = dmat[med_i]
    med_dist = float(np.median(d_med[d_med > 0])) if len(pts) > 1 else 0.0
    radius = max(CLUSTER_BASE_FRAC * diag, CLUSTER_MED_MULT * med_dist)
    keep = np.where(d_med <= radius)[0]
    info["n_valid"] = int(len(keep))
    info["spread_frac"] = round(med_dist / diag, 4)

    if hand_pts is not None and len(hand_pts):
        hd = float(np.min(np.linalg.norm(np.asarray(hand_pts, float) - pts[med_i], axis=1)))
        info["hand_dist_frac"] = round(hd / diag, 4)
        info["hand_prox"] = bool(hd <= HAND_PROX_FRAC * diag)
    return keep.tolist(), info


def accept_points(info):
    """Acceptance rule for one candidate's validated point set.

    A failed hand-proximity check rejects (points not near any tracked hand suggest the
    wrong instance). Otherwise accept >=2 validated points, or a single point whose
    proximity check passed.
    """
    if info["hand_prox"] is False:
        return False
    if info["n_valid"] >= 2:
        return True
    return info["n_valid"] == 1 and info["hand_prox"] is True
