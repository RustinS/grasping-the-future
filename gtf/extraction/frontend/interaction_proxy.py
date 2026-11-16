#!/usr/bin/env python3
"""Post-SAM3 interaction proxy: an optional early exit before the expensive GPU stages.

Right after SAM3 (object masks and hands available) the pipeline can already tell, for a
conservative subset of segments, that the interaction / interaction_sustain gates can never
pass: the tracked hand never comes near the object mask. Skipping those segments saves the
sam3d + object_pose + optimize stages.

Proxies (2D, a few seconds per segment on CPU):
    prox_rate  - fraction of hand-tracked frames where a projected hand vertex (either hand)
                 lands within D px (5% of the image width) of the dilated SAM3 object mask
    prox_run_s - longest contiguous such run, in seconds (the analog of interaction_sustain)

A segment is rejected only when both proxies are below their thresholds. Enabled with
`python -m gtf.extraction.action100m.pipeline --early-exit` (off by default).
"""

from pathlib import Path

import cv2
import numpy as np

FRAME_STRIDE = 2
VERT_STRIDE = 25
PROX_FRAC_W = 0.05

# Each threshold is 0.7x the lowest value observed on segments that pass all quality gates, so the early exit
# does not drop a segment the gates would keep.
PROX_RATE_MIN = 0.268
PROX_RUN_S_MIN = 1.26


def segment_interaction_proxies(seg_dir):
    """Compute {prox_rate, prox_run_s, n_hand_frames} for one segment, or None if inputs missing."""
    from gtf.utils import load_hand_data

    seg_dir = Path(seg_dir)
    mask_path = seg_dir / "SAM3" / "masks.npz"
    if not mask_path.exists():
        return None
    try:
        masks = np.load(mask_path)["masks"]
    except Exception:
        return None
    T, H, W = masks.shape
    hands = load_hand_data(str(seg_dir))
    if not hands:
        return None
    D = max(3, int(round(PROX_FRAC_W * W)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * D + 1, 2 * D + 1))
    frames = sorted({f for h in hands.values() for f in h["frame_to_idx"]})
    frames = [f for f in frames if 0 <= f < T][::FRAME_STRIDE]
    if not frames:
        return None
    near = []
    for f in frames:
        m = masks[f].astype(np.uint8)
        dm = cv2.dilate(m, kernel) if m.any() else None
        hit = False
        if dm is not None:
            for h in hands.values():
                idx = h["frame_to_idx"].get(f)
                if idx is None:
                    continue
                v = h["verts"][idx][::VERT_STRIDE]
                K = h["intr"][idx] if h["intr"].ndim == 3 else h["intr"]
                z = np.clip(v[:, 2], 1e-6, None)
                u = np.clip((K[0, 0] * v[:, 0] / z + K[0, 2]).astype(int), 0, W - 1)
                w_ = np.clip((K[1, 1] * v[:, 1] / z + K[1, 2]).astype(int), 0, H - 1)
                if dm[w_, u].any():
                    hit = True
                    break
        near.append(hit)
    near = np.asarray(near, bool)
    best = cur = 0
    for x in near:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return {
        "prox_rate": float(near.mean()),
        "prox_run_s": best * FRAME_STRIDE / 30.0,
        "n_hand_frames": int(len(frames)),
    }


def should_early_exit(seg_dir):
    """Return (reject: bool, proxies: dict|None). Conservative: never rejects on missing inputs."""
    p = segment_interaction_proxies(seg_dir)
    if p is None or p["n_hand_frames"] < 5:
        return False, p
    reject = p["prox_rate"] < PROX_RATE_MIN and p["prox_run_s"] < PROX_RUN_S_MIN
    return reject, p
