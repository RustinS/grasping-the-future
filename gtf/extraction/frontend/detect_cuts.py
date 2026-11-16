#!/usr/bin/env python3
"""Shot-boundary (scene-cut) detection for extraction segments.

In-the-wild Action100M clips are edited videos that often cut every few seconds. A cut
inside a segment breaks everything downstream of it (hands jump to a new person or camera,
object tracks die) while every per-frame extraction stage still succeeds. Physics and
silhouette gates cannot see a cut; the video can. The pipeline uses this detector in the
`cutcheck` step (which truncates the segment at its first cut and harvests later shots as
`{seg}__shotN` child segments) and in the scene_cut quality gate.

Detector: per consecutive-frame pair, HS-histogram correlation + changed-pixel fraction
on downscaled gray. A cut needs both a histogram break and a large changed area: fast
pans move pixels but shift the histogram slowly, while a cut breaks both at once.

Usage:
    python -m gtf.extraction.frontend.detect_cuts --video results/SEG/video.mp4
    python -m gtf.extraction.frontend.detect_cuts --segment SEG_NAME
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


CORR_HARD = 0.55  # histogram correlation below this alone = cut
CORR_SOFT = 0.80  # below this AND most pixels changed = cut
CHANGED_FRAC = 0.55  # fraction of pixels with |gray diff| > 30
MIN_GAP = 5  # merge cut candidates closer than this many frames


def detect_scene_cuts(video_path, corr_hard=CORR_HARD, corr_soft=CORR_SOFT, changed_frac=CHANGED_FRAC, min_gap=MIN_GAP, downscale_w=160):
    """Return (cuts, n_frames, fps): cut frame indices (each = first frame of a new shot)."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    prev_hist, prev_gray = None, None
    cuts, fi = [], 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        h = int(bgr.shape[0] * downscale_w / bgr.shape[1])
        small = cv2.resize(bgr, (downscale_w, max(h, 2)))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.int16)
        if prev_hist is not None:
            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            changed = float((np.abs(gray - prev_gray) > 30).mean())
            if corr < corr_hard or (corr < corr_soft and changed > changed_frac):
                if not cuts or fi - cuts[-1] >= min_gap:
                    cuts.append(fi)
        prev_hist, prev_gray = hist, gray
        fi += 1
    cap.release()
    return cuts, fi, fps


def main():
    ap = argparse.ArgumentParser(prog="python -m gtf.extraction.frontend.detect_cuts")
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--segment", type=str, default=None)
    args = ap.parse_args()
    from gtf.utils import RESULTS_DIR

    path = Path(args.video) if args.video else Path(RESULTS_DIR) / args.segment / "video.mp4"
    cuts, n, fps = detect_scene_cuts(path)
    print(json.dumps({"video": str(path), "n_frames": n, "fps": round(fps, 3), "cuts": cuts, "cut_times_s": [round(c / fps, 2) for c in cuts]}))


if __name__ == "__main__":
    main()
