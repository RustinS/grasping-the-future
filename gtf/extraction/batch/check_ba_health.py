#!/usr/bin/env python3
"""Bundle-adjustment health check of a batch: aborts a run whose object-pose solves exploded.

Reads the median reprojection error over the evidence frames of every segment's object_pose/object_poses.npz (the
bundle adjustment stores NaN for frames without evidence). Exits 1 when the exploded fraction (evidence median >
--exploded-px, default 30 px, or unreadable) exceeds --max-exploded-frac. run_split_extraction.sh runs it after the
heavy GPU stages and does not mark a failing run as successful (a broken environment can make most solves explode
without any error).

Usage: python -m gtf.extraction.batch.check_ba_health --seg-file LIST [--max-exploded-frac 0.4]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from gtf.paths import REPO_ROOT

RESULTS = Path(os.environ.get("ACTION100M_RESULTS_DIR") or REPO_ROOT / "results")


def evidence_median(seg_dir):
    """Median reprojection error (px) over the evidence frames of one segment.

    None without BA output or evidence; inf when the archive cannot be read; NaN (counted as exploded) when the BA kept
    no evidence frame although tracks.npz has a frame with at least four visible tracks."""
    op = seg_dir / "object_pose"
    p = op / "object_poses.npz"
    if not p.exists():
        return None
    try:
        re = np.load(p)["reprojection_errors"]
    except Exception:
        return float("inf")
    if np.isfinite(re).any():
        return float(np.nanmedian(re))
    try:
        nvis = np.load(op / "tracks.npz")["visibility"].sum(1)
    except Exception:
        return None
    return float("nan") if len(re) >= len(nvis) and (nvis >= 4).any() else None


def main():
    ap = argparse.ArgumentParser(prog="python -m gtf.extraction.batch.check_ba_health")
    ap.add_argument("--seg-file", required=True)
    ap.add_argument("--max-exploded-frac", type=float, default=0.4)
    ap.add_argument("--exploded-px", type=float, default=30.0)
    ap.add_argument("--min-solved", type=int, default=10, help="below this many solved segments the check passes trivially")
    args = ap.parse_args()

    segs = [s.strip() for s in Path(args.seg_file).read_text().splitlines() if s.strip() and not s.lstrip().startswith("#")]
    meds = []
    for s in segs:
        m = evidence_median(RESULTS / s)
        if m is not None:
            meds.append((s, m))
    if len(meds) < args.min_solved:
        print(f"[ba-health] only {len(meds)} solved segments (<{args.min_solved}); check skipped")
        return 0
    exploded = [(s, m) for s, m in meds if not np.isfinite(m) or m > args.exploded_px]
    frac = len(exploded) / len(meds)
    med_all = float(np.median([m for _, m in meds if np.isfinite(m)])) if any(np.isfinite(m) for _, m in meds) else float("nan")
    print(f"[ba-health] {len(meds)} solved | median evidence reproj {med_all:.1f}px | exploded (> {args.exploded_px}px): {len(exploded)} ({100 * frac:.0f}%)")
    for s, m in exploded[:10]:
        print(f"[ba-health]   exploded: {s} ({m:.0f}px)")
    if frac > args.max_exploded_frac:
        print(f"[ba-health] FAIL: exploded fraction {100 * frac:.0f}% > {100 * args.max_exploded_frac:.0f}%: the BA stage is broken; stopping before the optimizer")
        return 1
    print("[ba-health] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
