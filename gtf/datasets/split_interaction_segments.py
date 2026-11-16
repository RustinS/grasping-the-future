#!/usr/bin/env python3
"""Split ground-truth segments into interaction-window child segments.

For each parent segment, ``interaction_runs.interaction_windows`` finds the disjoint
windows where a hand touches or moves the object (extended backwards to include the
reach). Unless the detection covers the whole segment, each window becomes a child:

  results/{parent}__iw{k}/
    canonical/canonical.npz     # per-frame arrays sliced [start:end+1]
    canonical/metadata.json     # parent metadata + window provenance + n_frames
    video.mp4 -> ../{parent}/video.mp4  (symlink; frame_indices index the parent video)

The parent stays on disk with metadata ``interaction_children``; build_manifest skips it
so no frame is counted twice. A parent without any interaction gets
``interaction_none`` and is skipped as well. ``gravity_direction`` is recomputed at
each child's middle frame (sources with camera extrinsics).

Sources listed in ``ROW_STRIDE`` are then thinned to the training cadence: every
trainable segment (child or unsplit parent) keeps every n-th row. ``fps`` stays the video
rate and ``frame_indices`` keep the video frame numbers, so the effective cadence
``fps / median frame step`` is the decimated one.

Parents that already carry ``interaction_children``, ``interaction_none`` or
``fps_decimated_from`` are skipped, so the script can be re-run.

Usage:
  python -m gtf.datasets.split_interaction_segments --datasets taco oakink2 show3d ho3d h2o hocap
  python -m gtf.datasets.split_interaction_segments --datasets oakink2 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

from gtf.datasets.interaction_runs import interaction_windows, is_full_span
from gtf.utils import RESULTS_DIR
from gtf.utils.canonical import validate_canonical_npz

# Per-dataset detector overrides. SHOW3D hands are fits to tracked landmarks with
# 15-30 mm error, so grasping frames measure 3-5 cm from the object surface.
DETECTOR_OVERRIDES: dict[str, dict] = {
    "show3d": {"contact_m": 0.05},
}

# Per-dataset row stride of the exported segments. SHOW3D is recorded at 60 Hz; every
# second row is kept so it trains at ~30 Hz like the other sources.
ROW_STRIDE: dict[str, int] = {"show3d": 2}

# Per-frame keys (first dim == T). Everything else is copied verbatim.
_SLICE_KEYS_BASE = [
    "frame_indices",
    "source_frame_indices",
    "camera_extrinsics",
    "obj_rotation",
    "obj_translation",
    "obj_rotation_world",
    "obj_translation_world",
    "obj_verts",
]
_SLICE_KEYS_HAND = ["orient", "pose", "cam_t", "valid", "verts", "orient_world", "cam_t_world", "fit_mean_error_mm", "fit_p90_error_mm"]
_SLICE_KEYS = set(_SLICE_KEYS_BASE) | {f"hand_{side}_{k}" for side in ("left", "right") for k in _SLICE_KEYS_HAND}


def _take_rows(data: dict, rows: np.ndarray) -> dict:
    T = data["obj_rotation"].shape[0]
    out = {}
    for k, v in data.items():
        if k in _SLICE_KEYS:
            if v.shape[0] != T:
                raise ValueError(f"{k}: expected first dim {T}, got {v.shape}")
            out[k] = v[rows].copy()
        else:
            out[k] = v.copy()
    return out


def _slice_npz(data: dict, s: int, e: int) -> dict:
    out = _take_rows(data, np.arange(s, e + 1))
    # gravity_direction is defined at the segment's middle-frame camera.
    if "gravity_world" in out and "camera_extrinsics" in out:
        E = out["camera_extrinsics"]
        Em = E if E.ndim == 2 else E[E.shape[0] // 2]
        g = Em[:3, :3].astype(np.float64) @ out["gravity_world"].astype(np.float64)
        out["gravity_direction"] = (g / np.linalg.norm(g)).astype(np.float32)
    return out


def _decimate(data: dict, meta: dict, stride: int) -> tuple[dict, dict]:
    """Keep every ``stride``-th row; ``gravity_direction`` is kept from the full-rate segment."""
    T = data["obj_rotation"].shape[0]
    out = _take_rows(data, np.arange(0, T, stride))
    meta = dict(meta)
    meta["fps_decimated_from"] = float(data["fps"])
    meta["n_frames"] = int(out["obj_rotation"].shape[0])
    for key in ("interaction_window", "interaction_core"):
        if key in meta:
            meta[key] = [int(x) // stride for x in meta[key]]
    return out, meta


def _write_canonical(cdir: Path, data: dict, meta: dict) -> None:
    cdir.mkdir(parents=True, exist_ok=True)
    tmp = cdir / "canonical.tmp.npz"
    np.savez_compressed(str(tmp), **data)
    os.replace(tmp, cdir / "canonical.npz")
    ok, errors = validate_canonical_npz(str(cdir / "canonical.npz"), full_scan=False)
    hard = [x for x in errors if not str(x).startswith("WARNING")]
    if not ok and hard:
        raise RuntimeError(f"{cdir.parent.name} failed schema validation: {hard[:2]}")
    (cdir / "metadata.json").write_text(json.dumps(meta, indent=2))


def split_segment(args_tuple):
    seg_dir, results_dir, dry_run = args_tuple
    try:
        seg = Path(seg_dir)
        results = Path(results_dir)
        meta_path = seg / "canonical" / "metadata.json"
        meta = json.loads(meta_path.read_text())
        if meta.get("interaction_children") or meta.get("interaction_none") or meta.get("fps_decimated_from"):
            return {"segment": seg.name, "status": "skip_done"}

        with np.load(seg / "canonical" / "canonical.npz", allow_pickle=False) as d:
            data = {k: d[k] for k in d.files}
        T = data["obj_rotation"].shape[0]
        source = str(meta.get("source_dataset", ""))
        fps = float(data.get("fps", meta.get("fps", 30.0)))
        stride = int(ROW_STRIDE.get(source, 1))
        wins = interaction_windows(data, fps, **DETECTOR_OVERRIDES.get(source, {}))
        if not wins:
            # The object is never touched and never moves: no training value.
            if not dry_run:
                meta["interaction_none"] = True
                meta_path.write_text(json.dumps(meta, indent=2))
            return {"segment": seg.name, "status": "no_interaction", "T": T}
        if is_full_span(wins, T):
            if not dry_run and stride > 1:
                data, meta = _decimate(data, meta, stride)
                _write_canonical(seg / "canonical", data, meta)
            return {"segment": seg.name, "status": "full_span", "T": T}
        if dry_run:
            return {"segment": seg.name, "status": "would_split", "T": T, "windows": [[w.start, w.end] for w in wins], "kept": sum(w.n_frames for w in wins)}

        children = []
        try:
            for k, w in enumerate(wins):
                child_id = f"{seg.name}__iw{k}"
                cmeta = dict(meta)
                cmeta.update(
                    segment_id=child_id,
                    n_frames=int(w.n_frames),
                    parent_segment_id=seg.name,
                    interaction_window=[int(w.start), int(w.end)],
                    interaction_core=[int(w.core_start), int(w.core_end)],
                    interaction_contact_frac=round(w.contact_frac, 4),
                    interaction_moving_frac=round(w.moving_frac, 4),
                )
                sliced = _slice_npz(data, w.start, w.end)
                if stride > 1:
                    sliced, cmeta = _decimate(sliced, cmeta, stride)
                _write_canonical(results / child_id / "canonical", sliced, cmeta)
                vid = seg / "video.mp4"
                link = results / child_id / "video.mp4"
                if vid.exists() and not link.exists():
                    os.symlink(os.path.relpath(vid, link.parent), link)
                children.append(child_id)
        except Exception:
            for c in children + [f"{seg.name}__iw{len(children)}"]:
                shutil.rmtree(results / c, ignore_errors=True)
            raise

        meta["interaction_children"] = children
        meta["interaction_windows"] = [[int(w.start), int(w.end)] for w in wins]
        meta_path.write_text(json.dumps(meta, indent=2))
        return {"segment": seg.name, "status": "split", "T": T, "n_children": len(children), "kept": sum(w.n_frames for w in wins)}
    except Exception as exc:
        return {"segment": Path(seg_dir).name, "status": f"EXCEPTION: {exc}", "trace": traceback.format_exc()[-500:]}


def main():
    ap = argparse.ArgumentParser(prog="python -m gtf.datasets.split_interaction_segments", description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="*", required=True)
    ap.add_argument("--results-root", default=str(RESULTS_DIR))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--segments", nargs="*", default=None, help="Explicit parent segment directories")
    args = ap.parse_args()

    results = Path(args.results_root)
    if args.segments:
        dirs = args.segments
    else:
        dirs = []
        for ds in args.datasets:
            dirs += sorted(str(p) for p in results.glob(f"{ds}_*") if "__iw" not in p.name and (p / "canonical" / "canonical.npz").exists())
    if args.limit:
        dirs = dirs[: args.limit]
    print(f"scanning {len(dirs)} parents (dry_run={args.dry_run})", flush=True)

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = []
    with ctx.Pool(args.workers) as pool:
        for n, r in enumerate(pool.imap_unordered(split_segment, [(d, str(results), args.dry_run) for d in dirs], chunksize=4)):
            out.append(r)
            if r["status"].startswith("EXCEPTION"):
                print(f"  !! {r['segment']}: {r['status']}", flush=True)
            if (n + 1) % 300 == 0:
                print(f"  [{n + 1}/{len(dirs)}]", flush=True)

    print(Counter(r["status"][:22] for r in out))
    split = [r for r in out if r["status"] in ("split", "would_split")]
    if split:
        kept = sum(r["kept"] for r in split)
        total = sum(r["T"] for r in split)
        nch = sum(r.get("n_children", len(r.get("windows", []))) for r in split)
        print(f"split parents: {len(split)}  children: {nch}  frames kept {kept}/{total} ({kept / total * 100:.0f}%) on split parents")
    if any(r["status"].startswith("EXCEPTION") for r in out):
        sys.exit(1)


if __name__ == "__main__":
    main()
