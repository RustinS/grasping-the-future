#!/usr/bin/env python3
"""Estimate gravity from the scene caches of segments that have none, and write it back.

HO-3D (and H2O-3D) canonicals live in a static camera frame without a gravity direction.
Their scene caches, built first without gravity, hold camera-frame scene clouds. Per
segment, a RANSAC dominant-plane fit runs on up to 8 valid frames; "up" is the plane
normal signed toward the camera (cameras look down at the support), a plane whose normal
does not point into the scene is rejected as a wall, and gravity is the normalized mean
of the per-frame estimates. A segment is updated only when its per-frame estimates agree
(median spread <= 3 deg); the others keep has_gravity=false.

An updated segment gets ``gravity_direction`` in canonical.npz, ``has_gravity`` and
``gravity_source`` in metadata.json, and its scene cache is deleted: the scene stage then
rebuilds it gravity-aligned. Run it between two passes of gtf/datasets/scene/preprocess.py.

Segments listed in ``ho3d_scene_gravity.json`` are skipped: the HO-3D loader writes their
gravity from that table.

Usage:
    python -m gtf.datasets.estimate_scene_gravity [--manifest datasets/canonical_manifest.json] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

from gtf.paths import REPO_ROOT
from gtf.datasets.ho3d import recorded_scene_gravity

ROOT = REPO_ROOT

# Sources whose loaders write no gravity.
SOURCES = {"ho3d", "h2o3d"}
N_FRAMES_PROBE = 8
MAX_SPREAD_DEG = 3.0


def fit_plane_gravity(P: np.ndarray, ref: np.ndarray, rng: np.random.RandomState, n_ransac: int = 250, thresh: float = 0.02) -> tuple[np.ndarray, float] | None:
    """RANSAC dominant plane of one cloud; returns (gravity unit vector, inlier fraction) or None."""
    P = P[np.isfinite(P).all(axis=1)]
    if P.shape[0] < 200:
        return None
    best = (0, None, None)
    N = P.shape[0]
    for _ in range(n_ransac):
        idx = rng.choice(N, 3, replace=False)
        p0, p1, p2 = P[idx]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        if nn < 1e-8:
            continue
        n = n / nn
        d = -np.dot(n, p0)
        k = int((np.abs(P @ n + d) < thresh).sum())
        if k > best[0]:
            best = (k, n, d)
    k, n, d = best
    if n is None or k < 0.15 * N:
        return None
    # Least-squares refit on the inliers.
    Q = P[np.abs(P @ n + d) < thresh]
    c = Q.mean(0)
    _, _, Vt = np.linalg.svd(Q - c, full_matrices=False)
    n = Vt[-1] / np.linalg.norm(Vt[-1])
    # The normal points from the plane toward the camera center (the cloud origin).
    if np.dot(ref - c, n) < 0:
        n = -n
    g = -n
    # A support plane's gravity points into the scene content; a wall's is horizontal.
    mean_dir = P.mean(0) - ref
    mean_dir = mean_dir / (np.linalg.norm(mean_dir) + 1e-9)
    if float(np.dot(g, mean_dir)) < 0.15:
        return None
    return g.astype(np.float64), float(k / N)


def segment_gravity(scene_cache: Path, n_canonical_frames: int) -> dict:
    """Consensus camera-frame gravity of one segment from its scene cache."""
    out: dict = {}
    if not scene_cache.exists():
        out["error"] = "no scene cache"
        return out
    estimates = []
    rng = np.random.RandomState(0)
    with h5py.File(scene_cache, "r") as h5:
        nf = h5["P_scene"].shape[0]
        valid = np.asarray(h5["valid"])
        probe = np.linspace(0, min(n_canonical_frames, nf) - 1, N_FRAMES_PROBE * 2).round().astype(int)
        for i in [i for i in probe if valid[i]][:N_FRAMES_PROBE]:
            r = fit_plane_gravity(np.asarray(h5["P_scene"][i], dtype=np.float64), np.zeros(3, dtype=np.float64), rng)
            if r is not None:
                estimates.append(r[0])
    if len(estimates) < 3:
        out["error"] = f"too few frame estimates ({len(estimates)})"
        return out
    vs = np.stack(estimates)
    m = vs.mean(0)
    m = m / np.linalg.norm(m)
    out["g_cam"] = m.round(4).tolist()
    out["spread_med_deg"] = float(np.median(np.degrees(np.arccos(np.clip(vs @ m, -1, 1)))))
    return out


def update_segment(entry: dict, dry_run: bool) -> str:
    if str(entry["segment_id"]).split("__iw")[0] in recorded_scene_gravity():
        return "recorded"
    canonical_path = Path(entry["canonical_path"])
    with np.load(canonical_path, allow_pickle=False) as d:
        data = {k: d[k] for k in d.files}
    if "gravity_direction" in data:
        return "has_gravity"
    scene_cache = canonical_path.parent / "scene_cache_v2.h5"
    r = segment_gravity(scene_cache, int(data["obj_rotation"].shape[0]))
    if "error" in r:
        return f"skip: {r['error']}"
    if r["spread_med_deg"] > MAX_SPREAD_DEG:
        return f"skip: spread {r['spread_med_deg']:.1f} deg"
    if dry_run:
        return "would_update"
    g = np.asarray(r["g_cam"], dtype=np.float64)
    data["gravity_direction"] = (g / np.linalg.norm(g)).astype(np.float32)
    tmp = canonical_path.with_name("canonical.tmp.npz")
    np.savez_compressed(str(tmp), **data)
    os.replace(tmp, canonical_path)
    meta_path = canonical_path.parent / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["has_gravity"] = True
    meta["gravity_source"] = "scene_plane_consensus"
    meta_path.write_text(json.dumps(meta, indent=2))
    scene_cache.unlink()
    return "updated"


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m gtf.datasets.estimate_scene_gravity", description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default=str(ROOT / "datasets" / "canonical_manifest.json"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    entries = json.loads(Path(args.manifest).read_text())["entries"]
    entries = [e for e in entries if e.get("validated", True) and str(e.get("source_dataset", "")).lower() in SOURCES]
    counts: dict[str, int] = {}
    for entry in entries:
        status = update_segment(entry, args.dry_run)
        key = status.split(":")[0]
        counts[key] = counts.get(key, 0) + 1
        if status not in ("has_gravity", "recorded"):
            print(f"  {entry['segment_id']}: {status}")
    print(f"{len(entries)} segments: {counts}")


if __name__ == "__main__":
    main()
