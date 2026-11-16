#!/usr/bin/env python3
"""Convention validation for model-ready canonical.npz files.

Checks more than the schema:
  - MANO params must reconstruct stored hand vertices.
  - Object rotation/translation/scale must reconstruct stored object vertices.
  - Invalid frames must stay zero-filled.
  - Optional camera/depth conventions are sanity-checked when requested.

The dataset loaders call it after each conversion; it can also be run directly
on existing result directories.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from gtf.paths import MANO_DIR
from gtf.utils import RESULTS_DIR, rprint as print
from gtf.utils.canonical import mano_fk_verts, validate_canonical_npz

MAX_SAMPLED_OBJECT_LOAD_BYTES = 512 * 1024 * 1024
MIN_SAMPLED_OBJECT_FRAMES = 32


@dataclass
class ConventionReport:
    path: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


def _as_bool(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr).astype(bool)


def _frame_indices(valid: np.ndarray, max_frames: int | None) -> np.ndarray:
    idx = np.flatnonzero(valid)
    if max_frames is None or max_frames <= 0 or len(idx) <= max_frames:
        return idx
    # Deterministic coverage across the whole segment, not just the start.
    take = np.linspace(0, len(idx) - 1, max_frames).round().astype(np.int64)
    return idx[take]


def _sample_z(verts: np.ndarray, max_points: int = 200_000) -> np.ndarray:
    flat = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
    if flat.shape[0] <= max_points:
        return flat[:, 2].copy()
    idx = np.linspace(0, flat.shape[0] - 1, max_points).round().astype(np.int64)
    return flat[idx, 2].copy()


def _npz_array_meta(path: Path, key: str) -> tuple[tuple[int, ...], np.dtype]:
    with zipfile.ZipFile(path) as zf:
        with zf.open(f"{key}.npy") as f:
            version = np.lib.format.read_magic(f)
            shape, _, dtype = np.lib.format._read_array_header(f, version)
    return tuple(shape), np.dtype(dtype)


def _check_hand_fk(d: np.lib.npyio.NpzFile, side: str, report: ConventionReport, max_frames: int | None, fk_tol_m: float) -> None:
    valid = _as_bool(d[f"hand_{side}_valid"])
    if not valid.any():
        return

    idx = _frame_indices(valid, max_frames)
    orient = d[f"hand_{side}_orient"][idx].astype(np.float32)
    pose = d[f"hand_{side}_pose"][idx].astype(np.float32)
    cam_t = d[f"hand_{side}_cam_t"][idx].astype(np.float32)
    betas = d[f"hand_{side}_betas"].astype(np.float32)
    stored = d[f"hand_{side}_verts"][idx].astype(np.float32)

    try:
        recon = mano_fk_verts(orient, pose, betas, cam_t, side, str(MANO_DIR)).astype(np.float32)
    except Exception as e:  # pragma: no cover - diagnostic path
        report.add_error(f"{side} MANO FK failed: {e}")
        return

    err = np.linalg.norm(recon - stored, axis=-1)
    mean_err = float(err.mean())
    max_err = float(err.max())
    report.metrics[f"{side}_mano_fk_mean_m"] = mean_err
    report.metrics[f"{side}_mano_fk_max_m"] = max_err
    if max_err > fk_tol_m:
        report.add_error(f"{side} MANO FK mismatch: max={max_err:.6f}m mean={mean_err:.6f}m tol={fk_tol_m:.6f}m")


def _check_object_pose(
    path: Path,
    d: np.lib.npyio.NpzFile,
    report: ConventionReport,
    obj_tol_m: float,
    max_frames: int | None,
) -> np.ndarray | None:
    verts0 = d["obj_canonical_verts"].astype(np.float32)
    scale = d["obj_scale"].astype(np.float32).reshape(3)
    if np.any(scale <= 0) or not np.all(np.isfinite(scale)):
        report.add_error(f"obj_scale must be finite and positive, got {scale.tolist()}")
        return None

    rot = d["obj_rotation"].astype(np.float32)
    trans = d["obj_translation"].astype(np.float32)
    obj_max_frames = max_frames
    if max_frames is not None and max_frames > 0:
        obj_max_frames = max(max_frames, MIN_SAMPLED_OBJECT_FRAMES)
    frame_idx = _frame_indices(np.ones(rot.shape[0], dtype=bool), obj_max_frames)
    obj_shape, obj_dtype = _npz_array_meta(path, "obj_verts")
    obj_nbytes = int(np.prod(obj_shape, dtype=np.int64)) * obj_dtype.itemsize
    load_stored = max_frames is None or max_frames <= 0 or obj_nbytes <= MAX_SAMPLED_OBJECT_LOAD_BYTES

    scaled = verts0 * scale.reshape(1, 3)
    recon = np.empty((len(frame_idx), verts0.shape[0], 3), dtype=np.float32)
    for out_i, src_i in enumerate(frame_idx):
        R = Rotation.from_rotvec(rot[src_i]).as_matrix().astype(np.float32)
        recon[out_i] = scaled @ R.T + trans[src_i]
    obj_z = _sample_z(recon)

    report.metrics["obj_pose_checked_frames"] = float(len(frame_idx))
    report.metrics["obj_pose_total_frames"] = float(rot.shape[0])
    report.metrics["obj_verts_nbytes"] = float(obj_nbytes)
    if not load_stored:
        report.metrics["obj_pose_stored_check_skipped"] = 1.0
        return obj_z

    stored_all = d["obj_verts"].astype(np.float32)
    stored = stored_all[frame_idx]

    err = np.linalg.norm(recon - stored, axis=-1)
    mean_err = float(err.mean())
    max_err = float(err.max())
    report.metrics["obj_pose_mean_m"] = mean_err
    report.metrics["obj_pose_max_m"] = max_err
    if max_err > obj_tol_m:
        report.add_error(f"object pose mismatch: max={max_err:.6f}m mean={mean_err:.6f}m tol={obj_tol_m:.6f}m")
    return obj_z


def _check_camera_sanity(d: np.lib.npyio.NpzFile, report: ConventionReport, obj_z: np.ndarray | None, max_frames: int | None) -> None:
    """Soft checks for camera-frame canonicals.

    These catch obvious axis flips and scale errors; they are warnings so that unusual
    but valid camera placements are not rejected.
    """
    z_samples = []
    if obj_z is not None:
        z_samples.append(obj_z)
    for side in ("left", "right"):
        valid = _as_bool(d[f"hand_{side}_valid"])
        if valid.any():
            idx = _frame_indices(valid, max_frames)
            hv = d[f"hand_{side}_verts"][idx].astype(np.float32)
            z_samples.append(_sample_z(hv))
    if not z_samples:
        return

    z = np.concatenate(z_samples)
    z = z[np.isfinite(z)]
    if z.size == 0:
        report.add_error("camera sanity: no finite z coordinates")
        return
    frac_in_front = float((z > 0.02).mean())
    median_z = float(np.median(z))
    report.metrics["camera_frac_z_positive"] = frac_in_front
    report.metrics["camera_median_z_m"] = median_z
    if frac_in_front < 0.90:
        report.add_warning(f"camera sanity: only {frac_in_front:.3f} of geometry has positive OpenCV z")
    if not (0.05 <= median_z <= 8.0):
        report.add_warning(f"camera sanity: median z={median_z:.3f}m is unusual")


def _check_gravity(d: np.lib.npyio.NpzFile, report: ConventionReport) -> None:
    if "gravity_direction" not in d.files:
        return
    g = d["gravity_direction"].astype(np.float32)
    norm = float(np.linalg.norm(g))
    report.metrics["gravity_norm"] = norm
    if not np.isfinite(norm) or norm < 1e-5:
        report.add_error("gravity_direction is zero or non-finite")
    elif abs(norm - 1.0) > 0.05:
        report.add_warning(f"gravity_direction norm is {norm:.3f}, expected near 1")


def validate_canonical_conventions(
    canonical_path: str | Path,
    *,
    max_fk_frames: int | None = None,
    fk_tol_m: float = 2e-4,
    obj_tol_m: float = 2e-5,
    strict_camera: bool = False,
    schema_full_scan: bool = True,
) -> ConventionReport:
    path = Path(canonical_path)
    report = ConventionReport(path=path)

    ok, schema_errors = validate_canonical_npz(str(path), full_scan=schema_full_scan)
    for e in schema_errors:
        if e.startswith("WARNING:"):
            report.add_warning(e.removeprefix("WARNING:").strip())
        else:
            report.add_error(e)
    if not ok and report.errors:
        return report

    with np.load(str(path), allow_pickle=False) as d:
        obj_z = _check_object_pose(path, d, report, obj_tol_m=obj_tol_m, max_frames=max_fk_frames)
        for side in ("left", "right"):
            _check_hand_fk(d, side, report, max_frames=max_fk_frames, fk_tol_m=fk_tol_m)
        _check_gravity(d, report)
        if strict_camera:
            _check_camera_sanity(d, report, obj_z=obj_z, max_frames=max_fk_frames)

    return report


def _find_canonicals(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file() and p.name == "canonical.npz":
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(p.glob("*/canonical/canonical.npz")))
            if (p / "canonical.npz").exists():
                out.append(p / "canonical.npz")
    return sorted(set(out))


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m gtf.datasets.validate_conventions", description="Validate canonical convention consistency")
    parser.add_argument("paths", nargs="*", default=[str(RESULTS_DIR)], help="canonical.npz files or result roots")
    parser.add_argument("--max-fk-frames", type=int, default=0, help="0 = validate every valid frame")
    parser.add_argument("--fk-tol-m", type=float, default=2e-4)
    parser.add_argument("--obj-tol-m", type=float, default=2e-5)
    parser.add_argument("--strict-camera", action="store_true")
    parser.add_argument("--fast-schema", action="store_true", help="Use NPZ headers for schema checks instead of full-array NaN/Inf scans")
    parser.add_argument("--json", action="store_true", help="Emit JSON summary")
    args = parser.parse_args()

    paths = _find_canonicals(args.paths)
    reports = [
        validate_canonical_conventions(
            p,
            max_fk_frames=None if args.max_fk_frames <= 0 else args.max_fk_frames,
            fk_tol_m=args.fk_tol_m,
            obj_tol_m=args.obj_tol_m,
            strict_camera=args.strict_camera,
            schema_full_scan=not args.fast_schema,
        )
        for p in paths
    ]

    if args.json:
        payload = [
            {
                "path": str(r.path),
                "ok": r.ok,
                "errors": r.errors,
                "warnings": r.warnings,
                "metrics": r.metrics,
            }
            for r in reports
        ]
        print(json.dumps(payload, indent=2))
    else:
        for r in reports:
            if not r.ok:
                print(f"FAIL {r.path}: {r.errors[:2]}")
            elif r.warnings:
                print(f"WARN {r.path}: {r.warnings[:2]}")
        n_ok = sum(r.ok for r in reports)
        n_warn = sum(bool(r.warnings) for r in reports)
        print(f"Validated {n_ok}/{len(reports)} canonicals ({n_warn} with warnings)")

    if any(not r.ok for r in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
