#!/usr/bin/env python3
"""Build the training manifest from every converted segment.

Scans ``results/*/canonical/metadata.json``, validates each canonical.npz,
reads the quality-gate verdict and the scene cache when present, and writes
``datasets/canonical_manifest.json``.

Usage:
    python -m gtf.datasets.build_manifest
    python -m gtf.datasets.build_manifest --dataset hot3d
    python -m gtf.datasets.build_manifest --results-root /abs/results --output /abs/manifest.json
    python -m gtf.datasets.build_manifest --scene-cache-root DIR --output DIR/canonical_manifest.json
"""

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from gtf.paths import REPO_ROOT
from gtf.extraction.gates.interaction_metrics import interaction_fidelity_score
from gtf.utils import RESULTS_DIR, rprint as print
from gtf.utils.canonical import validate_canonical_npz

ROOT = REPO_ROOT

TIER_SCORES = {
    "gt_mocap": 0.95,
    "gt_fitted": 0.80,
    "gt_landmark_fitted": 0.65,
    "mixed": 0.75,
    "extracted": 0.50,
}

OUTPUT_PATH = ROOT / "datasets" / "canonical_manifest.json"

HDF5_SCENE_CACHE_REQUIRED_KEYS = {
    "P_scene",
    "scene_normals",
    "scene_geo_feats",
    "valid",
    "support_plane_y",
    "R_world_from_cam",
    "anchor_canonical_idx",
    "anchor_video_idx",
    "n_points_actual",
}

# A scene cache needs at least this fraction of valid frames. Per-frame builds skip
# close-up frames whose masks leave too few background points; the dataset treats an
# invalid frame as scene-free.
SCENE_CACHE_MIN_COVERAGE = 0.8


def _scene_cache_hdf5_status(path: Path, expected_frames: int | None = None) -> tuple[bool, bool, str | None, int]:
    """Validate a scene_cache_v2.h5: datasets and shapes, frame coverage, anchors, finiteness.

    Finiteness of the point payload is spot-checked at five evenly spaced valid frames.
    Returns (ok, has_semantic, error, valid_frame_count).
    """
    if not path.exists():
        return False, False, None, 0
    try:
        import h5py

        with h5py.File(str(path), "r") as h5:
            missing = sorted(HDF5_SCENE_CACHE_REQUIRED_KEYS - set(h5.keys()))
            if missing:
                return False, False, f"scene_cache_v2.h5 missing datasets: {missing[:3]}", 0
            p_shape = tuple(h5["P_scene"].shape)
            if len(p_shape) != 3 or p_shape[2] != 3:
                return False, False, f"scene_cache_v2.h5 bad P_scene shape: {p_shape}", 0
            n_frames, n_points = int(p_shape[0]), int(p_shape[1])
            if tuple(h5["scene_normals"].shape) != (n_frames, n_points, 3):
                return False, False, f"scene_cache_v2.h5 bad scene_normals shape: {tuple(h5['scene_normals'].shape)}", 0
            if tuple(h5["scene_geo_feats"].shape) != (n_frames, n_points, 12):
                return False, False, f"scene_cache_v2.h5 bad scene_geo_feats shape: {tuple(h5['scene_geo_feats'].shape)}", 0
            if tuple(h5["valid"].shape) != (n_frames,):
                return False, False, f"scene_cache_v2.h5 bad valid shape: {tuple(h5['valid'].shape)}", 0
            if tuple(h5["R_world_from_cam"].shape) != (n_frames, 3, 3):
                return False, False, f"scene_cache_v2.h5 bad R_world_from_cam shape: {tuple(h5['R_world_from_cam'].shape)}", 0
            if expected_frames is not None and n_frames != int(expected_frames):
                return False, False, f"scene_cache_v2.h5 stale frame axis: {n_frames}/{int(expected_frames)}", 0

            valid = np.asarray(h5["valid"][:], dtype=bool)
            valid_count = int(valid.sum())
            if expected_frames is not None and valid_count < max(3, int(np.ceil(SCENE_CACHE_MIN_COVERAGE * int(expected_frames)))):
                return False, False, f"scene_cache_v2.h5 incomplete: {valid_count}/{int(expected_frames)} frames", valid_count
            if expected_frames is None and valid_count <= 0:
                return False, False, "scene_cache_v2.h5 contains no valid frames", 0

            valid_idx = np.flatnonzero(valid)
            if valid_idx.size:
                anchors = np.asarray(h5["anchor_canonical_idx"][:]).reshape(-1)
                bad = valid_idx[anchors[valid_idx] != valid_idx]
                if bad.size:
                    i = int(bad[0])
                    return False, False, f"scene_cache_v2.h5 anchor mismatch at frame {i}: anchor_canonical_idx={int(anchors[i])} ({bad.size} bad frames)", valid_count
                rw = np.asarray(h5["R_world_from_cam"][:])[valid_idx]
                if not np.isfinite(rw).all():
                    return False, False, "scene_cache_v2.h5 non-finite R_world_from_cam on a valid frame", valid_count
                probe_idx = np.unique(valid_idx[np.round(np.linspace(0, valid_idx.size - 1, min(5, valid_idx.size))).astype(int)])
                for i in probe_idx:
                    for key in ("P_scene", "scene_normals", "scene_geo_feats"):
                        if not np.isfinite(np.asarray(h5[key][int(i)])).all():
                            return False, False, f"scene_cache_v2.h5 non-finite {key} at valid frame {int(i)}", valid_count

            has_semantic = bool(h5.attrs.get("has_semantic", "scene_sem_feats" in h5))
            if has_semantic:
                if "scene_sem_feats" not in h5:
                    return False, False, "scene_cache_v2.h5 has_semantic=true but scene_sem_feats missing", valid_count
                sem_shape = tuple(h5["scene_sem_feats"].shape)
                if len(sem_shape) != 3 or sem_shape[0] != n_frames or sem_shape[1] != n_points:
                    return False, False, f"scene_cache_v2.h5 bad scene_sem_feats shape: {sem_shape}", valid_count
                if valid_idx.size and not np.isfinite(np.asarray(h5["scene_sem_feats"][int(valid_idx[0])])).all():
                    return False, False, f"scene_cache_v2.h5 non-finite scene_sem_feats at valid frame {int(valid_idx[0])}", valid_count
        return True, has_semantic, None, valid_count
    except Exception as e:
        return False, False, f"scene_cache_v2.h5 unreadable: {type(e).__name__}: {e}", 0


def build_manifest(args):
    results_dir = Path(args.results_root or RESULTS_DIR)
    if not results_dir.exists():
        print(f"ERROR: {results_dir} does not exist")
        return

    metadata_files = sorted(results_dir.glob("*/canonical/metadata.json"))
    print(f"Found {len(metadata_files)} segments with canonical/metadata.json")

    entries = []
    dataset_counts = Counter()
    validation_counts = {"passed": 0, "failed": 0, "warnings": 0}
    gate_rejected = 0
    n_superseded = 0

    for meta_path in metadata_files:
        seg_dir = meta_path.parent.parent
        segment_name = seg_dir.name
        canonical_dir = meta_path.parent
        npz_path = canonical_dir / "canonical.npz"

        with open(meta_path) as f:
            metadata = json.load(f)

        source_dataset = metadata.get("source_dataset", "unknown")
        if args.dataset and source_dataset != args.dataset:
            continue

        # A parent split into interaction windows (split_interaction_segments.py) is
        # replaced by its __iwN children, and a segment without interaction is dropped.
        if metadata.get("interaction_children") or metadata.get("interaction_none"):
            n_superseded += 1
            continue

        entry = dict(metadata)
        entry["segment_id"] = metadata.get("segment_id", segment_name)
        entry["canonical_path"] = str(npz_path)
        entry["quality_score"] = TIER_SCORES.get(metadata.get("confidence_tier", "extracted"), 0.50)

        # Quality gate (extracted segments only): a failed segment is excluded; a passing one
        # gets an interaction-fidelity score in [0.7, 1] that scales its quality score and
        # feeds the dataset's extractor-confidence channel. Ground-truth segments have no
        # quality.json and keep their tier score.
        qpath = seg_dir / "quality" / "quality.json"
        if qpath.exists():
            with open(qpath) as f:
                q = json.load(f)
            entry["gate_pass"] = bool(q.get("pass"))
            entry["failed_gates"] = q.get("failed_gates") or []
            if entry["gate_pass"] is False and not args.include_gate_rejects:
                gate_rejected += 1
                continue
            ifs = interaction_fidelity_score(
                contact_cover=q.get("contact_cover"),
                rot_jerk=q.get("rot_jerk"),
                hand_drift=q.get("hand_drift"),
                obj_centroid=q.get("obj_centroid"),
                motion2d_corr=q.get("motion2d_corr"),
                area_ratio_drift=q.get("area_ratio_drift"),
            )
            entry["interaction_fidelity_score"] = round(ifs, 4)
            entry["tier_score"] = entry["quality_score"]
            entry["quality_score"] = round(entry["quality_score"] * ifs, 4)

        errors = []
        n_frames = None
        dynamic_present_anchor = False

        if not npz_path.exists():
            errors.append("canonical.npz not found")
        else:
            is_valid, schema_errors = validate_canonical_npz(str(npz_path), full_scan=not args.fast_schema)
            if not is_valid:
                errors.extend(schema_errors)

            d = np.load(str(npz_path))
            npz_keys = set(d.files)

            if metadata.get("has_gravity", False) and "gravity_direction" not in npz_keys:
                errors.append("has_gravity=true but gravity_direction missing from .npz")
            if not metadata.get("has_gravity", False) and "gravity_direction" in npz_keys:
                errors.append("WARNING: has_gravity=false but gravity_direction present in .npz")
                validation_counts["warnings"] += 1

            if metadata.get("has_support_plane", False) and "support_plane_height" not in npz_keys:
                errors.append("has_support_plane=true but support_plane_height missing from .npz")
            if not metadata.get("has_support_plane", False) and "support_plane_height" in npz_keys:
                errors.append("WARNING: has_support_plane=false but support_plane_height present in .npz")
                validation_counts["warnings"] += 1

            if "frame_indices" in npz_keys:
                n_frames = int(d["frame_indices"].shape[0])
                entry["n_frames"] = n_frames
            # A moving camera re-anchors every frame to the present camera, which needs a
            # scene row per frame.
            if "camera_extrinsics" in npz_keys and n_frames is not None:
                cam = d["camera_extrinsics"]
                if cam.ndim == 3 and cam.shape[0] == n_frames and cam.shape[0] > 1:
                    dynamic_present_anchor = bool(np.max(np.abs(cam - cam[:1])) > 1e-5)

        # `has_scene` means the segment video exists; the scene cache is the model-ready
        # point-cloud artifact built from it.
        video_path = seg_dir / "video.mp4"
        video_exists = video_path.exists()
        metadata_has_scene = bool(metadata.get("has_scene", False))
        if metadata_has_scene and not video_exists:
            errors.append("has_scene=true but video.mp4 not found in segment dir")
        if video_exists and not metadata_has_scene:
            errors.append("WARNING: has_scene=false but video.mp4 exists; manifest will mark has_scene=true")
            validation_counts["warnings"] += 1
        entry["has_video_file"] = video_exists
        entry["has_scene"] = video_exists

        scene_cache_path = canonical_dir / "scene_cache_v2.h5"
        if args.scene_cache_root:
            # Alternative caches (e.g. estimated depth) take precedence where they exist.
            alt = Path(args.scene_cache_root) / entry["segment_id"] / "scene_cache_v2.h5"
            if alt.exists():
                scene_cache_path = alt
        has_hdf5_scene, has_hdf5_semantic, scene_cache_error, scene_cache_frame_count = _scene_cache_hdf5_status(scene_cache_path, n_frames)
        entry["has_scene_features"] = has_hdf5_scene
        entry["has_scene_features_by_frame"] = has_hdf5_scene
        entry["requires_scene_features_by_frame"] = bool(dynamic_present_anchor)
        entry["scene_features_by_frame_count"] = scene_cache_frame_count
        entry["has_scene_semantic"] = bool(has_hdf5_semantic)
        if has_hdf5_scene:
            entry["scene_cache_path"] = str(scene_cache_path.resolve())
            entry["scene_cache_format"] = "hdf5_v2"
            entry["scene_cache_frame_count"] = scene_cache_frame_count
        else:
            for key in ("scene_cache_path", "scene_cache_format", "scene_cache_frame_count"):
                entry.pop(key, None)
        if scene_cache_error:
            errors.append(f"WARNING: {scene_cache_error}")
            validation_counts["warnings"] += 1

        real_errors = [e for e in errors if not e.startswith("WARNING:")]
        entry["validated"] = len(real_errors) == 0
        entry["has_complete_scene_sample"] = bool(entry["validated"] and has_hdf5_scene)
        if errors:
            entry["validation_errors"] = errors

        if entry["validated"]:
            validation_counts["passed"] += 1
        else:
            validation_counts["failed"] += 1
            print(f"  FAIL {segment_name}: {real_errors[:2]}")

        entries.append(entry)
        dataset_counts[source_dataset] += 1

    entries.sort(key=lambda e: (e.get("source_dataset", ""), e.get("segment_id", "")))

    manifest = {
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "total_segments": len(entries),
        "validated_segments": validation_counts["passed"],
        "datasets": dict(dataset_counts),
        "entries": entries,
    }

    out_path = Path(args.output) if args.output else OUTPUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=2)
    tmp_path.replace(out_path)

    print(f"\nManifest written to {out_path}")
    print(f"  Total: {len(entries)} segments")
    print(f"  Interaction-split parents and idle segments (excluded): {n_superseded}")
    print(f"  Gate-rejected (excluded): {gate_rejected}" + ("  [--include-gate-rejects to keep]" if not args.include_gate_rejects else "  [kept via --include-gate-rejects]"))
    print(f"  Validated: {validation_counts['passed']} passed, {validation_counts['failed']} failed, {validation_counts['warnings']} warnings")
    print(f"  Datasets: {dict(dataset_counts)}")
    gated = [e["interaction_fidelity_score"] for e in entries if "interaction_fidelity_score" in e]
    if gated:
        print(f"  Interaction-fidelity weight (gated segs, n={len(gated)}): mean {np.mean(gated):.3f}, min {min(gated):.3f}")
    tiers = Counter(e.get("confidence_tier", "unknown") for e in entries)
    print(f"  Tiers: {dict(tiers)}")


def main():
    parser = argparse.ArgumentParser(prog="python -m gtf.datasets.build_manifest", description="Build the training manifest")
    parser.add_argument("--results-root", type=str, default=str(RESULTS_DIR), help=f"Segment tree to scan (default: {RESULTS_DIR})")
    parser.add_argument("--dataset", type=str, help="Only this source dataset (e.g., hot3d, dexycb)")
    parser.add_argument("--include-gate-rejects", action="store_true", help="Keep segments that failed the quality gate (default: exclude them)")
    parser.add_argument("--output", type=str, default=None, help=f"Output path (default: {OUTPUT_PATH})")
    parser.add_argument("--fast-schema", action="store_true", help="Check NPZ headers only instead of scanning every array for NaN/Inf")
    parser.add_argument("--scene-cache-root", default=None, help="Use DIR/<segment_id>/scene_cache_v2.h5 where it exists (see gtf.datasets.scene.preprocess --cache-root)")
    build_manifest(parser.parse_args())


if __name__ == "__main__":
    main()
