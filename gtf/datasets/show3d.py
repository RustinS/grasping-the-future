#!/usr/bin/env python3
"""Stage, convert, and validate SHOW3D data.

SHOW3D ships object 6DoF tracks, camera calibration, and UmeTrack hand tracks,
but no MANO parameters, so this loader fits MANO to the released UmeTrack/HOT3D-order
landmarks and keeps only frames whose fit residual passes a per-frame threshold.

Segments are written at the 60 Hz source rate; ``split_interaction_segments.py``
later keeps every second row of each trainable segment.

Usage:
    python -m gtf.datasets.show3d --start-from download --resolve-from-index --max-seqs 0
    python -m gtf.datasets.show3d --start-from convert --max-seqs 0
    python -m gtf.datasets.show3d --start-from convert --subject-id ASC023 --scene-id aria_inspecting_f5e9 --out-root /abs/scratch
"""

from __future__ import annotations

import argparse
import json
import os
import re
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from gtf.paths import MANO_DIR, REPO_ROOT
from gtf.utils.hand_rigid import transform_hand_rigid
from gtf.utils import RESULTS_DIR, rprint as print
from gtf.utils.canonical import (
    _preload_torch_cuda_deps,
    load_mesh,
    mano_fk_verts,
    rotmat_to_aa,
    validate_canonical_npz,
)
from gtf.datasets.validate_conventions import validate_canonical_conventions

ROOT = REPO_ROOT

PIPELINE_STEPS = ["download", "convert", "validate"]

SHOW3D_DIR = ROOT / "datasets" / "show3d"

# The SHOW3D (Quest) world is Y-up.
_SHOW3D_GRAVITY_WORLD = np.array([0.0, -1.0, 0.0], dtype=np.float32)


def _gravity_direction_mid(w2c_frames: np.ndarray, g_world: np.ndarray) -> np.ndarray:
    """Camera-frame gravity at the segment's middle frame from the constant world gravity."""
    E = w2c_frames[len(w2c_frames) // 2]
    g = E[:3, :3] @ g_world
    return (g / np.linalg.norm(g)).astype(np.float32)


HF_REPO = "facebook/show3d-dataset"
FPS = 60.0

# SHOW3D object aliases that reuse HOT3D meshes. The SHOW3D README lists HOT3D library
# IDs; the HOT3D object models are addressed by the BOP IDs of gtf.datasets.hot3d.BOP_OBJECTS,
# so each alias maps to the BOP ID of the same object.
SHOW3D_TO_HOT3D_BOP_ID = {
    "brushholder": 1,  # holder_black
    "balandabowl": 2,  # bowl
    "mug2": 8,  # mug_patterned
    "mug": 9,  # mug_white
    "cansoup": 10,
    "canparmesan": 11,
    "cantomatosauce": 12,
    "mustard": 13,
    "bbq": 14,
    "ranch": 15,
    "vase": 16,
    "milk": 17,
    "orangejuice": 18,
    "waffles": 20,
    "vegetables": 21,
    "dumbbell": 22,
    "aria": 23,
    "birdhousetoy": 26,
    "dinotoy": 27,
    "keyboard": 28,
    "mouse": 31,
    "whiteboardmarker": 32,
}

# SHOW3D publishes UmeTrack/HOT3D landmarks in a different order from manopth's joints;
# the two index lists below pair them by meaning (SHOW3D 5 = MANO 0 = wrist, ...).
#
# SHOW3D: tips 0..4; wrist 5; thumb internals 6..7; index 8..10;
#         middle 11..13; ring 14..16; pinky 17..19; palm centre 20.
# MANO:   wrist 0; thumb 1..4; index 5..8; middle 9..12; ring 13..16;
#         pinky 17..20.  MANO thumb joint 1 and SHOW3D palm centre have no
#         direct counterpart, so both are omitted.
SHOW3D_FIT_LANDMARKS = np.asarray([5, 6, 7, 0, 8, 9, 10, 1, 11, 12, 13, 2, 14, 15, 16, 3, 17, 18, 19, 4], dtype=np.int64)
MANO_FIT_LANDMARKS = np.asarray([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], dtype=np.int64)
MANO_LANDMARK_MAPPING = "show3d_umetrack_to_manopth"
_MANO_LAYER_CACHE: dict[str, Any] = {}


@dataclass
class RawSceneReport:
    subject_id: str
    scene_id: str
    object_alias: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    object_frames: int = 0
    object_camera_z_positive_pct: float | None = None
    object_rotation_max_orth_err: float = 0.0
    object_rotation_max_det_err: float = 0.0
    left_frames: int = 0
    right_frames: int = 0
    left_camera_z_positive_pct: float | None = None
    right_camera_z_positive_pct: float | None = None
    hand_rotation_max_orth_err: float = 0.0
    hand_rotation_max_det_err: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.errors


def _safe_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text.strip()).strip("_")


def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _scene_filter(subject_id: str, scene_id: str, sequence: str) -> bool:
    if not sequence:
        return True
    needle = sequence.strip("/")
    return needle in scene_id or needle in subject_id or needle in f"{subject_id}/{scene_id}"


def _parse_pose_path(path: Path) -> tuple[str, str]:
    # .../object_pose/v1/scenes/<subject>/<scene>/object_pose.json
    return path.parts[-3], path.parts[-2]


def _discover_staged_scenes(root: Path, sequence: str = "") -> list[tuple[str, str]]:
    scenes: list[tuple[str, str]] = []
    for obj_path in sorted((root / "object_pose" / "v1" / "scenes").glob("*/*/object_pose.json")):
        subject_id, scene_id = _parse_pose_path(obj_path)
        hand_path = root / "hand_pose" / "v1" / "scenes" / subject_id / scene_id / "hand_pose.json"
        if hand_path.exists() and _scene_filter(subject_id, scene_id, sequence):
            scenes.append((subject_id, scene_id))
    return scenes


def _camera_calibration_path(root: Path, subject_id: str, scene_id: str, camera: str) -> Path:
    return root / "scenes" / subject_id / scene_id / "camera_calibration" / f"{camera}.json"


def _scene_file_patterns(subject_id: str, scene_id: str, camera: str) -> list[str]:
    cams = [camera]
    if camera == "both_headsets":
        cams = ["headset0", "headset1"]
    patterns = [
        "README.md",
        "hand_pose/README.md",
        "object_pose/README.md",
        "captions/README.md",
        "scenes/README.md",
        "dataset_index_train.parquet",
        "dataset_index_test.parquet",
        "hand_pose/v1/index.parquet",
        "object_pose/v1/index.parquet",
        f"object_pose/v1/scenes/{subject_id}/{scene_id}/object_pose.json",
        f"hand_pose/v1/scenes/{subject_id}/{scene_id}/hand_pose.json",
        f"captions/v1/scenes/{subject_id}/{scene_id}/caption.json",
        f"scenes/{subject_id}/{scene_id}/metadata/frame_info.json",
    ]
    for cam in cams:
        patterns.append(f"scenes/{subject_id}/{scene_id}/camera_calibration/{cam}.json")
        patterns.append(f"scenes/{subject_id}/{scene_id}/{cam}.mp4")
        patterns.append(f"scenes/{subject_id}/{scene_id}/blur_info/{cam}.mp4.json")
    return patterns


def _scene_is_staged(root: Path, subject_id: str, scene_id: str) -> bool:
    obj_path = root / "object_pose" / "v1" / "scenes" / subject_id / scene_id / "object_pose.json"
    hand_path = root / "hand_pose" / "v1" / "scenes" / subject_id / scene_id / "hand_pose.json"
    return obj_path.exists() and hand_path.exists()


def _filter_resolved_scenes(scenes: list[tuple[str, str]], args) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    root = Path(args.root)
    for item in scenes:
        object_alias = item[1].split("_", 1)[0]
        if not args.include_unmapped_objects and object_alias not in SHOW3D_TO_HOT3D_BOP_ID:
            continue
        if item in seen:
            continue
        if args.sequence and not _scene_filter(item[0], item[1], args.sequence):
            continue
        if args.skip_staged and _scene_is_staged(root, item[0], item[1]):
            continue
        seen.add(item)
        out.append(item)
    if args.seq_offset:
        out = out[args.seq_offset :]
    if args.max_seqs:
        out = out[: args.max_seqs]
    return out


def _resolve_index_scenes(root: Path, args) -> list[tuple[str, str]] | None:
    obj_index = root / "object_pose" / "v1" / "index.parquet"
    hand_index = root / "hand_pose" / "v1" / "index.parquet"
    if not obj_index.exists() or not hand_index.exists():
        return None

    try:
        import pandas as pd

        obj_df = pd.read_parquet(obj_index)
        hand_df = pd.read_parquet(hand_index, columns=["subject_id", "scene_id"])
    except Exception as e:
        print(f"  WARN could not read SHOW3D local indexes: {e}")
        return None

    hand_scenes = set(zip(hand_df["subject_id"].astype(str), hand_df["scene_id"].astype(str)))
    available_scenes: set[tuple[str, str]] | None = None
    dataset_indexes = [root / "dataset_index_train.parquet", root / "dataset_index_test.parquet"]
    if all(p.exists() for p in dataset_indexes):
        try:
            data_df = pd.concat(
                [pd.read_parquet(p, columns=["subject_id", "scene_id", "has_hand_pose", "has_object_pose"]) for p in dataset_indexes],
                ignore_index=True,
            )
            data_df = data_df[(data_df["has_hand_pose"].astype(bool)) & (data_df["has_object_pose"].astype(bool))]
            available_scenes = set(zip(data_df["subject_id"].astype(str), data_df["scene_id"].astype(str)))
        except Exception as e:
            print(f"  WARN could not read SHOW3D dataset availability indexes: {e}")

    scenes: list[tuple[str, str]] = []
    for row in obj_df.itertuples(index=False):
        subject_id = str(getattr(row, "subject_id"))
        scene_id = str(getattr(row, "scene_id"))
        item = (subject_id, scene_id)
        if item not in hand_scenes:
            continue
        if available_scenes is not None and item not in available_scenes:
            continue
        scenes.append(item)
    return _filter_resolved_scenes(scenes, args)


def _resolve_hf_scenes(args) -> list[tuple[str, str]]:
    from huggingface_hub import HfApi

    if args.subject_id and args.scene_id:
        return [(args.subject_id, args.scene_id)]

    root = Path(args.root)
    if args.resolve_from_index:
        scenes = _resolve_index_scenes(root, args)
        if scenes is not None:
            return scenes
        print("  WARN --resolve-from-index requested, but local SHOW3D indexes were unavailable; falling back to HF tree")

    try:
        files = HfApi().list_repo_files(HF_REPO, repo_type="dataset")
    except Exception as e:
        print(f"  WARN SHOW3D HF tree listing failed: {e}")
        scenes = _resolve_index_scenes(root, args)
        if scenes is not None:
            print("  Using local SHOW3D indexes for scene discovery")
            return scenes
        raise

    scenes: list[tuple[str, str]] = []
    pat = re.compile(r"^object_pose/v1/scenes/([^/]+)/([^/]+)/object_pose\.json$")
    for filename in files:
        m = pat.match(filename)
        if not m:
            continue
        scenes.append((m.group(1), m.group(2)))
    return _filter_resolved_scenes(scenes, args)


def _download_file(filename: str, root: Path, required: bool = False) -> bool:
    from huggingface_hub import hf_hub_download

    try:
        hf_hub_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            filename=filename,
            local_dir=str(root),
            cache_dir=str(root / ".hf_cache"),
        )
        return True
    except Exception as e:
        if required:
            raise
        print(f"  WARN could not download {filename}: {e}")
        return False


def run_download_step(args) -> None:
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    scenes = _resolve_hf_scenes(args)
    if not scenes:
        print("No SHOW3D scenes matched the requested filters; downloading READMEs/indexes only")
        for filename in [
            "README.md",
            "hand_pose/README.md",
            "object_pose/README.md",
            "captions/README.md",
            "scenes/README.md",
            "dataset_index_train.parquet",
            "dataset_index_test.parquet",
            "hand_pose/v1/index.parquet",
            "object_pose/v1/index.parquet",
        ]:
            _download_file(filename, root)
        return

    print(f"Staging {len(scenes)} SHOW3D scene(s) under {root}")
    for subject_id, scene_id in scenes:
        print(f"  {subject_id}/{scene_id}")
        for filename in _scene_file_patterns(subject_id, scene_id, args.camera):
            required = filename.endswith("object_pose.json") or filename.endswith("hand_pose.json")
            _download_file(filename, root, required=required)


def _rotation_errors(R: np.ndarray) -> tuple[float, float]:
    orth = float(np.linalg.norm(R.T @ R - np.eye(3), ord="fro"))
    det = float(abs(np.linalg.det(R) - 1.0))
    return orth, det


def _camera_from_world(calib: dict[str, Any], frame_key: str) -> np.ndarray | None:
    by_index = calib.get("T_WorldFromCamera_by_index", {})
    item = by_index.get(str(frame_key))
    if not item:
        return None
    T_wc = np.asarray(item.get("T_WorldFromCamera", []), dtype=np.float64)
    if T_wc.shape != (4, 4) or not np.isfinite(T_wc).all():
        return None
    # SHOW3D pose JSON stores world translations in millimeters. Canonical data
    # is meters, so convert the homogeneous transform translation before invert.
    T_wc = T_wc.copy()
    T_wc[:3, 3] *= 0.001
    return np.linalg.inv(T_wc)


def _to_camera_point(T_cw: np.ndarray | None, p_world_mm: np.ndarray) -> np.ndarray | None:
    if T_cw is None:
        return None
    ph = np.ones(4, dtype=np.float64)
    ph[:3] = np.asarray(p_world_mm, dtype=np.float64).reshape(3) * 0.001
    return (T_cw @ ph)[:3]


def _to_camera_points(T_cw: np.ndarray, p_world_mm: np.ndarray) -> np.ndarray:
    pts = np.asarray(p_world_mm, dtype=np.float64) * 0.001
    ph = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    return (T_cw @ ph.T).T[:, :3].astype(np.float32)


def _world_from_object_m(frame: dict[str, Any]) -> np.ndarray | None:
    R = np.asarray(frame.get("R", []), dtype=np.float64)
    t = np.asarray(frame.get("t", []), dtype=np.float64).reshape(-1)
    if R.shape != (3, 3) or t.shape != (3,) or not np.isfinite(R).all() or not np.isfinite(t).all():
        return None
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t * 0.001
    return T


def _split_contiguous_runs(frame_indices: np.ndarray, valid: np.ndarray, min_frames: int) -> list[np.ndarray]:
    runs: list[list[int]] = []
    cur: list[int] = []
    prev_frame: int | None = None
    for local_i, (frame_i, ok) in enumerate(zip(frame_indices.tolist(), valid.tolist())):
        if not ok:
            if len(cur) >= min_frames:
                runs.append(cur)
            cur = []
            prev_frame = None
            continue
        if prev_frame is None or frame_i == prev_frame + 1:
            cur.append(local_i)
        else:
            if len(cur) >= min_frames:
                runs.append(cur)
            cur = [local_i]
        prev_frame = frame_i
    if len(cur) >= min_frames:
        runs.append(cur)
    return [np.asarray(r, dtype=np.int64) for r in runs]


def _camera_hand_to_world_series(
    orient_cam: np.ndarray,
    trans_cam: np.ndarray,
    valid: np.ndarray,
    side: str,
    T_cw: np.ndarray,
    betas: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    # World fields use the shaped J0(betas) pivot, so the present-camera re-anchor
    # reproduces the camera-frame fit exactly.
    orient_world = np.zeros_like(orient_cam)
    trans_world = np.zeros_like(trans_cam)
    for i in np.flatnonzero(valid):
        T_wc = np.linalg.inv(T_cw[i]).astype(np.float32)
        o, tr = transform_hand_rigid(
            orient_cam[i : i + 1],
            trans_cam[i : i + 1],
            np.ones(1, dtype=bool),
            side,
            R=T_wc[:3, :3],
            t=T_wc[:3, 3],
            betas=betas,
        )
        orient_world[i] = o[0]
        trans_world[i] = tr[0]
    return orient_world.astype(np.float32), trans_world.astype(np.float32)


def _load_object_mesh(object_alias: str) -> tuple[np.ndarray, np.ndarray, Path, int, str]:
    if object_alias not in SHOW3D_TO_HOT3D_BOP_ID:
        raise ValueError(f"SHOW3D object alias {object_alias!r} has no local HOT3D BOP mesh mapping")
    from gtf.datasets.hot3d import BOP_OBJECTS, _convert_glb_to_obj

    bop_id = SHOW3D_TO_HOT3D_BOP_ID[object_alias]
    mesh_path = _convert_glb_to_obj(bop_id)
    verts, faces = load_mesh(str(mesh_path))
    category = BOP_OBJECTS.get(bop_id, ("", object_alias))[1]
    return verts.astype(np.float32), faces.astype(np.int32), mesh_path, bop_id, category


def validate_raw_scene(root: Path, subject_id: str, scene_id: str, args) -> RawSceneReport:
    object_alias = scene_id.split("_", 1)[0]
    report = RawSceneReport(subject_id=subject_id, scene_id=scene_id, object_alias=object_alias)

    obj_path = root / "object_pose" / "v1" / "scenes" / subject_id / scene_id / "object_pose.json"
    hand_path = root / "hand_pose" / "v1" / "scenes" / subject_id / scene_id / "hand_pose.json"
    calib_path = _camera_calibration_path(root, subject_id, scene_id, args.camera)

    if object_alias not in SHOW3D_TO_HOT3D_BOP_ID:
        report.warnings.append(f"object alias {object_alias!r} has no HOT3D canonical mesh mapping")
    if not calib_path.exists():
        report.warnings.append(f"camera calibration missing for {args.camera}; camera-depth convention not checked")

    try:
        obj = _load_json(obj_path)
        hand = _load_json(hand_path)
        calib = _load_json(calib_path) if calib_path.exists() else {}
    except Exception as e:
        report.errors.append(f"failed to load SHOW3D scene files: {e}")
        return report

    object_z: list[float] = []
    left_z: list[float] = []
    right_z: list[float] = []
    checked = 0
    for frame_key in sorted(obj.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
        if args.max_inspect_frames and checked >= args.max_inspect_frames:
            break
        frame = obj[frame_key]
        conf = float(frame.get("confidence", 0.0) or 0.0)
        if conf <= args.object_confidence:
            continue
        R = np.asarray(frame.get("R", []), dtype=np.float64)
        t = np.asarray(frame.get("t", []), dtype=np.float64).reshape(-1)
        if R.shape != (3, 3) or t.shape != (3,) or not np.isfinite(R).all() or not np.isfinite(t).all():
            report.errors.append(f"bad object pose at frame {frame_key}")
            continue
        orth, det = _rotation_errors(R)
        report.object_rotation_max_orth_err = max(report.object_rotation_max_orth_err, orth)
        report.object_rotation_max_det_err = max(report.object_rotation_max_det_err, det)
        T_cw = _camera_from_world(calib, frame_key) if calib else None
        p_cam = _to_camera_point(T_cw, t)
        if p_cam is not None:
            object_z.append(float(p_cam[2]))
        checked += 1
    report.object_frames = checked

    if report.object_rotation_max_orth_err > args.rotation_tol:
        report.errors.append(f"object rotations are not orthonormal enough: max error {report.object_rotation_max_orth_err:.3g}")
    if report.object_rotation_max_det_err > args.rotation_tol:
        report.errors.append(f"object rotation determinant deviates from +1: max error {report.object_rotation_max_det_err:.3g}")
    if object_z:
        report.object_camera_z_positive_pct = 100.0 * sum(z > 0 for z in object_z) / len(object_z)
        if report.object_camera_z_positive_pct < args.min_positive_z_pct:
            report.errors.append(f"object camera-z positive rate {report.object_camera_z_positive_pct:.1f}% < {args.min_positive_z_pct:.1f}%")
    elif calib:
        report.warnings.append("no valid object frame overlapped camera calibration for depth-axis check")

    hand_checked = {"0": 0, "1": 0}
    for frame_key in sorted(hand.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
        item = hand[frame_key]
        poses = item.get("hand_poses", {}) or {}
        T_cw = _camera_from_world(calib, frame_key) if calib else None
        for hand_id, z_list in (("0", left_z), ("1", right_z)):
            hp = poses.get(hand_id)
            if not hp:
                continue
            conf = float(hp.get("confidence", 0.0) or 0.0)
            if conf <= args.hand_confidence:
                continue
            if args.max_inspect_frames and hand_checked[hand_id] >= args.max_inspect_frames:
                continue
            joint_angles = np.asarray(hp.get("joint_angles", []), dtype=np.float64)
            R = np.asarray(hp.get("wrist_rotation", []), dtype=np.float64)
            t = np.asarray(hp.get("wrist_translation", []), dtype=np.float64)
            if joint_angles.shape != (22,):
                report.errors.append(f"bad UmeTrack joint_angles shape at frame {frame_key} hand {hand_id}: {joint_angles.shape}")
                continue
            if R.shape != (3, 3) or t.shape != (3,) or not np.isfinite(R).all() or not np.isfinite(t).all():
                report.errors.append(f"bad UmeTrack wrist pose at frame {frame_key} hand {hand_id}")
                continue
            orth, det = _rotation_errors(R)
            report.hand_rotation_max_orth_err = max(report.hand_rotation_max_orth_err, orth)
            report.hand_rotation_max_det_err = max(report.hand_rotation_max_det_err, det)
            p_cam = _to_camera_point(T_cw, t)
            if p_cam is not None:
                z_list.append(float(p_cam[2]))
            hand_checked[hand_id] += 1

    report.left_frames = hand_checked["0"]
    report.right_frames = hand_checked["1"]
    if report.hand_rotation_max_orth_err > args.rotation_tol:
        report.errors.append(f"hand wrist rotations are not orthonormal enough: max error {report.hand_rotation_max_orth_err:.3g}")
    if report.hand_rotation_max_det_err > args.rotation_tol:
        report.errors.append(f"hand wrist rotation determinant deviates from +1: max error {report.hand_rotation_max_det_err:.3g}")
    for side, z_vals in (("left", left_z), ("right", right_z)):
        if not z_vals:
            continue
        pct = 100.0 * sum(z > 0 for z in z_vals) / len(z_vals)
        if side == "left":
            report.left_camera_z_positive_pct = pct
        else:
            report.right_camera_z_positive_pct = pct
        if pct < args.min_positive_z_pct:
            report.errors.append(f"{side} wrist camera-z positive rate {pct:.1f}% < {args.min_positive_z_pct:.1f}%")

    if not args.max_inspect_frames and report.object_frames < args.min_frames:
        report.errors.append(f"only {report.object_frames} object frames above confidence threshold; need >= {args.min_frames}")
    elif report.object_frames == 0:
        report.warnings.append("no object frames above confidence threshold")
    if report.left_frames == 0 and report.right_frames == 0:
        report.warnings.append("no hand frames above confidence threshold")
    report.warnings.append("hand representation is UmeTrack landmarks fitted to MANO only during canonical export")
    return report


def _print_report(report: RawSceneReport, verbose: bool = False) -> None:
    bop_id = SHOW3D_TO_HOT3D_BOP_ID.get(report.object_alias)
    mesh_text = f"HOT3D BOP {bop_id}" if bop_id is not None else "no HOT3D mesh"
    z_text = "n/a" if report.object_camera_z_positive_pct is None else f"{report.object_camera_z_positive_pct:.1f}%"
    left_z = "n/a" if report.left_camera_z_positive_pct is None else f"{report.left_camera_z_positive_pct:.1f}%"
    right_z = "n/a" if report.right_camera_z_positive_pct is None else f"{report.right_camera_z_positive_pct:.1f}%"
    state = "OK" if report.ok else "FAIL"
    print(
        f"  {state} {report.subject_id}/{report.scene_id}: object={report.object_alias} ({mesh_text}), "
        f"object_frames={report.object_frames}, hand_frames=L{report.left_frames}/R{report.right_frames}, "
        f"z+ object={z_text} left={left_z} right={right_z}"
    )
    if verbose or not report.ok:
        for msg in report.errors[:5]:
            print(f"    ERROR {msg}")
        for msg in report.warnings[:5]:
            print(f"    WARN {msg}")


def _validate_staged_raw(args) -> list[RawSceneReport]:
    root = Path(args.root)
    scenes = _discover_staged_scenes(root, args.sequence)
    if args.subject_id:
        scenes = [s for s in scenes if s[0] == args.subject_id]
    if args.scene_id:
        scenes = [s for s in scenes if s[1] == args.scene_id]
    if args.seq_offset:
        scenes = scenes[args.seq_offset :]
    if args.max_seqs:
        scenes = scenes[: args.max_seqs]
    if not scenes:
        print(f"No staged SHOW3D scenes with both object and hand pose found under {root}")
        return []

    print(f"Validating {len(scenes)} staged SHOW3D raw scene(s)")
    reports = []
    for subject_id, scene_id in scenes:
        report = validate_raw_scene(root, subject_id, scene_id, args)
        reports.append(report)
        _print_report(report, verbose=args.print_warnings)
    return reports


@dataclass
class ManoFitResult:
    orient: np.ndarray
    pose: np.ndarray
    cam_t: np.ndarray
    betas: np.ndarray
    valid: np.ndarray
    mean_error_mm: np.ndarray
    p90_error_mm: np.ndarray
    attempted_frames: int = 0
    passed_frames: int = 0
    anatomy_report: dict[str, Any] = field(default_factory=dict)


def _select_uniform_indices(idx: np.ndarray, max_count: int) -> np.ndarray:
    if max_count <= 0 or len(idx) <= max_count:
        return idx
    take = np.linspace(0, len(idx) - 1, max_count).round().astype(np.int64)
    return idx[take]


def _write_scene_video(root: Path, subject_id: str, scene_id: str, camera: str, frame_indices: np.ndarray, out_path: Path) -> bool:
    video_src = root / "scenes" / subject_id / scene_id / f"{camera}.mp4"
    if not video_src.exists():
        return False
    try:
        import cv2
    except Exception:
        return False

    cap = cv2.VideoCapture(str(video_src))
    if not cap.isOpened():
        return False
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary name, check the decoded frame count, then replace.
    tmp_path = out_path.with_name(out_path.name + ".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height))
    if not writer.isOpened():
        cap.release()
        return False
    try:
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        for f in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, img = cap.read()
            writer.write(img if ok and img is not None else blank)
    finally:
        writer.release()
        cap.release()
    ok = tmp_path.exists() and tmp_path.stat().st_size > 0
    if ok:
        chk = cv2.VideoCapture(str(tmp_path))
        n_dec = int(chk.get(cv2.CAP_PROP_FRAME_COUNT))
        chk.release()
        ok = n_dec == len(frame_indices)
        if not ok:
            print(f"  [show3d video] decoded frame count {n_dec} != {len(frame_indices)}; {out_path} not written")
    if ok:
        os.replace(tmp_path, out_path)
    else:
        tmp_path.unlink(missing_ok=True)
    return ok


def _resolve_fit_device(torch, fit_device: str):
    requested = (fit_device or "auto").strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("--fit-device cuda requested, but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    if requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ValueError(f"--fit-device {fit_device} requested, but torch.cuda.is_available() is false")
        return torch.device(requested)
    raise ValueError(f"unsupported --fit-device {fit_device!r}; use auto, cpu, cuda, or cuda:<index>")


def _mano_layer(fit_device: str):
    _preload_torch_cuda_deps()
    import torch
    from manopth.manolayer import ManoLayer

    device = _resolve_fit_device(torch, fit_device)
    key = str(device)
    layer = _MANO_LAYER_CACHE.get(key)
    if layer is None:
        layer = ManoLayer(side="right", use_pca=False, ncomps=45, flat_hand_mean=True, mano_root=str(MANO_DIR)).to(device)
        layer.eval()
        for param in layer.parameters():
            param.requires_grad_(False)
        _MANO_LAYER_CACHE[key] = layer
    return torch, layer, device


def _initial_cam_t(target_np: np.ndarray, side: str) -> np.ndarray:
    # SHOW3D landmark 5 is the wrist; it initializes the translation.
    init = target_np[:, 5].astype(np.float32).copy()
    # Stored left-hand params are right-model pre-mirror params; mano_fk_verts
    # mirrors the FK output in x for side="left".
    if side == "left":
        init[:, 0] *= -1.0
    return init


def _fit_mano_batch(
    target_np: np.ndarray,
    side: str,
    betas_np: np.ndarray,
    *,
    fit_device: str,
    optimize_betas: bool,
    steps: int,
    lr: float,
    pose_reg: float,
    beta_reg: float,
):
    torch, layer, device = _mano_layer(fit_device)
    target = torch.as_tensor(target_np, dtype=torch.float32, device=device)
    n = int(target.shape[0])
    mano_idx = torch.as_tensor(MANO_FIT_LANDMARKS, dtype=torch.long, device=device)
    target_idx = torch.as_tensor(SHOW3D_FIT_LANDMARKS, dtype=torch.long, device=device)

    orient = torch.zeros(n, 3, dtype=torch.float32, device=device, requires_grad=True)
    pose = torch.zeros(n, 45, dtype=torch.float32, device=device, requires_grad=True)
    cam_t = torch.as_tensor(_initial_cam_t(target_np, side), dtype=torch.float32, device=device).requires_grad_(True)
    betas = torch.as_tensor(betas_np.astype(np.float32), dtype=torch.float32, device=device).requires_grad_(optimize_betas)

    params = [orient, pose, cam_t]
    if optimize_betas:
        params.append(betas)
    opt = torch.optim.Adam(params, lr=lr)

    for _ in range(max(1, int(steps))):
        opt.zero_grad(set_to_none=True)
        full_pose = torch.cat([orient, pose], dim=1)
        verts, joints = layer(full_pose, betas.unsqueeze(0).expand(n, -1), cam_t)
        joints = joints / 1000.0
        if side == "left":
            joints = joints.clone()
            joints[:, :, 0] *= -1.0
        pred = joints[:, mano_idx]
        tgt = target[:, target_idx]
        dist = torch.linalg.norm(pred - tgt, dim=-1)
        loss = dist.mean()
        loss = loss + float(pose_reg) * pose.square().mean()
        loss = loss + float(beta_reg) * betas.square().mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        full_pose = torch.cat([orient, pose], dim=1)
        verts, joints = layer(full_pose, betas.unsqueeze(0).expand(n, -1), cam_t)
        verts = verts / 1000.0
        joints = joints / 1000.0
        if side == "left":
            verts = verts.clone()
            joints = joints.clone()
            verts[:, :, 0] *= -1.0
            joints[:, :, 0] *= -1.0
        per_lm_mm = torch.linalg.norm(joints[:, mano_idx] - target[:, target_idx], dim=-1).cpu().numpy() * 1000.0

    return {
        "orient": orient.detach().cpu().numpy().astype(np.float32),
        "pose": pose.detach().cpu().numpy().astype(np.float32),
        "cam_t": cam_t.detach().cpu().numpy().astype(np.float32),
        "betas": betas.detach().cpu().numpy().astype(np.float32),
        "verts": verts.detach().cpu().numpy().astype(np.float32),
        "mean_error_mm": per_lm_mm.mean(axis=1).astype(np.float32),
        "p90_error_mm": np.percentile(per_lm_mm, 90, axis=1).astype(np.float32),
    }


def _evaluate_mano_params(
    target_np: np.ndarray,
    side: str,
    orient_np: np.ndarray,
    pose_np: np.ndarray,
    cam_t_np: np.ndarray,
    betas_np: np.ndarray,
    *,
    fit_device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate final constrained poses under the semantic landmark mapping."""

    torch, layer, device = _mano_layer(fit_device)
    target = torch.as_tensor(target_np, dtype=torch.float32, device=device)
    orient = torch.as_tensor(orient_np, dtype=torch.float32, device=device)
    pose = torch.as_tensor(pose_np, dtype=torch.float32, device=device)
    cam_t = torch.as_tensor(cam_t_np, dtype=torch.float32, device=device)
    betas = torch.as_tensor(betas_np, dtype=torch.float32, device=device).reshape(1, 10).expand(len(target_np), -1)
    mano_idx = torch.as_tensor(MANO_FIT_LANDMARKS, dtype=torch.long, device=device)
    target_idx = torch.as_tensor(SHOW3D_FIT_LANDMARKS, dtype=torch.long, device=device)
    with torch.no_grad():
        _, joints = layer(torch.cat([orient, pose], dim=1), betas, cam_t)
        joints = joints / 1000.0
        if side == "left":
            joints = torch.cat([-joints[..., :1], joints[..., 1:]], dim=-1)
        per_lm_mm = torch.linalg.norm(joints[:, mano_idx] - target[:, target_idx], dim=-1) * 1000.0
    per_lm = per_lm_mm.cpu().numpy()
    return per_lm.mean(axis=1).astype(np.float32), np.percentile(per_lm, 90, axis=1).astype(np.float32)


def _constrain_fit_articulation(
    targets: np.ndarray,
    side: str,
    out: ManoFitResult,
    args,
    frame_indices: np.ndarray | None = None,
) -> None:
    """Apply the swing-twist joint limits and temporal SO(3) smoothing to the fitted poses.

    Landmark residuals are recomputed afterwards; a frame that fails the thresholds
    after the correction is dropped.
    """

    rows = np.flatnonzero(out.valid)
    if not len(rows):
        return
    from gtf.extraction.stage2.articulation_filter import _bone_axes, clamp_swing_twist, smooth_articulation_karcher

    torch, layer, device = _mano_layer(args.fit_device)
    axes = _bone_axes(layer, out.betas, device)
    pose_before = out.pose[rows].copy()
    smooth_radius = int(args.fit_art_smooth)
    twist_mcp = float(args.fit_twist_mcp_deg)
    twist_pip = float(args.fit_twist_pip_deg)
    swing = float(args.fit_swing_deg)

    report: dict[str, Any] = {
        "mapping": MANO_LANDMARK_MAPPING,
        "smooth_radius": smooth_radius,
        "clamp": True,
        "twist_mcp_deg": twist_mcp,
        "twist_pip_deg": twist_pip,
        "swing_deg": swing,
        "n_prevalid": int(len(rows)),
    }
    constrained, before_stats = clamp_swing_twist(pose_before, axes, twist_mcp, twist_pip, swing)
    report["pre_twist_violations"] = int(before_stats["n_twist_viol"])
    report["pre_swing_violations"] = int(before_stats["n_swing_viol"])
    out.pose[rows] = constrained
    if smooth_radius > 0:
        if frame_indices is None or len(frame_indices) < 2:
            out.pose = smooth_articulation_karcher(out.pose, out.valid, smooth_radius)
        else:
            # Smooth separately inside spans that are contiguous in time and fit validity:
            # rows on either side of a tracking outage or a failed fit are not neighbours.
            fi = np.asarray(frame_indices, dtype=np.int64)
            positive_steps = np.diff(fi)
            positive_steps = positive_steps[positive_steps > 0]
            expected_step = int(np.median(positive_steps)) if len(positive_steps) else 1
            invalid_edge = ~(out.valid[:-1] & out.valid[1:])
            boundaries = np.flatnonzero((np.diff(fi) != expected_step) | invalid_edge) + 1
            for span in np.split(np.arange(len(fi)), boundaries):
                if len(span):
                    out.pose[span] = smooth_articulation_karcher(out.pose[span], out.valid[span], smooth_radius)
    # Smoothing is an SO(3) average and can move a boundary pose infinitesimally
    # outside a limit; project once more so the stored poses satisfy the limits exactly.
    out.pose[rows], final_stats = clamp_swing_twist(out.pose[rows], axes, twist_mcp, twist_pip, swing)
    report["post_smooth_twist_violations"] = int(final_stats["n_twist_viol"])
    report["post_smooth_swing_violations"] = int(final_stats["n_swing_viol"])

    # Measure the stored poses with the shaped-axis convention of training/eval
    # (micro-degree roundoff at an exact clamp boundary does not count).
    from gtf.model.repr.hand_anatomy import anatomy_violation_terms
    from gtf.model.repr.rotation import axis_angle_to_matrix

    with torch.no_grad():
        pose_t = torch.as_tensor(out.pose[rows], dtype=torch.float32, device=device).reshape(-1, 15, 3)
        axes_t = torch.as_tensor(axes, dtype=torch.float32, device=device).unsqueeze(0)
        authored_terms = anatomy_violation_terms(
            axis_angle_to_matrix(pose_t),
            axes_t,
            twist_mcp_deg=twist_mcp,
            twist_pip_deg=twist_pip,
            swing_deg=swing,
        )
    report["post_twist_violations"] = int(authored_terms["twist_violation"].sum().item())
    report["post_swing_violations"] = int(authored_terms["swing_violation"].sum().item())

    pose_after = out.pose[rows].copy()
    mean_mm, p90_mm = _evaluate_mano_params(
        targets[rows],
        side,
        out.orient[rows],
        out.pose[rows],
        out.cam_t[rows],
        out.betas,
        fit_device=args.fit_device,
    )
    out.mean_error_mm[rows] = mean_mm
    out.p90_error_mm[rows] = p90_mm
    keep = (mean_mm <= args.fit_mean_threshold_mm) & (p90_mm <= args.fit_p90_threshold_mm)
    rejected = rows[~keep]
    out.valid[rejected] = False
    out.orient[rejected] = 0
    out.pose[rejected] = 0
    out.cam_t[rejected] = 0
    report["n_final_valid"] = int(out.valid.sum())
    report["n_rejected_after_constraint"] = int((~keep).sum())
    report["final_mean_error_mm"] = float(mean_mm[keep].mean()) if keep.any() else None
    report["final_p90_error_mm"] = float(np.percentile(p90_mm[keep], 90)) if keep.any() else None
    report["pose_change_mean_rad"] = float(np.linalg.norm(pose_after - pose_before, axis=1).mean())
    out.anatomy_report = report


def _collect_hand_targets(
    hand: dict[str, Any],
    calib: dict[str, Any],
    frame_indices: np.ndarray,
    side_id: str,
    args,
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.zeros((len(frame_indices), 21, 3), dtype=np.float32)
    valid = np.zeros(len(frame_indices), dtype=bool)
    for local_i, frame_key in enumerate(frame_indices.tolist()):
        item = hand.get(str(frame_key), {})
        hp = (item.get("hand_poses") or {}).get(side_id)
        if not hp:
            continue
        conf = float(hp.get("confidence", 0.0) or 0.0)
        if conf <= args.hand_confidence or hp.get("landmarks_3d_mm") is None:
            continue
        T_cw = _camera_from_world(calib, str(frame_key))
        if T_cw is None:
            continue
        pts = _to_camera_points(T_cw, hp["landmarks_3d_mm"])
        if pts.shape != (21, 3) or not np.isfinite(pts).all():
            continue
        if float((pts[:, 2] > 0.02).mean()) < 0.8:
            continue
        targets[local_i] = pts
        valid[local_i] = True
    return targets, valid


def _fit_side_to_mano(
    targets: np.ndarray,
    target_valid: np.ndarray,
    side: str,
    args,
    frame_indices: np.ndarray | None = None,
) -> ManoFitResult:
    n_total = int(targets.shape[0])
    out = ManoFitResult(
        orient=np.zeros((n_total, 3), dtype=np.float32),
        pose=np.zeros((n_total, 45), dtype=np.float32),
        cam_t=np.zeros((n_total, 3), dtype=np.float32),
        betas=np.zeros(10, dtype=np.float32),
        valid=np.zeros(n_total, dtype=bool),
        mean_error_mm=np.zeros(n_total, dtype=np.float32),
        p90_error_mm=np.zeros(n_total, dtype=np.float32),
    )
    candidate_idx = np.flatnonzero(target_valid)
    candidate_idx = _select_uniform_indices(candidate_idx, args.fit_max_frames)
    out.attempted_frames = int(len(candidate_idx))
    if len(candidate_idx) < args.min_valid_hand_frames:
        return out

    beta_idx = _select_uniform_indices(candidate_idx, args.fit_betas_frames)
    beta_fit = _fit_mano_batch(
        targets[beta_idx],
        side,
        out.betas,
        fit_device=args.fit_device,
        optimize_betas=True,
        steps=args.fit_beta_steps,
        lr=args.fit_lr,
        pose_reg=args.fit_pose_reg,
        beta_reg=args.fit_beta_reg,
    )
    out.betas = beta_fit["betas"]

    for start in range(0, len(candidate_idx), args.fit_batch_size):
        batch_idx = candidate_idx[start : start + args.fit_batch_size]
        fit = _fit_mano_batch(
            targets[batch_idx],
            side,
            out.betas,
            fit_device=args.fit_device,
            optimize_betas=False,
            steps=args.fit_steps,
            lr=args.fit_lr,
            pose_reg=args.fit_pose_reg,
            beta_reg=0.0,
        )
        passed = (fit["mean_error_mm"] <= args.fit_mean_threshold_mm) & (fit["p90_error_mm"] <= args.fit_p90_threshold_mm)
        if passed.any():
            keep_idx = batch_idx[passed]
            out.orient[keep_idx] = fit["orient"][passed]
            out.pose[keep_idx] = fit["pose"][passed]
            out.cam_t[keep_idx] = fit["cam_t"][passed]
            out.valid[keep_idx] = True
        out.mean_error_mm[batch_idx] = fit["mean_error_mm"]
        out.p90_error_mm[batch_idx] = fit["p90_error_mm"]

    _constrain_fit_articulation(targets, side, out, args, frame_indices=frame_indices)
    out.passed_frames = int(out.valid.sum())
    return out


def _convert_scene(root: Path, subject_id: str, scene_id: str, args) -> list[str]:
    object_alias = scene_id.split("_", 1)[0]
    if object_alias not in SHOW3D_TO_HOT3D_BOP_ID:
        print(f"  {subject_id}/{scene_id}: skipped object alias {object_alias!r} (no local HOT3D mesh)")
        return []
    obj_path = root / "object_pose" / "v1" / "scenes" / subject_id / scene_id / "object_pose.json"
    hand_path = root / "hand_pose" / "v1" / "scenes" / subject_id / scene_id / "hand_pose.json"
    calib_path = _camera_calibration_path(root, subject_id, scene_id, args.camera)

    obj = _load_json(obj_path)
    hand = _load_json(hand_path)
    calib = _load_json(calib_path)
    verts0, faces, mesh_path, bop_id, category = _load_object_mesh(object_alias)

    frame_indices: list[int] = []
    obj_rotation: list[np.ndarray] = []
    obj_translation: list[np.ndarray] = []
    obj_rotation_world: list[np.ndarray] = []
    obj_translation_world: list[np.ndarray] = []
    obj_verts: list[np.ndarray] = []
    camera_extrinsics: list[np.ndarray] = []
    for frame_key in sorted(obj.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
        frame = obj[frame_key]
        if float(frame.get("confidence", 0.0) or 0.0) <= args.object_confidence:
            continue
        T_cw = _camera_from_world(calib, frame_key)
        T_wo = _world_from_object_m(frame)
        if T_cw is None or T_wo is None:
            continue
        T_co = (T_cw @ T_wo).astype(np.float32)
        if T_co[2, 3] <= 0.02:
            continue
        R = T_co[:3, :3]
        t = T_co[:3, 3]
        frame_indices.append(int(frame_key))
        camera_extrinsics.append(T_cw.astype(np.float32))
        obj_rotation_world.append(rotmat_to_aa(T_wo[None, :3, :3])[0].astype(np.float32))
        obj_translation_world.append(T_wo[:3, 3].astype(np.float32))
        obj_rotation.append(rotmat_to_aa(R[None])[0].astype(np.float32))
        obj_translation.append(t.astype(np.float32))
        obj_verts.append((verts0 @ R.T + t).astype(np.float32))

    if len(frame_indices) < args.min_frames:
        print(f"  {subject_id}/{scene_id}: no object-valid run with >= {args.min_frames} frames")
        return []

    frame_arr = np.asarray(frame_indices, dtype=np.int64)
    obj_rotation_arr = np.stack(obj_rotation).astype(np.float32)
    obj_translation_arr = np.stack(obj_translation).astype(np.float32)
    obj_rotation_world_arr = np.stack(obj_rotation_world).astype(np.float32)
    obj_translation_world_arr = np.stack(obj_translation_world).astype(np.float32)
    obj_verts_arr = np.stack(obj_verts).astype(np.float32)
    camera_extrinsics_arr = np.stack(camera_extrinsics).astype(np.float32)

    left_targets, left_target_valid = _collect_hand_targets(hand, calib, frame_arr, "0", args)
    right_targets, right_target_valid = _collect_hand_targets(hand, calib, frame_arr, "1", args)
    left = _fit_side_to_mano(left_targets, left_target_valid, "left", args, frame_indices=frame_arr)
    right = _fit_side_to_mano(right_targets, right_target_valid, "right", args, frame_indices=frame_arr)
    left_orient_world, left_trans_world = _camera_hand_to_world_series(left.orient, left.cam_t, left.valid, "left", camera_extrinsics_arr, betas=left.betas)
    right_orient_world, right_trans_world = _camera_hand_to_world_series(right.orient, right.cam_t, right.valid, "right", camera_extrinsics_arr, betas=right.betas)

    active_any = left.valid | right.valid
    runs = _split_contiguous_runs(frame_arr, active_any, args.min_frames)
    if args.max_runs:
        runs = runs[: args.max_runs]
    if not runs:
        print(
            f"  {subject_id}/{scene_id}: no hand-fit-valid run >= {args.min_frames} frames "
            f"(left {left.passed_frames}/{left.attempted_frames}, right {right.passed_frames}/{right.attempted_frames})"
        )
        return []

    out_root = Path(args.out_root or RESULTS_DIR)
    segments: list[str] = []
    K = np.asarray(
        [[calib["fx"], 0.0, calib["cx"]], [0.0, calib["fy"], calib["cy"]], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    side_data = {"left": left, "right": right}
    for run_i, run in enumerate(runs):
        segment_name = f"show3d_{_safe_name(subject_id)}__{_safe_name(scene_id)}_{args.camera}_r{run_i:03d}"
        canonical_dir = out_root / segment_name / "canonical"
        canonical_path = canonical_dir / "canonical.npz"
        if canonical_path.exists() and not args.force:
            segments.append(segment_name)
            continue

        save: dict[str, np.ndarray] = {
            "frame_indices": frame_arr[run].astype(np.int64),
            "fps": np.float32(FPS),
            "obj_canonical_verts": verts0,
            "obj_faces": faces,
            "obj_rotation": obj_rotation_arr[run],
            "obj_translation": obj_translation_arr[run],
            "obj_rotation_world": obj_rotation_world_arr[run],
            "obj_translation_world": obj_translation_world_arr[run],
            "obj_scale": np.ones(3, dtype=np.float32),
            "obj_verts": obj_verts_arr[run],
            "camera_K": K,
            "camera_extrinsics": camera_extrinsics_arr[run],
            "gravity_world": _SHOW3D_GRAVITY_WORLD.copy(),
            "gravity_direction": _gravity_direction_mid(camera_extrinsics_arr[run], _SHOW3D_GRAVITY_WORLD),
        }
        for side, fit in side_data.items():
            side_valid = fit.valid[run].copy()
            save[f"hand_{side}_orient"] = fit.orient[run].copy()
            save[f"hand_{side}_pose"] = fit.pose[run].copy()
            save[f"hand_{side}_cam_t"] = fit.cam_t[run].copy()
            if side == "left":
                save[f"hand_{side}_orient_world"] = left_orient_world[run].copy()
                save[f"hand_{side}_cam_t_world"] = left_trans_world[run].copy()
            else:
                save[f"hand_{side}_orient_world"] = right_orient_world[run].copy()
                save[f"hand_{side}_cam_t_world"] = right_trans_world[run].copy()
            save[f"hand_{side}_betas"] = fit.betas.astype(np.float32)
            save[f"hand_{side}_valid"] = side_valid
            save[f"hand_{side}_fit_mean_error_mm"] = fit.mean_error_mm[run].copy()
            save[f"hand_{side}_fit_p90_error_mm"] = fit.p90_error_mm[run].copy()
            if side_valid.any():
                hv = mano_fk_verts(
                    save[f"hand_{side}_orient"],
                    save[f"hand_{side}_pose"],
                    save[f"hand_{side}_betas"],
                    save[f"hand_{side}_cam_t"],
                    side,
                    str(MANO_DIR),
                )
                hv[~side_valid] = 0
            else:
                hv = np.zeros((len(run), 778, 3), dtype=np.float32)
            save[f"hand_{side}_verts"] = hv.astype(np.float32)

        canonical_dir.mkdir(parents=True, exist_ok=True)
        video_path = out_root / segment_name / "video.mp4"
        has_scene = False if args.no_video else _write_scene_video(root, subject_id, scene_id, args.camera, frame_arr[run], video_path)
        np.savez_compressed(str(canonical_path), **save)
        ok, schema_errors = validate_canonical_npz(str(canonical_path))
        if not ok:
            raise ValueError(f"{segment_name} schema validation failed: {schema_errors[:3]}")
        report = validate_canonical_conventions(
            canonical_path,
            max_fk_frames=None if args.max_validate_frames <= 0 else args.max_validate_frames,
            strict_camera=True,
        )
        if not report.ok:
            canonical_path.unlink(missing_ok=True)
            raise ValueError(f"{segment_name} convention validation failed: {report.errors[:3]}")
        if report.warnings:
            canonical_path.unlink(missing_ok=True)
            print(f"  SKIP {segment_name}: convention validation warnings {report.warnings[:3]}")
            continue

        fit_summary = {
            side: {
                "attempted_frames": int(fit.attempted_frames),
                "passed_frames": int(fit.passed_frames),
                "mean_error_mm": float(fit.mean_error_mm[fit.valid].mean()) if fit.valid.any() else None,
                "p90_error_mm": float(np.percentile(fit.p90_error_mm[fit.valid], 90)) if fit.valid.any() else None,
                "anatomy": fit.anatomy_report,
            }
            for side, fit in side_data.items()
        }
        metadata = {
            "source_dataset": "show3d",
            "segment_id": segment_name,
            "confidence_tier": "gt_landmark_fitted",
            "has_depth": bool(has_scene),
            "has_scene": has_scene,
            "has_gravity": True,
            "gravity_source": "world_axis_constant",
            "has_support_plane": False,
            "has_native_hand_pose": False,
            "has_native_object_pose": True,
            "has_native_umetrack_pose": True,
            "object_is_rigid": True,
            "primary_object_id": object_alias,
            "primary_object_category": category,
            "hot3d_bop_id": int(bop_id),
            "bimanual": bool(save["hand_left_valid"].any() and save["hand_right_valid"].any()),
            "viewpoint": "egocentric" if args.camera.startswith("headset") else "exocentric",
            "frame": f"show3d_{args.camera}_camera",
            "hand_source": "umetrack_landmarks_to_mano_fit",
            "object_source": "markerless_6d_pose",
            "depth_source": "moge2_video" if has_scene else "",
            "mano_convention": "right_model_x_mirror",
            "show3d_subject_id": subject_id,
            "show3d_scene_id": scene_id,
            "show3d_camera": args.camera,
            "mano_fit_landmark_mapping": MANO_LANDMARK_MAPPING,
            "mano_fit_eval_landmarks": SHOW3D_FIT_LANDMARKS.tolist(),
            "mano_fit_target_landmarks": SHOW3D_FIT_LANDMARKS.tolist(),
            "mano_fit_mano_landmarks": MANO_FIT_LANDMARKS.tolist(),
            "mano_fit_mean_threshold_mm": float(args.fit_mean_threshold_mm),
            "mano_fit_p90_threshold_mm": float(args.fit_p90_threshold_mm),
            "mano_fit_summary": fit_summary,
            "n_frames": int(len(run)),
            "fps": FPS,
            "active_sides": [s for s in ("left", "right") if save[f"hand_{s}_valid"].any()],
            "obj_mesh_path": str(mesh_path.relative_to(ROOT)) if mesh_path.is_relative_to(ROOT) else str(mesh_path),
        }
        with open(canonical_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        segments.append(segment_name)
        print(f"  {segment_name}: frames={len(run)}, left={int(save['hand_left_valid'].sum())}, right={int(save['hand_right_valid'].sum())}")
    return segments


def run_convert_step(args) -> list[str]:
    reports = _validate_staged_raw(args)
    if not reports:
        return []
    bad = [r for r in reports if not r.ok]
    if bad:
        print(f"SHOW3D conversion will skip {len(bad)} raw-failed scene(s)")

    root = Path(args.root)
    segments: list[str] = []
    failed = 0
    for report in reports:
        if not report.ok:
            continue
        try:
            new_segments = _convert_scene(root, report.subject_id, report.scene_id, args)
            segments.extend(new_segments)
        except Exception as e:
            failed += 1
            print(f"  ERROR {report.subject_id}/{report.scene_id}: {e}")
            traceback.print_exc()
    print(f"  Convert: {len(segments)} segments, {failed} failed scenes")
    if failed:
        raise SystemExit(1)
    return segments


def run_validate_step(args) -> None:
    reports = _validate_staged_raw(args)
    failed = sum(1 for r in reports if not r.ok)
    warned = sum(1 for r in reports if r.warnings)
    print(f"  Raw validate: {len(reports) - failed} passed, {failed} failed, {warned} warned ({len(reports)} total)")

    out_root = Path(args.out_root or RESULTS_DIR)
    canonicals = sorted(out_root.glob("show3d_*/canonical/canonical.npz"))
    c_passed = c_failed = c_warned = 0
    for p in canonicals:
        report = validate_canonical_conventions(
            p,
            max_fk_frames=None if args.max_validate_frames <= 0 else args.max_validate_frames,
            strict_camera=True,
        )
        if report.ok:
            c_passed += 1
            c_warned += int(bool(report.warnings))
            if report.warnings and args.print_warnings:
                print(f"  WARN {p.parent.parent.name}: {report.warnings[:2]}")
        else:
            c_failed += 1
            print(f"  FAIL {p.parent.parent.name}: {report.errors[:2]}")
    print(f"  Canonical validate: {c_passed} passed, {c_failed} failed, {c_warned} warned ({len(canonicals)} total)")
    if failed and args.strict_raw_validation:
        raise SystemExit(1)
    if c_failed:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m gtf.datasets.show3d", description="Stage/convert/validate SHOW3D data into canonical format")
    parser.add_argument("--start-from", choices=PIPELINE_STEPS, default="download")
    parser.add_argument("--root", type=str, default=str(SHOW3D_DIR))
    parser.add_argument("--out-root", type=str, default="")
    parser.add_argument("--subject-id", type=str, default="")
    parser.add_argument("--scene-id", type=str, default="")
    parser.add_argument("--sequence", type=str, default="")
    parser.add_argument("--camera", type=str, default="headset0")
    parser.add_argument("--seq-offset", type=int, default=0, help="Skip this many discovered SHOW3D scenes before --max-seqs")
    parser.add_argument("--max-seqs", type=int, default=1)
    parser.add_argument("--max-inspect-frames", type=int, default=0)
    parser.add_argument("--object-confidence", type=float, default=0.0)
    parser.add_argument("--hand-confidence", type=float, default=0.5)
    parser.add_argument("--rotation-tol", type=float, default=1e-3)
    parser.add_argument("--min-positive-z-pct", type=float, default=95.0)
    parser.add_argument("--min-frames", type=int, default=30)
    parser.add_argument("--min-valid-hand-frames", type=int, default=10)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--max-validate-frames", type=int, default=0)
    parser.add_argument("--fit-max-frames", type=int, default=0, help="Cap the fitted hand frames per side (0 = all)")
    parser.add_argument("--fit-device", type=str, default="auto", help="MANO fitting device: auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--fit-betas-frames", type=int, default=48)
    parser.add_argument("--fit-beta-steps", type=int, default=450)
    parser.add_argument("--fit-steps", type=int, default=450)
    parser.add_argument("--fit-batch-size", type=int, default=256)
    parser.add_argument("--fit-lr", type=float, default=0.03)
    parser.add_argument("--fit-pose-reg", type=float, default=0.0015)
    parser.add_argument("--fit-beta-reg", type=float, default=0.0003)
    parser.add_argument("--fit-art-smooth", type=int, default=1, help="Per-joint SO(3) temporal smoothing radius after the landmark fit")
    parser.add_argument("--fit-twist-mcp-deg", type=float, default=25.0)
    parser.add_argument("--fit-twist-pip-deg", type=float, default=15.0)
    parser.add_argument("--fit-swing-deg", type=float, default=135.0)
    parser.add_argument("--fit-mean-threshold-mm", type=float, default=25.0)
    parser.add_argument("--fit-p90-threshold-mm", type=float, default=55.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-video", action="store_true", help="Skip writing per-segment video.mp4")
    parser.add_argument("--include-unmapped-objects", action="store_true", help="Download/attempt SHOW3D-only object aliases without local HOT3D meshes")
    parser.add_argument("--resolve-from-index", action="store_true", help="Use local SHOW3D parquet indexes for scene discovery instead of listing the HF tree")
    parser.add_argument("--skip-staged", action="store_true", help="Skip scenes whose object and hand pose JSONs are already staged locally")
    parser.add_argument("--strict-raw-validation", action="store_true", help="Fail validation if staged raw scenes are rejected before canonical conversion")
    parser.add_argument("--download-only", action="store_true", help="Stop after the download step")
    parser.add_argument("--convert-only", action="store_true", help="Stop after conversion; new segments are validated while written")
    parser.add_argument("--print-warnings", action="store_true")
    args = parser.parse_args()

    if args.camera == "both_headsets" and args.start_from != "download":
        raise ValueError("--camera both_headsets is only valid for staging; validate one camera at a time")

    start_idx = PIPELINE_STEPS.index(args.start_from)
    active = set(PIPELINE_STEPS[start_idx:])
    if "download" in active:
        run_download_step(args)
        if args.download_only:
            return
    if "convert" in active:
        run_convert_step(args)
        if args.convert_only:
            return
    if "validate" in active:
        run_validate_step(args)


if __name__ == "__main__":
    main()
