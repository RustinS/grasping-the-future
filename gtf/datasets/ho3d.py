#!/usr/bin/env python3
"""Ingest HO-3D and H2O-3D into model-ready canonical.npz files.

The official HO-3D/H2O-3D annotations are stored in OpenGL camera coordinates
(objects/hands face along negative z). This loader converts them to the OpenCV
camera convention (positive z) with C = diag(1, -1, -1), then writes canonical
segments with one rigid YCB object.

HO-3D has no gravity annotation. ``ho3d_scene_gravity.json`` holds, per segment, the
camera-frame gravity of a RANSAC support-plane fit on its scene point clouds
(gtf/datasets/estimate_scene_gravity.py), or null where the fit was rejected; this
loader writes it. Segments not in the table get no gravity here;
gtf/datasets/estimate_scene_gravity.py estimates it from their scene caches.

Usage:
    python -m gtf.datasets.ho3d --start-from download
    python -m gtf.datasets.ho3d --start-from convert --split train
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import shutil
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from gtf.paths import MANO_DIR, REPO_ROOT
from gtf.datasets.validate_conventions import validate_canonical_conventions
from gtf.utils.hand_rigid import transform_hand_rigid
from gtf.utils import RESULTS_DIR, rprint as print
from gtf.utils.canonical import (
    load_mesh,
    mano_fk_verts,
    mirror_left_hand_params,
    rotmat_to_aa,
    validate_canonical_npz,
)

ROOT = REPO_ROOT

PIPELINE_STEPS = ["download", "convert", "validate"]

HO3D_DIR = ROOT / "datasets" / "ho3d"
H2O3D_DIR = ROOT / "datasets" / "h2o3d"
YCB_MODELS_DIR = ROOT / "datasets" / "dexycb" / "models"
FPS = 30.0
HO3D_KAGGLE_DATASET = "marcmarais/ho3d-v3"

OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0]).astype(np.float32)

SCENE_GRAVITY_TABLE = Path(__file__).with_name("ho3d_scene_gravity.json")
_SCENE_GRAVITY: dict[str, list[float] | None] | None = None


def recorded_scene_gravity() -> dict[str, list[float] | None]:
    """Segment id -> camera-frame gravity from ``ho3d_scene_gravity.json`` (None: fit rejected)."""
    global _SCENE_GRAVITY
    if _SCENE_GRAVITY is None:
        _SCENE_GRAVITY = json.loads(SCENE_GRAVITY_TABLE.read_text())
    return _SCENE_GRAVITY


OFFICIAL_LINKS = {
    "h2o3d_v1": "https://1drv.ms/u/s!AsG9HA3ULXQRgpAq-GOOV2jMYZFmkQ?e=AEEg13",
    "h2o3d_v1_folder": "https://onedrive.live.com/?authkey=%21APhjjldozGGRZpE&id=11742DD40D1CBDC1%2134858&cid=11742DD40D1CBDC1",
}


def _safe_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text.strip()).strip("_")


def _read_pickle(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        try:
            return pickle.load(f, encoding="latin1")
        except TypeError:
            f.seek(0)
            return pickle.load(f)


def _to_vec(value: Any, n: int) -> np.ndarray:
    if value is None:
        return np.zeros(n, dtype=np.float32)
    return np.asarray(value, dtype=np.float32).reshape(n)


def _discover_sequences(root: Path, split: str) -> list[Path]:
    split_dir = root / split
    if not split_dir.exists():
        return []
    return sorted(p for p in split_dir.iterdir() if (p / "meta").is_dir())


def _resolve_dataset_root(root: Path, dataset: str, split: str) -> Path:
    if (root / split).exists():
        return root
    nested_names = ("HO3D_v3", "HO3D_v2") if dataset == "ho3d" else ("H2O3D_v1", "H2O3D")
    for name in nested_names:
        nested = root / name
        if (nested / split).exists():
            return nested
    return root


def _frame_ids(seq_dir: Path) -> list[str]:
    return sorted(p.stem for p in (seq_dir / "meta").glob("*.pkl"))


def _find_mesh(ycb_root: Path, obj_name: str) -> Path:
    candidates = [
        ycb_root / obj_name / "textured_simple.obj",
        ycb_root / obj_name / "textured.obj",
        ycb_root / obj_name / "model.obj",
        ycb_root / obj_name / "model_normalized.obj",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"No YCB mesh for {obj_name} under {ycb_root}")


def _split_valid_runs(valid: np.ndarray, min_frames: int) -> list[np.ndarray]:
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return []
    runs = []
    start = prev = int(idx[0])
    for x_raw in idx[1:]:
        x = int(x_raw)
        if x != prev + 1:
            if prev - start + 1 >= min_frames:
                runs.append(np.arange(start, prev + 1, dtype=np.int64))
            start = x
        prev = x
    if prev - start + 1 >= min_frames:
        runs.append(np.arange(start, prev + 1, dtype=np.int64))
    return runs


def _convert_hand_arrays(
    poses: np.ndarray,
    trans: np.ndarray,
    valid: np.ndarray,
    side: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    orient = poses[:, :3].astype(np.float32)
    finger = poses[:, 3:48].astype(np.float32)
    trans = trans.astype(np.float32)
    valid = valid.astype(bool)
    if side == "left" and valid.any():
        o, p, tr = mirror_left_hand_params(orient[valid], finger[valid], trans[valid])
        orient = orient.copy()
        finger = finger.copy()
        trans = trans.copy()
        orient[valid] = o
        finger[valid] = p
        trans[valid] = tr
    orient_cv, trans_cv = transform_hand_rigid(
        orient,
        trans,
        valid,
        side,
        R=OPENGL_TO_OPENCV,
        t=np.zeros(3, dtype=np.float32),
    )
    return orient_cv.astype(np.float32), finger.astype(np.float32), trans_cv.astype(np.float32), valid


def _read_sequence_arrays(seq_dir: Path, dataset: str, max_frames: int = 0) -> dict[str, Any]:
    ids = _frame_ids(seq_dir)
    if max_frames:
        ids = ids[:max_frames]
    n = len(ids)
    if n == 0:
        raise ValueError(f"No meta frames in {seq_dir}")

    right_pose = np.zeros((n, 48), dtype=np.float32)
    right_trans = np.zeros((n, 3), dtype=np.float32)
    left_pose = np.zeros((n, 48), dtype=np.float32)
    left_trans = np.zeros((n, 3), dtype=np.float32)
    right_valid = np.zeros(n, dtype=bool)
    left_valid = np.zeros(n, dtype=bool)
    obj_rot = np.zeros((n, 3), dtype=np.float32)
    obj_trans = np.zeros((n, 3), dtype=np.float32)
    obj_valid = np.zeros(n, dtype=bool)
    cam_k = None
    betas = np.zeros(10, dtype=np.float32)
    obj_name = ""

    for i, fid in enumerate(ids):
        anno = _read_pickle(seq_dir / "meta" / f"{fid}.pkl")
        cam_raw = anno.get("camMat")
        if cam_raw is None or np.asarray(cam_raw).size != 9:
            continue
        cam_k = np.asarray(cam_raw, dtype=np.float32).reshape(3, 3)
        betas = _to_vec(anno.get("handBeta"), 10)
        obj_name = str(anno.get("objName", obj_name))

        if anno.get("objRot") is not None and anno.get("objTrans") is not None:
            obj_rot[i] = _to_vec(anno["objRot"], 3)
            obj_trans[i] = _to_vec(anno["objTrans"], 3)
            obj_valid[i] = np.isfinite(obj_rot[i]).all() and np.isfinite(obj_trans[i]).all()

        if dataset == "ho3d":
            if anno.get("handPose") is not None and anno.get("handTrans") is not None:
                right_pose[i] = _to_vec(anno["handPose"], 48)
                right_trans[i] = _to_vec(anno["handTrans"], 3)
                right_valid[i] = np.isfinite(right_pose[i]).all() and np.isfinite(right_trans[i]).all()
        else:
            if anno.get("rightHandPose") is not None and anno.get("rightHandTrans") is not None:
                right_pose[i] = _to_vec(anno["rightHandPose"], 48)
                right_trans[i] = _to_vec(anno["rightHandTrans"], 3)
                right_valid[i] = np.isfinite(right_pose[i]).all() and np.isfinite(right_trans[i]).all()
            if anno.get("leftHandPose") is not None and anno.get("leftHandTrans") is not None:
                left_pose[i] = _to_vec(anno["leftHandPose"], 48)
                left_trans[i] = _to_vec(anno["leftHandTrans"], 3)
                left_valid[i] = np.isfinite(left_pose[i]).all() and np.isfinite(left_trans[i]).all()

    if cam_k is None or not obj_name:
        raise ValueError(f"Missing camera/object metadata in {seq_dir}")

    right = _convert_hand_arrays(right_pose, right_trans, right_valid, "right")
    left = _convert_hand_arrays(left_pose, left_trans, left_valid, "left")
    return {
        "frame_ids": ids,
        "camera_K": cam_k,
        "betas": betas.astype(np.float32),
        "object_name": obj_name,
        "object_rot": obj_rot,
        "object_trans": obj_trans,
        "object_valid": obj_valid,
        "right": right,
        "left": left,
    }


def _write_video(seq_dir: Path, frame_ids: list[str], frames: np.ndarray, out_path: Path) -> bool:
    try:
        import cv2
    except Exception:
        return False
    first = None
    for f in frames:
        fid = frame_ids[int(f)]
        for ext in (".jpg", ".png"):
            p = seq_dir / "rgb" / f"{fid}{ext}"
            if p.exists():
                first = cv2.imread(str(p))
                break
        if first is not None:
            break
    if first is None:
        return False
    h, w = first.shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    if not writer.isOpened():
        return False
    for f in frames:
        fid = frame_ids[int(f)]
        img = None
        for ext in (".jpg", ".png"):
            p = seq_dir / "rgb" / f"{fid}{ext}"
            if p.exists():
                img = cv2.imread(str(p))
                break
        if img is None:
            img = np.zeros_like(first)
        writer.write(img)
    writer.release()
    return out_path.exists() and out_path.stat().st_size > 0


def run_download_step(args) -> None:
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    if args.dataset != "ho3d":
        print("H2O-3D official releases are shared as OneDrive folders:")
        for name in ("h2o3d_v1", "h2o3d_v1_folder"):
            print(f"  {name}: {OFFICIAL_LINKS[name]}")
        print("Automatic H2O-3D download is not implemented because these OneDrive links require interactive access from some environments.")
        print(f"Stage H2O-3D under {H2O3D_DIR}, then rerun with --start-from convert.")
        return

    try:
        import kagglehub
    except ImportError as e:
        raise SystemExit("HO-3D download needs kagglehub (pip install kagglehub), or stage HO3D_v3 manually.") from e

    path = Path(kagglehub.dataset_download(args.kaggle_dataset))
    src = path / "HO3D_v3"
    if not src.exists():
        candidates = sorted(path.glob("**/HO3D_v3"))
        if candidates:
            src = candidates[0]
    if not src.exists():
        raise FileNotFoundError(f"Kaggle dataset downloaded to {path}, but HO3D_v3 was not found")

    dest = root / "HO3D_v3"
    if dest.exists():
        print(f"HO-3D already staged at {dest}")
        return
    try:
        dest.symlink_to(src, target_is_directory=True)
        print(f"Staged HO-3D symlink: {dest} -> {src}")
    except OSError:
        print(f"Symlink failed; copying HO-3D tree to {dest}")
        shutil.copytree(src, dest)


def _convert_sequence(seq_dir: Path, dataset: str, ycb_root: Path, args) -> list[str]:
    arrays = _read_sequence_arrays(seq_dir, dataset, args.max_frames)
    frame_ids: list[str] = arrays["frame_ids"]
    hand_active = arrays["right"][3] | arrays["left"][3]
    valid = hand_active & arrays["object_valid"]
    runs = _split_valid_runs(valid, args.min_frames)
    if args.max_runs:
        runs = runs[: args.max_runs]
    if not runs:
        return []

    mesh_path = _find_mesh(ycb_root, arrays["object_name"])
    verts0, faces = load_mesh(str(mesh_path))
    verts0 = verts0.astype(np.float32)
    faces = faces.astype(np.int32)
    out_root = Path(args.out_root or RESULTS_DIR)
    segments = []
    for run_i, frames in enumerate(runs):
        segment_name = f"{dataset}_{seq_dir.name}_{_safe_name(arrays['object_name'])}_r{run_i:03d}"
        canonical_dir = out_root / segment_name / "canonical"
        canonical_path = canonical_dir / "canonical.npz"
        if canonical_path.exists() and not args.force:
            segments.append(segment_name)
            continue

        obj_R_gl = Rotation.from_rotvec(arrays["object_rot"][frames]).as_matrix().astype(np.float32)
        obj_R_cv = np.einsum("ij,tjk->tik", OPENGL_TO_OPENCV, obj_R_gl).astype(np.float32)
        obj_t_cv = (arrays["object_trans"][frames] @ OPENGL_TO_OPENCV.T).astype(np.float32)
        obj_rotation = rotmat_to_aa(obj_R_cv).astype(np.float32)
        obj_verts = np.empty((len(frames), verts0.shape[0], 3), dtype=np.float32)
        for i in range(len(frames)):
            obj_verts[i] = verts0 @ obj_R_cv[i].T + obj_t_cv[i]

        save = {
            "frame_indices": np.array([int(frame_ids[int(f)]) for f in frames], dtype=np.int64),
            "fps": np.float32(FPS),
            "obj_canonical_verts": verts0,
            "obj_faces": faces,
            "obj_rotation": obj_rotation,
            "obj_translation": obj_t_cv,
            "obj_scale": np.ones(3, dtype=np.float32),
            "obj_verts": obj_verts,
            "camera_K": arrays["camera_K"],
        }
        gravity = recorded_scene_gravity().get(segment_name)
        if gravity is not None:
            save["gravity_direction"] = np.asarray(gravity, dtype=np.float32)
        for side in ("right", "left"):
            orient, pose, trans, side_valid = arrays[side]
            run_valid = side_valid[frames]
            save[f"hand_{side}_orient"] = orient[frames].copy()
            save[f"hand_{side}_pose"] = pose[frames].copy()
            save[f"hand_{side}_cam_t"] = trans[frames].copy()
            save[f"hand_{side}_betas"] = arrays["betas"]
            save[f"hand_{side}_valid"] = run_valid.copy()
            if run_valid.any():
                hv = mano_fk_verts(
                    save[f"hand_{side}_orient"],
                    save[f"hand_{side}_pose"],
                    arrays["betas"],
                    save[f"hand_{side}_cam_t"],
                    side,
                    str(MANO_DIR),
                )
                hv[~run_valid] = 0
            else:
                hv = np.zeros((len(frames), 778, 3), dtype=np.float32)
            save[f"hand_{side}_verts"] = hv.astype(np.float32)

        canonical_dir.mkdir(parents=True, exist_ok=True)
        video_path = out_root / segment_name / "video.mp4"
        has_scene = False if args.no_video else _write_video(seq_dir, frame_ids, frames, video_path)
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
            raise ValueError(f"{segment_name} convention validation failed: {report.errors[:3]}")

        metadata = {
            "source_dataset": dataset,
            "segment_id": segment_name,
            "confidence_tier": "gt_fitted",
            "has_depth": (seq_dir / "depth").is_dir(),
            "has_scene": has_scene,
            "has_gravity": gravity is not None,
            "has_support_plane": False,
            "has_native_hand_pose": True,
            "has_native_object_pose": True,
            "object_is_rigid": True,
            "primary_object_id": arrays["object_name"],
            "primary_object_category": arrays["object_name"],
            "bimanual": bool(save["hand_left_valid"].any() and save["hand_right_valid"].any()),
            "viewpoint": "exocentric",
            "frame": "opencv_camera_from_official_opengl",
            **({"gravity_source": "scene_plane_consensus"} if gravity is not None else {}),
            "hand_source": "gt_fitted",
            "object_source": "gt_6d_pose",
            "depth_source": "rgbd_depth",
            "mano_convention": "right_model_x_mirror",
            "split": seq_dir.parent.name,
            "sequence": seq_dir.name,
            "n_frames": int(len(frames)),
            "fps": FPS,
            "active_sides": [s for s in ("left", "right") if save[f"hand_{s}_valid"].any()],
            "obj_mesh_path": str(mesh_path.relative_to(ROOT)) if mesh_path.is_relative_to(ROOT) else str(mesh_path),
        }
        with open(canonical_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        segments.append(segment_name)
    return segments


def run_convert_step(args) -> list[str]:
    root = _resolve_dataset_root(Path(args.root), args.dataset, args.split)
    if not root.exists():
        print(f"ERROR: {args.dataset} root not found: {root}")
        return []
    ycb_root = Path(args.ycb_models_root)
    if not ycb_root.exists():
        print(f"ERROR: YCB model root not found: {ycb_root}")
        return []

    seqs = _discover_sequences(root, args.split)
    if args.sequence:
        seqs = [p for p in seqs if args.sequence in p.name]
    if args.max_seqs:
        seqs = seqs[: args.max_seqs]
    if not seqs:
        print(f"No {args.dataset} sequences found under {root}/{args.split}")
        return []

    print(f"Converting {len(seqs)} {args.dataset} sequences")
    segments: list[str] = []
    failed = 0
    for seq in seqs:
        try:
            new_segments = _convert_sequence(seq, args.dataset, ycb_root, args)
            segments.extend(new_segments)
            print(f"  {seq.parent.name}/{seq.name}: {len(new_segments)} segments")
        except Exception as e:
            failed += 1
            print(f"  ERROR {seq}: {e}")
            traceback.print_exc()
    print(f"  Convert: {len(segments)} segments, {failed} failed sequences")
    return segments


def run_validate_step(args) -> None:
    out_root = Path(args.out_root or RESULTS_DIR)
    paths = sorted(out_root.glob(f"{args.dataset}_*/canonical/canonical.npz"))
    if args.sequence:
        paths = [p for p in paths if args.sequence in str(p)]
    if not paths:
        print(f"No {args.dataset} canonicals found under {out_root}")
        return
    passed = failed = warned = 0
    for p in paths:
        report = validate_canonical_conventions(
            p,
            max_fk_frames=None if args.max_validate_frames <= 0 else args.max_validate_frames,
            strict_camera=True,
        )
        if report.ok:
            passed += 1
            warned += int(bool(report.warnings))
            if report.warnings and args.print_warnings:
                print(f"  WARN {p.parent.parent.name}: {report.warnings[:2]}")
        else:
            failed += 1
            print(f"  FAIL {p.parent.parent.name}: {report.errors[:2]}")
    print(f"  Validate: {passed} passed, {failed} failed, {warned} warned ({len(paths)} total)")
    if failed:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m gtf.datasets.ho3d", description="Ingest HO-3D/H2O-3D into canonical format")
    parser.add_argument("--start-from", choices=PIPELINE_STEPS, default="download")
    parser.add_argument("--dataset", choices=["ho3d", "h2o3d"], default="ho3d")
    parser.add_argument("--root", type=str, default="")
    parser.add_argument("--out-root", type=str, default="")
    parser.add_argument("--ycb-models-root", type=str, default=str(YCB_MODELS_DIR))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--sequence", type=str, default="")
    parser.add_argument("--max-seqs", type=int, default=0)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--min-frames", type=int, default=30)
    parser.add_argument("--max-validate-frames", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--download-only", action="store_true", help="Stop after the download step")
    parser.add_argument("--kaggle-dataset", type=str, default=HO3D_KAGGLE_DATASET)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--print-warnings", action="store_true")
    args = parser.parse_args()

    if not args.root:
        args.root = str(HO3D_DIR if args.dataset == "ho3d" else H2O3D_DIR)
    start_idx = PIPELINE_STEPS.index(args.start_from)
    active = set(PIPELINE_STEPS[start_idx:])
    if "download" in active:
        run_download_step(args)
        if args.download_only:
            return
    if "convert" in active:
        run_convert_step(args)
    if "validate" in active:
        run_validate_step(args)


if __name__ == "__main__":
    main()
