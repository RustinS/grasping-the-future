#!/usr/bin/env python3
"""Ingest H2O into model-ready canonical.npz files.

H2O provides true MANO_LEFT/MANO_RIGHT parameters, per-frame object 6D poses,
camera poses, RGB-D, object meshes, and interaction labels. This loader writes
camera-frame canonicals under results/, each in the static egocentric camera of
its middle frame.

Expected H2O layout, per sequence (official viewer layout):
    subject1/h1/0/
      cam4/
        cam_intrinsics.txt
        cam_pose/000000.txt          # camera-to-world, 4x4
        hand_pose_mano/000000.txt    # true left/right MANO params
        obj_pose_rt/000000.txt       # object index + 4x4 object-to-cam
        rgb/000000.png               # optional
        depth/000000.png             # optional

Download the H2O archives (object.zip, label_split.zip, subject{1..4}_pose_v1_1.tar.gz
and the RGB-D frames) from https://h2odataset.ethz.ch after registering, into --root;
``--start-from extract`` unpacks them there.

Usage:
    python -m gtf.datasets.h2o --start-from extract --root datasets/h2o
    python -m gtf.datasets.h2o --start-from convert --root datasets/h2o
"""

from __future__ import annotations

import argparse
import json
import re
import tarfile
import traceback
import zipfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from gtf.paths import MANO_DIR, REPO_ROOT
from gtf.datasets.validate_conventions import validate_canonical_conventions
from gtf.utils.hand_rigid import transform_hand_rigid
from gtf.utils import RESULTS_DIR, rprint as print
from gtf.utils.canonical import (
    estimate_gravity,
    load_mesh,
    mano_fk_verts,
    mirror_left_hand_params,
    rotmat_to_aa,
    validate_canonical_npz,
)

ROOT = REPO_ROOT

PIPELINE_STEPS = ["extract", "convert", "validate"]

H2O_DIR = ROOT / "datasets" / "h2o"
OBJECT_ROOT_NAMES = ["object", "objects", "object_models", "models"]  # where the meshes may sit under the dataset root
FPS = 30.0

# Object index mapping from the official h2oplayer class_names list.
H2O_CLASSES = [
    "background",
    "book",
    "espresso",
    "lotion",
    "spray",
    "milk",
    "cocoa",
    "pringles",
    "cappuccino",
]

H2O_ARCHIVES = (
    "object.zip",
    "label_split.zip",
    "subject1_pose_v1_1.tar.gz",
    "subject2_pose_v1_1.tar.gz",
    "subject3_pose_v1_1.tar.gz",
    "subject4_pose_v1_1.tar.gz",
)

H2O_CATEGORY = {
    "book": "book",
    "espresso": "box",
    "lotion": "bottle",
    "lotion_spray": "spray",
    "milk": "carton",
    "ovomaltine": "can",
    "pringles": "can",
    "starbucks": "cup",
}


def _safe_name(text: str) -> str:
    text = text.strip().replace("/", "_")
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("_")


def _loadtxt(path: Path) -> np.ndarray:
    return np.loadtxt(str(path), dtype=np.float64).reshape(-1)


def _extract_h2o_archive(path: Path, root: Path, force: bool = False) -> None:
    done = root / f".{path.name}.extracted"
    if done.exists() and not force:
        return
    top_levels: set[str] = set()
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            top_levels = {n.split("/", 1)[0] for n in zf.namelist() if n and not n.startswith(".")}
            if top_levels and all((root / name).exists() for name in top_levels):
                done.touch()
                return
            zf.extractall(root)
    elif path.name.endswith(".tar.gz"):
        with tarfile.open(path) as tf:
            top_levels = {m.name.split("/", 1)[0] for m in tf.getmembers() if m.name and not m.name.startswith(".")}
            if top_levels and all((root / name).exists() for name in top_levels):
                done.touch()
                return
            tf.extractall(root)
    else:
        return
    done.touch()


def _read_cam_pose(path: Path) -> np.ndarray:
    vals = _loadtxt(path)
    if vals.size != 16:
        raise ValueError(f"{path} expected 16 values, got {vals.size}")
    return vals.reshape(4, 4).astype(np.float32)


def _read_intrinsics(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    vals = _loadtxt(path)
    if vals.size < 4:
        return None
    fx, fy, cx, cy = vals[:4]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _read_object_pose(path: Path) -> tuple[int, np.ndarray] | None:
    if not path.exists():
        return None
    vals = _loadtxt(path)
    if vals.size < 17:
        return None
    obj_idx = int(round(float(vals[0])))
    T = vals[1:17].reshape(4, 4).astype(np.float32)
    return obj_idx, T


def _read_hand_pose(path: Path) -> dict[str, dict[str, np.ndarray | bool]]:
    """Read one H2O hand_pose_mano file.

    Layout from h2oplayer:
      left:  flag[0], trans[1:4], pose[4:52], betas[52:62]
      right: flag[62], trans[63:66], pose[66:114], betas[114:124]
    Poses use the true MANO side models; the caller mirrors the left hand into
    the right-model x-mirror convention.
    """
    out = {
        "left": {"valid": False},
        "right": {"valid": False},
    }
    if not path.exists():
        return out
    vals = _loadtxt(path).astype(np.float32)
    if vals.size < 124:
        return out

    left_valid = bool(vals[0] != 0)
    if left_valid:
        out["left"] = {
            "valid": True,
            "trans": vals[1:4].astype(np.float32),
            "orient": vals[4:7].astype(np.float32),
            "pose": vals[7:52].astype(np.float32),
            "betas": vals[52:62].astype(np.float32),
        }

    right_valid = bool(vals[62] != 0)
    if right_valid:
        out["right"] = {
            "valid": True,
            "trans": vals[63:66].astype(np.float32),
            "orient": vals[66:69].astype(np.float32),
            "pose": vals[69:114].astype(np.float32),
            "betas": vals[114:124].astype(np.float32),
        }
    return out


def _discover_sequence_dirs(root: Path) -> list[Path]:
    seqs = []
    for cam4 in sorted(root.rglob("cam4")):
        if not cam4.is_dir():
            continue
        if (cam4 / "hand_pose_mano").is_dir() and (cam4 / "obj_pose_rt").is_dir() and (cam4 / "cam_pose").is_dir():
            seqs.append(cam4.parent)
    return seqs


def _object_name(obj_idx: int) -> str:
    if 0 <= obj_idx < len(H2O_CLASSES):
        return H2O_CLASSES[obj_idx]
    return f"object_{obj_idx:02d}"


def _find_object_mesh(object_root: Path, obj_name: str) -> Path:
    mesh_aliases = {
        "spray": ["lotion_spray", "spray"],
        "pringles": ["pringles", "chips"],
        "cappuccino": ["cappuccino", "starbucks"],
    }
    names = [obj_name, *mesh_aliases.get(obj_name, [])]
    candidates = []
    for name in dict.fromkeys(names):
        for ext in (".obj", ".ply"):
            candidates.extend(
                [
                    object_root / obj_name / f"{name}{ext}",
                    object_root / name / f"{name}{ext}",
                    object_root / f"{name}{ext}",
                    object_root / obj_name / f"{name}_cm{ext}",
                    object_root / name / f"{name}_cm{ext}",
                ]
            )
    for p in candidates:
        if p.exists():
            return p
    for name in dict.fromkeys(names):
        matches = sorted(object_root.glob(f"**/{name}.obj")) + sorted(object_root.glob(f"**/{name}.ply"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No mesh for {obj_name} under {object_root}")


def _resolve_object_root(args) -> Path:
    if args.object_root:
        return Path(args.object_root)
    candidates = [Path(args.root) / name for name in OBJECT_ROOT_NAMES]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def _group_frames_by_object(seq_dir: Path, min_frames: int) -> list[tuple[int, list[int]]]:
    pose_dir = seq_dir / "cam4" / "obj_pose_rt"
    items: list[tuple[int, int]] = []
    for p in sorted(pose_dir.glob("*.txt")):
        try:
            frame = int(p.stem)
        except ValueError:
            continue
        obj = _read_object_pose(p)
        if obj is None:
            continue
        obj_idx, _T = obj
        if obj_idx <= 0:
            continue
        items.append((frame, obj_idx))

    groups: list[tuple[int, list[int]]] = []
    cur_obj = None
    cur_frames: list[int] = []
    prev_frame = None
    for frame, obj_idx in items:
        contiguous = prev_frame is None or frame == prev_frame + 1
        if cur_obj is None or obj_idx != cur_obj or not contiguous:
            if cur_obj is not None and len(cur_frames) >= min_frames:
                groups.append((cur_obj, cur_frames))
            cur_obj = obj_idx
            cur_frames = [frame]
        else:
            cur_frames.append(frame)
        prev_frame = frame
    if cur_obj is not None and len(cur_frames) >= min_frames:
        groups.append((cur_obj, cur_frames))
    return groups


def _estimate_anchor_gravity(seq_dir: Path, anchor_frame: int) -> tuple[np.ndarray | None, float | None]:
    import cv2

    cam4 = seq_dir / "cam4"
    K = _read_intrinsics(cam4 / "cam_intrinsics.txt")
    if K is None:
        return None, None
    depth_path = cam4 / "depth" / f"{anchor_frame:06d}.png"
    if not depth_path.exists():
        return None, None
    depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        return None, None
    result = estimate_gravity(depth_img.astype(np.float32) / 1000.0, K)
    if result is None:
        return None, None
    gravity, support = result
    return gravity.astype(np.float32), support


def _write_video(seq_dir: Path, frames: list[int], video_path: Path) -> bool:
    rgb_dir = seq_dir / "cam4" / "rgb"
    if not rgb_dir.exists():
        return False
    try:
        import cv2
    except Exception:
        return False

    first = None
    for f in frames:
        for ext in (".png", ".jpg", ".jpeg"):
            p = rgb_dir / f"{f:06d}{ext}"
            if p.exists():
                first = cv2.imread(str(p))
                break
        if first is not None:
            break
    if first is None:
        return False

    video_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    if not writer.isOpened():
        return False
    for f in frames:
        img = None
        for ext in (".png", ".jpg", ".jpeg"):
            p = rgb_dir / f"{f:06d}{ext}"
            if p.exists():
                img = cv2.imread(str(p))
                break
        if img is None:
            img = np.zeros_like(first)
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        writer.write(img)
    writer.release()
    return video_path.exists() and video_path.stat().st_size > 0


def _convert_group(seq_dir: Path, obj_idx: int, frames: list[int], group_idx: int, args, object_root: Path) -> str | None:
    cam4 = seq_dir / "cam4"
    obj_name = _object_name(obj_idx)
    obj_category = H2O_CATEGORY.get(obj_name, obj_name)
    rel_seq = _safe_name(str(seq_dir.relative_to(Path(args.root))))
    segment_name = f"h2o_{rel_seq}_{obj_name}_g{group_idx:03d}"
    seg_dir = Path(args.out_root or RESULTS_DIR) / segment_name
    canonical_dir = seg_dir / "canonical"
    canonical_path = canonical_dir / "canonical.npz"
    if canonical_path.exists() and not args.force:
        return segment_name

    mesh_path = _find_object_mesh(object_root, obj_name)
    obj_canonical_verts, obj_faces = load_mesh(str(mesh_path))
    obj_canonical_verts = obj_canonical_verts.astype(np.float32)
    obj_faces = obj_faces.astype(np.int32)

    anchor_frame = frames[len(frames) // 2]
    T_anchor_world = np.linalg.inv(_read_cam_pose(cam4 / "cam_pose" / f"{anchor_frame:06d}.txt")).astype(np.float32)

    T = len(frames)
    hand_orient = {s: np.zeros((T, 3), dtype=np.float32) for s in ("left", "right")}
    hand_pose = {s: np.zeros((T, 45), dtype=np.float32) for s in ("left", "right")}
    hand_cam_t = {s: np.zeros((T, 3), dtype=np.float32) for s in ("left", "right")}
    hand_valid = {s: np.zeros(T, dtype=bool) for s in ("left", "right")}
    beta_samples = {s: [] for s in ("left", "right")}

    obj_rotation = np.zeros((T, 3), dtype=np.float32)
    obj_translation = np.zeros((T, 3), dtype=np.float32)

    for i, frame in enumerate(frames):
        T_current_world = _read_cam_pose(cam4 / "cam_pose" / f"{frame:06d}.txt")
        T_anchor_current = (T_anchor_world @ T_current_world).astype(np.float32)
        R_ac = T_anchor_current[:3, :3]
        t_ac = T_anchor_current[:3, 3]

        obj = _read_object_pose(cam4 / "obj_pose_rt" / f"{frame:06d}.txt")
        if obj is None or obj[0] != obj_idx:
            raise ValueError(f"missing object pose for frame {frame}")
        T_obj_cam = obj[1]
        T_obj_anchor = (T_anchor_current @ T_obj_cam).astype(np.float32)
        obj_rotation[i] = rotmat_to_aa(T_obj_anchor[:3, :3].reshape(1, 3, 3))[0]
        obj_translation[i] = T_obj_anchor[:3, 3]

        hands = _read_hand_pose(cam4 / "hand_pose_mano" / f"{frame:06d}.txt")
        for side in ("left", "right"):
            h = hands[side]
            if not h.get("valid", False):
                continue
            orient = np.asarray(h["orient"], dtype=np.float32).reshape(1, 3)
            pose = np.asarray(h["pose"], dtype=np.float32).reshape(1, 45)
            trans = np.asarray(h["trans"], dtype=np.float32).reshape(1, 3)

            if side == "left":
                # H2O is true MANO_LEFT, so mirror orient, fingers, and trans.
                orient, pose, trans = mirror_left_hand_params(orient, pose, trans)

            orient_t, trans_t = transform_hand_rigid(
                orient,
                trans,
                np.array([True]),
                side,
                R=R_ac,
                t=t_ac,
            )
            hand_orient[side][i] = orient_t[0]
            hand_pose[side][i] = pose[0]
            hand_cam_t[side][i] = trans_t[0]
            hand_valid[side][i] = True
            beta_samples[side].append(np.asarray(h["betas"], dtype=np.float32))

    betas = {}
    for side in ("left", "right"):
        if beta_samples[side]:
            betas[side] = np.median(np.stack(beta_samples[side], axis=0), axis=0).astype(np.float32)
        else:
            betas[side] = np.zeros(10, dtype=np.float32)

    if not (hand_valid["left"].any() or hand_valid["right"].any()):
        return None

    hand_verts = {}
    for side in ("left", "right"):
        if hand_valid[side].any():
            hand_verts[side] = mano_fk_verts(hand_orient[side], hand_pose[side], betas[side], hand_cam_t[side], side, str(MANO_DIR)).astype(np.float32)
            hand_verts[side][~hand_valid[side]] = 0
        else:
            hand_verts[side] = np.zeros((T, 778, 3), dtype=np.float32)

    obj_scale = np.ones(3, dtype=np.float32)
    obj_verts = np.zeros((T, obj_canonical_verts.shape[0], 3), dtype=np.float32)
    for i in range(T):
        R_i = Rotation.from_rotvec(obj_rotation[i]).as_matrix().astype(np.float32)
        obj_verts[i] = obj_canonical_verts @ R_i.T + obj_translation[i]

    gravity_direction, support_plane_height = _estimate_anchor_gravity(seq_dir, anchor_frame)
    camera_K = _read_intrinsics(cam4 / "cam_intrinsics.txt")

    canonical_dir.mkdir(parents=True, exist_ok=True)
    video_path = seg_dir / "video.mp4"
    has_scene = False
    if not args.no_video:
        has_scene = _write_video(seq_dir, frames, video_path)

    save = {
        "frame_indices": np.array(frames, dtype=np.int64),
        "fps": np.float32(FPS),
        "obj_canonical_verts": obj_canonical_verts,
        "obj_faces": obj_faces,
        "obj_rotation": obj_rotation,
        "obj_translation": obj_translation,
        "obj_scale": obj_scale,
        "obj_verts": obj_verts,
    }
    if camera_K is not None:
        save["camera_K"] = camera_K.astype(np.float32)
    for side in ("left", "right"):
        save[f"hand_{side}_orient"] = hand_orient[side]
        save[f"hand_{side}_pose"] = hand_pose[side]
        save[f"hand_{side}_cam_t"] = hand_cam_t[side]
        save[f"hand_{side}_betas"] = betas[side]
        save[f"hand_{side}_valid"] = hand_valid[side]
        save[f"hand_{side}_verts"] = hand_verts[side]
    if gravity_direction is not None:
        save["gravity_direction"] = gravity_direction.astype(np.float32)
    if support_plane_height is not None:
        save["support_plane_height"] = np.float32(support_plane_height)

    np.savez_compressed(str(canonical_path), **save)
    ok, schema_errors = validate_canonical_npz(str(canonical_path))
    if not ok:
        raise ValueError(f"{segment_name} schema validation failed: {schema_errors[:3]}")

    active = [s for s in ("left", "right") if hand_valid[s].any()]
    metadata = {
        "source_dataset": "h2o",
        "segment_id": segment_name,
        "confidence_tier": "gt_fitted",
        "has_depth": (cam4 / "depth" / f"{anchor_frame:06d}.png").exists(),
        "has_scene": has_scene,
        "has_gravity": gravity_direction is not None,
        "has_support_plane": support_plane_height is not None,
        "has_native_hand_pose": True,
        "has_native_object_pose": True,
        "object_is_rigid": True,
        "primary_object_id": obj_name,
        "primary_object_category": obj_category,
        "bimanual": len(active) > 1,
        "viewpoint": "egocentric",
        "frame": "anchor_camera_static",
        "hand_source": "gt_fitted",
        "object_source": "gt_6d_pose",
        "depth_source": "h2o_rgbd" if (cam4 / "depth").exists() else "none",
        "mano_convention": "right_model_x_mirror",
        "h2o_sequence": str(seq_dir.relative_to(Path(args.root))),
        "h2o_object_index": int(obj_idx),
        "anchor_frame": int(anchor_frame),
        "n_frames": int(T),
        "fps": FPS,
        "active_sides": active,
        "obj_mesh_path": str(mesh_path.relative_to(ROOT)) if mesh_path.is_relative_to(ROOT) else str(mesh_path),
    }
    with open(canonical_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    report = validate_canonical_conventions(
        canonical_path,
        max_fk_frames=None if args.max_validate_frames <= 0 else args.max_validate_frames,
        strict_camera=True,
    )
    if not report.ok:
        raise ValueError(f"{segment_name} convention validation failed: {report.errors[:3]}")
    if report.warnings and args.print_warnings:
        print(f"  WARN {segment_name}: {report.warnings[:2]}")

    return segment_name


def run_extract_step(args) -> None:
    root = Path(args.root)
    missing = [name for name in H2O_ARCHIVES if not (root / name).exists()]
    if missing:
        raise SystemExit(f"H2O archives not found in {root}: {', '.join(missing)}. Download them from https://h2odataset.ethz.ch (registration required).")
    for filename in H2O_ARCHIVES:
        print(f"Extracting H2O {filename}")
        _extract_h2o_archive(root / filename, root, force=args.force_extract)


def run_convert_step(args) -> list[str]:
    root = Path(args.root)
    object_root = _resolve_object_root(args)
    if not root.exists():
        print(f"ERROR: H2O root not found: {root}")
        return []
    if not object_root.exists():
        print(f"ERROR: H2O object root not found: {object_root}")
        return []

    seq_dirs = _discover_sequence_dirs(root)
    if args.sequence:
        wanted = args.sequence
        seq_dirs = [p for p in seq_dirs if wanted in str(p.relative_to(root))]
    if args.max_seqs:
        seq_dirs = seq_dirs[: args.max_seqs]
    if not seq_dirs:
        print(f"No H2O sequences found under {root}")
        return []

    print(f"Converting {len(seq_dirs)} H2O sequences from {root}")
    done = skipped = failed = 0
    segments: list[str] = []
    for seq_i, seq_dir in enumerate(seq_dirs):
        groups = _group_frames_by_object(seq_dir, args.min_frames)
        if args.max_groups:
            groups = groups[: args.max_groups]
        if not groups:
            skipped += 1
            continue
        print(f"  [{seq_i + 1}/{len(seq_dirs)}] {seq_dir.relative_to(root)}: {len(groups)} object groups")
        for group_idx, (obj_idx, frames) in enumerate(groups):
            try:
                seg = _convert_group(seq_dir, obj_idx, frames, group_idx, args, object_root)
                if seg:
                    segments.append(seg)
                    done += 1
                else:
                    skipped += 1
            except Exception as e:
                failed += 1
                print(f"    ERROR obj={obj_idx} group={group_idx}: {e}")
                traceback.print_exc()
    print(f"  Convert: {done} segments created, {skipped} skipped, {failed} failed")
    return segments


def run_validate_step(args) -> None:
    out_root = Path(args.out_root or RESULTS_DIR)
    paths = sorted(out_root.glob("h2o_*/canonical/canonical.npz"))
    if args.sequence:
        paths = [p for p in paths if args.sequence in str(p)]
    if not paths:
        print(f"No H2O canonicals found under {out_root}")
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
            if report.warnings:
                warned += 1
                if args.print_warnings:
                    print(f"  WARN {p.parent.parent.name}: {report.warnings[:2]}")
        else:
            failed += 1
            print(f"  FAIL {p.parent.parent.name}: {report.errors[:2]}")
    print(f"  Validate: {passed} passed, {failed} failed, {warned} warned ({len(paths)} total)")
    if failed:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m gtf.datasets.h2o", description="Ingest H2O into canonical format")
    parser.add_argument("--start-from", choices=PIPELINE_STEPS, default="convert")
    parser.add_argument("--root", type=str, default=str(H2O_DIR), help="Extracted H2O dataset root")
    parser.add_argument("--object-root", type=str, default="", help="H2O object mesh root")
    parser.add_argument("--out-root", type=str, default="", help="Output root for canonicals (default: results)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--force-extract", action="store_true", help="Re-extract the H2O archives")
    parser.add_argument("--extract-only", action="store_true", help="Stop after the extract step")
    parser.add_argument("--sequence", type=str, default="", help="Substring filter for sequence path")
    parser.add_argument("--max-seqs", type=int, default=0)
    parser.add_argument("--max-groups", type=int, default=0, help="Cap object groups per sequence")
    parser.add_argument("--min-frames", type=int, default=30)
    parser.add_argument("--max-validate-frames", type=int, default=0, help="0 = validate every valid frame")
    parser.add_argument("--no-video", action="store_true", help="Skip RGB video writing")
    parser.add_argument("--print-warnings", action="store_true")
    args = parser.parse_args()

    start_idx = PIPELINE_STEPS.index(args.start_from)
    active_steps = set(PIPELINE_STEPS[start_idx:])
    if "extract" in active_steps:
        run_extract_step(args)
        if args.extract_only:
            return
    if "convert" in active_steps:
        run_convert_step(args)
    if "validate" in active_steps:
        run_validate_step(args)


if __name__ == "__main__":
    main()
