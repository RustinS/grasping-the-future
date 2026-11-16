#!/usr/bin/env python3
"""Ingest TACO into model-ready canonical.npz files.

TACO is bimanual and usually contains two rigid objects per sequence
(``tool`` and ``target``). A canonical holds one rigid object, so this loader
writes one canonical per selected object role and records the paired object in
the metadata.

TACO hand poses are full 48D MANO axis-angle params in the world frame. Left
hands are mirrored into the ``right_model_x_mirror`` convention before the
official per-frame egocentric world-to-camera transform is applied. The
canonical keeps both the per-frame camera fields and the world fields with the
per-frame ``camera_extrinsics`` (world-to-camera).

``--start-from download`` fetches the official release from Hugging Face (leolyliu/TACO).

Usage:
    python -m gtf.datasets.taco --start-from download --include-media
    python -m gtf.datasets.taco --start-from convert
    python -m gtf.datasets.taco --start-from convert --triplet "(brush, brush, bowl)" --max-seqs 1 --out-root /abs/scratch
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import traceback
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from gtf.paths import MANO_DIR, REPO_ROOT
from gtf.datasets.validate_conventions import validate_canonical_conventions
from gtf.utils.hand_rigid import transform_hand_rigid
from gtf.utils import RESULTS_DIR, rprint as print
from gtf.utils.canonical import (
    _preload_torch_cuda_deps,
    load_mesh,
    mano_fk_verts,
    mirror_left_hand_params,
    root_trans_to_manopth_cam_t,
    rotmat_to_aa,
    validate_canonical_npz,
)

ROOT = REPO_ROOT

PIPELINE_STEPS = ["download", "convert", "validate"]

TACO_DIR = ROOT / "datasets" / "taco"

# The TACO mocap world is Z-up.
_TACO_GRAVITY_WORLD = np.array([0.0, 0.0, -1.0], dtype=np.float32)


def _gravity_direction_mid(w2c_frames: np.ndarray, g_world: np.ndarray) -> np.ndarray:
    """Camera-frame gravity at the segment's middle frame from the constant world gravity."""
    E = w2c_frames[len(w2c_frames) // 2]
    g = E[:3, :3] @ g_world
    return (g / np.linalg.norm(g)).astype(np.float32)


FPS = 30.0
# The official TACO release on Hugging Face.
HF_REPO = "leolyliu/TACO"
HF_ARCHIVES = (
    "Hand_Poses.zip",
    "Object_Poses.zip",
    "Egocentric_Camera_Parameters.zip",
    "Object_Models.zip",
)
HF_MEDIA_FILES = (
    "Egocentric_RGB_Videos.zip",
    "Egocentric_Depth_Videos.zip",
)


def _safe_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text.strip()).strip("_")


def _extract_zip_once(zip_path: Path, root: Path, force: bool = False) -> None:
    done = root / f".{zip_path.name}.extracted"
    if done.exists() and not force:
        return
    with zipfile.ZipFile(zip_path) as zf:
        top_levels = {n.split("/", 1)[0] for n in zf.namelist() if n and not n.startswith(".")}
        if top_levels and all((root / name).exists() for name in top_levels):
            done.touch()
            return
        zf.extractall(root)
    done.touch()


def _download_hf_file(repo_id: str, filename: str, root: Path) -> None:
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        local_dir=str(root),
        cache_dir=str(root / ".hf_cache"),
    )


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _read_pickle(path: Path) -> Any:
    _preload_torch_cuda_deps()
    with open(path, "rb") as f:
        return pickle.load(f)


def _read_shape(path: Path) -> np.ndarray:
    if not path.exists():
        return np.zeros(10, dtype=np.float32)
    data = _read_pickle(path)
    beta = _to_numpy(data["hand_shape"]).reshape(10)
    return beta.astype(np.float32)


def _sorted_pose_keys(data: dict[Any, Any]) -> list[Any]:
    def key_fn(k: Any) -> tuple[int, str]:
        text = str(k)
        nums = re.findall(r"\d+", text)
        return (int(nums[-1]) if nums else 0, text)

    return sorted(data.keys(), key=key_fn)


def _read_hand_pose(path: Path, n_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    orient = np.zeros((n_frames, 3), dtype=np.float32)
    pose = np.zeros((n_frames, 45), dtype=np.float32)
    trans = np.zeros((n_frames, 3), dtype=np.float32)
    valid = np.zeros(n_frames, dtype=bool)
    if not path.exists():
        return orient, pose, trans, valid

    if path.is_file():
        data = _read_pickle(path)
        items = [data[k] for k in _sorted_pose_keys(data)]
    else:
        items = [_read_pickle(p) for p in sorted(path.glob("*.pkl"))]

    count = min(n_frames, len(items))
    for i in range(count):
        item = items[i]
        theta = _to_numpy(item["hand_pose"]).reshape(48).astype(np.float32)
        tr = _to_numpy(item["hand_trans"]).reshape(3).astype(np.float32)
        if np.isfinite(theta).all() and np.isfinite(tr).all():
            orient[i] = theta[:3]
            pose[i] = theta[3:48]
            trans[i] = tr
            valid[i] = True
    return orient, pose, trans, valid


def _transform_hand_per_frame(
    orient: np.ndarray,
    pose: np.ndarray,
    trans: np.ndarray,
    valid: np.ndarray,
    side: str,
    w2c: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    orient = orient.astype(np.float32)
    pose = pose.astype(np.float32)
    trans = trans.astype(np.float32)
    valid = valid.astype(bool)
    # TACO ``hand_trans`` is the root-joint world position (manotorch center_idx=0
    # semantics); convert it to the manopth cam_t before mirroring. Right-hand FK then
    # reproduces TACO's hand_joints.npy exactly.
    trans = root_trans_to_manopth_cam_t(trans, beta, side, str(MANO_DIR))
    if side == "left" and valid.any():
        o, p, tr = mirror_left_hand_params(orient[valid], pose[valid], trans[valid])
        orient = orient.copy()
        pose = pose.copy()
        trans = trans.copy()
        orient[valid] = o
        pose[valid] = p
        trans[valid] = tr
    orient_world = orient.copy()
    trans_world = trans.copy()

    orient_out = np.zeros_like(orient)
    trans_out = np.zeros_like(trans)
    for i in np.flatnonzero(valid):
        o, tr = transform_hand_rigid(
            orient[i : i + 1],
            trans[i : i + 1],
            np.ones(1, dtype=bool),
            side,
            R=w2c[i, :3, :3],
            t=w2c[i, :3, 3],
            # Shaped pivot J0(betas): the camera fields are the exact rigid re-expression
            # of the world fields, the inverse of the present-camera re-anchor.
            betas=beta,
        )
        orient_out[i] = o[0]
        trans_out[i] = tr[0]
    return orient_out.astype(np.float32), pose.astype(np.float32), trans_out.astype(np.float32), valid, orient_world.astype(np.float32), trans_world.astype(np.float32)


def _discover_sequences(root: Path, list_file: str = "") -> list[tuple[str, str]]:
    if list_file:
        seqs: list[tuple[str, str]] = []
        for raw in Path(list_file).read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                seqs.append((parts[0], parts[1]))
            elif "/" in line:
                triplet, seq = line.rsplit("/", 1)
                seqs.append((triplet, seq))
        return seqs

    hand_root = root / "Hand_Poses"
    out = []
    if not hand_root.exists():
        return out
    for triplet_dir in sorted(p for p in hand_root.iterdir() if p.is_dir()):
        for seq_dir in sorted(p for p in triplet_dir.iterdir() if p.is_dir()):
            rel_obj = root / "Object_Poses" / triplet_dir.name / seq_dir.name
            rel_cam = root / "Egocentric_Camera_Parameters" / triplet_dir.name / seq_dir.name
            if rel_obj.is_dir() and rel_cam.is_dir():
                out.append((triplet_dir.name, seq_dir.name))
    return out


def _role_files(object_pose_dir: Path) -> dict[str, Path]:
    out = {}
    for role in ("tool", "target"):
        matches = sorted(object_pose_dir.glob(f"{role}_*.npy"))
        if matches:
            out[role] = matches[0]
    return out


def _object_name_from_pose(path: Path, role: str) -> str:
    stem = path.stem
    prefix = f"{role}_"
    return stem[len(prefix) :] if stem.startswith(prefix) else stem


def _find_object_mesh(object_root: Path, object_name: str) -> Path:
    candidates = [
        object_root / f"{object_name}_cm.obj",
        object_root / f"{object_name}.obj",
        object_root / object_name / f"{object_name}_cm.obj",
        object_root / object_name / f"{object_name}.obj",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"No TACO object mesh found for {object_name} under {object_root}")


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


def _camera_z_is_plausible(
    obj_cam: np.ndarray,
    verts0: np.ndarray,
    hand_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    frames: np.ndarray,
) -> tuple[bool, float, float]:
    stride = max(1, len(verts0) // 2048)
    verts_sample = verts0[::stride]
    obj_sample = (
        np.einsum("vj,tij->tvi", verts_sample, obj_cam[:, :3, :3])
        + obj_cam[:, None, :3, 3]
    )
    z_samples = [obj_sample[..., 2].reshape(-1)]
    for _side, (_orient, _pose, trans, _beta, valid) in hand_data.items():
        run_valid = valid[frames]
        if run_valid.any():
            z_samples.append(trans[frames][run_valid, 2])
    z = np.concatenate(z_samples)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return False, 0.0, float("nan")
    frac_in_front = float((z > 0.02).mean())
    median_z = float(np.median(z))
    return frac_in_front >= 0.90 and 0.05 <= median_z <= 8.0, frac_in_front, median_z


def _write_video(video_path: Path, frames: np.ndarray, out_path: Path) -> bool:
    if not video_path.exists():
        return False
    try:
        import cv2
    except Exception:
        return False

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    media_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    mapped_frames = _map_pose_frames_to_media(video_path, frames, media_frames)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height))
    if not writer.isOpened():
        cap.release()
        return False
    last_img = None
    for f in mapped_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if not ok:
            img = last_img if last_img is not None else np.zeros((height, width, 3), dtype=np.uint8)
        writer.write(img)
        last_img = img
    writer.release()
    cap.release()
    return out_path.exists() and out_path.stat().st_size > 0


def _pose_frame_count_for_media(video_path: Path) -> int | None:
    """Return TACO pose/camera frame count matching an egocentric RGB video."""
    try:
        rel = video_path.resolve().relative_to((TACO_DIR / "Egocentric_RGB_Videos").resolve())
    except ValueError:
        return None
    if len(rel.parts) < 3:
        return None
    triplet, sequence = rel.parts[0], rel.parts[1]
    cam_path = TACO_DIR / "Egocentric_Camera_Parameters" / triplet / sequence / "egocentric_frame_extrinsic.npy"
    if not cam_path.exists():
        return None
    try:
        return int(np.load(str(cam_path), mmap_mode="r").shape[0])
    except Exception:
        return None


def _map_pose_frames_to_media(video_path: Path, frames: np.ndarray, media_frames: int) -> np.ndarray:
    """Pose frame ids -> frames of a media file, rescaled when the media has fewer frames than the poses."""
    frames = np.asarray(frames, dtype=np.int64)
    if frames.size == 0:
        return frames
    if media_frames <= 0:
        return frames
    pose_frames = _pose_frame_count_for_media(video_path)
    if pose_frames is not None and pose_frames > 1 and media_frames < pose_frames:
        scale = float(media_frames - 1) / float(pose_frames - 1)
        mapped = np.rint(frames.astype(np.float64) * scale).astype(np.int64)
    else:
        mapped = frames
    return np.clip(mapped, 0, int(media_frames) - 1)


def run_download_step(args) -> None:
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    files = (*HF_ARCHIVES, *(HF_MEDIA_FILES if args.include_media else ()))
    for filename in files:
        print(f"Downloading TACO {filename}")
        _download_hf_file(HF_REPO, filename, root)
    for filename in files:
        print(f"Extracting TACO {filename}")
        _extract_zip_once(root / filename, root, force=args.force_download)


def _resolve_object_root(root: Path, object_model_root: str) -> Path:
    if object_model_root:
        return Path(object_model_root)
    for rel in ("Object_Models", "object_models", "models"):
        p = root / rel
        if p.exists():
            return p
    return root / "Object_Models"


def _convert_sequence(root: Path, object_root: Path, triplet: str, sequence: str, args) -> list[str]:
    hand_dir = root / "Hand_Poses" / triplet / sequence
    object_pose_dir = root / "Object_Poses" / triplet / sequence
    roles = _role_files(object_pose_dir)
    if not roles:
        raise FileNotFoundError(f"No tool_/target_ poses in {object_pose_dir}")
    first_pose = np.load(next(iter(roles.values()))).astype(np.float32)

    camera_dir = root / "Egocentric_Camera_Parameters" / triplet / sequence
    K = np.loadtxt(camera_dir / "egocentric_intrinsic.txt").astype(np.float32).reshape(3, 3)
    w2c = np.load(camera_dir / "egocentric_frame_extrinsic.npy").astype(np.float32)
    n_frames = min(first_pose.shape[0], w2c.shape[0])
    has_depth = (root / "Egocentric_Depth_Videos" / triplet / sequence / "egocentric_depth.avi").exists()
    video_src = root / "Egocentric_RGB_Videos" / triplet / sequence / "color.mp4"

    right_beta = _read_shape(hand_dir / "right_hand_shape.pkl")
    left_beta = _read_shape(hand_dir / "left_hand_shape.pkl")
    hand_data = {}
    hand_world = {}
    for side in ("right", "left"):
        beta = right_beta if side == "right" else left_beta
        orient, pose, trans, valid = _read_hand_pose(hand_dir / f"{side}_hand.pkl", n_frames)
        orient, pose, trans, valid, orient_w, trans_w = _transform_hand_per_frame(orient, pose, trans, valid, side, w2c[:n_frames], beta)
        hand_data[side] = (orient, pose, trans, beta, valid)
        hand_world[side] = (orient_w, trans_w)

    active_any = hand_data["right"][4] | hand_data["left"][4]
    finite_cam = np.isfinite(w2c[:n_frames]).all(axis=(1, 2))
    out_root = Path(args.out_root or RESULTS_DIR)
    output_roles = list(roles) if args.object_role == "both" else [args.object_role]
    segments: list[str] = []

    for role in output_roles:
        if role not in roles:
            continue
        obj_name = _object_name_from_pose(roles[role], role)
        pair_names = {r: _object_name_from_pose(p, r) for r, p in roles.items()}
        obj_world = np.load(roles[role]).astype(np.float32)[:n_frames]
        finite_obj = np.isfinite(obj_world).all(axis=(1, 2))
        runs = _split_valid_runs(active_any & finite_cam & finite_obj, args.min_frames)
        if args.max_runs:
            runs = runs[: args.max_runs]
        if not runs:
            continue

        mesh_path = _find_object_mesh(object_root, obj_name)
        verts0, faces = load_mesh(str(mesh_path))
        verts0 = verts0.astype(np.float32)
        if mesh_path.stem.endswith("_cm"):
            verts0 *= 0.01
        faces = faces.astype(np.int32)

        for run_i, frames in enumerate(runs):
            segment_name = f"taco_{_safe_name(triplet)}_{sequence}_{role}_{_safe_name(obj_name)}_r{run_i:03d}"
            canonical_dir = out_root / segment_name / "canonical"
            canonical_path = canonical_dir / "canonical.npz"

            obj_cam = np.einsum("tij,tjk->tik", w2c[frames], obj_world[frames]).astype(np.float32)
            obj_rotation = rotmat_to_aa(obj_cam[:, :3, :3]).astype(np.float32)
            obj_translation = obj_cam[:, :3, 3].astype(np.float32)
            obj_rotation_world = rotmat_to_aa(obj_world[frames, :3, :3]).astype(np.float32)
            obj_translation_world = obj_world[frames, :3, 3].astype(np.float32)
            camera_ok, frac_in_front, median_z = _camera_z_is_plausible(obj_cam, verts0, hand_data, frames)
            if not camera_ok:
                print(
                    f"  SKIP {triplet}/{sequence}/{role}: camera z sanity failed "
                    f"(frac={frac_in_front:.3f}, median_z={median_z:.3f}m)"
                )
                continue
            if canonical_path.exists() and not args.force:
                segments.append(segment_name)
                continue

            obj_verts = np.empty((len(frames), verts0.shape[0], 3), dtype=np.float32)
            for i in range(len(frames)):
                obj_verts[i] = verts0 @ obj_cam[i, :3, :3].T + obj_translation[i]

            save = {
                "frame_indices": frames.astype(np.int64),
                "fps": np.float32(FPS),
                "obj_canonical_verts": verts0,
                "obj_faces": faces,
                "obj_rotation": obj_rotation,
                "obj_translation": obj_translation,
                "obj_rotation_world": obj_rotation_world,
                "obj_translation_world": obj_translation_world,
                "obj_scale": np.ones(3, dtype=np.float32),
                "obj_verts": obj_verts,
                "camera_K": K,
                "camera_extrinsics": w2c[frames].astype(np.float32),
                "gravity_world": _TACO_GRAVITY_WORLD.copy(),
                "gravity_direction": _gravity_direction_mid(w2c[frames], _TACO_GRAVITY_WORLD),
            }
            for side in ("right", "left"):
                orient, pose, trans, beta, valid = hand_data[side]
                orient_w, trans_w = hand_world[side]
                run_valid = valid[frames]
                save[f"hand_{side}_orient"] = orient[frames].copy()
                save[f"hand_{side}_pose"] = pose[frames].copy()
                save[f"hand_{side}_cam_t"] = trans[frames].copy()
                save[f"hand_{side}_orient_world"] = orient_w[frames].copy()
                save[f"hand_{side}_cam_t_world"] = trans_w[frames].copy()
                save[f"hand_{side}_betas"] = beta.astype(np.float32)
                save[f"hand_{side}_valid"] = run_valid.copy()
                if run_valid.any():
                    hv = mano_fk_verts(
                        save[f"hand_{side}_orient"],
                        save[f"hand_{side}_pose"],
                        beta.astype(np.float32),
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
            has_scene = False if args.no_video else _write_video(video_src, frames, video_path)
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
                "source_dataset": "taco",
                "segment_id": segment_name,
                "confidence_tier": "gt_fitted",
                "has_depth": bool(has_depth),
                "has_scene": has_scene,
                "has_gravity": True,
                "gravity_source": "world_axis_constant",
                "has_support_plane": False,
                "has_native_hand_pose": True,
                "has_native_object_pose": True,
                "object_is_rigid": True,
                "primary_object_id": obj_name,
                "primary_object_role": role,
                "primary_object_category": obj_name,
                "paired_object_ids": pair_names,
                "bimanual": bool(save["hand_left_valid"].any() and save["hand_right_valid"].any()),
                "viewpoint": "egocentric",
                "frame": "taco_egocentric_camera",
                "hand_source": "gt_fitted",
                "object_source": "gt_6d_pose",
                "depth_source": "egocentric_depth_video",
                "mano_convention": "right_model_x_mirror",
                "taco_triplet": triplet,
                "taco_sequence": sequence,
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
    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: TACO root not found: {root}")
        return []
    object_root = _resolve_object_root(root, args.object_model_root)
    if not object_root.exists():
        print(f"ERROR: TACO object model root not found: {object_root}")
        return []

    seqs = _discover_sequences(root, args.list_file)
    if args.triplet:
        seqs = [s for s in seqs if args.triplet in s[0]]
    if args.sequence:
        seqs = [s for s in seqs if args.sequence in s[1]]
    if args.seq_offset:
        seqs = seqs[args.seq_offset :]
    if args.max_seqs:
        seqs = seqs[: args.max_seqs]
    if not seqs:
        print(f"No TACO sequences found under {root}")
        return []

    print(f"Converting {len(seqs)} TACO sequences")
    segments: list[str] = []
    failed = 0
    for triplet, sequence in seqs:
        try:
            new_segments = _convert_sequence(root, object_root, triplet, sequence, args)
            segments.extend(new_segments)
            print(f"  {triplet}/{sequence}: {len(new_segments)} segments")
        except Exception as e:
            failed += 1
            print(f"  ERROR {triplet}/{sequence}: {e}")
            traceback.print_exc()
    print(f"  Convert: {len(segments)} segments, {failed} failed sequences")
    return segments


def run_validate_step(args) -> None:
    out_root = Path(args.out_root or RESULTS_DIR)
    paths = sorted(out_root.glob("taco_*/canonical/canonical.npz"))
    if args.sequence:
        paths = [p for p in paths if args.sequence in str(p)]
    if not paths:
        print(f"No TACO canonicals found under {out_root}")
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
    parser = argparse.ArgumentParser(prog="python -m gtf.datasets.taco", description="Ingest TACO into canonical format")
    parser.add_argument("--start-from", choices=PIPELINE_STEPS, default="download")
    parser.add_argument("--root", type=str, default=str(TACO_DIR))
    parser.add_argument("--out-root", type=str, default="")
    parser.add_argument("--object-model-root", type=str, default="")
    parser.add_argument("--object-role", choices=["tool", "target", "both"], default="both")
    parser.add_argument("--triplet", type=str, default="")
    parser.add_argument("--sequence", type=str, default="")
    parser.add_argument("--list-file", type=str, default="")
    parser.add_argument("--seq-offset", type=int, default=0, help="Skip this many discovered sequences before --max-seqs")
    parser.add_argument("--max-seqs", type=int, default=0)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--min-frames", type=int, default=30)
    parser.add_argument("--max-validate-frames", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--include-media",
        action="store_true",
        help="Also download and extract the egocentric RGB and depth videos (about 116 GB)",
    )
    parser.add_argument("--download-only", action="store_true", help="Stop after the download step")
    parser.add_argument("--convert-only", action="store_true", help="Stop after conversion; new segments are still validated while written")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--print-warnings", action="store_true")
    args = parser.parse_args()

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
