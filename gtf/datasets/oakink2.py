#!/usr/bin/env python3
"""Ingest OakInk2 rigid-object sequences into canonical.npz files.

OakInk2 annotations provide world-frame MANO quaternions and object transforms
at the 120 Hz mocap rate, plus per-view camera extrinsics. The loader emits one
canonical per rigid object instance (one rigid object per canonical); articulated
parts are skipped via ``object_affordance/instance_id``. Mocap is decimated to
30 Hz and the segment video holds exactly one frame per kept row, so
``frame_indices`` are video frame numbers (``source_frame_indices`` keeps the
mocap frame ids).

Usage:
    python -m gtf.datasets.oakink2 --start-from download --max-seqs 0
    python -m gtf.datasets.oakink2 --start-from convert --max-seqs 0
    python -m gtf.datasets.oakink2 --start-from convert --sequence scene_01__A001 --out-root /abs/scratch
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import shutil
import tarfile
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

OAKINK2_DIR = ROOT / "datasets" / "oakink2"

# The OakInk2 world is Y-up.
_OAKINK2_GRAVITY_WORLD = np.array([0.0, -1.0, 0.0], dtype=np.float32)


def _gravity_direction_mid(w2c_frames: np.ndarray, g_world: np.ndarray) -> np.ndarray:
    """Camera-frame gravity at the segment's middle frame from the constant world gravity."""
    E = w2c_frames[len(w2c_frames) // 2]
    g = E[:3, :3] @ g_world
    return (g / np.linalg.norm(g)).astype(np.float32)


HF_REPO = "kelvin34501/OakInk-v2"
FPS_MOCAP = 120.0
FPS_TARGET = 30.0
_VIDEO_WRITE_CACHE: dict[tuple[str, str, str, float], Path] = {}


def _safe_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.@-]+", "_", text.strip()).strip("_")


def _safe_extract_tar(tar_path: Path, out_dir: Path) -> None:
    out_dir = out_dir.resolve()
    with tarfile.open(tar_path, "r:*") as tf:
        for member in tf.getmembers():
            target = (out_dir / member.name).resolve()
            if not str(target).startswith(str(out_dir)):
                raise ValueError(f"Unsafe tar member path: {member.name}")
        tf.extractall(out_dir)


def _read_pickle(path: Path) -> dict[str, Any]:
    _preload_torch_cuda_deps()
    with open(path, "rb") as f:
        return pickle.load(f)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _quat_wxyz_to_aa(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32)
    flat = quat_wxyz.reshape(-1, 4)
    quat_xyzw = np.stack([flat[:, 1], flat[:, 2], flat[:, 3], flat[:, 0]], axis=1)
    aa = Rotation.from_quat(quat_xyzw).as_rotvec().astype(np.float32)
    return aa.reshape(quat_wxyz.shape[:-1] + (3,))


def _discover_annos(root: Path) -> list[Path]:
    return sorted((root / "anno_preview").glob("*.pkl"))


def _load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def _rigid_instance_ids(root: Path) -> set[str]:
    ids = _load_json(root / "object_affordance" / "instance_id.json", default=None)
    if isinstance(ids, list):
        return {str(x) for x in ids}
    return set()


def _object_desc(root: Path) -> dict[str, str]:
    data = _load_json(root / "object_repair" / "obj_desc.json", default={}) or {}
    out = {}
    for oid, item in data.items():
        if isinstance(item, dict):
            out[str(oid)] = str(item.get("obj_name", oid))
    return out


def _find_mesh(root: Path, obj_id: str, obj_offset: str) -> Path:
    obj_dir = root / obj_offset / "align_ds" / obj_id
    if not obj_dir.exists():
        raise FileNotFoundError(f"No OakInk2 object model directory for {obj_id}: {obj_dir}")
    candidates = [p for p in sorted(obj_dir.iterdir()) if p.suffix.lower() in {".obj", ".ply"}]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one mesh for {obj_id}, found {len(candidates)} under {obj_dir}")
    return candidates[0]


def _select_camera(anno: dict[str, Any], camera: str) -> tuple[str, np.ndarray]:
    cam_name = camera
    if camera == "first":
        cam_name = str(anno["cam_selection"][0])
    if cam_name not in anno["cam_intr"] or cam_name not in anno["cam_extr"]:
        raise KeyError(f"Camera {cam_name} not in OakInk2 annotation; available={list(anno['cam_intr'])}")
    first_fid = sorted(anno["cam_intr"][cam_name].keys())[0]
    K = np.asarray(anno["cam_intr"][cam_name][first_fid], dtype=np.float32).reshape(3, 3)
    return cam_name, K


def _nearest_camera_fid(camera_fids: np.ndarray, mocap_fid: int) -> int:
    pos = int(np.searchsorted(camera_fids, int(mocap_fid)))
    if pos <= 0:
        return int(camera_fids[0])
    if pos >= len(camera_fids):
        return int(camera_fids[-1])
    before = int(camera_fids[pos - 1])
    after = int(camera_fids[pos])
    return before if abs(int(mocap_fid) - before) <= abs(after - int(mocap_fid)) else after


def _camera_transforms_for_mocap(anno: dict[str, Any], cam_name: str, mocap_frames: list[int]) -> np.ndarray:
    extr = anno["cam_extr"][cam_name]
    camera_fids = np.asarray(sorted(int(k) for k in extr.keys()), dtype=np.int64)
    if camera_fids.size == 0:
        raise ValueError(f"OakInk2 camera {cam_name} has no extrinsic frames")
    out = np.empty((len(mocap_frames), 4, 4), dtype=np.float32)
    for i, fid in enumerate(mocap_frames):
        cam_fid = _nearest_camera_fid(camera_fids, int(fid))
        out[i] = np.asarray(extr[cam_fid], dtype=np.float32).reshape(4, 4)
    return out


def _transform_hand_rigid_series(
    orient: np.ndarray,
    trans: np.ndarray,
    valid: np.ndarray,
    side: str,
    cam_T: np.ndarray,
    betas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # Per-frame world-to-camera transforms (T, 4, 4). Shaped pivot J0(betas): the camera
    # fields are the exact rigid re-expression of the world fields, the inverse of the
    # present-camera re-anchor.
    orient_out = np.zeros_like(orient)
    trans_out = np.zeros_like(trans)
    for i in np.flatnonzero(valid):
        o, tr = transform_hand_rigid(
            orient[i : i + 1],
            trans[i : i + 1],
            np.ones(1, dtype=bool),
            side,
            R=cam_T[i, :3, :3],
            t=cam_T[i, :3, 3],
            betas=betas,
        )
        orient_out[i] = o[0]
        trans_out[i] = tr[0]
    return orient_out, trans_out


def _collect_hand(
    anno: dict[str, Any],
    side_code: str,
    side: str,
    mocap_frames: list[int],
    cam_T: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(mocap_frames)
    pose_q = np.zeros((n, 16, 4), dtype=np.float32)
    trans = np.zeros((n, 3), dtype=np.float32)
    betas = np.zeros(10, dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    pose_key = f"{side_code}__pose_coeffs"
    trans_key = f"{side_code}__tsl"
    beta_key = f"{side_code}__betas"
    for i, fid in enumerate(mocap_frames):
        item = anno["raw_mano"].get(fid)
        if not item or pose_key not in item or trans_key not in item:
            continue
        pq = _to_numpy(item[pose_key]).reshape(16, 4).astype(np.float32)
        tr = _to_numpy(item[trans_key]).reshape(3).astype(np.float32)
        be = _to_numpy(item[beta_key]).reshape(10).astype(np.float32)
        if np.isfinite(pq).all() and np.isfinite(tr).all() and np.isfinite(be).all():
            pose_q[i] = pq
            trans[i] = tr
            betas = be
            valid[i] = True
    aa = _quat_wxyz_to_aa(pose_q)
    orient = aa[:, 0, :]
    pose = aa[:, 1:, :].reshape(n, 45)
    # OakInk2 ``__tsl`` is the root-joint world position (manotorch center_idx=0
    # semantics); convert it to the manopth cam_t before mirroring.
    trans = root_trans_to_manopth_cam_t(trans, betas, side, str(MANO_DIR))
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
    orient_c, trans_c = _transform_hand_rigid_series(orient, trans, valid, side, cam_T, betas=betas)
    return (
        orient_c.astype(np.float32),
        pose.astype(np.float32),
        trans_c.astype(np.float32),
        betas.astype(np.float32),
        valid,
        orient_world.astype(np.float32),
        trans_world.astype(np.float32),
    )


def _resample_mocap_frames(mocap_frames: list[int], target_fps: float) -> tuple[list[int], float]:
    target_fps = float(target_fps)
    if target_fps <= 0 or target_fps >= FPS_MOCAP:
        return mocap_frames, FPS_MOCAP
    stride = max(1, int(round(FPS_MOCAP / target_fps)))
    return mocap_frames[::stride], FPS_MOCAP / stride


def _camera_serial_for_name(anno: dict[str, Any], camera: str) -> str:
    cam_def = anno.get("cam_def", {})
    for serial, name in cam_def.items():
        if str(name) == str(camera):
            return str(serial)
    if str(camera) in cam_def:
        return str(camera)
    raise KeyError(f"OakInk2 camera {camera!r} not found in cam_def={cam_def}")


def _tar_frame_index(tar_path: Path, serial: str) -> dict[int, str]:
    index: dict[int, str] = {}
    if not tar_path.exists():
        return index
    with tarfile.open(tar_path, "r:*") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            p = Path(member.name)
            if len(p.parts) < 2 or p.parts[-2] != serial or p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            try:
                index[int(p.stem)] = member.name
            except ValueError:
                continue
    return index


def _decode_tar_image(tf: tarfile.TarFile, member_name: str):
    try:
        import cv2
    except Exception:
        return None
    f = tf.extractfile(member_name)
    if f is None:
        return None
    data = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _write_oakink2_video(anno: dict[str, Any], sequence: str, camera: str, raw_frame_indices: np.ndarray, fps: float, out_path: Path) -> bool:
    try:
        import cv2
    except Exception:
        return False
    tar_path = OAKINK2_DIR / "data" / f"{sequence}.tar"
    if not tar_path.exists():
        return False
    frames_arr = np.ascontiguousarray(np.asarray(raw_frame_indices, dtype=np.int64))
    digest = hashlib.blake2b(frames_arr.view(np.uint8), digest_size=16).hexdigest()
    cache_key = (sequence, camera, digest, float(fps))
    cached = _VIDEO_WRITE_CACHE.get(cache_key)
    if cached is not None and cached.exists():
        # Another object of the same sequence/camera wrote this video already.
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_link = out_path.with_name(out_path.name + ".tmp")
        try:
            tmp_link.unlink(missing_ok=True)
            os.link(cached, tmp_link)
        except OSError:
            shutil.copy2(cached, tmp_link)
        os.replace(tmp_link, out_path)
        return out_path.exists() and out_path.stat().st_size > 0

    serial = _camera_serial_for_name(anno, camera)
    camera_frames = np.asarray([int(x) for x in anno.get("frame_id_list", [])], dtype=np.int64)
    if camera_frames.size == 0:
        return False
    mapped_frames = [_nearest_camera_fid(camera_frames, int(f)) for f in raw_frame_indices]
    index = _tar_frame_index(tar_path, serial)
    if not index:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary name, check the decoded frame count, then replace.
    tmp_path = out_path.with_name(out_path.name + ".tmp.mp4")
    with tarfile.open(tar_path, "r:*") as tf:
        first = None
        for fid in mapped_frames:
            member = index.get(int(fid))
            if member is None:
                continue
            first = _decode_tar_image(tf, member)
            if first is not None:
                break
        if first is None:
            return False
        h, w = first.shape[:2]
        writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
        if not writer.isOpened():
            return False
        last = first
        for fid in mapped_frames:
            img = None
            member = index.get(int(fid))
            if member is not None:
                img = _decode_tar_image(tf, member)
            if img is None:
                img = last
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            writer.write(img)
            last = img
        writer.release()
    ok = tmp_path.exists() and tmp_path.stat().st_size > 0
    if ok:
        cap = cv2.VideoCapture(str(tmp_path))
        n_dec = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        ok = n_dec == len(mapped_frames)
        if not ok:
            print(f"  [oakink2 video] decoded frame count {n_dec} != {len(mapped_frames)}; {out_path} not written")
    if ok:
        os.replace(tmp_path, out_path)
        _VIDEO_WRITE_CACHE.setdefault(cache_key, out_path)
    else:
        tmp_path.unlink(missing_ok=True)
    return ok


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


def run_download_step(args) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    files = api.list_repo_files(HF_REPO, repo_type="dataset")
    annos = [p for p in files if p.startswith("anno_preview/") and p.endswith(".pkl")]
    if args.sequence:
        annos = [p for p in annos if args.sequence in p]
    if args.max_seqs:
        annos = annos[: args.max_seqs]
    if not annos:
        raise ValueError("No OakInk2 annotations matched download filters")

    wanted = ["object_repair.tar", "object_affordance.tar", "program.tar", *annos]
    for filename in wanted:
        print(f"Downloading OakInk2 {filename}")
        hf_hub_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            filename=filename,
            local_dir=str(root),
            cache_dir=str(root / ".hf_cache"),
        )

    for tar_name in ("object_repair.tar", "object_affordance.tar", "program.tar"):
        done = root / f".{tar_name}.extracted"
        if done.exists() and not args.force_download:
            continue
        print(f"Extracting {tar_name}")
        _safe_extract_tar(root / tar_name, root)
        done.touch()


def _convert_anno(anno_path: Path, args) -> list[str]:
    root = Path(args.root)
    anno = _read_pickle(anno_path)
    cam_name, K = _select_camera(anno, args.camera)
    mocap_frames = list(anno["mocap_frame_id_list"])
    mocap_frames, fps_out = _resample_mocap_frames(mocap_frames, args.target_fps)
    n = len(mocap_frames)
    cam_T = _camera_transforms_for_mocap(anno, cam_name, mocap_frames)
    right = _collect_hand(anno, "rh", "right", mocap_frames, cam_T)
    left = _collect_hand(anno, "lh", "left", mocap_frames, cam_T)
    active_any = right[4] | left[4]

    rigid_ids = _rigid_instance_ids(root)
    obj_desc = _object_desc(root)
    obj_ids = [str(o) for o in anno["obj_list"]]
    if rigid_ids:
        obj_ids = [o for o in obj_ids if o in rigid_ids]
    if args.object_id:
        obj_ids = [o for o in obj_ids if args.object_id in o]

    out_root = Path(args.out_root or RESULTS_DIR)
    seq_token = anno_path.stem
    segments = []
    for obj_id in obj_ids:
        if obj_id not in anno["obj_transf"]:
            continue
        obj_world = np.zeros((n, 4, 4), dtype=np.float32)
        obj_valid = np.zeros(n, dtype=bool)
        for i, fid in enumerate(mocap_frames):
            if fid not in anno["obj_transf"][obj_id]:
                continue
            T = np.asarray(anno["obj_transf"][obj_id][fid], dtype=np.float32).reshape(4, 4)
            if np.isfinite(T).all():
                obj_world[i] = T
                obj_valid[i] = True
        valid = active_any & obj_valid
        runs = _split_valid_runs(valid, args.min_frames)
        if args.max_runs:
            runs = runs[: args.max_runs]
        if not runs:
            continue

        mesh_path = _find_mesh(root, obj_id, args.obj_offset)
        verts0, faces = load_mesh(str(mesh_path))
        verts0 = verts0.astype(np.float32)
        faces = faces.astype(np.int32)
        obj_name = obj_desc.get(obj_id, obj_id)

        for run_i, frames in enumerate(runs):
            segment_name = f"oakink2_{_safe_name(seq_token)}_{_safe_name(obj_id)}_{cam_name}_r{run_i:03d}"
            canonical_dir = out_root / segment_name / "canonical"
            canonical_path = canonical_dir / "canonical.npz"
            if canonical_path.exists() and not args.force:
                segments.append(segment_name)
                continue

            obj_cam = np.einsum("tij,tjk->tik", cam_T[frames], obj_world[frames]).astype(np.float32)
            obj_rotation = rotmat_to_aa(obj_cam[:, :3, :3]).astype(np.float32)
            obj_translation = obj_cam[:, :3, 3].astype(np.float32)
            obj_rotation_world = rotmat_to_aa(obj_world[frames, :3, :3]).astype(np.float32)
            obj_translation_world = obj_world[frames, :3, 3].astype(np.float32)
            obj_verts = np.empty((len(frames), verts0.shape[0], 3), dtype=np.float32)
            for i in range(len(frames)):
                obj_verts[i] = verts0 @ obj_cam[i, :3, :3].T + obj_translation[i]

            save = {
                # The video holds one frame per kept row, so frame_indices are video
                # frames and the effective cadence fps / median step is fps_out.
                "frame_indices": np.arange(len(frames), dtype=np.int64),
                "source_frame_indices": np.asarray([mocap_frames[int(f)] for f in frames], dtype=np.int64),
                "fps": np.float32(fps_out),
                "obj_canonical_verts": verts0,
                "obj_faces": faces,
                "obj_rotation": obj_rotation,
                "obj_translation": obj_translation,
                "obj_rotation_world": obj_rotation_world,
                "obj_translation_world": obj_translation_world,
                "obj_scale": np.ones(3, dtype=np.float32),
                "obj_verts": obj_verts,
                "camera_K": K,
                "camera_extrinsics": cam_T[frames].astype(np.float32),
                "gravity_world": _OAKINK2_GRAVITY_WORLD.copy(),
                "gravity_direction": _gravity_direction_mid(cam_T[frames], _OAKINK2_GRAVITY_WORLD),
            }
            for side, data in (("right", right), ("left", left)):
                orient, pose, trans, betas, side_valid, orient_w, trans_w = data
                run_valid = side_valid[frames]
                save[f"hand_{side}_orient"] = orient[frames].copy()
                save[f"hand_{side}_pose"] = pose[frames].copy()
                save[f"hand_{side}_cam_t"] = trans[frames].copy()
                save[f"hand_{side}_orient_world"] = orient_w[frames].copy()
                save[f"hand_{side}_cam_t_world"] = trans_w[frames].copy()
                save[f"hand_{side}_betas"] = betas.astype(np.float32)
                save[f"hand_{side}_valid"] = run_valid.copy()
                if run_valid.any():
                    hv = mano_fk_verts(
                        save[f"hand_{side}_orient"],
                        save[f"hand_{side}_pose"],
                        betas.astype(np.float32),
                        save[f"hand_{side}_cam_t"],
                        side,
                        str(MANO_DIR),
                    )
                    hv[~run_valid] = 0
                else:
                    hv = np.zeros((len(frames), 778, 3), dtype=np.float32)
                save[f"hand_{side}_verts"] = hv.astype(np.float32)

            canonical_dir.mkdir(parents=True, exist_ok=True)
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

            video_path = out_root / segment_name / "video.mp4"
            # The video writer maps mocap frame ids to camera images.
            has_scene = False if args.no_video else _write_oakink2_video(anno, seq_token, cam_name, save["source_frame_indices"], fps_out, video_path)

            metadata = {
                "source_dataset": "oakink2",
                "segment_id": segment_name,
                "confidence_tier": "gt_mocap",
                "has_depth": bool(has_scene),
                "has_scene": has_scene,
                "has_gravity": True,
                "gravity_source": "world_axis_constant",
                "has_support_plane": False,
                "has_native_hand_pose": True,
                "has_native_object_pose": True,
                "object_is_rigid": True,
                "primary_object_id": obj_id,
                "primary_object_category": obj_name,
                "bimanual": bool(save["hand_left_valid"].any() and save["hand_right_valid"].any()),
                "viewpoint": "egocentric" if cam_name == "egocentric" else "exocentric",
                "frame": f"oakink2_{'dynamic' if cam_name == 'egocentric' else 'static'}_{cam_name}_camera",
                "hand_source": "gt_mocap_mano",
                "object_source": "gt_mocap_6d_pose",
                "depth_source": "moge2_video" if has_scene else "",
                "mano_convention": "right_model_x_mirror",
                "oakink2_sequence": seq_token,
                "oakink2_camera": cam_name,
                "n_frames": int(len(frames)),
                "fps": fps_out,
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
        print(f"ERROR: OakInk2 root not found: {root}")
        return []
    if not (root / args.obj_offset / "align_ds").exists():
        print(f"ERROR: OakInk2 object models not found: {root / args.obj_offset / 'align_ds'}")
        return []
    annos = _discover_annos(root)
    if args.sequence:
        annos = [p for p in annos if args.sequence in p.name]
    if args.seq_offset:
        annos = annos[args.seq_offset :]
    if args.max_seqs:
        annos = annos[: args.max_seqs]
    if not annos:
        print(f"No OakInk2 annotations found under {root / 'anno_preview'}")
        return []

    print(f"Converting {len(annos)} OakInk2 annotations")
    segments: list[str] = []
    failed = 0
    for anno_path in annos:
        try:
            new_segments = _convert_anno(anno_path, args)
            segments.extend(new_segments)
            print(f"  {anno_path.name}: {len(new_segments)} segments")
        except Exception as e:
            failed += 1
            print(f"  ERROR {anno_path}: {e}")
            traceback.print_exc()
    print(f"  Convert: {len(segments)} segments, {failed} failed sequences")
    return segments


def run_validate_step(args) -> None:
    out_root = Path(args.out_root or RESULTS_DIR)
    paths = sorted(out_root.glob("oakink2_*/canonical/canonical.npz"))
    if args.sequence:
        paths = [p for p in paths if _safe_name(args.sequence) in str(p)]
    if not paths:
        print(f"No OakInk2 canonicals found under {out_root}")
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
    parser = argparse.ArgumentParser(prog="python -m gtf.datasets.oakink2", description="Ingest OakInk2 rigid-object data into canonical format")
    parser.add_argument("--start-from", choices=PIPELINE_STEPS, default="download")
    parser.add_argument("--root", type=str, default=str(OAKINK2_DIR))
    parser.add_argument("--out-root", type=str, default="")
    parser.add_argument("--sequence", type=str, default="")
    parser.add_argument("--object-id", type=str, default="")
    parser.add_argument("--camera", type=str, default="egocentric")
    parser.add_argument("--obj-offset", type=str, default="object_repair")
    parser.add_argument("--target-fps", type=float, default=FPS_TARGET, help="Output canonical fps after mocap decimation; default 30.")
    parser.add_argument("--seq-offset", type=int, default=0, help="Skip this many discovered annotations before --max-seqs")
    parser.add_argument("--max-seqs", type=int, default=1)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--min-frames", type=int, default=30)
    parser.add_argument("--max-validate-frames", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-video", action="store_true", help="Skip writing video.mp4 (the scene caches need it)")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--download-only", action="store_true", help="Stop after the download step")
    parser.add_argument("--convert-only", action="store_true", help="Stop after conversion; new segments are still validated while written")
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
