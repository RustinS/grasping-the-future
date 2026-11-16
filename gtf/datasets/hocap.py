#!/usr/bin/env python3
"""Ingest HO-Cap into model-ready canonical.npz files.

HO-Cap provides RGB-D videos, true-side MANO PCA hand poses, object 6DoF poses,
object meshes, camera calibration, and image labels. This loader downloads the
official public Box ZIP archives into ``datasets/hocap`` and writes camera-frame
canonicals under ``results/hocap_*``.

HO-Cap uses the true ``MANO_LEFT`` model for left hands, while canonicals store left
hands as right-model x-mirrors: ``MANO_LEFT.pkl`` expands the left PCA coefficients
before mirroring, and sequences with a left hand are skipped without that file.

Usage:
    python -m gtf.datasets.hocap --start-from download --subject all
    python -m gtf.datasets.hocap --start-from convert --subject all
"""

from __future__ import annotations

import argparse
import json
import traceback
import zipfile
from pathlib import Path

import numpy as np
import requests
import yaml
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from gtf.paths import MANO_DIR, mano_model_path, REPO_ROOT
from gtf.datasets.validate_conventions import validate_canonical_conventions
from gtf.utils.hand_rigid import transform_hand_rigid
from gtf.utils import RESULTS_DIR, rprint as print
from gtf.utils.canonical import (
    load_mesh,
    mano_fk_verts,
    mano_pca_to_aa,
    mirror_left_hand_params,
    rotmat_to_aa,
    validate_canonical_npz,
)

ROOT = REPO_ROOT

PIPELINE_STEPS = ["download", "convert", "validate"]

HOCAP_DIR = ROOT / "datasets" / "hocap"
MANO_RIGHT = Path(mano_model_path("right"))
MANO_LEFT = Path(mano_model_path("left"))
FPS = 30.0

# Camera-frame gravity of rig camera 0, the consensus of RANSAC support-plane fits on its
# scene point clouds (0.0-0.6 deg spread across sequences). Other cameras use the world
# Z axis of the calibration board.
CAM0_GRAVITY = np.array([0.0151, 0.3800, 0.9250], dtype=np.float64)

BOX_URLS = {
    "models": "https://utdallas.box.com/shared/static/con44iqej33weg9f3rpxof61eh3x2x21.zip",
    "calibration": "https://utdallas.box.com/shared/static/nlp4c6vtd0n8o0entxlh1vxdpcdeh0h8.zip",
    "poses": "https://utdallas.box.com/shared/static/2lofbp2yd005d8o213ns77mdrtxg8eep.zip",
    "subject_1": "https://utdallas.box.com/shared/static/w0voy9bixtxyclo52841xyamock2lxpt.zip",
    "subject_2": "https://utdallas.box.com/shared/static/j498kxxrkvaf674tvmt4su4ad0bz9s9f.zip",
    "subject_3": "https://utdallas.box.com/shared/static/shklq33yaoozh9gm681nxwnq0o3y0y1d.zip",
    "subject_4": "https://utdallas.box.com/shared/static/dew68k7b3ya09t40818gpfxm95oa4yeq.zip",
    "subject_5": "https://utdallas.box.com/shared/static/mutor2a09kudze1yw173gsfetsru7ces.zip",
    "subject_6": "https://utdallas.box.com/shared/static/iyja7rdbjx2ksgjhmdu6mx3zqvaurdni.zip",
    "subject_7": "https://utdallas.box.com/shared/static/4g5qyig6i4uz1rgrzkcu9n4mhdjs74m2.zip",
    "subject_8": "https://utdallas.box.com/shared/static/khrb5guy8rdwnoqi4euk2w0mk5lslxkn.zip",
    "subject_9": "https://utdallas.box.com/shared/static/3x5yitydmbmwolq9bty5dd2udu5v52fc.zip",
}

OBJECT_DESCRIPTORS = {
    "G01_1": "fruit_snacks",
    "G01_2": "water_softener_bottle",
    "G01_3": "coconut_milk_carton",
    "G01_4": "hammer",
    "G02_1": "whole_milk_carton",
    "G02_2": "cooked_ham",
    "G02_3": "chocolate_drink_powder",
    "G02_4": "sauce_bottle",
    "G04_1": "flapjack_mix",
    "G04_2": "chocolate_fudge",
    "G04_3": "herring_fillets_can",
    "G04_4": "crackers_box",
    "G05_1": "vegetable_oil_spread",
    "G05_2": "body_lotion",
    "G05_3": "peanut_chocolate_box",
    "G05_4": "sponge",
    "G06_1": "dvd",
    "G06_2": "pumpkin_creamer",
    "G06_3": "yellow_mustard",
    "G06_4": "dish_brush",
    "G07_1": "game_controller",
    "G07_2": "hot_cocoa_mix",
    "G07_3": "dandruff_shampoo",
    "G07_4": "toy_axe",
    "G09_1": "candy_box",
    "G09_2": "toy_car",
    "G09_3": "toothpaste_box",
    "G09_4": "chocolate_syrup_bottle",
    "G10_1": "soup_mix",
    "G10_2": "gel_toothpaste",
    "G10_3": "chocolate_biscuit_sticks",
    "G10_4": "body_wash",
    "G11_1": "coconut_water",
    "G11_2": "baby_powder",
    "G11_3": "baking_soda",
    "G11_4": "chocolate_bar",
    "G15_1": "mens_body_wash",
    "G15_2": "dandelion_tea",
    "G15_3": "cooking_spray",
    "G15_4": "joy_controller",
    "G16_1": "toilet_cleaner",
    "G16_2": "laundry_detergent",
    "G16_3": "small_coconut_water",
    "G16_4": "fabric_softener_sheets",
    "G18_1": "green_tea_latte_mix",
    "G18_2": "projector_remote",
    "G18_3": "right_shoe",
    "G18_4": "left_shoe",
    "G19_1": "electric_screwdriver",
    "G19_2": "blue_spatula",
    "G19_3": "deodorant",
    "G19_4": "ping_pong_paddle",
    "G20_1": "cappuccino_mix",
    "G20_2": "mustard_bottle",
    "G20_3": "toilet_cleaner",
    "G20_4": "dog_toy_bone",
    "G21_1": "moisturizing_lotion",
    "G21_2": "playing_cards",
    "G21_3": "pink_spatula",
    "G21_4": "blue_brush",
    "G22_1": "chocolate_bar",
    "G22_2": "mayomust_sauce",
    "G22_3": "soup_mix",
    "G22_4": "gray_spatula",
}


def _category_from_descriptor(desc: str) -> str:
    for token in ("bottle", "carton", "box", "can", "brush", "spatula", "remote", "shoe", "hammer", "car", "paddle", "controller"):
        if token in desc:
            return token
    if "milk" in desc or "water" in desc:
        return "carton"
    if "lotion" in desc or "shampoo" in desc or "cleaner" in desc or "wash" in desc:
        return "bottle"
    return desc.split("_")[-1]


def _read_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _safe_extract(zip_path: Path, out_dir: Path) -> None:
    out_dir = out_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            target = (out_dir / member.filename).resolve()
            if not str(target).startswith(str(out_dir)):
                raise ValueError(f"Unsafe zip member path: {member.filename}")
        zf.extractall(out_dir)


def _download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    resume = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={resume}-"} if resume else {}
    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) + resume
        mode = "ab" if resume and r.status_code == 206 else "wb"
        with tmp.open(mode) as f, tqdm(total=total, initial=resume if mode == "ab" else 0, unit="B", unit_scale=True, ncols=80, desc=out_path.name) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    tmp.replace(out_path)


def _target_archives(subject: str) -> list[str]:
    base = ["models", "calibration", "poses"]
    if subject == "all":
        return base + [f"subject_{i}" for i in range(1, 10)]
    return base + [subject]


def run_download_step(args) -> None:
    HOCAP_DIR.mkdir(parents=True, exist_ok=True)
    names = _target_archives(args.subject)
    for name in names:
        zip_path = HOCAP_DIR / f"{name}.zip"
        done = HOCAP_DIR / f".{name}.extracted"
        if done.exists() and not args.force_download:
            print(f"  {name}: already extracted")
            continue
        if not zip_path.exists() or args.force_download:
            print(f"Downloading HO-Cap {name}")
            _download_file(BOX_URLS[name], zip_path)
        print(f"Extracting {zip_path.name}")
        _safe_extract(zip_path, HOCAP_DIR)
        done.touch()


def _quat_to_mat(qt: np.ndarray) -> np.ndarray:
    qt = np.asarray(qt, dtype=np.float32)
    q = qt[..., :4]
    t = qt[..., 4:7]
    out = np.broadcast_to(np.eye(4, dtype=np.float32), qt.shape[:-1] + (4, 4)).copy()
    out[..., :3, :3] = Rotation.from_quat(q.reshape(-1, 4)).as_matrix().reshape(q.shape[:-1] + (3, 3)).astype(np.float32)
    out[..., :3, 3] = t
    return out


def _load_extrinsics(dataset_root: Path, sequence_meta: dict) -> tuple[list[str], np.ndarray]:
    calib_root = dataset_root / "calibration"
    serials = list(sequence_meta["realsense"]["serials"])
    ext_name = sequence_meta["extrinsics"]
    data = _read_yaml(calib_root / "extrinsics" / ext_name)

    def mat(values):
        return np.array([values[0:4], values[4:8], values[8:12], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

    tag_1_inv = np.linalg.inv(mat(data["extrinsics"]["tag_1"]))
    extr2master = np.stack([mat(data["extrinsics"][s]) for s in serials], axis=0)
    cam2world = np.stack([tag_1_inv @ t for t in extr2master], axis=0).astype(np.float32)
    return serials, cam2world


def _load_intrinsic(dataset_root: Path, serial: str) -> np.ndarray:
    data = _read_yaml(dataset_root / "calibration" / "intrinsics" / f"{serial}.yaml")["color"]
    return np.array([[data["fx"], 0.0, data["ppx"]], [0.0, data["fy"], data["ppy"]], [0.0, 0.0, 1.0]], dtype=np.float32)


def _load_betas(dataset_root: Path, subject_id: str) -> np.ndarray:
    data = _read_yaml(dataset_root / "calibration" / "mano" / f"{subject_id}.yaml")
    return np.asarray(data["betas"], dtype=np.float32)


def _discover_sequences(root: Path, subject: str) -> list[Path]:
    subjects = [p for p in sorted(root.glob("subject_*")) if p.is_dir()]
    if subject != "all":
        subjects = [p for p in subjects if p.name == subject]
    seqs = []
    for sub in subjects:
        for seq in sorted(sub.iterdir()):
            if (seq / "meta.yaml").exists() and (seq / "poses_m.npy").exists() and (seq / "poses_o.npy").exists():
                seqs.append(seq)
    return seqs


def _split_valid_runs(valid: np.ndarray, min_frames: int) -> list[np.ndarray]:
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return []
    runs = []
    start = idx[0]
    prev = idx[0]
    for x in idx[1:]:
        if x != prev + 1:
            if prev - start + 1 >= min_frames:
                runs.append(np.arange(start, prev + 1, dtype=np.int64))
            start = x
        prev = x
    if prev - start + 1 >= min_frames:
        runs.append(np.arange(start, prev + 1, dtype=np.int64))
    return runs


def _convert_hand_pose(raw_pose: np.ndarray, side: str, R_wc: np.ndarray, t_wc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    orient = raw_pose[:, :3].astype(np.float32)
    pca = raw_pose[:, 3:48].astype(np.float32)
    trans = raw_pose[:, 48:51].astype(np.float32)
    if side == "right":
        pose = mano_pca_to_aa(pca, 45, str(MANO_RIGHT))
    else:
        if not MANO_LEFT.exists():
            raise FileNotFoundError(f"HO-Cap left-hand conversion requires {MANO_LEFT}")
        pose = mano_pca_to_aa(pca, 45, str(MANO_LEFT))
        orient, pose, trans = mirror_left_hand_params(orient, pose, trans)
    valid = np.ones(raw_pose.shape[0], dtype=bool)
    orient_c, trans_c = transform_hand_rigid(orient, trans, valid, side, R=R_wc, t=t_wc)
    return orient_c.astype(np.float32), pose.astype(np.float32), trans_c.astype(np.float32)


def _write_video(seq_dir: Path, serial: str, frames: np.ndarray, out_path: Path) -> bool:
    try:
        import cv2
    except Exception:
        return False
    first = None
    for f in frames:
        p = seq_dir / serial / f"color_{int(f):06d}.jpg"
        if p.exists():
            first = cv2.imread(str(p))
            break
    if first is None:
        return False
    h, w = first.shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    if not writer.isOpened():
        return False
    for f in frames:
        p = seq_dir / serial / f"color_{int(f):06d}.jpg"
        img = cv2.imread(str(p)) if p.exists() else None
        if img is None:
            img = np.zeros_like(first)
        writer.write(img)
    writer.release()
    return out_path.exists() and out_path.stat().st_size > 0


def _convert_sequence(seq_dir: Path, args) -> list[str]:
    dataset_root = Path(args.root)
    meta = _read_yaml(seq_dir / "meta.yaml")
    object_ids = list(meta["object_ids"])
    mano_sides = list(meta["mano_sides"])
    serials, cam2world = _load_extrinsics(dataset_root, meta)
    if args.camera_idx >= len(serials):
        raise ValueError(f"camera_idx {args.camera_idx} out of range for {seq_dir}")
    serial = serials[args.camera_idx]
    w2c = np.linalg.inv(cam2world[args.camera_idx]).astype(np.float32)
    if args.camera_idx == 0:
        gravity_direction = (CAM0_GRAVITY / np.linalg.norm(CAM0_GRAVITY)).astype(np.float32)
        gravity_source = "scene_plane_consensus"
    else:
        # The HO-Cap world (calibration board on the table) is Z-up.
        gravity_direction = (w2c[:3, :3] @ np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float32)
        gravity_source = "world_axis_constant"
    R_wc = w2c[:3, :3]
    t_wc = w2c[:3, 3]
    K = _load_intrinsic(dataset_root, serial)

    betas = _load_betas(dataset_root, meta["subject_id"])
    poses_m = np.load(seq_dir / "poses_m.npy").astype(np.float32)  # (2, T, 51)
    poses_o = _quat_to_mat(np.load(seq_dir / "poses_o.npy").astype(np.float32))  # (O, T, 4, 4)
    T = int(meta["num_frames"])

    hand_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for side in ("right", "left"):
        if side not in mano_sides:
            hand_data[side] = (
                np.zeros((T, 3), dtype=np.float32),
                np.zeros((T, 45), dtype=np.float32),
                np.zeros((T, 3), dtype=np.float32),
                np.zeros(T, dtype=bool),
            )
            continue
        raw = poses_m[0 if side == "right" else 1]
        valid = np.isfinite(raw).all(axis=1) & (raw[:, 0] != -1)
        orient = np.zeros((T, 3), dtype=np.float32)
        pose = np.zeros((T, 45), dtype=np.float32)
        trans = np.zeros((T, 3), dtype=np.float32)
        if valid.any():
            o, p, tr = _convert_hand_pose(raw[valid], side, R_wc, t_wc)
            orient[valid] = o
            pose[valid] = p
            trans[valid] = tr
        hand_data[side] = (orient, pose, trans, valid)

    active_any = hand_data["right"][3] | hand_data["left"][3]
    segments = []
    out_root = Path(args.out_root or RESULTS_DIR)

    for obj_idx, obj_id in enumerate(object_ids):
        if args.object_id and obj_id != args.object_id:
            continue
        mesh_path = dataset_root / "models" / obj_id / "cleaned_mesh_10000.obj"
        if not mesh_path.exists():
            mesh_path = dataset_root / "models" / obj_id / "textured_mesh.obj"
        if not mesh_path.exists():
            raise FileNotFoundError(f"No mesh found for {obj_id}")

        obj_world = poses_o[obj_idx]
        finite_obj = np.isfinite(obj_world).all(axis=(1, 2))
        runs = _split_valid_runs(active_any & finite_obj, args.min_frames)
        if args.max_runs:
            runs = runs[: args.max_runs]
        if not runs:
            continue

        verts0, faces = load_mesh(str(mesh_path))
        verts0 = verts0.astype(np.float32)
        faces = faces.astype(np.int32)
        desc = OBJECT_DESCRIPTORS.get(obj_id, obj_id.lower())
        category = _category_from_descriptor(desc)

        for run_i, frames in enumerate(runs):
            segment_name = f"hocap_{meta['subject_id']}_{seq_dir.name}_{obj_id}_cam{args.camera_idx}_r{run_i:03d}"
            canonical_dir = out_root / segment_name / "canonical"
            canonical_path = canonical_dir / "canonical.npz"
            if canonical_path.exists() and not args.force:
                segments.append(segment_name)
                continue

            obj_cam = np.einsum("ij,tjk->tik", w2c, obj_world[frames]).astype(np.float32)
            obj_rotation = rotmat_to_aa(obj_cam[:, :3, :3]).astype(np.float32)
            obj_translation = obj_cam[:, :3, 3].astype(np.float32)
            obj_verts = np.zeros((len(frames), verts0.shape[0], 3), dtype=np.float32)
            for i in range(len(frames)):
                obj_verts[i] = verts0 @ obj_cam[i, :3, :3].T + obj_translation[i]

            save = {
                "frame_indices": frames.astype(np.int64),
                "fps": np.float32(FPS),
                "obj_canonical_verts": verts0,
                "obj_faces": faces,
                "obj_rotation": obj_rotation,
                "obj_translation": obj_translation,
                "obj_scale": np.ones(3, dtype=np.float32),
                "obj_verts": obj_verts,
                "gravity_direction": gravity_direction,
                "camera_K": K,
            }
            for side in ("right", "left"):
                orient, pose, trans, valid = hand_data[side]
                run_valid = valid[frames]
                save[f"hand_{side}_orient"] = orient[frames].copy()
                save[f"hand_{side}_pose"] = pose[frames].copy()
                save[f"hand_{side}_cam_t"] = trans[frames].copy()
                save[f"hand_{side}_betas"] = betas.astype(np.float32)
                save[f"hand_{side}_valid"] = run_valid.copy()
                if run_valid.any():
                    hv = mano_fk_verts(save[f"hand_{side}_orient"], save[f"hand_{side}_pose"], betas.astype(np.float32), save[f"hand_{side}_cam_t"], side, str(MANO_DIR))
                    hv[~run_valid] = 0
                else:
                    hv = np.zeros((len(frames), 778, 3), dtype=np.float32)
                save[f"hand_{side}_verts"] = hv.astype(np.float32)

            canonical_dir.mkdir(parents=True, exist_ok=True)
            video_path = out_root / segment_name / "video.mp4"
            has_scene = False if args.no_video else _write_video(seq_dir, serial, frames, video_path)
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
                "source_dataset": "hocap",
                "segment_id": segment_name,
                "confidence_tier": "gt_fitted",
                "has_depth": (seq_dir / serial / f"depth_{int(frames[len(frames)//2]):06d}.png").exists(),
                "has_scene": has_scene,
                "has_gravity": True,
                "gravity_source": gravity_source,
                "has_support_plane": False,
                "has_native_hand_pose": True,
                "has_native_object_pose": True,
                "object_is_rigid": True,
                "primary_object_id": obj_id,
                "primary_object_category": category,
                "bimanual": bool(save["hand_left_valid"].any() and save["hand_right_valid"].any()),
                "viewpoint": "exocentric",
                "frame": "realsense_static_camera",
                "hand_source": "gt_fitted",
                "object_source": "gt_6d_pose",
                "depth_source": "realsense",
                "mano_convention": "right_model_x_mirror",
                "hocap_subject": meta["subject_id"],
                "hocap_sequence": seq_dir.name,
                "hocap_camera_serial": serial,
                "hocap_camera_idx": int(args.camera_idx),
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
        print(f"ERROR: HO-Cap root not found: {root}")
        return []
    seqs = _discover_sequences(root, args.subject)
    if args.sequence:
        seqs = [p for p in seqs if args.sequence in p.name]
    if args.max_seqs:
        seqs = seqs[: args.max_seqs]
    if not seqs:
        print(f"No HO-Cap sequences found under {root}")
        return []

    print(f"Converting {len(seqs)} HO-Cap sequences")
    done = failed = skipped = 0
    segments: list[str] = []
    for seq in seqs:
        try:
            if not MANO_LEFT.exists():
                meta = _read_yaml(seq / "meta.yaml")
                if "left" in list(meta.get("mano_sides", [])):
                    skipped += 1
                    print(f"  SKIP {seq.parent.name}/{seq.name}: MANO_LEFT.pkl required for left-hand HO-Cap PCA")
                    continue
            new_segments = _convert_sequence(seq, args)
            segments.extend(new_segments)
            done += len(new_segments)
            print(f"  {seq.parent.name}/{seq.name}: {len(new_segments)} segments")
        except Exception as e:
            failed += 1
            print(f"  ERROR {seq}: {e}")
            traceback.print_exc()
    print(f"  Convert: {done} segments, {failed} failed sequences, {skipped} skipped sequences")
    return segments


def run_validate_step(args) -> None:
    out_root = Path(args.out_root or RESULTS_DIR)
    paths = sorted(out_root.glob("hocap_*/canonical/canonical.npz"))
    if args.sequence:
        paths = [p for p in paths if args.sequence in str(p)]
    if not paths:
        print(f"No HO-Cap canonicals found under {out_root}")
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
    parser = argparse.ArgumentParser(prog="python -m gtf.datasets.hocap", description="Ingest HO-Cap into canonical format")
    parser.add_argument("--start-from", choices=PIPELINE_STEPS, default="download")
    parser.add_argument("--root", type=str, default=str(HOCAP_DIR))
    parser.add_argument("--out-root", type=str, default="")
    parser.add_argument("--subject", type=str, default="subject_1", choices=["all"] + [f"subject_{i}" for i in range(1, 10)])
    parser.add_argument("--sequence", type=str, default="")
    parser.add_argument("--object-id", type=str, default="")
    parser.add_argument("--camera-idx", type=int, default=0)
    parser.add_argument("--max-seqs", type=int, default=0)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--min-frames", type=int, default=30)
    parser.add_argument("--max-validate-frames", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--download-only", action="store_true", help="Stop after the download step")
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
    if "validate" in active:
        run_validate_step(args)


if __name__ == "__main__":
    main()
