#!/usr/bin/env python3
"""Ingest DexYCB into camera-frame training canonicals.

Downloads the per-subject archives from dex-ycb.github.io and writes one
canonical per (sequence, camera) for three spread cameras (indices 0, 2, 5)
under `results/dexycb_{subject}_{sequence}_cam{N}/canonical/`. Everything
stays in that camera's optical frame, in meters as in the DexYCB labels; the
gravity direction is estimated from the sequence's middle-frame depth map.

Usage:
    python -m gtf.datasets.dexycb                                   # full: download → convert
    python -m gtf.datasets.dexycb --start-from convert --subject s01
    python -m gtf.datasets.dexycb --start-from convert --force
    python -m gtf.datasets.dexycb --start-from convert --subject s01 --max-seqs 2 --no-video --out-root /abs/scratch
"""

import argparse
import json
import subprocess
import tarfile
import traceback
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from gtf.paths import MANO_DIR, mano_model_path, REPO_ROOT
from gtf.utils import RESULTS_DIR, rprint as print
from gtf.utils.canonical import (
    mano_pca_to_aa,
    mirror_left_hand_params,
    rotmat_to_aa,
    estimate_gravity,
    mano_fk_verts,
    load_mesh,
)

ROOT = REPO_ROOT

PIPELINE_STEPS = ["download", "convert"]

DEXYCB_DIR = ROOT / "datasets" / "dexycb"
MANO_MODEL = mano_model_path("right")

FPS = 30.0

DEXYCB_SUBJECTS = {
    "s01": "20200709-subject-01",
    "s02": "20200813-subject-02",
    "s03": "20200820-subject-03",
    "s04": "20200903-subject-04",
    "s05": "20200908-subject-05",
    "s06": "20200918-subject-06",
    "s07": "20200928-subject-07",
    "s08": "20201002-subject-08",
    "s09": "20201015-subject-09",
    "s10": "20201022-subject-10",
}

# 8 cameras; canonicals are written for 3 spread views (indices 0, 2, 5).
ALL_CAMERA_SERIALS = [
    "836212060125",
    "839512060362",
    "840412060917",
    "841412060263",
    "932122060857",
    "932122060861",
    "932122061900",
    "932122062010",
]
SELECTED_CAMERA_INDICES = [0, 2, 5]
SELECTED_CAMERAS = [ALL_CAMERA_SERIALS[i] for i in SELECTED_CAMERA_INDICES]

YCB_OBJECTS = {
    1: "002_master_chef_can",
    2: "003_cracker_box",
    3: "004_sugar_box",
    4: "005_tomato_soup_can",
    5: "006_mustard_bottle",
    6: "007_tuna_fish_can",
    7: "008_pudding_box",
    8: "009_gelatin_box",
    9: "010_potted_meat_can",
    10: "011_banana",
    11: "019_pitcher_base",
    12: "021_bleach_cleanser",
    13: "024_bowl",
    14: "025_mug",
    15: "035_power_drill",
    16: "036_wood_block",
    17: "037_scissors",
    18: "040_large_marker",
    19: "052_extra_large_clamp",
    20: "061_foam_brick",
}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _gdown(gdrive_id, output_path):
    """Download a file from Google Drive using gdown."""
    import gdown

    url = f"https://drive.google.com/uc?id={gdrive_id}"
    gdown.download(url, str(output_path), quiet=False)


# Google Drive file IDs for DexYCB (from https://dex-ycb.github.io/)
GDRIVE_IDS = {
    "calibration.tar.gz": "1UAwVKT4Rgb1fLcFoa1o71_-0NtSvvLAQ",
    "models.tar.gz": "1cAzlQBpcTatI5ykYQ8ziQiHLUG_a_UpM",
    "20200709-subject-01.tar.gz": "1Ehh92wDE3CWAiKG7E9E73HjN2Xk2XfEk",
    "20200813-subject-02.tar.gz": "1Uo7MLqTbXEa-8s7YQZ3duugJ1nXFEo62",
    "20200820-subject-03.tar.gz": "1FkUxas8sv8UcVGgAzmSZlJw1eI5W5CXq",
    "20200903-subject-04.tar.gz": "14up6qsTpvgEyqOQ5hir-QbjMB_dHfdpA",
    "20200908-subject-05.tar.gz": "1NBA_FPyGWOQF5-X9ueAat5g8lDMz-EmS",
    "20200918-subject-06.tar.gz": "1UWIN2-wOBZX2T0dkAi4ctAAW8KffkXMQ",
    "20200928-subject-07.tar.gz": "1oWEYD_o3PVh39pLzMlJcArkDtMj4nzI0",
    "20201002-subject-08.tar.gz": "1GTNZwhWbs7Mfez0krTgXwLPndvrw1Ztv",
    "20201015-subject-09.tar.gz": "1j0BLkaCjIuwjakmywKdOO9vynHTWR0UH",
    "20201022-subject-10.tar.gz": "1FvFlRfX-p5a5sAWoKEGc17zKJWwKaSB-",
}


def run_download_step(args):
    """Download DexYCB archives from Google Drive via gdown."""
    DEXYCB_DIR.mkdir(parents=True, exist_ok=True)

    for archive_name in ["calibration.tar.gz", "models.tar.gz"]:
        local_path = DEXYCB_DIR / archive_name
        extract_marker = DEXYCB_DIR / f".{archive_name}.done"

        if extract_marker.exists():
            continue

        gdrive_id = GDRIVE_IDS.get(archive_name)
        if not local_path.exists() and gdrive_id:
            print(f"Downloading {archive_name} from Google Drive...")
            try:
                _gdown(gdrive_id, local_path)
                print(f"  Downloaded {local_path.stat().st_size / 1e6:.1f} MB")
            except Exception as e:
                print(f"  FAILED: {e}")
                continue

        if local_path.exists():
            print(f"  Extracting {archive_name}...")
            with tarfile.open(str(local_path), "r:gz") as tf:
                tf.extractall(str(DEXYCB_DIR))
            extract_marker.touch()

    subjects = _get_target_subjects(args)
    for subject_key, subject_dir_name in subjects:
        archive_name = f"{subject_dir_name}.tar.gz"
        local_path = DEXYCB_DIR / archive_name
        extract_marker = DEXYCB_DIR / f".{archive_name}.done"

        if extract_marker.exists():
            print(f"  {subject_key} already extracted")
            continue

        gdrive_id = GDRIVE_IDS.get(archive_name)
        if not local_path.exists() and gdrive_id:
            print(f"Downloading {subject_key} ({archive_name}) from Google Drive...")
            try:
                _gdown(gdrive_id, local_path)
                print(f"  Downloaded {local_path.stat().st_size / 1e9:.1f} GB")
            except Exception as e:
                print(f"  FAILED: {e}")
                continue
        elif not local_path.exists():
            print(f"  FAILED: no Google Drive ID for {archive_name}")
            continue

        print(f"  Extracting {subject_key}...")
        with tarfile.open(str(local_path), "r:gz") as tf:
            tf.extractall(str(DEXYCB_DIR))
        extract_marker.touch()

    print("Download complete.")


def _get_target_subjects(args):
    """Get list of (key, dir_name) for subjects to process."""
    if args.subject:
        key = args.subject
        if key not in DEXYCB_SUBJECTS:
            print(f"WARNING: unknown subject {key}")
            return []
        return [(key, DEXYCB_SUBJECTS[key])]
    return list(DEXYCB_SUBJECTS.items())


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def _load_intrinsics(serial):
    """Load camera intrinsics from calibration."""
    intr_path = DEXYCB_DIR / "calibration" / "intrinsics" / f"{serial}_640x480.yml"
    if not intr_path.exists():
        return None
    with open(intr_path) as f:
        data = yaml.unsafe_load(f)  # DexYCB uses !!python/tuple tags
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = data["color"]["fx"]
    K[1, 1] = data["color"]["fy"]
    K[0, 2] = data["color"]["ppx"]
    K[1, 2] = data["color"]["ppy"]
    return K


def _load_betas(subject_dir_name):
    """Load MANO betas from calibration for a subject.

    The MANO calibration folders are named `mano_<calib-date>_..._subject-NN_right` and
    the calibration date differs from the session date for some subjects (subject-07,
    subject-09), so they are matched by the subject token, which is unique.
    """
    calib_dir = DEXYCB_DIR / "calibration"
    mano_dirs = sorted(calib_dir.glob("mano_*"))
    subj_token = "subject-" + subject_dir_name.split("-")[-1]  # "20200928-subject-07" -> "subject-07"
    for md in mano_dirs:
        if subj_token in md.name:
            mano_yml = md / "mano.yml"
            if mano_yml.exists():
                with open(mano_yml) as f:
                    data = yaml.safe_load(f)
                return np.array(data["betas"], dtype=np.float32)
    return None


def _load_ycb_mesh(ycb_name):
    """Load a YCB object CAD mesh (meters). DexYCB pose_y places it directly."""
    mesh_path = DEXYCB_DIR / "models" / ycb_name / "textured_simple.obj"
    if not mesh_path.exists():
        raise FileNotFoundError(f"No mesh for {ycb_name} under {DEXYCB_DIR / 'models'}")
    verts, faces = load_mesh(str(mesh_path))
    return verts.astype(np.float32), faces.astype(np.int32)


def _convert_one_sequence_canonical(subject_key, subject_dir_name, seq_name, cam_serial, cam_idx, out_root, force, write_video=True):
    """Write the camera-frame canonical of one (sequence, camera). Returns segment_name or None.

    Everything stays in the camera's optical frame (gravity alignment happens in the dataset),
    in meters as given by pose_m/pose_y. Left hand: orient + cam_t are mirrored into the
    right-model x-mirror convention; the finger pose stays the right-basis PCA decode, which
    already matches the right model.
    """
    seq_dir = DEXYCB_DIR / subject_dir_name / seq_name
    cam_dir = seq_dir / cam_serial
    meta_path = seq_dir / "meta.yml"
    if not cam_dir.exists() or not meta_path.exists():
        return None
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    ycb_ids = meta["ycb_ids"]
    active_side = meta["mano_sides"][0]
    grasp_ind = meta.get("ycb_grasp_ind", 0)
    T = meta["num_frames"]
    ycb_name = YCB_OBJECTS.get(ycb_ids[grasp_ind])
    if ycb_name is None:
        return None
    seq_short = seq_name.replace(subject_dir_name.split("-")[0] + "_", "")
    segment_name = f"dexycb_{subject_key}_{seq_short}_cam{cam_idx}"
    seg_dir = Path(out_root) / segment_name
    canonical_dir = seg_dir / "canonical"
    if not force and (canonical_dir / "canonical.npz").exists():
        return segment_name
    betas = _load_betas(subject_dir_name)
    if betas is None:
        return None
    try:
        obj_canonical_verts, obj_faces = _load_ycb_mesh(ycb_name)
    except FileNotFoundError:
        return None

    hand_orient = {s: np.zeros((T, 3), np.float32) for s in ("left", "right")}
    hand_pose = {s: np.zeros((T, 45), np.float32) for s in ("left", "right")}
    hand_cam_t = {s: np.zeros((T, 3), np.float32) for s in ("left", "right")}
    hand_valid = {s: np.zeros(T, bool) for s in ("left", "right")}
    obj_rotation = np.zeros((T, 3), np.float32)
    obj_translation = np.zeros((T, 3), np.float32)

    for t in range(T):
        label_path = cam_dir / f"labels_{t:06d}.npz"
        if not label_path.exists():
            continue
        data = np.load(str(label_path))
        pose_m = data["pose_m"]  # (1,51) orient(3)+pca(45)+trans(3), meters
        pose_y = data["pose_y"]  # (N,3,4) object R|t, meters
        if np.any(pose_m[0] != 0):
            global_orient = pose_m[0, :3].astype(np.float32)
            finger_aa = mano_pca_to_aa(pose_m[0, 3:48].astype(np.float32), 45, MANO_MODEL)
            cam_t_m = pose_m[0, 48:51].astype(np.float32)
            if active_side == "left":
                global_orient, _finger_m, cam_t_m = mirror_left_hand_params(global_orient.reshape(1, 3), finger_aa.reshape(1, 45), cam_t_m.reshape(1, 3))
                global_orient = global_orient[0]
                cam_t_m = cam_t_m[0]
                # finger_aa keeps the right-basis PCA decode
            hand_orient[active_side][t] = global_orient
            hand_pose[active_side][t] = finger_aa
            hand_cam_t[active_side][t] = cam_t_m
            hand_valid[active_side][t] = True
        if grasp_ind < len(pose_y):
            R_obj = pose_y[grasp_ind][:3, :3]
            t_obj = pose_y[grasp_ind][:3, 3]
            if np.any(R_obj != 0):
                obj_rotation[t] = rotmat_to_aa(R_obj.reshape(1, 3, 3))[0].astype(np.float32)
                obj_translation[t] = t_obj.astype(np.float32)

    for side in ("left", "right"):
        inv = ~hand_valid[side]
        hand_orient[side][inv] = 0
        hand_pose[side][inv] = 0
        hand_cam_t[side][inv] = 0
    if hand_valid[active_side].sum() < 10:
        return None

    hand_verts = {}
    for side in ("left", "right"):
        if hand_valid[side].any():
            hand_verts[side] = mano_fk_verts(hand_orient[side], hand_pose[side], betas, hand_cam_t[side], side, str(MANO_DIR))
        else:
            hand_verts[side] = np.zeros((T, 778, 3), np.float32)
        hand_verts[side][~hand_valid[side]] = 0

    obj_scale = np.ones(3, np.float32)
    obj_verts = np.zeros((T, len(obj_canonical_verts), 3), np.float32)
    for t in range(T):
        if np.any(obj_rotation[t] != 0) or np.any(obj_translation[t] != 0):
            R_t = Rotation.from_rotvec(obj_rotation[t]).as_matrix()
            obj_verts[t] = (R_t @ (obj_canonical_verts * obj_scale).T).T + obj_translation[t]

    gravity_direction = support_plane_height = None
    depth_path = cam_dir / f"aligned_depth_to_color_{T // 2:06d}.png"
    intr = _load_intrinsics(cam_serial)
    if depth_path.exists() and intr is not None:
        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_img is not None:
            res = estimate_gravity(depth_img.astype(np.float32) / 1000.0, intr)
            if res is not None:
                gravity_direction, support_plane_height = res

    canonical_dir.mkdir(parents=True, exist_ok=True)
    video_path = seg_dir / "video.mp4"
    if write_video and (force or not video_path.exists()):
        if sorted(cam_dir.glob("color_*.jpg")):
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-framerate",
                        str(FPS),
                        "-i",
                        str(cam_dir / "color_%06d.jpg"),
                        "-c:v",
                        "libx264",
                        "-crf",
                        "18",
                        "-pix_fmt",
                        "yuv420p",
                        "-loglevel",
                        "error",
                        str(video_path),
                    ],
                    check=True,
                    capture_output=True,
                )
            except Exception as e:
                print(f"    WARNING: video encode failed for {segment_name}: {e}")

    save = {
        "frame_indices": np.arange(T, dtype=np.int64),
        "fps": np.float32(FPS),
        "obj_canonical_verts": obj_canonical_verts,
        "obj_faces": obj_faces,
        "obj_rotation": obj_rotation,
        "obj_translation": obj_translation,
        "obj_scale": obj_scale,
        "obj_verts": obj_verts,
    }
    for side in ("left", "right"):
        save[f"hand_{side}_orient"] = hand_orient[side]
        save[f"hand_{side}_pose"] = hand_pose[side]
        save[f"hand_{side}_cam_t"] = hand_cam_t[side]
        save[f"hand_{side}_betas"] = betas
        save[f"hand_{side}_valid"] = hand_valid[side]
        save[f"hand_{side}_verts"] = hand_verts[side]
    if gravity_direction is not None:
        save["gravity_direction"] = gravity_direction.astype(np.float32)
    if support_plane_height is not None:
        save["support_plane_height"] = np.float32(support_plane_height)
    np.savez_compressed(str(canonical_dir / "canonical.npz"), **save)

    category = ycb_name.split("_", 1)[1] if "_" in ycb_name else ycb_name
    metadata = {
        "source_dataset": "dexycb",
        "segment_id": segment_name,
        "confidence_tier": "gt_fitted",
        "has_depth": True,
        "has_scene": video_path.exists(),
        "has_gravity": gravity_direction is not None,
        "has_support_plane": support_plane_height is not None,
        "has_native_hand_pose": True,
        "has_native_object_pose": True,
        "object_is_rigid": True,
        "primary_object_id": ycb_name,
        "primary_object_category": category,
        "bimanual": False,
        "viewpoint": "exocentric",
        "hand_source": "gt_fitted",
        "object_source": "gt_ycb_cad",
        "depth_source": "realsense_d415",
        "mano_convention": "right_model_x_mirror",
        "dexycb_subject": subject_key,
        "dexycb_camera_serial": cam_serial,
        "dexycb_camera_idx": cam_idx,
        "active_side": active_side,
        "n_frames": T,
        "fps": FPS,
    }
    with open(canonical_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return segment_name


def run_convert_canonical_step(args):
    """Produce camera-frame training canonicals: results/dexycb_*_cam{0,2,5}/canonical/."""
    subjects = _get_target_subjects(args)
    out_root = args.out_root or RESULTS_DIR
    seq_filter = set(args.seq.split(",")) if args.seq else None
    done = skipped = failed = 0
    segments = []
    for subject_key, subject_dir_name in subjects:
        subject_dir = DEXYCB_DIR / subject_dir_name
        if not subject_dir.exists():
            print(f"  {subject_key}: dir not found, skipping")
            continue
        seq_names = sorted([d.name for d in subject_dir.iterdir() if d.is_dir()])
        if seq_filter:
            seq_names = [s for s in seq_names if s in seq_filter or s.split("_")[-1] in seq_filter]
        if args.max_seqs:
            seq_names = seq_names[: args.max_seqs]
        for seq_name in seq_names:
            for cam_serial, cam_idx in zip(SELECTED_CAMERAS, SELECTED_CAMERA_INDICES):
                try:
                    seg = _convert_one_sequence_canonical(subject_key, subject_dir_name, seq_name, cam_serial, cam_idx, out_root, args.force, write_video=not args.no_video)
                    if seg:
                        segments.append(seg)
                        done += 1
                    else:
                        skipped += 1
                except Exception as e:
                    print(f"    ERROR {subject_key}/{seq_name}/cam{cam_idx}: {e}")
                    traceback.print_exc()
                    failed += 1
        print(f"  {subject_key}: {done} canonicals so far")
    print(f"  Canonical convert: {done} created, {skipped} skipped, {failed} failed -> {out_root}")
    return segments


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(prog="python -m gtf.datasets.dexycb", description="Ingest DexYCB into canonical format")
    parser.add_argument("--start-from", choices=PIPELINE_STEPS, default=PIPELINE_STEPS[0])
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--subject", type=str, help="Process single subject (e.g., s01)")
    parser.add_argument("--out-root", type=str, default="", help="Output root (default: results).")
    parser.add_argument("--max-seqs", type=int, default=0, help="Cap sequences per subject (validation runs).")
    parser.add_argument("--seq", type=str, default="", help="Comma list of sequence names / short ids to restrict to.")
    parser.add_argument("--no-video", action="store_true", help="Skip video encoding (faster validation; full runs need it).")
    args = parser.parse_args()

    start_idx = PIPELINE_STEPS.index(args.start_from)
    active_steps = set(PIPELINE_STEPS[start_idx:])

    if "download" in active_steps:
        run_download_step(args)

    if "convert" in active_steps:
        run_convert_canonical_step(args)


if __name__ == "__main__":
    main()
