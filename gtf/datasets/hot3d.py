#!/usr/bin/env python3
"""Ingest HOT3D-Clips (Aria train split) into training canonicals.

Downloads clip tars and object models from HuggingFace, parses the native
annotations (MoCap MANO + object 6DoF in the world frame), and writes one egocentric
canonical per (clip, manipulated object) pair under
`results/hot3d_aria_{clip}_obj{NN}/` (canonical/ + video.mp4 of the RGB stream),
in the single static anchor-camera frame (see `_write_egocentric_canonical`).

Object meshes are converted once from GLB to OBJ in their native orientation
(millimeters to meters only).

Usage:
    python -m gtf.datasets.hot3d                                   # full: download → convert
    python -m gtf.datasets.hot3d --start-from download --max-clips 10
    python -m gtf.datasets.hot3d --start-from convert --clip clip-001849
    python -m gtf.datasets.hot3d --start-from convert --force
"""

import argparse
import json
import os
import tarfile
import traceback
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from gtf.paths import MANO_DIR, mano_model_path, REPO_ROOT
from gtf.utils import RESULTS_DIR, rprint as print
from gtf.utils.canonical import (
    mano_fk_verts,
    mano_pca_to_aa,
    mirror_left_hand_params,
)

ROOT = REPO_ROOT

PIPELINE_STEPS = ["download", "convert"]

HOT3D_DIR = ROOT / "datasets" / "hot3d"
MANO_MODEL = mano_model_path("right")

# Native-orientation OBJ meshes (mm → m only)
OBJ_MESH_DIR = ROOT / "datasets" / "hot3d" / "object_models_obj"

HF_REPO = "bop-benchmark/hot3d"
CAMERA_STREAM = "214-1"
FPS = 30.0

DEFAULT_PROX_MIN_FRAMES = 20
# Manipulation gate: a (hand, object) pair counts when the hand mesh comes within
# DEFAULT_SURFACE_CONTACT_M of the object surface (nearest mesh vertex) for at least
# DEFAULT_PROX_MIN_FRAMES frames. It does not need motion, so a held-still object counts.
# The object pose origin can sit 50-100 mm from the graspable surface, so the distance is
# measured to the surface; a cheap wrist-to-origin pre-filter (< DEFAULT_PREFILTER_ORIGIN_M)
# narrows the candidates first. Real grasps measure ~16 mm hand-to-surface, drive-bys > 50 mm.
DEFAULT_SURFACE_CONTACT_M = 0.02
DEFAULT_PREFILTER_ORIGIN_M = 0.30

# Normalize HOT3D BOP category names (e.g. "mug_patterned") to lowercase
# simple-category tokens (e.g. "mug", "bowl", "bottle") for the canonical's
# `primary_object_category`. A name without an entry falls back to its first token.
CATEGORY_NORMALIZE = {
    "holder_black": "holder",
    "bowl": "bowl",
    "plate_bamboo": "plate",
    "spoon_wooden": "spoon",
    "potato_masher": "masher",
    "spatula_red": "spatula",
    "coffee_pot": "pot",
    "mug_patterned": "mug",
    "mug_white": "mug",
    "can_soup": "can",
    "can_parmesan": "can",
    "can_tomato_sauce": "can",
    "bottle_mustard": "bottle",
    "bottle_bbq": "bottle",
    "bottle_ranch": "bottle",
    "vase": "vase",
    "carton_milk": "carton",
    "carton_oj": "carton",
    "flask": "flask",
    "food_waffles": "food",
    "food_vegetables": "food",
    "dumbbell_5lb": "dumbbell",
    "aria_small": "device",
    "cellphone": "phone",
    "holder_gray": "holder",
    "birdhouse_toy": "toy",
    "dino_toy": "toy",
    "keyboard": "keyboard",
    "whiteboard_eraser": "eraser",
    "puzzle_toy": "toy",
    "mouse": "mouse",
    "whiteboard_marker": "marker",
    "dvd_remote": "remote",
}


# ---------------------------------------------------------------------------
# Object model info
# ---------------------------------------------------------------------------

# BOP ID → (object_uid, category name) mapping from HOT3D documentation
BOP_OBJECTS = {
    1: ("232501673989606", "holder_black"),
    2: ("194930206998778", "bowl"),
    3: ("248247003992116", "plate_bamboo"),
    4: ("225397651484143", "spoon_wooden"),
    5: ("79582884925181", "potato_masher"),
    6: ("270231216246839", "spatula_red"),
    7: ("228358276546933", "coffee_pot"),
    8: ("117658302265452", "mug_patterned"),
    9: ("223371871635142", "mug_white"),
    10: ("98604936546412", "can_soup"),
    11: ("171735388712249", "can_parmesan"),
    12: ("183136364331389", "can_tomato_sauce"),
    13: ("261746112525368", "bottle_mustard"),
    14: ("106434519822892", "bottle_bbq"),
    15: ("70709727230291", "bottle_ranch"),
    16: ("163340972252026", "vase"),
    17: ("125863066770940", "carton_milk"),
    18: ("204462113746498", "carton_oj"),
    19: ("4111539686391", "flask"),
    20: ("253405647833885", "food_waffles"),
    21: ("96945373046044", "food_vegetables"),
    22: ("27078911029651", "dumbbell_5lb"),
    23: ("111305855142671", "aria_small"),
    24: ("5462893327580", "cellphone"),
    25: ("143541090750632", "holder_gray"),
    26: ("195041665639898", "birdhouse_toy"),
    27: ("208243983021975", "dino_toy"),
    28: ("37787722328019", "keyboard"),
    29: ("258906041248094", "whiteboard_eraser"),
    30: ("106957734975303", "puzzle_toy"),
    31: ("265826671143948", "mouse"),
    32: ("238686662724712", "whiteboard_marker"),
    33: ("249541253457812", "dvd_remote"),
}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _get_train_aria_clip_ids():
    """Read clip IDs from the split file."""
    split_path = HOT3D_DIR / "split" / "hot3d_train_train.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    clips = set()
    with open(split_path) as f:
        for line in f:
            clip_id = line.strip().split(",")[0]
            clips.add(clip_id)
    return sorted(clips)


def run_download_step(args):
    """Download HOT3D train_aria tars and object models from HuggingFace."""
    from huggingface_hub import hf_hub_download

    obj_model_dir = HOT3D_DIR / "object_models"
    if not any(obj_model_dir.glob("*.ply")) and not any(obj_model_dir.glob("*.json")):
        print("Downloading object models...")
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=HF_REPO, repo_type="dataset", allow_patterns=["object_models/*"], local_dir=str(HOT3D_DIR))
            n_files = len(list(obj_model_dir.iterdir())) if obj_model_dir.exists() else 0
            print(f"  Downloaded {n_files} object model files")
        except Exception as e:
            print(f"  WARNING: Could not download object models: {e}")
            print("  You may need to download them manually.")
    else:
        print(f"Object models already present in {obj_model_dir}")

    all_clips = _get_train_aria_clip_ids()
    if args.clip:
        clips = [args.clip] if args.clip in all_clips else []
        if not clips:
            print(f"WARNING: {args.clip} not found in train_aria split")
            return []
    elif args.max_clips:
        clips = all_clips[: args.max_clips]
    else:
        clips = all_clips

    print(f"Downloading {len(clips)}/{len(all_clips)} train_aria clips...")
    downloaded = []
    for i, clip_id in enumerate(clips):
        tar_name = f"train_aria/{clip_id}.tar"
        local_path = HOT3D_DIR / tar_name
        if local_path.exists() and local_path.stat().st_size > 1000:
            downloaded.append(clip_id)
            continue
        try:
            hf_hub_download(repo_id=HF_REPO, filename=tar_name, repo_type="dataset", local_dir=str(HOT3D_DIR))
            downloaded.append(clip_id)
            if (i + 1) % 50 == 0:
                print(f"  [{i + 1}/{len(clips)}] downloaded")
        except Exception as e:
            print(f"  FAILED {clip_id}: {e}")

    print(f"  Downloaded {len(downloaded)} clips")
    return downloaded


# ---------------------------------------------------------------------------
# Clip parsing
# ---------------------------------------------------------------------------


def _parse_tar(tar_path, with_images=False):
    """Parse a HOT3D clip tar into structured per-frame data.

    With ``with_images`` the JPEG of the RGB stream (CAMERA_STREAM) is kept per frame.

    Returns:
        dict with keys: frames (list of frame dicts), betas (10,), clip_id
    """
    frames = {}
    betas = None
    clip_id = Path(tar_path).stem

    with tarfile.open(tar_path, "r") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            name = os.path.basename(member.name)

            if name == "__hand_shapes.json__":
                data = json.load(tf.extractfile(member))
                betas = np.array(data["mano"], dtype=np.float32)
                continue

            parts = name.split(".")
            if len(parts) < 2:
                continue

            frame_id = parts[0]
            if not frame_id.isdigit():
                continue

            if frame_id not in frames:
                frames[frame_id] = {"frame_id": frame_id}

            if name.endswith(".hands.json"):
                frames[frame_id]["hands"] = json.load(tf.extractfile(member))
            elif name.endswith(".objects.json"):
                frames[frame_id]["objects"] = json.load(tf.extractfile(member))
            elif name.endswith(".cameras.json"):
                frames[frame_id]["cameras"] = json.load(tf.extractfile(member))
            elif with_images and name.endswith(f".image_{CAMERA_STREAM}.jpg"):
                frames[frame_id]["image"] = tf.extractfile(member).read()

    sorted_ids = sorted(frames.keys(), key=lambda x: int(x))
    frame_list = [frames[fid] for fid in sorted_ids]

    return {"frames": frame_list, "betas": betas, "clip_id": clip_id}


def _parse_camera_transform(cam_data):
    """Extract T_world_from_camera for the RGB stream from cameras.json."""
    for stream_key, stream_data in cam_data.items():
        stream_label = stream_data.get("stream_label", "")
        if CAMERA_STREAM in stream_label or CAMERA_STREAM in stream_key:
            xform = stream_data.get("T_world_from_camera", {})
            q = xform.get("quaternion_wxyz", None)
            t = xform.get("translation_xyz", None)
            if q is not None and t is not None:
                R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R
                T[:3, 3] = t
                return T
    return None


def _parse_object_pose(obj_data):
    """Parse T_world_from_object from objects.json entry."""
    xform = obj_data.get("T_world_from_object", {})
    q = xform.get("quaternion_wxyz", None)
    t = xform.get("translation_xyz", None)
    if q is None or t is None:
        return None, None
    R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    return R.astype(np.float64), np.array(t, dtype=np.float64)


# ---------------------------------------------------------------------------
# Manipulation detection
# ---------------------------------------------------------------------------


# MANO layer and object meshes of the surface-contact gate, loaded once.
_DET_MANO_LAYER = None
_NATIVE_MESH_CACHE: dict[int, np.ndarray] = {}


def _det_fk_world(orient_aa, pose_aa, betas, cam_t, side):
    """World-frame MANO verts via a cached right-hand layer (left = x-mirror)."""
    import torch

    global _DET_MANO_LAYER
    if _DET_MANO_LAYER is None:
        from gtf.utils.canonical import _import_manolayer

        ManoLayer = _import_manolayer()
        _DET_MANO_LAYER = ManoLayer(side="right", use_pca=False, ncomps=45, flat_hand_mean=True, mano_root=str(MANO_DIR))
    T = orient_aa.shape[0]
    with torch.no_grad():
        full_pose = torch.cat([torch.from_numpy(orient_aa).float(), torch.from_numpy(pose_aa).float()], dim=1)
        bt = torch.from_numpy(betas).float().unsqueeze(0).expand(T, -1)
        verts, _ = _DET_MANO_LAYER(full_pose, bt, torch.from_numpy(cam_t).float())
        verts = verts.numpy().astype(np.float32) / 1000.0
    if side == "left":
        verts[:, :, 0] *= -1
    return verts


def _native_obj_verts(bop_id):
    """Native-orientation object vertices (meters), subsampled for the KDTree; cached per object."""
    if bop_id not in _NATIVE_MESH_CACHE:
        from gtf.utils.canonical import load_mesh

        verts, _ = load_mesh(str(_convert_glb_to_obj(bop_id)))
        v = np.asarray(verts, dtype=np.float32)
        if len(v) > 500:
            v = v[np.linspace(0, len(v) - 1, 500).astype(int)]
        _NATIVE_MESH_CACHE[bop_id] = v
    return _NATIVE_MESH_CACHE[bop_id]


def _surface_contact_frames(frames, obj_key, bop_id, side, betas, contact_m):
    """Count frames where the hand mesh is within `contact_m` of the object surface.

    FK the hand in the world frame (left orient/trans mirrored, fingers un-mirrored)
    and place the native object mesh per frame. A held-still object still counts.
    """
    T = len(frames)
    orient = np.zeros((T, 3), np.float32)
    pose = np.zeros((T, 45), np.float32)
    cam_t = np.zeros((T, 3), np.float32)
    val = np.zeros(T, bool)
    orot = np.zeros((T, 3), np.float32)
    otr = np.zeros((T, 3), np.float32)
    ov = np.zeros(T, bool)
    for t, f in enumerate(frames):
        h = f.get("hands", {}).get(side, {})
        mp = h.get("mano_pose", {}) if "mano_pose" in h else {}
        wx, th = mp.get("wrist_xform"), mp.get("thetas")
        if wx is not None and th is not None and len(wx) >= 6 and len(th) >= 15:
            orient[t] = wx[:3]
            cam_t[t] = wx[3:6]
            pose[t] = mano_pca_to_aa(np.array(th, np.float32), 15, MANO_MODEL)
            val[t] = True
        o = f.get("objects", {}).get(obj_key)
        if o:
            d = o[0] if isinstance(o, list) and o else o
            R, tt = _parse_object_pose(d)
            if R is not None:
                orot[t] = Rotation.from_matrix(R).as_rotvec()
                otr[t] = tt
                ov[t] = True
    both = val & ov
    if both.sum() < 5:
        return 0
    if side == "left":
        orient, _pose_m, cam_t = mirror_left_hand_params(orient, pose, cam_t)
    hv = _det_fk_world(orient, pose, betas, cam_t, side)
    nv = _native_obj_verts(bop_id)
    cf = 0
    for t in np.where(both)[0]:
        ow = nv @ Rotation.from_rotvec(orot[t]).as_matrix().T.astype(np.float32) + otr[t]
        if cKDTree(ow).query(hv[t])[0].min() < contact_m:
            cf += 1
    return cf


def _detect_manipulated_objects(
    clip_data,
    prox_min_frames,
    *,
    surface_contact_m: float = DEFAULT_SURFACE_CONTACT_M,
    prefilter_origin_m: float = DEFAULT_PREFILTER_ORIGIN_M,
):
    """Detect which objects are manipulated by which hands (geometry-aware).

    A (hand, object) pair counts as manipulation when the hand mesh comes within
    `surface_contact_m` of the object surface for >= `prox_min_frames` frames.
    A cheap wrist-to-pose-origin pre-filter (< `prefilter_origin_m`) narrows
    candidates before the FK contact check. A held-still object still counts.

    Returns list of dicts: {bop_id, obj_key, category, sides: set('left','right')}.
    """
    frames = clip_data["frames"]
    T = len(frames)
    betas = clip_data.get("betas")
    betas = np.zeros(10, np.float32) if betas is None else np.asarray(betas, np.float32)

    # Collect per-frame wrist positions (world space) and object positions
    hand_positions = {"left": [None] * T, "right": [None] * T}
    object_positions = {}  # uid → list of (frame_idx, position)

    for t, frame in enumerate(frames):
        hands = frame.get("hands", {})
        for side in ("left", "right"):
            if side in hands and "mano_pose" in hands[side]:
                wrist_xform = hands[side]["mano_pose"].get("wrist_xform", None)
                if wrist_xform is not None and len(wrist_xform) >= 6:
                    hand_positions[side][t] = np.array(wrist_xform[3:6], dtype=np.float64)

        objects = frame.get("objects", {})
        for obj_key, obj_entry in objects.items():
            # objects.json values are lists wrapping a dict: [{pose_dict}]
            obj_data = obj_entry[0] if isinstance(obj_entry, list) and obj_entry else obj_entry
            _, obj_t = _parse_object_pose(obj_data)
            if obj_t is not None:
                if obj_key not in object_positions:
                    object_positions[obj_key] = []
                object_positions[obj_key].append((t, obj_t))

    # For each object, check proximity to each hand
    # objects.json keys are BOP IDs as strings (e.g., "1", "24")
    results = {}
    for obj_key, obj_frames in object_positions.items():
        try:
            bop_id = int(obj_key)
        except ValueError:
            continue
        if bop_id not in BOP_OBJECTS:
            continue

        interacting_sides = set()
        for side in ("left", "right"):
            # Generous pre-filter on the wrist-to-origin distance (the origin can sit far
            # from the surface); the FK contact check decides.
            near = sum(1 for t, obj_pos in obj_frames if hand_positions[side][t] is not None and float(np.linalg.norm(hand_positions[side][t] - obj_pos)) < prefilter_origin_m)
            if near < prox_min_frames:
                continue
            if _surface_contact_frames(frames, obj_key, bop_id, side, betas, surface_contact_m) >= prox_min_frames:
                interacting_sides.add(side)

        if interacting_sides:
            _, category = BOP_OBJECTS[bop_id]
            results[obj_key] = {"bop_id": bop_id, "obj_key": obj_key, "category": category, "sides": interacting_sides}

    return list(results.values())


# ---------------------------------------------------------------------------
# Mesh conversion
# ---------------------------------------------------------------------------


def _convert_glb_to_obj(bop_id: int) -> Path:
    """Convert a HOT3D GLB to a native-orientation OBJ in meters (mm → m only).

    HOT3D meshes ship as GLB files in BOP millimeter units. Only the scale is applied:
    the per-frame pose (T_world_from_object) is expressed in the native GLB frame, so
    no axis change may be baked into the mesh.

    Returns the absolute path to the OBJ on disk.
    """
    import trimesh

    OBJ_MESH_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OBJ_MESH_DIR / f"obj_{bop_id:06d}.obj"
    if out_path.exists():
        return out_path

    glb_path = HOT3D_DIR / "object_models" / f"obj_{bop_id:06d}.glb"
    if not glb_path.exists():
        ply_path = HOT3D_DIR / "object_models" / f"obj_{bop_id:06d}.ply"
        if ply_path.exists():
            loaded = trimesh.load(str(ply_path))
        else:
            raise FileNotFoundError(f"No mesh for BOP object {bop_id} in {HOT3D_DIR / 'object_models'}")
    else:
        loaded = trimesh.load(str(glb_path))

    # Flatten Scene → Trimesh while preserving the first geometry's TextureVisuals
    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) == 0:
            raise ValueError(f"Empty scene in {glb_path}")
        # Use the first (and typically only) geometry to keep TextureVisuals intact.
        # Scene.to_geometry() merges textures which can lose PBR info.
        mesh = list(loaded.geometry.values())[0]
        if len(loaded.geometry) > 1:
            print(f"  note: {glb_path.name} has {len(loaded.geometry)} geoms, using first")
    else:
        mesh = loaded

    mesh.vertices = mesh.vertices * 0.001

    # Export as OBJ. Trimesh writes an .mtl sidecar and a PNG atlas alongside.
    mesh.export(str(out_path))
    return out_path


def _load_object_mesh(bop_id):
    """Load HOT3D object mesh verts/faces from the OBJ (native GLB orientation, meters; see `_convert_glb_to_obj`)."""
    from gtf.utils.canonical import load_mesh

    verts, faces = load_mesh(str(_convert_glb_to_obj(bop_id)))
    return verts.astype(np.float32), faces.astype(np.int32)


def _write_video(frames, out_path: Path) -> bool:
    """Write the clip's RGB-stream JPEGs as video.mp4 (one video frame per canonical row)."""
    import cv2

    first = next((f["image"] for f in frames if f.get("image")), None)
    if first is None:
        return False
    first_img = cv2.imdecode(np.frombuffer(first, dtype=np.uint8), cv2.IMREAD_COLOR)
    if first_img is None:
        return False
    h, w = first_img.shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    if not writer.isOpened():
        return False
    last = first_img
    for f in frames:
        img = cv2.imdecode(np.frombuffer(f["image"], dtype=np.uint8), cv2.IMREAD_COLOR) if f.get("image") else None
        if img is None:
            img = last
        writer.write(img)
        last = img
    writer.release()
    cap = cv2.VideoCapture(str(tmp_path))
    ok = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(frames)
    cap.release()
    if ok:
        os.replace(tmp_path, out_path)
    else:
        tmp_path.unlink(missing_ok=True)
    return ok


def _convert_one_clip(clip_data, manip_obj, args):
    """Write the egocentric canonical (and video) of one clip + manipulated object.

    Args:
        clip_data: parsed clip dict from _parse_tar
        manip_obj: dict with bop_id, obj_key, category, sides
        args: CLI args
    Returns:
        segment_name, or None if the segment was skipped or failed
    """
    clip_id = clip_data["clip_id"]
    bop_id = manip_obj["bop_id"]
    segment_name = f"hot3d_aria_{clip_id}_obj{bop_id:02d}"
    seg_dir = Path(args.out_root or RESULTS_DIR) / segment_name

    if not args.force and (seg_dir / "canonical" / "canonical.npz").exists():
        return segment_name

    betas = clip_data["betas"]
    if betas is None:
        print(f"  WARNING: {clip_id} missing betas, skipping")
        return None

    # Load the object mesh (converts GLB → OBJ once, native orientation).
    try:
        obj_verts, obj_faces = _load_object_mesh(bop_id)
    except FileNotFoundError as e:
        print(f"  WARNING: {e}, skipping")
        return None

    if not args.no_video and (args.force or not (seg_dir / "video.mp4").exists()):
        _write_video(clip_data["frames"], seg_dir / "video.mp4")
    try:
        written = _write_egocentric_canonical(clip_data, manip_obj, seg_dir, obj_verts, obj_faces, betas)
    except Exception as e:
        print(f"  WARNING: egocentric canonical export failed for {segment_name}: {e}")
        traceback.print_exc()
        return None
    return segment_name if written else None


def _write_egocentric_canonical(clip_data, manip_obj, seg_dir, native_verts, native_faces, betas):
    """Write the segment's egocentric canonical ({seg_dir}/canonical/).

    The camera-frame fields live in the single static anchor Aria camera (the clip's
    middle-frame pose), not the per-frame moving camera (that would add head motion to
    every trajectory) and not the world frame (scene preprocessing projects the anchor-frame
    verts into the video frame). Like DexYCB, the canonical is a static camera plus a
    gravity_direction. The world-frame fields and per-frame camera_extrinsics are stored
    too, for re-anchoring to the present camera.

    Left hands: only orient + trans are mirrored into the right-model convention; the
    finger PCA decode stays un-mirrored (same rule as DexYCB).
    """
    frames = clip_data["frames"]
    clip_id = clip_data["clip_id"]
    bop_id = manip_obj["bop_id"]
    active_sides = manip_obj["sides"]
    obj_key = manip_obj["obj_key"]
    segment_name = seg_dir.name
    T = len(frames)

    # Anchor frame = T // 2, falling back to the nearest frame with a camera transform.
    anchor = T // 2
    T_world_cam = _parse_camera_transform(frames[anchor].get("cameras")) if frames[anchor].get("cameras") else None
    if T_world_cam is None:
        for off in range(1, T):
            for a in (anchor - off, anchor + off):
                if 0 <= a < T and frames[a].get("cameras"):
                    cand = _parse_camera_transform(frames[a].get("cameras"))
                    if cand is not None:
                        T_world_cam, anchor = cand, a
                        break
            if T_world_cam is not None:
                break
    if T_world_cam is None:
        print(f"  WARNING: {segment_name} has no anchor camera transform — skipping canonical")
        return False

    T_cam_world = np.linalg.inv(T_world_cam)
    R_cw = T_cam_world[:3, :3].astype(np.float32)  # world → anchor camera
    t_cw = T_cam_world[:3, 3].astype(np.float32)
    camera_extrinsics = np.repeat(T_cam_world.astype(np.float32)[None, :, :], T, axis=0)
    for i, frame in enumerate(frames):
        cams_i = frame.get("cameras")
        if not cams_i:
            continue
        T_world_cam_i = _parse_camera_transform(cams_i)
        if T_world_cam_i is not None:
            camera_extrinsics[i] = np.linalg.inv(T_world_cam_i).astype(np.float32)

    hand_orient = {s: np.zeros((T, 3), dtype=np.float32) for s in ("left", "right")}
    hand_pose = {s: np.zeros((T, 45), dtype=np.float32) for s in ("left", "right")}
    hand_cam_t = {s: np.zeros((T, 3), dtype=np.float32) for s in ("left", "right")}
    hand_valid = {s: np.zeros(T, dtype=bool) for s in ("left", "right")}
    # World-frame params; their FK gives the reference verts for the wrist-pivot
    # cam_t correction below.
    hand_orient_w = {s: np.zeros((T, 3), dtype=np.float32) for s in ("left", "right")}
    hand_cam_t_w = {s: np.zeros((T, 3), dtype=np.float32) for s in ("left", "right")}
    obj_rotation = np.zeros((T, 3), dtype=np.float32)
    obj_translation = np.zeros((T, 3), dtype=np.float32)
    obj_rotation_world = np.zeros((T, 3), dtype=np.float32)
    obj_translation_world = np.zeros((T, 3), dtype=np.float32)
    obj_valid = np.zeros(T, dtype=bool)

    for t, frame in enumerate(frames):
        hands = frame.get("hands", {})
        for side in ("left", "right"):
            if side not in active_sides or side not in hands or "mano_pose" not in hands[side]:
                continue
            mano_pose = hands[side]["mano_pose"]
            wrist_xform = mano_pose.get("wrist_xform")
            thetas = mano_pose.get("thetas")
            if wrist_xform is None or thetas is None or len(wrist_xform) < 6 or len(thetas) < 15:
                continue

            orient_world = np.array(wrist_xform[:3], dtype=np.float32)
            trans_world = np.array(wrist_xform[3:6], dtype=np.float32)
            finger_aa = mano_pca_to_aa(np.array(thetas, dtype=np.float32), 15, MANO_MODEL)

            # World-frame params (left mirrored in the world frame, fingers un-mirrored).
            if side == "left":
                o_wm, _f_wm, t_wm = mirror_left_hand_params(orient_world.reshape(1, 3), finger_aa.reshape(1, 45), trans_world.reshape(1, 3))
                orient_w_corr, trans_w_corr = o_wm[0], t_wm[0]
            else:
                orient_w_corr, trans_w_corr = orient_world, trans_world
            hand_orient_w[side][t] = orient_w_corr
            hand_cam_t_w[side][t] = trans_w_corr

            # Anchor-camera params. Composing R_cw with the global orient gives the
            # camera-frame orientation (for left, mirrored in the camera frame). The
            # naive cam_t carries a wrist-pivot offset (MANO's global orient rotates
            # about the wrist joint, not the origin); it is corrected after FK below.
            R_o_cam = R_cw @ Rotation.from_rotvec(orient_world).as_matrix().astype(np.float32)
            orient_cam = Rotation.from_matrix(R_o_cam).as_rotvec().astype(np.float32)
            trans_cam = (R_cw @ trans_world + t_cw).astype(np.float32)
            if side == "left":
                # Camera-frame mirror of orient+trans only (fingers stay un-mirrored).
                o_m, _f_m, tr_m = mirror_left_hand_params(orient_cam.reshape(1, 3), finger_aa.reshape(1, 45), trans_cam.reshape(1, 3))
                orient_cam, trans_cam = o_m[0], tr_m[0]

            hand_orient[side][t] = orient_cam
            hand_pose[side][t] = finger_aa
            hand_cam_t[side][t] = trans_cam
            hand_valid[side][t] = True

        objects = frame.get("objects", {})
        if obj_key in objects:
            obj_entry = objects[obj_key]
            obj_data = obj_entry[0] if isinstance(obj_entry, list) and obj_entry else obj_entry
            R_obj_world, t_obj_world = _parse_object_pose(obj_data)
            if R_obj_world is not None:
                obj_rotation_world[t] = Rotation.from_matrix(R_obj_world.astype(np.float32)).as_rotvec().astype(np.float32)
                obj_translation_world[t] = t_obj_world.astype(np.float32)
                R_obj_cam = R_cw @ R_obj_world.astype(np.float32)
                t_obj_cam = (R_cw @ t_obj_world.astype(np.float32) + t_cw).astype(np.float32)
                obj_rotation[t] = Rotation.from_matrix(R_obj_cam).as_rotvec().astype(np.float32)
                obj_translation[t] = t_obj_cam
                obj_valid[t] = True

    if not (hand_valid["left"].any() or hand_valid["right"].any()) or not obj_valid.any():
        print(f"  skip canonical {segment_name}: no valid hand/object frames")
        return False

    # FK + wrist-pivot cam_t correction. Re-expressing MANO params across the
    # world→camera rotation leaves the global orient's pivot (the wrist joint) in the
    # wrong place, displacing the hand by ~(I−R_cw)·J_wrist. FK the world hand,
    # rigid-transform it to the camera (reference), then solve the translation offset
    # so the stored params reproduce it (v_naive and v_target differ only by a shift).
    # This is the shaped-pivot (J0(betas)) re-expression, so world and camera fields
    # are exact inverses under the present-camera re-anchor.
    betas_f = betas.astype(np.float32)
    hand_verts = {}
    for side in ("left", "right"):
        if not hand_valid[side].any():
            hand_verts[side] = np.zeros((T, 778, 3), dtype=np.float32)
            continue
        v_world = mano_fk_verts(hand_orient_w[side], hand_pose[side], betas_f, hand_cam_t_w[side], side, str(MANO_DIR)).astype(np.float32)
        v_target = (np.einsum("ij,tvj->tvi", R_cw, v_world) + t_cw).astype(np.float32)
        v_naive = mano_fk_verts(hand_orient[side], hand_pose[side], betas_f, hand_cam_t[side], side, str(MANO_DIR)).astype(np.float32)
        delta = (v_target - v_naive).mean(axis=1).astype(np.float32)  # (T,3) output-space translation
        delta[~hand_valid[side]] = 0
        # cam_t lives PRE x-mirror: mano_fk_verts mirrors the output for left hands
        # (verts[...,0] *= -1), so a +delta shift of the output needs +M·delta on
        # cam_t (M = diag(-1,1,1)). For right, M = I.
        cam_delta = delta.copy()
        if side == "left":
            cam_delta[:, 0] *= -1
        hand_cam_t[side] = (hand_cam_t[side] + cam_delta).astype(np.float32)
        hv = (v_naive + delta[:, None, :]).astype(np.float32)  # == v_target
        hv[~hand_valid[side]] = 0
        hand_verts[side] = hv

    # Object verts in the anchor-camera frame (native mesh + camera-frame pose).
    native_verts = native_verts.astype(np.float32)
    obj_verts = np.zeros((T, len(native_verts), 3), dtype=np.float32)
    for t in range(T):
        if obj_valid[t]:
            R_t = Rotation.from_rotvec(obj_rotation[t]).as_matrix().astype(np.float32)
            obj_verts[t] = native_verts @ R_t.T + obj_translation[t]

    # gravity_direction in the anchor-camera frame: world is +Z up → down = -Z.
    gravity_direction = (R_cw @ np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float32)

    canonical_dir = seg_dir / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "frame_indices": np.arange(T, dtype=np.int64),
        "fps": np.float32(FPS),
        "obj_canonical_verts": native_verts,
        "obj_faces": native_faces.astype(np.int32),
        "obj_rotation": obj_rotation,
        "obj_translation": obj_translation,
        "obj_rotation_world": obj_rotation_world,
        "obj_translation_world": obj_translation_world,
        "obj_scale": np.ones(3, dtype=np.float32),
        "obj_verts": obj_verts,
        "gravity_direction": gravity_direction,
        "gravity_world": np.array([0.0, 0.0, -1.0], dtype=np.float32),
        "camera_extrinsics": camera_extrinsics.astype(np.float32),
    }
    for side in ("left", "right"):
        save_dict[f"hand_{side}_orient"] = hand_orient[side]
        save_dict[f"hand_{side}_pose"] = hand_pose[side]
        save_dict[f"hand_{side}_cam_t"] = hand_cam_t[side]
        save_dict[f"hand_{side}_orient_world"] = hand_orient_w[side]
        save_dict[f"hand_{side}_cam_t_world"] = hand_cam_t_w[side]
        save_dict[f"hand_{side}_betas"] = betas.astype(np.float32)
        save_dict[f"hand_{side}_valid"] = hand_valid[side]
        save_dict[f"hand_{side}_verts"] = hand_verts[side]
    np.savez_compressed(str(canonical_dir / "canonical.npz"), **save_dict)

    _, raw_category = BOP_OBJECTS[bop_id]
    obj_category = CATEGORY_NORMALIZE.get(raw_category, raw_category.split("_")[0].lower())
    active = sorted([s for s in ("left", "right") if hand_valid[s].any()])
    metadata = {
        "source_dataset": "hot3d",
        "segment_id": segment_name,
        "confidence_tier": "gt_mocap",
        "has_depth": (seg_dir / "video.mp4").exists(),
        "has_scene": (seg_dir / "video.mp4").exists(),
        "has_gravity": True,
        "has_support_plane": False,
        "has_native_hand_pose": True,
        "has_native_object_pose": True,
        "object_is_rigid": True,
        "primary_object_id": f"obj_{bop_id:06d}",
        "primary_object_category": obj_category,
        "bimanual": len(active) > 1,
        "viewpoint": "egocentric",
        "frame": "anchor_camera_static",
        "hand_source": "mocap_gt",
        "object_source": "mocap_gt",
        "depth_source": "moge2_video" if (seg_dir / "video.mp4").exists() else "none",
        "mano_convention": "right_model_x_mirror",
        "hot3d_clip_id": clip_id,
        "anchor_frame": int(anchor),
        "n_frames": int(T),
        "fps": FPS,
        "active_sides": active,
    }
    with open(canonical_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return True


def run_convert_step(args):
    """Convert downloaded HOT3D clips to canonical format."""
    train_aria_dir = HOT3D_DIR / "train_aria"
    if not train_aria_dir.exists():
        print("ERROR: train_aria/ directory not found. Run download step first.")
        return []

    # Discover tars
    if args.clip:
        tar_files = [train_aria_dir / f"{args.clip}.tar"]
        tar_files = [t for t in tar_files if t.exists()]
    else:
        tar_files = sorted(train_aria_dir.glob("clip-*.tar"))
        if args.max_clips:
            tar_files = tar_files[: args.max_clips]
        if args.num_shards > 1:
            tar_files = [t for i, t in enumerate(tar_files) if i % args.num_shards == args.shard_idx]

    if not tar_files:
        print("No clip tars found to convert.")
        return []

    print(f"Converting {len(tar_files)} clips...")
    done = skipped = failed = 0
    segments = []

    for i, tar_path in enumerate(tar_files):
        clip_id = tar_path.stem
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i + 1}/{len(tar_files)}] {clip_id}")

        try:
            clip_data = _parse_tar(str(tar_path), with_images=not args.no_video)

            # The hand annotations must hold wrist_xform as 6 floats (axis-angle + translation).
            if i == 0 and clip_data["frames"]:
                frame0 = clip_data["frames"][0]
                hands0 = frame0.get("hands", {})
                for side in ("left", "right"):
                    if side in hands0 and "mano_pose" in hands0[side]:
                        wxt = hands0[side]["mano_pose"].get("wrist_xform")
                        if not isinstance(wxt, list) or len(wxt) != 6:
                            print(f"  ERROR: unexpected wrist_xform format in {clip_id} ({type(wxt).__name__}); aborting")
                            return segments
                        break

            manip_objects = _detect_manipulated_objects(clip_data, args.prox_min_frames)

            if not manip_objects:
                skipped += 1
                continue

            for manip_obj in manip_objects:
                seg_name = _convert_one_clip(clip_data, manip_obj, args)
                if seg_name:
                    segments.append(seg_name)
                    done += 1
                else:
                    failed += 1

        except Exception as e:
            print(f"  ERROR on {clip_id}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"  Convert: {done} segments created, {skipped} clips skipped (no manipulation), {failed} failed")
    return segments


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(prog="python -m gtf.datasets.hot3d", description="Ingest HOT3D-Clips into canonical format")
    parser.add_argument("--start-from", choices=PIPELINE_STEPS, default=PIPELINE_STEPS[0])
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--clip", type=str, help="Process single clip (e.g., clip-001849)")
    parser.add_argument("--max-clips", type=int, help="Limit number of clips to process")
    parser.add_argument("--out-root", type=str, default="", help="Output root (default: results)")
    parser.add_argument("--no-video", action="store_true", help="Skip writing video.mp4")
    parser.add_argument("--num-shards", type=int, default=1, help="Shard clips across N processes (convert step)")
    parser.add_argument("--shard-idx", type=int, default=0, help="This process's shard index [0, num_shards)")
    parser.add_argument("--prox-min-frames", type=int, default=DEFAULT_PROX_MIN_FRAMES, help="Min hand-surface contact frames to count as manipulation")
    args = parser.parse_args()

    start_idx = PIPELINE_STEPS.index(args.start_from)
    active_steps = set(PIPELINE_STEPS[start_idx:])

    if "download" in active_steps:
        run_download_step(args)

    if "convert" in active_steps:
        run_convert_step(args)


if __name__ == "__main__":
    main()
