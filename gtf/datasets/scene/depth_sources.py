"""Per-source depth loaders for scene preprocessing.

Each loader returns an ``AnchorDepthResult`` with depth, RGB, intrinsics, and
hand/object segmentation masks for one canonical frame (the "anchor"). The masks
exclude hand and object pixels from the scene point cloud; where a source has no
native masks they are projected from the canonical hand/object vertices.

``anchor_canonical_idx`` is the canonical row and ``anchor_video_idx`` its
``frame_indices`` value. Action100M keyframes are sparse, so its depth payloads
are indexed by video frame, not canonical row.
"""

from __future__ import annotations

import atexit
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import logging
import os
import time
from typing import Any

import cv2
import numpy as np
import torch

from gtf.paths import REPO_ROOT
from gtf.datasets.dexycb import DEXYCB_SUBJECTS, _load_intrinsics

ROOT = REPO_ROOT

logger = logging.getLogger(__name__)

DEXYCB_DIR = ROOT / "datasets" / "dexycb"
H2O_DIR = ROOT / "datasets" / "h2o"
HO3D_DIR = ROOT / "datasets" / "ho3d"
H2O3D_DIR = ROOT / "datasets" / "h2o3d"
HOCAP_DIR = ROOT / "datasets" / "hocap"
TACO_DIR = ROOT / "datasets" / "taco"

# MoGe-2 metric checkpoint for the sources without usable depth.
MOGE_MODEL_ID = "yjh001/metricanything_student_pointmap"
# Sources whose depth comes from MoGe on the segment video.
MOGE_SOURCES = {"hot3d", "show3d", "oakink2"}

# DexYCB seg label convention (from labels_*.npz): background=0, YCB objects use
# their 1-based YCB index (1-21), hand pixels are 255.
DEXYCB_HAND_LABEL = 255
MAX_VERTS_FOR_MASK = 12000
MAX_VIDEO_CAPTURE_CACHE = 2
VIDEO_DECODER_THREADS = 16
_VIDEO_CAPTURE_CACHE: OrderedDict[str, tuple[cv2.VideoCapture, int]] = OrderedDict()


def _open_video_capture(video_path: str | Path) -> cv2.VideoCapture:
    """Open an FFmpeg decoder with a bounded thread count (default OpenCV backend as fallback).

    A default capture can start one FFmpeg thread per core; 16 threads decode the
    same frames.
    """
    key = str(Path(video_path).resolve())
    try:
        cap = cv2.VideoCapture(
            key,
            cv2.CAP_FFMPEG,
            [cv2.CAP_PROP_N_THREADS, int(VIDEO_DECODER_THREADS)],
        )
    except (AttributeError, TypeError, cv2.error):
        cap = cv2.VideoCapture(key)
    if cap.isOpened():
        return cap
    cap.release()
    return cv2.VideoCapture(key)


def _close_video_capture_cache() -> None:
    while _VIDEO_CAPTURE_CACHE:
        _key, (cap, _last_frame) = _VIDEO_CAPTURE_CACHE.popitem(last=False)
        cap.release()


atexit.register(_close_video_capture_cache)


@dataclass
class AnchorDepthResult:
    depth: np.ndarray  # (H, W) float32 meters, 0 = invalid
    rgb: np.ndarray  # (H, W, 3) uint8
    intrinsics: np.ndarray  # (3, 3) pixel-unit K
    anchor_canonical_idx: int
    anchor_video_idx: int
    hand_seg_mask: np.ndarray | None  # (H, W) bool; True = hand pixels
    obj_seg_mask: np.ndarray | None  # (H, W) bool; True = object pixels


def _canonical_seg_dir(entry: dict) -> Path:
    return Path(entry["canonical_path"]).parent.parent


def load_anchor_depth(
    entry: dict,
    canonical_data: dict[str, Any],
    moge_model: Any | None = None,
    anchor_canonical_idx: int | None = None,
) -> AnchorDepthResult | None:
    """Dispatch on ``entry['source_dataset']``.

    Returns None when an input is missing (missing file, unreadable image, zero-frame
    video). Any other exception propagates, so ``--strict`` stops on it.
    """
    source = str(entry.get("source_dataset", "")).lower()
    if source == "dexycb":
        return _load_dexycb_anchor(entry, canonical_data, anchor_canonical_idx, moge_model=moge_model)
    if source == "hot3d":
        if moge_model is None:
            logger.warning(f"{entry['segment_id']}: HOT3D requires moge_model=...")
            return None
        return _load_hot3d_anchor(entry, canonical_data, moge_model, anchor_canonical_idx)
    if source == "action100m":
        return _load_action100m_anchor(entry, canonical_data, anchor_canonical_idx)
    if source == "hocap":
        return _load_hocap_anchor(entry, canonical_data, anchor_canonical_idx)
    if source == "h2o":
        return _load_h2o_anchor(entry, canonical_data, anchor_canonical_idx)
    if source in {"ho3d", "h2o3d"}:
        return _load_ho3d_anchor(entry, canonical_data, source, anchor_canonical_idx)
    if source == "taco":
        native = _load_taco_anchor(entry, canonical_data, anchor_canonical_idx)
        if native is not None:
            return native
        if moge_model is None:
            logger.warning(f"{entry['segment_id']}: TACO missing native depth and moge_model=...")
            return None
        return _load_video_moge_anchor(entry, canonical_data, moge_model, anchor_canonical_idx)
    if source in {"show3d", "oakink2"}:
        if moge_model is None:
            logger.warning(f"{entry['segment_id']}: {source} RGB-only scene preprocessing requires moge_model=...")
            return None
        return _load_video_moge_anchor(entry, canonical_data, moge_model, anchor_canonical_idx)
    # Any other source: MoGe-2 depth on the segment's video.mp4.
    if moge_model is None:
        logger.warning(f"{entry['segment_id']}: source {source!r} needs moge_model=...")
        return None
    return _load_video_moge_anchor(entry, canonical_data, moge_model, anchor_canonical_idx)


# ---------------------------------------------------------------------------
# DexYCB
# ---------------------------------------------------------------------------


def _dexycb_cam_dir(entry: dict) -> Path:
    subject_key = str(entry["dexycb_subject"])
    subject_dir = DEXYCB_SUBJECTS[subject_key]
    date_prefix = subject_dir.split("-")[0]
    # segment_id format: "dexycb_{subject_key}_{time_suffix}_cam{cam_idx}"
    seg_id: str = str(entry["segment_id"])
    # Strip cam suffix, then subject_key prefix.
    without_cam = seg_id.rsplit("_cam", 1)[0]
    time_suffix = without_cam.split(f"dexycb_{subject_key}_", 1)[-1]
    seq_name = f"{date_prefix}_{time_suffix}"
    cam_serial = str(entry["dexycb_camera_serial"])
    return DEXYCB_DIR / subject_dir / seq_name / cam_serial


def _dexycb_primary_class_id(cam_dir: Path) -> int | None:
    """Return the YCB-Video class id of the grasped object for this sequence.

    Reads ``meta.yml`` in the sequence dir (one level above cam_dir) and uses
    ``ycb_ids[ycb_grasp_ind]``. Returns None if the file or fields are missing.
    """
    meta_path = cam_dir.parent / "meta.yml"
    if not meta_path.exists():
        return None
    import yaml

    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    ycb_ids = meta.get("ycb_ids")
    grasp_ind = meta.get("ycb_grasp_ind")
    if ycb_ids is None or grasp_ind is None:
        return None
    if not (0 <= int(grasp_ind) < len(ycb_ids)):
        return None
    return int(ycb_ids[int(grasp_ind)])


def _dexycb_intrinsics(entry: dict, canonical_data: dict[str, Any]) -> np.ndarray | None:
    if "camera_K" in canonical_data:
        K = canonical_data["camera_K"]
        return K.numpy().astype(np.float32) if hasattr(K, "numpy") else np.asarray(K, dtype=np.float32)
    intrinsics = _load_intrinsics(str(entry["dexycb_camera_serial"]))
    return None if intrinsics is None else intrinsics.astype(np.float32)


def _load_dexycb_masks(cam_dir: Path, entry: dict, anchor_video: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    labels_path = cam_dir / f"labels_{int(anchor_video):06d}.npz"
    hand_mask = None
    obj_mask = None
    if not labels_path.exists():
        return hand_mask, obj_mask
    try:
        with np.load(str(labels_path), allow_pickle=False) as lb:
            seg = lb["seg"]
        hand_mask = seg == DEXYCB_HAND_LABEL
        primary_class = _dexycb_primary_class_id(cam_dir)
        if primary_class is not None:
            obj_mask = seg == int(primary_class)
        else:
            logger.warning(f"{entry['segment_id']}: could not resolve primary YCB class; leaving obj_mask=None")
    except Exception as e:
        logger.warning(f"{entry['segment_id']}: seg load failed: {e}")
    return hand_mask, obj_mask


def _load_dexycb_anchor(
    entry: dict,
    canonical_data: dict[str, Any],
    anchor_canonical_idx: int | None = None,
    moge_model: Any | None = None,
) -> AnchorDepthResult | None:
    cam_dir = _dexycb_cam_dir(entry)
    if not cam_dir.exists():
        logger.warning(f"{entry['segment_id']}: cam_dir missing {cam_dir}")
        return None

    anchor_canonical, anchor_video = _anchor_indices(canonical_data, anchor_canonical_idx)

    color_path = cam_dir / f"color_{anchor_video:06d}.jpg"
    bgr = cv2.imread(str(color_path))
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    intrinsics = _dexycb_intrinsics(entry, canonical_data)
    if intrinsics is None:
        return None

    depth_path = cam_dir / f"aligned_depth_to_color_{anchor_video:06d}.png"
    depth = None
    if depth_path.exists():
        depth_u16 = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_u16 is not None:
            depth = depth_u16.astype(np.float32) / 1000.0  # mm → m
    if depth is None:
        if moge_model is None:
            return None
        depth, _moge_intrinsics = _run_moge(moge_model, rgb)

    # Masks from labels_*.npz (0 = background, 1-21 = YCB-Video class ids, 255 = hand). Only
    # the grasped object (meta.yml: ycb_ids[ycb_grasp_ind]) is masked; the other YCB objects
    # are scene content and stay in the point cloud.
    hand_mask, obj_mask = _load_dexycb_masks(cam_dir, entry, anchor_video)

    return AnchorDepthResult(
        depth=depth,
        rgb=rgb,
        intrinsics=intrinsics,
        anchor_canonical_idx=anchor_canonical,
        anchor_video_idx=anchor_video,
        hand_seg_mask=hand_mask,
        obj_seg_mask=obj_mask,
    )


# ---------------------------------------------------------------------------
# HOT3D
# ---------------------------------------------------------------------------


def _decode_video_frame_bgr_cached(video_path: Path, frame_idx: int) -> np.ndarray | None:
    # Interaction children symlink their parent's video: key the decoder cache by the target.
    key = str(Path(video_path).resolve())
    item = _VIDEO_CAPTURE_CACHE.pop(key, None)
    if item is None:
        cap = _open_video_capture(key)
        last_frame = -1
    else:
        cap, last_frame = item
        if not cap.isOpened():
            cap.release()
            cap = _open_video_capture(key)
            last_frame = -1
    if not cap.isOpened():
        return None

    frame_idx = int(frame_idx)
    if frame_idx != last_frame + 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
    try:
        ok, bgr = cap.read()
    finally:
        _VIDEO_CAPTURE_CACHE[key] = (cap, frame_idx if ok else -1)
        _VIDEO_CAPTURE_CACHE.move_to_end(key)
        while len(_VIDEO_CAPTURE_CACHE) > MAX_VIDEO_CAPTURE_CACHE:
            _old_key, (old_cap, _old_last_frame) = _VIDEO_CAPTURE_CACHE.popitem(last=False)
            old_cap.release()
    if not ok or bgr is None:
        return None
    return bgr


def _decode_video_frame(video_path: Path, frame_idx: int) -> np.ndarray | None:
    bgr = _decode_video_frame_bgr_cached(video_path, frame_idx)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


_VIDEO_NFRAMES_CACHE: dict[str, int] = {}
_CLIP_START_CACHE: dict[str, int] = {}


def _video_n_frames(video_path: Path) -> int:
    key = str(Path(video_path).resolve())
    if key not in _VIDEO_NFRAMES_CACHE:
        # Retry once, then raise: a guessed length would bypass the frame-range check.
        n, last_err = -1, None
        for _attempt in range(2):
            try:
                cap = _open_video_capture(key)
                opened = cap.isOpened()
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else -1
                cap.release()
                if opened and n > 0:
                    break
                last_err = RuntimeError(("reported %d frames" % n) if opened else "could not open the file")
            except Exception as e:  # noqa: BLE001
                last_err = e
            time.sleep(0.5)
        if n <= 0:
            raise RuntimeError(f"_video_n_frames: video unreadable after retry: {key} ({last_err})")
        _VIDEO_NFRAMES_CACHE[key] = n
    return _VIDEO_NFRAMES_CACHE[key]


def canonical_row_to_video_frame(canonical_data: dict[str, Any], seg_dir: Path, row: int) -> int:
    """Map a canonical row to the frame index of the segment's video.mp4.

    Two bindings exist:
      * direct: frame_indices are frames of the video itself (1:1 sources with
        fi = arange, OakInk2 parents and their __iwN children, which symlink the
        parent video, Action100M keyframes). video_frame = fi[row].
      * clip_local: frame_indices are session frame numbers while video.mp4 is a clip
        starting at the parent canonical's fi[0] (SHOW3D, including its decimated
        segments, and HO3D). video_frame = fi[row] - clip_start, where clip_start is
        this segment's fi[0], or the symlink-target segment's fi[0] for an interaction
        child.

    Other sources use direct when fi[-1] < n_video. A result outside the video raises.
    """
    fi = canonical_data["frame_indices"]
    _seg_name = Path(seg_dir).name
    fi_row = int(fi[row].item() if hasattr(fi[row], "item") else fi[row])
    fi_first = int(fi[0].item() if hasattr(fi[0], "item") else fi[0])
    fi_last = int(fi[-1].item() if hasattr(fi[-1], "item") else fi[-1])
    video_path = Path(seg_dir) / "video.mp4"
    n_video = _video_n_frames(video_path)
    if _seg_name.startswith(("show3d", "ho3d")):
        _bind = "clip_local"
    elif _seg_name.startswith("oakink2"):
        _bind = "direct"
    else:
        _bind = "direct" if fi_last < n_video else "clip_local"
    if _bind == "direct":
        out = fi_row
    else:
        clip_start = fi_first
        real = video_path.resolve()
        if real.parent != Path(seg_dir).resolve():
            key = str(real.parent)
            if key not in _CLIP_START_CACHE:
                # The parent canonical's fi[0] is the clip origin of a symlinked child.
                parent_cp = real.parent / "canonical" / "canonical.npz"
                if not parent_cp.exists():
                    raise RuntimeError(f"canonical_row_to_video_frame: {Path(seg_dir).name} links to {real.parent}, whose canonical {parent_cp} is missing")
                start, last_err = None, None
                for _attempt in range(2):
                    try:
                        with np.load(parent_cp, allow_pickle=True) as pd:
                            start = int(np.asarray(pd["frame_indices"])[0])
                        last_err = None
                        break
                    except Exception as e:  # noqa: BLE001
                        last_err = e
                        time.sleep(0.5)
                if start is None:
                    raise RuntimeError(f"canonical_row_to_video_frame: parent canonical unreadable after retry: {parent_cp} ({last_err})")
                _CLIP_START_CACHE[key] = start
            clip_start = _CLIP_START_CACHE[key]
        out = fi_row - clip_start
    if not (0 <= out < n_video):
        raise ValueError(f"canonical_row_to_video_frame: row {row} → video frame {out} outside [0, {n_video}) for {seg_dir} (fi0={fi_first}, fiN={fi_last})")
    return out


def _anchor_indices(canonical_data: dict[str, Any], anchor_canonical_idx: int | None = None) -> tuple[int, int]:
    frame_indices = canonical_data["frame_indices"]
    n_frames = int(frame_indices.shape[0])
    anchor_canonical = n_frames // 2 if anchor_canonical_idx is None else int(np.clip(anchor_canonical_idx, 0, n_frames - 1))
    anchor_video = int(frame_indices[anchor_canonical].item() if hasattr(frame_indices[anchor_canonical], "item") else frame_indices[anchor_canonical])
    return anchor_canonical, anchor_video


def _masks_from_canonical_verts(
    canonical_data: dict[str, Any],
    anchor_canonical: int,
    intrinsics: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    hand_verts_list = []
    for side in ("left", "right"):
        valid_key = f"hand_{side}_valid"
        verts_key = f"hand_{side}_verts"
        if valid_key in canonical_data and verts_key in canonical_data and bool(canonical_data[valid_key][anchor_canonical]):
            v = canonical_data[verts_key][anchor_canonical]
            if hasattr(v, "numpy"):
                v = v.numpy()
            hand_verts_list.append(np.asarray(v, dtype=np.float32))
    hand_verts_cam = np.concatenate(hand_verts_list, axis=0) if hand_verts_list else np.zeros((0, 3), dtype=np.float32)
    hand_mask = _project_mask_from_verts(hand_verts_cam, intrinsics, image_shape)

    obj_mask = None
    obj_verts = canonical_data.get("obj_verts")
    if obj_verts is not None:
        obj_verts_cam = obj_verts[anchor_canonical]
        if hasattr(obj_verts_cam, "numpy"):
            obj_verts_cam = obj_verts_cam.numpy()
        obj_mask = _project_mask_from_verts(np.asarray(obj_verts_cam, dtype=np.float32), intrinsics, image_shape)
    return hand_mask, obj_mask


def _model_device(model: Any) -> torch.device:
    return next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cpu")


@torch.inference_mode()
def _run_moge(moge_model: Any, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    H, W = rgb.shape[:2]
    img = torch.from_numpy(rgb).to(device=_model_device(moge_model), dtype=torch.float32).permute(2, 0, 1) / 255.0
    out = moge_model.infer(img)
    depth = out["depth"].detach().cpu().numpy().astype(np.float32)
    K_norm = out["intrinsics"].detach().cpu().numpy().copy()
    K_px = K_norm.astype(np.float32)
    K_px[0, 0] *= W
    K_px[1, 1] *= H
    K_px[0, 2] *= W
    K_px[1, 2] *= H
    return depth, K_px


@torch.inference_mode()
def _run_moge_batch(moge_model: Any, rgbs: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run MoGe on a same-resolution RGB batch and return per-frame depth/K."""
    if not rgbs:
        return []
    shapes = {tuple(rgb.shape) for rgb in rgbs}
    if len(shapes) != 1:
        raise ValueError(f"batched MoGe requires same-resolution frames, got {sorted(shapes)}")
    H, W = rgbs[0].shape[:2]
    imgs = np.stack(rgbs, axis=0)
    imgs_t = torch.from_numpy(imgs).to(device=_model_device(moge_model), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    out = moge_model.infer(imgs_t)
    depths = out["depth"].detach().cpu().numpy().astype(np.float32)
    K_norm = out["intrinsics"].detach().cpu().numpy().copy()
    K_px = K_norm.astype(np.float32)
    K_px[:, 0, 0] *= W
    K_px[:, 1, 1] *= H
    K_px[:, 0, 2] *= W
    K_px[:, 1, 2] *= H
    return [(depths[i], K_px[i]) for i in range(len(rgbs))]


def _project_mask_from_verts(
    verts_cam: np.ndarray,
    intrinsics: np.ndarray,
    image_shape: tuple[int, int],
    dilate_px: int = 6,
) -> np.ndarray:
    """Project 3D verts (camera frame, meters) to a 2D pixel mask via convex hull.

    Used where a source has no per-pixel segmentation masks.
    """
    H, W = image_shape
    mask = np.zeros((H, W), dtype=np.uint8)
    if verts_cam.size == 0:
        return mask.astype(bool)
    if verts_cam.shape[0] > MAX_VERTS_FOR_MASK:
        idx = np.linspace(0, verts_cam.shape[0] - 1, MAX_VERTS_FOR_MASK, dtype=np.int64)
        verts_cam = verts_cam[idx]
    z = verts_cam[:, 2]
    valid = z > 1e-3
    if not valid.any():
        return mask.astype(bool)
    v = verts_cam[valid]
    uv = (intrinsics @ v.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    uv[:, 0] = np.clip(uv[:, 0], 0, W - 1)
    uv[:, 1] = np.clip(uv[:, 1], 0, H - 1)
    pts = uv.astype(np.int32)
    if len(pts) >= 3:
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 1)
    if dilate_px > 0:
        k = np.ones((dilate_px, dilate_px), dtype=np.uint8)
        mask = cv2.dilate(mask, k)
    return mask.astype(bool)


def _load_hot3d_anchor(
    entry: dict,
    canonical_data: dict[str, Any],
    moge_model: Any,
    anchor_canonical_idx: int | None = None,
) -> AnchorDepthResult | None:
    seg_dir = _canonical_seg_dir(entry)
    video_path = seg_dir / "video.mp4"
    if not video_path.exists():
        return None

    anchor_canonical, anchor_video = _anchor_indices(canonical_data, anchor_canonical_idx)

    rgb = _decode_video_frame(video_path, anchor_video)
    if rgb is None:
        return None

    depth, intrinsics = _run_moge(moge_model, rgb)
    if canonical_data.get("obj_verts") is None:
        return None
    hand_mask, obj_mask = _masks_from_canonical_verts(canonical_data, anchor_canonical, intrinsics, depth.shape)

    return AnchorDepthResult(
        depth=depth,
        rgb=rgb,
        intrinsics=intrinsics,
        anchor_canonical_idx=anchor_canonical,
        anchor_video_idx=anchor_video,
        hand_seg_mask=hand_mask,
        obj_seg_mask=obj_mask,
    )


# ---------------------------------------------------------------------------
# Action100M
# ---------------------------------------------------------------------------


_ACTION100M_DEPTH_KEYS = ("depth", "intrinsics_px", "mask")


def _load_action100m_depth(path: Path, image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Read one extraction depth payload (object_pose/depth/{frame:05d}.npz) and check its schema.

    Returns (depth with masked-out pixels set to 0, pixel-space intrinsics).
    """
    with np.load(str(path), allow_pickle=False) as d:
        missing = set(_ACTION100M_DEPTH_KEYS) - set(d.files)
        if missing:
            raise ValueError(f"{path}: missing keys {sorted(missing)}")
        depth, intrinsics, mask = (np.asarray(d[k]) for k in _ACTION100M_DEPTH_KEYS)
    if depth.shape != image_shape or mask.shape != image_shape:
        raise ValueError(f"{path}: depth/mask shape {depth.shape}/{mask.shape} != decoded RGB {image_shape}")
    if depth.dtype != np.float32 or intrinsics.dtype != np.float32 or mask.dtype != np.uint8:
        raise ValueError(f"{path}: expected float32/float32/uint8, got {depth.dtype}/{intrinsics.dtype}/{mask.dtype}")
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError(f"{path}: intrinsics are malformed/non-finite")
    if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0 or not np.allclose(intrinsics[2], [0.0, 0.0, 1.0], atol=1e-5):
        raise ValueError(f"{path}: intrinsics are not a valid pixel-space K")
    if not np.isin(mask, (0, 1)).all() or not bool(mask.any()):
        raise ValueError(f"{path}: mask is non-binary or has no valid pixels")
    valid_depth = depth[mask.astype(bool)]
    if not np.isfinite(valid_depth).all() or np.any(valid_depth <= 0):
        raise ValueError(f"{path}: valid-mask depth must be finite and positive")
    if np.isnan(depth).any() or np.isneginf(depth).any():
        raise ValueError(f"{path}: depth contains NaN or -inf")
    # The mask marks valid depth, not the object: masked-out pixels are dropped from the scene.
    return np.where(mask.astype(bool), depth, 0.0), intrinsics


def _load_action100m_anchor(
    entry: dict,
    canonical_data: dict[str, Any],
    anchor_canonical_idx: int | None = None,
) -> AnchorDepthResult | None:
    seg_dir = _canonical_seg_dir(entry)
    video_path = seg_dir / "video.mp4"
    anchor_canonical, anchor_video = _anchor_indices(canonical_data, anchor_canonical_idx)
    depth_path = seg_dir / "object_pose" / "depth" / f"{int(anchor_video):05d}.npz"
    if not depth_path.exists():
        return None
    rgb = _decode_video_frame(video_path, anchor_video) if video_path.exists() else None
    if rgb is None:
        return None
    depth, intrinsics = _load_action100m_depth(depth_path, rgb.shape[:2])
    hand_mask, obj_mask = _masks_from_canonical_verts(canonical_data, anchor_canonical, intrinsics, depth.shape[:2])
    return AnchorDepthResult(
        depth=depth,
        rgb=rgb,
        intrinsics=intrinsics,
        anchor_canonical_idx=anchor_canonical,
        anchor_video_idx=anchor_video,
        hand_seg_mask=hand_mask,
        obj_seg_mask=obj_mask,
    )


# ---------------------------------------------------------------------------
# HO-Cap, H2O, HO3D, TACO
# ---------------------------------------------------------------------------


def _project_depth_to_color(
    depth_m: np.ndarray,
    K_depth: np.ndarray,
    T_color_from_depth: np.ndarray,
    K_color: np.ndarray,
    color_shape: tuple[int, int],
) -> np.ndarray:
    """Z-buffer a raw depth image into the color camera frame."""
    H_d, W_d = depth_m.shape
    H_c, W_c = color_shape
    valid = (depth_m > 0.05) & (depth_m < 10.0) & np.isfinite(depth_m)
    if not valid.any():
        return np.zeros((H_c, W_c), dtype=np.float32)

    yy, xx = np.nonzero(valid)
    z = depth_m[yy, xx].astype(np.float32)
    x = (xx.astype(np.float32) - float(K_depth[0, 2])) * z / float(K_depth[0, 0])
    y = (yy.astype(np.float32) - float(K_depth[1, 2])) * z / float(K_depth[1, 1])
    pts_d = np.stack([x, y, z, np.ones_like(z)], axis=1)
    pts_c = (T_color_from_depth.astype(np.float32) @ pts_d.T).T[:, :3]
    zc = pts_c[:, 2]
    keep = (zc > 0.05) & (zc < 10.0) & np.isfinite(pts_c).all(axis=1)
    if not keep.any():
        return np.zeros((H_c, W_c), dtype=np.float32)

    pts_c = pts_c[keep]
    zc = zc[keep]
    uv = (K_color.astype(np.float32) @ pts_c.T).T
    u = np.rint(uv[:, 0] / uv[:, 2]).astype(np.int32)
    v = np.rint(uv[:, 1] / uv[:, 2]).astype(np.int32)
    in_bounds = (u >= 0) & (u < W_c) & (v >= 0) & (v < H_c)
    if not in_bounds.any():
        return np.zeros((H_c, W_c), dtype=np.float32)

    flat = np.full(H_c * W_c, np.inf, dtype=np.float32)
    flat_idx = v[in_bounds] * W_c + u[in_bounds]
    np.minimum.at(flat, flat_idx, zc[in_bounds].astype(np.float32))
    aligned = flat.reshape(H_c, W_c)
    aligned[~np.isfinite(aligned)] = 0.0
    return aligned.astype(np.float32)


def _load_hocap_depth_calibration(
    serial: str,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    path = HOCAP_DIR / "calibration" / "intrinsics" / f"{serial}.yaml"
    if not path.exists():
        return None, None, None
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f)

    def k(stream: str) -> np.ndarray:
        item = data[stream]
        return np.array(
            [
                [item["fx"], 0.0, item["ppx"]],
                [0.0, item["fy"], item["ppy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    K_color = k("color")
    K_depth = k("depth")
    T_raw = np.asarray(data.get("depth2color", np.eye(4).reshape(-1)), dtype=np.float32).reshape(-1)
    if T_raw.size == 12:
        T = np.eye(4, dtype=np.float32)
        T[:3, :4] = T_raw.reshape(3, 4)
    else:
        T = T_raw.reshape(4, 4)
    return K_color, K_depth, T


def _load_hocap_anchor(entry: dict, canonical_data: dict[str, Any], anchor_canonical_idx: int | None = None) -> AnchorDepthResult | None:
    subject = str(entry.get("hocap_subject", ""))
    sequence = str(entry.get("hocap_sequence", ""))
    serial = str(entry.get("hocap_camera_serial", ""))
    if not subject or not sequence or not serial:
        return None
    cam_dir = HOCAP_DIR / subject / sequence / serial
    if not cam_dir.exists():
        return None

    anchor_canonical, anchor_video = _anchor_indices(canonical_data, anchor_canonical_idx)
    color_path = cam_dir / f"color_{anchor_video:06d}.jpg"
    depth_path = cam_dir / f"depth_{anchor_video:06d}.png"
    bgr = cv2.imread(str(color_path))
    depth_u16 = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if bgr is None or depth_u16 is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    raw_depth = depth_u16.astype(np.float32) / 1000.0

    K_color, K_depth, T_color_from_depth = _load_hocap_depth_calibration(serial)
    if K_color is None or K_depth is None or T_color_from_depth is None:
        K = canonical_data.get("camera_K")
        if K is None:
            return None
        K_color = K.numpy().astype(np.float32) if hasattr(K, "numpy") else np.asarray(K, dtype=np.float32)
        depth = raw_depth
    else:
        depth = _project_depth_to_color(raw_depth, K_depth, T_color_from_depth, K_color, rgb.shape[:2])

    hand_mask, obj_mask = _masks_from_canonical_verts(canonical_data, anchor_canonical, K_color, depth.shape)
    return AnchorDepthResult(
        depth=depth,
        rgb=rgb,
        intrinsics=K_color.astype(np.float32),
        anchor_canonical_idx=anchor_canonical,
        anchor_video_idx=anchor_video,
        hand_seg_mask=hand_mask,
        obj_seg_mask=obj_mask,
    )


def _find_numbered_image(root: Path, stem: int, exts: tuple[str, ...]) -> Path | None:
    names = [str(stem), f"{stem:04d}", f"{stem:05d}", f"{stem:06d}"]
    for name in dict.fromkeys(names):
        for ext in exts:
            p = root / f"{name}{ext}"
            if p.exists():
                return p
    return None


def _read_h2o_intrinsics(seq_dir: Path, canonical_data: dict[str, Any]) -> np.ndarray | None:
    K = canonical_data.get("camera_K")
    if K is not None:
        return K.numpy().astype(np.float32) if hasattr(K, "numpy") else np.asarray(K, dtype=np.float32)
    intr_path = seq_dir / "cam4" / "cam_intrinsics.txt"
    if not intr_path.exists():
        return None
    vals = np.loadtxt(str(intr_path), dtype=np.float32).reshape(-1)
    if vals.size < 4:
        return None
    fx, fy, cx, cy = vals[:4]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _load_h2o_anchor(entry: dict, canonical_data: dict[str, Any], anchor_canonical_idx: int | None = None) -> AnchorDepthResult | None:
    rel = str(entry.get("h2o_sequence", ""))
    if not rel:
        return None
    seq_dir = H2O_DIR / rel
    cam4 = seq_dir / "cam4"
    if not cam4.exists():
        return None

    anchor_canonical, anchor_video = _anchor_indices(canonical_data, anchor_canonical_idx)
    rgb_path = _find_numbered_image(cam4 / "rgb", anchor_video, (".png", ".jpg", ".jpeg"))
    depth_path = _find_numbered_image(cam4 / "depth", anchor_video, (".png", ".tiff", ".tif"))
    if rgb_path is None or depth_path is None:
        return None
    bgr = cv2.imread(str(rgb_path))
    depth_u16 = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    K = _read_h2o_intrinsics(seq_dir, canonical_data)
    if bgr is None or depth_u16 is None or K is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth = depth_u16.astype(np.float32) / 1000.0
    if depth.shape != rgb.shape[:2]:
        depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    hand_mask, obj_mask = _masks_from_canonical_verts(canonical_data, anchor_canonical, K, depth.shape)
    return AnchorDepthResult(
        depth=depth,
        rgb=rgb,
        intrinsics=K.astype(np.float32),
        anchor_canonical_idx=anchor_canonical,
        anchor_video_idx=anchor_video,
        hand_seg_mask=hand_mask,
        obj_seg_mask=obj_mask,
    )


# Meters per unit of the HO-3D / H2O-3D depth PNGs (the datasets' official reader).
HO3D_DEPTH_SCALE_M = 0.00012498664727900177


def _resolve_ho3d_seq_dir(entry: dict, source: str) -> Path | None:
    root = H2O3D_DIR if source == "h2o3d" else HO3D_DIR
    split = str(entry.get("split", "train"))
    seq = str(entry.get("sequence", ""))
    if not seq:
        return None
    nested_names = ("H2O3D_v1", "H2O3D") if source == "h2o3d" else ("HO3D_v3", "HO3D_v2")
    candidates = [
        root / split / seq,
        *(root / name / split / seq for name in nested_names),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_ho3d_anchor(
    entry: dict,
    canonical_data: dict[str, Any],
    source: str,
    anchor_canonical_idx: int | None = None,
) -> AnchorDepthResult | None:
    seq_dir = _resolve_ho3d_seq_dir(entry, source)
    if seq_dir is None:
        return None
    anchor_canonical, anchor_video = _anchor_indices(canonical_data, anchor_canonical_idx)
    rgb_path = _find_numbered_image(seq_dir / "rgb", anchor_video, (".png", ".jpg", ".jpeg"))
    depth_path = _find_numbered_image(seq_dir / "depth", anchor_video, (".png", ".tiff", ".tif"))
    if rgb_path is None or depth_path is None:
        return None
    bgr = cv2.imread(str(rgb_path))
    depth_u16 = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    K = canonical_data.get("camera_K")
    if bgr is None or depth_u16 is None or K is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    intrinsics = K.numpy().astype(np.float32) if hasattr(K, "numpy") else np.asarray(K, dtype=np.float32)
    if depth_u16.ndim == 3 and depth_u16.shape[2] >= 3 and depth_u16.dtype == np.uint8:
        # Official HO-3D encoding: 16-bit depth split into R (low byte) and G (high byte).
        depth = (depth_u16[:, :, 2].astype(np.float32) + depth_u16[:, :, 1].astype(np.float32) * 256.0) * HO3D_DEPTH_SCALE_M
    elif depth_u16.ndim == 2 and depth_u16.dtype == np.uint16:
        depth = depth_u16.astype(np.float32) / 1000.0
    else:
        logger.warning(f"{entry['segment_id']}: unsupported {source.upper()} depth image {depth_u16.dtype} {depth_u16.shape} in {depth_path}")
        return None
    if depth.shape != rgb.shape[:2]:
        depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    hand_mask, obj_mask = _masks_from_canonical_verts(canonical_data, anchor_canonical, intrinsics, depth.shape)
    return AnchorDepthResult(
        depth=depth,
        rgb=rgb,
        intrinsics=intrinsics,
        anchor_canonical_idx=anchor_canonical,
        anchor_video_idx=anchor_video,
        hand_seg_mask=hand_mask,
        obj_seg_mask=obj_mask,
    )


def _find_taco_depth_video(depth_dir: Path) -> Path | None:
    if not depth_dir.exists():
        return None
    candidates = []
    for ext in ("*.avi", "*.mp4", "*.mkv", "*.mov"):
        candidates.extend(sorted(depth_dir.glob(ext)))
    return candidates[0] if candidates else None


def _video_frame_count(video_path: Path) -> int:
    cap = _open_video_capture(video_path)
    if not cap.isOpened():
        return 0
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()


def _taco_pose_frame_count(triplet: str, sequence: str) -> int | None:
    cam_path = TACO_DIR / "Egocentric_Camera_Parameters" / triplet / sequence / "egocentric_frame_extrinsic.npy"
    if not cam_path.exists():
        return None
    try:
        return int(np.load(str(cam_path), mmap_mode="r").shape[0])
    except Exception:
        return None


def _map_taco_pose_frame_to_media(frame_idx: int, triplet: str, sequence: str, media_path: Path) -> int:
    media_frames = _video_frame_count(media_path)
    if media_frames <= 0:
        return int(frame_idx)
    pose_frames = _taco_pose_frame_count(triplet, sequence)
    if pose_frames is not None and pose_frames > 1 and media_frames < pose_frames:
        scale = float(media_frames - 1) / float(pose_frames - 1)
        mapped = int(round(float(frame_idx) * scale))
    else:
        mapped = int(frame_idx)
    return int(np.clip(mapped, 0, media_frames - 1))


# TACO egocentric depth: FFV1 gray16le, one uint16 per pixel in units of 0.25 mm
# (RealSense L515), one depth frame per pose frame.
TACO_DEPTH_UNITS_PER_M = 4000.0


def _decode_taco_depth_m(depth_path: Path, frame_idx: int) -> np.ndarray | None:
    """Decode one TACO depth frame in meters (OpenCV's default BGR conversion would keep only the high byte)."""
    cap = cv2.VideoCapture(str(depth_path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_CONVERT_RGB, 0])
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
        ok, raw = cap.read()
    finally:
        cap.release()
    if not ok or raw is None or raw.dtype != np.uint16:
        return None
    return raw.reshape(raw.shape[0], raw.shape[1]).astype(np.float32) / TACO_DEPTH_UNITS_PER_M


def _load_taco_anchor(entry: dict, canonical_data: dict[str, Any], anchor_canonical_idx: int | None = None) -> AnchorDepthResult | None:
    triplet = str(entry.get("taco_triplet", ""))
    sequence = str(entry.get("taco_sequence", ""))
    if not triplet or not sequence:
        return None
    rgb_path = TACO_DIR / "Egocentric_RGB_Videos" / triplet / sequence / "color.mp4"
    depth_path = _find_taco_depth_video(TACO_DIR / "Egocentric_Depth_Videos" / triplet / sequence)
    if depth_path is None or not rgb_path.exists():
        return None

    anchor_canonical, anchor_video = _anchor_indices(canonical_data, anchor_canonical_idx)
    rgb_frame = _map_taco_pose_frame_to_media(anchor_video, triplet, sequence, rgb_path)
    depth_frame = _map_taco_pose_frame_to_media(anchor_video, triplet, sequence, depth_path)
    bgr = _decode_video_frame_bgr_cached(rgb_path, rgb_frame)
    depth = _decode_taco_depth_m(depth_path, depth_frame)
    K = canonical_data.get("camera_K")
    if bgr is None or depth is None or K is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if depth.shape != rgb.shape[:2]:
        # Some early sequences store raw 1024x768 sensor depth that is not registered to the
        # color image and ships no depth calibration; the caller falls back to MoGe.
        logger.warning(f"{entry['segment_id']}: TACO depth {depth.shape} is not registered to the color image; using MoGe")
        return None

    intrinsics = K.numpy().astype(np.float32) if hasattr(K, "numpy") else np.asarray(K, dtype=np.float32)
    hand_mask, obj_mask = _masks_from_canonical_verts(canonical_data, anchor_canonical, intrinsics, depth.shape)
    return AnchorDepthResult(
        depth=depth,
        rgb=rgb,
        intrinsics=intrinsics,
        anchor_canonical_idx=anchor_canonical,
        anchor_video_idx=anchor_video,
        hand_seg_mask=hand_mask,
        obj_seg_mask=obj_mask,
    )


def _load_video_moge_anchor(
    entry: dict,
    canonical_data: dict[str, Any],
    moge_model: Any,
    anchor_canonical_idx: int | None = None,
) -> AnchorDepthResult | None:
    seg_dir = _canonical_seg_dir(entry)
    video_path = seg_dir / "video.mp4"
    if not video_path.exists():
        return None
    anchor_canonical, anchor_video = _anchor_indices(canonical_data, anchor_canonical_idx)
    # Seek through the row → video-frame binding: interaction children symlink the
    # parent video, and decimated SHOW3D segments step 2 video frames per row.
    rgb = _decode_video_frame(
        video_path,
        canonical_row_to_video_frame(canonical_data, seg_dir, anchor_canonical),
    )
    if rgb is None:
        return None
    depth, intrinsics = _run_moge(moge_model, rgb)
    hand_mask, obj_mask = _masks_from_canonical_verts(canonical_data, anchor_canonical, intrinsics, depth.shape)
    return AnchorDepthResult(
        depth=depth,
        rgb=rgb,
        intrinsics=intrinsics,
        anchor_canonical_idx=anchor_canonical,
        anchor_video_idx=anchor_video,
        hand_seg_mask=hand_mask,
        obj_seg_mask=obj_mask,
    )


# ---------------------------------------------------------------------------
# Canonical loading helper (includes the per-frame verts)
# ---------------------------------------------------------------------------


_SCENE_CANONICAL_FIELDS = (
    "frame_indices",
    "fps",
    "hand_left_orient",
    "hand_left_pose",
    "hand_left_cam_t",
    "hand_left_betas",
    "hand_left_valid",
    "hand_left_verts",
    "hand_right_orient",
    "hand_right_pose",
    "hand_right_cam_t",
    "hand_right_betas",
    "hand_right_valid",
    "hand_right_verts",
    "obj_rotation",
    "obj_translation",
    "obj_verts",
    "obj_canonical_verts",
    "obj_faces",
    "obj_scale",
    "gravity_direction",
    "support_plane_height",
    "camera_K",
    "camera_extrinsics",
    "gravity_world",
    "hand_left_orient_world",
    "hand_left_cam_t_world",
    "hand_right_orient_world",
    "hand_right_cam_t_world",
    "obj_rotation_world",
    "obj_translation_world",
)


def load_canonical_for_scene(canonical_path: str | os.PathLike) -> dict[str, Any]:
    """Load the canonical.npz fields needed for scene preprocessing.

    Includes the per-frame vertex arrays (hand_*_verts, obj_verts) that gravity_align
    rotates and the scene features measure distances against.
    """
    result: dict[str, Any] = {}
    with np.load(str(canonical_path), allow_pickle=False) as npz:
        for key in _SCENE_CANONICAL_FIELDS:
            if key not in npz.files:
                continue
            v = npz[key]
            if v.ndim == 0:
                if v.dtype.kind in "fiu":
                    result[key] = float(v) if v.dtype.kind == "f" else int(v)
                else:
                    result[key] = v.item()
            else:
                result[key] = torch.from_numpy(v.copy())
    return result
