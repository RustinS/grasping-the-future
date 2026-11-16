#!/usr/bin/env python3
"""Build the per-frame scene caches (``canonical/scene_cache_v2.h5``) of every manifest segment.

For every canonical frame (the "present" frame the model conditions on):
    re-express the canonical in that frame's camera (moving-camera sources)
    → load the frame's depth + RGB + hand/object masks (source-specific)
    → gravity-align the canonical and rotate the unprojected scene points with the same R
    → crop around the manipulation, FPS-select n_points, compute 12-D geometric features
      and 32-D DINOv3 semantic features (global PCA basis from gtf/datasets/scene/fit_pca.py)
    → write one row of scene_cache_v2.h5

Frames whose masks leave too few background points stay invalid; a segment whose
cache covers fewer than 80% of its frames is discarded. Each cache is built into a
``.rebuild`` sibling and moved into place when complete; existing valid caches are
skipped unless --force.

``--depth estimated`` replaces the sensor depth of the RGB-D sources (DexYCB, H2O, HO-3D,
HO-Cap, TACO) with MoGe-2 depth of the same frame, conditioned on the field of view of the
sensor-aligned intrinsics; frames without sensor depth use the normal path. With
``--cache-root DIR`` the caches go to DIR/<segment_id>/scene_cache_v2.h5 instead of next to
the canonical; ``build_manifest.py --scene-cache-root DIR`` then writes a manifest that uses
them (other segments keep their normal caches).

Usage:
    python -m gtf.datasets.scene.preprocess
    python -m gtf.datasets.scene.preprocess --sources taco,show3d --num-shards 8 --shard-idx 3
    python -m gtf.datasets.scene.preprocess --frame-batch-size 32 --scene-cpu-workers 4 --strict
    python -m gtf.datasets.scene.preprocess --depth estimated --cache-root datasets/scene_estimated_depth
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
import json
import logging
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from gtf.paths import MANO_DIR, REPO_ROOT
from gtf.datasets.build_manifest import SCENE_CACHE_MIN_COVERAGE, _scene_cache_hdf5_status
from gtf.datasets.scene.depth_sources import (
    MOGE_MODEL_ID,
    MOGE_SOURCES,
    TACO_DIR,
    AnchorDepthResult,
    _anchor_indices,
    _canonical_seg_dir,
    _decode_video_frame,
    canonical_row_to_video_frame,
    _masks_from_canonical_verts,
    _run_moge_batch,
    load_anchor_depth,
    load_canonical_for_scene,
)
from gtf.datasets.scene.scene_pipeline import (
    PreparedSceneCloud,
    SceneFeatureFinishTask,
    batch_fps_indices,
    finish_scene_features,
    finish_scene_features_task,
    initialize_scene_cpu_worker,
    prepare_scene_cloud,
)
from gtf.model.repr.canonicalize import compute_gravity_alignment, gravity_align
from gtf.utils import load_mano_faces
from gtf.utils.canonical import mano_fk_verts, rotmat_to_aa
from gtf.utils.hand_rigid import transform_hand_rigid

ROOT = REPO_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Sources that bring their own depth: sensor depth, or the extraction pipeline's depth for Action100M.
# Sources in MOGE_SOURCES and sources named in neither set use MoGe-2 on the segment video.
DEPTH_SOURCES = {"dexycb", "h2o", "ho3d", "h2o3d", "hocap", "taco", "action100m"}
# Sources with sensor depth, replaced by MoGe-2 under --depth estimated.
SENSOR_DEPTH_SOURCES = {"dexycb", "h2o", "ho3d", "h2o3d", "hocap", "taco"}

_MANO_FACES_SCENE = None


def _mano_faces_for_scene() -> np.ndarray:
    """MANO faces (cached) for the point-to-triangle hand distance.

    Both hands share one (1552, 3) topology and the unsigned distance does not depend
    on the winding, so the right face set indexes each hand's 778 verts.
    """
    global _MANO_FACES_SCENE
    if _MANO_FACES_SCENE is None:
        faces_right, _faces_left = load_mano_faces()
        _MANO_FACES_SCENE = np.asarray(faces_right, dtype=np.int64)
    return _MANO_FACES_SCENE


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m gtf.datasets.scene.preprocess", description="Build per-frame scene caches")
    p.add_argument("--manifest", default=str(ROOT / "datasets" / "canonical_manifest.json"))
    p.add_argument("--sources", default=None, help="Comma-separated source datasets (default: all)")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--pca-basis", default=str(ROOT / "datasets" / "scene_dinov3_pca.pt"))
    p.add_argument("--n-points", type=int, default=4096)
    p.add_argument("--crop-radius", type=float, default=0.5)
    p.add_argument("--normal-k", type=int, default=30)
    p.add_argument("--limit", type=int, default=0, help="Process at most N segments")
    p.add_argument("--force", action="store_true", help="Rebuild existing valid caches")
    p.add_argument("--strict", action="store_true", help="Exit nonzero on the first failed segment")
    p.add_argument("--frame-batch-size", type=int, default=32, help="Frames per batched MoGe/DINO/FPS forward")
    p.add_argument("--scene-cpu-workers", type=int, default=0, help="Spawn workers for the CPU feature tail (0: in-process)")
    p.add_argument("--depth", choices=("native", "estimated"), default="native", help="Sensor depth, or MoGe-2 depth for the RGB-D sources")
    p.add_argument("--cache-root", default=None, help="Write caches to DIR/<segment_id>/scene_cache_v2.h5 (default: next to the canonical)")
    return p.parse_args()


def _entry_n_frames(entry: dict) -> int:
    n_frames = entry.get("n_frames")
    if n_frames is not None:
        try:
            return max(0, int(n_frames))
        except Exception:
            pass
    try:
        with np.load(str(entry["canonical_path"]), allow_pickle=False) as npz:
            return int(npz["frame_indices"].shape[0])
    except Exception:
        return 0


def _has_taco_native_rgbd(entry: dict) -> bool:
    triplet = str(entry.get("taco_triplet", ""))
    sequence = str(entry.get("taco_sequence", ""))
    if not triplet or not sequence:
        return False
    rgb = TACO_DIR / "Egocentric_RGB_Videos" / triplet / sequence / "color.mp4"
    depth_dir = TACO_DIR / "Egocentric_Depth_Videos" / triplet / sequence
    return rgb.exists() and depth_dir.exists() and any(depth_dir.glob("*.avi"))


def _has_scene_media(entry: dict) -> bool:
    """True when the media a scene cache is built from exist (the segment video, or TACO's native RGB-D)."""
    if (_canonical_seg_dir(entry) / "video.mp4").exists():
        return True
    return str(entry.get("source_dataset", "")).lower() == "taco" and _has_taco_native_rgbd(entry)


def _cache_path(entry: dict, cache_root: str | None) -> Path:
    if cache_root:
        return Path(cache_root) / str(entry["segment_id"]) / "scene_cache_v2.h5"
    return Path(entry["canonical_path"]).parent / "scene_cache_v2.h5"


def _estimated_depth_anchor(entry, scene_canonical, moge_model, anchor_idx) -> AnchorDepthResult | None:
    """The sensor-depth anchor with its depth replaced by MoGe-2 (None when the frame has no sensor depth).

    MoGe runs on the RGB at the sensor depth map's resolution, conditioned on the horizontal
    field of view of the sensor-aligned intrinsics; pixels outside its validity mask get 0.
    """
    import cv2

    native = load_anchor_depth(entry, scene_canonical, None, anchor_canonical_idx=int(anchor_idx))
    if native is None:
        return None
    H, W = native.depth.shape
    rgb = native.rgb if native.rgb.shape[:2] == (H, W) else cv2.resize(native.rgb, (W, H), interpolation=cv2.INTER_AREA)
    device = next(moge_model.parameters()).device
    image = torch.from_numpy(rgb).to(device, dtype=torch.float32).permute(2, 0, 1) / 255.0
    fov_x = float(np.degrees(2 * np.arctan(W / (2 * float(native.intrinsics[0, 0])))))
    with torch.inference_mode():
        out = moge_model.infer(image, fov_x=fov_x)
    depth = out["depth"].detach().float().cpu().numpy().astype(np.float32)
    mask = out["mask"].detach().cpu().numpy().astype(bool) if out.get("mask") is not None else np.isfinite(depth)
    native.depth = np.where(mask & np.isfinite(depth), depth, 0.0).astype(np.float32)
    return native


def _select_entries(manifest: dict, sources: set[str] | None) -> list[dict]:
    entries = [e for e in manifest["entries"] if e.get("validated", True)]
    if sources is not None:
        entries = [e for e in entries if str(e.get("source_dataset", "")).lower() in sources]
    entries = [e for e in entries if _has_scene_media(e)]
    entries.sort(key=lambda e: str(e["segment_id"]))
    return entries


def _select_shard(entries: list[dict], num_shards: int, shard_idx: int) -> list[dict]:
    """Greedy shard assignment balanced by frame count."""
    if num_shards <= 1:
        return entries
    buckets: list[list[dict]] = [[] for _ in range(num_shards)]
    totals = [0 for _ in range(num_shards)]
    for entry in sorted(entries, key=lambda e: (_entry_n_frames(e), str(e["segment_id"])), reverse=True):
        j = min(range(num_shards), key=lambda k: totals[k])
        buckets[j].append(entry)
        totals[j] += _entry_n_frames(entry)
    bucket = buckets[shard_idx]
    bucket.sort(key=lambda e: str(e["segment_id"]))
    return bucket


def _load_pca_basis(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        sys.exit(f"Missing PCA basis at {path}. Run python -m gtf.datasets.scene.fit_pca first.")
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    return {
        "components": np.asarray(ckpt["components"], dtype=np.float32),
        "mean": np.asarray(ckpt["mean"], dtype=np.float32),
    }


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _has_dynamic_present_scene_data(canonical_data: dict[str, Any]) -> bool:
    required = {
        "camera_extrinsics",
        "hand_left_orient_world",
        "hand_left_cam_t_world",
        "hand_right_orient_world",
        "hand_right_cam_t_world",
        "obj_rotation_world",
        "obj_translation_world",
    }
    if not required.issubset(canonical_data.keys()):
        return False
    cam = _to_numpy(canonical_data["camera_extrinsics"]).astype(np.float32)
    if cam.ndim != 3 or cam.shape[0] < 2:
        return False
    return bool(np.max(np.abs(cam - cam[:1])) > 1e-5)


def _prepare_present_scene_canonical(canonical_data: dict[str, Any], anchor_idx: int) -> dict[str, Any]:
    """Return the canonical fields expressed in the camera of frame ``anchor_idx``.

    Only moving-camera canonicals (world fields + per-frame extrinsics) change. Only
    the anchor-frame poses and verts are materialized: the scene build needs them for
    masks, the crop center and the surface distances.
    """
    if not _has_dynamic_present_scene_data(canonical_data):
        return canonical_data

    frame_indices = _to_numpy(canonical_data["frame_indices"])
    T = int(frame_indices.shape[0])
    anchor_idx = int(np.clip(anchor_idx, 0, T - 1))
    cam = _to_numpy(canonical_data["camera_extrinsics"]).astype(np.float32)
    E = cam[anchor_idx]
    R_cw = E[:3, :3]
    t_cw = E[:3, 3]

    result = dict(canonical_data)
    dtype = np.float32

    obj_rot_w = _to_numpy(canonical_data["obj_rotation_world"]).astype(dtype)
    obj_trans_w = _to_numpy(canonical_data["obj_translation_world"]).astype(dtype)
    obj_R = np.zeros((T, 3), dtype=dtype)
    obj_t = np.zeros((T, 3), dtype=dtype)
    Rw = Rotation.from_rotvec(obj_rot_w[anchor_idx]).as_matrix().astype(dtype)
    Rc = R_cw @ Rw
    obj_R[anchor_idx] = rotmat_to_aa(Rc[None])[0].astype(dtype)
    obj_t[anchor_idx] = (R_cw @ obj_trans_w[anchor_idx] + t_cw).astype(dtype)
    result["obj_rotation"] = torch.from_numpy(obj_R)
    result["obj_translation"] = torch.from_numpy(obj_t)

    verts0 = _to_numpy(canonical_data["obj_canonical_verts"]).astype(dtype)
    if "obj_scale" in canonical_data:
        scale = _to_numpy(canonical_data["obj_scale"]).astype(dtype).reshape(-1)
        if scale.size in (1, 3):
            verts0 = verts0 * scale
    obj_verts = np.zeros((T, verts0.shape[0], 3), dtype=dtype)
    R_anchor = Rotation.from_rotvec(obj_R[anchor_idx]).as_matrix().astype(dtype)
    obj_verts[anchor_idx] = verts0 @ R_anchor.T + obj_t[anchor_idx]
    result["obj_verts"] = torch.from_numpy(obj_verts)

    for side in ("left", "right"):
        orient_w = _to_numpy(canonical_data[f"hand_{side}_orient_world"]).astype(dtype)
        trans_w = _to_numpy(canonical_data[f"hand_{side}_cam_t_world"]).astype(dtype)
        valid_full = _to_numpy(canonical_data[f"hand_{side}_valid"]).astype(bool)
        valid = np.zeros(T, dtype=bool)
        valid[anchor_idx] = bool(valid_full[anchor_idx])
        orient_c = np.zeros((T, 3), dtype=dtype)
        trans_c = np.zeros((T, 3), dtype=dtype)
        if bool(valid[anchor_idx]):
            # World fields are authored with the shaped J0(betas) wrist pivot; the same
            # pivot inverts them exactly (left hands are M-conjugated inside).
            o_one, t_one = transform_hand_rigid(
                orient_w[anchor_idx : anchor_idx + 1],
                trans_w[anchor_idx : anchor_idx + 1],
                np.ones(1, dtype=bool),
                side,
                R=R_cw,
                t=t_cw,
                betas=_to_numpy(canonical_data[f"hand_{side}_betas"]).astype(np.float32),
            )
            orient_c[anchor_idx] = o_one[0]
            trans_c[anchor_idx] = t_one[0]
        result[f"hand_{side}_orient"] = torch.from_numpy(orient_c.astype(dtype))
        result[f"hand_{side}_cam_t"] = torch.from_numpy(trans_c.astype(dtype))
        result[f"hand_{side}_valid"] = torch.from_numpy(valid)

        hand_verts = np.zeros((T, 778, 3), dtype=dtype)
        if bool(valid[anchor_idx]):
            pose = _to_numpy(canonical_data[f"hand_{side}_pose"])[anchor_idx : anchor_idx + 1].astype(dtype)
            betas = _to_numpy(canonical_data[f"hand_{side}_betas"]).astype(dtype)
            hand_verts[anchor_idx] = mano_fk_verts(
                orient_c[anchor_idx : anchor_idx + 1],
                pose,
                betas,
                trans_c[anchor_idx : anchor_idx + 1],
                side,
                MANO_DIR,
            )[0].astype(dtype)
        result[f"hand_{side}_verts"] = torch.from_numpy(hand_verts)

    gravity_world = canonical_data.get("gravity_world")
    if gravity_world is not None:
        g = _to_numpy(gravity_world).astype(dtype)
        if np.linalg.norm(g) > 1e-6:
            result["gravity_direction"] = torch.from_numpy((R_cw @ g).astype(dtype))
    elif "gravity_direction" in result:
        result.pop("gravity_direction", None)
    return result


def _features_to_payload(features: dict[str, Any], R_world_from_cam: np.ndarray, anchor: AnchorDepthResult) -> dict[str, Any]:
    """Wrap scene features with the per-frame cache metadata."""
    has_sem = features["scene_sem_feats"] is not None
    return {
        "P_scene": features["P_scene"].astype(np.float32),
        "scene_normals": features["scene_normals"].astype(np.float32),
        "scene_geo_feats": features["scene_geo_feats"].astype(np.float32),
        "scene_sem_feats": features["scene_sem_feats"].astype(np.float32) if has_sem else None,
        "support_plane_y": np.float32(features["support_plane_y"] if features["support_plane_y"] is not None else np.nan),
        "R_world_from_cam": np.asarray(R_world_from_cam, dtype=np.float32),
        "anchor_canonical_idx": np.int64(anchor.anchor_canonical_idx),
        "anchor_video_idx": np.int64(anchor.anchor_video_idx),
        "n_points_actual": np.int64(features["n_points_actual"]),
    }


_COMPRESSION = {"compression": "gzip", "compression_opts": 4, "shuffle": True}


def _create_hdf5_scene_cache(path: Path, n_frames: int, n_points: int, semantic_dim: int) -> Any:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    h5 = h5py.File(str(path), "w")
    per_frame = min(n_frames, 1024)
    h5.create_dataset("P_scene", shape=(n_frames, n_points, 3), dtype="float32", chunks=(1, n_points, 3), fillvalue=0, **_COMPRESSION)
    h5.create_dataset("scene_normals", shape=(n_frames, n_points, 3), dtype="float32", chunks=(1, n_points, 3), fillvalue=0, **_COMPRESSION)
    h5.create_dataset("scene_geo_feats", shape=(n_frames, n_points, 12), dtype="float32", chunks=(1, n_points, 12), fillvalue=0, **_COMPRESSION)
    h5.create_dataset("scene_sem_feats", shape=(n_frames, n_points, semantic_dim), dtype="float32", chunks=(1, n_points, semantic_dim), fillvalue=0, **_COMPRESSION)
    h5.create_dataset("valid", shape=(n_frames,), dtype="bool", chunks=(per_frame,), fillvalue=False)
    h5.create_dataset("support_plane_y", shape=(n_frames,), dtype="float32", chunks=(per_frame,), fillvalue=np.nan)
    h5.create_dataset("R_world_from_cam", shape=(n_frames, 3, 3), dtype="float32", chunks=(1, 3, 3), fillvalue=0)
    h5.create_dataset("anchor_canonical_idx", shape=(n_frames,), dtype="int64", chunks=(per_frame,), fillvalue=-1)
    h5.create_dataset("anchor_video_idx", shape=(n_frames,), dtype="int64", chunks=(per_frame,), fillvalue=-1)
    h5.create_dataset("n_points_actual", shape=(n_frames,), dtype="int64", chunks=(per_frame,), fillvalue=0)
    h5.attrs["format_version"] = "scene_cache_v2"
    h5.attrs["n_points"] = int(n_points)
    h5.attrs["has_semantic"] = True
    h5.attrs["semantic_dim"] = int(semantic_dim)
    return h5


def _write_hdf5_frame(h5: Any, canonical_idx: int, payload: dict[str, Any]) -> None:
    idx = int(canonical_idx)
    h5["P_scene"][idx] = payload["P_scene"]
    h5["scene_normals"][idx] = payload["scene_normals"]
    h5["scene_geo_feats"][idx] = payload["scene_geo_feats"]
    sem = payload.get("scene_sem_feats")
    h5["scene_sem_feats"][idx] = np.zeros(h5["scene_sem_feats"].shape[1:], dtype=np.float32) if sem is None else sem
    h5["support_plane_y"][idx] = np.float32(payload["support_plane_y"])
    h5["R_world_from_cam"][idx] = payload["R_world_from_cam"]
    h5["anchor_canonical_idx"][idx] = np.int64(payload["anchor_canonical_idx"])
    h5["anchor_video_idx"][idx] = np.int64(payload["anchor_video_idx"])
    h5["n_points_actual"][idx] = np.int64(payload["n_points_actual"])
    h5["valid"][idx] = True


def _batched_moge_anchors(
    entry: dict,
    scene_canonicals: list[dict[str, Any]],
    frame_ids: list[int],
    moge_model: Any,
) -> list[AnchorDepthResult | None]:
    """MoGe depth for a batch of frames of an RGB-only source."""
    seg_dir = _canonical_seg_dir(entry)
    video_path = seg_dir / "video.mp4"
    if not video_path.exists():
        return [None] * len(frame_ids)

    records: list[tuple[int, int, dict[str, Any], np.ndarray] | None] = []
    rgbs: list[np.ndarray] = []
    for scene_canonical, anchor_idx in zip(scene_canonicals, frame_ids):
        anchor_canonical, anchor_video = _anchor_indices(scene_canonical, anchor_idx)
        rgb = _decode_video_frame(video_path, canonical_row_to_video_frame(scene_canonical, seg_dir, anchor_canonical))
        if rgb is None:
            records.append(None)
            continue
        records.append((anchor_canonical, anchor_video, scene_canonical, rgb))
        rgbs.append(rgb)
    if not rgbs:
        return [None] * len(frame_ids)

    moge_outputs = _run_moge_batch(moge_model, rgbs)
    out: list[AnchorDepthResult | None] = []
    j = 0
    for rec in records:
        if rec is None:
            out.append(None)
            continue
        anchor_canonical, anchor_video, scene_canonical, rgb = rec
        depth, intrinsics = moge_outputs[j]
        j += 1
        hand_mask, obj_mask = _masks_from_canonical_verts(scene_canonical, anchor_canonical, intrinsics, depth.shape)
        out.append(
            AnchorDepthResult(
                depth=depth,
                rgb=rgb,
                intrinsics=intrinsics,
                anchor_canonical_idx=anchor_canonical,
                anchor_video_idx=anchor_video,
                hand_seg_mask=hand_mask,
                obj_seg_mask=obj_mask,
            )
        )
    return out


def _batched_dino_patch_features(
    dino_extractor: Any,
    pca_basis: dict[str, np.ndarray],
    anchors: list[AnchorDepthResult | None],
) -> tuple[list[np.ndarray | None], int, bool]:
    """DINOv3 patch features of a batch, PCA-projected on the GPU (per-frame CPU fallback)."""
    patches: list[np.ndarray | None] = [None] * len(anchors)
    valid_positions = [i for i, anchor in enumerate(anchors) if anchor is not None]
    projected = True
    if valid_positions:
        rgbs = [anchors[i].rgb for i in valid_positions]
        try:
            batch_patches = dino_extractor.extract_projected_patch_features_batch(rgbs, pca_basis)
        except Exception as e:
            logger.warning(f"batched DINO failed ({type(e).__name__}: {e}); falling back to per-frame DINO for this chunk")
            batch_patches = [dino_extractor.extract_patch_features(rgb) for rgb in rgbs]
            projected = False
        for i, patch in zip(valid_positions, batch_patches):
            patches[i] = patch
    # Read after the forward: the extractor sets patch_size when it loads the model.
    patch_size = getattr(dino_extractor, "patch_size", None)
    return patches, 16 if patch_size is None else int(patch_size), projected


@dataclass
class _PreparedFrame:
    """One frame ready for the CPU feature tail, plus the cache metadata the parent keeps."""

    anchor: AnchorDepthResult
    R_world_from_cam: np.ndarray
    task: SceneFeatureFinishTask


def _prepare_frame_batch(
    entry: dict,
    moge_model: Any | None,
    dino_extractor: Any,
    pca_basis: dict[str, np.ndarray],
    args: argparse.Namespace,
    canonical_data: dict[str, Any],
    frame_ids: list[int],
) -> list[_PreparedFrame | None]:
    """Depth, DINO, gravity alignment, cropping and FPS for one batch of frames (GPU side)."""
    scene_canonicals = [_prepare_present_scene_canonical(canonical_data, int(anchor_idx)) for anchor_idx in frame_ids]
    anchors: list[AnchorDepthResult | None] | None = None
    if args.depth == "estimated" and str(entry.get("source_dataset", "")).lower() in SENSOR_DEPTH_SOURCES:
        anchors = [
            _estimated_depth_anchor(entry, scene_canonical, moge_model, anchor_idx)
            or load_anchor_depth(entry, scene_canonical, moge_model, anchor_canonical_idx=int(anchor_idx))
            for scene_canonical, anchor_idx in zip(scene_canonicals, frame_ids)
        ]
    elif moge_model is not None and _uses_moge(entry):
        try:
            anchors = _batched_moge_anchors(entry, scene_canonicals, frame_ids, moge_model)
        except Exception as e:
            logger.warning(f"{entry['segment_id']}: batched MoGe failed ({type(e).__name__}: {e}); falling back to per-frame anchors")
            anchors = None
    if anchors is None:
        anchors = [
            load_anchor_depth(entry, scene_canonical, moge_model, anchor_canonical_idx=int(anchor_idx))
            for scene_canonical, anchor_idx in zip(scene_canonicals, frame_ids)
        ]
    patch_features, patch_size, projected = _batched_dino_patch_features(dino_extractor, pca_basis, anchors)

    prepared_by_pos: list[tuple[PreparedSceneCloud, np.ndarray, AnchorDepthResult, np.ndarray | None] | None] = []
    for scene_canonical, anchor, patches in zip(scene_canonicals, anchors, patch_features):
        if anchor is None:
            prepared_by_pos.append(None)
            continue
        gravity = scene_canonical.get("gravity_direction")
        # Without a gravity estimate the canonical stays in camera space (identity).
        canonical_aligned = gravity_align(scene_canonical, gravity)
        R_world_from_cam = compute_gravity_alignment(gravity).numpy().astype(np.float32) if gravity is not None else np.eye(3, dtype=np.float32)
        prepared = prepare_scene_cloud(
            depth=anchor.depth,
            intrinsics=anchor.intrinsics,
            hand_mask=anchor.hand_seg_mask,
            obj_mask=anchor.obj_seg_mask,
            canonical_aligned=canonical_aligned,
            R_world_from_cam=R_world_from_cam,
            anchor_canonical_idx=anchor.anchor_canonical_idx,
            crop_radius_m=float(args.crop_radius),
            hand_faces=_mano_faces_for_scene(),
        )
        prepared_by_pos.append(None if prepared is None else (prepared, R_world_from_cam, anchor, patches))

    valid_positions = [pos for pos, value in enumerate(prepared_by_pos) if value is not None]
    fps_by_valid = batch_fps_indices(
        [prepared_by_pos[pos][0].pts_world for pos in valid_positions],  # type: ignore[index]
        int(args.n_points),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    out: list[_PreparedFrame | None] = [None] * len(frame_ids)
    for pos, fps_indices in zip(valid_positions, fps_by_valid):
        prepared, R_world_from_cam, anchor, patches = prepared_by_pos[pos]  # type: ignore[misc]
        task = SceneFeatureFinishTask(
            task_id=int(anchor.anchor_canonical_idx),
            prepared=prepared,
            fps_indices=fps_indices,
            normal_k=int(args.normal_k),
            pca_basis=pca_basis,
            dino_patch_features=patches,
            dino_patch_size=int(patch_size),
            dino_patch_features_projected=bool(projected),
        )
        out[pos] = _PreparedFrame(anchor=anchor, R_world_from_cam=R_world_from_cam, task=task)
    return out


def _finish_inline(prepared: list[_PreparedFrame | None]) -> list[dict[str, Any] | None]:
    payloads: list[dict[str, Any] | None] = []
    for item in prepared:
        if item is None:
            payloads.append(None)
            continue
        t = item.task
        features = finish_scene_features(
            t.prepared,
            t.fps_indices,
            normal_k=t.normal_k,
            pca_basis=t.pca_basis,
            dino_patch_features=t.dino_patch_features,
            dino_patch_size=t.dino_patch_size,
            dino_patch_features_projected=t.dino_patch_features_projected,
        )
        payloads.append(_features_to_payload(features, item.R_world_from_cam, item.anchor))
    return payloads


def _submit(prepared: list[_PreparedFrame | None], pool: ProcessPoolExecutor) -> list[tuple[_PreparedFrame, Future] | None]:
    return [None if item is None else (item, pool.submit(finish_scene_features_task, item.task)) for item in prepared]


def _collect(submitted: list[tuple[_PreparedFrame, Future] | None]) -> list[dict[str, Any] | None]:
    payloads: list[dict[str, Any] | None] = []
    for item in submitted:
        if item is None:
            payloads.append(None)
            continue
        frame, future = item
        result = future.result()
        if int(result.task_id) != int(frame.anchor.anchor_canonical_idx):
            raise RuntimeError(f"scene CPU worker returned task {result.task_id}, expected {frame.anchor.anchor_canonical_idx}")
        payloads.append(_features_to_payload(result.payload, frame.R_world_from_cam, frame.anchor))
    return payloads


def _iter_frame_payload_batches(entry, moge_model, dino_extractor, pca_basis, args, canonical_data, frame_ids, scene_cpu_pool):
    """Yield (frame ids, payloads) per batch; with a CPU pool, batch k's tail overlaps batch k+1's GPU work."""
    batch_size = max(1, int(args.frame_batch_size))
    chunks = [frame_ids[start : start + batch_size] for start in range(0, len(frame_ids), batch_size)]
    if scene_cpu_pool is None:
        for chunk in chunks:
            yield chunk, _finish_inline(_prepare_frame_batch(entry, moge_model, dino_extractor, pca_basis, args, canonical_data, chunk))
        return
    previous = None
    for chunk in chunks:
        try:
            current = (chunk, _submit(_prepare_frame_batch(entry, moge_model, dino_extractor, pca_basis, args, canonical_data, chunk), scene_cpu_pool))
        except Exception:
            if previous is not None:
                yield previous[0], _collect(previous[1])
            raise
        if previous is not None:
            yield previous[0], _collect(previous[1])
        previous = current
    if previous is not None:
        yield previous[0], _collect(previous[1])


def _uses_moge(entry: dict) -> bool:
    source = str(entry.get("source_dataset", "")).lower()
    return source in MOGE_SOURCES or source not in DEPTH_SOURCES or (source == "taco" and not _has_taco_native_rgbd(entry))


def _build_segment_cache(entry, moge_model, dino_extractor, pca_basis, args, scene_cpu_pool) -> tuple[int, int, bool]:
    """Build one segment's scene_cache_v2.h5. Returns (ok_frames, failed_frames, accepted)."""
    canonical_data = load_canonical_for_scene(entry["canonical_path"])
    n_frames = int(_to_numpy(canonical_data["frame_indices"]).shape[0])
    frame_ids = list(range(n_frames))
    semantic_dim = int(np.asarray(pca_basis["components"]).shape[0])
    out_path = _cache_path(entry, args.cache_root)
    build_path = out_path.with_name(out_path.name + ".rebuild")
    build_path.unlink(missing_ok=True)

    n_ok = n_fail = 0
    with _create_hdf5_scene_cache(build_path, n_frames, int(args.n_points), semantic_dim) as h5:
        h5.attrs["depth"] = str(args.depth)
        for chunk, payloads in _iter_frame_payload_batches(entry, moge_model, dino_extractor, pca_basis, args, canonical_data, frame_ids, scene_cpu_pool):
            for anchor_idx, payload in zip(chunk, payloads):
                if payload is None:
                    n_fail += 1
                    continue
                _write_hdf5_frame(h5, int(anchor_idx), payload)
                n_ok += 1
            h5.flush()
    coverage = n_ok / max(n_ok + n_fail, 1)
    accepted = n_ok > 0 and coverage >= SCENE_CACHE_MIN_COVERAGE
    if accepted:
        build_path.replace(out_path)
    else:
        build_path.unlink(missing_ok=True)
        logger.warning(f"{entry['segment_id']}: coverage {coverage:.3f} < {SCENE_CACHE_MIN_COVERAGE} (ok={n_ok}, failed={n_fail}); no cache written")
    return n_ok, n_fail, accepted


def main() -> None:
    args = _parse_args()
    if args.num_shards < 1 or args.shard_idx < 0 or args.shard_idx >= args.num_shards:
        sys.exit(f"invalid sharding: num_shards={args.num_shards}, shard_idx={args.shard_idx}")
    if int(args.scene_cpu_workers) < 0:
        sys.exit(f"--scene-cpu-workers must be >= 0, got {args.scene_cpu_workers}")
    sources = None
    if args.sources is not None:
        sources = {part.strip().lower() for part in str(args.sources).split(",") if part.strip()}

    with open(args.manifest) as f:
        manifest = json.load(f)
    entries = _select_entries(manifest, sources)
    if args.depth == "estimated":
        entries = [e for e in entries if str(e.get("source_dataset", "")).lower() in SENSOR_DEPTH_SOURCES]
    shard = _select_shard(entries, int(args.num_shards), int(args.shard_idx))
    if args.limit and args.limit > 0:
        shard = shard[: int(args.limit)]
    logger.info(f"manifest: {len(entries)} segments with scene media; shard {args.shard_idx}/{args.num_shards}: {len(shard)} segments")
    if not shard:
        return

    scene_cpu_pool: ProcessPoolExecutor | None = None
    if int(args.scene_cpu_workers) > 0:
        # Spawn the workers before CUDA is initialized: they receive only NumPy tasks.
        scene_cpu_pool = ProcessPoolExecutor(
            max_workers=int(args.scene_cpu_workers),
            mp_context=mp.get_context("spawn"),
            initializer=initialize_scene_cpu_worker,
        )
        for future in [scene_cpu_pool.submit(os.getpid) for _ in range(int(args.scene_cpu_workers))]:
            future.result()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    moge_model = None
    # DexYCB and TACO fall back to MoGe where native depth is missing or unusable.
    if args.depth == "estimated" or any(_uses_moge(e) or str(e["source_dataset"]).lower() in {"dexycb", "taco"} for e in shard):
        from moge.model.v2 import MoGeModel

        moge_model = MoGeModel.from_pretrained(MOGE_MODEL_ID).to(device).eval()

    from gtf.datasets.scene.dinov3_semantic import DinoV3Extractor

    dino_extractor = DinoV3Extractor(device=device)
    pca_basis = _load_pca_basis(Path(args.pca_basis))

    n_ok = n_fail = n_skip = n_reject = 0
    for entry in tqdm(shard, desc=f"shard {args.shard_idx}/{args.num_shards}"):
        try:
            cache_path = _cache_path(entry, args.cache_root)
            if not args.force and _scene_cache_hdf5_status(cache_path, _entry_n_frames(entry) or None)[0]:
                n_skip += 1
                continue
            ok_i, fail_i, accepted = _build_segment_cache(entry, moge_model, dino_extractor, pca_basis, args, scene_cpu_pool)
            n_ok += ok_i
            n_fail += fail_i
            n_reject += int(not accepted)
        except Exception as e:
            logger.warning(f"{entry['segment_id']}: {type(e).__name__}: {e}")
            n_reject += 1
            if args.strict:
                raise

    logger.info(f"done: frames ok={n_ok}, frames failed={n_fail}, segments skipped (valid cache)={n_skip}, segments rejected={n_reject}, total={len(shard)}")
    if scene_cpu_pool is not None:
        scene_cpu_pool.shutdown(wait=True, cancel_futures=True)
    if args.strict and n_reject > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
