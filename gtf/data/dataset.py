"""GTF dataset: canonical.npz → model-ready tensors.

Loads segments from a manifest, cuts them into real-time windows, and returns
gravity-aligned, tokenized (but unnormalized, uncentered) data with relation
features, contact labels, per-window confidence features and the cached Uni3D
object embeddings. Normalization and centering happen in the training step.

Pipeline:
  canonical.npz → load fields → gravity_align → tokenize full segment →
  relation/contact preprocessing → cache deterministic segment artifacts →
  slice/pad window (optionally strided) → confidence features → return dict

Usage:
    from gtf.data.dataset import GTFDataset, gtf_collate
    dataset = GTFDataset(split="train", uni3d_cache_path="datasets/uni3d_embeddings.pt")
    loader = DataLoader(dataset, batch_size=32, collate_fn=gtf_collate, num_workers=4)
"""

from collections import OrderedDict
import hashlib
import logging
import os
import time

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from gtf.paths import MANO_DIR
from gtf.data.compute_stats import get_split, median_frame_step
from gtf.data.confidence import TIER_SCORE, compute_confidence_features
from gtf.data.startup_metadata import load_manifest
from gtf.model.repr.canonicalize import gravity_align
from gtf.model.repr.mano import MANOHelper
from gtf.model.repr.mesh import sanitize_mesh_faces
from gtf.model.repr.pseudo_labels import compute_contact_labels
from gtf.model.repr.relation import compute_relation_features
from gtf.model.repr.tokenizer import (
    HAND_CAM_T_SLICE,
    HAND_ORIENT_SLICE,
    HAND_POSE_SLICE,
    OBJ_ROTATION_SLICE,
    OBJ_TRANSLATION_SLICE,
    StateTokenizer,
)

logger = logging.getLogger(__name__)

# Fields needed from canonical.npz — excludes large per-frame vertex arrays
# (hand_*_verts, obj_verts) which are not used for training.
_NEEDED_FIELDS = {
    "hand_left_orient",
    "hand_left_pose",
    "hand_left_cam_t",
    "hand_left_betas",
    "hand_left_valid",
    "hand_right_orient",
    "hand_right_pose",
    "hand_right_cam_t",
    "hand_right_betas",
    "hand_right_valid",
    "obj_rotation",
    "obj_translation",
    "obj_canonical_verts",
    "obj_faces",
    "obj_scale",
    "fps",
    "frame_indices",
}

# Timestamp convention of canonical ``frame_times_s`` (per-row capture times in
# seconds). Action100M canonicals must carry them; MoCap canonicals without them
# get a uniform grid at their effective cadence.
_FRAME_TIME_CONVENTION = "video_best_effort_pts_seconds_v1"

# Optional canonical fields (every other _NEEDED_FIELDS entry is required).
_OPTIONAL_FIELDS = {"gravity_direction", "support_plane_height"}

# Optional fields of moving-camera sources: world-frame states plus per-frame
# camera extrinsics, used to anchor a window to its present camera after the
# observation mask is sampled.
_PRESENT_ANCHOR_FIELDS = {
    "camera_extrinsics",
    "gravity_world",
    "hand_left_orient_world",
    "hand_left_cam_t_world",
    "hand_right_orient_world",
    "hand_right_cam_t_world",
    "obj_rotation_world",
    "obj_translation_world",
}

# Max retries on corrupt npz load
_MAX_RETRIES = 3

_DEFAULT_SEGMENT_CACHE_MAX_BYTES = 32 << 30


def _deterministic_obj_points(segment_entry: dict):
    """One fixed Uni3D input cloud per segment (area-weighted surface samples).

    The generator is seeded with crc32(segment_id), so every process, worker and
    epoch sees the same cloud. This is the input the Uni3D embedding cache is
    built from. ``segment_entry`` needs ``segment_id``, ``obj_canonical_verts``,
    ``obj_vertex_normals`` and ``obj_faces``.
    """
    import zlib

    from gtf.model.arch.conditioning import sample_obj_points

    _gen = torch.Generator().manual_seed(zlib.crc32(str(segment_entry["segment_id"]).encode()) & 0x7FFFFFFF)
    return sample_obj_points(
        segment_entry["obj_canonical_verts"],
        segment_entry["obj_vertex_normals"],
        generator=_gen,
        obj_faces=segment_entry.get("obj_faces"),
    )


def _tensor_tree_nbytes(value) -> int:
    """Estimate tensor payload bytes in a nested structure."""
    if isinstance(value, Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_tree_nbytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_tree_nbytes(v) for v in value)
    return 0


def _load_npz_fields(path: str) -> dict[str, Tensor | float | str]:
    """Load needed fields from canonical.npz, converting to tensors.

    Skips large vertex arrays (hand_*_verts, obj_verts) to save I/O. A missing
    required field raises KeyError (surfaces data corruption).

    Args:
        path: Path to canonical.npz file.

    Returns:
        Dict mapping field name → Tensor (float for fps, str for the time convention).
    """
    npz = np.load(path, allow_pickle=False)
    result: dict[str, Tensor | float] = {}

    for key in _NEEDED_FIELDS:
        if key in npz:
            v = npz[key]
            if v.ndim == 0:
                # Scalar (fps)
                result[key] = float(v)
            else:
                result[key] = torch.from_numpy(v.copy())
        elif key not in _OPTIONAL_FIELDS:
            raise KeyError(f"Required field '{key}' missing from {path}")

    if "frame_times_s" in npz:
        result["frame_times_s"] = torch.from_numpy(np.asarray(npz["frame_times_s"]).copy())
    if "frame_time_convention" in npz:
        result["frame_time_convention"] = str(np.asarray(npz["frame_time_convention"]).item())

    # Load optional fields
    if "gravity_direction" in npz:
        result["gravity_direction"] = torch.from_numpy(npz["gravity_direction"].copy())
    if "support_plane_height" in npz:
        v = npz["support_plane_height"]
        result["support_plane_height"] = float(v) if v.ndim == 0 else float(v.item())
    for key in _PRESENT_ANCHOR_FIELDS:
        if key in npz:
            v = npz[key]
            result[key] = torch.from_numpy(v.copy())

    npz.close()
    return result


def _scene_cache_path_for_entry(entry: dict) -> str | None:
    """Return an existing HDF5 per-frame scene cache path, if any."""
    cache_path = entry.get("scene_cache_path")
    if cache_path and os.path.exists(cache_path):
        return cache_path
    canonical_path = entry.get("canonical_path")
    if canonical_path:
        candidate = os.path.join(os.path.dirname(canonical_path), "scene_cache_v2.h5")
        if os.path.exists(candidate):
            return candidate
    return None


def _read_scene_features_hdf5_row(h5, canonical_idx: int) -> dict[str, Tensor] | None:
    """Read one raw frame row from an open scene cache (None if out of range or invalid)."""
    n_frames = int(h5["P_scene"].shape[0])
    idx = int(canonical_idx)
    if idx < 0 or idx >= n_frames:
        return None
    if "valid" in h5 and not bool(h5["valid"][idx]):
        return None
    scene_features = {
        "P_scene": torch.from_numpy(np.asarray(h5["P_scene"][idx], dtype=np.float32).copy()),
        "scene_normals": torch.from_numpy(np.asarray(h5["scene_normals"][idx], dtype=np.float32).copy()),
        "scene_geo_feats": torch.from_numpy(np.asarray(h5["scene_geo_feats"][idx], dtype=np.float32).copy()),
    }
    if "scene_sem_feats" in h5 and bool(h5.attrs.get("has_semantic", True)):
        scene_features["scene_sem_feats"] = torch.from_numpy(np.asarray(h5["scene_sem_feats"][idx], dtype=np.float32).copy())
    return scene_features


_SCENE_FEATURE_CACHE = None


def clear_scene_feature_cache() -> None:
    """Release this process's bounded read-only H5 handles and raw frame tensors."""
    global _SCENE_FEATURE_CACHE
    if _SCENE_FEATURE_CACHE is not None:
        _SCENE_FEATURE_CACHE.close()
        _SCENE_FEATURE_CACHE = None


def _load_scene_features_hdf5(cache_path: str, canonical_idx: int) -> dict[str, Tensor] | None:
    global _SCENE_FEATURE_CACHE
    if _SCENE_FEATURE_CACHE is None:
        import atexit

        from gtf.data.scene_row_cache import SceneRowCache

        _SCENE_FEATURE_CACHE = SceneRowCache(_read_scene_features_hdf5_row, max_open_files=32, max_bytes=128 * 1024**2)
        atexit.register(clear_scene_feature_cache)
    return _SCENE_FEATURE_CACHE(cache_path, canonical_idx)


def load_present_scene_features(batch: dict, present_frames: Tensor, strict: bool = False) -> list[dict[str, Tensor] | None] | None:
    """Read each sample's scene row at its given (present) frame from the per-frame scene cache.

    Returns None when no sample has a row. With ``strict`` (evaluation) a read
    failure raises instead of leaving the sample scene-free; training logs it and
    continues.
    """
    B = int(present_frames.shape[0])
    result: list[dict[str, Tensor] | None] = [None] * B

    cache_paths = batch.get("scene_cache_path")
    if not cache_paths:
        return None
    canonical_frame_indices = batch.get("canonical_frame_indices")
    if isinstance(canonical_frame_indices, Tensor):
        canonical_frame_indices_cpu = canonical_frame_indices.detach().cpu()
    else:
        canonical_frame_indices_cpu = canonical_frame_indices
    present_cpu = present_frames.detach().cpu()

    for b in range(B):
        local_idx = int(present_cpu[b].item())
        try:
            canonical_idx = int(canonical_frame_indices_cpu[b][local_idx].item()) if canonical_frame_indices_cpu is not None else local_idx
        except Exception:
            canonical_idx = local_idx

        cache_path = cache_paths[b] if b < len(cache_paths) else None
        if cache_path:
            try:
                result[b] = _load_scene_features_hdf5(cache_path, canonical_idx)
            except Exception as e:
                if strict:
                    raise RuntimeError(f"present scene cache {cache_path} frame {canonical_idx} failed to load: {e}") from e
                logger.warning(f"failed to load present scene cache {cache_path} frame {canonical_idx}: {e}")

    return None if all(v is None for v in result) else result


def load_observed_scene_features(
    batch: dict,
    present_frames: Tensor,
    obs_mask: Tensor,
    presence_mask: Tensor,
    valid_frames: Tensor | None = None,
    strict: bool = False,
) -> list[dict[str, Tensor] | None] | None:
    """Scene rows from the last fully-observed frame, rigidly re-expressed in the present frame.

    A per-frame scene row carries d_obj/d_hand distance fields and an
    interaction-centered crop for its own frame. When the present frame has a
    masked entity (hand_to_object, object_to_hand, goal tasks), the present row
    would encode the masked entity's position, so the scene content is taken from
    the last fully-observed frame and transformed into the present gravity-aligned
    camera frame via the stored extrinsics; the states stay present-anchored. When
    the fully-observed frame is the present frame this is a no-op, and
    static-camera samples (no anchor metadata) share one camera frame and load
    untransformed.

    With ``strict`` (evaluation) a transform failure raises; training logs it and
    keeps the untransformed row.
    """
    from gtf.model.repr.present_anchor import scene_frame_transforms, select_fully_observed_frames, transform_scene_row

    f_obs = select_fully_observed_frames(obs_mask, presence_mask, valid_frames)
    rows = load_present_scene_features(batch, f_obs, strict=strict)
    if rows is None:
        return None

    same = (f_obs == present_frames.to(f_obs.device)).cpu()
    if bool(same.all()):
        return rows

    tf_from = scene_frame_transforms(batch, f_obs)
    tf_to = scene_frame_transforms(batch, present_frames)
    if tf_from is None or tf_to is None:
        return rows  # static-camera sources: one shared frame

    has_anchor = batch.get("has_present_anchor")
    if isinstance(has_anchor, Tensor):
        has_anchor = has_anchor.cpu()
    R_f, t_f = tf_from
    R_t, t_t = tf_to
    for b in range(len(rows)):
        if rows[b] is None or bool(same[b]):
            continue
        if has_anchor is not None and not bool(has_anchor[b]):
            continue
        try:
            rows[b] = transform_scene_row(rows[b], R_f[b].cpu(), t_f[b].cpu(), R_t[b].cpu(), t_t[b].cpu())
        except Exception as e:
            # A failed transform leaves the row in the observed (not present) camera frame.
            if strict:
                raise RuntimeError(f"observed-scene transform failed for sample {b}: {e}") from e
            logger.warning(f"observed-scene transform failed for sample {b}: {e}")
    return rows


def _has_present_anchor_fields(data: dict, T: int) -> bool:
    required = _PRESENT_ANCHOR_FIELDS - {"gravity_world"}
    if not all(k in data for k in required):
        return False
    cam = data["camera_extrinsics"]
    if not isinstance(cam, Tensor):
        return False
    if cam.ndim == 2 and tuple(cam.shape) == (4, 4):
        return True
    return cam.ndim == 3 and cam.shape[0] == T and tuple(cam.shape[1:]) == (4, 4)


def _entry_has_safe_scene_features(entry: dict) -> bool:
    """True if the segment has a per-frame scene cache usable for every window.

    The manifest's ``has_complete_scene_sample`` flag is honored only while the
    cache file still exists.
    """
    if not bool(entry.get("has_complete_scene_sample", True)):
        return False
    return _scene_cache_path_for_entry(entry) is not None


def _dataset_aux_cache_dir(manifest_path: str) -> str:
    """Directory for derived dataset-index caches.

    Defaults to the manifest directory; GTF_DATASET_CACHE_DIR redirects
    these disposable caches (e.g. when the manifest directory is read-only).
    """

    override = os.environ.get("GTF_DATASET_CACHE_DIR", "").strip()
    if not override:
        return os.path.dirname(manifest_path)
    cache_dir = os.path.abspath(os.path.expanduser(override))
    os.makedirs(cache_dir, exist_ok=True)
    if not os.path.isdir(cache_dir):
        raise RuntimeError(f"GTF_DATASET_CACHE_DIR is not a directory: {cache_dir}")
    return cache_dir


def _index_cache_identity(manifest_path: str, segments: list[dict]) -> str:
    """Key of the per-segment index caches (hand validity, frame times).

    Cache rows are addressed by segment index, so the key binds the ordered
    segment population (id, canonical path, frame count) and the manifest file's
    size and mtime. Rewriting canonicals in place without touching the manifest
    requires deleting the cache files.
    """
    h = hashlib.sha256()
    for s in segments:
        h.update(f"{s['segment_id']}\t{s.get('canonical_path', '')}\t{s.get('n_frames', '')}\n".encode())
    try:
        st = os.stat(manifest_path)
        h.update(f"{st.st_size}:{st.st_mtime_ns}".encode())
    except OSError:
        h.update(b"<missing>")
    return h.hexdigest()[:16]


def _load_hand_valid_cache(
    manifest_path: str,
    segments: list[dict],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Load per-segment hand_left_valid / hand_right_valid arrays with disk caching.

    The first call scans all canonical.npz files and writes a compact cache in
    the derived-cache directory; later calls (including DDP ranks) read it
    instead of opening every canonical again.

    Returns:
        (seg_left_valid, seg_right_valid): dicts mapping seg_idx → (T,) bool array.
    """
    cache_path = os.path.join(
        _dataset_aux_cache_dir(manifest_path),
        f".hand_valid_cache_v3_{_index_cache_identity(manifest_path, segments)}.npz",
    )

    seg_left_valid: dict[int, np.ndarray] = {}
    seg_right_valid: dict[int, np.ndarray] = {}

    # Try loading cache
    if os.path.exists(cache_path):
        try:
            cache = np.load(cache_path, allow_pickle=False)
            for si in range(len(segments)):
                lk, rk = f"l{si}", f"r{si}"
                if lk in cache and rk in cache:
                    seg_left_valid[si] = cache[lk]
                    seg_right_valid[si] = cache[rk]
            if len(seg_left_valid) == len(segments):
                return seg_left_valid, seg_right_valid
            # Cache incomplete — fall through to rebuild
            seg_left_valid.clear()
            seg_right_valid.clear()
        except Exception:
            pass

    # Scan canonical.npz files. A read failure is retried once and then raised:
    # substituting all-zero validity would bias the hand-balanced sampler.
    for si, ent in enumerate(segments):
        canon_path = ent.get("canonical_path")
        if canon_path:
            last_err = None
            for _attempt in range(2):
                try:
                    npz = np.load(canon_path, allow_pickle=False)
                    seg_left_valid[si] = npz["hand_left_valid"].astype(bool) if "hand_left_valid" in npz else np.zeros(ent["n_frames"], dtype=bool)
                    seg_right_valid[si] = npz["hand_right_valid"].astype(bool) if "hand_right_valid" in npz else np.zeros(ent["n_frames"], dtype=bool)
                    npz.close()
                    last_err = None
                    break
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    time.sleep(0.5)
            if last_err is not None:
                raise RuntimeError(f"hand-validity scan: canonical unreadable after retry: {canon_path} ({last_err})") from last_err
        else:
            seg_left_valid[si] = np.zeros(ent["n_frames"], dtype=bool)
            seg_right_valid[si] = np.zeros(ent["n_frames"], dtype=bool)

    # Write cache (atomic via tmp + rename; concurrent writers produce identical content)
    try:
        save_dict = {}
        for si in range(len(segments)):
            save_dict[f"l{si}"] = seg_left_valid[si]
            save_dict[f"r{si}"] = seg_right_valid[si]
        tmp_path = cache_path + f".tmp.{os.getpid()}"
        np.savez_compressed(tmp_path, **save_dict)
        # np.savez_compressed appends .npz if not already present
        tmp_actual = tmp_path if tmp_path.endswith(".npz") else tmp_path + ".npz"
        os.replace(tmp_actual, cache_path)
    except Exception:
        pass  # Non-fatal; next run will re-scan

    return seg_left_valid, seg_right_valid


def _validate_frame_times(frame_times, *, n_frames: int, context: str) -> np.ndarray:
    """Validate one canonical's row times and return float64 seconds."""
    times = np.asarray(frame_times, dtype=np.float64).reshape(-1)
    if times.shape[0] != int(n_frames):
        raise ValueError(f"{context}: frame_times_s length {times.shape[0]} != n_frames {n_frames}")
    if times.size == 0:
        raise ValueError(f"{context}: frame_times_s is empty")
    if not np.isfinite(times).all():
        raise ValueError(f"{context}: frame_times_s contains NaN/Inf")
    if times.size > 1 and not np.all(np.diff(times) > 0.0):
        raise ValueError(f"{context}: frame_times_s must be strictly increasing")
    return times


def _load_frame_times_cache(manifest_path: str, segments: list[dict]) -> dict[int, np.ndarray]:
    """Load canonical row times for timestamp-defined window construction.

    Canonical files are scanned once into one concatenated ragged cache. An
    Action100M canonical without valid ``frame_times_s`` is a hard error;
    MoCap canonicals without timestamps get a uniform grid at their effective
    cadence (stored fps / median frame step).
    """
    cache_path = os.path.join(
        _dataset_aux_cache_dir(manifest_path),
        f".frame_times_cache_v2_{_index_cache_identity(manifest_path, segments)}.npz",
    )
    if os.path.exists(cache_path):
        try:
            cache = np.load(cache_path, allow_pickle=False)
            values = np.asarray(cache["values"], dtype=np.float64)
            offsets = np.asarray(cache["offsets"], dtype=np.int64)
            if offsets.shape == (len(segments) + 1,) and offsets[0] == 0 and offsets[-1] == values.shape[0] and np.all(offsets[1:] >= offsets[:-1]):
                out = {}
                for si, entry in enumerate(segments):
                    out[si] = _validate_frame_times(
                        values[offsets[si] : offsets[si + 1]],
                        n_frames=int(entry["n_frames"]),
                        context=f"{entry['segment_id']} (timing cache)",
                    )
                cache.close()
                return out
            cache.close()
        except Exception:
            # A partial/corrupt cache is never authoritative; rescan canonicals.
            pass

    arrays: list[np.ndarray] = []
    offsets = [0]
    for entry in segments:
        path = entry.get("canonical_path")
        if not path:
            raise ValueError(f"{entry['segment_id']}: frame-time windowing requires canonical_path")
        last_err = None
        for _attempt in range(2):
            try:
                with np.load(path, allow_pickle=False) as npz:
                    convention = str(np.asarray(npz["frame_time_convention"]).item()) if "frame_time_convention" in npz else None
                    if convention is not None and convention != _FRAME_TIME_CONVENTION:
                        raise ValueError(f"unknown frame_time_convention={convention!r}")
                    if "frame_times_s" in npz:
                        if convention is None and str(entry.get("source_dataset", "")).lower() == "action100m":
                            raise KeyError("Action100M canonical is missing frame_time_convention")
                        times = _validate_frame_times(
                            npz["frame_times_s"],
                            n_frames=int(entry["n_frames"]),
                            context=f"{entry['segment_id']} ({path})",
                        )
                    elif str(entry.get("source_dataset", "")).lower() == "action100m":
                        raise KeyError("Action100M canonical is missing frame_times_s")
                    elif convention is not None:
                        raise KeyError("frame_time_convention is present but frame_times_s is missing")
                    else:
                        # MoCap canonical without timestamps: uniform grid at its
                        # effective cadence.
                        n_frames = int(entry["n_frames"])
                        video_fps = float(npz["fps"]) if "fps" in npz else 30.0
                        step = median_frame_step(np.asarray(npz["frame_indices"])) if "frame_indices" in npz else 1.0
                        effective_fps = video_fps / max(step, 1e-6)
                        times = np.arange(n_frames, dtype=np.float64) / effective_fps
                arrays.append(times)
                offsets.append(offsets[-1] + times.shape[0])
                last_err = None
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(0.5)
        if last_err is not None:
            raise RuntimeError(f"frame-time scan: canonical unreadable/invalid after retry: {path} ({last_err})") from last_err

    values = np.concatenate(arrays) if arrays else np.empty((0,), dtype=np.float64)
    offsets_np = np.asarray(offsets, dtype=np.int64)
    try:
        tmp_path = cache_path + f".tmp.{os.getpid()}.npz"
        np.savez_compressed(tmp_path, values=values, offsets=offsets_np)
        os.replace(tmp_path, cache_path)
    except Exception:
        pass
    return {si: values[offsets_np[si] : offsets_np[si + 1]] for si in range(len(segments))}


def _timestamp_windows(
    frame_times_s,
    *,
    window_seconds: float,
    stride_seconds: float,
    max_frames: int,
) -> tuple[list[tuple[int, int]], int]:
    """Construct half-open row windows from timestamps.

    Regular windows begin at the first row at/after each real-time stride mark
    and include rows in ``[mark, mark + window_seconds)``. A final trailing
    window is appended when the stride grid does not end flush with the segment.
    ``max_frames`` is the fixed tensor length T; the returned count reports
    real-time windows truncated by that cap.
    """
    times = _validate_frame_times(frame_times_s, n_frames=len(frame_times_s), context="timestamp window construction")
    n = int(times.shape[0])
    if n == 1:
        return [(0, 1)], 0

    edge = np.diff(times)
    nominal_dt = float(np.median(edge))
    segment_end = float(times[-1] + nominal_dt)  # exclusive end of the final row
    available_s = segment_end - float(times[0])
    if available_s <= window_seconds:
        if n > max_frames:
            out = [(0, max_frames)]
            trailing = (n - max_frames, n)
            if trailing != out[-1]:
                out.append(trailing)
            return out, len(out)
        return [(0, n)], 0

    n_regular = int(np.floor((available_s - window_seconds) / stride_seconds + 1e-9)) + 1
    windows: list[tuple[int, int]] = []
    n_clamped = 0

    def _left_boundary_index(mark: float) -> int:
        # Boundary marks add/subtract decimal durations to binary timestamps.
        # Treat only roundoff-scale differences as equality so a uniform
        # sequence cannot lose its boundary row (e.g. 587 rows at 30 Hz must
        # start the trailing window at 437, not 438). Real timestamp
        # differences are orders of magnitude larger than this tolerance.
        scale = max(
            1.0,
            abs(float(mark)),
            abs(float(times[0])),
            abs(segment_end),
        )
        tolerance = 16.0 * np.finfo(np.float64).eps * scale
        return int(np.searchsorted(times, float(mark) - tolerance, side="left"))

    def _append(start: int, end: int) -> None:
        nonlocal n_clamped
        start = min(max(int(start), 0), n - 1)
        end = min(max(int(end), start + 1), n)
        if end - start > max_frames:
            end = start + max_frames
            n_clamped += 1
        item = (start, end)
        if not windows or windows[-1] != item:
            windows.append(item)

    t0 = float(times[0])
    for k in range(n_regular):
        mark = t0 + k * stride_seconds
        start = _left_boundary_index(mark)
        end = _left_boundary_index(mark + window_seconds)
        _append(start, end)

    # The last row represents [times[-1], segment_end). Choose the first row in
    # the final window's real-time interval (for a uniform cadence this is
    # ``n - round(window_seconds * fps)``).
    trailing_mark = segment_end - window_seconds
    trailing_start = _left_boundary_index(trailing_mark)
    if n - trailing_start > max_frames:
        trailing_start = n - max_frames
    _append(trailing_start, n)
    return windows, n_clamped


class GTFDataset(Dataset):
    """PyTorch Dataset connecting canonical.npz files to model-ready tensors.

    Returns gravity-aligned, tokenized data with relation features, contact labels,
    confidence features and (when a cache is attached) the Uni3D object
    embeddings. Normalization and centering happen in the training step.

    Segments without usable scene features are skipped. Windows are defined in
    real time from each canonical's frame timestamps and padded to the fixed
    length ``window_size``.

    Args:
        manifest_path: Path to canonical_manifest.json.
        window_size: Fixed window length T in frames (a real-time window longer
            than T frames is truncated).
        split: One of "train", "val", "test". Ignored when segment_ids is provided.
        mano_root: Path to MANO model files.
        segment_ids: If provided, restrict to these segment IDs (bypasses split filter).
        segment_cache_max_bytes: Byte cap for the per-process LRU cache of deterministic
            segment artifacts. Set to 0 to disable.
        aug_stride_prob: Train split only: probability of re-slicing a window at a
            random temporal stride (same real-time extent, coarser cadence).
        aug_stride_target_hz: The largest stride brings a source down to about this rate.
        aug_stride_max: Upper bound on the stride.
        window_seconds: Window length in seconds.
        stride_seconds: Step between window starts in seconds.
        uni3d_cache_path: Uni3D embedding cache (gtf/data/uni3d_cache.py). Without
            it, samples carry no object embeddings.
    """

    def __init__(
        self,
        manifest_path: str = "datasets/canonical_manifest.json",
        window_size: int = 150,
        split: str = "train",
        mano_root: str = MANO_DIR,
        segment_ids: list[str] | None = None,
        segment_cache_max_bytes: int = _DEFAULT_SEGMENT_CACHE_MAX_BYTES,
        aug_stride_prob: float = 0.0,
        aug_stride_target_hz: float = 10.0,
        aug_stride_max: int = 6,
        window_seconds: float = 5.0,
        stride_seconds: float = 2.5,
        uni3d_cache_path: str | None = None,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.window_seconds = float(window_seconds)
        self.stride_seconds = float(stride_seconds)
        if self.window_seconds <= 0.0 or self.stride_seconds <= 0.0:
            raise ValueError(f"window_seconds and stride_seconds must be positive, got {self.window_seconds} and {self.stride_seconds}")
        self.split = split
        self.segment_cache_max_bytes = max(0, int(segment_cache_max_bytes))
        self.uni3d_cache_path = str(uni3d_cache_path or "")
        self._uni3d_cache_lookup = None
        # Stride augmentation (train only): with prob, re-slice the window at a random
        # temporal stride s ∈ {1..max_stride} — the same motion at a coarser cadence —
        # so dt conditioning learns physics rather than source identity.
        self.aug_stride_prob = float(aug_stride_prob)
        self.aug_stride_target_hz = float(aug_stride_target_hz)
        self.aug_stride_max = max(1, int(aug_stride_max))
        # Number of training windows replaced by a neighbor after a failed load.
        self.load_substitutions = 0

        # Load and filter manifest
        manifest = load_manifest(manifest_path)

        self.segments: list[dict] = []
        self.windows: list[tuple[int, int, int]] = []  # (segment_idx, start, end)
        segment_ids_set = set(segment_ids) if segment_ids is not None else None

        n_skipped_no_scene = 0
        for entry in manifest["entries"]:
            if not entry.get("validated", True):
                continue
            # When segment_ids is provided, use explicit filter; otherwise use split
            if segment_ids_set is not None:
                if entry["segment_id"] not in segment_ids_set:
                    continue
            elif get_split(entry) != split:
                continue
            if not _entry_has_safe_scene_features(entry):
                n_skipped_no_scene += 1
                continue
            self.segments.append(entry)

        # Real-time windows from the canonical frame timestamps, capped at the
        # fixed tensor length window_size.
        n_clamped = 0
        frame_times = _load_frame_times_cache(manifest_path, self.segments)
        for seg_idx in range(len(self.segments)):
            seg_windows, n_seg_clamped = _timestamp_windows(
                frame_times[seg_idx],
                window_seconds=self.window_seconds,
                stride_seconds=self.stride_seconds,
                max_frames=window_size,
            )
            self.windows.extend((seg_idx, start, end) for start, end in seg_windows)
            n_clamped += n_seg_clamped
        if n_clamped:
            logger.warning(
                f"GTFDataset({split}): {n_clamped} real-time windows exceed fixed window_size={window_size} "
                f"and are truncated (< {self.window_seconds:g}s of real time)."
            )

        # Per-window left/right valid frame counts for the hand-balanced sampler.
        seg_left_valid, seg_right_valid = _load_hand_valid_cache(manifest_path, self.segments)
        self._window_left_valid_count: list[int] = []
        self._window_right_valid_count: list[int] = []
        for si, start, end in self.windows:
            lv = seg_left_valid.get(si)
            rv = seg_right_valid.get(si)
            self._window_left_valid_count.append(int(lv[start:end].sum()) if lv is not None else 0)
            self._window_right_valid_count.append(int(rv[start:end].sum()) if rv is not None else 0)

        # Shared objects
        self.tokenizer = StateTokenizer()
        self.mano_helper = MANOHelper(mano_root=mano_root)
        self.mano_helper.eval()
        for p in self.mano_helper.parameters():
            p.requires_grad = False

        # Process-local LRU cache of deterministic segment artifacts.
        self._segment_cache: OrderedDict[int, tuple[dict, int]] = OrderedDict()
        self._segment_cache_bytes = 0

        # Validate the embedding cache once (it must cover every selected
        # segment) before any worker serves a window.
        if self.uni3d_cache_path:
            from gtf.data.uni3d_cache import load_uni3d_cache

            self._uni3d_cache_lookup, _cache_meta, _cache_digest = load_uni3d_cache(
                self.uni3d_cache_path,
                required_segment_ids=[str(entry["segment_id"]) for entry in self.segments],
            )

        logger.info(f"GTFDataset({split}): {len(self.segments)} segments, {len(self.windows)} windows ({n_skipped_no_scene} segments without scene features skipped)")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        """Load one windowed sample.

        Deterministic per-segment preprocessing is cached by segment index within
        the current worker process; window slicing, the stride draw and the
        confidence features are per access. On the train split an unreadable
        canonical is replaced by the next window (counted in
        ``load_substitutions``); on val/test it raises so the evaluation cohort
        stays exact.
        """
        for attempt in range(_MAX_RETRIES):
            actual_idx = (idx + attempt) % len(self)
            seg_idx, start, end = self.windows[actual_idx]
            manifest_entry = self.segments[seg_idx]

            segment_entry = self._cache_get(seg_idx)
            if segment_entry is None:
                try:
                    data = _load_npz_fields(manifest_entry["canonical_path"])
                except Exception as e:
                    logger.warning(f"Failed to load {manifest_entry['canonical_path']}: {e}")
                    self.load_substitutions += 1
                    if self.split != "train":
                        raise RuntimeError(f"split={self.split}: failed to load {manifest_entry['canonical_path']}: {e}") from e
                    if attempt == _MAX_RETRIES - 1:
                        raise RuntimeError(f"Failed to load NPZ for index {idx} after {_MAX_RETRIES} attempts") from e
                    continue

                # Only the file read is retried; a processing error raises.
                segment_entry = self._build_segment_entry(data, manifest_entry)
                self._cache_put(seg_idx, segment_entry)

            stride = self._draw_stride(segment_entry)
            return self._build_window_sample(segment_entry, start, end, stride=stride)

        raise RuntimeError(f"Failed to load NPZ for index {idx} after {_MAX_RETRIES} attempts")

    def _cache_get(self, seg_idx: int) -> dict | None:
        if self.segment_cache_max_bytes <= 0:
            return None
        cached = self._segment_cache.pop(seg_idx, None)
        if cached is None:
            return None
        self._segment_cache[seg_idx] = cached
        return cached[0]

    def _cache_put(self, seg_idx: int, segment_entry: dict) -> None:
        if self.segment_cache_max_bytes <= 0:
            return

        entry_bytes = _tensor_tree_nbytes(segment_entry)
        if entry_bytes > self.segment_cache_max_bytes:
            return

        if seg_idx in self._segment_cache:
            _old_entry, old_bytes = self._segment_cache.pop(seg_idx)
            self._segment_cache_bytes -= old_bytes

        while self._segment_cache and self._segment_cache_bytes + entry_bytes > self.segment_cache_max_bytes:
            _evicted_key, (_evicted_entry, evicted_bytes) = self._segment_cache.popitem(last=False)
            self._segment_cache_bytes -= evicted_bytes

        if self._segment_cache_bytes + entry_bytes > self.segment_cache_max_bytes:
            return

        self._segment_cache[seg_idx] = (segment_entry, entry_bytes)
        self._segment_cache_bytes += entry_bytes

    def _build_segment_entry(self, data: dict, manifest_entry: dict) -> dict:
        """Build deterministic full-segment artifacts for cache/storage."""
        raw_data = data
        has_gravity = manifest_entry.get("has_gravity", False)
        gravity_dir = data.get("gravity_direction") if has_gravity else None
        if gravity_dir is not None:
            gravity_dir = gravity_dir.float()
        data = gravity_align(data, gravity_dir)

        tokens = self.tokenizer.encode(data)
        presence_mask = tokens["presence_mask"]
        T_seg = presence_mask.shape[0]

        has_present_anchor = _has_present_anchor_fields(raw_data, T_seg)
        if has_present_anchor:
            camera_extrinsics = raw_data["camera_extrinsics"].float()
            if camera_extrinsics.ndim == 2:
                camera_extrinsics = camera_extrinsics.view(1, 4, 4).expand(T_seg, 4, 4).clone()
            world_data = dict(raw_data)
            for key in ("hand_left_orient", "hand_left_cam_t", "hand_right_orient", "hand_right_cam_t", "obj_rotation", "obj_translation"):
                world_data[key] = raw_data[f"{key}_world"]
            world_tokens = self.tokenizer.encode(world_data)
            gravity_world = raw_data.get("gravity_world")
            if isinstance(gravity_world, Tensor):
                gravity_world = gravity_world.float()
            else:
                gravity_world = torch.zeros(3, dtype=tokens["obj_state"].dtype)
        else:
            camera_extrinsics = torch.eye(4, dtype=tokens["obj_state"].dtype).view(1, 4, 4).expand(T_seg, 4, 4).clone()
            world_tokens = {
                "left_state": tokens["left_state"],
                "right_state": tokens["right_state"],
                "obj_state": tokens["obj_state"],
            }
            gravity_world = torch.zeros(3, dtype=tokens["obj_state"].dtype)

        obj_canonical_verts_raw = data["obj_canonical_verts"] if isinstance(data["obj_canonical_verts"], Tensor) else torch.from_numpy(data["obj_canonical_verts"])
        obj_faces_raw = data["obj_faces"] if isinstance(data["obj_faces"], Tensor) else torch.from_numpy(data["obj_faces"])
        obj_faces = sanitize_mesh_faces(
            obj_canonical_verts_raw,
            obj_faces_raw,
            context=f"{manifest_entry['segment_id']} object mesh",
        )
        obj_canonical_verts = obj_canonical_verts_raw.float()
        obj_scale = data["obj_scale"].float()
        obj_rot_6d = tokens["obj_state"][:, OBJ_ROTATION_SLICE]
        obj_trans = tokens["obj_state"][:, OBJ_TRANSLATION_SLICE]

        for side, state_slot, rel_key in [("left", 0, "left_rel"), ("right", 1, "right_rel")]:
            hand_valid = presence_mask[:, state_slot]
            if hand_valid.any():
                orient_6d = tokens[f"{side}_state"][:, HAND_ORIENT_SLICE]
                cam_t = tokens[f"{side}_state"][:, HAND_CAM_T_SLICE]
                pose_6d = tokens[f"{side}_state"][:, HAND_POSE_SLICE]
                betas = data[f"hand_{side}_betas"].float() if isinstance(data[f"hand_{side}_betas"], Tensor) else torch.zeros(10)

                with torch.no_grad():
                    rel = compute_relation_features(
                        orient_6d,
                        cam_t,
                        pose_6d,
                        betas,
                        side,
                        obj_rot_6d,
                        obj_trans,
                        obj_canonical_verts,
                        obj_scale,
                        self.mano_helper,
                        obj_faces=obj_faces,
                    )
                rel[~hand_valid] = 0.0
                tokens[rel_key] = rel
            else:
                tokens[rel_key] = torch.zeros(T_seg, 14, dtype=tokens["obj_state"].dtype)

        fingertip_dists = torch.stack([tokens["left_rel"][:, 9:], tokens["right_rel"][:, 9:]], dim=1)
        hand_valid_2 = torch.stack([presence_mask[:, 0], presence_mask[:, 1]], dim=1)
        contact_labels = compute_contact_labels(fingertip_dists, hand_valid=hand_valid_2)

        betas_left = data["hand_left_betas"].float() if isinstance(data.get("hand_left_betas"), Tensor) else torch.zeros(10)
        betas_right = data["hand_right_betas"].float() if isinstance(data.get("hand_right_betas"), Tensor) else torch.zeros(10)

        fps_val = data.get("fps", 30.0)
        if isinstance(fps_val, Tensor):
            fps_val = fps_val.item()
        else:
            fps_val = float(fps_val)
        # Effective cadence: the stored fps is the video rate; keyframe-subsampled
        # canonicals (Action100M stride ~3, OakInk2 stride 4) tick slower. The
        # stride augmentation and timestamp-less MoCap canonicals use this rate.
        video_fps_val = fps_val
        fps_val = video_fps_val / median_frame_step(data["frame_indices"])

        _source_is_a100m = str(manifest_entry.get("source_dataset", "")).lower() == "action100m"
        _time_convention = raw_data.get("frame_time_convention")
        if _time_convention is not None and _time_convention != _FRAME_TIME_CONVENTION:
            raise ValueError(f"{manifest_entry['segment_id']}: unknown frame_time_convention={_time_convention!r}")
        if _source_is_a100m and _time_convention is None:
            raise KeyError(f"{manifest_entry['segment_id']}: Action100M canonical requires frame_time_convention")
        if "frame_times_s" not in raw_data and (_source_is_a100m or _time_convention is not None):
            raise KeyError(f"{manifest_entry['segment_id']}: frame_time_convention requires canonical frame_times_s")
        if "frame_times_s" in raw_data:
            _times_np = _validate_frame_times(
                raw_data["frame_times_s"].detach().cpu().numpy(),
                n_frames=T_seg,
                context=f"{manifest_entry['segment_id']} canonical",
            )
        else:
            # MoCap canonical without timestamps: uniform grid at the effective cadence.
            _times_np = np.arange(T_seg, dtype=np.float64) / fps_val
        # Keep absolute timestamps in float64 while the segment is cached: video
        # timestamps can have a large origin, and casting before subtracting the
        # window origin can erase millisecond-scale edge differences.
        frame_times_s = torch.from_numpy(_times_np.copy())

        # Support plane: transform to gravity-aligned Y coordinate
        # In gravity-aligned frame (+Y=down): plane_y = -support_plane_height
        has_support_plane = manifest_entry.get("has_support_plane", False)
        if has_support_plane and "support_plane_height" in data and has_gravity:
            support_plane_y = -float(data["support_plane_height"])
        else:
            support_plane_y = None

        # Scene rows are read from the per-frame cache at the anchor frame in the
        # training/evaluation step.
        scene_cache_path = _scene_cache_path_for_entry(manifest_entry)

        # Constant confidence feature 6: tier score, down-weighted by the manifest's
        # interaction-fidelity score (absent → 1.0).
        tier = str(manifest_entry.get("confidence_tier", "extracted"))
        extractor_score = TIER_SCORE.get(tier, 0.5) * float(manifest_entry.get("interaction_fidelity_score", 1.0))

        entry = {
            "left_state": tokens["left_state"],
            "right_state": tokens["right_state"],
            "obj_state": tokens["obj_state"],
            "left_state_world": world_tokens["left_state"],
            "right_state_world": world_tokens["right_state"],
            "obj_state_world": world_tokens["obj_state"],
            "camera_extrinsics": camera_extrinsics,
            "gravity_world": gravity_world,
            "has_present_anchor": torch.tensor(bool(has_present_anchor), dtype=torch.bool),
            "left_rel": tokens["left_rel"],
            "right_rel": tokens["right_rel"],
            "presence_mask": presence_mask,
            "contact_labels": contact_labels,
            "betas_left": betas_left,
            "betas_right": betas_right,
            "obj_canonical_verts": obj_canonical_verts,
            "obj_faces": obj_faces,
            "obj_scale": obj_scale,
            "support_plane_y": support_plane_y,
            "_confidence_extractor_score": float(extractor_score),
            "scene_cache_path": scene_cache_path,
            "fps": fps_val,
            "video_fps": video_fps_val,
            "frame_indices": data["frame_indices"].long(),
            "frame_times_s": frame_times_s,
            "segment_id": manifest_entry["segment_id"],
            "source_dataset": str(manifest_entry.get("source_dataset", "unknown")),
        }
        return entry

    def _draw_stride(self, segment_entry: dict) -> int:
        """Draw a temporal stride for one window (stride augmentation; 1 = no-op).

        Only fires on the train split with aug_stride_prob > 0. max_stride derives from
        the segment's effective cadence, so 7.5/10 Hz sources never stride (max 1)
        while 30/60 Hz sources reach about target_hz. Uses the per-worker torch RNG.
        """
        if self.aug_stride_prob <= 0.0 or self.split != "train":
            return 1
        max_stride = min(self.aug_stride_max, max(1, int(round(segment_entry["fps"] / self.aug_stride_target_hz))))
        if max_stride <= 1:
            return 1
        if torch.rand(()).item() >= self.aug_stride_prob:
            return 1
        return int(torch.randint(1, max_stride + 1, ()).item())

    def _slice_time_tensor(self, tensor: Tensor, start: int, end: int, pad_value: float = 0.0) -> Tensor:
        """Slice a temporal tensor and pad to window_size with ``pad_value`` when needed."""
        window = tensor[start:end]
        pad = self.window_size - (end - start)
        if pad <= 0:
            return window
        if pad_value == 0.0:
            pad_tensor = torch.zeros((pad, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
        else:
            pad_tensor = torch.full((pad, *tensor.shape[1:]), pad_value, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([window, pad_tensor], dim=0)

    def _slice_time_strided(self, tensor: Tensor, start: int, end: int, stride: int, pad_value: float = 0.0) -> Tensor:
        """Every ``stride``-th frame of ``[start, end)``, padded to window_size like ``_slice_time_tensor``."""
        window = tensor[start:end:stride]
        pad = self.window_size - window.shape[0]
        if pad <= 0:
            return window
        if pad_value == 0.0:
            pad_tensor = torch.zeros((pad, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
        else:
            pad_tensor = torch.full((pad, *tensor.shape[1:]), pad_value, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([window, pad_tensor], dim=0)

    def _build_window_sample(self, segment_entry: dict, start: int, end: int, stride: int = 1) -> dict:
        """Slice a cached segment entry into a model-ready fixed-length window.

        ``stride`` > 1 (train-time stride augmentation) takes every stride-th frame
        of the window's real-time extent — the same interaction at a coarser
        cadence; the frame times, edge dt and ``fps`` follow the selected frames.
        """
        T = self.window_size
        if stride == 1:
            T_real = end - start

            def _slice(x: Tensor, pad_value: float = 0.0) -> Tensor:
                return self._slice_time_tensor(x, start, end, pad_value)
        else:
            n_seg = int(segment_entry["frame_indices"].shape[0])
            end_s = min(start + T * stride, min(end, n_seg))
            T_real = (end_s - start + stride - 1) // stride

            def _slice(x: Tensor, pad_value: float = 0.0) -> Tensor:
                return self._slice_time_strided(x, start, end_s, stride, pad_value)

        valid_frames = torch.ones(T, dtype=torch.bool)
        if T_real < T:
            valid_frames[T_real:] = False

        left_state = _slice(segment_entry["left_state"])
        right_state = _slice(segment_entry["right_state"])
        obj_state = _slice(segment_entry["obj_state"])
        left_state_world = _slice(segment_entry["left_state_world"])
        right_state_world = _slice(segment_entry["right_state_world"])
        obj_state_world = _slice(segment_entry["obj_state_world"])
        camera_extrinsics = _slice(segment_entry["camera_extrinsics"])
        frame_indices = _slice(segment_entry["frame_indices"], pad_value=-1)
        if stride == 1:
            canonical_frame_indices = torch.arange(start, end, dtype=torch.long)
        else:
            canonical_frame_indices = torch.arange(start, start + T_real * stride, stride, dtype=torch.long)
        if T_real < T:
            canonical_frame_indices = torch.cat([canonical_frame_indices, torch.full((T - T_real,), -1, dtype=torch.long)])

        # Frame times relative to the window start (float64) and per-transition
        # dt (float32). Padded rows repeat the last time / the nominal dt.
        sample_fps = segment_entry["fps"] if stride == 1 else segment_entry["fps"] / stride
        source_times = segment_entry["frame_times_s"]
        selected_idx = canonical_frame_indices[:T_real]
        selected_times = source_times.index_select(0, selected_idx).to(dtype=torch.float64)
        relative_times = selected_times - selected_times[0]
        if T_real > 1:
            valid_edge_dt = selected_times[1:] - selected_times[:-1]
            if not bool(torch.isfinite(valid_edge_dt).all()) or not bool((valid_edge_dt > 0).all()):
                raise ValueError(f"{segment_entry['segment_id']}: sliced frame_times_s produced nonpositive/nonfinite edge dt")
            nominal_dt = valid_edge_dt.median()
            # Representative cadence of the window, from its timestamps.
            sample_fps = 1.0 / float(nominal_dt)
        else:
            nominal_dt = torch.tensor(1.0 / max(float(sample_fps), 1e-6), dtype=source_times.dtype)
            valid_edge_dt = source_times.new_empty((0,))

        # Relative times stay float64: in float32 a 5 s window can move ±0.1 s
        # confidence boundaries by one row at 10 Hz.
        frame_times_s = torch.full((T,), float(relative_times[-1]), dtype=relative_times.dtype)
        frame_times_s[:T_real] = relative_times
        edge_dt_s = torch.full((T - 1,), float(nominal_dt), dtype=torch.float32)
        if valid_edge_dt.numel():
            edge_dt_s[: valid_edge_dt.shape[0]] = valid_edge_dt.float()

        left_rel = _slice(segment_entry["left_rel"])
        right_rel = _slice(segment_entry["right_rel"])
        presence_mask = _slice(segment_entry["presence_mask"])
        contact_labels = _slice(segment_entry["contact_labels"])

        betas_left = segment_entry["betas_left"]
        betas_right = segment_entry["betas_right"]
        if not presence_mask[:, 0].any():
            betas_left = torch.zeros_like(betas_left)
        if not presence_mask[:, 1].any():
            betas_right = torch.zeros_like(betas_right)

        # Confidence features from the window alone (no ground truth outside the
        # window), at the window's own cadence. Padded frames get the neutral 0.5.
        confidence_window = compute_confidence_features(
            tokens={"left_state": left_state[:T_real], "right_state": right_state[:T_real], "obj_state": obj_state[:T_real]},
            presence_mask=presence_mask[:T_real],
            relation_features={"left_rel": left_rel[:T_real], "right_rel": right_rel[:T_real]},
            extractor_score=float(segment_entry["_confidence_extractor_score"]),
            frame_times_s=frame_times_s[:T_real],
        )
        if T_real < T:
            confidence_window = torch.cat([confidence_window, torch.full((T - T_real, 5, 7), 0.5, dtype=confidence_window.dtype)], dim=0)

        sample = {
            # Tokens (unnormalized, uncentered)
            "left_state": left_state,  # (T, 99)
            "right_state": right_state,  # (T, 99)
            "obj_state": obj_state,  # (T, 9)
            "left_state_world": left_state_world,  # (T, 99)
            "right_state_world": right_state_world,  # (T, 99)
            "obj_state_world": obj_state_world,  # (T, 9)
            "camera_extrinsics": camera_extrinsics,  # (T, 4, 4)
            "gravity_world": segment_entry["gravity_world"],  # (3,)
            "has_present_anchor": segment_entry["has_present_anchor"],  # bool scalar
            "left_rel": left_rel,  # (T, 14)
            "right_rel": right_rel,  # (T, 14)
            # Masks
            "presence_mask": presence_mask,  # (T, 5) bool
            "valid_frames": valid_frames,  # (T,) bool
            # Labels
            "contact_labels": contact_labels,  # (T, 2, 5) bool
            # Conditioning
            "betas_left": betas_left,  # (10,)
            "betas_right": betas_right,  # (10,)
            "obj_canonical_verts": segment_entry["obj_canonical_verts"],  # (V, 3)
            "obj_faces": segment_entry["obj_faces"],  # (F, 3) int64
            "obj_scale": segment_entry["obj_scale"],  # (3,)
            # Scene
            "support_plane_y": segment_entry["support_plane_y"],  # float or None
            "confidence_features": confidence_window,  # (T, 5, 7)
            "scene_cache_path": segment_entry.get("scene_cache_path"),
            # Meta
            "fps": sample_fps,
            "video_fps": segment_entry.get("video_fps", segment_entry["fps"]),
            "frame_indices": frame_indices,
            "canonical_frame_indices": canonical_frame_indices,
            "frame_times_s": frame_times_s,  # (T,) float64, relative to the window start
            "edge_dt_s": edge_dt_s,  # (T-1,) float32
            "segment_id": segment_entry["segment_id"],
            "source_dataset": segment_entry.get("source_dataset", "unknown"),
        }
        _uni3d_lookup = self._uni3d_cache_lookup
        if _uni3d_lookup is not None:
            obj_embedding_bf16, obj_embedding_fp32 = _uni3d_lookup[segment_entry["segment_id"]]
            sample["obj_embedding_bf16"] = obj_embedding_bf16  # (1024,) bf16-autocast output (training)
            sample["obj_embedding_fp32"] = obj_embedding_fp32  # (1024,) fp32 output (AMP-off evaluation)
        return sample


# Collation. Fixed-size tensor keys — torch.stack to (B, ...)
_STACKABLE = {
    "left_state",
    "right_state",
    "obj_state",
    "left_state_world",
    "right_state_world",
    "obj_state_world",
    "camera_extrinsics",
    "gravity_world",
    "has_present_anchor",
    "frame_indices",
    "canonical_frame_indices",
    "frame_times_s",
    "edge_dt_s",
    "left_rel",
    "right_rel",
    "presence_mask",
    "valid_frames",
    "contact_labels",
    "betas_left",
    "betas_right",
    "obj_scale",
    "confidence_features",
    "obj_embedding_bf16",
    "obj_embedding_fp32",
}

# Variable-size tensor keys — keep as list[Tensor]
_LIST_TENSORS = {"obj_canonical_verts", "obj_faces"}

# Metadata keys — keep as list
_META = {"fps", "video_fps", "segment_id", "source_dataset", "scene_cache_path"}


def gtf_collate(batch: list[dict]) -> dict:
    """Custom collate for GTFDataset.

    Stacks fixed-size tensors along batch dimension (B, T, D).
    Variable-size tensors (obj_canonical_verts, obj_faces) kept as lists.
    Metadata (fps, segment_id) kept as lists.
    Optional per-sample scalars (support_plane_y) get tensor + availability mask.

    Args:
        batch: List of dicts from GTFDataset.__getitem__.

    Returns:
        Collated batch dict with (B, ...) tensors and lists.
    """
    result = {}
    for key in batch[0]:
        if key in _STACKABLE:
            result[key] = torch.stack([b[key] for b in batch])
        elif key in _LIST_TENSORS:
            result[key] = [b[key] for b in batch]
        elif key in _META:
            result[key] = [b[key] for b in batch]
        elif key == "support_plane_y":
            # Optional per-sample scalar: tensor + availability mask
            values = [b[key] for b in batch]
            has_values = [v is not None for v in values]
            if any(has_values):
                result["support_plane_y"] = torch.tensor([v if v is not None else 0.0 for v in values], dtype=torch.float32)
                result["has_support_plane"] = torch.tensor(has_values, dtype=torch.bool)
            else:
                result["support_plane_y"] = None
                result["has_support_plane"] = None
        else:
            # Unknown key — attempt stack, fall back to list
            try:
                result[key] = torch.stack([b[key] for b in batch])
            except (TypeError, RuntimeError):
                result[key] = [b[key] for b in batch]
    return result
