"""Metric depth for the object-pose stage: the depth model, SAM 3D keyframe depth, and the depth cache contract."""

from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
import os
from zipfile import BadZipFile
import numpy as np
import torch
from pathlib import Path
from gtf.utils import rprint as print
from gtf.utils.npz import savez_compressed_level
from gtf.extraction.object_pose.pose_cache import (
    OBJECT_POSE_PIPELINE_VERSION,
    _binding_with_digest,
    _file_stat_record,
    _object_pose_done_matches,
    _output_artifact_binding,
    _output_artifact_matches,
    _publish_object_pose_done,
    _write_json_atomic,
)


DEPTH_MODEL_REPO = "yjh001/metricanything_student_pointmap"
DEPTH_MODEL_REVISION = "8f1f08c53c683d4e19864601dfaf78515f29a63b"
DEPTH_CACHE_METADATA_NAME = "depth_metadata.json"


def _load_depth_cache(depth_dir, T, frame_shape=None):
    """Load and validate one exact per-frame metric-depth roster."""
    results = []
    for t in range(T):
        npz_path = os.path.join(depth_dir, f"{t:05d}.npz")
        if not os.path.exists(npz_path):
            return None
        try:
            with np.load(npz_path, allow_pickle=False) as data:
                if not {"depth", "intrinsics_px", "mask"}.issubset(data.files):
                    return None
                result = {key: np.asarray(data[key]) for key in ("depth", "intrinsics_px", "mask")}
        except (OSError, ValueError, KeyError, EOFError, BadZipFile):
            return None
        depth = result["depth"]
        intrinsics = result["intrinsics_px"]
        mask = result["mask"]
        if (
            depth.ndim != 2
            or mask.shape != depth.shape
            or intrinsics.shape != (3, 3)
            or (frame_shape is not None and depth.shape != tuple(frame_shape))
            or not np.isfinite(depth).all()
            or not np.isfinite(intrinsics).all()
            or not np.isfinite(mask).all()
        ):
            return None
        results.append(result)
    return results


class DepthEstimator:
    """Wraps MoGe v2 (MetricAnything) for per-frame metric depth + intrinsics."""

    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        self.model = None
        self._model_device = None

    def _ensure_model(self):
        if self.model is None:
            from moge.model.v2 import MoGeModel

            self.model = MoGeModel.from_pretrained(
                DEPTH_MODEL_REPO,
                revision=DEPTH_MODEL_REVISION,
            )
        if self._model_device != self.device:
            self.model = self.model.to(self.device)
            self._model_device = self.device
            self.model.eval()
        return self.model

    def offload(self):
        """Move MoGe off GPU before CoTracker allocates attention tensors."""

        if self.model is None or self._model_device == torch.device("cpu"):
            return
        self.model = self.model.to("cpu")
        self._model_device = torch.device("cpu")
        gc.collect()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def infer_frame(self, frame_rgb: np.ndarray) -> dict:
        """Run MoGe on a single frame. Returns depth (H,W), intrinsics_px (3,3), mask (H,W)."""
        model = self._ensure_model()
        H, W = frame_rgb.shape[:2]
        image_tensor = torch.tensor(frame_rgb / 255.0, dtype=torch.float32, device=self.device).permute(2, 0, 1)
        # Opt-in camera calibration: with EXTRACT_KNOWN_FOV_X_DEG set, MoGe is
        # conditioned on that horizontal FoV (degrees) instead of estimating it.
        known_fov = os.environ.get("EXTRACT_KNOWN_FOV_X_DEG", "").strip()
        if known_fov:
            output = model.infer(image_tensor, fov_x=float(known_fov))
        else:
            output = model.infer(image_tensor)

        depth = output["depth"].cpu().numpy()
        intrinsics_norm = output["intrinsics"].cpu().numpy()
        mask = output["mask"].cpu().numpy()

        K_px = intrinsics_norm.copy()
        K_px[0, 0] *= W
        K_px[1, 1] *= H
        K_px[0, 2] *= W
        K_px[1, 2] *= H

        return {
            "depth": depth.astype(np.float32),
            "intrinsics_px": K_px.astype(np.float32),
            "mask": mask.astype(np.uint8),
        }


def _load_sam3d_depth(
    sam3d_dir,
    frame_index,
    frame_shape,
    allowed_frames,
    aggregate=None,
):
    """SAM 3D's MoGe output for one successful keyframe (from moge_keyframes.npz), or None."""
    if not sam3d_dir or frame_index not in allowed_frames or aggregate is None:
        return None
    suffix = f"_{frame_index:05d}"
    try:
        result = {key: aggregate[f"{key}{suffix}"] for key in ("depth", "intrinsics_px", "mask")}
    except (OSError, ValueError, KeyError):
        return None
    H, W = frame_shape
    if result["depth"].shape != (H, W) or result["mask"].shape != (H, W) or result["intrinsics_px"].shape != (3, 3):
        return None
    return result


def _sam3d_success_frames(sam3d_dir):
    if not sam3d_dir or not os.path.exists(os.path.join(sam3d_dir, ".sam3d_done")):
        return set()
    try:
        with open(os.path.join(sam3d_dir, "metadata.json")) as f:
            metadata = json.load(f)
        return {int(row["frame_index"]) for row in metadata.get("frames", []) if row.get("status") == "ok"}
    except (OSError, ValueError, KeyError):
        return set()


def _canonicalize_generated_depth_result(result, frame_shape):
    """Make invalid-mask depth pixels storage-safe without inventing evidence.

    MetricAnything can emit ``+inf`` at pixels where its own validity mask is
    false.  Those values are never usable depth evidence, but persisting them
    makes the content-bound cache fail its all-finite contract after an
    otherwise successful inference.  Canonicalize *only* non-finite values
    outside the validity mask to zero.  A non-finite value inside the mask is
    a real model-output failure and remains fail-closed.

    Normal finite outputs are returned byte-for-byte unchanged.
    """

    required = {"depth", "intrinsics_px", "mask"}
    if not isinstance(result, dict) or not required.issubset(result):
        raise ValueError("metric-depth inference returned a malformed result")
    depth = np.asarray(result["depth"])
    intrinsics = np.asarray(result["intrinsics_px"])
    mask = np.asarray(result["mask"])
    expected_shape = tuple(int(value) for value in frame_shape)
    if depth.shape != expected_shape or mask.shape != expected_shape or intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all() or not np.isfinite(mask).all():
        raise ValueError("metric-depth inference returned malformed arrays")
    nonfinite = ~np.isfinite(depth)
    if not nonfinite.any():
        return result
    if np.any(nonfinite & mask.astype(bool)):
        raise ValueError("metric-depth inference returned non-finite valid pixels")
    canonical = dict(result)
    canonical_depth = depth.copy()
    canonical_depth[nonfinite] = 0
    canonical["depth"] = canonical_depth
    return canonical


def run_depth_estimation(
    depth_estimator,
    frames,
    depth_dir,
    sam3d_dir=None,
    output_bindings=None,
    output_set_binding=None,
):
    """Run MoGe on all frames and write one depth file per frame (depth_dir/{t:05d}.npz).

    Keyframes that SAM 3D finished reuse its MoGe output. Returns (per-frame results, number of reused frames).
    """
    os.makedirs(depth_dir, exist_ok=True)
    results = []
    reused = 0
    allowed_frames = _sam3d_success_frames(sam3d_dir)
    aggregate = None
    aggregate_path = os.path.join(sam3d_dir, "moge_keyframes.npz") if sam3d_dir else None
    if allowed_frames and aggregate_path and os.path.exists(aggregate_path):
        try:
            aggregate = np.load(aggregate_path, allow_pickle=False)
        except (OSError, ValueError):
            aggregate = None
    writer_workers = int(os.environ.get("OBJECT_POSE_DEPTH_WRITERS", "2"))
    if writer_workers < 1:
        raise ValueError("OBJECT_POSE_DEPTH_WRITERS must be positive")
    pending_writes = []
    depth_set_hasher = hashlib.sha256() if output_set_binding is not None else None

    def finish_write(frame_index, npz_path, future):
        binding = future.result()
        if output_bindings is None:
            return
        if not isinstance(binding, dict) or not isinstance(binding.get("sha256"), str) or not isinstance(binding.get("size_bytes"), int):
            raise RuntimeError("depth writer did not return a content binding")
        path = Path(npz_path)
        stat = path.stat()
        if path.is_symlink() or not path.is_file() or stat.st_size != binding["size_bytes"]:
            raise RuntimeError(f"depth writer published an invalid artifact: {path}")
        payload = binding.pop("payload", None)
        if depth_set_hasher is not None:
            if not isinstance(payload, bytes):
                raise RuntimeError("depth writer did not return its exact payload")
            # The same digest as `_sha256_files` over the written files (name,
            # NUL, bytes, NUL per file), computed from the in-memory payload so
            # the files are not read back.
            depth_set_hasher.update(path.name.encode())
            depth_set_hasher.update(b"\0")
            depth_set_hasher.update(payload)
            depth_set_hasher.update(b"\0")
        output_bindings[int(frame_index)] = dict(binding)

    try:
        # Frame caches are independent and published atomically, so a small
        # bounded writer pool (OBJECT_POSE_DEPTH_WRITERS, default 2) compresses
        # and writes earlier frames while the GPU infers the next one. The
        # returned arrays are copied before they are queued so a reused
        # model-side buffer cannot change the cache or the in-memory result.
        with ThreadPoolExecutor(max_workers=writer_workers) as writer_pool:
            for t in range(len(frames)):
                npz_path = os.path.join(depth_dir, f"{t:05d}.npz")
                result = _load_sam3d_depth(
                    sam3d_dir,
                    t,
                    frames[t].shape[:2],
                    allowed_frames,
                    aggregate=aggregate,
                )
                if result is None:
                    result = depth_estimator.infer_frame(frames[t])
                else:
                    reused += 1
                result = _canonicalize_generated_depth_result(result, frames[t].shape[:2])
                result = {key: np.array(value, copy=True) for key, value in result.items()}
                results.append(result)
                pending_writes.append(
                    (
                        t,
                        npz_path,
                        writer_pool.submit(
                            savez_compressed_level,
                            npz_path,
                            compresslevel=1,
                            allow_pickle=False,
                            return_sha256=output_bindings is not None,
                            return_payload=output_set_binding is not None,
                            **result,
                        ),
                    )
                )
                # Bound both memory and outstanding writes. Waiting in
                # submission order also makes the first write error fail
                # the stage deterministically.
                if len(pending_writes) >= writer_workers:
                    finish_write(*pending_writes.pop(0))
            for pending in pending_writes:
                finish_write(*pending)
    finally:
        if aggregate is not None:
            aggregate.close()
    if reused:
        print(f"  Reused SAM3D MoGe depth for {reused}/{len(frames)} frames")
    if output_set_binding is not None and depth_set_hasher is not None and output_bindings is not None and len(output_bindings) == len(frames):
        output_set_binding["sha256"] = depth_set_hasher.hexdigest()
    return results, reused


def _depth_source_paths(sam3d_dir):
    """Return the conservative set of SAM3D files that may source depth."""

    records = []
    success_frames = []
    sam3d_dir = None if sam3d_dir is None else Path(sam3d_dir)
    if sam3d_dir is None:
        return success_frames, records
    marker = sam3d_dir / ".sam3d_done"
    metadata = sam3d_dir / "metadata.json"
    if marker.is_file():
        records.append((".sam3d_done", marker))
    if metadata.is_file():
        records.append(("metadata.json", metadata))
    success_frames = sorted(_sam3d_success_frames(str(sam3d_dir)))
    aggregate = sam3d_dir / "moge_keyframes.npz"
    if success_frames and aggregate.is_file():
        records.append(("moge_keyframes.npz", aggregate))
    return success_frames, records


def _depth_input_binding(
    *,
    video_path,
    sam3d_dir,
    n_frames,
    decoded_frame_shape,
):
    """Bind metric-depth generation to its exact RGB/model/source recipe.

    Depth is upstream of tracking, camera estimation, and BA.  Binding only
    the produced depth bytes is insufficient: after the video is replaced those
    bytes would still be internally consistent but belong to the other video.
    The SAM3D marker, metadata and moge_keyframes.npz that supply reused
    keyframe depth are bound too.
    """

    n_frames = int(n_frames)
    shape = tuple(int(value) for value in decoded_frame_shape)
    if n_frames < 1 or len(shape) != 4 or shape[0] != n_frames:
        raise ValueError("depth input binding received an invalid decoded roster")
    video_binding = _output_artifact_binding(video_path)
    success_frames, source_paths = _depth_source_paths(sam3d_dir)
    sam3d_records = [
        {
            "name": name,
            "sha256": _output_artifact_binding(path)["sha256"],
        }
        for name, path in source_paths
    ]
    return _binding_with_digest(
        {
            "format": "object-pose-depth-input-binding-v1",
            "pipeline_version": OBJECT_POSE_PIPELINE_VERSION,
            "n_video_frames": n_frames,
            "decoded_frame_shape": list(shape),
            "video_sha256": video_binding["sha256"],
            "model_repo": DEPTH_MODEL_REPO,
            "model_revision": DEPTH_MODEL_REVISION,
            "sam3d_success_frames": success_frames,
            "sam3d_source_files": sam3d_records,
        }
    )


def _depth_input_stat_binding(
    *,
    video_path,
    sam3d_dir,
    n_frames,
    decoded_frame_shape,
):
    """Cheap identity for the already content-bound depth source roster."""

    n_frames = int(n_frames)
    shape = tuple(int(value) for value in decoded_frame_shape)
    success_frames, source_paths = _depth_source_paths(sam3d_dir)
    return _binding_with_digest(
        {
            "format": "object-pose-depth-input-stat-binding-v1",
            "pipeline_version": OBJECT_POSE_PIPELINE_VERSION,
            "n_video_frames": n_frames,
            "decoded_frame_shape": list(shape),
            "model_repo": DEPTH_MODEL_REPO,
            "model_revision": DEPTH_MODEL_REVISION,
            "sam3d_success_frames": success_frames,
            "files": [
                {"name": "video.mp4", **_file_stat_record(video_path)},
                *[{"name": name, **_file_stat_record(path)} for name, path in source_paths],
            ],
        }
    )


def _depth_input_and_stat_binding(**kwargs):
    """Content-bind one stable depth-source snapshot."""

    for _attempt in range(2):
        before = _depth_input_stat_binding(**kwargs)
        content = _depth_input_binding(**kwargs)
        after = _depth_input_stat_binding(**kwargs)
        if before == after:
            return content, after
    raise RuntimeError("metric-depth inputs changed while they were being bound")


def _depth_output_binding(
    depth_dir,
    n_frames,
    frame_shape,
    *,
    produced_bindings=None,
    produced_set_sha256=None,
    validated_results=None,
):
    """Validate and content-bind the exact depth files consumed downstream."""

    depth_dir = Path(depth_dir)
    use_producer_bindings = (
        isinstance(produced_bindings, dict) and len(produced_bindings) == int(n_frames) and isinstance(validated_results, list) and len(validated_results) == int(n_frames)
    )
    if use_producer_bindings:
        expected_shape = tuple(int(value) for value in frame_shape)
        for result in validated_results:
            if not isinstance(result, dict) or not {
                "depth",
                "intrinsics_px",
                "mask",
            }.issubset(result):
                raise ValueError("metric-depth producer returned malformed arrays")
            depth = np.asarray(result["depth"])
            intrinsics = np.asarray(result["intrinsics_px"])
            mask = np.asarray(result["mask"])
            if (
                depth.shape != expected_shape
                or mask.shape != expected_shape
                or intrinsics.shape != (3, 3)
                or not np.isfinite(depth).all()
                or not np.isfinite(intrinsics).all()
                or not np.isfinite(mask).all()
            ):
                raise ValueError("metric-depth producer returned invalid arrays")
    else:
        loaded = _load_depth_cache(depth_dir, int(n_frames), frame_shape=frame_shape)
        if loaded is None:
            raise ValueError("metric-depth cache is missing, corrupt, or malformed")
    files = []
    for frame in range(int(n_frames)):
        path = depth_dir / f"{frame:05d}.npz"
        if use_producer_bindings:
            produced = produced_bindings.get(frame)
            stat = _file_stat_record(path)
            if not isinstance(produced, dict) or not isinstance(produced.get("sha256"), str) or len(produced["sha256"]) != 64 or produced.get("size_bytes") != stat["size"]:
                raise ValueError("metric-depth producer binding is malformed")
            # The digest was calculated over the complete in-memory NPZ payload
            # handed to the atomic writer. On resume, `_output_artifact_matches`
            # hashes the persisted bytes, so any later or partial mutation still
            # invalidates the cache fail-closed.
            binding = {
                "format": "object-pose-output-artifact-binding-v1",
                "sha256": produced["sha256"],
                "stat": stat,
            }
        else:
            binding = _output_artifact_binding(path)
        files.append(
            {
                "name": path.name,
                "format": binding["format"],
                "sha256": binding["sha256"],
                "stat": binding["stat"],
            }
        )
    payload = {
        "format": "object-pose-depth-output-binding-v1",
        "n_frames": int(n_frames),
        "frame_shape": [int(value) for value in frame_shape],
        "files": files,
    }
    if use_producer_bindings and isinstance(produced_set_sha256, str) and len(produced_set_sha256) == 64:
        payload["full_frame_depth_set_sha256"] = produced_set_sha256
    return _binding_with_digest(payload)


def _depth_output_matches(depth_dir, expected, n_frames, frame_shape):
    if (
        not isinstance(expected, dict)
        or expected.get("format") != "object-pose-depth-output-binding-v1"
        or expected.get("n_frames") != int(n_frames)
        or expected.get("frame_shape") != [int(value) for value in frame_shape]
    ):
        return False
    records = expected.get("files")
    if not isinstance(records, list) or len(records) != int(n_frames):
        return False
    for frame, record in enumerate(records):
        name = f"{frame:05d}.npz"
        if not isinstance(record, dict) or record.get("name") != name:
            return False
        if not _output_artifact_matches(Path(depth_dir) / name, record):
            return False
    return True


def publish_depth_cache_contract(
    *,
    video_path,
    depth_dir,
    pose_dir,
    sam3d_dir,
    n_frames,
    decoded_frame_shape,
    sam3d_depth_frames_reused,
    produced_bindings=None,
    produced_set_sha256=None,
    validated_results=None,
):
    """Atomically attest a complete precomputed or freshly inferred depth cache."""

    shape = tuple(int(value) for value in decoded_frame_shape)
    input_binding, input_stat_binding = _depth_input_and_stat_binding(
        video_path=video_path,
        sam3d_dir=sam3d_dir,
        n_frames=n_frames,
        decoded_frame_shape=shape,
    )
    output_binding = _depth_output_binding(
        depth_dir,
        n_frames,
        shape[1:3],
        produced_bindings=produced_bindings,
        produced_set_sha256=produced_set_sha256,
        validated_results=validated_results,
    )
    payload = {
        "format": "object-pose-depth-cache-contract-v1",
        "pipeline_version": OBJECT_POSE_PIPELINE_VERSION,
        "sam3d_depth_frames_reused": int(sam3d_depth_frames_reused),
        "input_binding": input_binding,
        "input_stat_binding": input_stat_binding,
        "output_binding": output_binding,
    }
    pose_dir = Path(pose_dir)
    metadata_path = pose_dir / DEPTH_CACHE_METADATA_NAME
    marker_path = pose_dir / ".depth_done"
    _write_json_atomic(metadata_path, payload)
    _publish_object_pose_done(marker_path, metadata_path)
    return payload


def _load_matching_depth_cache_contract(
    *,
    video_path,
    depth_dir,
    pose_dir,
    sam3d_dir,
    n_frames,
    decoded_frame_shape,
):
    """Return a verified depth contract, or ``None`` on any stale/corrupt input."""

    try:
        pose_dir = Path(pose_dir)
        metadata_path = pose_dir / DEPTH_CACHE_METADATA_NAME
        marker_path = pose_dir / ".depth_done"
        payload = json.loads(metadata_path.read_text())
        if (
            not isinstance(payload, dict)
            or payload.get("format") != "object-pose-depth-cache-contract-v1"
            or payload.get("pipeline_version") != OBJECT_POSE_PIPELINE_VERSION
            or not _object_pose_done_matches(marker_path, metadata_path)
        ):
            return None
        # Stat identity is not content identity: on some filesystems a
        # same-size overwrite keeps the complete stat tuple.  Always
        # recompute the exact input binding before accepting cached depth.
        current_input, _stable_stat = _depth_input_and_stat_binding(
            video_path=video_path,
            sam3d_dir=sam3d_dir,
            n_frames=n_frames,
            decoded_frame_shape=decoded_frame_shape,
        )
        if payload.get("input_binding") != current_input:
            return None
        if not _depth_output_matches(
            depth_dir,
            payload.get("output_binding"),
            n_frames,
            tuple(int(value) for value in decoded_frame_shape)[1:3],
        ):
            return None
        return payload
    except (
        OSError,
        TypeError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        EOFError,
        BadZipFile,
    ):
        return None
