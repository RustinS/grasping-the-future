"""Content bindings, cache checks and atomic writes of the object-pose stage."""

import hashlib
import json
import os
import tempfile
from zipfile import BadZipFile
import numpy as np
from pathlib import Path


from gtf.paths import REPO_ROOT

ROOT = str(REPO_ROOT)
OBJECT_POSE_PIPELINE_VERSION = 5


class ObjectPoseTerminalReject(RuntimeError):
    """A deterministic input-quality failure that retries cannot repair."""


def _npz_scalar(value, default=None):
    """Convert a 0-d npz value to a Python scalar."""
    if value is None:
        return default
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def _sha256_files(paths):
    """Content-bind an ordered collection of extraction inputs."""
    digest = hashlib.sha256()
    for path in sorted((Path(path) for path in paths), key=lambda value: str(value)):
        digest.update(path.name.encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256_file(path):
    """Hash one regular input file without folding in its host-specific path."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"required input is not a file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding_with_digest(payload):
    """Return a canonical JSON binding with its own SHA-256 digest."""

    output = dict(payload)
    encoded = json.dumps(output, sort_keys=True, separators=(",", ":")).encode()
    output["input_digest_sha256"] = hashlib.sha256(encoded).hexdigest()
    return output


def _object_pose_data_binding(
    *,
    video_path,
    masks_path,
    depth_dir,
    n_frames,
    decoded_frame_shape,
    verified_depth_output_binding=None,
):
    """Content-bind the effective shared inputs to object-pose stages.

    ``.depth_done`` is deliberately not evidence: it contains only ``done``.
    The binding hashes the exact depth files actually read for decoded rows
    ``0..T-1``. Extra files are ignored because they are not runtime inputs.
    """

    n_frames = int(n_frames)
    shape = tuple(int(value) for value in decoded_frame_shape)
    if n_frames < 1 or len(shape) != 4 or shape[0] != n_frames:
        raise ValueError("object-pose input binding requires a non-empty decoded T,H,W,C shape")
    depth_dir = Path(depth_dir)
    depth_paths = [depth_dir / f"{frame:05d}.npz" for frame in range(n_frames)]
    if any(not path.is_file() or path.is_symlink() for path in depth_paths):
        raise ValueError("object-pose input binding requires one regular depth file per decoded video frame")
    depth_set_sha256 = None
    if isinstance(verified_depth_output_binding, dict):
        records = verified_depth_output_binding.get("files")
        candidate = verified_depth_output_binding.get("full_frame_depth_set_sha256")
        if isinstance(candidate, str) and len(candidate) == 64 and isinstance(records, list) and len(records) == n_frames:
            # The caller has just content-verified the depth contract (or just
            # atomically produced it). Bind the same aggregate digest while
            # requiring the exact verified file generations to remain present.
            expected_stats = [record.get("stat") for record in records]
            if all(isinstance(expected, dict) and _file_stat_record(path) == expected for path, expected in zip(depth_paths, expected_stats)):
                depth_set_sha256 = candidate
    if depth_set_sha256 is None:
        depth_set_sha256 = _sha256_files(depth_paths)

    return _binding_with_digest(
        {
            "format": "object-pose-shared-input-binding-v1",
            "pipeline_version": OBJECT_POSE_PIPELINE_VERSION,
            "n_video_frames": n_frames,
            "decoded_frame_shape": list(shape),
            "video_sha256": _sha256_file(video_path),
            "object_masks_sha256": _sha256_file(masks_path),
            "full_frame_depth_count": len(depth_paths),
            "full_frame_depth_set_sha256": depth_set_sha256,
        }
    )


def _object_pose_input_stat_binding(*, video_path, masks_path, depth_dir, n_frames):
    """Cheap unchanged-file identity for the content-bound cache fast path.

    Device/inode/size/mtime/ctime make ordinary replacements and in-place
    rewrites observable. A mismatch never invalidates good data by itself; it
    falls back to the exact byte hashes in :func:`_object_pose_data_binding`.
    """

    paths = [
        ("video", Path(video_path)),
        ("object_masks", Path(masks_path)),
    ] + [(f"depth/{frame:05d}", Path(depth_dir) / f"{frame:05d}.npz") for frame in range(int(n_frames))]
    records = []
    for label, path in paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"object-pose stat binding requires a regular file: {path}")
        stat = path.stat()
        records.append(
            {
                "label": label,
                "device": int(stat.st_dev),
                "inode": int(stat.st_ino),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "ctime_ns": int(stat.st_ctime_ns),
            }
        )
    return _binding_with_digest(
        {
            "format": "object-pose-shared-input-stat-binding-v1",
            "n_video_frames": int(n_frames),
            "files": records,
        }
    )


def _object_pose_data_and_stat_binding(
    *,
    video_path,
    masks_path,
    depth_dir,
    n_frames,
    decoded_frame_shape,
    verified_depth_output_binding=None,
):
    """Content-bind a stable input snapshot without a hash/stat race."""

    for _attempt in range(2):
        before = _object_pose_input_stat_binding(
            video_path=video_path,
            masks_path=masks_path,
            depth_dir=depth_dir,
            n_frames=n_frames,
        )
        content = _object_pose_data_binding(
            video_path=video_path,
            masks_path=masks_path,
            depth_dir=depth_dir,
            n_frames=n_frames,
            decoded_frame_shape=decoded_frame_shape,
            verified_depth_output_binding=verified_depth_output_binding,
        )
        after = _object_pose_input_stat_binding(
            video_path=video_path,
            masks_path=masks_path,
            depth_dir=depth_dir,
            n_frames=n_frames,
        )
        if before == after:
            return content, after
    raise RuntimeError("object-pose inputs changed while their content binding was computed")


def _file_stat_record(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"object-pose artifact must be a regular file: {path}")
    stat = path.stat()
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
    }


def _output_artifact_binding(path):
    """Publish exact output bytes plus a cheap unchanged-file identity."""

    for _attempt in range(2):
        before = _file_stat_record(path)
        digest = _sha256_file(path)
        after = _file_stat_record(path)
        if before == after:
            return {
                "format": "object-pose-output-artifact-binding-v1",
                "sha256": digest,
                "stat": after,
            }
    raise RuntimeError(f"object-pose output changed while hashing: {path}")


def _output_artifact_matches(path, expected):
    if (
        not isinstance(expected, dict)
        or expected.get("format") != "object-pose-output-artifact-binding-v1"
        or not isinstance(expected.get("sha256"), str)
        or not isinstance(expected.get("stat"), dict)
    ):
        return False
    # Do not treat a matching stat tuple as proof of matching bytes.  Some
    # filesystems preserve device/inode/size/mtime/ctime across a same-size overwrite, so
    # that shortcut can silently adopt a modified extraction artifact.  The
    # stat record remains useful provenance, but the SHA-256 is authoritative.
    current = _output_artifact_binding(path)
    return current["sha256"] == expected["sha256"]


def _load_previous_stage_npz(metadata, name, path):
    """Load a reusable stage only when prior metadata binds its exact bytes.

    The complete-cache preflight uses these output bindings to notice a
    changed/truncated NPZ.  Re-entering ``process_segment`` must apply the same
    rule; otherwise a valid-looking edit could be adopted and re-attested on
    the next run.  Any read or binding failure is a cache miss, never a crash.
    """

    try:
        expected = metadata.get("output_artifact_bindings", {}).get(name)
        if not _output_artifact_matches(path, expected):
            return None
        with np.load(path, allow_pickle=False) as archive:
            return {key: np.asarray(archive[key]).copy() for key in archive.files}
    except (OSError, ValueError, KeyError, TypeError, BadZipFile, EOFError):
        return None


def _object_pose_stage_binding(shared, stage, **settings):
    """Derive one stage cache key from the shared content binding."""

    if not isinstance(shared, dict) or shared.get("format") != "object-pose-shared-input-binding-v1" or not isinstance(shared.get("input_digest_sha256"), str):
        raise ValueError("object-pose stage binding received a malformed base")
    return _binding_with_digest(
        {
            "format": f"object-pose-{stage}-input-binding-v1",
            "pipeline_version": OBJECT_POSE_PIPELINE_VERSION,
            "shared_input_digest_sha256": shared["input_digest_sha256"],
            **settings,
        }
    )


def _cache_binding_matches(cache, key, expected):
    """Fail closed on a missing, malformed, or mismatched embedded binding."""

    digest = _npz_scalar(cache.get(f"{key}_input_digest_sha256"), "")
    payload = _npz_scalar(cache.get(f"{key}_input_binding_json"), "")
    if digest != expected.get("input_digest_sha256") or not isinstance(payload, str):
        return False
    try:
        decoded = json.loads(payload)
    except (TypeError, json.JSONDecodeError):
        return False
    return decoded == expected


def _binding_cache_fields(key, binding):
    return {
        f"{key}_input_binding_json": np.array(json.dumps(binding, sort_keys=True, separators=(",", ":"))),
        f"{key}_input_digest_sha256": np.array(binding["input_digest_sha256"]),
    }


def _savez_compressed_atomic(path, **arrays):
    """Publish a verified NPZ without exposing a partial cache file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".npz", dir=path.parent, delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        np.savez_compressed(temporary, **arrays)
        with np.load(temporary, allow_pickle=False) as archive:
            if set(archive.files) != set(arrays):
                raise ValueError(f"temporary cache key roster changed for {path}")
            for key in archive.files:
                np.asarray(archive[key])
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path, payload):
    """Atomically publish strict JSON and verify its exact decoded value."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_text = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".json", dir=path.parent)
    temporary = Path(temporary_text)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if json.loads(temporary.read_text()) != payload:
            raise ValueError(f"temporary JSON verification failed for {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_object_pose_done(marker_path, metadata_path):
    """Bind the final marker to the exact atomic metadata publication."""

    marker_path = Path(marker_path)
    fd, temporary_text = tempfile.mkstemp(prefix=f".{marker_path.name}.", dir=marker_path.parent)
    temporary = Path(temporary_text)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(_sha256_file(metadata_path) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, marker_path)
    finally:
        temporary.unlink(missing_ok=True)


def _object_pose_done_matches(marker_path, metadata_path):
    try:
        return Path(marker_path).read_text().strip() == _sha256_file(metadata_path)
    except OSError:
        return False
