"""Content-bound decoded-video timing for exact-timing canonicals.

The frame times of an Action100M canonical are ffprobe's decoded-frame
``best_effort_timestamp_time``. When ffprobe is not installed, PyAV reads the same
decoded-frame PTS as a rational value. Both paths round to ffprobe's six-decimal
representation before any array is written or hashed, and the times are bound to
the SHA-256 of the video file.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np


FRAME_TIME_CONVENTION = "video_best_effort_pts_seconds_v1"
FRAME_TIME_SOURCE = "ffprobe.frame.best_effort_timestamp_time"
FFPROBE_PTS_DECIMALS = 6


class VideoTimingError(RuntimeError):
    """The media cannot satisfy the exact decoded-frame timing contract."""


@dataclass(frozen=True)
class VideoTiming:
    """An immutable, content-bound full decoded-frame timing probe."""

    video_sha256: str
    frame_times_s: np.ndarray
    decoded_frame_times_sha256: str
    backend: str

    def select(self, frame_indices: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        indices = validate_frame_indices(frame_indices, len(self.frame_times_s))
        selected = np.ascontiguousarray(self.frame_times_s[indices], dtype="<f8")
        if selected.size > 1 and np.any(np.diff(selected) <= 0):
            raise VideoTimingError(
                "selected decoded video PTS must be strictly increasing"
            )
        return selected, {
            "frame_time_convention": FRAME_TIME_CONVENTION,
            "frame_time_source": FRAME_TIME_SOURCE,
            "pts_probe_backend": self.backend,
            "video_sha256": self.video_sha256,
            "decoded_video_frames": int(self.frame_times_s.size),
            "decoded_frame_times_sha256": self.decoded_frame_times_sha256,
            "selected_frame_times_sha256": array_sha256(selected, "<f8"),
        }


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray, dtype: str) -> str:
    """SHA-256 over the array's shape, dtype and bytes (after casting to ``dtype``)."""

    normalized = np.ascontiguousarray(np.asarray(value, dtype=np.dtype(dtype)))
    digest = hashlib.sha256()
    digest.update(str(normalized.shape).encode())
    digest.update(b"\0")
    digest.update(normalized.dtype.str.encode())
    digest.update(b"\0")
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def _increasing_indices(value: np.ndarray) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size == 0 or raw.dtype.kind not in "iu":
        raise VideoTimingError("frame_indices must be a non-empty integer vector")
    indices = raw.astype(np.int64, copy=False)
    if indices.size > 1 and np.any(np.diff(indices) <= 0):
        raise VideoTimingError("frame_indices must be strictly increasing")
    return indices


def validate_frame_indices(value: np.ndarray, decoded_frames: int) -> np.ndarray:
    indices = _increasing_indices(value)
    if int(indices[0]) < 0 or int(indices[-1]) >= decoded_frames:
        raise VideoTimingError(
            "canonical frame_indices are outside the decoded video frame roster"
        )
    return indices


def validate_exact_timing_arrays(
    frame_indices: np.ndarray,
    frame_times_s: np.ndarray,
    frame_time_convention: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the canonical-side exact timing fields without probing media."""

    indices = _increasing_indices(frame_indices)
    times = np.asarray(frame_times_s)
    if times.dtype != np.dtype("float64") or times.ndim != 1:
        raise VideoTimingError("frame_times_s must be a one-dimensional float64 array")
    if times.shape != indices.shape:
        raise VideoTimingError("frame_times_s does not match frame_indices")
    if not np.isfinite(times).all():
        raise VideoTimingError("frame_times_s contains a non-finite value")
    if times.size > 1 and np.any(np.diff(times) <= 0):
        raise VideoTimingError("frame_times_s must be strictly increasing")

    convention_array = np.asarray(frame_time_convention)
    if convention_array.shape != ():
        raise VideoTimingError("frame_time_convention must be scalar")
    convention = str(convention_array.item())
    if convention != FRAME_TIME_CONVENTION:
        raise VideoTimingError(
            f"frame_time_convention must be {FRAME_TIME_CONVENTION!r}"
        )
    return indices, times


def probe_video_timing(video_path: Path | str, *, timeout_s: float = 600.0) -> VideoTiming:
    """Decode all presentation timestamps and bind them to stable media bytes."""

    path = Path(video_path).resolve(strict=True)
    before = _stat_fingerprint(path)
    video_digest = file_sha256(path)
    ffprobe = shutil.which("ffprobe")
    if ffprobe is not None:
        values = _probe_ffprobe(path, ffprobe, timeout_s)
        backend = "ffprobe.best_effort_timestamp_time"
    else:
        values = _probe_pyav(path)
        backend = "pyav.frame.pts*time_base"
    after = _stat_fingerprint(path)
    if before != after:
        raise VideoTimingError(f"video changed while it was hashed/probed: {path}")
    values = _validate_decoded_times(values)
    return VideoTiming(
        video_sha256=video_digest,
        frame_times_s=values,
        decoded_frame_times_sha256=array_sha256(values, "<f8"),
        backend=backend,
    )


def probe_selected_video_timing(
    video_path: Path | str,
    frame_indices: np.ndarray,
    *,
    timeout_s: float = 600.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    return probe_video_timing(video_path, timeout_s=timeout_s).select(frame_indices)


def _stat_fingerprint(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    if not path.is_file():
        raise VideoTimingError(f"expected a regular video file: {path}")
    return int(stat.st_dev), int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns)


def _validate_decoded_times(value: np.ndarray) -> np.ndarray:
    times = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    if times.ndim != 1 or times.size < 2:
        raise VideoTimingError("video probe returned fewer than two decoded frames")
    if not np.isfinite(times).all():
        raise VideoTimingError("decoded video PTS contains a non-finite value")
    if np.any(np.diff(times) <= 0):
        raise VideoTimingError("decoded video PTS must be strictly increasing")
    return times


def _probe_ffprobe(path: Path, ffprobe: str, timeout_s: float) -> np.ndarray:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "frame=best_effort_timestamp_time",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise VideoTimingError(f"ffprobe PTS probe timed out for {path}") from exc
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        raise VideoTimingError(
            f"ffprobe PTS probe failed for {path}: {stderr.strip() or exc}"
        ) from exc
    try:
        records = json.loads(result.stdout).get("frames")
    except (AttributeError, json.JSONDecodeError) as exc:
        raise VideoTimingError(f"ffprobe returned invalid PTS JSON: {exc}") from exc
    if not isinstance(records, list):
        raise VideoTimingError("ffprobe returned no decoded frame roster")
    values: list[float] = []
    for index, record in enumerate(records):
        raw = record.get("best_effort_timestamp_time") if isinstance(record, dict) else None
        try:
            timestamp = float(raw)
        except (TypeError, ValueError) as exc:
            raise VideoTimingError(
                f"decoded video frame {index} has invalid PTS {raw!r}"
            ) from exc
        if not math.isfinite(timestamp):
            raise VideoTimingError(
                f"decoded video frame {index} has non-finite PTS {timestamp!r}"
            )
        values.append(timestamp)
    return np.asarray(values, dtype=np.float64)


def _probe_pyav(path: Path) -> np.ndarray:
    try:
        import av
    except ImportError as exc:
        raise VideoTimingError(
            "ffprobe is unavailable and PyAV cannot be imported for exact PTS authoring"
        ) from exc
    values: list[float] = []
    try:
        with av.open(str(path), mode="r") as container:
            streams = list(container.streams.video)
            if not streams:
                raise VideoTimingError("PyAV found no video stream")
            for index, frame in enumerate(container.decode(streams[0])):
                if frame.pts is None or frame.time_base is None:
                    raise VideoTimingError(
                        f"decoded video frame {index} lacks PTS/time_base"
                    )
                # ffprobe renders timestamp_time with six decimal places.  It
                # multiplies after converting the rational time base to double;
                # keep that exact order to reproduce its half-microsecond cases.
                raw = float(frame.pts) * float(frame.time_base)
                timestamp = float(f"{raw:.{FFPROBE_PTS_DECIMALS}f}")
                if not math.isfinite(timestamp):
                    raise VideoTimingError(
                        f"decoded video frame {index} has non-finite PTS {timestamp!r}"
                    )
                values.append(timestamp)
    except VideoTimingError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise VideoTimingError(
            f"PyAV PTS probe failed for {path}: {type(exc).__name__}: {exc}"
        ) from exc
    return np.asarray(values, dtype=np.float64)
