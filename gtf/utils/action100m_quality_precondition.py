"""Fail-closed terminal record for Action100M segments too short for the quality gates.

A segment whose optimizer output (optimized_mano/optimized.npz) has fewer than
MINIMUM_KEYFRAMES keyframes cannot be scored by the 21 quality gates. It is not an
action reject, and a retry cannot add keyframes to the same optimizer output, so the
refine step writes a separate terminal record, quality/precondition_reject.json, that
marks the segment as not admissible for training. The record is bound to the hash of
the optimizer output and is checked against it whenever it is read.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any
import uuid

import numpy as np

from gtf.utils.video_timing import file_sha256

FORMAT = "action100m-quality-precondition-reject-v1"
REASON = "insufficient_temporal_support"
STAGE = "refine"
MINIMUM_KEYFRAMES = 3
RELATIVE_PATH = Path("quality/precondition_reject.json")
OPTIMIZED_RELATIVE_PATH = Path("optimized_mano/optimized.npz")
_DOWNSTREAM_CONFLICTS = (
    Path("optimized_stage2/optimized.npz"),
    Path("canonical/canonical.npz"),
    Path("quality/quality.json"),
)
_REQUIRED_KEYS = frozenset(
    {
        "format",
        "segment",
        "stage",
        "reason",
        "training_admissible",
        "minimum_required_keyframes",
        "observed_keyframes",
        "frame_indices",
        "frame_indices_sha256",
        "optimized_mano_path",
        "optimized_mano_sha256",
        "optimized_mano_size_bytes",
    }
)


class QualityPreconditionError(ValueError):
    """Quality-precondition evidence is absent, malformed, or inconsistent."""


def _validate_segment_identity(segment: str) -> None:
    """Require one literal child name, never a traversal or absolute path."""

    if (
        not isinstance(segment, str)
        or not segment
        or segment in {".", ".."}
        or "\x00" in segment
        or Path(segment).is_absolute()
        or Path(segment).parts != (segment,)
    ):
        raise QualityPreconditionError(f"unsafe segment identity: {segment!r}")


def _directory_open_flags() -> int:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return flags


def _open_quality_directory(segment_dir: Path, *, create: bool) -> tuple[int, int | None]:
    """Open ``quality/`` relative to a non-symlink segment directory.

    Directory-relative descriptors keep publication inside the inspected segment
    even if another process renames paths while the create-once record is written.
    """

    try:
        segment_fd = os.open(segment_dir, _directory_open_flags())
    except OSError as exc:
        raise QualityPreconditionError(
            f"segment directory is absent or unsafe: {segment_dir}: {exc}"
        ) from exc
    try:
        try:
            quality_fd = os.open(
                "quality",
                _directory_open_flags(),
                dir_fd=segment_fd,
            )
        except FileNotFoundError:
            if not create:
                return segment_fd, None
            try:
                os.mkdir("quality", mode=0o755, dir_fd=segment_fd)
            except FileExistsError:
                # A concurrent creator won.  The O_NOFOLLOW open below still
                # proves that the winner published a real directory.
                pass
            quality_fd = os.open(
                "quality",
                _directory_open_flags(),
                dir_fd=segment_fd,
            )
        except OSError as exc:
            raise QualityPreconditionError(
                f"quality directory is absent or unsafe: {segment_dir / 'quality'}: {exc}"
            ) from exc
        if not stat.S_ISDIR(os.fstat(quality_fd).st_mode):
            os.close(quality_fd)
            raise QualityPreconditionError(
                f"quality directory is absent or unsafe: {segment_dir / 'quality'}"
            )
        return segment_fd, quality_fd
    except Exception:
        os.close(segment_fd)
        raise


def _read_record_from_directory(quality_fd: int, path: Path) -> dict[str, Any] | None:
    name = RELATIVE_PATH.name
    try:
        record_fd = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=quality_fd,
        )
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise QualityPreconditionError(
            f"quality-precondition record is not a regular file: {path}: {exc}"
        ) from exc
    try:
        if not stat.S_ISREG(os.fstat(record_fd).st_mode):
            raise QualityPreconditionError(
                f"quality-precondition record is not a regular file: {path}"
            )
        try:
            with os.fdopen(record_fd, "r", encoding="utf-8") as stream:
                record_fd = -1
                record = json.load(stream)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise QualityPreconditionError(
                f"quality-precondition record is unreadable: {path}: {exc}"
            ) from exc
    finally:
        if record_fd >= 0:
            os.close(record_fd)
    if not isinstance(record, dict):
        raise QualityPreconditionError(
            f"quality-precondition record has the wrong schema: {path}"
        )
    return record


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_regular_nonsymlink(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def _typed_equal(left: Any, right: Any) -> bool:
    """JSON equality that does not accept ``True == 1`` type confusion."""

    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _typed_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _typed_equal(a, b) for a, b in zip(left, right)
        )
    return bool(left == right)


def _optimized_evidence(segment_dir: Path) -> dict[str, Any]:
    optimized_parent = segment_dir / OPTIMIZED_RELATIVE_PATH.parent
    if not optimized_parent.is_dir() or optimized_parent.is_symlink():
        raise QualityPreconditionError(
            f"optimized MANO directory is absent or unsafe: {optimized_parent}"
        )
    try:
        if optimized_parent.resolve(strict=True) != optimized_parent.absolute():
            raise QualityPreconditionError(
                f"optimized MANO directory traverses a symlink: {optimized_parent}"
            )
    except OSError as exc:
        raise QualityPreconditionError(
            f"optimized MANO directory is absent or unsafe: {optimized_parent}: {exc}"
        ) from exc
    optimized = segment_dir / OPTIMIZED_RELATIVE_PATH
    if not _is_regular_nonsymlink(optimized):
        raise QualityPreconditionError(
            f"optimized MANO input is absent or unsafe: {optimized}"
        )

    try:
        before = optimized.stat()
        before_sha = file_sha256(optimized)
        with np.load(optimized, allow_pickle=False) as archive:
            if "frame_indices" not in archive.files:
                raise QualityPreconditionError(
                    f"optimized MANO input has no frame_indices: {optimized}"
                )
            raw = np.asarray(archive["frame_indices"])
        after_sha = file_sha256(optimized)
        after = optimized.stat()
    except QualityPreconditionError:
        raise
    except Exception as exc:  # noqa: BLE001 - malformed NPZ must fail closed
        raise QualityPreconditionError(
            f"cannot read optimized MANO input: {optimized}: {exc}"
        ) from exc

    before_signature = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_signature = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_signature != after_signature or before_sha != after_sha:
        raise QualityPreconditionError(
            f"optimized MANO input changed while inspected: {optimized}"
        )
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
        raise QualityPreconditionError(
            f"optimized MANO frame_indices must be a 1-D integer array: {optimized}"
        )
    frame_indices = [int(value) for value in raw.tolist()]
    if (
        not frame_indices
        or len(frame_indices) >= MINIMUM_KEYFRAMES
        or any(value < 0 for value in frame_indices)
        or any(right <= left for left, right in zip(frame_indices, frame_indices[1:]))
    ):
        raise QualityPreconditionError(
            "insufficient-temporal-support evidence requires a strictly increasing "
            f"1..{MINIMUM_KEYFRAMES - 1}-frame roster: {optimized}"
        )
    return {
        "frame_indices": frame_indices,
        "frame_indices_sha256": _sha256_json(frame_indices),
        "optimized_mano_sha256": before_sha,
        "optimized_mano_size_bytes": before.st_size,
    }


def _expected_record(segment_dir: Path, segment: str) -> dict[str, Any]:
    _validate_segment_identity(segment)
    if not segment_dir.is_dir() or segment_dir.is_symlink():
        raise QualityPreconditionError(
            f"segment directory is absent or unsafe: {segment_dir}"
        )
    if segment_dir.name != segment:
        raise QualityPreconditionError(
            f"segment directory identity mismatch: {segment_dir.name!r} != {segment!r}"
        )
    conflicts = [str(path) for relative in _DOWNSTREAM_CONFLICTS if (path := segment_dir / relative).exists() or path.is_symlink()]
    if conflicts:
        raise QualityPreconditionError(
            "quality-precondition rejection conflicts with downstream output: "
            + ", ".join(conflicts)
        )
    evidence = _optimized_evidence(segment_dir)
    return {
        "format": FORMAT,
        "segment": segment,
        "stage": STAGE,
        "reason": REASON,
        "training_admissible": False,
        "minimum_required_keyframes": MINIMUM_KEYFRAMES,
        "observed_keyframes": len(evidence["frame_indices"]),
        "frame_indices": evidence["frame_indices"],
        "frame_indices_sha256": evidence["frame_indices_sha256"],
        "optimized_mano_path": OPTIMIZED_RELATIVE_PATH.as_posix(),
        "optimized_mano_sha256": evidence["optimized_mano_sha256"],
        "optimized_mano_size_bytes": evidence["optimized_mano_size_bytes"],
    }


def load_quality_precondition_reject(
    segment_dir: str | Path,
    *,
    expected_segment: str | None = None,
) -> dict[str, Any] | None:
    """Return a validated terminal record, ``None`` when no record exists.

    A dangling symlink or malformed/conflicting record is an error rather than
    absence.  Validation re-hashes the live optimizer input every time.
    """

    segment_dir = Path(segment_dir)
    segment = expected_segment if expected_segment is not None else segment_dir.name
    _validate_segment_identity(segment)
    path = segment_dir / RELATIVE_PATH
    try:
        segment_fd, quality_fd = _open_quality_directory(segment_dir, create=False)
        if quality_fd is None:
            return None
        try:
            record = _read_record_from_directory(quality_fd, path)
        finally:
            os.close(quality_fd)
    finally:
        if "segment_fd" in locals():
            os.close(segment_fd)
    if record is None:
        return None
    if set(record) != _REQUIRED_KEYS:
        raise QualityPreconditionError(
            f"quality-precondition record has the wrong schema: {path}"
        )
    expected = _expected_record(segment_dir, segment)
    if not _typed_equal(record, expected):
        raise QualityPreconditionError(
            f"quality-precondition record is stale or mismatched: {path}"
        )
    return record


def publish_quality_precondition_reject(
    segment_dir: str | Path,
    *,
    expected_segment: str | None = None,
) -> dict[str, Any]:
    """Atomically create an idempotent, content-bound terminal record.

    The target is create-once.  A concurrent identical publisher is accepted;
    a conflicting pre-existing record is refused and never overwritten.
    """

    segment_dir = Path(segment_dir)
    segment = expected_segment if expected_segment is not None else segment_dir.name
    expected = _expected_record(segment_dir, segment)
    path = segment_dir / RELATIVE_PATH
    payload = json.dumps(
        expected,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    segment_fd, quality_fd = _open_quality_directory(segment_dir, create=True)
    assert quality_fd is not None
    temporary_name = f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    temporary_fd = -1
    try:
        existing = _read_record_from_directory(quality_fd, path)
        if existing is not None:
            if not _typed_equal(existing, expected):
                raise QualityPreconditionError(
                    f"quality-precondition record conflicts with current input: {path}"
                )
            return existing
        temporary_fd = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=quality_fd,
        )
        with os.fdopen(temporary_fd, "w", encoding="utf-8") as stream:
            temporary_fd = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if not _typed_equal(_expected_record(segment_dir, segment), expected):
            raise QualityPreconditionError(
                "quality-precondition input changed before publication: "
                f"{segment_dir}"
            )
        try:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=quality_fd,
                dst_dir_fd=quality_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            existing = _read_record_from_directory(quality_fd, path)
            if not _typed_equal(existing, expected):
                raise QualityPreconditionError(
                    f"concurrent quality-precondition record conflicts: {path}"
                )
            return existing
        os.fsync(quality_fd)
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        try:
            os.unlink(temporary_name, dir_fd=quality_fd)
        except FileNotFoundError:
            pass
        os.close(quality_fd)
        os.close(segment_fd)
    published = load_quality_precondition_reject(
        segment_dir, expected_segment=segment
    )
    if published is None:
        raise QualityPreconditionError(
            f"quality-precondition record disappeared after publication: {path}"
        )
    return published
