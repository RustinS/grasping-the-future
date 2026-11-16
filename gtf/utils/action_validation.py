"""Fail-closed Action100M action-validation resolution.

A segment carries one ``valid_action/{object}/result.json`` row per validated object
noun. This module is the single interpretation of those rows (which object the segment
is about and whether it is valid) and owns the transactional rejection that the
pipeline stages use. Unreadable, incomplete or ambiguous rows never resolve to valid.
It has no project imports, so any extraction stage can use it without importing model code.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

REJECT_TRANSACTION_MARKER = ".reject_in_progress.json"
REJECT_TRANSACTION_FORMAT = "action100m-valid-action-reject-v1"

ActionValidationStatus = Literal[
    "valid",
    "invalid",
    "missing",
    "ambiguous",
    "unreadable",
]


@dataclass(frozen=True)
class ActionValidationResolution:
    """Resolved validation state for one segment.

    ``valid`` is true only when a single effective object can be selected from
    readable ``valid=true`` rows.
    """

    status: ActionValidationStatus
    valid: bool
    object_name: str | None
    result_path: Path | None
    metadata_object: str | None
    valid_objects: tuple[str, ...]
    invalid_objects: tuple[str, ...]
    unreadable_objects: tuple[str, ...]
    reasons: tuple[str, ...]


def _metadata_object(segment_dir: Path) -> tuple[str | None, str | None]:
    path = segment_dir / "metadata.json"
    if not path.exists():
        return None, None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"metadata unreadable: {type(exc).__name__}"
    if not isinstance(payload, dict):
        return None, "metadata is not a JSON object"
    value = payload.get("object")
    if value is None:
        return None, None
    if not isinstance(value, str) or not value.strip():
        return None, "metadata object is not a non-empty string"
    return value.strip(), None


def _read_result(path: Path) -> tuple[bool | None, str | None]:
    if not path.exists():
        return None, "missing result.json"
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"unreadable result.json: {type(exc).__name__}"
    if not isinstance(payload, dict):
        return None, "result.json is not a JSON object"
    value = payload.get("valid")
    if value is True:
        return True, None
    if value is False:
        return False, None
    return None, "result.json has no boolean valid field"


def _result_rows(
    segment_dir: Path,
) -> tuple[dict[str, tuple[Path, bool | None]], tuple[str, ...]]:
    va_dir = segment_dir / "valid_action"
    rows: dict[str, tuple[Path, bool | None]] = {}
    reasons: list[str] = []
    if not va_dir.is_dir():
        return {}, ()
    try:
        paths = sorted(
            path / "result.json" for path in va_dir.iterdir() if path.is_dir()
        )
    except OSError as exc:
        return {}, (f"valid_action unreadable: {type(exc).__name__}",)

    for result_path in paths:
        object_name = result_path.parent.name
        if object_name in rows:
            rows[object_name] = (result_path, None)
            reasons.append(f"{object_name}: duplicate action-result paths")
            continue
        valid, error = _read_result(result_path)
        rows[object_name] = (result_path, valid)
        if error is not None:
            reasons.append(f"{object_name}: {error}")
    return rows, tuple(reasons)


def _reject_transaction_status(segment_dir: Path) -> str | None:
    marker = segment_dir / "valid_action" / REJECT_TRANSACTION_MARKER
    if not marker.exists():
        return None
    try:
        payload = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return f"rejection transaction marker unreadable: {type(exc).__name__}"
    if (
        not isinstance(payload, dict)
        or payload.get("format") != REJECT_TRANSACTION_FORMAT
        or not isinstance(payload.get("objects"), list)
    ):
        return "rejection transaction marker is malformed"
    return "rejection transaction is incomplete"


def resolve_action_validation(
    segment_dir: str | Path,
    requested_object: str | None = None,
) -> ActionValidationResolution:
    """Resolve one segment's effective action object and validity.

    With ``requested_object`` only that exact row counts.  Object-agnostic callers
    (no ``requested_object``) use these deterministic rules:

    * an unreadable/incomplete row fails closed;
    * a valid metadata row disambiguates multiple valid rows;
    * otherwise exactly one valid row wins, even when metadata names a false row;
    * multiple valid rows without a valid metadata tie-break fail closed.

    With no validation rows, the metadata noun remains available so the validation
    stage can create its first row, but ``valid`` remains false.
    """

    segment_dir = Path(segment_dir)
    metadata_object, metadata_error = _metadata_object(segment_dir)
    rows, row_reasons = _result_rows(segment_dir)
    reasons = list(row_reasons)
    if metadata_error is not None:
        reasons.append(metadata_error)
    transaction_status = _reject_transaction_status(segment_dir)
    if transaction_status is not None:
        reasons.append(transaction_status)
        return ActionValidationResolution(
            status="unreadable",
            valid=False,
            object_name=None,
            result_path=None,
            metadata_object=metadata_object,
            valid_objects=tuple(
                sorted(name for name, (_, valid) in rows.items() if valid is True)
            ),
            invalid_objects=tuple(
                sorted(name for name, (_, valid) in rows.items() if valid is False)
            ),
            unreadable_objects=tuple(
                sorted(name for name, (_, valid) in rows.items() if valid is None)
            ),
            reasons=tuple(reasons),
        )

    if requested_object is not None:
        requested_object = str(requested_object).strip()
        result_path = segment_dir / "valid_action" / requested_object / "result.json"
        valid, error = _read_result(result_path)
        if error is not None:
            reasons.append(f"{requested_object}: {error}")
            status: ActionValidationStatus = (
                "missing" if not result_path.exists() else "unreadable"
            )
        else:
            status = "valid" if valid is True else "invalid"
        return ActionValidationResolution(
            status=status,
            valid=valid is True,
            object_name=requested_object,
            result_path=result_path,
            metadata_object=metadata_object,
            valid_objects=(requested_object,) if valid is True else (),
            invalid_objects=(requested_object,) if valid is False else (),
            unreadable_objects=(requested_object,) if valid is None else (),
            reasons=tuple(reasons),
        )

    valid_objects = tuple(
        sorted(name for name, (_, valid) in rows.items() if valid is True)
    )
    invalid_objects = tuple(
        sorted(name for name, (_, valid) in rows.items() if valid is False)
    )
    unreadable_objects = tuple(
        sorted(name for name, (_, valid) in rows.items() if valid is None)
    )

    # A partially-written or malformed alternative row may conceal a second valid
    # object, so object-agnostic resolution stops rather than silently choosing.
    if unreadable_objects or (row_reasons and not rows):
        return ActionValidationResolution(
            status="unreadable",
            valid=False,
            object_name=None,
            result_path=None,
            metadata_object=metadata_object,
            valid_objects=valid_objects,
            invalid_objects=invalid_objects,
            unreadable_objects=unreadable_objects,
            reasons=tuple(reasons),
        )

    selected: str | None = None
    status = "missing"
    if valid_objects:
        if metadata_object in valid_objects:
            selected = metadata_object
        elif len(valid_objects) == 1:
            selected = valid_objects[0]
        else:
            reasons.append(
                "multiple valid action rows have no valid metadata tie-break: "
                + ", ".join(valid_objects)
            )
            status = "ambiguous"
        if selected is not None:
            status = "valid"
    elif invalid_objects:
        status = "invalid"
        if metadata_object in invalid_objects:
            selected = metadata_object
        elif len(invalid_objects) == 1:
            selected = invalid_objects[0]
    else:
        selected = metadata_object

    result_path = (
        rows[selected][0] if selected is not None and selected in rows else None
    )
    return ActionValidationResolution(
        status=status,
        valid=status == "valid",
        object_name=selected,
        result_path=result_path,
        metadata_object=metadata_object,
        valid_objects=valid_objects,
        invalid_objects=invalid_objects,
        unreadable_objects=unreadable_objects,
        reasons=tuple(reasons),
    )


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
        _fsync_dir(path.parent)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_reject_transaction(segment_dir: Path) -> dict | None:
    marker = segment_dir / "valid_action" / REJECT_TRANSACTION_MARKER
    if not marker.exists():
        return None
    try:
        payload = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"cannot resume {segment_dir.name}: unreadable rejection transaction"
        ) from exc
    objects = payload.get("objects") if isinstance(payload, dict) else None
    reason = payload.get("reason") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("format") != REJECT_TRANSACTION_FORMAT
        or not isinstance(objects, list)
        or not objects
        or any(
            not isinstance(value, str) or not value or Path(value).name != value
            for value in objects
        )
        or len(objects) != len(set(objects))
        or not isinstance(reason, str)
        or not reason
    ):
        raise RuntimeError(
            f"cannot resume {segment_dir.name}: malformed rejection transaction"
        )
    return payload


def reject_action_validation(
    segment_dir: str | Path,
    reason: str,
    requested_object: str | None = None,
) -> tuple[str, ...]:
    """Persist one coherent rejection and return the object rows changed.

    An explicit-object run changes only that exact row.  An object-agnostic run
    invalidates *all* currently-valid rows so another stale true row cannot keep the
    segment alive.  When no row exists yet, the metadata noun seeds the rejection.
    Malformed rows and target-less segments raise rather than recording a partial or
    ineffective rejection.
    """

    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reject reason must be a non-empty string")
    reason = reason.strip()
    segment_dir = Path(segment_dir)
    transaction = _read_reject_transaction(segment_dir)
    if transaction is not None and transaction["reason"] != reason:
        raise RuntimeError(
            f"cannot reject {segment_dir.name}: an interrupted rejection has a "
            "different reason"
        )

    if requested_object is not None:
        targets = (str(requested_object).strip(),)
        if transaction is not None and list(targets) != transaction["objects"]:
            raise RuntimeError(
                f"cannot reject {segment_dir.name}: an interrupted rejection has "
                "different objects"
            )
    else:
        rows, row_reasons = _result_rows(segment_dir)
        if row_reasons:
            raise RuntimeError(
                f"cannot reject {segment_dir.name}: " + "; ".join(row_reasons)
            )
        targets = (
            tuple(transaction["objects"])
            if transaction is not None
            else tuple(
                sorted(name for name, (_, valid) in rows.items() if valid is True)
            )
        )
        if (
            not targets
            and transaction is None
            and rows
            and all(valid is False for _, valid in rows.values())
        ):
            # A retry after an already-completed rejection is still a coherent
            # segment-wide operation. Re-stamp every readable false row rather
            # than failing merely because there is no true row left to flip.
            targets = tuple(sorted(rows))
        if not targets and transaction is None:
            resolution = resolve_action_validation(segment_dir)
            targets = (
                (resolution.object_name,) if resolution.object_name is not None else ()
            )
    if not targets or any(not target for target in targets):
        raise RuntimeError(
            f"cannot reject {segment_dir.name}: no unambiguous action object"
        )

    payloads: dict[str, dict] = {}
    for object_name in targets:
        result_path = segment_dir / "valid_action" / object_name / "result.json"
        if result_path.exists():
            try:
                payload = json.loads(result_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"cannot reject {segment_dir.name}: unreadable {result_path}"
                ) from exc
            if not isinstance(payload, dict):
                raise RuntimeError(
                    f"cannot reject {segment_dir.name}: {result_path} is not an object"
                )
        else:
            payload = {"object": object_name}
        payload["valid"] = False
        payload["reject_reason"] = reason
        payloads[object_name] = payload

    marker = segment_dir / "valid_action" / REJECT_TRANSACTION_MARKER
    if transaction is None:
        _atomic_write(
            marker,
            json.dumps(
                {
                    "format": REJECT_TRANSACTION_FORMAT,
                    "objects": list(payloads),
                    "reason": reason,
                },
                indent=2,
            )
            + "\n",
        )
    for object_name, payload in payloads.items():
        object_dir = segment_dir / "valid_action" / object_name
        _atomic_write(object_dir / "result.json", json.dumps(payload, indent=2) + "\n")
        _atomic_write(object_dir / ".done", "done\n")
    marker.unlink()
    _fsync_dir(marker.parent)
    return tuple(payloads)


def resume_action_validation_rejection(
    segment_dir: str | Path,
) -> tuple[str, ...] | None:
    """Finish an interrupted rejection, or return ``None`` when none is pending.

    Resolution deliberately treats a pending transaction as unreadable and fails
    closed.  Pipeline retries must therefore replay the transaction before using
    validity as a stage filter; otherwise the rejected segment can remain in an
    infinite incomplete/retry loop.
    """

    segment_dir = Path(segment_dir)
    transaction = _read_reject_transaction(segment_dir)
    if transaction is None:
        return None
    return reject_action_validation(segment_dir, transaction["reason"])
