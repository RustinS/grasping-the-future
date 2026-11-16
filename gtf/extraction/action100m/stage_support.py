"""Shared pieces of the Action100M pipeline steps: directories, atomic writes, stage queues, staleness checks.

The step modules import RESULTS_DIR from here by name. To run the pipeline on another results directory, set
ACTION100M_RESULTS_DIR before the first import (the orchestrator's subprocesses inherit it).
"""

import json
import math
import os
import tempfile
import zipfile
from pathlib import Path
import numpy as np
from gtf.utils import RESULTS_DIR as _SHARED_RESULTS_DIR, SEGMENTS_DIR as _SHARED_SEGMENTS_DIR
from gtf.extraction import chains

RESULTS_DIR = Path(_SHARED_RESULTS_DIR)
SEGMENTS_DIR = Path(_SHARED_SEGMENTS_DIR)
MAX_OPTIMIZED_CAMERA_ABS_M = 20.0


def _write_text_atomic(path: str | Path, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_json_atomic(path: str | Path, payload: dict) -> None:
    _write_text_atomic(
        path,
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
    )


def _optimized_output_complete(path: str | Path) -> bool:
    """Cheap structural check of an optimized.npz (optimizer or stage-2 output): the required arrays exist, the
    vertices are finite and within MAX_OPTIMIZED_CAMERA_ABS_M metres of the camera, and the stored losses are finite."""

    try:
        with np.load(path, allow_pickle=True) as archive:
            required = {
                "frame_indices",
                "obj_verts",
                "canonical_verts",
                "obj_faces",
            }
            if not required.issubset(archive.files) or archive["frame_indices"].size == 0:
                return False
            for name in ("obj_verts", "canonical_verts"):
                value = np.asarray(archive[name])
                if value.size == 0 or not np.isfinite(value).all():
                    return False
            camera_arrays = ["obj_verts"] + [name for name in archive.files if name.startswith("hand_") and name.endswith("_verts")]
            for name in camera_arrays:
                value = np.asarray(archive[name])
                if not np.isfinite(value).all():
                    return False
                if value.size and float(np.abs(value.astype(np.float64, copy=False)).max()) > MAX_OPTIMIZED_CAMERA_ABS_M:
                    return False
            if "optimization_losses_json" not in archive.files:
                return True
            losses = json.loads(
                str(archive["optimization_losses_json"].item()),
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON constant {value}")),
            )
            return all(not isinstance(value, (int, float)) or math.isfinite(value) for value in losses.values())
    except (OSError, ValueError, TypeError, EOFError, zipfile.BadZipFile):
        return False


def _stage_work_items(stage, names, args):
    """The segments this process works on: all of `names`, or the ones it claims from the split driver's stage queue."""
    queue_db = getattr(args, "stage_queue_db", None)
    if not queue_db:
        return iter(names)
    from gtf.extraction.action100m.stage_queue import iter_queued_stage_names

    worker = getattr(args, "stage_worker_id", None) or f"pid-{os.getpid()}"
    return iter_queued_stage_names(queue_db, stage, worker, names)


def _raise_queued_stage_failure(args, stage, name, detail):
    """Keep a failed queue item unacknowledged so the split driver can retry it.

    ``iter_queued_stage_names`` acknowledges the item it yielded last only when
    its consumer asks for the next one. Stage loops catch per-segment exceptions so
    a multi-segment run can continue; in queue mode that would acknowledge the
    failed item. Raising here instead ends the worker, the row stays ``running``,
    and ``run_split_extraction.sh`` retries it in a fresh process.
    """

    if getattr(args, "stage_queue_db", None):
        raise RuntimeError(f"{stage} queue item {name} failed: {detail}")


def _cuda_context_is_unusable(error):
    """Return whether continuing in this process can corrupt later GPU work."""

    messages = []
    current = error
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(str(current).lower())
        current = current.__cause__ or current.__context__
    detail = "\n".join(messages)
    fatal_fragments = (
        "device-side assert",
        "an illegal memory access was encountered",
        "cudaerrorillegaladdress",
        "cuda_error_illegal_address",
        "unspecified launch failure",
        "cuda error 700",
        "cuda error 710",
        "context is destroyed",
    )
    return any(fragment in detail for fragment in fatal_fragments)


def _drain_stage_queue(stage, args):
    """Acknowledge every queued item of a stage that has nothing to do (all segments complete)."""
    if getattr(args, "stage_queue_db", None):
        for _ in _stage_work_items(stage, (), args):
            pass


def stage2_subdir(name):
    """Input subdir for export/quality-gate: the stage-2 refined trajectory when present
    (gtf/extraction/stage2/refine_trajectory.py), else the raw optimizer output."""
    return "optimized_stage2" if (RESULTS_DIR / name / "optimized_stage2" / "optimized.npz").exists() else "optimized_mano"


def _quality_precondition_rejected(name):
    """Whether refine rejected the segment for too few keyframes (a current quality/precondition_reject.json).

    Malformed or stale records raise rather than being treated as absent.
    """

    from gtf.utils.action100m_quality_precondition import (
        load_quality_precondition_reject,
    )

    return (
        load_quality_precondition_reject(
            RESULTS_DIR / name,
            expected_segment=name,
        )
        is not None
    )


def _stale_vs_upstream(out_path, *upstream_paths):
    """True when *out_path* exists but an upstream output is newer: re-running a stage invalidates
    the stages below it (mtime sentinel)."""
    out_path = Path(out_path)
    if not out_path.exists():
        return False
    out_m = out_path.stat().st_mtime
    for up in upstream_paths:
        up = Path(up)
        if up.exists() and up.stat().st_mtime > out_m:
            return True
    return False


def _claim_chain(segment_names, args):
    """The run's chain, after binding every segment directory to it (chains.claim; raises on a chain mix)."""
    chain = chains.get(getattr(args, "chain", None))
    for name in segment_names:
        chains.claim(RESULTS_DIR / name, chain)
    return chain


def _expand_declared_interaction_window_children(segment_names):
    """Add the `__iwN` children each segment's committed canonical export declares."""

    from gtf.extraction.export.export_canonical import declared_interaction_window_children

    expanded = []
    for name in segment_names:
        if name not in expanded:
            expanded.append(name)
        canonical_path = RESULTS_DIR / name / "canonical" / "canonical.npz"
        if not canonical_path.is_file():
            continue
        try:
            children = declared_interaction_window_children(RESULTS_DIR / name, expected_segment=name)
        except ValueError as exc:
            raise RuntimeError(f"refusing stale interaction-window discovery for {name}: {exc}") from exc
        for child in children:
            if child not in expanded:
                expanded.append(child)
    return expanded
