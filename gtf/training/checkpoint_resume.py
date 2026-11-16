"""Resume only complete training state, ordered by progress rather than file age.

The checks assume Lightning automatic optimization and a per-step LR scheduler.
"""

import logging
import math
from pathlib import Path
import re

import torch

_LOG = logging.getLogger(__name__)
_STEP = re.compile(r"(?:^|[-_])step[=_-]?(\d+)")


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"invalid {name}: {value!r}")
    return value


def validate_checkpoint_payload(checkpoint):
    """Return completed optimizer steps; reject incomplete or inconsistent state."""
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint is not a dictionary")
    step = _integer(checkpoint.get("global_step"), "global_step", 1)
    epoch = _integer(checkpoint.get("epoch"), "epoch")
    if not isinstance(checkpoint.get("state_dict"), dict) or not checkpoint["state_dict"]:
        raise ValueError("missing model state")
    optimizers = checkpoint.get("optimizer_states")
    if not isinstance(optimizers, list) or not optimizers:
        raise ValueError("missing optimizer state")
    for optimizer in optimizers:
        if not isinstance(optimizer, dict) or not optimizer.get("state") or not optimizer.get("param_groups"):
            raise ValueError("empty optimizer state or parameter groups")
        state_steps = []
        for state in optimizer["state"].values():
            if "step" not in state:
                continue  # E.g. SGD momentum state has no per-parameter clock.
            counter = state["step"]
            if isinstance(counter, torch.Tensor):
                if counter.numel() != 1:
                    raise ValueError("non-scalar optimizer step")
                counter = counter.item()
            if not isinstance(counter, (int, float)) or not math.isfinite(counter) or int(counter) != counter:
                raise ValueError("invalid optimizer step")
            if counter < 0 or counter > step:
                raise ValueError("partially completed optimizer update")
            state_steps.append(int(counter))
        if state_steps and max(state_steps) != step:
            raise ValueError("optimizer state lags completed training progress")
    schedulers = checkpoint.get("lr_schedulers")
    if not isinstance(schedulers, list) or not schedulers:
        raise ValueError("missing scheduler state")
    for scheduler in schedulers:
        if not isinstance(scheduler, dict) or scheduler.get("last_epoch") != step:
            raise ValueError("per-step scheduler disagrees with global_step")
    try:
        loop = checkpoint["loops"]["fit_loop"]
        completed = loop["epoch_loop.automatic_optimization.optim_progress"]["optimizer"]["step"]["total"]["completed"]
        completed_epoch = loop["epoch_progress"]["current"]["completed"]
    except (KeyError, TypeError) as exc:
        raise ValueError("missing automatic-optimization progress") from exc
    if completed != step or completed_epoch != epoch:
        raise ValueError("restored optimizer/epoch progress disagrees with checkpoint header")
    # Batch and _batches_that_stepped counters may legitimately lag by one in
    # ModelCheckpoint.on_train_batch_end. Do not compare those to global_step.
    return step


def checkpoint_step(path):
    """Read metadata without loading 1+ GB of parameter/optimizer storage into RAM."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError as exc:
        if "mmap can only be used with files saved with" not in str(exc):
            raise
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return validate_checkpoint_payload(checkpoint)


def resolve_resume_path(resume, checkpoint_dir):
    """Choose the highest validated step, including versioned Lightning saves.

    Unknown-name aliases are inspected first. Numeric snapshot names are then
    inspected in descending order; the lower ones cannot improve the choice and
    need not all be deserialized on every DDP rank. A numeric filename must agree
    with its payload. Existing-but-unusable files never silently start a new run.
    """
    if resume is None or resume in ("null", "None"):
        return None
    if resume != "auto":
        path = Path(checkpoint_dir) / "last.ckpt" if resume == "last" else Path(resume)
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        checkpoint_step(path)
        return str(path)

    candidates = list(Path(checkpoint_dir).glob("*.ckpt"))
    if not candidates:
        return None

    def hint(path):
        match = _STEP.search(path.name)
        return int(match[1]) if match else None

    # Unknown names first, followed by greatest filename step. The content, not
    # this hint or the mtime, is the authority for the final choice.
    candidates.sort(key=lambda p: (hint(p) is None, hint(p) or -1), reverse=True)
    best = None
    rejected = []
    for path in candidates:
        named_step = hint(path)
        if best is not None and named_step is not None and named_step < best[0]:
            break
        try:
            step = checkpoint_step(path)
            if named_step is not None and step != named_step:
                raise ValueError(f"filename step {named_step} != payload step {step}")
            candidate = (step, path.stat().st_mtime_ns, str(path))
            if best is None or candidate > best:
                best = candidate
        except Exception as exc:
            rejected.append(f"{path.name}: {type(exc).__name__}: {exc}")
            _LOG.warning("Ignoring unusable resume checkpoint %s: %s", path, exc)
    if best is None:
        raise ValueError("No complete training checkpoint available; refusing a fresh start. " + "; ".join(rejected))
    _LOG.info("Resume selected step %d: %s", best[0], best[2])
    return best[2]
