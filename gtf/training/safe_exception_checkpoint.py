"""Best-effort exception snapshots for replicated DDP (or single-device) training.

Unlike ``Trainer.save_checkpoint``, this exceptional path deliberately has no
barrier: another rank may have failed before it reached the same callback.
Only rank zero dumps and writes its complete, locally replicated training state.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import uuid

from lightning.pytorch.callbacks import OnExceptionCheckpoint
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy

from gtf.training.checkpoint_resume import validate_checkpoint_payload


log = logging.getLogger(__name__)


class SafeOnExceptionCheckpoint(OnExceptionCheckpoint):
    """Preserve the previous rescue point unless a new complete step is saved.

    The armed flag is process-local and is never restored from checkpoint state.
    ``on_train_start`` runs after model, loop, optimizer, and scheduler restoration;
    ``setup`` and ``on_fit_start`` are too early for this purpose.

    Only the replicated DDP and single-device strategies are supported. Sharded
    strategies require a different, collective-aware save protocol.
    """

    def __init__(self, dirpath: str | Path, filename: str = "on_exception") -> None:
        super().__init__(dirpath=dirpath, filename=filename)
        self._train_started = False
        self._restored_step: int | None = None

    def on_train_start(self, trainer, pl_module) -> None:
        self._restored_step = int(trainer.global_step)
        self._train_started = True

    def on_exception(self, trainer, pl_module, exception: BaseException) -> None:
        # Do not introduce a collective, including on the non-writing ranks.
        if not trainer.is_global_zero:
            return
        if not isinstance(trainer.strategy, (DDPStrategy, SingleDeviceStrategy)):
            log.warning("Skipping exception checkpoint: strategy is not replicated DDP/single-device")
            return
        if not self._train_started or self._restored_step is None:
            log.warning("Skipping exception checkpoint: training state was not fully restored")
            return

        temporary: Path | None = None
        try:
            step = int(trainer.global_step)
            if step <= self._restored_step:
                log.warning("Skipping exception checkpoint: no new completed optimizer step")
                return

            # For the supported strategies, these state-dump hooks are local. Validate before any write so a partially completed
            # optimizer/scheduler update cannot replace the previous snapshot.
            checkpoint = trainer._checkpoint_connector.dump_checkpoint(weights_only=False)
            validated_step = validate_checkpoint_payload(checkpoint)
            if validated_step != step:
                raise ValueError(f"snapshot step {validated_step} != runtime step {step}")

            target = Path(self.ckpt_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
            # Strategy.save_checkpoint has no barrier for the supported
            # strategies; Trainer.save_checkpoint has one and must not be used.
            trainer.strategy.save_checkpoint(checkpoint, str(temporary))
            os.replace(temporary, target)
            log.warning("Saved validated exception checkpoint at step %s: %s", step, target)
        except Exception:
            # Never mask the original training failure or overwrite a prior
            # rescue point with an incomplete/failed save.
            log.warning("Exception checkpoint skipped; previous checkpoint preserved", exc_info=True)
        finally:
            self._train_started = False
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    log.warning("Could not remove incomplete checkpoint temporary: %s", temporary)

    def teardown(self, trainer, pl_module, stage: str) -> None:
        # Stock OnExceptionCheckpoint deletes its file here. Retain validated
        # rescue points; the progress-based resolver can safely rank them.
        self._train_started = False
