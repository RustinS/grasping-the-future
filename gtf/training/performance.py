"""Low-overhead wall-clock training throughput.

Measurements are between rank-zero optimizer-step completion hooks, not CUDA
kernel timings or a distributed maximum. They include intervening input waits,
checkpoint/logging overhead and validation, so they describe effective training
throughput. No CUDA synchronization, tensor readback, or collective is added.
"""

import math
import time
import warnings

import lightning.pytorch as pl


class TrainingSpeedMonitor(pl.Callback):
    """Send interval-average speed to every configured logger (including W&B).

    Use Lightning's optimizer ``global_step``, not the microbatch index. Actual
    rank-zero window counts are accumulated across microbatches and multiplied
    by world size; global windows/sec assumes equal per-rank batch sizes, as
    with the training sampler. It counts sampled windows, not unique
    windows, frames, or applicable loss terms.

    The first completed update after every start/resume establishes the timing
    baseline, excluding restore/setup and first-batch warm-up. Nothing is saved
    in checkpoints: wall-clock gaps while preempted must not become step time.
    """

    def __init__(self, log_every: int = 50) -> None:
        if log_every < 1:
            raise ValueError("Speed logging interval must be positive")
        self.log_every = int(log_every)
        self._warned_loggers: set[int] = set()
        self._reset()

    def _reset(self) -> None:
        self._last_step = 0
        self._last_end: float | None = None
        self._pending_windows = 0
        self._pending_known = True
        self._clear_interval()

    def _clear_interval(self) -> None:
        self._seconds = 0.0
        self._steps = 0
        self._windows = 0
        self._windows_known = True
        self._max_step_seconds = 0.0

    def on_train_start(self, trainer, pl_module) -> None:
        self._reset()
        self._last_step = int(trainer.global_step)

    @staticmethod
    def _window_count(batch) -> int | None:
        # shape metadata only: no .item(), device transfer or GPU synchronization.
        if isinstance(batch, dict):
            state = batch.get("left_state")
            shape = getattr(state, "shape", ())
            if len(shape) > 0:
                return int(shape[0])
        return None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if not trainer.is_global_zero or not trainer.loggers:
            return
        if outputs is not None:
            count = self._window_count(batch)
            self._pending_known &= count is not None
            self._pending_windows += count or 0

        step = int(trainer.global_step)
        if step <= self._last_step:
            return  # Accumulation microbatch, not a completed optimizer update.
        now = time.perf_counter()
        if self._last_end is not None:
            seconds = now - self._last_end
            updates = step - self._last_step
            if math.isfinite(seconds) and seconds > 0:
                self._seconds += seconds
                self._steps += updates
                self._windows += self._pending_windows * int(trainer.world_size)
                self._windows_known &= self._pending_known
                self._max_step_seconds = max(self._max_step_seconds, seconds / updates)
        self._last_step, self._last_end = step, now
        self._pending_windows, self._pending_known = 0, True
        if self._steps >= self.log_every or (trainer.max_steps > 0 and step >= trainer.max_steps):
            self._emit(trainer)

    def _emit(self, trainer) -> None:
        if not self._steps or self._seconds <= 0:
            return
        metrics = {
            "perf/step_time_s": self._seconds / self._steps,
            "perf/steps_per_second": self._steps / self._seconds,
            "perf/step_time_max_s": self._max_step_seconds,
        }
        if self._windows_known:
            metrics["perf/windows_per_second"] = self._windows / self._seconds
        for logger in trainer.loggers:
            try:
                # Lightning's WandbLogger maps this to trainer/global_step;
                # no wandb.init(), new run, or competing W&B step clock.
                logger.log_metrics(metrics, step=int(trainer.global_step))
            except Exception as error:
                # Observability must not kill a multi-node training run.
                key = id(logger)
                if key not in self._warned_loggers:
                    warnings.warn(f"Speed metrics skipped for {type(logger).__name__}: {error}", RuntimeWarning)
                    self._warned_loggers.add(key)
        self._clear_interval()

    def on_train_end(self, trainer, pl_module) -> None:
        if trainer.is_global_zero and trainer.loggers:
            # Flush partial reporting intervals without counting final teardown
            # or inventing another optimizer update.
            self._emit(trainer)
