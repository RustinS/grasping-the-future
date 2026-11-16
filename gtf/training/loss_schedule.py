"""Step-based loss weight scheduler for GTF training.

Per-loss linear ramps keyed by `lambda_*` names (matching `cfg.loss.lambda_*`). Weights without a ramp keep
their static value from `cfg.loss`. Weights with a ramp interpolate linearly between ``start_val`` and
``end_val`` over ``[start_step, end_step]``, clamped outside that range.

Usage:

    schedule = LossSchedule(
        ramps={"lambda_aux_contact": LossRampConfig(0, 5000, 1e-9, 0.1)},
        static_values={"lambda_aux_contact": 0.1},
    )
    current = schedule(global_step)  # -> dict[str, float]
"""

from gtf.config import LossRampConfig


def _linear_interp(step: int, ramp: LossRampConfig) -> float:
    """Compute the linearly-interpolated value at ``step``."""
    if step <= ramp.start_step:
        return float(ramp.start_val)
    if step >= ramp.end_step:
        return float(ramp.end_val)
    span = max(1, ramp.end_step - ramp.start_step)
    frac = (step - ramp.start_step) / span
    return float(ramp.start_val + frac * (ramp.end_val - ramp.start_val))


class LossSchedule:
    """Per-loss linear-ramp scheduler keyed by ``lambda_*`` names.

    Pass an empty ``ramps`` dict to disable scheduling (returns static values).
    """

    def __init__(self, ramps: dict[str, LossRampConfig], static_values: dict[str, float]) -> None:
        self.ramps = dict(ramps)
        self.static = {k: float(v) for k, v in static_values.items()}

    def __call__(self, global_step: int) -> dict[str, float]:
        """Return current per-loss weights at ``global_step``."""
        out = dict(self.static)
        for key, ramp in self.ramps.items():
            out[key] = _linear_interp(int(global_step), ramp)
        return out
