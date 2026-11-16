"""Deterministic source curriculum for weighted window sampling.

The window sampler already encodes the within-source, confidence-tier and
hand-balance policy. The curriculum does not replace those relative weights: it
only assigns a requested share of the sampling mass to one coarse pool
(Action100M versus the curated MoCap sources), keeping every weight ratio inside
each pool.

This module holds pure tensor operations and schedule evaluation. The training
integration passes the optimizer ``global_step``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class ScheduleKnot:
    """One optimizer-step/value control point."""

    step: int
    value: float

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError(f"schedule step must be nonnegative, got {self.step}")
        if not math.isfinite(self.value):
            raise ValueError(f"schedule value must be finite, got {self.value}")


@dataclass(frozen=True)
class PiecewiseSchedule:
    """Piecewise-linear schedule evaluated in optimizer-step space."""

    knots: tuple[ScheduleKnot, ...]

    def __post_init__(self) -> None:
        if not self.knots:
            raise ValueError("a schedule needs at least one knot")
        steps = [k.step for k in self.knots]
        if steps != sorted(set(steps)):
            raise ValueError(f"schedule knot steps must be strictly increasing, got {steps}")

    @classmethod
    def from_pairs(cls, pairs: Sequence[tuple[int, float]]) -> "PiecewiseSchedule":
        return cls(tuple(ScheduleKnot(int(step), float(value)) for step, value in pairs))

    def value_at(self, step: int) -> float:
        """Return the clamped schedule value at an optimizer step."""

        step = max(0, int(step))
        if step <= self.knots[0].step:
            return self.knots[0].value
        if step >= self.knots[-1].step:
            return self.knots[-1].value
        for left, right in zip(self.knots[:-1], self.knots[1:], strict=True):
            if left.step <= step <= right.step:
                u = (step - left.step) / float(right.step - left.step)
                return left.value + u * (right.value - left.value)
        raise RuntimeError("schedule interval lookup failed")

    def mean_over_steps(self, n_steps: int) -> float:
        """Exact discrete mean over steps ``[0, n_steps)``.

        This is the whole-run exposure used by the window-sampled normalization
        statistics.
        """

        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")
        return sum(self.value_at(step) for step in range(int(n_steps))) / float(n_steps)


def _as_bool_mask(mask: Tensor | Sequence[bool], n: int, *, name: str) -> Tensor:
    out = torch.as_tensor(mask, dtype=torch.bool)
    if out.shape != (n,):
        raise ValueError(f"{name} must have shape {(n,)}, got {tuple(out.shape)}")
    return out


def rescale_binary_mass(
    weights: Tensor,
    selected: Tensor | Sequence[bool],
    target_mass: float,
    *,
    name: str = "selected",
) -> Tensor:
    """Set one pool's mass while preserving all within-pool weight ratios."""

    w = torch.as_tensor(weights, dtype=torch.double).clone()
    if w.ndim != 1 or not bool(torch.isfinite(w).all()) or bool((w < 0).any()):
        raise ValueError("weights must be a finite nonnegative 1-D tensor")
    if not 0.0 <= float(target_mass) <= 1.0:
        raise ValueError(f"target_mass must be in [0,1], got {target_mass}")
    m = _as_bool_mask(selected, w.numel(), name=name)
    selected_mass = w[m].sum()
    other_mass = w[~m].sum()
    target_mass = float(target_mass)

    if target_mass == 0.0:
        w[m] = 0.0
        return w
    if target_mass == 1.0:
        w[~m] = 0.0
        return w
    if float(selected_mass) <= 0.0 or float(other_mass) <= 0.0:
        raise ValueError(f"cannot assign {name} mass {target_mass:.4f}: selected={float(selected_mass):.6g}, other={float(other_mass):.6g}")

    # Scale selected relative to the complement. torch.multinomial does not need
    # normalized weights, and skipping the normalization keeps more precision.
    scale = (target_mass / (1.0 - target_mass)) * (other_mass / selected_mass)
    w[m] *= scale
    return w


def apply_source_curriculum(
    base_weights: Tensor,
    *,
    is_extracted: Tensor | Sequence[bool],
    extracted_mass: float,
) -> Tensor:
    """Give the extracted (Action100M) pool ``extracted_mass`` of the total mass."""

    w = torch.as_tensor(base_weights, dtype=torch.double)
    ext = _as_bool_mask(is_extracted, w.numel(), name="is_extracted")
    return rescale_binary_mass(w, ext, extracted_mass, name="extracted")


def curriculum_mass_report(weights: Tensor, *, is_extracted: Tensor) -> dict[str, float]:
    """Small logging helper for realized sampler mass."""

    w = torch.as_tensor(weights, dtype=torch.double)
    total = w.sum().clamp(min=torch.finfo(torch.double).tiny)
    ext = _as_bool_mask(is_extracted, w.numel(), name="is_extracted")
    return {
        "extracted": float(w[ext].sum() / total),
        "curated": float(w[~ext].sum() / total),
    }
