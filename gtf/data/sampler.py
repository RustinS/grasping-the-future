"""Weighted window sampling for GTF with DDP partitioning.

Provides:
  - ``compute_window_weights``: per-window weights from a ``GTFDataset``
    (flattened source mass, confidence-tier boost, left/right hand balance).
  - ``StepScheduledDistributedWeightedSampler``: weighted-with-replacement
    sampling whose Action100M share follows a schedule over optimizer steps,
    partitioned deterministically across DDP ranks.

Lightning calls ``set_epoch`` on custom samplers at each epoch start, so defining
``set_epoch`` here is sufficient.
"""

import logging
import math
from collections import Counter
from collections.abc import Iterator
from typing import Callable

import torch
from torch import Tensor
from torch.utils.data import Sampler

from gtf.data.curriculum import PiecewiseSchedule, apply_source_curriculum

DEFAULT_TIER_BOOST: dict[str, float] = {"gt_mocap": 3.0, "gt_fitted": 2.0, "gt_landmark_fitted": 1.5, "extracted": 1.0}


class StepScheduledDistributedWeightedSampler(Sampler[int]):
    """Optimizer-step-aware two-pool curriculum with exact DDP partitioning.

    A local optimizer step consumes ``batch_size * accumulate_grad_batches``
    samples per rank. For every such block, the requested mass of ``selected``
    (Action100M) is evaluated at the corresponding optimizer ``global_step``.
    All ranks construct one identical global draw (same seed) and rank ``r``
    takes ``r::num_replicas``. Duplicates within and across ranks are expected
    for weighted sampling with replacement.

    Pool membership is Bernoulli with the scheduled probability; indices inside
    each pool are drawn from the base weights restricted to that pool, so the
    confidence-tier, hand-balance and within-source ratios survive.

    ``weights`` holds the *expected whole-run exposure* (the schedule's discrete
    mean over ``exposure_steps``). Window-sampled normalization statistics use
    this vector; the iterator itself follows the per-step schedule.
    """

    def __init__(
        self,
        base_weights: Tensor | list[float],
        selected: Tensor | list[bool],
        schedule: PiecewiseSchedule,
        *,
        num_samples: int,
        num_replicas: int,
        rank: int,
        batch_size: int,
        accumulate_grad_batches: int,
        start_global_step: int = 0,
        start_epoch: int = 0,
        exposure_steps: int,
        global_step_getter: Callable[[], int] | None = None,
        seed: int = 0,
    ) -> None:
        if num_replicas < 1:
            raise ValueError(f"num_replicas must be >= 1, got {num_replicas}")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank {rank} must be in [0, {num_replicas})")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        if batch_size <= 0 or accumulate_grad_batches <= 0:
            raise ValueError(f"batch_size and accumulate_grad_batches must be positive, got {batch_size} and {accumulate_grad_batches}")
        if start_global_step < 0 or start_epoch < 0:
            raise ValueError("start_global_step and start_epoch must be nonnegative")
        if exposure_steps <= 0:
            raise ValueError(f"exposure_steps must be positive, got {exposure_steps}")
        if any(not 0.0 <= knot.value <= 1.0 for knot in schedule.knots):
            raise ValueError("all curriculum schedule masses must lie in [0,1]")

        self.base_weights = torch.as_tensor(base_weights, dtype=torch.double).clone()
        self.selected = torch.as_tensor(selected, dtype=torch.bool).clone()
        if self.base_weights.ndim != 1:
            raise ValueError(f"base_weights must be 1D, got shape {tuple(self.base_weights.shape)}")
        if self.selected.shape != self.base_weights.shape:
            raise ValueError(f"selected shape {tuple(self.selected.shape)} != weights shape {tuple(self.base_weights.shape)}")
        if not bool(torch.isfinite(self.base_weights).all()) or bool((self.base_weights < 0).any()):
            raise ValueError("base_weights must be finite and non-negative")
        selected_mass = float(self.base_weights[self.selected].sum())
        other_mass = float(self.base_weights[~self.selected].sum())
        needs_selected = any(k.value > 0.0 for k in schedule.knots)
        needs_other = any(k.value < 1.0 for k in schedule.knots)
        if needs_selected and selected_mass <= 0.0:
            raise ValueError("curriculum assigns positive mass to an empty selected pool")
        if needs_other and other_mass <= 0.0:
            raise ValueError("curriculum assigns positive mass to an empty complement pool")

        self.schedule = schedule
        self.num_samples = int(num_samples)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.batch_size = int(batch_size)
        self.accumulate_grad_batches = int(accumulate_grad_batches)
        self.samples_per_rank_step = self.batch_size * self.accumulate_grad_batches
        if self.num_samples % self.samples_per_rank_step != 0:
            raise ValueError(
                "num_samples must be a multiple of batch_size * accumulate_grad_batches "
                "so every scheduled optimizer-step block has identical window mass; "
                f"got {self.num_samples} % {self.samples_per_rank_step}"
            )
        self.steps_per_epoch = math.ceil(self.num_samples / self.samples_per_rank_step)
        self.start_global_step = int(start_global_step)
        self.start_epoch = int(start_epoch)
        self.global_step_getter = global_step_getter
        self.exposure_steps = int(exposure_steps)
        self.seed = int(seed)
        self.epoch = self.start_epoch

        # Expected exposure over the whole run. A mixture of fixed conditional
        # pool distributions is linear in the schedule mass, so the discrete mean
        # is exact.
        mean_mass = self.schedule.mean_over_steps(self.exposure_steps)
        self.weights = apply_source_curriculum(
            self.base_weights,
            is_extracted=self.selected,
            extracted_mass=mean_mass,
        )

    def _epoch_start_step(self) -> int:
        if self.global_step_getter is not None:
            step = int(self.global_step_getter())
            if step < 0:
                raise RuntimeError(f"global_step_getter returned a negative step: {step}")
            return step
        return self.start_global_step + (self.epoch - self.start_epoch) * self.steps_per_epoch

    def _pool_probabilities(self) -> Tensor:
        """Scheduled selected-pool probability for each global draw position."""

        epoch_start = self._epoch_start_step()
        n_steps = math.ceil(self.num_samples / self.samples_per_rank_step)
        by_step = torch.tensor(
            [self.schedule.value_at(epoch_start + offset) for offset in range(n_steps)],
            dtype=torch.double,
        )
        by_local = by_step.repeat_interleave(self.samples_per_rank_step)[: self.num_samples]
        # Global draw ordering is [local_position, rank]. Partitioning the result
        # by rank::world therefore gives every rank the same optimizer-step mix.
        return by_local.repeat_interleave(self.num_replicas)

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        probabilities = self._pool_probabilities()
        selected_draw = torch.rand(probabilities.numel(), generator=g, dtype=torch.double) < probabilities
        all_indices = torch.empty(probabilities.numel(), dtype=torch.long)

        for take_selected in (True, False):
            slots = (selected_draw == take_selected).nonzero(as_tuple=True)[0]
            if slots.numel() == 0:
                continue
            pool = self.selected if take_selected else ~self.selected
            pool_indices = (pool & (self.base_weights > 0)).nonzero(as_tuple=True)[0]
            if pool_indices.numel() == 0:
                raise RuntimeError("curriculum drew from a pool with zero positive-weight windows")
            choices = torch.multinomial(
                self.base_weights[pool_indices],
                int(slots.numel()),
                replacement=True,
                generator=g,
            )
            all_indices[slots] = pool_indices[choices]

        rank_indices = all_indices[self.rank :: self.num_replicas]
        return iter(rank_indices.tolist())

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set the Lightning epoch; the schedule stays aligned to the optimizer step."""

        self.epoch = int(epoch)


def compute_window_weights(
    dataset,
    alpha: float = 0.25,
    tier_boost: dict[str, float] | None = None,
    hand_balance_target: float = 0.5,
    hand_balance_max_boost: float = 10.0,
) -> Tensor:
    """Compute per-window sampling weights from a GTFDataset.

    Per-window weight::

        source_mass[src] = N_windows_in_src ** alpha
        source_mass_norm = source_mass / sum(source_mass)
        weight_i = (source_mass_norm[src_i] / N_windows_in_src[src_i]) * tier_boost[tier_i]

    This flattens the share of large sources (alpha < 1) and boosts curated
    tiers. Counts are windows, not segments.

    With ``hand_balance_target > 0``, left-hand-present windows are boosted so
    that the expected sampled left-valid frames reach the target share of all
    (left + right) valid frames. Boosting a left-present window multiplies both
    its left and its right frame mass, so with L / R_l / R_o = left mass / right
    mass inside left-present windows / right mass elsewhere, the boosted share
    is b*L / (b*L + b*R_l + R_o) and the exact solve is
    ``b = p * R_o / (L * (1 - p) - p * R_l)``. When the target is unreachable
    (denominator <= 0, or R_o == 0) or the solve exceeds
    ``hand_balance_max_boost``, the boost is capped and the shortfall logged.

    Args:
        dataset: Object with ``.windows`` list[(seg_idx, start, end)], ``.segments``
            list[dict] exposing ``source_dataset`` and ``confidence_tier``, and the
            per-window ``_window_left_valid_count`` / ``_window_right_valid_count``.
        alpha: Source mass exponent (1.0 = proportional, 0.0 = uniform per source).
        tier_boost: {tier_name: multiplier}. Default: DEFAULT_TIER_BOOST.
        hand_balance_target: Target L/(L+R) expected valid-frame ratio (0 = no balancing).
        hand_balance_max_boost: Cap on the left-window boost.

    Returns:
        (N,) double tensor of unnormalized per-window weights.
    """
    if tier_boost is None:
        tier_boost = DEFAULT_TIER_BOOST

    source_per_window: list[str] = []
    tier_per_window: list[str] = []
    for seg_idx, _start, _end in dataset.windows:
        entry = dataset.segments[seg_idx]
        source_per_window.append(str(entry.get("source_dataset", "unknown")))
        tier_per_window.append(str(entry.get("confidence_tier", "extracted")))

    n_per_source = Counter(source_per_window)

    # Source mass, flattened and normalized.
    source_mass = {src: (count**alpha) for src, count in n_per_source.items()}
    total_mass = sum(source_mass.values())
    if total_mass <= 0:
        raise ValueError("Total source mass is zero; check dataset contents.")
    source_mass_norm = {src: m / total_mass for src, m in source_mass.items()}

    weights = torch.empty(len(source_per_window), dtype=torch.double)
    for i, (src, tier) in enumerate(zip(source_per_window, tier_per_window)):
        per_window_source = source_mass_norm[src] / n_per_source[src]
        boost = tier_boost.get(tier, 1.0)
        weights[i] = per_window_source * boost

    # Frame-level left/right hand balance.
    if hand_balance_target > 0:
        left_counts = torch.tensor(dataset._window_left_valid_count, dtype=torch.double)
        right_counts = torch.tensor(dataset._window_right_valid_count, dtype=torch.double)

        expected_L = (weights * left_counts).sum().item()
        expected_R = (weights * right_counts).sum().item()

        if expected_L > 0 and expected_R > 0:
            r = expected_L / (expected_L + expected_R)
            p = float(hand_balance_target)
            _cap = max(1.0, float(hand_balance_max_boost))
            _note = ""
            if r < p:
                left_mask = left_counts > 0
                # All left mass lives in left-present windows, so L == expected_L.
                L = expected_L
                R_l = (weights[left_mask] * right_counts[left_mask]).sum().item()
                R_o = expected_R - R_l
                denom = L * (1.0 - p) - p * R_l
                if denom > 0.0 and R_o > 0.0:
                    boost_exact = p * R_o / denom
                    boost_factor = min(boost_exact, _cap)
                    if boost_factor < boost_exact:
                        _note = f" (exact solve {boost_exact:.2f} capped at {_cap:.1f} — target not reached)"
                else:
                    # Asymptote L/(L+R_l) <= p (or R_o == 0: ratio boost-invariant):
                    # no finite uniform boost of left-present windows reaches p.
                    boost_factor = _cap if R_o > 0.0 else 1.0
                    _note = f" (target unreachable: asymptote {L / (L + R_l):.3f} <= target — capped at {boost_factor:.1f})"
                weights[left_mask] *= boost_factor
            else:
                boost_factor = 1.0

            new_expected_L = (weights * left_counts).sum().item()
            new_expected_R = (weights * right_counts).sum().item()
            new_r = new_expected_L / (new_expected_L + new_expected_R) if (new_expected_L + new_expected_R) > 0 else 0.0
            logging.getLogger("sampler").info(
                f"hand_balance: target_frame_ratio={p:.2f}, before={r:.3f}, after={new_r:.3f}, "
                f"boost={boost_factor:.2f}, left_windows={int((left_counts > 0).sum())}/{len(left_counts)}{_note}"
            )

    return weights
