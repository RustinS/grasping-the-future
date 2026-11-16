"""Masking regime sampler for flow matching training.

Samples observation masks that define which tokens are observed (context)
vs masked (target) during training, with dependency-aware relation masking.

Token slot indices: 0=left_state, 1=right_state, 2=obj_state, 3=left_rel, 4=right_rel.

Three token states result from presence_mask and obs_mask:
- Present + observed (presence=True, obs=True): clean data, context in attention.
- Present + masked (presence=True, obs=False): noised, flow matching target.
- Absent (presence=False): zero-filled, invisible via -inf attention, no loss.
"""

import torch
from torch import Tensor

# "full_mask" (nothing observed) is the fallback for windows with fewer than two
# real frames, which cannot be split into context and target.
_REGIME_IDS = {
    "future_prediction": 0,
    "hand_to_object": 1,
    "object_to_hand": 2,
    "random_block": 3,
    "goal_infilling": 4,
    "temporal_infilling": 5,
    "full_mask": 6,
}


def sample_mask_batched(
    presence_mask: Tensor,
    valid_frames: Tensor,
    regime_probs: dict[str, float],
) -> Tensor:
    """Sample one observation mask per window of a batch.

    Each window draws a regime from ``regime_probs``:

    - ``future_prediction``: the first K frames are observed, K uniform in [1, T_real - 1].
    - ``hand_to_object``: hands observed everywhere, the object only at frame 0.
    - ``object_to_hand``: the object observed everywhere, hands only at frame 0.
    - ``random_block``: per state entity, with probability 0.5 one contiguous block is
      masked; frame 0 stays visible and at least one present token after frame 0 is masked.
    - ``goal_infilling``: a 1-5 frame prefix and 1-3 frame suffix are observed for all
      entities (50%), hands only (25%) or the object only (25%); frame 0 is fully visible.
    - ``temporal_infilling``: everything observed except one contiguous all-entity gap of
      20-50% of the window, strictly between frame 0 and the last frame.

    Windows with fewer than two real frames are fully masked; infilling regimes fall
    back to future prediction on windows with fewer than three. Random numbers are
    drawn in a fixed rectangular layout (independent of the sampled regimes) and no
    tensor values are read back to the host.

    Args:
        presence_mask: ``(B, T, 5)`` bool tensor. True where a token is present.
        valid_frames: ``(B, T)`` bool tensor. True for real frames, which form a prefix.
        regime_probs: Probability weights over the regime names above (sum ~1).

    Returns:
        ``(B, T, 5)`` bool observation mask. True means observed/context. A relation
        slot is observed iff its hand and the object are observed; absent tokens are
        never observed.
    """
    if presence_mask.ndim != 3 or presence_mask.shape[-1] != 5:
        raise ValueError(f"presence_mask must have shape (B, T, 5), got {tuple(presence_mask.shape)}")
    if valid_frames.shape != presence_mask.shape[:2]:
        raise ValueError(f"valid_frames must have shape {tuple(presence_mask.shape[:2])}, got {tuple(valid_frames.shape)}")
    if presence_mask.dtype != torch.bool or valid_frames.dtype != torch.bool:
        raise TypeError("presence_mask and valid_frames must be bool tensors")
    if presence_mask.device != valid_frames.device:
        raise ValueError("presence_mask and valid_frames must be on the same device")

    B, T, _ = presence_mask.shape
    device = presence_mask.device
    names, probs = _validate_regime_probs(regime_probs)

    regime_u = torch.rand(B, device=device)
    future_u = torch.rand(B, device=device)
    rb_apply_u = torch.rand(B, 3, device=device)
    rb_len_u = torch.rand(B, 3, device=device)
    rb_start_u = torch.rand(B, 3, device=device)
    rb_force_time_u = torch.rand(B, device=device)
    rb_force_entity_u = torch.rand(B, device=device)
    goal_prefix_u = torch.rand(B, device=device)
    goal_suffix_u = torch.rand(B, device=device)
    goal_entity_u = torch.rand(B, device=device)
    temporal_len_u = torch.rand(B, device=device)
    temporal_start_u = torch.rand(B, device=device)

    prob_tensor = torch.tensor(probs, dtype=torch.float32, device=device)
    prob_tensor = prob_tensor / prob_tensor.sum()
    boundaries = prob_tensor.cumsum(0)[:-1]
    sampled_indices = torch.searchsorted(boundaries, regime_u, right=True)
    code_lookup = torch.tensor([_REGIME_IDS[name] for name in names], dtype=torch.long, device=device)
    regime = code_lookup[sampled_indices]

    t_real = valid_frames.sum(dim=1, dtype=torch.long)
    full_id = _REGIME_IDS["full_mask"]
    future_id = _REGIME_IDS["future_prediction"]
    regime = torch.where((t_real < 2) & (regime != full_id), full_id, regime)
    infill = (regime == _REGIME_IDS["goal_infilling"]) | (regime == _REGIME_IDS["temporal_infilling"])
    regime = torch.where((t_real < 3) & infill, future_id, regime)

    time = torch.arange(T, device=device).view(1, T, 1)
    slot = torch.arange(3, device=device).view(1, 1, 3)
    real = time < t_real.view(B, 1, 1)

    # future_prediction: K is uniform over [1, T_real - 1].
    future_choices = (t_real - 1).clamp_min(1)
    future_k = 1 + _uniform_index(future_u, future_choices)
    future = real & (time < future_k.view(B, 1, 1))

    # Entity-conditioned regimes reveal the other entity at frame 0.
    hand_to_object = real & ((slot < 2) | (time == 0))
    object_to_hand = real & ((slot == 2) | (time == 0))

    random_block = _sample_random_blocks_batched(
        presence_mask=presence_mask[:, :, :3],
        real=real,
        t_real=t_real,
        apply_u=rb_apply_u,
        length_u=rb_len_u,
        start_u=rb_start_u,
        force_time_u=rb_force_time_u,
        force_entity_u=rb_force_entity_u,
    )

    # goal_infilling: prefix in [1, min(6, T) - 1], suffix in [1, max(min(4, T - prefix), 2) - 1],
    # shortened so that at least one frame stays masked.
    prefix_choices = (torch.minimum(t_real, torch.full_like(t_real, 6)) - 1).clamp_min(1)
    n_prefix = 1 + _uniform_index(goal_prefix_u, prefix_choices)
    max_suffix = torch.minimum(torch.full_like(t_real, 4), t_real - n_prefix)
    suffix_choices = (torch.maximum(max_suffix, torch.full_like(max_suffix, 2)) - 1).clamp_min(1)
    n_suffix = 1 + _uniform_index(goal_suffix_u, suffix_choices)
    n_suffix = torch.where(
        n_prefix + n_suffix >= t_real,
        (t_real - n_prefix - 1).clamp_min(1),
        n_suffix,
    )
    endpoint = (time < n_prefix.view(B, 1, 1)) | ((time >= (t_real - n_suffix).view(B, 1, 1)) & real)
    goal_all = goal_entity_u < 0.5
    goal_hands = (goal_entity_u >= 0.5) & (goal_entity_u < 0.75)
    entity_selected = goal_all.view(B, 1, 1) | (goal_hands.view(B, 1, 1) & (slot < 2)) | ((goal_entity_u >= 0.75).view(B, 1, 1) & (slot == 2))
    goal = real & endpoint & entity_selected
    goal = goal | (real & (time == 0))  # frame 0 is fully visible

    # temporal_infilling: a contiguous all-entity gap strictly between anchors.
    gap_len = torch.floor(t_real.to(torch.float32) * (0.2 + 0.3 * temporal_len_u)).to(torch.long)
    gap_len = torch.maximum(gap_len, torch.ones_like(gap_len))
    gap_len = torch.minimum(gap_len, (t_real - 2).clamp_min(1))
    gap_start_choices = (t_real - gap_len - 1).clamp_min(1)
    gap_start = 1 + _uniform_index(temporal_start_u, gap_start_choices)
    temporal = real & ~((time >= gap_start.view(B, 1, 1)) & (time < (gap_start + gap_len).view(B, 1, 1)))

    state_obs = torch.zeros(B, T, 3, dtype=torch.bool, device=device)
    candidates = (
        (future_id, future),
        (_REGIME_IDS["hand_to_object"], hand_to_object),
        (_REGIME_IDS["object_to_hand"], object_to_hand),
        (_REGIME_IDS["random_block"], random_block),
        (_REGIME_IDS["goal_infilling"], goal),
        (_REGIME_IDS["temporal_infilling"], temporal),
    )
    for regime_id, candidate in candidates:
        state_obs = torch.where((regime == regime_id).view(B, 1, 1), candidate, state_obs)

    obs_mask = torch.zeros(B, T, 5, dtype=torch.bool, device=device)
    obs_mask[:, :, :3] = state_obs
    obs_mask[:, :, 3] = state_obs[:, :, 0] & state_obs[:, :, 2]
    obs_mask[:, :, 4] = state_obs[:, :, 1] & state_obs[:, :, 2]
    obs_mask &= presence_mask
    return obs_mask


def _validate_regime_probs(regime_probs: dict[str, float]) -> tuple[list[str], list[float]]:
    if not regime_probs:
        raise ValueError("regime_probs must not be empty")
    names = list(regime_probs.keys())
    unknown = [name for name in names if name not in _REGIME_IDS]
    if unknown:
        raise ValueError(f"Unknown masking regime(s): {unknown}")
    probs = [float(regime_probs[name]) for name in names]
    total = sum(probs)
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Regime probabilities sum to {total:.4f}, expected ~1.0")
    if any(prob < 0.0 for prob in probs):
        raise ValueError("Regime probabilities must be non-negative")
    return names, probs


def _uniform_index(u: Tensor, n: Tensor) -> Tensor:
    """Map U[0,1) draws to uniform integer indices [0, n) without host reads."""
    n = n.clamp_min(1)
    index = torch.floor(u * n.to(u.dtype)).to(torch.long)
    return torch.minimum(index, n - 1)


def _sample_random_blocks_batched(
    *,
    presence_mask: Tensor,
    real: Tensor,
    t_real: Tensor,
    apply_u: Tensor,
    length_u: Tensor,
    start_u: Tensor,
    force_time_u: Tensor,
    force_entity_u: Tensor,
) -> Tensor:
    """Random-block state mask ``(B, T, 3)`` with a guaranteed present masked target."""
    B, _, _ = presence_mask.shape
    time = torch.arange(presence_mask.shape[1], device=presence_mask.device).view(1, -1, 1)
    slot = torch.arange(3, device=presence_mask.device).view(1, 1, 3)

    length_choices = (t_real - 1).clamp_min(1).view(B, 1).expand(-1, 3)
    block_len = 1 + _uniform_index(length_u, length_choices)
    start_choices = (t_real.view(B, 1) - block_len + 1).clamp_min(1)
    block_start = _uniform_index(start_u, start_choices)
    in_block = (time >= block_start.view(B, 1, 3)) & (time < (block_start + block_len).view(B, 1, 3))
    obs = real.expand(-1, -1, 3) & ~((apply_u < 0.5).view(B, 1, 3) & in_block)
    obs = obs | (real.expand(-1, -1, 3) & (time == 0))

    # all_obs is restricted to real-prefix tokens (padded False values do not count).
    all_obs = (obs | ~real.expand(-1, -1, 3)).all(dim=(1, 2))
    random_t = 1 + _uniform_index(force_time_u, (t_real - 1).clamp_min(1))
    random_e = _uniform_index(force_entity_u, torch.full_like(t_real, 3))
    arbitrary_force = all_obs & (t_real >= 2)

    # Guarantee a PRESENT masked target after frame 0: when the drawn blocks masked
    # no present token, mask one present token of a random eligible entity.
    eligible = presence_mask & real.expand(-1, -1, 3) & (time > 0)
    already_present_target = ((~obs) & eligible).any(dim=(1, 2))
    entity_eligible = eligible.any(dim=1)
    n_entities = entity_eligible.sum(dim=1, dtype=torch.long)
    has_eligible = n_entities > 0

    entity_rank = _uniform_index(force_entity_u, n_entities.clamp_min(1))
    selected_entity = entity_eligible & (entity_eligible.cumsum(dim=1, dtype=torch.long) - 1 == entity_rank.view(B, 1))
    selected_rows = eligible & selected_entity.view(B, 1, 3)
    n_rows = selected_rows.sum(dim=(1, 2), dtype=torch.long)
    row_rank = _uniform_index(force_time_u, n_rows.clamp_min(1))
    forced_present = selected_rows & (selected_rows.cumsum(dim=1, dtype=torch.long) - 1 == row_rank.view(B, 1, 1))
    force_present = has_eligible & ~already_present_target
    obs &= ~(forced_present & force_present.view(B, 1, 1))

    # Degenerate window (nothing present after frame 0): mask an arbitrary token
    # only when no random block was drawn at all.
    arbitrary_force &= ~has_eligible

    arbitrary_target = (time == random_t.view(B, 1, 1)) & (slot == random_e.view(B, 1, 1)) & arbitrary_force.view(B, 1, 1)
    obs &= ~arbitrary_target
    return obs
