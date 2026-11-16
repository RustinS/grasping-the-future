"""7D confidence features per token per frame.

Produces a (T, 5, 7) tensor consumed by `gtf.model.arch.embeddings.InputEmbedding`.
Five token slots: [left_state, right_state, obj_state, left_rel, right_rel].
Seven features per slot:

    0: temporal_smoothness   -- 1 - normalized residual from a local moving average
    1: geometric_consistency -- 1 - normalized fingertip-distance jitter
    2-4: reserved            -- constant 0.5
    5: temporal_coverage     -- fraction of nearby frames where the token is present
    6: extractor_score       -- confidence-tier constant (see TIER_SCORE)

Neighborhoods are defined in physical time (from each window's frame timestamps),
so the same motion scores the same at every frame rate. Absent tokens
(presence_mask=False) get all 0.5 (neutral). All values are clamped to [0, 1].
The embedding layer ignores relation-slot values and derives relation confidence
from state confidences, so slots 3/4 are copies of slots 0/1.
"""

import torch
from torch import Tensor

TIER_SCORE: dict[str, float] = {
    "gt_mocap": 0.95,
    "gt_fitted": 0.80,
    "gt_landmark_fitted": 0.65,
    "extracted": 0.5,
}

# Fallback (neutral) confidence value for absent tokens and unknown features.
NEUTRAL = 0.5

# Normalization scales for the temporal smoothness residuals (per-type, meters).
# They act as soft gates.
SMOOTHNESS_SCALE_TRANSLATION = 0.05  # 5 cm residual is "very jittery"
SMOOTHNESS_SCALE_ROTATION = 0.1  # 0.1 rad residual is "very jittery"

# Jitter scale for fingertip distances (1 cm deviation is "very jittery").
JITTER_SCALE = 0.01

# Temporal coverage window: ±5 frames at 30 Hz.
COVERAGE_RADIUS = 5

# Half-width (seconds) of the smoothness / jitter neighborhoods (±3 frames at 30 Hz).
SMOOTHNESS_RADIUS_S = 0.1


def _time_boundary_tolerance(times: Tensor, radius_s: float) -> Tensor:
    """Roundoff-only tolerance for inclusive physical-time neighborhoods.

    Relative timestamps are float64. The tolerance is far below the source
    timestamp quantum while still covering arithmetic error from adding or
    subtracting the radius at the end of a five-second window. The final
    singleton dimension broadcasts over time (and over a pairwise time matrix
    in ``neutralize_confidence_leak``).
    """

    if times.dtype != torch.float64:
        raise TypeError("physical-time boundary calculations require float64")
    if times.shape[-1] == 0:
        return torch.zeros(
            times.shape[:-1] + (1,),
            device=times.device,
            dtype=times.dtype,
        )
    scale = times.abs().amax(dim=-1, keepdim=True).clamp_min(max(1.0, abs(float(radius_s))))
    return scale * (16.0 * torch.finfo(torch.float64).eps)


def _time_window_bounds(
    frame_times_s: Tensor,
    radius_s: float,
    *,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Inclusive row bounds for centered physical-time neighborhoods."""

    if frame_times_s.dtype != torch.float64:
        raise TypeError("physical-time neighborhoods require float64 frame_times_s")
    times = frame_times_s.to(device=device)
    if times.ndim != 1:
        raise ValueError(f"frame_times_s must be one-dimensional, got {tuple(times.shape)}")
    if times.numel() > 1 and not bool((times[1:] > times[:-1]).all()):
        raise ValueError("frame_times_s must be strictly increasing")
    tolerance = _time_boundary_tolerance(times, radius_s)
    lower = times - float(radius_s) - tolerance
    upper = times + float(radius_s) + tolerance
    start = torch.searchsorted(times, lower, right=False)
    end = torch.searchsorted(times, upper, right=True)
    return times, start, end


def _time_window_average(signal: Tensor, frame_times_s: Tensor, radius_s: float) -> Tensor:
    """Centered moving average (equal weight per row) over a physical-time radius."""

    times, start, end = _time_window_bounds(
        frame_times_s,
        radius_s,
        device=signal.device,
    )
    if times.shape[0] != signal.shape[0]:
        raise ValueError(f"frame_times_s must be ({signal.shape[0]},), got {tuple(times.shape)}")
    zero = torch.zeros((1,) + signal.shape[1:], device=signal.device, dtype=signal.dtype)
    csum = torch.cat([zero, torch.cumsum(signal, dim=0)], dim=0)
    denom = (end - start).to(signal.dtype).clamp_min(1)
    while denom.ndim < signal.ndim:
        denom = denom.unsqueeze(-1)
    return (csum[end] - csum[start]) / denom


def _time_window_support(
    frame_times_s: Tensor,
    radius_s: float,
) -> Tensor:
    """Number of rows inside each centered physical-time neighborhood."""

    _, start, end = _time_window_bounds(frame_times_s, radius_s)
    return end - start


def _trans_residual_smoothness(
    signal: Tensor,
    scale: float,
    frame_times_s: Tensor,
    radius_s: float = SMOOTHNESS_RADIUS_S,
) -> Tensor:
    """1 - normalized residual from a centered physical-time moving average.

    Args:
        signal: (T, D) trajectory.
        scale: Normalization scale.
        frame_times_s: (T,) float64 frame times.
        radius_s: Neighborhood half-width in seconds.

    Returns:
        (T,) smoothness values in [0, 1].
    """
    T = signal.shape[0]
    if T == 0:
        return torch.zeros(0, device=signal.device, dtype=signal.dtype)

    avg = _time_window_average(signal, frame_times_s, radius_s)
    residual = (signal - avg).norm(dim=-1)  # (T,)
    smoothness = 1.0 - (residual / scale).clamp(0.0, 1.0)
    # Fewer than three samples inside the neighborhood (possible below 10 Hz)
    # cannot estimate local smoothness: report the neutral value.
    support = _time_window_support(frame_times_s, radius_s).to(device=smoothness.device)
    return torch.where(support >= 3, smoothness, torch.full_like(smoothness, NEUTRAL))


def _fingertip_jitter(
    fd: Tensor,
    scale: float,
    frame_times_s: Tensor,
    radius_s: float = SMOOTHNESS_RADIUS_S,
) -> Tensor:
    """Per-frame jitter of fingertip distances.

    Jitter is the mean absolute deviation of the distances from their centered
    physical-time moving average (position units), so noise amplitude and smooth
    relative motion are both close to cadence-invariant.

    Args:
        fd: (T, 5) fingertip distances per frame.
        scale: Jitter normalization scale (meters).
        frame_times_s: (T,) float64 frame times.
        radius_s: Neighborhood half-width in seconds.

    Returns:
        (T,) consistency in [0, 1]; high = smooth, low = jittery.
    """
    T = fd.shape[0]
    if T < 2:
        return torch.full((T,), NEUTRAL, device=fd.device, dtype=fd.dtype)

    avg = _time_window_average(fd, frame_times_s, radius_s)
    jitter_t = (fd - avg).abs().mean(dim=-1)  # (T,)
    consistency = 1.0 - (jitter_t / scale).clamp(0.0, 1.0)
    support = _time_window_support(frame_times_s, radius_s).to(device=consistency.device)
    return torch.where(support >= 3, consistency, torch.full_like(consistency, NEUTRAL))


def _temporal_coverage(
    presence_col: Tensor,
    frame_times_s: Tensor,
    radius_s: float = COVERAGE_RADIUS / 30.0,
) -> Tensor:
    """Fraction of rows within ±radius_s where the token is present.

    Args:
        presence_col: (T,) bool presence mask for one token type.
        frame_times_s: (T,) float64 frame times.
        radius_s: Neighborhood half-width in seconds.

    Returns:
        (T,) float in [0, 1].
    """
    T = presence_col.shape[0]
    if T == 0:
        return torch.zeros(0, device=presence_col.device, dtype=torch.float32)

    pm = presence_col.float()
    return _time_window_average(pm.unsqueeze(-1), frame_times_s, radius_s).squeeze(-1)


def compute_confidence_features(
    tokens: dict[str, Tensor],
    presence_mask: Tensor,
    relation_features: dict[str, Tensor],
    extractor_score: float,
    frame_times_s: Tensor,
) -> Tensor:
    """Compute per-token confidence features for one window.

    Args:
        tokens: dict with keys left_state, right_state, obj_state — each (T, D).
        presence_mask: (T, 5) bool.
        relation_features: dict with keys left_rel, right_rel — each (T, 14).
        extractor_score: scalar tier score (see TIER_SCORE).
        frame_times_s: (T,) float64 frame times (seconds, strictly increasing).

    Returns:
        (T, 5, 7) tensor in [0, 1]. Slots 3/4 are copies of 0/1 for consistency with
        the embedding layer contract.
    """
    left_state = tokens["left_state"]
    right_state = tokens["right_state"]
    obj_state = tokens["obj_state"]

    T = presence_mask.shape[0]
    device = presence_mask.device
    dtype = left_state.dtype

    conf = torch.full((T, 5, 7), NEUTRAL, device=device, dtype=dtype)

    # hand_state layout: [orient_6d(6) | cam_t(3) | pose_6d(90)] = 99D
    # obj_state layout:  [rotation_6d(6) | translation(3)]        = 9D
    for slot_idx, state in enumerate((left_state, right_state, obj_state)):
        trans = state[:, 6:9]
        orient = state[:, 0:6]

        # Feature 0: temporal smoothness on translation + orientation.
        smooth_trans = _trans_residual_smoothness(trans, SMOOTHNESS_SCALE_TRANSLATION, frame_times_s)
        smooth_rot = _trans_residual_smoothness(orient, SMOOTHNESS_SCALE_ROTATION, frame_times_s)
        conf[:, slot_idx, 0] = (smooth_trans + smooth_rot) * 0.5

        # Feature 5: temporal coverage.
        conf[:, slot_idx, 5] = _temporal_coverage(presence_mask[:, slot_idx], frame_times_s).to(dtype)

        # Feature 6: extractor score (constant per segment)
        conf[:, slot_idx, 6] = float(extractor_score)

    # --- Feature 1: geometric consistency from fingertip jitter ---
    left_fd = relation_features["left_rel"][:, 9:14]  # (T, 5)
    right_fd = relation_features["right_rel"][:, 9:14]  # (T, 5)
    left_consistency = _fingertip_jitter(left_fd, JITTER_SCALE, frame_times_s)
    right_consistency = _fingertip_jitter(right_fd, JITTER_SCALE, frame_times_s)

    conf[:, 0, 1] = left_consistency
    conf[:, 1, 1] = right_consistency

    # Obj consistency: average of present hands
    left_present_f = presence_mask[:, 0].float()
    right_present_f = presence_mask[:, 1].float()
    denom = (left_present_f + right_present_f).clamp(min=1.0)
    obj_consistency = (left_consistency * left_present_f + right_consistency * right_present_f) / denom
    # Frames with no present hand fall back to neutral.
    any_hand = presence_mask[:, 0] | presence_mask[:, 1]
    obj_consistency = torch.where(any_hand, obj_consistency, torch.full_like(obj_consistency, NEUTRAL))
    conf[:, 2, 1] = obj_consistency

    # --- Absent tokens: every feature neutral (0.5) ---
    for slot_idx in range(3):
        absent = ~presence_mask[:, slot_idx]
        if absent.any():
            conf[absent, slot_idx, :] = NEUTRAL

    # --- Relation slots 3/4 copy from state slots 0/1 ---
    conf[:, 3, :] = conf[:, 0, :]
    conf[:, 4, :] = conf[:, 1, :]

    return conf.clamp(0.0, 1.0)


def neutralize_confidence_leak(
    conf: Tensor,
    obs_mask: Tensor,
    presence_mask: Tensor | None = None,
    valid_frames: Tensor | None = None,
    frame_times_s: Tensor | None = None,
) -> Tensor:
    """Hide masked ground truth from the confidence features (train, eval and live eval).

    Features 0 (temporal smoothness) and 1 (fingertip jitter, a cross-entity
    hand-object quantity) are computed from the ground-truth trajectory around
    each frame, so an observed slot near a masked one carries information about
    the masked motion. Rule: feature 0 → 0.5 within ±0.1 s of that entity's
    masked frames; feature 1 additionally within ±0.1 s of the pair partner's
    masked frames (hand partner = object; object partners = both hands).
    Features 2-6 are mask-independent and untouched. Relation slots 3/4 mirror
    the neutralized state slots 0/1.

    Only genuinely hidden ground truth is a source: ``(~obs) & presence & valid``
    when the masks are given. Absent-entity frames carry no ground truth and
    padding is not a target.

    Args:
        conf: (B, T, 5, 7) confidence features (not mutated).
        obs_mask: (B, T, 5) bool — True = observed.
        presence_mask: (B, T, 5) or (B, T, 3) bool — entity present per frame.
        valid_frames: (B, T) bool — real (non-padded) frames.
        frame_times_s: (B, T) float64 relative frame times of each window.

    Returns a new (B, T, 5, 7) tensor.
    """
    assert conf.dim() == 4 and obs_mask.dim() == 3, (conf.shape, obs_mask.shape)
    if frame_times_s is None:
        raise ValueError("neutralize_confidence_leak requires frame_times_s")
    # Inputs may arrive on different devices (GPU features, CPU masks).
    hidden = ~obs_mask[:, :, :3].to(conf.device)
    if presence_mask is not None:
        hidden = hidden & presence_mask[:, :, :3].bool().to(conf.device)
    if valid_frames is not None:
        hidden = hidden & valid_frames.bool().to(conf.device).unsqueeze(-1)
    masked = hidden.to(conf.dtype).permute(0, 2, 1)  # (B, 3, T)
    if frame_times_s.dtype != torch.float64:
        raise TypeError("confidence neutralization requires float64 frame_times_s")
    times = frame_times_s.to(device=masked.device)
    if tuple(times.shape) != (masked.shape[0], masked.shape[2]):
        raise ValueError(f"frame_times_s must be {(masked.shape[0], masked.shape[2])}, got {tuple(times.shape)}")
    # Same ±0.1 s receptive field as the confidence computation; hidden source
    # frames are already presence/valid gated above.
    tolerance = _time_boundary_tolerance(times, SMOOTHNESS_RADIUS_S)
    near = (times[:, :, None] - times[:, None, :]).abs() <= (SMOOTHNESS_RADIUS_S + tolerance[:, :, None])
    dil = torch.einsum("bij,bej->bei", near.to(masked.dtype), masked).clamp_max_(1.0)
    dil = dil.permute(0, 2, 1) > 0.5  # (B, T, 3) — masked within the time radius
    f0 = dil  # feature 0: entity-local receptive field
    f1 = torch.stack(  # feature 1: entity ∪ pair partner(s)
        [dil[:, :, 0] | dil[:, :, 2], dil[:, :, 1] | dil[:, :, 2], dil[:, :, 2] | dil[:, :, 0] | dil[:, :, 1]],
        dim=-1,
    )
    out = conf.clone()
    neutral = torch.full((), NEUTRAL, device=conf.device, dtype=conf.dtype)
    out[:, :, :3, 0] = torch.where(f0, neutral, out[:, :, :3, 0])
    out[:, :, :3, 1] = torch.where(f1, neutral, out[:, :, :3, 1])
    out[:, :, 3, :] = out[:, :, 0, :]
    out[:, :, 4, :] = out[:, :, 1, :]
    return out
