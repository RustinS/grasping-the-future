"""Flow matching forward process for conditional OT formulation.

Interpolation: x_t = (1-t)*noise + t*data  (t=0: noise, t=1: data)
Velocity target: v = data - noise  (constant along straight-line OT paths)
"""

import torch
from torch import Tensor

TOKEN_TYPES = ["left_state", "right_state", "obj_state", "left_rel", "right_rel"]
TOKEN_SLOTS = {name: idx for idx, name in enumerate(TOKEN_TYPES)}


def sample_timestep(
    batch_size: int,
    device: torch.device,
    eps: float = 1e-5,
) -> Tensor:
    """Sample flow matching timesteps uniformly in [eps, 1-eps].

    Args:
        batch_size: Number of timesteps to sample.
        device: Target device.
        eps: Margin from 0 and 1 to avoid endpoint singularities.

    Returns:
        (B,) tensor of timesteps in [eps, 1-eps].
    """
    return torch.rand(batch_size, device=device) * (1.0 - 2 * eps) + eps


def interpolate(x_data: Tensor, noise: Tensor, t: Tensor) -> Tensor:
    """Compute flow matching interpolation x_t = (1-t)*noise + t*x_data.

    Convention: t=0 gives noise, t=1 gives data. Flow goes noise -> data.

    Args:
        x_data: (B, ...) clean data.
        noise: (B, ...) noise samples, same shape as x_data.
        t: (B,) timesteps. Will be reshaped for broadcasting.

    Returns:
        (B, ...) interpolated values.
    """
    t_expanded = t.reshape(-1, *([1] * (x_data.ndim - 1)))
    return (1.0 - t_expanded) * noise + t_expanded * x_data


def velocity_target(x_data: Tensor, noise: Tensor) -> Tensor:
    """Compute the conditional flow matching velocity target.

    v = x_data - noise (independent of t, constant velocity along OT paths).

    Args:
        x_data: (B, ...) clean data.
        noise: (B, ...) noise samples.

    Returns:
        (B, ...) velocity targets.
    """
    return x_data - noise


def apply_flow_noise(
    tokens_dict: dict[str, Tensor],
    obs_mask: Tensor,
    presence_mask: Tensor,
    t: Tensor,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Apply flow matching noise to masked tokens.

    For each token type independently:
    - Masked + present tokens: interpolated between noise and data
    - Observed + present tokens: kept as clean data
    - Absent tokens: kept as zeros

    Args:
        tokens_dict: Dict mapping token type name to (B, T, D_type) tensors.
        obs_mask: (B, T, 5) bool. True = observed.
        presence_mask: (B, T, 5) bool. True = present.
        t: (B,) flow timesteps.

    Returns:
        x_t_dict: Dict of (B, T, D_type) noised token tensors.
        v_target_dict: Dict of (B, T, D_type) velocity target tensors.
            Only meaningful for masked+present tokens.
    """
    x_t_dict = {}
    v_target_dict = {}

    for name in TOKEN_TYPES:
        slot_idx = TOKEN_SLOTS[name]
        x_data = tokens_dict[name]  # (B, T, D)

        # Independent noise per token, per frame, per batch element
        noise = torch.randn_like(x_data)

        # Per-slot masks: (B, T) -> (B, T, 1) for broadcasting with (B, T, D)
        is_observed = obs_mask[:, :, slot_idx].unsqueeze(-1)
        is_absent = (~presence_mask[:, :, slot_idx]).unsqueeze(-1)

        # Compute interpolation and velocity for all positions
        x_t = interpolate(x_data, noise, t)  # (B, T, D)
        v = velocity_target(x_data, noise)  # (B, T, D)

        # Assemble: observed=clean, masked+present=noised, absent=zeros
        x_t_out = torch.where(is_observed, x_data, x_t)
        x_t_out = torch.where(is_absent, torch.zeros_like(x_t_out), x_t_out)

        v_target_out = torch.where(is_observed, torch.zeros_like(v), v)
        v_target_out = torch.where(is_absent, torch.zeros_like(v_target_out), v_target_out)

        x_t_dict[name] = x_t_out
        v_target_dict[name] = v_target_out

    return x_t_dict, v_target_dict


def estimate_x0(x_t: Tensor, v_pred: Tensor, t: Tensor) -> Tensor:
    """Estimate clean data from noised state and predicted velocity.

    From x_t = (1-t)*noise + t*data and v = data - noise:
    data = x_t + (1-t)*v

    Args:
        x_t: (B, ...) current noised state.
        v_pred: (B, ...) predicted velocity.
        t: (B,) timesteps. Will be reshaped for broadcasting.

    Returns:
        (B, ...) estimated clean data.
    """
    t_expanded = t.reshape(-1, *([1] * (x_t.ndim - 1)))
    return x_t + (1.0 - t_expanded) * v_pred
