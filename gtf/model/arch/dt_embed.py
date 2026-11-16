"""Frame-cadence (dt) conditioning embed for the AdaLN stream.

The corpus mixes cadences (7.5 to 60 Hz), so a fixed-length window spans very
different durations and per-step velocities. DtEmbed maps the per-window log
dt-ratio (log(dt * dt_ref_fps); 0 at the reference cadence) to a (B, D) vector
added to the shared AdaLN timestep stream, reaching every block.

The final layer is zero-initialized, so the embed is an exact no-op at init.
"""

import torch
from torch import Tensor, nn


class DtEmbed(nn.Module):
    """(B, 1) log dt-ratio → (B, D) additive AdaLN conditioning vector."""

    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        # Zero-init the final layer (weight + bias): any input maps to zero at init.
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, log_dt_ratio: Tensor) -> Tensor:
        return self.mlp(log_dt_ratio)


def log_dt_ratio(dt: Tensor | None, dt_ref_fps: float, like: Tensor) -> Tensor:
    """Build the (B, 1) DtEmbed input from per-sample dt seconds.

    dt=None → zeros = log-ratio 0 = the dt_ref_fps reference cadence; the embed
    still runs, so its parameters stay in the autograd graph every step (required
    by DDP with find_unused_parameters=False).
    """
    B = like.shape[0]
    if dt is None:
        return like.new_zeros(B, 1)
    return torch.log((dt.to(device=like.device, dtype=like.dtype) * float(dt_ref_fps)).clamp_min(1e-6)).reshape(B, 1)
