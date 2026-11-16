"""AdaLN-Zero timestep conditioning for flow matching transformer blocks.

Implements the DiT (Scalable Diffusion Models with Transformers) convention:
- Sinusoidal timestep embedding → MLP → per-sublayer (gamma, beta, alpha)
- Zero-initialized: at init, gamma=0, beta=0, alpha=0
- Applied as: x_out = x + alpha * sublayer((1 + gamma) * LayerNorm(x) + beta)
- At init this reduces to identity: x_out = x

The model creates one shared TimestepEmbedder; each block creates N AdaLNModulation instances
(N=3 for factorized blocks: temporal attn, cross-token attn, FFN;
 N=2 for global blocks: global attn, FFN).
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


def sinusoidal_timestep_embedding(t: Tensor, dim: int) -> Tensor:
    """Sinusoidal embedding for continuous timestep t in [0, 1].

    Args:
        t: (B,) float timesteps in [0, 1].
        dim: Embedding dimension (must be even).

    Returns:
        (B, dim) sinusoidal embedding.
    """
    half = dim // 2
    # Use float32 for frequency computation regardless of input dtype
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.unsqueeze(-1).float() * freqs.unsqueeze(0)  # (B, half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
    return emb.to(t.dtype)


class TimestepEmbedder(nn.Module):
    """Embeds flow matching timestep t in [0,1] via sinusoidal encoding + MLP.

    Architecture: sinusoidal(t, freq_dim) → Linear(freq_dim, d_model) → SiLU → Linear(d_model, d_model)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        freq_dim = d_model // 4
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, t: Tensor) -> Tensor:
        """Embed timesteps.

        Args:
            t: (B,) float timesteps in [0, 1].

        Returns:
            (B, d_model) timestep embeddings.
        """
        sinusoidal = sinusoidal_timestep_embedding(t, self.freq_dim)
        return self.mlp(sinusoidal)


class AdaLNModulation(nn.Module):
    """Produces (gamma, beta, alpha) modulation parameters for one sublayer.

    Zero-initialized: at init, output is all zeros → gamma=0, beta=0, alpha=0.
    Applied by the block as: x_out = x + alpha * sublayer((1 + gamma) * LN(x) + beta)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 3 * d_model)
        # Zero-init: critical for training stability
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, t_emb: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute modulation parameters from timestep embedding.

        Args:
            t_emb: (B, d_model) timestep embedding.

        Returns:
            gamma: (B, d_model) — scale modulation (used as 1 + gamma).
            beta: (B, d_model) — shift modulation.
            alpha: (B, d_model) — residual gate (0 at init = closed).
        """
        raw = self.linear(t_emb)  # (B, 3 * d_model)
        gamma, beta, alpha = raw.chunk(3, dim=-1)
        return gamma, beta, alpha
