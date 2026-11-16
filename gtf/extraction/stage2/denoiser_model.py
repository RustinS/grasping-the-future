"""Stage-2 trajectory denoiser: a compact bidirectional temporal transformer.

Input  = the full noisy window (normalized 207-D state per frame + validity flags);
         denoising is offline (whole extracted segment available), so no causal mask.
Output = a residual added to the (clamped) noisy input in normalized space
         (checkpoints/denoiser/v0.pt).

Residual prediction makes the identity the trivial baseline: an untrained model with a
zero-init head is a no-op, so anything it learns is signal over "return the input".
The 3 entities are packed into one token per frame (the MLP mixes them); temporal
attention is the working part: extraction noise is a slowly drifting bias (strong lag-1
autocorrelation), so the model's job is cross-time + cross-entity consistency.
"""

import torch
from torch import Tensor, nn

from gtf.extraction.stage2.stage2_common import STATE_DIM, WINDOW_T


class TrajDenoiser(nn.Module):
    def __init__(
        self,
        d_model: int = 384,
        n_layers: int = 8,
        n_heads: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.0,
        max_t: int = WINDOW_T,
        input_clamp: float = 12.0,
    ):
        super().__init__()
        self.input_clamp = input_clamp
        in_dim = STATE_DIM + 4  # states + 3 entity-valid flags + frame_valid
        self.in_proj = nn.Sequential(nn.Linear(in_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_t, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, STATE_DIM)
        nn.init.zeros_(self.head.weight)  # zero-init head → exact no-op at init
        nn.init.zeros_(self.head.bias)

    def forward(self, noisy_norm: Tensor, valid: Tensor, frame_valid: Tensor) -> Tensor:
        """noisy_norm (B,T,207), valid (B,T,3) bool, frame_valid (B,T) bool → denoised normalized states (B,T,207)."""
        x = noisy_norm.clamp(-self.input_clamp, self.input_clamp)
        feats = torch.cat([x, valid.float(), frame_valid.float().unsqueeze(-1)], dim=-1)
        h = self.in_proj(feats) + self.pos_embed[:, : x.shape[1]]
        # a window must keep at least one unmasked key (an all-masked row gives NaN attention)
        h = self.encoder(h, src_key_padding_mask=~frame_valid.bool())
        return x + self.head(h)
