"""Factorized and global transformer blocks for GTF.

FactorizedBlock: 3 sublayers (temporal self-attn, cross-token attn, FFN) — used for
    3 out of every 4 layers. Factorizes the (T, 5) spatiotemporal grid into separate
    temporal and cross-token attention operations.

GlobalBlock: 2 sublayers (full spatiotemporal attn, FFN) — every 4th layer.
    Full attention across all tokens at all frames.

Both blocks:
  - Use AdaLN-Zero conditioning on flow timestep t
  - Prepend conditioning tokens to K/V (not Q) for both attention types
  - Use presence_mask to exclude absent entities (→ -inf in attention)
  - Use cond_mask to exclude absent conditioning tokens (→ -inf in attention)
  - Re-zero absent tokens at end of forward to maintain the zero invariant
  - Use SDPA (auto-dispatches to FlashAttention-2 when possible)

Token grid shape: (B, T, 5, D) where 5 = [left_state, right_state, obj_state, left_rel, right_rel]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from gtf.model.arch.adaln import AdaLNModulation
from gtf.config import ModelConfig


def _multihead_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    n_heads: int,
    attn_mask: Tensor | None,
    dropout_p: float = 0.0,
    training: bool = True,
) -> Tensor:
    """Multi-head attention using PyTorch SDPA.

    Args:
        q: (B, S_q, D) queries.
        k: (B, S_kv, D) keys.
        v: (B, S_kv, D) values.
        n_heads: Number of attention heads.
        attn_mask: (B, 1, S_q, S_kv) float mask. 0.0 = attend, -inf = block.
            Or None for no masking.
        dropout_p: Dropout probability (only applied during training).
        training: Whether in training mode. SDPA doesn't respect model.eval(),
            so dropout_p must be explicitly set to 0.0 for inference.

    Returns:
        (B, S_q, D) attention output.
    """
    B, S_q, D = q.shape
    d_head = D // n_heads
    effective_dropout = dropout_p if training else 0.0

    # Reshape: (B, S, D) → (B, n_heads, S, d_head) for SDPA
    q = q.reshape(B, S_q, n_heads, d_head).transpose(1, 2)  # (B, n_heads, S_q, d_head)
    k = k.reshape(B, -1, n_heads, d_head).transpose(1, 2)  # (B, n_heads, S_kv, d_head)
    v = v.reshape(B, -1, n_heads, d_head).transpose(1, 2)  # (B, n_heads, S_kv, d_head)

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=effective_dropout)
    # (B, n_heads, S_q, d_head) → (B, S_q, D)
    return out.transpose(1, 2).reshape(B, S_q, D)


def _build_kv_mask(kv_valid: Tensor) -> Tensor:
    """Convert bool validity mask to float attention mask.

    Args:
        kv_valid: (..., S_kv) bool. True = attend, False = block.

    Returns:
        (..., 1, 1, S_kv) float. 0.0 where valid, -inf where blocked.
        Broadcasts over n_heads and S_q dimensions.
    """
    # Add dimensions for (n_heads=1, S_q=1) broadcasting
    mask = kv_valid.unsqueeze(-2).unsqueeze(-2)  # (..., 1, 1, S_kv)
    return torch.where(mask, 0.0, float("-inf"))


class FactorizedBlock(nn.Module):
    """Factorized transformer block with temporal + cross-token attention.

    Sub-layers:
        (a) Temporal self-attention: per token type, across frames
        (b) Cross-token attention: per frame, across 5 token types
        (c) FFN: per token, per frame

    Each sub-layer uses AdaLN-Zero conditioning on flow timestep.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D = cfg.d_model

        # --- Sub-layer (a): Temporal self-attention ---
        self.temporal_ln = nn.LayerNorm(D)
        self.temporal_adaln = AdaLNModulation(D)
        self.temporal_qkv = nn.Linear(D, 3 * D)
        self.temporal_cond_kv = nn.Linear(D, 2 * D)  # conditioning → K, V only
        self.temporal_out = nn.Linear(D, D)

        # --- Sub-layer (b): Cross-token attention ---
        self.cross_ln = nn.LayerNorm(D)
        self.cross_adaln = AdaLNModulation(D)
        self.cross_qkv = nn.Linear(D, 3 * D)
        self.cross_cond_kv = nn.Linear(D, 2 * D)
        self.cross_out = nn.Linear(D, D)

        # --- Sub-layer (c): FFN ---
        self.ffn_ln = nn.LayerNorm(D)
        self.ffn_adaln = AdaLNModulation(D)
        self.ffn = nn.Sequential(
            nn.Linear(D, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, D),
        )

        self.n_heads = cfg.n_heads
        self.dropout_p = cfg.dropout

    def forward(
        self,
        x: Tensor,
        cond_tokens: Tensor,
        cond_mask: Tensor,
        t_emb: Tensor,
        presence_mask: Tensor,
    ) -> Tensor:
        """Forward pass through factorized block.

        Args:
            x: (B, T, 5, D) token grid.
            cond_tokens: (B, N_c, D) conditioning tokens.
            cond_mask: (B, N_c) bool. True = real conditioning token.
            t_emb: (B, D) timestep embedding (from shared TimestepEmbedder).
            presence_mask: (B, T, 5) bool. True = entity present.

        Returns:
            (B, T, 5, D) updated token grid. Absent tokens re-zeroed.
        """
        B, T, S, D = x.shape  # S = 5
        N_c = cond_tokens.shape[1]

        # (a) Temporal self-attention — per token type
        gamma, beta, alpha = self.temporal_adaln(t_emb)  # each (B, D)

        # AdaLN: (1 + gamma) * LN(x) + beta
        x_normed = self.temporal_ln(x)
        x_mod = (1.0 + gamma[:, None, None, :]) * x_normed + beta[:, None, None, :]

        # Reshape for per-type temporal attention: (B, T, 5, D) → (B*5, T, D)
        x_flat = x_mod.permute(0, 2, 1, 3).reshape(B * S, T, D)

        # QKV from main tokens
        qkv = self.temporal_qkv(x_flat)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B*5, T, D)

        # Conditioning K,V: tile for each of 5 token types
        cond_kv = self.temporal_cond_kv(cond_tokens)  # (B, N_c, 2D)
        cond_k, cond_v = cond_kv.chunk(2, dim=-1)  # each (B, N_c, D)
        cond_k = cond_k.unsqueeze(1).expand(B, S, N_c, D).reshape(B * S, N_c, D)
        cond_v = cond_v.unsqueeze(1).expand(B, S, N_c, D).reshape(B * S, N_c, D)

        # Prepend conditioning to K, V
        k = torch.cat([cond_k, k], dim=1)  # (B*5, N_c + T, D)
        v = torch.cat([cond_v, v], dim=1)

        # Attention mask: cond_mask + presence per token type
        # presence_mask: (B, T, 5) → per-type: (B*5, T)
        pres_per_type = presence_mask.permute(0, 2, 1).reshape(B * S, T)
        cond_mask_tiled = cond_mask.unsqueeze(1).expand(B, S, N_c).reshape(B * S, N_c)
        kv_valid = torch.cat([cond_mask_tiled, pres_per_type], dim=1)  # (B*5, N_c + T)
        attn_mask = _build_kv_mask(kv_valid)  # (B*5, 1, 1, N_c + T)

        out = _multihead_attention(q, k, v, self.n_heads, attn_mask, self.dropout_p, self.training)
        out = self.temporal_out(out)  # (B*5, T, D)
        out = out.reshape(B, S, T, D).permute(0, 2, 1, 3)  # (B, T, 5, D)

        x = x + alpha[:, None, None, :] * out

        # (b) Cross-token attention — per frame
        gamma, beta, alpha = self.cross_adaln(t_emb)

        x_normed = self.cross_ln(x)
        x_mod = (1.0 + gamma[:, None, None, :]) * x_normed + beta[:, None, None, :]

        # Reshape for per-frame cross-token attention: (B, T, 5, D) → (B*T, 5, D)
        x_flat = x_mod.reshape(B * T, S, D)

        qkv = self.cross_qkv(x_flat)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B*T, 5, D)

        # Conditioning K,V: tile for each frame
        cond_kv = self.cross_cond_kv(cond_tokens)  # (B, N_c, 2D)
        cond_k, cond_v = cond_kv.chunk(2, dim=-1)
        cond_k = cond_k.unsqueeze(1).expand(B, T, N_c, D).reshape(B * T, N_c, D)
        cond_v = cond_v.unsqueeze(1).expand(B, T, N_c, D).reshape(B * T, N_c, D)

        k = torch.cat([cond_k, k], dim=1)  # (B*T, N_c + 5, D)
        v = torch.cat([cond_v, v], dim=1)

        # Attention mask: cond_mask + presence per frame
        pres_per_frame = presence_mask.reshape(B * T, S)  # (B*T, 5)
        cond_mask_tiled = cond_mask.unsqueeze(1).expand(B, T, N_c).reshape(B * T, N_c)
        kv_valid = torch.cat([cond_mask_tiled, pres_per_frame], dim=1)  # (B*T, N_c + 5)
        attn_mask = _build_kv_mask(kv_valid)

        out = _multihead_attention(q, k, v, self.n_heads, attn_mask, self.dropout_p, self.training)
        out = self.cross_out(out)  # (B*T, 5, D)
        out = out.reshape(B, T, S, D)

        x = x + alpha[:, None, None, :] * out

        # (c) FFN
        gamma, beta, alpha = self.ffn_adaln(t_emb)

        x_normed = self.ffn_ln(x)
        x_mod = (1.0 + gamma[:, None, None, :]) * x_normed + beta[:, None, None, :]
        out = self.ffn(x_mod)

        x = x + alpha[:, None, None, :] * out

        # Re-zero absent tokens to maintain the zero invariant
        x = x * presence_mask.unsqueeze(-1).float()

        return x


class GlobalBlock(nn.Module):
    """Global spatiotemporal transformer block.

    Flattens the (T, 5) grid to T*5 tokens and runs full self-attention.
    Used every global_block_interval layers (default: every 4th).

    Sub-layers:
        (a) Full spatiotemporal self-attention
        (b) FFN
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D = cfg.d_model

        # --- Sub-layer (a): Full spatiotemporal attention ---
        self.attn_ln = nn.LayerNorm(D)
        self.attn_adaln = AdaLNModulation(D)
        self.attn_qkv = nn.Linear(D, 3 * D)
        self.attn_cond_kv = nn.Linear(D, 2 * D)
        self.attn_out = nn.Linear(D, D)

        # --- Sub-layer (b): FFN ---
        self.ffn_ln = nn.LayerNorm(D)
        self.ffn_adaln = AdaLNModulation(D)
        self.ffn = nn.Sequential(
            nn.Linear(D, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, D),
        )

        self.n_heads = cfg.n_heads
        self.dropout_p = cfg.dropout

    def forward(
        self,
        x: Tensor,
        cond_tokens: Tensor,
        cond_mask: Tensor,
        t_emb: Tensor,
        presence_mask: Tensor,
    ) -> Tensor:
        """Forward pass through global block.

        Args:
            x: (B, T, 5, D) token grid.
            cond_tokens: (B, N_c, D) conditioning tokens.
            cond_mask: (B, N_c) bool.
            t_emb: (B, D) timestep embedding.
            presence_mask: (B, T, 5) bool.

        Returns:
            (B, T, 5, D) updated token grid. Absent tokens re-zeroed.
        """
        B, T, S, D = x.shape

        # (a) Full spatiotemporal attention
        gamma, beta, alpha = self.attn_adaln(t_emb)

        x_normed = self.attn_ln(x)
        x_mod = (1.0 + gamma[:, None, None, :]) * x_normed + beta[:, None, None, :]

        # Flatten: (B, T, 5, D) → (B, T*5, D)
        x_flat = x_mod.reshape(B, T * S, D)

        qkv = self.attn_qkv(x_flat)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, T*5, D)

        # Conditioning K,V
        cond_kv = self.attn_cond_kv(cond_tokens)  # (B, N_c, 2D)
        cond_k, cond_v = cond_kv.chunk(2, dim=-1)
        k = torch.cat([cond_k, k], dim=1)  # (B, N_c + T*5, D)
        v = torch.cat([cond_v, v], dim=1)

        # Attention mask: cond_mask + flattened presence
        pres_flat = presence_mask.reshape(B, T * S)  # (B, T*5)
        kv_valid = torch.cat([cond_mask, pres_flat], dim=1)  # (B, N_c + T*5)
        attn_mask = _build_kv_mask(kv_valid)  # (B, 1, 1, N_c + T*5)

        out = _multihead_attention(q, k, v, self.n_heads, attn_mask, self.dropout_p, self.training)
        out = self.attn_out(out)  # (B, T*5, D)
        out = out.reshape(B, T, S, D)

        x = x + alpha[:, None, None, :] * out

        # (b) FFN
        gamma, beta, alpha = self.ffn_adaln(t_emb)

        x_normed = self.ffn_ln(x)
        x_mod = (1.0 + gamma[:, None, None, :]) * x_normed + beta[:, None, None, :]
        out = self.ffn(x_mod)

        x = x + alpha[:, None, None, :] * out

        # Re-zero absent tokens
        x = x * presence_mask.unsqueeze(-1).float()

        return x
