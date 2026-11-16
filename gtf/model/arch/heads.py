"""Output heads for the GTF transformer.

Velocity prediction heads (one per token type, shared for left/right) plus
auxiliary classification heads (contact, phase). The contact head is trained with a
per-finger BCE; the phase head receives no training signal.

Token slot order: [0=left_state, 1=right_state, 2=obj_state, 3=left_rel, 4=right_rel]
"""

import torch
import torch.nn as nn
from torch import Tensor

from gtf.config import ModelConfig


class OutputHeads(nn.Module):
    """Velocity prediction + auxiliary classification heads.

    Velocity heads: single linear layer per token type (no activation — velocity is unbounded).
    Auxiliary heads: 2-layer GELU MLPs for contact and phase classification.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D = cfg.d_model
        dims = cfg.state_dims

        # --- Primary velocity heads (shared where noted) ---
        self.hand_state_head = nn.Linear(D, dims.left_state)  # D → 99, shared for left/right
        self.obj_state_head = nn.Linear(D, dims.obj_state)  # D → 9
        self.relation_head = nn.Linear(D, dims.left_rel)  # D → 14, shared for left_rel/right_rel

        # --- Auxiliary heads ---
        # Contact: concat(left_state, right_state, obj_state) → 10 binary outputs
        self.contact_head = nn.Sequential(
            nn.Linear(D * 3, D),
            nn.GELU(),
            nn.Linear(D, 10),  # → reshape to (B, T, 2, 5)
        )

        # Phase: masked mean of 5 token types → 5 logits (no loss is applied to them)
        self.phase_head = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, 5),
        )

    def forward(self, x: Tensor, presence_mask: Tensor) -> dict:
        """Apply output heads to transformer output.

        Args:
            x: (B, T, 5, D) transformer output grid. Absent tokens are zero
                (maintained by block re-zeroing).
            presence_mask: (B, T, 5) bool. True = entity present. Needed for
                phase head masked mean.

        Returns:
            Dict with:
                'v_pred': dict mapping token name to (B, T, D_type) velocity.
                'contact_logits': (B, T, 2, 5) for BCE loss.
                'phase_logits': (B, T, 5) phase-head logits (no loss).
        """
        B, T, _, D = x.shape

        # Extract per-type tokens
        left_state_tok = x[:, :, 0, :]  # (B, T, D)
        right_state_tok = x[:, :, 1, :]  # (B, T, D)
        obj_state_tok = x[:, :, 2, :]  # (B, T, D)
        left_rel_tok = x[:, :, 3, :]  # (B, T, D)
        right_rel_tok = x[:, :, 4, :]  # (B, T, D)

        # --- Velocity predictions (shared modules for left/right) ---
        v_pred = {
            "left_state": self.hand_state_head(left_state_tok),  # (B, T, 99)
            "right_state": self.hand_state_head(right_state_tok),  # same module
            "obj_state": self.obj_state_head(obj_state_tok),  # (B, T, 9)
            "left_rel": self.relation_head(left_rel_tok),  # (B, T, 14)
            "right_rel": self.relation_head(right_rel_tok),  # same module
        }

        # --- Contact head ---
        # Input: concat(left_state, right_state, obj_state) per frame
        contact_input = torch.cat([left_state_tok, right_state_tok, obj_state_tok], dim=-1)  # (B, T, 3D)
        contact_logits = self.contact_head(contact_input)  # (B, T, 10)
        contact_logits = contact_logits.reshape(B, T, 2, 5)  # 2 hands × 5 fingers

        # --- Phase head ---
        # Input: masked mean over present tokens only (avoids dilution from absent slots)
        # presence_mask: (B, T, 5) bool
        pres_float = presence_mask.unsqueeze(-1).float()  # (B, T, 5, 1)
        n_present = presence_mask.sum(dim=2, keepdim=True).clamp(min=1).unsqueeze(-1)  # (B, T, 1, 1)
        phase_input = (x * pres_float).sum(dim=2) / n_present.squeeze(-1)  # (B, T, D)
        phase_logits = self.phase_head(phase_input)  # (B, T, 5)

        return {
            "v_pred": v_pred,
            "contact_logits": contact_logits,
            "phase_logits": phase_logits,
        }
