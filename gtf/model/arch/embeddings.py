"""Input embedding layers for the GTF transformer.

Projects per-token-type state vectors to d_model and adds the anchor side-channel,
type, physical-time position, observation-mask and confidence embeddings.
Weight sharing:
- hand_state_proj / hand_anchor_proj: shared for left_state and right_state
- relation_proj: shared for left_rel and right_rel
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from gtf.config import ModelConfig
from gtf.model.repr.timing import frame_times_from_edge_dt, normalize_edge_dt


def sinusoidal_position_embedding(positions: Tensor, d_model: int) -> Tensor:
    """Compute sinusoidal position embeddings for frame/time positions (any length, computed on the fly).

    Args:
        positions: (T,) integer or floating-point positions.
        d_model: Embedding dimension (should be even).

    Returns:
        (T, d_model) position embeddings.
    """
    half = d_model // 2
    # Use float32 for frequency computation
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=positions.device, dtype=torch.float32) / half)
    args = positions.unsqueeze(-1).float() * freqs.unsqueeze(0)  # (T, half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (T, d_model) or (T, d_model-1)
    # Handle odd d_model
    if d_model % 2 == 1:
        emb = torch.cat([emb, torch.zeros(positions.shape[0], 1, device=positions.device)], dim=-1)
    return emb


def physical_time_positions(
    frame_dt: Tensor | None,
    *,
    batch_size: int,
    n_frames: int,
    device: torch.device,
    dt_ref_fps: float,
) -> Tensor | None:
    """Return ``(B,T)`` elapsed-time positions in reference-frame units.

    ``frame_dt`` holds ``(B,T-1)`` seconds for the transitions ``t→t+1``; the
    positions are its cumulative sum times ``dt_ref_fps`` (equal to frame indices
    at the reference cadence). None returns None (frame-index positions).
    """
    if frame_dt is None:
        return None
    edge_dt = normalize_edge_dt(frame_dt, n_frames=n_frames, batch_size=batch_size, device=device, dtype=torch.float32, validate=False)
    return frame_times_from_edge_dt(edge_dt) * float(dt_ref_fps)


def summarize_frame_dt(
    frame_dt: Tensor | None,
    presence_mask: Tensor,
) -> Tensor | None:
    """Reduce per-transition timing to the per-window scalar consumed by ``DtEmbed``.

    The arithmetic mean covers only transitions whose endpoint frames are valid
    (at least one token present), excluding padded tails.
    """
    if frame_dt is None:
        return None
    B, T = presence_mask.shape[:2]
    edge_dt = normalize_edge_dt(frame_dt, n_frames=T, batch_size=B, device=presence_mask.device, dtype=torch.float32, validate=False)
    valid_frames = presence_mask.any(dim=-1)
    active_edges = valid_frames[:, :-1] & valid_frames[:, 1:]
    weights = active_edges.to(edge_dt.dtype)
    counts = weights.sum(dim=1)
    totals = (edge_dt * weights).sum(dim=1)
    # A one-frame/all-padded sample has no physical transition. It is irrelevant to
    # the loss, but keep DtEmbed finite without a CUDA scalar read.
    if edge_dt.shape[1]:
        fallback = edge_dt.mean(dim=1)
    else:
        fallback = edge_dt.new_full((B,), 1.0 / 30.0)
    return torch.where(counts > 0, totals / counts.clamp_min(1), fallback)


def _make_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    """2-layer GELU MLP for input projection."""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.GELU(),
        nn.Linear(out_dim, out_dim),
    )


class InputEmbedding(nn.Module):
    """Projects per-token-type state vectors to d_model and adds embeddings.

    Output: (B, T, 5, D) token grid. Presence-masked tokens are exactly zero.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D = cfg.d_model
        dims = cfg.state_dims

        # State input MLPs — shared between left/right hand and left/right relation
        self.hand_state_proj = _make_mlp(dims.left_state, D)  # 99 → D, shared for left/right
        self.obj_state_proj = _make_mlp(dims.obj_state, D)  # 9 → D
        self.relation_proj = _make_mlp(dims.left_rel, D)  # 14 → D, shared for left_rel/right_rel

        # Token type embedding: 5 types
        self.type_embedding = nn.Embedding(cfg.n_token_types, D)

        # Observation mask embedding: 0=observed, 1=masked
        self.obs_mask_embedding = nn.Embedding(2, D)

        # Confidence embeddings
        self.state_confidence_proj = _make_mlp(cfg.confidence_dim, D)  # 7 → D for state tokens
        self.relation_confidence_proj = _make_mlp(cfg.confidence_dim * 2, D)  # 14 → D for relation tokens

        # Anchor side-channel: zero-init additive projections of [absolute-normalized
        # state at observed frames, observed indicator] per state token (see
        # delta_codec.build_anchor_feats). In Δ-codec space an observed future frame
        # otherwise carries only derivatives, so its absolute position would be invisible.
        self.hand_anchor_proj = nn.Linear(dims.left_state + 1, D)
        self.obj_anchor_proj = nn.Linear(dims.obj_state + 1, D)
        for m in (self.hand_anchor_proj, self.obj_anchor_proj):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

        # Temporal position embedding over elapsed physical time in reference-fps
        # units, so mixed-cadence windows share one time axis.
        self.dt_ref_fps = float(cfg.dt_ref_fps)

        # Dropout after assembly, before presence masking
        self.dropout = nn.Dropout(cfg.dropout)

        self.d_model = D

    def project_anchor_features(self, anchor_feats: dict[str, Tensor]) -> Tensor:
        """Project clean state anchors to the three state slots as ``(B,T,3,D)``."""
        required = ("left_state", "right_state", "obj_state")
        absent = [name for name in required if name not in anchor_feats]
        if absent:
            raise ValueError(f"anchor_feats omitted {absent}")
        return torch.stack(
            (
                self.hand_anchor_proj(anchor_feats["left_state"]),
                self.hand_anchor_proj(anchor_feats["right_state"]),
                self.obj_anchor_proj(anchor_feats["obj_state"]),
            ),
            dim=2,
        )

    def forward(
        self,
        tokens_dict: dict[str, Tensor],
        obs_mask: Tensor,
        presence_mask: Tensor,
        confidence_features: Tensor | None = None,
        anchor_feats: dict[str, Tensor] | None = None,
        frame_dt: Tensor | None = None,
    ) -> Tensor:
        """Embed and assemble the token grid.

        Args:
            tokens_dict: Dict mapping token name to (B, T, D_type) state tensors.
                Keys: left_state(99), right_state(99), obj_state(9), left_rel(14), right_rel(14).
            obs_mask: (B, T, 5) bool. True = observed.
            presence_mask: (B, T, 5) bool. True = present entity.
            confidence_features: (B, T, 5, 7) or None. Per-token confidence features;
                None uses a constant neutral 0.5.
            anchor_feats: Optional anchor side-channel inputs (build_anchor_feats).
            frame_dt: ``(B,T-1)`` transition durations in seconds; None uses
                frame-index positions.

        Returns:
            (B, T, 5, D) embedded token grid. Absent tokens are exactly 0.
        """
        B, T = obs_mask.shape[:2]
        D = self.d_model
        device = obs_mask.device
        dtype = next(self.parameters()).dtype

        # --- 1. Project states to D ---
        # Shared MLPs: same Module instance for left/right hand and left/right relation
        projections = [
            self.hand_state_proj(tokens_dict["left_state"]),  # (B, T, D)
            self.hand_state_proj(tokens_dict["right_state"]),  # same module
            self.obj_state_proj(tokens_dict["obj_state"]),
            self.relation_proj(tokens_dict["left_rel"]),  # (B, T, D)
            self.relation_proj(tokens_dict["right_rel"]),  # same module
        ]
        x = torch.stack(projections, dim=2)  # (B, T, 5, D)

        # --- 1b. Anchor side-channel (additive) ---
        if anchor_feats is not None:
            x = x.clone()
            anchor_projection = self.project_anchor_features(anchor_feats)
            x[:, :, :3] = x[:, :, :3] + anchor_projection

        # --- 2. Token type embedding ---
        type_ids = torch.arange(5, device=device)  # (5,)
        type_emb = self.type_embedding(type_ids)  # (5, D)
        x = x + type_emb.reshape(1, 1, 5, D)

        # --- 3. Temporal position embedding (sinusoidal, on-the-fly) ---
        positions = physical_time_positions(frame_dt, batch_size=B, n_frames=T, device=device, dt_ref_fps=self.dt_ref_fps)
        if positions is not None:
            temporal_emb = sinusoidal_position_embedding(positions.reshape(-1), D).reshape(B, T, D)
            x = x + temporal_emb.reshape(B, T, 1, D).to(x.dtype)
        else:
            frame_indices = torch.arange(T, device=device)  # (T,)
            temporal_emb = sinusoidal_position_embedding(frame_indices, D)  # (T, D)
            x = x + temporal_emb.reshape(1, T, 1, D)

        # --- 4. Observation mask embedding ---
        obs_ids = (~obs_mask).long()  # True→observed→0, False→masked→1
        obs_emb = self.obs_mask_embedding(obs_ids)  # (B, T, 5, D)
        x = x + obs_emb

        # --- 5. Confidence embedding ---
        if confidence_features is None:
            # Constant 0.5 neutral confidence
            state_conf = torch.full((B, T, 3, 7), 0.5, device=device, dtype=dtype)
            # Relation: concat(hand_conf, obj_conf) = 14D, filled with 0.5
            rel_conf = torch.full((B, T, 2, 14), 0.5, device=device, dtype=dtype)
        else:
            state_conf = confidence_features[:, :, :3, :]  # (B, T, 3, 7)
            # left_rel confidence = concat(left_state_conf, obj_state_conf)
            left_rel_conf = torch.cat([confidence_features[:, :, 0, :], confidence_features[:, :, 2, :]], dim=-1)  # (B, T, 14)
            # right_rel confidence = concat(right_state_conf, obj_state_conf)
            right_rel_conf = torch.cat([confidence_features[:, :, 1, :], confidence_features[:, :, 2, :]], dim=-1)  # (B, T, 14)
            rel_conf = torch.stack([left_rel_conf, right_rel_conf], dim=2)  # (B, T, 2, 14)

        # Project confidence
        state_conf_emb = self.state_confidence_proj(state_conf)  # (B, T, 3, D)
        rel_conf_emb = self.relation_confidence_proj(rel_conf)  # (B, T, 2, D)
        conf_emb = torch.cat([state_conf_emb, rel_conf_emb], dim=2)  # (B, T, 5, D)
        x = x + conf_emb

        # --- 6. Dropout ---
        x = self.dropout(x)

        # --- 7. Presence masking: absent tokens → exactly 0 ---
        x = x * presence_mask.unsqueeze(-1).float()

        return x
