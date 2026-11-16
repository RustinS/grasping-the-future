"""GTF: masked spatiotemporal completion model for hand-object manipulation.

  InputEmbedding → [FactorizedBlock × 9, GlobalBlock × 3] → OutputHeads
with conditioning tokens (cached Uni3D object shape, hand betas, object scale, scene)
and AdaLN-Zero conditioning on the flow timestep and the window's frame cadence.

The model receives already-noised tokens (x_t). Flow noise, masking, centering,
and the Δ-codec transform are the training loop's (or the sampler's) responsibility.
"""

import torch
import torch.nn as nn
from torch import Tensor

from gtf.paths import MANO_DIR
from gtf.model.arch.adaln import TimestepEmbedder
from gtf.model.arch.blocks import FactorizedBlock, GlobalBlock
from gtf.model.arch.conditioning import ConditioningEncoder
from gtf.model.arch.dt_embed import DtEmbed, log_dt_ratio
from gtf.model.arch.embeddings import InputEmbedding, summarize_frame_dt
from gtf.model.arch.heads import OutputHeads
from gtf.config import GTFConfig
from gtf.model.repr.mano import MANOHelper


class GTFModel(nn.Module):
    """Complete GTF model for masked spatiotemporal flow matching.

    Architecture:
        - InputEmbedding: per-token-type MLPs + anchor side-channel + type/temporal/obs/confidence embeddings
        - ConditioningEncoder: cached Uni3D object shape + hand betas + obj scale + LitePT scene tokens
        - TimestepEmbedder + DtEmbed: shared AdaLN conditioning (flow timestep + log frame cadence)
        - Block stack: 9 FactorizedBlock + 3 GlobalBlock (every global_block_interval-th layer)
        - OutputHeads: per-type velocity heads + contact/phase auxiliary heads

    The MANOHelper is registered as a frozen submodule for device management
    and checkpointing. It is not used in forward; the losses access it as
    model.mano_helper.
    """

    def __init__(
        self,
        cfg: GTFConfig,
        mano_root: str = MANO_DIR,
    ):
        super().__init__()
        model_cfg = cfg.model
        self.dt_ref_fps = float(model_cfg.dt_ref_fps)

        # 1. Input embedding
        self.input_embedding = InputEmbedding(model_cfg)

        # 2. Conditioning encoder
        self.conditioning_encoder = ConditioningEncoder(model_cfg)

        # 3. Timestep embedding — one shared instance
        self.timestep_embedder = TimestepEmbedder(model_cfg.d_model)

        # 4. Block stack: GlobalBlock every global_block_interval layers, rest Factorized
        # (n_layers=12, interval=4: GlobalBlock at i=3,7,11 → 9F + 3G)
        self.blocks = nn.ModuleList()
        for i in range(model_cfg.n_layers):
            if (i + 1) % model_cfg.global_block_interval == 0:
                self.blocks.append(GlobalBlock(model_cfg))
            else:
                self.blocks.append(FactorizedBlock(model_cfg))

        # 5. Output heads
        self.output_heads = OutputHeads(model_cfg)

        # 6. Frame-cadence conditioning: a per-window log(dt · dt_ref_fps) embedding
        # added to the shared AdaLN stream (zero-init final layer).
        self.dt_embed = DtEmbed(model_cfg.d_model)

        # 7. MANO helper (frozen, for loss-time FK — not used in forward)
        self.mano_helper = MANOHelper(mano_root=mano_root)
        for param in self.mano_helper.parameters():
            param.requires_grad = False

    def _encode_conditioning(self, conditioning: dict[str, Tensor], hand_presence: Tensor, origin: Tensor | None = None):
        """Run the conditioning encoder → (cond_tokens, cond_mask).

        Depends only on `conditioning`, hand_presence and the per-window centering
        origin (not on x_t or t), so the output is identical across all Euler steps
        and noise samples of a window; see encode_conditioning() for the cache entry
        point used by the samplers.
        """
        return self.conditioning_encoder(
            conditioning.get("betas_left"),
            conditioning.get("betas_right"),
            conditioning["obj_scale"],
            hand_presence,
            scene_features=conditioning.get("scene_features"),
            origin=origin,
            obj_embedding_bf16=conditioning.get("obj_embedding_bf16"),
            obj_embedding_fp32=conditioning.get("obj_embedding_fp32"),
        )

    def encode_conditioning(self, conditioning: dict[str, Tensor], presence_mask: Tensor, origin: Tensor | None = None) -> tuple:
        """Precompute the cacheable conditioning for a sampler (see _encode_conditioning).

        Derives hand_presence exactly as forward() does and returns the
        (cond_tokens, cond_mask) tuple to pass back as forward(cond_cache=...).
        The caller passes the same centering origin the state tokens use.
        """
        hand_presence = conditioning.get("hand_presence")
        if hand_presence is None:
            hand_presence = torch.stack([presence_mask[:, :, 0].any(dim=1), presence_mask[:, :, 1].any(dim=1)], dim=1)
        return self._encode_conditioning(conditioning, hand_presence, origin=origin)

    def forward(
        self,
        x_t: dict[str, Tensor],
        t: Tensor,
        obs_mask: Tensor,
        presence_mask: Tensor,
        conditioning: dict[str, Tensor],
        origin: Tensor | None = None,
        cond_cache: tuple | None = None,
        dt: Tensor | None = None,
        anchor_feats: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass through the GTF model.

        Args:
            x_t: Dict of per-type noised tensors in Δ-codec space:
                - left_state: (B, T, 99)
                - right_state: (B, T, 99)
                - obj_state: (B, T, 9)
                - left_rel: (B, T, 14)
                - right_rel: (B, T, 14)
            t: (B,) flow timestep in [0, 1].
            obs_mask: (B, T, 5) bool. True = observed (context).
            presence_mask: (B, T, 5) bool. True = entity present.
            conditioning: Dict with segment-level data:
                - obj_embedding_bf16: (B, 1024) cached Uni3D embedding computed under
                  CUDA bf16 autocast (used when bf16 autocast is active)
                - obj_embedding_fp32: (B, 1024) cached Uni3D embedding computed in fp32
                  (used when autocast is off)
                - betas_left: (B, 10) or None
                - betas_right: (B, 10) or None
                - obj_scale: (B, 3)
                - hand_presence: (B, 2) bool, optional — derived from presence_mask if absent
                - confidence_features: (B, T, 5, 7) or None (None → neutral 0.5)
                - scene_features: list[dict | None] of per-sample scene point clouds
            origin: (B, 3) centering origin; the scene point cloud is shifted into
                the states' centered frame with it.
            cond_cache: Optional precomputed encode_conditioning() output.
            dt: (B, T-1) elapsed seconds of each frame transition. Drives the
                physical-time position embedding and the cadence embedding; None
                means the dt_ref_fps reference cadence.
            anchor_feats: Optional anchor side-channel inputs from
                ``delta_codec.build_anchor_feats`` (observed absolute states).

        Returns:
            Dict with:
                - v_pred: dict {left_state(B,T,99), right_state(B,T,99),
                    obj_state(B,T,9), left_rel(B,T,14), right_rel(B,T,14)}
                - contact_logits: (B, T, 2, 5) raw logits for BCE
                - phase_logits: (B, T, 5) raw logits of the phase head (no loss)
        """
        # 1. Derive hand_presence from presence_mask if not provided
        hand_presence = conditioning.get("hand_presence")
        if hand_presence is None:
            hand_presence = torch.stack(
                [
                    presence_mask[:, :, 0].any(dim=1),  # left hand ever present
                    presence_mask[:, :, 1].any(dim=1),  # right hand ever present
                ],
                dim=1,
            )  # (B, 2)

        # 2. Input embedding: per-type tensors → (B, T, 5, D) token grid. The per-sample
        # transition durations drive the temporal position embedding.
        x = self.input_embedding(
            x_t,
            obs_mask,
            presence_mask,
            conditioning.get("confidence_features"),
            anchor_feats=anchor_feats,
            frame_dt=dt,
        )

        # 3. Conditioning tokens: (B, N_cond, D), (B, N_cond) bool.
        if cond_cache is not None:
            cond_tokens, cond_mask = cond_cache
        else:
            cond_tokens, cond_mask = self._encode_conditioning(conditioning, hand_presence, origin=origin)

        # 4. AdaLN conditioning: flow timestep embedding plus the cadence embedding of the
        # window's mean transition duration over valid frames. dt=None gives a zero
        # log-ratio (the reference cadence) and keeps dt_embed in the autograd graph.
        t_emb = self.timestep_embedder(t)
        dt_summary = summarize_frame_dt(dt, presence_mask)
        t_emb = t_emb + self.dt_embed(log_dt_ratio(dt_summary, self.dt_ref_fps, t_emb))

        # 5. Block stack — both block types have identical forward signatures.
        for block in self.blocks:
            x = block(x, cond_tokens, cond_mask, t_emb, presence_mask)

        # 6. Output heads: (B, T, 5, D) → v_pred dict + aux logits
        return self.output_heads(x, presence_mask)
