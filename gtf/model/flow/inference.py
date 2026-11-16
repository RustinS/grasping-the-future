"""ODE inference pipeline for GTF generation.

Fixed-step Euler integration from noise (t=0) to data (t=1):
center → encode into Δ-codec space → ODE loop → decode → un-center → axis-angle.

Train/infer parity: centering and the codec transform match the training step exactly.
"""

import torch
import torch.nn as nn
from torch import Tensor

from gtf.model.flow.matching import TOKEN_TYPES
from gtf.model.repr.delta_codec import DeltaCodec, build_anchor_feats
from gtf.model.repr.present_anchor import select_present_frames
from gtf.model.repr.tokenizer import StateTokenizer, center_tokens_at_present, center_translations

__all__ = ["generate"]


@torch.no_grad()
def generate(
    model: nn.Module,
    tokens: dict[str, Tensor],
    obs_mask: Tensor,
    presence_mask: Tensor,
    conditioning: dict[str, Tensor],
    stats: dict,
    n_steps: int = 20,
    return_tokens: bool = False,
    phys_dt: Tensor | None = None,
) -> dict[str, Tensor] | tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Generate predictions via fixed-step Euler ODE integration.

    Receives unnormalized, uncentered tokenized data. Centers at the present frame,
    encodes into Δ-codec space, runs the ODE from noise to data on the masked present
    slots, decodes, un-centers, and converts to canonical axis-angle format.

    Args:
        model: GTFModel in eval mode.
        tokens: {left_state(B,T,99), right_state(B,T,99), obj_state(B,T,9),
                 left_rel(B,T,14), right_rel(B,T,14)} — unnormalized, uncentered.
        obs_mask: (B, T, 5) bool. True = observed (context).
        presence_mask: (B, T, 5) bool. True = entity present.
        conditioning: Dict with cached object embeddings, betas_left, betas_right,
            obj_scale, confidence_features, scene_features.
        stats: Normalization stats from load_stats().
        n_steps: Number of Euler integration steps.
        return_tokens: If True, also return the un-centered decoded token dict
            (includes left_rel, right_rel).
        phys_dt: (B, T-1) elapsed seconds of each frame transition (required).

    Returns:
        If return_tokens=False: Dict with canonical field names (axis-angle).
        If return_tokens=True: (decoded_dict, tokens_final_dict) where tokens_final
            contains all 5 token types in un-centered, denormalized space.
    """
    if phys_dt is None:
        raise ValueError("generate() requires phys_dt (per-transition seconds, (B, T-1))")
    device = obs_mask.device
    B = obs_mask.shape[0]

    # 1. Center at the latest observed present frame (raw meters).
    present_frames = select_present_frames(obs_mask, presence_mask)
    tokens_centered, origin = center_tokens_at_present(tokens, obs_mask, presence_mask, present_frames)

    # 2. Encode into Δ-codec space: state tokens become run-start absolutes plus
    # per-second derivatives and the obs mask transforms alongside; relations stay
    # absolute-normalized. The original obs mask is kept for the decode.
    codec = DeltaCodec(stats)
    obs_mask_orig = obs_mask
    tokens_norm = codec.encode(tokens_centered, presence_mask, phys_dt)
    obs_mask = codec.transform_masks(obs_mask, presence_mask)

    # Anchor side-channel: observed absolute positions as input features, built from
    # the clean centered tokens and the original obs mask, exactly as in training.
    anchor_feats = build_anchor_feats(tokens_centered, obs_mask_orig, stats)

    # 3. Initialize x_current: noise for masked+present, normalized values for observed.
    mask_expanded = ~obs_mask & presence_mask  # (B, T, 5) masked+present
    x_current = {}
    for k, name in enumerate(TOKEN_TYPES):
        if name not in tokens_norm:
            continue
        type_mask = mask_expanded[:, :, k].unsqueeze(-1)  # (B, T, 1)
        noise = torch.randn_like(tokens_norm[name])
        x_current[name] = torch.where(type_mask, noise, tokens_norm[name])

    # The conditioning encode depends only on `conditioning` and the centering origin,
    # so it is computed once and reused by every Euler step.
    cond_cache = model.encode_conditioning(conditioning, presence_mask, origin=origin)

    # 4. Euler ODE, t from 0 (noise) to 1 (data); observed slots are held fixed.
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((B,), i * dt, device=device)
        output = model(
            x_current,
            t,
            obs_mask,
            presence_mask,
            conditioning,
            origin=origin,
            cond_cache=cond_cache,
            dt=phys_dt,
            anchor_feats=anchor_feats,
        )
        v_use = output["v_pred"]
        for k, name in enumerate(TOKEN_TYPES):
            if name not in x_current:
                continue
            type_mask = mask_expanded[:, :, k].unsqueeze(-1).float()
            x_current[name] = x_current[name] + type_mask * v_use[name] * dt

    # 5. Decode: integrate derivatives back to centered absolute values (anchored at
    # run-start absolutes), close each masked gap onto its observed anchors and
    # restore originally-observed frames exactly (DeltaCodec.decode_with_observed).
    tokens_raw = codec.decode_with_observed(x_current, tokens_centered, obs_mask_orig, presence_mask, phys_dt)

    # 6. Un-center: center_translations subtracts, so pass -origin.
    tokens_final = center_translations(tokens_raw, -origin)

    # 7. Decode to axis-angle.
    decoded = StateTokenizer().decode(tokens_final)

    if return_tokens:
        return decoded, tokens_final
    return decoded
