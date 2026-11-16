"""Training losses for GTF.

Token losses act on the generated slots only: ``loss_mask = ~obs_mask & presence_mask & valid_frames``.
Per-window terms are averaged inside each window first and then over the windows that carry mass
(``ddp_window_mean``), so a long 30 Hz window does not outweigh a short 7.5 Hz one.

- ``combined_loss``: component-weighted flow matching + the auxiliary contact BCE.
- ``loss_energy_score``: energy score on M sampled rollouts, plus support-surface penetration of the
  same samples.
- ``loss_attachment``: decoded hand pose in the object body frame over GT contact runs.

The decoded x0 losses (position, rotation, articulation, smoothness, anatomy, MANO prior, decoded
contact) are computed in ``gtf/training/trainer.py``.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from gtf.model.flow.es_fast.search import pen_below_support
from gtf.model.flow.matching import TOKEN_SLOTS, TOKEN_TYPES
from gtf.model.repr.canonicalize import _LEFT_MIRROR_DIAG
from gtf.model.repr.rotation import rotation_6d_to_matrix
from gtf.model.repr.timing import normalize_edge_dt
from gtf.model.repr.tokenizer import HAND_CAM_T_SLICE, HAND_ORIENT_SLICE, HAND_POSE_SLICE, OBJ_ROTATION_SLICE, OBJ_TRANSLATION_SLICE

_STATE_NAMES = ("left_state", "right_state", "obj_state")
_TRANSLATION_SLICE = {"left_state": HAND_CAM_T_SLICE, "right_state": HAND_CAM_T_SLICE, "obj_state": OBJ_TRANSLATION_SLICE}


def ddp_window_mean(per_window: Tensor, has: Tensor) -> Tensor:
    """Global mean of a per-window quantity over the windows with mass, exact under DDP.

    per_window: (B,) values (may be NaN where ``~has``; those entries are dropped by indexing).
    has: (B,) bool.

    A plain ``per_window[has].mean()`` is a per-rank mean, and DDP's gradient averaging would then
    weight every rank equally whatever its window count. With per-rank sum S_r and count C_r this
    returns ``W * S_r / sum_r C_r``: after DDP's 1/W averaging both the value and the gradient are
    the mean over the global window set. The count all-reduce is detached and runs on every rank
    (a rank without windows contributes C_r = 0 and a grad-connected zero), so ranks never desync.
    Single process: S_r / C_r.
    """
    if bool(has.any()):
        s_r = per_window[has].sum()
    else:
        s_r = torch.nan_to_num(per_window).sum() * 0.0  # grad-connected finite zero (DDP static graph)
    c_r = has.sum().to(torch.float32)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world = float(torch.distributed.get_world_size())
        c_g = c_r.detach().clone()
        torch.distributed.all_reduce(c_g, op=torch.distributed.ReduceOp.SUM)
        return world * s_r / c_g.clamp(min=1.0)
    return s_r / c_r.clamp(min=1.0)


# Named component streams of the weighted flow-matching objective: (group, token type, channels).
_FM_STREAMS: tuple[tuple[str, str, slice], ...] = (
    ("hand_rot", "left_state", HAND_ORIENT_SLICE),
    ("hand_trans", "left_state", HAND_CAM_T_SLICE),
    ("hand_pose", "left_state", HAND_POSE_SLICE),
    ("hand_rot", "right_state", HAND_ORIENT_SLICE),
    ("hand_trans", "right_state", HAND_CAM_T_SLICE),
    ("hand_pose", "right_state", HAND_POSE_SLICE),
    ("obj_rot", "obj_state", OBJ_ROTATION_SLICE),
    ("obj_trans", "obj_state", OBJ_TRANSLATION_SLICE),
    ("relation", "left_rel", slice(None)),
    ("relation", "right_rel", slice(None)),
)


def loss_fm_per_sample(v_pred: dict[str, Tensor], v_target: dict[str, Tensor], loss_mask: Tensor, fm_weights: dict[str, float]) -> Tensor:
    """Per-window component-weighted flow-matching MSE, (B,).

    Per window and component group (hand_trans / hand_rot / hand_pose / obj_trans / obj_rot /
    relation): masked per-element MSE over the group's channels, averaged over the group's active
    streams (e.g. both hands), then the weighted mean ``sum(w * L) / sum(w)`` over the window's
    active groups. Missing weights default to 1. A window with no active group returns NaN.
    """
    B = loss_mask.shape[0]
    device = loss_mask.device
    group_sums: dict[str, Tensor] = {}
    group_counts: dict[str, Tensor] = {}
    for group, name, sl in _FM_STREAMS:
        k = TOKEN_SLOTS[name]
        mask_k = loss_mask[:, :, k]  # (B, T)
        diff_sq = (v_pred[name][..., sl] - v_target[name][..., sl]).pow(2).mean(dim=-1)  # (B, T)
        cnt = mask_k.sum(dim=1)
        s = (diff_sq * mask_k).sum(dim=1)
        val = torch.where(cnt > 0, s / cnt.clamp(min=1), torch.zeros_like(s))
        has = (cnt > 0).to(val.dtype)
        group_sums[group] = group_sums.get(group, torch.zeros(B, device=device)) + val
        group_counts[group] = group_counts.get(group, torch.zeros(B, device=device)) + has
    num = torch.zeros(B, device=device)
    den = torch.zeros(B, device=device)
    for g, sm in group_sums.items():
        cnt_g = group_counts[g]
        active = cnt_g > 0
        gl = torch.where(active, sm / cnt_g.clamp(min=1.0), torch.zeros_like(sm))
        w = float(fm_weights.get(g, 1.0))
        num = num + w * gl * active.to(num.dtype)
        den = den + w * active.to(den.dtype)
    per = num / den.clamp(min=1e-8)
    return torch.where(den > 0, per, torch.full_like(per, float("nan")))


def loss_aux_contact(
    contact_logits: Tensor,
    contact_labels: Tensor,
    loss_mask: Tensor,
    presence_mask: Tensor,
    valid_frames: Tensor,
) -> Tensor:
    """Per-finger contact BCE of the contact head, averaged per window, then over windows.

    Args:
        contact_logits: (B, T, 2, 5) raw logits (left/right hand x 5 fingers).
        contact_labels: (B, T, 2, 5) bool ground truth.
        loss_mask: (B, T, 5) bool; a hand's frames count where its relation slot is generated.
        presence_mask: (B, T, 5) bool; the hand and the object must both be present.
        valid_frames: (B, T) bool.
    """
    B = contact_logits.shape[0]
    # The grad-connected zero keeps the contact head in the DDP graph when no frame qualifies.
    l_vec = contact_logits.sum() * 0.0 + torch.zeros(B, device=contact_logits.device, dtype=contact_logits.dtype)
    n_vec = torch.zeros(B, device=contact_logits.device, dtype=contact_logits.dtype)
    for h in range(2):
        rel_slot = 3 + h
        mask = (loss_mask[:, :, rel_slot] & presence_mask[:, :, h] & presence_mask[:, :, 2] & valid_frames).unsqueeze(-1).float()  # (B, T, 1)
        bce = F.binary_cross_entropy_with_logits(contact_logits[:, :, h, :], contact_labels[:, :, h, :].float(), reduction="none")  # (B, T, 5)
        l_vec = l_vec + (bce * mask).sum(dim=(1, 2))
        n_vec = n_vec + mask.sum(dim=(1, 2)) * 5.0
    has = n_vec > 0
    return ddp_window_mean(l_vec / n_vec.clamp(min=1.0), has)


def combined_loss(
    v_pred: dict[str, Tensor],
    v_target: dict[str, Tensor],
    loss_mask: Tensor,
    presence_mask: Tensor,
    valid_frames: Tensor,
    contact_labels: Tensor,
    contact_logits: Tensor,
    phase_logits: Tensor,
    lambda_aux_contact: float,
    fm_weights: dict[str, float],
) -> tuple[Tensor, dict[str, Tensor]]:
    """Flow matching + weighted auxiliary contact BCE.

    Args:
        v_pred, v_target: {token type: (B, T, D)} predicted / target velocity.
        loss_mask: (B, T, 5) bool.
        presence_mask: (B, T, 5) bool.
        valid_frames: (B, T) bool.
        contact_labels, contact_logits: (B, T, 2, 5).
        phase_logits: (B, T, 5) output of the phase head, which has no loss; a zero term keeps its
            parameters in the DDP graph.
        lambda_aux_contact: contact BCE weight (may follow a ramp schedule).
        fm_weights: flow-matching component weights (see ``loss_fm_per_sample``).

    Returns:
        (total, {"loss_fm", "loss_aux_contact", "loss_total"}).
    """
    per = loss_fm_per_sample(v_pred, v_target, loss_mask, fm_weights)
    l_fm = ddp_window_mean(per, ~torch.isnan(per))
    loss_dict = {"loss_fm": l_fm}
    total = l_fm

    l_aux_contact = loss_aux_contact(contact_logits, contact_labels, loss_mask, presence_mask, valid_frames)
    loss_dict["loss_aux_contact"] = l_aux_contact
    total = total + lambda_aux_contact * l_aux_contact

    total = total + phase_logits.sum() * 0.0

    loss_dict["loss_total"] = total
    return total, loss_dict


# Energy score on sampled rollouts


def _make_rollout_step_fn(
    model: torch.nn.Module,
    obs_mask: Tensor,
    presence_mask: Tensor,
    conditioning: dict,
    origin: Tensor,
    phys_dt: Tensor,
    cond_cache,
    anchor_feats: dict,
):
    """One checkpointable Euler-step velocity call.

    The returned callable maps (x_left, x_right, x_obj, x_lrel, x_rrel, t_k) to the five predicted
    velocities. Everything constant across steps is closure-captured: the masks, the conditioning
    and its cached encoding, the centering origin, the physical frame intervals ``phys_dt`` (the
    model's dt conditioning, not the flow-time step) and the observed-anchor features.
    """

    def step_fn(x_left, x_right, x_obj, x_lrel, x_rrel, t_k):
        x = {
            "left_state": x_left,
            "right_state": x_right,
            "obj_state": x_obj,
            "left_rel": x_lrel,
            "right_rel": x_rrel,
        }
        out = model(x, t_k, obs_mask, presence_mask, conditioning, origin=origin, dt=phys_dt, cond_cache=cond_cache, anchor_feats=anchor_feats)
        v = out["v_pred"]
        return (v["left_state"], v["right_state"], v["obj_state"], v["left_rel"], v["right_rel"])

    return step_fn


def _es_states(x_norm: dict[str, Tensor], presence_mask: Tensor, codec, codec_dt: Tensor) -> dict[str, Tensor]:
    """Integrate codec tokens to the three (B, T, D) states: centered meters and 6D rotations."""
    return codec.decode_state_fast(x_norm, presence_mask, codec_dt)


def _es_positions(x_norm: dict[str, Tensor], presence_mask: Tensor, codec, codec_dt: Tensor) -> dict[str, Tensor]:
    """Per-entity (B, T, 3) translations in centered meters."""
    full = _es_states(x_norm, presence_mask, codec, codec_dt)
    return {name: full[name][:, :, _TRANSLATION_SLICE[name]] for name in _STATE_NAMES}


def _es_displacement(
    x_norm: dict[str, Tensor],
    loss_mask: Tensor,
    present_frames: Tensor,
    codec_dt: Tensor,
    codec,
    presence_mask: Tensor,
    valid_frames: Tensor,
    rel_weight: float,
    path_points: int,
    ref_pos: dict[str, Tensor],
    rot_weight: float,
    ref_full: dict[str, Tensor] | None,
    dense: bool = False,
) -> Tensor:
    """Joint displacement-from-present statistic in meters, (B, D).

    Displacements are taken from the present-frame position of the clean chain (``ref_pos``), so a
    sample's integration drift is not cancelled by its own drifted reference.

    Landmark form (``dense=False``), per state entity (left hand, right hand, object): the position
    at P = ``path_points`` intermediate qualifying frames (fractions k/(P+1) through the sorted
    generated frames) and at the endpoint (last generated frame), minus the present-frame position.
    Entities without a generated frame contribute zeros in both prediction and GT.

    ``rel_weight > 0`` appends, per hand, the relative hand-object displacement
    ``(p_h(f) - p_obj(f)) - (p_h(pf) - p_obj(pf))`` scaled by ``rel_weight``. Frames qualify where the
    hand-object relation slot is generated and the object is present and valid (an observed member
    contributes its clean position). The left hand is flipped to physical coordinates first (left
    cam_t lives in mirrored right-model coordinates; the constant root-joint offset cancels).

    ``rot_weight > 0`` appends per-entity endpoint relative rotations R_pf^T R_end, embedded as the
    flattened (R_rel - I) / sqrt(2): continuous through 180 degrees and close to the angle for
    small rotations.

    Dense form (``dense=True``): every generated frame instead of the landmarks, for the
    translation and relative blocks only, each block scaled by sqrt((P + 1) / n_frames) so a
    constant per-frame displacement has the same squared-distance budget as the P + 1 landmarks.

    The statistic dimension is static (zero-filled where frames are missing), and an energy score
    over any fixed statistic map stays proper.
    """
    B, T, _ = loss_mask.shape
    device = x_norm["obj_state"].device
    pos = _es_positions(x_norm, presence_mask, codec, codec_dt)
    ref = ref_pos

    b_idx = torch.arange(B, device=device)
    pf = present_frames.clamp(min=0, max=T - 1)
    P = int(path_points)
    idx_range = torch.arange(T, device=device).view(1, T)

    def _qual_frames(qmask: Tensor) -> tuple[list[Tensor], Tensor]:
        """Frame indices [path_1..path_P, endpoint] (each (B,)) and ``has`` (B,) for a (B, T) mask."""
        has = qmask.any(dim=1)
        ef = torch.where(qmask, idx_range, torch.full_like(idx_range, -1)).max(dim=1).values.clamp(min=0)
        frames: list[Tensor] = []
        if P > 0:
            n_qual = qmask.sum(dim=1)  # (B,)
            csum = torch.cumsum(qmask.long(), dim=1)  # (B, T) rank of each qualifying frame
            for kk in range(1, P + 1):
                frac = kk / (P + 1)
                jdx = torch.round(frac * (n_qual.float() - 1).clamp(min=0)).long()
                rank = (jdx + 1).clamp(min=1)
                match = (csum == rank.unsqueeze(1)) & qmask  # one True per window with frames
                frames.append(match.float().argmax(dim=1))
        frames.append(ef)
        return frames, has

    def _full_block(displacement: Tensor, qmask: Tensor) -> Tensor:
        # torch.where, not a product: inactive padding may hold NaNs.
        count = qmask.sum(dim=1).clamp(min=1).to(displacement.dtype)
        scale = ((P + 1) / count).sqrt().view(B, 1, 1)
        values = torch.where(qmask.unsqueeze(-1), displacement, torch.zeros_like(displacement))
        return (values * scale).flatten(1)

    parts: list[Tensor] = []
    for name in _STATE_NAMES:
        m = loss_mask[:, :, TOKEN_SLOTS[name]]  # (B, T)
        pcol = pos[name]
        p_pf = ref[name][b_idx, pf]
        if dense:
            parts.append(_full_block(pcol - p_pf.unsqueeze(1), m))
            continue
        frames, has = _qual_frames(m)
        hcol = has.unsqueeze(-1).to(pcol.dtype)
        for fr in frames:
            parts.append((pcol[b_idx, fr] - p_pf) * hcol)

    if rel_weight > 0.0:
        obj_meaningful = presence_mask[:, :, TOKEN_SLOTS["obj_state"]].bool() & valid_frames.bool()  # (B, T)
        p_obj = pos["obj_state"]
        p_obj_pf = ref["obj_state"][b_idx, pf]
        for name in ("left_state", "right_state"):
            rel_slot = TOKEN_SLOTS["left_rel" if name == "left_state" else "right_rel"]
            rel_q = loss_mask[:, :, rel_slot] & obj_meaningful
            p_h = pos[name]
            p_h_ref = ref[name][b_idx, pf]
            if name == "left_state":
                mflip = torch.tensor([-1.0, 1.0, 1.0], device=p_h.device, dtype=p_h.dtype)
                p_h = p_h * mflip
                p_h_ref = p_h_ref * mflip
            rel_pf = p_h_ref - p_obj_pf
            if dense:
                parts.append(_full_block(p_h - p_obj - rel_pf.unsqueeze(1), rel_q) * rel_weight)
                continue
            frames, has_rel = _qual_frames(rel_q)
            hcol = has_rel.unsqueeze(-1).to(p_h.dtype)
            for fr in frames:
                rel_f = p_h[b_idx, fr] - p_obj[b_idx, fr]
                parts.append((rel_f - rel_pf) * rel_weight * hcol)

    if rot_weight > 0.0:
        full = _es_states(x_norm, presence_mask, codec, codec_dt)
        eye = torch.eye(3, device=x_norm["obj_state"].device, dtype=x_norm["obj_state"].dtype)
        for name in _STATE_NAMES:
            sl = OBJ_ROTATION_SLICE if name == "obj_state" else HAND_ORIENT_SLICE
            m = loss_mask[:, :, TOKEN_SLOTS[name]]
            has = m.any(dim=1)
            ef = torch.where(m, idx_range, torch.full_like(idx_range, -1)).max(dim=1).values.clamp(min=0)
            R_ef = rotation_6d_to_matrix(full[name][b_idx, ef][:, sl])
            R_pf = rotation_6d_to_matrix(ref_full[name][b_idx, pf][:, sl])
            R_rel = R_pf.transpose(-1, -2) @ R_ef  # (B, 3, 3)
            emb = ((R_rel - eye) * (1.0 / math.sqrt(2.0))).flatten(-2)  # (B, 9)
            parts.append(emb * rot_weight * has.unsqueeze(-1).to(emb.dtype))

    return torch.cat(parts, dim=-1)


def _energy_score(X: list[Tensor], y: Tensor) -> Tensor:
    """Unbiased per-window energy score (1/M) sum ||X_i - y|| - 1/2 mean_{i<j} ||X_i - X_j||, (B,)."""
    M = len(X)
    acc = 0.0
    for i in range(M):
        acc = acc + (X[i] - y).norm(dim=-1)
    term1 = acc / M
    spread = 0.0
    n_pairs = 0
    for i in range(M):
        for j in range(i + 1, M):
            spread = spread + (X[i] - X[j]).norm(dim=-1)
            n_pairs += 1
    term2 = spread / max(n_pairs, 1)
    return term1 - 0.5 * term2


# A scene point is a horizontal support surface when |n_y| >= cos(30 deg). The normal's sign is
# noisy, so only its absolute up-component is used; "up" is gravity-up (-y in the model frame).
_SUPPORT_COS_HORIZ = 0.8660254037844387


def _support_penetration_states(
    full_states: dict[str, Tensor],
    loss_mask: Tensor,
    origin: Tensor,
    scene_list: list,
    obj_verts_list: list,
    obj_scale: Tensor,
    source_list: list,
    allow_sources: set,
    margin_m: float,
    band_r: float,
    topk: int,
) -> tuple[Tensor, Tensor]:
    """Per-window one-sided support penetration of the posed object mesh.

    Every object vertex, scaled and posed by the decoded per-frame (R, t) at each generated object
    frame, is checked against the present-frame scene's horizontal support points (see
    ``pen_below_support``). With ``topk > 0`` only the k deepest vertices of each frame count.
    Windows from sources outside ``allow_sources`` or without support points are skipped.

    States are centered meters; adding ``origin`` back gives the gravity-aligned present-camera
    frame the scene points live in.

    Returns (sum (B,), count (B,)).
    """
    device = full_states["obj_state"].device
    B = full_states["obj_state"].shape[0]
    sum_vec = torch.zeros(B, device=device, dtype=torch.float32)
    cnt_vec = torch.zeros(B, device=device, dtype=torch.float32)

    R_obj = rotation_6d_to_matrix(full_states["obj_state"][..., OBJ_ROTATION_SLICE].float())  # (B, T, 3, 3)
    t_obj = full_states["obj_state"][..., OBJ_TRANSLATION_SLICE].float()  # (B, T, 3) centered

    for b in range(B):
        if str(source_list[b]) not in allow_sources:
            continue
        scene = scene_list[b]
        if scene is None:
            continue
        p_scene = scene["P_scene"].to(device).float()  # (N, 3)
        normals = scene["scene_normals"].to(device).float()
        support = p_scene[normals[:, 1].abs() >= _SUPPORT_COS_HORIZ]  # (Ns, 3)
        if support.shape[0] == 0:
            continue
        org = origin[b].to(device).float()

        obj_frames = loss_mask[b, :, TOKEN_SLOTS["obj_state"]].nonzero(as_tuple=True)[0]
        if obj_frames.numel() > 0:
            vc = obj_verts_list[b].to(device).float()  # (V, 3) canonical rest pose
            vs = vc * obj_scale[b].to(device).float()  # (V, 3) metric canonical
            r_f = R_obj[b, obj_frames]  # (F, 3, 3)
            t_f = t_obj[b, obj_frames] + org  # (F, 3) world
            v_world = torch.einsum("fij,pj->fpi", r_f, vs) + t_f.unsqueeze(1)  # (F, V, 3)
            pen = pen_below_support(v_world, support, margin_m, band_r)  # (F, V)
            if topk > 0:
                k = min(int(topk), pen.shape[1])
                pen = pen.topk(k, dim=1).values  # (F, k)
            sum_vec[b] = sum_vec[b] + pen.sum()
            cnt_vec[b] = cnt_vec[b] + float(pen.numel())

    return sum_vec, cnt_vec


def loss_energy_score(
    model: torch.nn.Module,
    x0_norm: dict[str, Tensor],
    obs_mask: Tensor,
    presence_mask: Tensor,
    valid_frames: Tensor,
    conditioning: dict,
    present_frames: Tensor,
    M: int,
    K: int,
    origin: Tensor,
    phys_dt: Tensor,
    codec,
    codec_dt: Tensor,
    es_rel_weight: float,
    es_path_points: int,
    es_rot_weight: float,
    anchor_feats: dict,
    support_pen: dict | None,
    es_dense_weight: float,
) -> Tensor | tuple[Tensor, Tensor]:
    """Proper energy score on M full-chain model samples.

    ES(P, y) = (1/M) sum_i ||X_i - y|| - 1/(2 M (M-1)) sum_{i != j} ||X_i - X_j||   (unbiased, M >= 2)

    X_i are displacement statistics (``_es_displacement``) of M independent K-step Euler
    generations from pure noise at the generated slots, and y is the statistic of the clean
    tokens. The expected score is minimized only when the sample law equals the true conditional
    law (Gneiting and Raftery 2007): collapsed samples lose the spread bonus, over-dispersed ones
    pay in the first term. The joint statistic also rewards hand-object co-movement.

    The loss is the landmark score (with endpoint rotations) plus ``es_dense_weight`` times the
    dense translation/relative score; both use the same M samples, and the dense score has no
    rotation block so endpoint rotations are not counted twice.

    Gradients flow through all M x K model calls, each recomputed in backward (activation
    checkpointing). The conditioning encoding is computed once and reused across calls.

    Args:
        x0_norm: clean codec tokens (dict per token type).
        present_frames: (B,) present-frame index of each window.
        origin: (B, 3) centering origin.
        phys_dt: physical frame intervals for the model's dt conditioning.
        codec, codec_dt: the delta codec and its (B, T-1) edge durations.
        anchor_feats: observed-anchor side-channel features for the model.
        support_pen: None, or the support-penetration context (keys: scene_list, obj_verts_list,
            obj_scale, source_list, allow_sources, margin_m, band_r, topk). When given, the same
            samples are also scored for support penetration and ``(es, support_pen)`` is returned.

    Returns:
        The scalar ES, or ``(es, support_pen)``.
    """
    assert M >= 2, "energy score spread term needs M >= 2 samples"
    if int(es_path_points) < 0:
        raise ValueError("es_path_points must be nonnegative")
    if not math.isfinite(es_dense_weight) or es_dense_weight < 0.0:
        raise ValueError("es_dense_weight must be finite and nonnegative")
    device = x0_norm["obj_state"].device
    B = x0_norm["obj_state"].shape[0]

    update_mask = (~obs_mask & presence_mask).unsqueeze(-1).float()  # (B, T, 5, 1)
    cond_cache = model.encode_conditioning(conditioning, presence_mask, origin=origin)
    step_fn = _make_rollout_step_fn(model, obs_mask, presence_mask, conditioning, origin, phys_dt, cond_cache, anchor_feats)
    flow_dt = 1.0 / K

    samples = []
    for _ in range(M):
        # Pure noise at generated slots, clean values elsewhere.
        x = {}
        for k, name in enumerate(TOKEN_TYPES):
            noise = torch.randn_like(x0_norm[name])
            keep = update_mask[:, :, k, :]
            x[name] = keep * noise + (1.0 - keep) * x0_norm[name]
        for step in range(K):
            t_k = torch.full((B,), step * flow_dt, device=device)
            v_tuple = torch.utils.checkpoint.checkpoint(step_fn, x["left_state"], x["right_state"], x["obj_state"], x["left_rel"], x["right_rel"], t_k, use_reentrant=False)
            new_x = {}
            for k, name in enumerate(TOKEN_TYPES):
                new_x[name] = x[name] + update_mask[:, :, k, :] * v_tuple[k] * flow_dt
            x = new_x
        samples.append(x)

    # Decoding, rotations and norms run in fp32.
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    with torch.amp.autocast(amp_device, enabled=False):
        loss_mask = ~obs_mask & presence_mask & valid_frames.unsqueeze(-1)
        x0_f = {k: v.float() for k, v in x0_norm.items()}
        # One clean-chain reference for every statistic (a constant).
        ref_pos = {k: v.detach() for k, v in _es_positions(x0_f, presence_mask, codec, codec_dt).items()}
        ref_full = None
        if es_rot_weight > 0.0:
            ref_full = {k: v.detach() for k, v in _es_states(x0_f, presence_mask, codec, codec_dt).items()}
        stat_kw = dict(
            codec=codec,
            presence_mask=presence_mask,
            valid_frames=valid_frames,
            rel_weight=es_rel_weight,
            path_points=es_path_points,
            ref_pos=ref_pos,
            rot_weight=es_rot_weight,
            ref_full=ref_full,
        )
        y = _es_displacement(x0_f, loss_mask, present_frames, codec_dt, **stat_kw)  # (B, D)
        X = [_es_displacement({k: v.float() for k, v in s.items()}, loss_mask, present_frames, codec_dt, **stat_kw) for s in samples]

        has_any = loss_mask[:, :, :3].any(dim=(1, 2)).float()  # windows with a generated state entity
        es = _energy_score(X, y) * has_any
        if es_dense_weight > 0.0:
            # A separate summand, so the dense vector's norm does not attenuate the landmark gradients.
            dense_kw = dict(stat_kw, rot_weight=0.0, dense=True)
            y_dense = _es_displacement(x0_f, loss_mask, present_frames, codec_dt, **dense_kw)
            X_dense = [_es_displacement({k: v.float() for k, v in s.items()}, loss_mask, present_frames, codec_dt, **dense_kw) for s in samples]
            es_dense = _energy_score(X_dense, y_dense) * has_any
            es = es + es_dense_weight * es_dense
        denom = has_any.sum().clamp(min=1.0)
        es_scalar = es.sum() / denom

    if support_pen is None:
        return es_scalar

    # Support-surface penetration of the same samples (no extra model calls): per window, the
    # mean penetration depth in meters over every evaluated vertex of all M samples.
    with torch.amp.autocast(amp_device, enabled=False):
        total_pen = torch.zeros(loss_mask.shape[0], device=device, dtype=torch.float32)
        total_cnt = torch.zeros(loss_mask.shape[0], device=device, dtype=torch.float32)
        for s in samples:
            full = _es_states({k: v.float() for k, v in s.items()}, presence_mask, codec, codec_dt)
            sp, cnt = _support_penetration_states(
                full,
                loss_mask,
                origin,
                support_pen["scene_list"],
                support_pen["obj_verts_list"],
                support_pen["obj_scale"],
                support_pen["source_list"],
                support_pen["allow_sources"],
                support_pen["margin_m"],
                support_pen["band_r"],
                int(support_pen["topk"]),
            )
            total_pen = total_pen + sp
            total_cnt = total_cnt + cnt
        has = total_cnt > 0
        support_pen_val = ddp_window_mean(total_pen / total_cnt.clamp(min=1.0), has)
    return es_scalar, support_pen_val


# Decoded object-frame attachment.
#
# The hand is supervised in the OBJECT BODY frame, q(t) = R_o(t)^T (p(t) - p_o(t)), which a
# co-rotating grasp holds constant (a world-frame offset changes by (R - I) r under object
# rotation and would penalize correct in-hand co-rotation).
#   * Coupling: one residual on q gives equal and opposite gradients on the hand and object
#     translations plus a term on R_o.
#   * Placement: object-frame position fixes where the fingers sit relative to the object,
#     inside or outside, which an unsigned surface distance cannot.
#   * GT matching, not a barrier: the target is the GT q, so sliding, in-hand rotation, regrasp
#     and release stay possible, and there is no conflict with flow matching at GT contacts.
# Frames come from contiguous GT-contact runs, because keeping a grasp over time is the target.


def hand_in_object_frame(orient_6d: Tensor, pose_6d: Tensor, betas: Tensor, cam_t: Tensor, side: str, obj_rot_6d: Tensor, obj_trans: Tensor, mano_helper):
    """Hand pose in the object body frame: (q_wrist (T, 3), rot_in_obj (T, 3, 3), tips_canon (T, 5, 3)).

    Matches ``compute_relation_features`` dims 0:9 (same FK root joint, same M-conjugation, same
    R_o^T), without its triangle query.

    Left-hand convention: MANO runs the right model and mirrors the FK output by M = diag(-1, 1, 1),
    so ``joints``/``fingertips`` are physical while the raw ``cam_t``/orient live in mirrored
    coordinates. The physical wrist is therefore the FK root joint (never cam_t), and the left
    parameter rotation is M-conjugated (R -> M R M) before it is expressed in the object frame.

    Centering-invariant: centering shifts the physical hand and the object by the same origin, so
    q and tips_canon are the same on centered and uncentered inputs.

    ``betas`` is (10,) per segment or (T, 10) per row.
    """
    T = orient_6d.shape[0]
    betas_t = betas.unsqueeze(0).expand(T, -1) if betas.dim() == 1 else betas
    _, joints, fingertips = mano_helper(orient_6d, pose_6d, betas_t, cam_t, side=side)
    obj_R = rotation_6d_to_matrix(obj_rot_6d)  # (T, 3, 3)
    obj_R_inv = obj_R.transpose(-1, -2)
    hand_R = rotation_6d_to_matrix(orient_6d)
    if side == "left":
        m = torch.tensor(_LEFT_MIRROR_DIAG, device=hand_R.device, dtype=hand_R.dtype)
        hand_R = m.view(1, 3, 1) * hand_R * m.view(1, 1, 3)
    q_wrist = (obj_R_inv @ (joints[:, 0] - obj_trans).unsqueeze(-1)).squeeze(-1)  # (T, 3)
    rot_in_obj = obj_R_inv @ hand_R  # (T, 3, 3)
    tips_canon = torch.einsum("tij,tfj->tfi", obj_R_inv, fingertips - obj_trans.unsqueeze(-2))  # (T, 5, 3)
    return q_wrist, rot_in_obj, tips_canon


def contact_run_frames(
    contact: Tensor,
    supervised: Tensor,
    max_frames: int,
    min_run: int,
    *,
    edge_dt: Tensor,
) -> Tensor:
    """Deterministic, physical-time-stratified frames from contiguous contact runs.

    Runs (of ``contact & supervised``) shorter than ``min_run`` are dropped; the rest are taken
    longest-first in physical time until ``max_frames`` frames are used (0 = no limit).
    ``max_frames`` is a compute budget, not a temporal prefix: a run that does not fit is covered by
    adjacent frame pairs spread uniformly in physical time, always including its first and last
    frames, so consecutive samples remain for the relative-velocity term and the covered span does
    not depend on the frame rate.

    contact, supervised: (T,) bool. edge_dt: (T-1,) seconds. Returns a sorted LongTensor.
    """
    m = (contact & supervised).nonzero(as_tuple=False).squeeze(-1)
    if m.numel() == 0:
        return m
    m_list = m.tolist()  # one device transfer instead of a sync per selected edge
    runs, start, prev = [], m_list[0], m_list[0]
    for v in m_list[1:]:
        if v != prev + 1:
            runs.append((start, prev))
            start = v
        prev = v
    runs.append((start, prev))
    runs = [r for r in runs if (r[1] - r[0] + 1) >= min_run]
    if not runs:
        return m[:0]
    if edge_dt.ndim != 1 or edge_dt.numel() != contact.numel() - 1:
        raise ValueError(f"edge_dt must be ({contact.numel() - 1},), got {tuple(edge_dt.shape)}")
    if not bool(torch.isfinite(edge_dt).all()) or not bool((edge_dt > 0).all()):
        raise ValueError("edge_dt must contain finite positive edge durations")
    # Selection is non-differentiable bookkeeping; the small farthest-point loop runs on CPU.
    edge_dt_cpu = edge_dt.detach().float().cpu()
    frame_time = torch.cat((edge_dt_cpu.new_zeros(1), edge_dt_cpu.cumsum(dim=0)))

    # Longest in physical time first; stable tie-breaks keep the selection deterministic.
    runs.sort(key=lambda r: (-float((frame_time[r[1]] - frame_time[r[0]]).item()), -(r[1] - r[0] + 1), r[0]))

    def _stratified_run(a: int, b: int, n: int) -> Tensor:
        length = b - a + 1
        if n >= length:
            return torch.arange(a, b + 1, device=contact.device)
        if n <= 0:
            return torch.empty(0, device=contact.device, dtype=torch.long)
        if n == 1:
            return torch.tensor([b], device=contact.device, dtype=torch.long)
        if n == 2:
            # Both endpoints (no velocity edge).
            return torch.tensor([a, b], device=contact.device, dtype=torch.long)
        if n == 3:
            # Both endpoints plus the late edge, where integration drift is largest.
            return torch.tensor([a, b - 1, b], device=contact.device, dtype=torch.long)

        # Farthest-point sampling of adjacent edges in physical time. The first and last edges are
        # mandatory; neighboring selected edges may share a frame, so spare capacity is filled.
        edge_idx = torch.arange(a, b)
        edge_time = 0.5 * (frame_time[a:b] + frame_time[a + 1 : b + 1])
        n_pairs = min(max(2, n // 2), int(edge_idx.numel()))
        chosen = [0, int(edge_idx.numel()) - 1]
        while len(set(chosen)) < n_pairs:
            uniq = sorted(set(chosen))
            dist = torch.stack([(edge_time - edge_time[j]).abs() for j in uniq]).amin(dim=0)
            dist[torch.tensor(uniq)] = -1
            nxt = int(dist.argmax().item())
            if nxt in uniq:
                break
            chosen.append(nxt)

        selected: set[int] = set()
        for j in sorted(set(chosen)):
            e = int(edge_idx[j].item())
            selected.update((e, e + 1))
        while len(selected) < n:
            available = [f for f in range(a, b + 1) if f not in selected]
            if not available:
                break
            sel_t = frame_time[torch.tensor(sorted(selected))]
            cand = torch.tensor(available)
            min_dist = (frame_time[cand, None] - sel_t[None, :]).abs().amin(dim=1)
            selected.add(available[int(min_dist.argmax().item())])
        return torch.tensor(sorted(selected), device=contact.device, dtype=torch.long)

    keep, budget = [], int(max_frames) if max_frames > 0 else 10**9
    for a, b in runs:
        if budget <= 0:
            break
        run_len = b - a + 1
        n = min(run_len, budget)
        # A truncated run needs at least 3 samples to cover both endpoints and keep one edge;
        # a one- or two-frame remainder would be a biased fragment.
        if n < min(run_len, 3):
            continue
        chosen = _stratified_run(a, b, n)
        if chosen.numel():
            keep.append(chosen)
            budget -= int(chosen.numel())
    if not keep:
        return m[:0]
    return torch.sort(torch.cat(keep)).values


def loss_attachment(
    *,
    dec_states: dict[str, Tensor],
    clean_states: dict[str, Tensor],
    clean_rel: dict[str, Tensor],
    sup_gate: dict[str, Tensor],
    presence_mask: Tensor,
    valid_frames: Tensor,
    betas: dict[str, Tensor],
    mano_helper,
    edge_dt: Tensor,
    contact_thr: float,
    max_frames: int,
    min_run: int,
) -> dict[str, tuple[Tensor, Tensor]]:
    """Decoded object-frame attachment over GT contact runs (see the block comment above).

        attach_pos  ||q_pred - q_gt|| + mean_f ||tip_pred_f - tip_gt_f||   wrist and fingertips in the object frame
        attach_rot  ||R_pred - R_gt||_F                                     R_o^T R_h, chordal
        attach_vel  ||dq_pred/dt - dq_gt/dt||                               over adjacent selected frames

    A (window, hand) frame is supervised where the hand or the object depends on a generated slot
    (``sup_gate[f"{side}_state"] | sup_gate["obj_state"]``), both are present, and the frame is
    valid; contact means a GT minimum fingertip distance <= ``contact_thr`` (relation dims 9:14).
    At each observed/generated boundary the clean observed neighbor is added to the geometry so the
    entry velocity jump is supervised; it does not count in the position/rotation means.

    GT q and R_o^T R_h come from the clean relation tokens (dims 0:3 and 3:9), which are
    frame-invariant; only the GT fingertips need FK (under no_grad).

    Args:
        dec_states / clean_states: {state name: (B, T, D)} uncentered decoded / clean states.
        clean_rel: {"left"/"right": (B, T, 14)} clean relation tokens.
        sup_gate: {state name: (B, T) bool}.
        betas: {"left"/"right": (B, 10)}.
        edge_dt: (B,) or (B, T-1) seconds.

    Returns {component: (sum (B,), count (B,))} for a per-window mean.
    """
    B = presence_mask.shape[0]
    T = presence_mask.shape[1]
    dev = presence_mask.device
    edge_dt_norm = normalize_edge_dt(edge_dt, n_frames=T, batch_size=B, device=dev, dtype=torch.float32, validate=False)
    # One host copy for the non-differentiable run selection; velocities use the device tensor.
    edge_dt_select = edge_dt_norm.detach().cpu()
    comps = ("attach_pos", "attach_rot", "attach_vel")
    acc: dict[str, list[Tensor]] = {c: [] for c in comps}
    cnt: dict[str, list[Tensor]] = {c: [] for c in comps}

    # Phase 1: per (window, hand) contact-run selection. ``score_fr`` counts in the
    # position/rotation means; ``geom_fr`` adds the clean observed neighbors at boundaries.
    sup_by_side = {side: sup_gate[f"{side}_state"] | sup_gate["obj_state"] for side in ("left", "right")}
    sel: dict[str, list[tuple[int, Tensor, Tensor]]] = {"left": [], "right": []}
    for b in range(B):
        for side, k in (("left", 0), ("right", 1)):
            sup = sup_by_side[side]
            contact = clean_rel[side][b, :, 9:14].float().min(dim=-1).values <= contact_thr
            supervised = sup[b] & presence_mask[b, :, k].bool() & presence_mask[b, :, 2].bool() & valid_frames[b].bool()
            fr = contact_run_frames(contact, supervised, max_frames, min_run, edge_dt=edge_dt_select[b])
            if fr.numel():
                available = contact & presence_mask[b, :, k].bool() & presence_mask[b, :, 2].bool() & valid_frames[b].bool()
                neighbours = []
                left = fr - 1
                left_ok = (fr > 0) & available[left.clamp(min=0)] & ~sup[b, left.clamp(min=0)]
                if bool(left_ok.any()):
                    neighbours.append(left[left_ok])
                right = fr + 1
                right_ok = (fr + 1 < T) & available[right.clamp(max=T - 1)] & ~sup[b, right.clamp(max=T - 1)]
                if bool(right_ok.any()):
                    neighbours.append(right[right_ok])
                geom_fr = torch.unique(torch.cat((fr, *neighbours)), sorted=True) if neighbours else fr
                sel[side].append((b, fr, geom_fr))

    # Phase 2: one fused MANO call per side (prediction and clean) over every selected frame.
    for side in ("left", "right"):
        items = sel[side]
        if not items:
            continue
        idx_b = torch.cat([torch.full_like(geom_fr, b) for b, _, geom_fr in items])
        idx_f = torch.cat([geom_fr for _, _, geom_fr in items])
        score_mask = torch.cat([torch.isin(geom_fr, score_fr) for _, score_fr, geom_fr in items])
        hn = f"{side}_state"
        st = dec_states[hn][idx_b, idx_f].float()
        ob = dec_states["obj_state"][idx_b, idx_f].float()
        bet = betas[side][idx_b].float()
        q_p, R_p, tips_p = hand_in_object_frame(st[:, 0:6], st[:, 9:99], bet, st[:, 6:9], side, ob[:, 0:6], ob[:, 6:9], mano_helper)

        rel_sel = clean_rel[side][idx_b, idx_f].float()
        q_g = rel_sel[:, 0:3]
        R_g = rotation_6d_to_matrix(rel_sel[:, 3:9])

        pos = (q_p - q_g).norm(dim=-1)
        with torch.no_grad():
            stc = clean_states[hn][idx_b, idx_f].float()
            obc = clean_states["obj_state"][idx_b, idx_f].float()
            _, _, tips_g = hand_in_object_frame(stc[:, 0:6], stc[:, 9:99], bet, stc[:, 6:9], side, obc[:, 0:6], obc[:, 6:9], mano_helper)
        pos = pos + (tips_p - tips_g).norm(dim=-1).mean(dim=-1)
        pos = pos[score_mask]
        rot = (R_p - R_g).flatten(1).norm(dim=-1)[score_mask]
        idx_b_score = idx_b[score_mask]
        ones = torch.ones_like(pos)
        z = torch.zeros(B, device=dev, dtype=torch.float32)
        acc["attach_pos"].append(z.index_add(0, idx_b_score, pos))
        cnt["attach_pos"].append(z.index_add(0, idx_b_score, ones))
        acc["attach_rot"].append(z.index_add(0, idx_b_score, rot))
        cnt["attach_rot"].append(z.index_add(0, idx_b_score, ones))

        # Relative velocity: pair consecutive rows of the same window that are consecutive frames
        # (distinct runs are separated by a gap, so a pair never straddles two runs).
        if idx_b.numel() >= 2:
            adj = (idx_b[1:] == idx_b[:-1]) & ((idx_f[1:] - idx_f[:-1]) == 1)
            if bool(adj.any()):
                vb = idx_b[1:][adj]
                dq = (q_p[1:] - q_p[:-1])[adj] - (q_g[1:] - q_g[:-1])[adj]
                vf = idx_f[1:][adj]
                h = edge_dt_norm[vb, vf - 1].to(device=dq.device, dtype=dq.dtype).clamp(min=1e-6)
                dq = dq / h.unsqueeze(-1)
                v = dq.norm(dim=-1)
                acc["attach_vel"].append(z.index_add(0, vb, v))
                cnt["attach_vel"].append(z.index_add(0, vb, torch.ones_like(v)))

    zero = torch.zeros(B, device=dev, dtype=torch.float32)
    return {c: ((torch.stack(acc[c]).sum(0) if acc[c] else zero), (torch.stack(cnt[c]).sum(0) if cnt[c] else zero)) for c in comps}
