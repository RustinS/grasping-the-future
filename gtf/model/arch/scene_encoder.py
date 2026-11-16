"""Scene encoder + global query branch for the scene conditioning tokens.

Pipeline per batch:
    list[dict|None] scene_features  (B entries; each dict has a 4096-point cloud)
    → pack_scene_batch              flat tensors (N_total, ...) for the backbone
    → SceneEncoder                  LitePT-S, flat (N_total, scene_feature_dim) features
    → GlobalSceneBranch             (B, 8, D) global conditioning tokens
                                    (missing rows filled with the learned baseline)

All scene clouds have N=4096 points, so flat batching needs no padding.

Backbone: LitePT-S (``third-party/litept/``, prs-eth/LitePT), ScanNet-pretrained,
v1m1 small config (5 encoder stages, 4 decoder stages of depth 0 — the decoder
only unpools). The input embedding stem is re-initialized for the 44-D input
(the pretraining input is 6-D XYZ+RGB). The body is frozen at the pretrained
values: only the stem, the output projection and GlobalSceneBranch train, and
``SceneEncoder.train()`` keeps the frozen body in eval mode (no BN running-stat
drift, no DropPath).

spconv's ``implicit_gemm`` kernel finds no algorithm for bf16 inputs under
inference_mode, so ``SceneEncoder.forward`` disables autocast around the
backbone: the spconv stack runs in fp32 inside the bf16-mixed model.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from gtf.paths import THIRD_PARTY_DIR

logger = logging.getLogger(__name__)

# The LitePT package is vendored under third-party/.
_THIRD_PARTY = str(THIRD_PARTY_DIR)
if _THIRD_PARTY not in sys.path:
    sys.path.insert(0, _THIRD_PARTY)

# LitePT-S architecture of upstream configs/scannet/semseg-litept-small-v1m1.py (prs-eth/LitePT), so the
# ScanNet-pretrained checkpoint loads. dec_depths=(0, 0, 0, 0): the decoder stages only unpool.
LITEPT_S_ARCH_KWARGS: dict = dict(
    order=("z", "z-trans", "hilbert", "hilbert-trans"),
    stride=(2, 2, 2, 2),
    enc_depths=(2, 2, 2, 6, 2),
    enc_channels=(36, 72, 144, 252, 504),
    enc_num_head=(2, 4, 8, 14, 28),
    enc_patch_size=(1024, 1024, 1024, 1024, 1024),
    enc_conv=(True, True, True, False, False),
    enc_attn=(False, False, False, True, True),
    enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
    dec_depths=(0, 0, 0, 0),
    dec_channels=(72, 72, 144, 252),
    dec_num_head=(4, 4, 8, 14),
    dec_patch_size=(1024, 1024, 1024, 1024),
    dec_conv=(False, False, False, False),
    dec_attn=(False, False, False, False),
    dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
    mlp_ratio=4,
    qkv_bias=True,
    qk_scale=None,
    attn_drop=0.0,
    proj_drop=0.0,
    drop_path=0.3,
    shuffle_orders=True,
    pre_norm=True,
    enc_mode=False,
)
LITEPT_S_OUT_DIM = 72  # = dec_channels[0]


def pack_scene_batch(
    scene_list: list[dict | None],
    *,
    device: torch.device,
    scene_semantic_dim: int = 32,
    dropout_p: float = 0.0,
    training: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Pack list[dict|None] (length B, at least one present) into flat backbone input tensors.

    Device: scene_features arrive from the collator as list[dict[str, Tensor]],
    which generic batch device transfers do not recurse into, so this function
    moves each tensor to ``device``.

    Semantic dropout: during training, zeroes the semantic slab of a whole
    segment with probability dropout_p.

    Returns:
        coord:        (B_present * N, 3) float
        feat:         (B_present * N, in_channels) float — geo + sem (zeros if missing)
        batch_idx:    (B_present * N,) long — values in [0, B_present)
        present_mask: (B,) bool — True for samples that contributed
    """
    present_mask = torch.tensor(
        [s is not None for s in scene_list],
        dtype=torch.bool,
    )

    present_dicts = [s for s in scene_list if s is not None]

    coord_list: list[Tensor] = []
    feat_list: list[Tensor] = []
    counts: list[int] = []
    for s in present_dicts:
        P = s["P_scene"].to(device)  # (N, 3)
        geo = s["scene_geo_feats"].to(device)  # (N, 12)
        sem = s.get("scene_sem_feats")
        if sem is None or sem.numel() == 0 or sem.shape[-1] == 0:
            sem = torch.zeros(P.shape[0], scene_semantic_dim, device=device, dtype=geo.dtype)
        else:
            sem = sem.to(device)
            # Pad/truncate semantic to scene_semantic_dim
            if sem.shape[-1] < scene_semantic_dim:
                pad = torch.zeros(sem.shape[0], scene_semantic_dim - sem.shape[-1], device=device, dtype=sem.dtype)
                sem = torch.cat([sem, pad], dim=-1)
            elif sem.shape[-1] > scene_semantic_dim:
                sem = sem[:, :scene_semantic_dim]

            # Training-time semantic dropout — gated per-segment
            if training and dropout_p > 0.0:
                if torch.rand(1, device=device).item() < dropout_p:
                    sem = torch.zeros_like(sem)

        coord_list.append(P)
        feat_list.append(torch.cat([geo, sem], dim=-1))  # (N, 12 + scene_semantic_dim)
        counts.append(int(P.shape[0]))

    coord = torch.cat(coord_list, dim=0)  # (N_total, 3)
    feat = torch.cat(feat_list, dim=0)  # (N_total, in_channels)

    # batch_idx: 0,0,...,0, 1,1,...,1, ...
    batch_idx = torch.repeat_interleave(
        torch.arange(len(present_dicts), device=device, dtype=torch.long),
        torch.tensor(counts, device=device, dtype=torch.long),
    )

    return coord, feat, batch_idx, present_mask.to(device)


def _load_pointcept_segmentor_weights(
    model: nn.Module,
    weights_path: str | Path,
    *,
    drop_head_prefixes: tuple[str, ...] = ("seg_head.", "cls_head.", "aux_head."),
    min_matched_frac: float = 0.8,
    backbone_name: str = "scene",
) -> dict:
    """Load Pointcept-style pretrained weights into a bare backbone.

    The LitePT ScanNet checkpoint is wrapped as DefaultSegmentorV2
    (``module.backbone.<bare_key>`` + ``module.seg_head.*``): the ``module.`` and
    ``backbone.`` prefixes are stripped and segmentation-head keys dropped.
    Keys whose shapes differ from the model (the re-initialized 44-D input stem)
    are skipped. A missing file raises.

    Args:
        model: target backbone module.
        weights_path: path to a .pth file.
        drop_head_prefixes: prefixes of segmentation-head keys to drop.
        min_matched_frac: raise if fewer than this fraction of model params match.
        backbone_name: used only in log messages.

    Returns:
        Load report (n_matched, matched_frac, n_dropped_shape, …).
    """
    ck = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    raw_sd = ck["state_dict"] if "state_dict" in ck else ck

    # Strip module. + (optional) backbone. prefixes, drop head keys
    stripped: dict[str, Tensor] = {}
    n_head_dropped = 0
    for k, v in raw_sd.items():
        new_k = k
        if new_k.startswith("module."):
            new_k = new_k[len("module.") :]
        if any(new_k.startswith(p) for p in drop_head_prefixes):
            n_head_dropped += 1
            continue
        if new_k.startswith("backbone."):
            new_k = new_k[len("backbone.") :]
        stripped[new_k] = v

    # Shape-filter — strict=False does NOT skip shape mismatches on matched keys.
    model_sd = model.state_dict()
    filtered: dict[str, Tensor] = {}
    dropped_shape: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    unexpected: list[str] = []
    for k, v in stripped.items():
        if k not in model_sd:
            unexpected.append(k)
            continue
        if model_sd[k].shape != v.shape:
            dropped_shape.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
            continue
        filtered[k] = v

    n_matched = len(filtered)
    n_total = len(model_sd)
    missing_keys = [k for k in model_sd if k not in filtered]

    model.load_state_dict(filtered, strict=False)

    load_report = dict(
        n_matched=n_matched,
        n_total_model_params=n_total,
        matched_frac=n_matched / max(n_total, 1),
        n_dropped_shape=len(dropped_shape),
        n_head_dropped=n_head_dropped,
        n_missing=len(missing_keys),
        n_unexpected=len(unexpected),
        dropped_shape_samples=dropped_shape[:3],
    )
    logger.info(
        "%s weight load: matched=%d/%d (%.1f%%), dropped_shape=%d, head_dropped=%d, missing=%d, unexpected=%d",
        backbone_name,
        load_report["n_matched"],
        load_report["n_total_model_params"],
        load_report["matched_frac"] * 100,
        load_report["n_dropped_shape"],
        load_report["n_head_dropped"],
        load_report["n_missing"],
        load_report["n_unexpected"],
    )
    if load_report["matched_frac"] < min_matched_frac:
        raise RuntimeError(
            f"{backbone_name} weight load matched only {load_report['matched_frac']:.1%} "
            f"of params (threshold {min_matched_frac:.0%}) — check checkpoint format. "
            f"Sample shape drops: {load_report['dropped_shape_samples']}"
        )
    return load_report


def _freeze_backbone_body(backbone: nn.Module, *, backbone_name: str) -> dict:
    """Set ``requires_grad=False`` on every backbone parameter except those under
    ``embedding.`` (the input stem stays trainable to absorb the 44-D input).

    Only flips ``requires_grad``; putting the body in ``eval()`` is handled by
    ``SceneEncoder._apply_body_eval_mode``. Returns (and logs) a freeze report.
    """
    n_total = 0
    n_trainable = 0
    n_stem_params = 0
    n_body_params = 0
    for name, p in backbone.named_parameters():
        n_total += 1
        if name.startswith("embedding."):
            n_trainable += 1
            n_stem_params += p.numel()
        else:
            p.requires_grad = False
            n_body_params += p.numel()
    freeze_report = dict(
        n_trainable_tensors=n_trainable,
        n_total_tensors=n_total,
        trainable_frac=n_trainable / max(n_total, 1),
        n_stem_params=n_stem_params,
        n_body_params_frozen=n_body_params,
    )
    logger.info(
        "%s freeze: trainable=%d/%d tensors (%.1f%%); stem_params=%d, body_params_frozen=%d",
        backbone_name,
        freeze_report["n_trainable_tensors"],
        freeze_report["n_total_tensors"],
        freeze_report["trainable_frac"] * 100,
        freeze_report["n_stem_params"],
        freeze_report["n_body_params_frozen"],
    )
    return freeze_report


class SceneEncoder(nn.Module):
    """LitePT-S scene backbone (frozen pretrained body) over per-point features.

    Inner attribute ``self.litept`` (native out_dim 72) followed by a linear
    ``out_proj`` to ``scene_feature_dim``.

    Forward:
        coord:     (N_total, 3)
        feat:      (N_total, in_channels)
        batch_idx: (N_total,) long  — only used for offset computation
        → returns ``(f_per_point (N_total, scene_feature_dim), offset (B_present,))``.
    """

    def __init__(
        self,
        in_channels: int = 44,
        scene_feature_dim: int = 64,
        weights_path: str | None = None,
        grid_size: float = 0.005,
    ):
        super().__init__()
        from litept.model import LitePT

        self.in_channels = int(in_channels)
        self.scene_feature_dim = int(scene_feature_dim)
        self.grid_size = float(grid_size)

        self.litept = LitePT(in_channels=self.in_channels, **LITEPT_S_ARCH_KWARGS)
        if weights_path is not None:
            _load_pointcept_segmentor_weights(self.litept, weights_path, backbone_name="LitePT-S-ScanNet")

        # Projection that pins the public output dim (72 → scene_feature_dim).
        if LITEPT_S_OUT_DIM != self.scene_feature_dim:
            self.out_proj = nn.Linear(LITEPT_S_OUT_DIM, self.scene_feature_dim)
        else:
            self.out_proj = nn.Identity()

        # Freeze the body; the stem (`embedding.*`) stays trainable. The train()
        # override below keeps the body in eval mode across model.train() calls.
        _freeze_backbone_body(self.litept, backbone_name="litept_s")
        self._apply_body_eval_mode()

    def _apply_body_eval_mode(self) -> None:
        """Put the backbone body in eval and the stem in SceneEncoder's mode.

        The body's BN running stats and DropPath stay static, while the trainable
        stem follows the outer module's mode (its BN learns 44-D running stats in
        training and uses them in evaluation).
        """
        self.litept.eval()  # body + stem → eval (recursive)
        self.litept.embedding.train(self.training)  # stem (and its children) → SceneEncoder mode

    def train(self, mode: bool = True):
        """Re-apply the body-eval policy after the default recursive train()/eval()."""
        super().train(mode)
        self._apply_body_eval_mode()
        return self

    def forward(self, coord: Tensor, feat: Tensor, batch_idx: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Returns:
            f_per_point: (N_total, scene_feature_dim)
            offset:      (B_present,) long

        Autocast is disabled around the backbone (spconv implicit_gemm has no
        working bf16 path); the output projection runs in the caller's autocast.
        """
        # offset: cumulative count per batch index
        counts = torch.bincount(batch_idx, minlength=int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else 0)
        offset = torch.cumsum(counts, dim=0)
        with torch.amp.autocast("cuda", enabled=False):
            coord_fp32 = coord.float()
            feat_fp32 = feat.float()
            inp = {"feat": feat_fp32, "coord": coord_fp32, "grid_size": self.grid_size, "offset": offset}
            out = self.litept(inp)
            # LitePT returns a Point dict-subclass with .feat
            f_per_point = out.feat if hasattr(out, "feat") else out["feat"]
        f_per_point = self.out_proj(f_per_point)
        return f_per_point, offset


class _CrossAttnBlock(nn.Module):
    """Cross-attn(q=queries, kv=points) → self-attn(queries) → FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.cross_ln_q = nn.LayerNorm(d_model)
        self.cross_ln_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.self_ln = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn_ln = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, queries: Tensor, kv: Tensor) -> Tensor:
        # queries: (B, n_queries, D)  kv: (B, N_kv, D)
        q_in = self.cross_ln_q(queries)
        kv_in = self.cross_ln_kv(kv)
        ca, _ = self.cross_attn(q_in, kv_in, kv_in, need_weights=False)
        queries = queries + ca

        q_in = self.self_ln(queries)
        sa, _ = self.self_attn(q_in, q_in, q_in, need_weights=False)
        queries = queries + sa

        queries = queries + self.ffn(self.ffn_ln(queries))
        return queries


class _CoordPE(nn.Module):
    """Learned positional encoding of 3D coordinates to d_model.

    Two-stage: Fourier-style sinusoidal expansion to 3 × 2 × n_freq bands, then
    a small MLP to d_model.
    """

    def __init__(self, d_model: int, n_freq: int = 32):
        super().__init__()
        self.n_freq = n_freq
        self.register_buffer(
            "freqs",
            torch.exp(torch.arange(n_freq, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / n_freq)),
            persistent=False,
        )
        feat_dim = 3 * 2 * n_freq
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, coord: Tensor) -> Tensor:
        # coord: (..., 3) → fourier (..., 3 * 2 * n_freq) → proj (..., D)
        parts = []
        for axis in range(3):
            c = coord[..., axis : axis + 1] * self.freqs.to(coord.dtype)  # (..., n_freq)
            parts.append(torch.sin(c))
            parts.append(torch.cos(c))
        feat = torch.cat(parts, dim=-1)
        return self.proj(feat)


class GlobalSceneBranch(nn.Module):
    """8 learned queries → 2 × [cross_attn(q=queries, kv=F_per_point), self_attn, FFN].

    Adds a learned positional encoding of P_scene to the KV side so queries can
    localize layout.

    Output: (B, n_queries, d_model) — FULL batch size, not B_present. Missing
    rows are filled by the caller's learned baseline tokens with the
    gradient-safe form
        out = baseline + present_mask * (computed - baseline)
    so the baseline Parameter stays gradient-connected even when every row in
    a batch has a real scene.
    """

    def __init__(
        self,
        d_model: int,
        n_queries: int = 8,
        n_layers: int = 2,
        scene_feature_dim: int = 64,
        n_heads: int = 8,
        d_ff: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.d_model = d_model
        self.n_queries = n_queries
        # KV projection from the backbone channel dim → transformer d_model.
        self.kv_proj = nn.Linear(scene_feature_dim, d_model)
        self.pos_enc = _CoordPE(d_model)
        # Learned query tokens, initialized small.
        self.queries = nn.Parameter(torch.zeros(n_queries, d_model))
        nn.init.normal_(self.queries, std=0.02)
        self.blocks = nn.ModuleList([_CrossAttnBlock(d_model, n_heads, d_ff, dropout=dropout) for _ in range(n_layers)])

    def forward(
        self,
        f_per_point: Tensor,  # (N_total, scene_feature_dim) from SceneEncoder
        coord: Tensor,  # (N_total, 3) same order as f_per_point
        offset: Tensor,  # (B_present,) cumulative counts
        B: int,
        present_mask: Tensor,  # (B,) bool
        baseline_tokens: Tensor,  # (n_queries, d_model) — fallback parameter
    ) -> Tensor:
        device = baseline_tokens.device
        D = self.d_model

        baseline_expanded = baseline_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, Q, D)

        n_present = int(present_mask.sum().item())
        if n_present == 0:
            return baseline_expanded.contiguous()

        # All present samples have the same N — flat-reshape.
        counts = torch.diff(torch.cat([torch.zeros(1, device=device, dtype=offset.dtype), offset]))
        N_per = int(counts.max().item())
        assert (counts == N_per).all(), f"GlobalSceneBranch assumes uniform N, got {counts.tolist()}"

        kv_proj = self.kv_proj(f_per_point)  # (N_total, D)
        kv_proj = kv_proj + self.pos_enc(coord)  # additive learned PE
        kv_proj = kv_proj.view(n_present, N_per, D)  # (B_present, N, D)

        queries = self.queries.unsqueeze(0).expand(n_present, -1, -1).contiguous()  # (B_present, Q, D)
        for blk in self.blocks:
            queries = blk(queries, kv_proj)
        # queries: (B_present, Q, D)

        # Scatter back to (B, Q, D) using the gradient-safe form
        #   out = baseline + present_mask * (computed - baseline)
        present_idx = present_mask.nonzero(as_tuple=False).flatten()  # (B_present,)
        computed = baseline_expanded.clone()
        computed[present_idx] = queries
        mask_f = present_mask.to(computed.dtype).view(B, 1, 1)
        return baseline_expanded + mask_f * (computed - baseline_expanded)
