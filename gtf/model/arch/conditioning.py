"""Segment-level conditioning token encoder for GTF.

Computes conditioning tokens from object shape (a cached frozen Uni3D embedding),
hand shape (MANO betas), object scale, and scene context (LitePT scene encoder +
global query branch, gtf/model/arch/scene_encoder.py). All conditioning tokens are
always "observed" in attention — never masked by obs_mask.

The Uni3D loader and the point-cloud helpers here are used to build the cached
object embeddings (gtf/datasets/build_uni3d_cache.py) and by the dataset.
"""

import sys

import torch
import torch.nn as nn
from torch import Tensor

from gtf.config import ModelConfig
from gtf.paths import THIRD_PARTY_DIR

# Uni3D model args for the giant variant
UNI3D_ARGS = {
    "pc_model": "eva_giant_patch14_560",
    "pc_feat_dim": 1408,
    "embed_dim": 1024,
    "group_size": 64,
    "num_group": 512,
    "pc_encoder_dim": 512,
    "npoints": 10000,
    "pretrained_pc": "",
    "drop_path_rate": 0.0,
    "patch_dropout": 0.0,
}


def _load_uni3d(weights_path: str) -> nn.Module:
    """Load frozen Uni3D model from checkpoint.

    Args:
        weights_path: Path to Uni3D-g model.pt (DeepSpeed checkpoint).

    Returns:
        Uni3D model with all parameters frozen.
    """
    # Add Uni3D source to path temporarily
    uni3d_dir = str(THIRD_PARTY_DIR / "uni3d")
    added = False
    if uni3d_dir not in sys.path:
        sys.path.insert(0, uni3d_dir)
        added = True

    try:
        from easydict import EasyDict
        from models.uni3d import create_uni3d

        model = create_uni3d(EasyDict(UNI3D_ARGS))

        # Load DeepSpeed checkpoint
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["module"]
        # Strip 'module.' prefix if present
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
    finally:
        if added and uni3d_dir in sys.path:
            sys.path.remove(uni3d_dir)

    # Freeze everything
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model


def compute_vertex_normals(verts: Tensor, faces: Tensor) -> Tensor:
    """Compute area-weighted vertex normals from mesh faces.

    Args:
        verts: (V, 3) mesh vertices.
        faces: (F, 3) triangle face indices (int64).

    Returns:
        (V, 3) unit vertex normals.
    """
    v0 = verts[faces[:, 0]]  # (F, 3)
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    # Face normals (area-weighted — cross product magnitude = 2 * triangle area)
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (F, 3)

    # Accumulate to vertices
    vertex_normals = torch.zeros_like(verts)  # (V, 3)
    vertex_normals.index_add_(0, faces[:, 0], face_normals)
    vertex_normals.index_add_(0, faces[:, 1], face_normals)
    vertex_normals.index_add_(0, faces[:, 2], face_normals)

    # Normalize
    norms = vertex_normals.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return vertex_normals / norms


def sample_obj_points(
    obj_canonical_verts: Tensor,
    obj_vertex_normals: Tensor,
    n_points: int = 10000,
    generator: torch.Generator | None = None,
    obj_faces: Tensor | None = None,
) -> Tensor:
    """Sample and normalize an object point cloud (the Uni3D input).

    With faces, points are sampled on the SURFACE: faces drawn proportionally to
    triangle area, uniform barycentric coordinates, face normals — independent of
    the mesh tessellation. Without faces (or with a zero-area mesh) mesh vertices
    are drawn with their vertex normals.

    Args:
        obj_canonical_verts: (V, 3) canonical mesh vertices.
        obj_vertex_normals: (V, 3) unit vertex normals.
        n_points: Number of points to sample.
        generator: optional torch.Generator for a deterministic draw.
        obj_faces: (F, 3) long triangles.

    Returns:
        (n_points, 6) point cloud: [xyz(3), normal(3)], centered and normalized.
    """
    V = obj_canonical_verts.shape[0]
    device = obj_canonical_verts.device

    if obj_faces is not None and obj_faces.numel() >= 3:
        f = obj_faces.long()
        v0, v1, v2 = obj_canonical_verts[f[:, 0]], obj_canonical_verts[f[:, 1]], obj_canonical_verts[f[:, 2]]
        cross = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (F, 3)
        area = cross.norm(dim=-1)  # 2×area, proportional is enough
        if float(area.sum()) > 0:
            fidx = torch.multinomial(area, n_points, replacement=True, generator=generator)
            r1 = torch.rand(n_points, 1, device=device, generator=generator)
            r2 = torch.rand(n_points, 1, device=device, generator=generator)
            su = r1.sqrt()
            b0, b1 = 1.0 - su, su * (1.0 - r2)
            b2 = su * r2
            xyz = b0 * v0[fidx] + b1 * v1[fidx] + b2 * v2[fidx]  # (N, 3)
            nrm = torch.nn.functional.normalize(cross[fidx], dim=-1)
            centroid = xyz.mean(dim=0, keepdim=True)
            xyz = xyz - centroid
            max_dist = xyz.norm(dim=-1).max().clamp(min=1e-6)
            return torch.cat([xyz / max_dist, nrm], dim=-1)

    # Vertex draw
    if V >= n_points:
        indices = torch.randperm(V, device=device, generator=generator)[:n_points]
    else:
        # Oversample with replacement
        indices = torch.randint(V, (n_points,), device=device, generator=generator)

    xyz = obj_canonical_verts[indices]  # (N, 3)
    nrm = obj_vertex_normals[indices]  # (N, 3)

    # Center at origin
    centroid = xyz.mean(dim=0, keepdim=True)
    xyz = xyz - centroid

    # Normalize to unit sphere
    max_dist = xyz.norm(dim=-1).max().clamp(min=1e-6)
    xyz = xyz / max_dist

    return torch.cat([xyz, nrm], dim=-1)  # (N, 6)


class ObjectShapeEncoder(nn.Module):
    """Trainable projection of a cached frozen Uni3D-g object embedding.

    (B, 1024) → MLP → (B, n_tokens, D). The embedding is precomputed at both
    precisions the model runs at, because Uni3D's fp32 and bf16 outputs differ
    materially: CUDA bf16 autocast selects ``obj_embedding_bf16`` and AMP-off
    evaluation selects ``obj_embedding_fp32``.
    """

    def __init__(self, d_model: int, n_tokens: int = 4):
        super().__init__()
        self.uni3d_dim = 1024  # Uni3D-g embed_dim (trans2embed output)
        self.n_tokens = n_tokens
        self.d_model = d_model

        # Trainable projection: 1024 → n_tokens * D
        proj_dim = n_tokens * d_model
        self.proj = nn.Sequential(
            nn.Linear(self.uni3d_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def project_embedding(self, obj_embedding: Tensor) -> Tensor:
        """Project a native frozen-Uni3D embedding to conditioning tokens."""
        if obj_embedding.ndim != 2 or obj_embedding.shape[-1] != self.uni3d_dim:
            raise ValueError(f"obj_embedding must have shape (B, 1024), got {tuple(obj_embedding.shape)}")
        tokens = self.proj(obj_embedding)  # (B, n_tokens * D)
        return tokens.reshape(-1, self.n_tokens, self.d_model)

    @staticmethod
    def _select_cached_embedding(
        obj_embedding_bf16: Tensor | None,
        obj_embedding_fp32: Tensor | None,
    ) -> Tensor:
        """Select the embedding produced under the active CUDA precision.

        Selection follows the real CUDA autocast state, not input dtype or
        module mode; up/down-casting one cached tensor cannot reproduce the other.
        """
        cuda_amp = torch.is_autocast_enabled("cuda")
        if cuda_amp:
            amp_dtype = torch.get_autocast_dtype("cuda")
            if amp_dtype != torch.bfloat16:
                raise RuntimeError(f"Cached Uni3D supports CUDA bf16 autocast or AMP-off fp32; active CUDA autocast dtype is {amp_dtype}.")
            selected = obj_embedding_bf16
            key = "obj_embedding_bf16"
            expected_dtype = torch.bfloat16
        else:
            selected = obj_embedding_fp32
            key = "obj_embedding_fp32"
            expected_dtype = torch.float32

        if selected is None:
            mode = "CUDA bf16 autocast" if cuda_amp else "AMP-off fp32"
            raise RuntimeError(f"conditioning[{key!r}] with shape (B, 1024) is required for the active {mode} path.")
        if selected.dtype != expected_dtype:
            raise ValueError(f"{key} must have dtype {expected_dtype}, got {selected.dtype}")
        return selected

    def forward(
        self,
        obj_embedding_bf16: Tensor | None = None,
        obj_embedding_fp32: Tensor | None = None,
    ) -> Tensor:
        """Project the cached Uni3D embedding of the active precision.

        Args:
            obj_embedding_bf16: Uni3D output (B, 1024) computed under CUDA bf16 autocast.
            obj_embedding_fp32: Uni3D output (B, 1024) computed with AMP disabled.

        Returns:
            (B, n_tokens, D) shape conditioning tokens.
        """
        embed = self._select_cached_embedding(obj_embedding_bf16, obj_embedding_fp32)
        return self.project_embedding(embed)


class ConditioningEncoder(nn.Module):
    """Computes all segment-level conditioning tokens.

    Output token sequence (N_cond = n_obj_shape_tokens + n_scene_global_tokens + 3):
    [obj_shape × 4 | scene_global × 8 | left_hand_shape × 1 | right_hand_shape × 1 | obj_scale × 1]
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        from gtf.model.arch.scene_encoder import GlobalSceneBranch, SceneEncoder

        D = cfg.d_model

        # 1. Object shape encoder (cached Uni3D embedding + trainable projection)
        self.obj_shape_encoder = ObjectShapeEncoder(d_model=D, n_tokens=cfg.n_obj_shape_tokens)

        # 2. Hand shape embedding — shared for left/right (same MANO betas space)
        self.hand_shape_proj = nn.Sequential(
            nn.Linear(10, D),
            nn.GELU(),
            nn.Linear(D, D),
        )

        # 3. Object scale embedding
        self.scale_proj = nn.Sequential(
            nn.Linear(3, D),
            nn.GELU(),
            nn.Linear(D, D),
        )

        # 4. Scene tokens. The learned baseline tokens stand in for samples without
        # scene features; GlobalSceneBranch returns
        # `baseline + present_mask * (computed - baseline)`, which keeps the baseline
        # gradient-connected in every batch (DDP find_unused_parameters=False).
        self.scene_global_tokens = nn.Parameter(torch.zeros(cfg.n_scene_global_tokens, D))
        self.scene_encoder = SceneEncoder(
            in_channels=int(cfg.scene_in_channels),
            scene_feature_dim=int(cfg.scene_feature_dim),
            # An empty path skips the pretrained backbone (a checkpoint load supplies its weights).
            weights_path=str(cfg.litept_weights_path) or None,
        )
        self.scene_branch = GlobalSceneBranch(
            d_model=D,
            n_queries=cfg.n_scene_global_tokens,
            n_layers=2,
            scene_feature_dim=int(cfg.scene_feature_dim),
            n_heads=cfg.n_heads,
        )
        self.scene_semantic_dim = int(cfg.scene_semantic_dim)
        self.scene_features_dropout = float(cfg.scene_features_dropout)

        self.d_model = D
        self.n_obj_shape_tokens = cfg.n_obj_shape_tokens
        self.n_scene_global_tokens = cfg.n_scene_global_tokens

        # A learned per-slot null-conditioning embedding. It is part of the checkpoint
        # but not used by the forward pass; a zero-weight term keeps it in the autograd
        # graph (DDP find_unused_parameters=False). Created after n_obj_shape_tokens /
        # n_scene_global_tokens, which n_cond_tokens reads.
        self.null_cond_tokens = nn.Parameter(torch.zeros(self.n_cond_tokens, D))

    @property
    def n_cond_tokens(self) -> int:
        """Total number of conditioning tokens."""
        return self.n_obj_shape_tokens + self.n_scene_global_tokens + 3  # +3 = left_hand + right_hand + scale

    @property
    def _left_hand_slot(self) -> int:
        return self.n_obj_shape_tokens + self.n_scene_global_tokens

    @property
    def _right_hand_slot(self) -> int:
        return self.n_obj_shape_tokens + self.n_scene_global_tokens + 1

    def forward(
        self,
        betas_left: Tensor | None,
        betas_right: Tensor | None,
        obj_scale: Tensor,
        hand_presence: Tensor,
        scene_features: list[dict | None] | None = None,
        origin: Tensor | None = None,
        obj_embedding_bf16: Tensor | None = None,
        obj_embedding_fp32: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute conditioning token sequence.

        Args:
            betas_left: (B, 10) left hand shape params, or None.
            betas_right: (B, 10) right hand shape params, or None.
            obj_scale: (B, 3) object scale factors.
            hand_presence: (B, 2) bool — [left_present, right_present].
            scene_features: list[dict|None] length B, from the collator (each present
                dict has P_scene / scene_geo_feats / [scene_sem_feats]). Samples
                without scene features get the learned baseline tokens.
            origin: (B, 3) centering origin of the state tokens. The scene points
                (P_scene and the xyz copy in the geometric features) are shifted by
                -origin at the encoder input so the scene shares the states' frame;
                the batch rows themselves are not modified.
            obj_embedding_bf16: (B, 1024) cached Uni3D embedding from CUDA bf16 autocast.
            obj_embedding_fp32: (B, 1024) cached Uni3D embedding from AMP-off fp32.

        Returns:
            cond_tokens: (B, N_cond, D) conditioning token sequence.
            cond_mask: (B, N_cond) bool — True for real tokens, False for absent hands.
        """
        from gtf.model.arch.scene_encoder import pack_scene_batch

        B = obj_scale.shape[0]
        device = obj_scale.device
        D = self.d_model

        # 1. Object shape tokens.
        obj_shape_tokens = self.obj_shape_encoder(obj_embedding_bf16=obj_embedding_bf16, obj_embedding_fp32=obj_embedding_fp32)

        # 2. Scene global tokens.
        scene_list = scene_features if scene_features is not None else [None] * B
        any_present = any(s is not None for s in scene_list)
        if any_present:
            coord, feat, batch_idx, present = pack_scene_batch(
                scene_list,
                scene_semantic_dim=self.scene_semantic_dim,
                dropout_p=self.scene_features_dropout,
                training=self.training,
                device=device,
            )
            # Shift the scene into the states' centered frame: coord and the xyz copy
            # in feat[:, 0:3]. The other channels (normals, distances, gravity angle,
            # curvature, density, semantics) are translation-invariant. batch_idx is
            # compact over present samples, so map it back to batch rows first.
            if origin is not None:
                _orig_rows = present.nonzero(as_tuple=True)[0]  # (B_present,) original rows
                org_pts = origin.to(device=coord.device, dtype=coord.dtype)[_orig_rows[batch_idx]]  # (N_total, 3)
                coord = coord - org_pts
                feat = feat.clone()
                feat[:, 0:3] = feat[:, 0:3] - org_pts
            f_per_point, offset = self.scene_encoder(coord, feat, batch_idx)
            scene_tokens = self.scene_branch(
                f_per_point,
                coord,
                offset,
                B=B,
                present_mask=present,
                baseline_tokens=self.scene_global_tokens,
            )
        else:
            # No sample in the batch has scene features: use the baseline tokens, plus a
            # zero-weight ghost forward over a tiny jittered cloud so the scene encoder
            # and branch parameters stay in the graph (DDP find_unused_parameters=False).
            # The serialization needs a non-degenerate cloud, hence the 1 cm jitter.
            N_g = 64
            ghost_coord = torch.randn(N_g, 3, device=device, dtype=torch.float32) * 0.01
            ghost_feat = torch.zeros(N_g, self.scene_encoder.in_channels, device=device)
            ghost_batch = torch.zeros(N_g, device=device, dtype=torch.long)
            f_ghost, ghost_offset = self.scene_encoder(ghost_coord, ghost_feat, ghost_batch)
            ghost_present = torch.zeros(B, dtype=torch.bool, device=device)
            ghost_present[0] = True
            ghost_tokens = self.scene_branch(
                f_ghost,
                ghost_coord,
                ghost_offset,
                B=B,
                present_mask=ghost_present,
                baseline_tokens=self.scene_global_tokens,
            )
            baseline = self.scene_global_tokens.unsqueeze(0).expand(B, -1, -1)
            scene_tokens = baseline + 0.0 * ghost_tokens.sum()

        # 3. Hand shape tokens
        if betas_left is not None:
            left_shape = self.hand_shape_proj(betas_left)  # (B, D)
        else:
            left_shape = torch.zeros(B, D, device=device)
        left_shape = left_shape.unsqueeze(1)  # (B, 1, D)

        if betas_right is not None:
            right_shape = self.hand_shape_proj(betas_right)  # same module, shared
        else:
            right_shape = torch.zeros(B, D, device=device)
        right_shape = right_shape.unsqueeze(1)  # (B, 1, D)

        # Zero out absent hand shape tokens
        left_shape = left_shape * hand_presence[:, 0:1].unsqueeze(-1).float()
        right_shape = right_shape * hand_presence[:, 1:2].unsqueeze(-1).float()

        # 4. Object scale token
        log_scale = torch.log(obj_scale.clamp(min=1e-6))  # (B, 3) — scale=1 → 0
        scale_token = self.scale_proj(log_scale).unsqueeze(1)  # (B, 1, D)

        # 5. Concatenate: [obj_shape(n_obj_shape_tokens) | scene(n_scene_global_tokens) | left(1) | right(1) | scale(1)]
        cond_tokens = torch.cat([obj_shape_tokens, scene_tokens, left_shape, right_shape, scale_token], dim=1)

        # 6. Build mask (True = real, False = absent)
        cond_mask = torch.ones(B, self.n_cond_tokens, dtype=torch.bool, device=device)
        cond_mask[:, self._left_hand_slot] = hand_presence[:, 0]
        cond_mask[:, self._right_hand_slot] = hand_presence[:, 1]

        # 7. Keep null_cond_tokens in the autograd graph (zero contribution).
        cond_tokens = cond_tokens + 0.0 * self.null_cond_tokens.sum()

        return cond_tokens, cond_mask
