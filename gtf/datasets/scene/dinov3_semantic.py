"""DINOv3 ViT-L/16 patch feature extractor for scene preprocessing.

Uses `transformers.DINOv3ViTModel` (transformers >= 4.57). Weights are fetched
from the gated `facebook/dinov3-vitl16-pretrain-lvd1689m` repo on first use
(requires HF license acceptance + auth token in the active environment).

Per-pixel semantic features are obtained by:
  1. pad RGB to a multiple of `patch_size` (16 for ViT-L/16)
  2. ImageNet-normalize
  3. forward through DINOv3 → `(1, 1 + num_register_tokens + H_p*W_p, C)`
  4. strip CLS + register tokens → `(1, H_p*W_p, C)` → reshape to `(H_p, W_p, C)`

The resulting (H_p, W_p, 1024) features are then projected through one global PCA
basis (`gtf/datasets/scene/fit_pca.py`), giving a 32-D scene semantic channel with
the same meaning across the corpus.
"""

from __future__ import annotations

import numpy as np
import torch

DEFAULT_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"

# ImageNet normalization — the DINOv3 image processor
# `DINOv3ViTImageProcessorFast` uses these as its defaults.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DinoV3Extractor:
    """Lazy-loaded DINOv3 ViT-L/16 wrapper.

    Example:
        extractor = DinoV3Extractor(device="cuda")
        feats = extractor.extract_patch_features(rgb)  # (H_p, W_p, 1024)

    The extractor exposes `patch_size` after the first forward call so callers
    can convert pixel coordinates into patch-grid indices via
    `pixel_to_patch_index(pix_u, pix_v, H_p, W_p, patch_size)`.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str | torch.device = "cuda"):
        self.model_id = model_id
        self.device = torch.device(device)
        self.model: torch.nn.Module | None = None
        self.patch_size: int | None = None
        self.num_skip_tokens: int | None = None  # CLS + register tokens
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

    def _load(self) -> None:
        from transformers import DINOv3ViTModel

        self.model = DINOv3ViTModel.from_pretrained(self.model_id).to(self.device).eval()
        self.patch_size = int(self.model.config.patch_size)
        self.num_skip_tokens = 1 + int(self.model.config.num_register_tokens)

    def _patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """ImageNet-normalize (N, 3, H_pad, W_pad) pixels in [0, 1]; return the (N, H_p*W_p, C) patch tokens."""
        if self._mean is None or self._mean.device != self.device:
            self._mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
            self._std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        assert self.model is not None and self._std is not None
        x = (x - self._mean) / self._std
        out = self.model(pixel_values=x).last_hidden_state  # (N, 1 + n_reg + H_p*W_p, C)
        return out[:, self.num_skip_tokens :, :]

    @torch.inference_mode()
    def extract_patch_features(self, rgb: np.ndarray) -> np.ndarray:
        """Extract per-patch features from an RGB image.

        Args:
            rgb: (H, W, 3) uint8 RGB image. Any H, W ≥ patch_size are supported;
                the image is zero-padded on the bottom/right to the next multiple
                of patch_size. DINOv3 uses RoPE so no interpolation is required.

        Returns:
            (H_p, W_p, C) float32 patch features where C = 1024 for ViT-L.
            H_p = ceil(H / patch_size), W_p = ceil(W / patch_size).
        """
        if self.model is None:
            self._load()
        assert self.model is not None and self.patch_size is not None and self.num_skip_tokens is not None
        _check_rgb(rgb)

        ps = int(self.patch_size)
        H, W = rgb.shape[:2]
        H_p = (H + ps - 1) // ps
        W_p = (W + ps - 1) // ps
        H_pad = H_p * ps
        W_pad = W_p * ps

        if H_pad != H or W_pad != W:
            canvas = np.zeros((H_pad, W_pad, 3), dtype=np.uint8)
            canvas[:H, :W] = rgb
        else:
            canvas = rgb

        x = torch.from_numpy(canvas).to(device=self.device, dtype=torch.float32) / 255.0  # (H_pad, W_pad, 3)
        x = x.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H_pad, W_pad)
        patches = self._patch_tokens(x)
        C = patches.shape[-1]
        return patches.reshape(H_p, W_p, C).detach().cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def extract_projected_patch_features_batch(self, rgbs: list[np.ndarray], pca_basis: dict[str, np.ndarray]) -> list[np.ndarray]:
        """Extract DINO patch features and apply global PCA on GPU before CPU transfer."""
        if not rgbs:
            return []
        if self.model is None:
            self._load()
        assert self.model is not None and self.patch_size is not None and self.num_skip_tokens is not None

        shapes = {tuple(rgb.shape) for rgb in rgbs}
        if len(shapes) != 1:
            raise ValueError(f"batched DINO requires same-resolution frames, got {sorted(shapes)}")
        for rgb in rgbs:
            _check_rgb(rgb)

        ps = int(self.patch_size)
        H, W = rgbs[0].shape[:2]
        H_p = (H + ps - 1) // ps
        W_p = (W + ps - 1) // ps
        H_pad = H_p * ps
        W_pad = W_p * ps
        if H_pad != H or W_pad != W:
            canvas = np.zeros((len(rgbs), H_pad, W_pad, 3), dtype=np.uint8)
            for i, rgb in enumerate(rgbs):
                canvas[i, :H, :W] = rgb
        else:
            canvas = np.stack(rgbs, axis=0)

        x = torch.from_numpy(canvas).to(device=self.device, dtype=torch.float32) / 255.0
        x = x.permute(0, 3, 1, 2)
        patches = self._patch_tokens(x).float()
        C = patches.shape[-1]
        components = torch.as_tensor(pca_basis["components"], device=self.device, dtype=torch.float32)
        mean = torch.as_tensor(pca_basis["mean"], device=self.device, dtype=torch.float32)
        if components.shape[1] != C:
            raise ValueError(f"PCA basis dim mismatch: components {tuple(components.shape)} vs DINOv3 feat dim {C}")
        projected = (patches - mean.view(1, 1, C)) @ components.t()
        K = projected.shape[-1]
        arr = projected.reshape(len(rgbs), H_p, W_p, K).detach().cpu().numpy().astype(np.float32)
        return [arr[i] for i in range(len(rgbs))]


def _check_rgb(rgb: np.ndarray) -> None:
    if rgb.dtype != np.uint8:
        raise ValueError(f"rgb must be uint8, got {rgb.dtype}")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must be (H, W, 3), got {rgb.shape}")


def pixel_to_patch_index(
    pix_u: np.ndarray,
    pix_v: np.ndarray,
    H_p: int,
    W_p: int,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert pixel coordinates to patch-grid indices.

    Args:
        pix_u: (M,) int pixel x coordinates.
        pix_v: (M,) int pixel y coordinates.
        H_p, W_p: patch grid height and width.
        patch_size: integer patch size of the backbone (16 for DINOv3 ViT-L/16).

    Returns:
        (patch_v, patch_u) int arrays clipped to [0, H_p-1] and [0, W_p-1].
    """
    patch_v = np.clip(pix_v // int(patch_size), 0, H_p - 1)
    patch_u = np.clip(pix_u // int(patch_size), 0, W_p - 1)
    return patch_v, patch_u
