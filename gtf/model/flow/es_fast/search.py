"""Grad-free nearest-support search for the support-penetration loss.

Squared distances are formed directly (no norm/matmul expansion, so no cancellation). Ties pick the
first support index. On CUDA with Triton, an FP32 screen certifies clear winners and every
ambiguous query is re-checked against all support points in FP64; returned distances are FP64
square roots rounded to FP32. Without CUDA or Triton, a chunked all-support FP64 search with the
same tie rule is used. Coordinates must be finite FP32.
"""

import warnings
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _load_kernels():
    try:
        from . import _kernels
    except ModuleNotFoundError as error:
        if error.name != "triton" and not str(error.name).startswith("triton."):
            raise
        return None
    return _kernels


def _check(query, support):
    if query.dtype != torch.float32 or support.dtype != torch.float32:
        raise ValueError("Support search expects FP32 geometry")
    if query.ndim != 2 or support.ndim != 2 or query.shape[1] != 3 or support.shape[1] != 3:
        raise ValueError("Expected query/support shaped (N,3)")
    if query.device != support.device:
        raise ValueError("Support search inputs must be on the same device")
    if len(support) == 0 or len(support) >= 2147483647:
        raise ValueError("Expected between 1 and 2147483646 support points")


def reference_search(query, support, *, chunk_queries=256, chunk_support=4096):
    """Memory-bounded all-support FP64 search (both axes chunked). Returns (distance, index)."""
    _check(query, support)
    if chunk_queries < 1 or chunk_support < 1:
        raise ValueError("Reference search chunk sizes must be positive")
    distances, indices = [], []
    with torch.no_grad():
        for start in range(0, len(query), chunk_queries):
            q = query[start:start + chunk_queries].detach().double()
            best = torch.full((len(q),), float("inf"), device=q.device, dtype=torch.float64)
            chosen = torch.zeros(len(q), device=q.device, dtype=torch.int64)
            for offset in range(0, len(support), chunk_support):
                s = support[offset:offset + chunk_support].detach().double()
                delta = q[:, None] - s[None]
                square = (delta[..., 0].square() + delta[..., 1].square()) + delta[..., 2].square()
                value, index = square.min(1)
                # Strict improvement keeps the first index across blocks.
                improve = value < best
                chosen = torch.where(improve, index + offset, chosen)
                best = torch.minimum(best, value)
            distances.append(best.sqrt().float())
            indices.append(chosen)
    if not distances:
        return query.new_empty((0,)), torch.empty(0, device=query.device, dtype=torch.int64)
    return torch.cat(distances), torch.cat(indices)


def nearest_support(query, support):
    """Nearest support point of each query: (distance (N,), index (N,)).

    CUDA with Triton uses the screen/refine kernels (no host synchronization or copy); otherwise
    the FP64 ``reference_search`` (with a warning on CUDA, where it is much slower).
    """
    _check(query, support)
    kernels = _load_kernels() if query.is_cuda else None
    if kernels is None:
        if query.is_cuda:
            warnings.warn("Triton unavailable: using the slower all-support FP64 support search", RuntimeWarning, stacklevel=2)
        return reference_search(query, support)
    with torch.no_grad():
        query, support = query.detach().contiguous(), support.detach().contiguous()
        distance = query.new_empty(len(query))
        index = torch.empty(len(query), device=query.device, dtype=torch.int64)
        ambiguous = torch.empty(len(query), device=query.device, dtype=torch.bool)
        if len(query):
            launch = ((len(query) + 31) // 32,)
            args = (query, support, distance, index, ambiguous, len(query), len(support), 32, 128)
            kernels._screen_kernel[launch](*args, num_warps=4, enable_fp_fusion=False)
            kernels._refine_kernel[launch](*args, num_warps=4, enable_fp_fusion=False)
        return distance, index


def pen_below_support(pts, support, margin, band_r):
    """One-sided penetration depth of query points below a horizontal support surface.

    Gravity-up is -y. Each point is compared with its nearest support point in 3D: the depth is
    ``relu((pt_y - support_y) - margin)``, so ``margin`` absorbs GT soft-contact residuals. A point
    is eligible when the lateral (xz) offset to that support point is within ``band_r``; a point
    sinking straight down keeps a near-zero lateral offset, so its penalty grows monotonically with
    depth, while free-space points beside the surface gate out. The depth target is the nearest
    support in 3D rather than the highest support in a lateral column: on cluttered tabletops a
    neighboring object's top would otherwise define the surface.

    pts: (..., 3) gravity-aligned points; support: (N, 3). Returns (...,) depths in meters.
    Gradients reach ``pts`` only through its y coordinate; the search and the gate are grad-free.
    """
    if pts.numel() == 0 or support.numel() == 0:
        return pts.new_zeros(pts.shape[:-1])
    flat = pts.reshape(-1, 3)
    with torch.no_grad():
        _, nn_i = nearest_support(flat, support)
        lat = (flat.detach()[:, [0, 2]] - support[nn_i][:, [0, 2]]).norm(dim=1)
        has_nbr = lat <= band_r
    depth = flat[:, 1] - support[nn_i, 1] - margin
    return torch.where(has_nbr, torch.relu(depth), torch.zeros_like(depth)).reshape(pts.shape[:-1])
