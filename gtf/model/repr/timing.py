"""Temporal-grid helpers shared by the codec, losses, and evaluation.

Video windows are sampled at source-frame indices that dither between integer
steps, and some videos are variable-frame-rate, so timing is represented as one
elapsed time per frame transition (edge):

    edge_dt[:, i] = time[:, i + 1] - time[:, i]

The helpers accept that explicit ``(B, T - 1)`` edge grid as well as scalar or
per-window ``(B,)`` durations. They do not infer time from source frame indices:
decoded media timestamps are the authority for VFR video.
"""

from __future__ import annotations

import torch
from torch import Tensor


def normalize_edge_dt(
    dt: Tensor | float,
    n_frames: int,
    *,
    batch_size: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    validate: bool = True,
) -> Tensor:
    """Return ``dt`` as ``(B, T-1)`` edge durations.

    Accepted inputs are:

    - a scalar;
    - ``(B,)`` per-window scalar durations;
    - ``(T-1,)`` for one unbatched trajectory;
    - ``(B, T-1)`` explicit edge durations.

    ``expand`` is used for scalar inputs, so they do not allocate a dense timing grid.
    """

    if n_frames < 1:
        raise ValueError(f"n_frames must be positive, got {n_frames}")
    n_edges = n_frames - 1
    if not isinstance(dt, Tensor):
        dt = torch.as_tensor(dt, device=device, dtype=dtype or torch.float32)
    else:
        dt = dt.to(device=device if device is not None else dt.device, dtype=dtype or dt.dtype)

    if dt.ndim == 0:
        out = dt.reshape(1, 1).expand(batch_size, n_edges)
    elif dt.ndim == 1:
        if dt.numel() == batch_size:
            out = dt.reshape(batch_size, 1).expand(batch_size, n_edges)
        elif batch_size == 1 and dt.numel() == n_edges:
            out = dt.reshape(1, n_edges)
        else:
            raise ValueError(
                f"1-D dt must have B={batch_size} scalar entries"
                f" or T-1={n_edges} edges for B=1, got {tuple(dt.shape)}"
            )
    elif dt.ndim == 2:
        if tuple(dt.shape) != (batch_size, n_edges):
            raise ValueError(f"edge dt must have shape {(batch_size, n_edges)}, got {tuple(dt.shape)}")
        out = dt
    else:
        raise ValueError(f"dt must be scalar, (B,), or (B,T-1); got {tuple(dt.shape)}")

    if validate and out.numel():
        if not bool(torch.isfinite(out).all()):
            raise ValueError("edge dt contains NaN or Inf")
        if not bool((out > 0).all()):
            raise ValueError("edge dt must be strictly positive")
    return out


def frame_times_from_edge_dt(edge_dt: Tensor) -> Tensor:
    """Integrate ``(B,T-1)`` edge durations to ``(B,T)`` elapsed times starting at 0."""

    if edge_dt.ndim != 2:
        raise ValueError(f"edge_dt must be (B,T-1), got {tuple(edge_dt.shape)}")
    start_t = torch.zeros(edge_dt.shape[0], 1, device=edge_dt.device, dtype=edge_dt.dtype)
    return torch.cat([start_t, start_t + torch.cumsum(edge_dt, dim=1)], dim=1)


def physical_finite_differences(
    x: Tensor,
    dt: Tensor | float,
    *,
    validate: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return velocity, acceleration, and jerk on a possibly nonuniform grid.

    ``x`` is ``(B,T,D)``.  Velocity lives on interval midpoints.  Acceleration
    uses the distance between adjacent velocity midpoints,
    ``(h[i-1] + h[i]) / 2``.  Algebraically this is twice the second divided
    difference.  Jerk is six times the third divided difference, so the
    denominator between consecutive acceleration estimates is the mean span
    of their four samples, ``(h[i] + h[i+1] + h[i+2]) / 3``.  This is the
    canonical nonuniform extension of the uniform third finite difference and
    exactly recovers the third derivative of a cubic.  Where three consecutive
    edges are equal, a ``where`` branch uses the edge itself as the denominator
    (avoiding the rounding of the mean).
    """

    if x.ndim < 3:
        raise ValueError(f"x must be at least (B,T,D), got {tuple(x.shape)}")
    B, T = x.shape[:2]
    edge = normalize_edge_dt(dt, T, batch_size=B, device=x.device, dtype=x.dtype, validate=validate)
    scale_shape = (B, T - 1) + (1,) * (x.ndim - 2)
    h = edge.reshape(scale_shape)
    vel = (x[:, 1:] - x[:, :-1]) / h

    if T < 3:
        acc = x[:, :0]
    else:
        acc_scale_shape = (B, T - 2) + (1,) * (x.ndim - 2)
        h_mid = (0.5 * (edge[:, :-1] + edge[:, 1:])).reshape(acc_scale_shape)
        acc = (vel[:, 1:] - vel[:, :-1]) / h_mid

    if T < 4:
        jerk = x[:, :0]
    else:
        jerk_scale_shape = (B, T - 3) + (1,) * (x.ndim - 2)
        h0, h1, h2 = edge[:, :-2], edge[:, 1:-1], edge[:, 2:]
        h_span = (h0 + h1 + h2) / 3.0
        # Uniform triple: use the edge itself (exact), else the mean span.
        h_acc = torch.where((h0 == h1) & (h1 == h2), h1, h_span).reshape(
            jerk_scale_shape
        )
        jerk = (acc[:, 1:] - acc[:, :-1]) / h_acc
    return vel, acc, jerk

