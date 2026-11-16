"""Bandwidth-calibrated temporal smoothing for recovered trajectory states.

The forecasting model trains on whole trajectories, so temporal smoothness is an output
axis next to accuracy: re-anchoring jumps and per-frame jitter in the extraction chains
become velocity targets the model would learn to imitate. This module provides

  smooth_trans    masked moving average of a 3D series over valid rows within ±radius
  smooth_rot      one-step Karcher mean on SO(3): R_t ← R_t·exp(mean_j log(R_tᵀ R_j)).
                  Local averaging (no increment recomposition, hence no global drift); a
                  constant-angular-velocity chain is preserved exactly in the interior
                  (same-axis logs commute, symmetric window ⇒ tangent residuals cancel);
                  white rotational jitter is averaged away.
  apply_temporal_filter   smooths exactly the rigid channels stage2_npz.rebuild_npz applies
                  (hand orient + cam_t per valid side, object rot + trans); articulation
                  passes through untouched.

The smoothing radius is the smallest whose jerk matches the jerk of ground-truth
trajectories: oversmoothing erases real dynamics, which is as wrong for training as jitter.
"""

import torch

from gtf.extraction.stage2.stage2_common import HAND_ORIENT, LEFT, OBJ, RIGHT, axis_angle_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """Batched matrix log on SO(3) → axis-angle (..., 3)."""
    tr = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    theta = torch.arccos(((tr - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7))
    w = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], dim=-1)
    s = theta / (2 * torch.sin(theta)).clamp(min=1e-7)
    return w * s.unsqueeze(-1)


def smooth_trans(x: torch.Tensor, rows: torch.Tensor, radius: int) -> torch.Tensor:
    """Masked moving average of (K,3) over valid rows within ±radius keyframes. Only valid
    rows are read or written; linear motion is preserved exactly in the interior."""
    if radius <= 0 or len(rows) == 0:
        return x.clone()
    out = x.clone()
    K = x.shape[0]
    vmask = torch.zeros(K, dtype=torch.bool, device=x.device)
    vmask[rows] = True
    for t in rows.tolist():
        lo, hi = max(0, t - radius), min(K, t + radius + 1)
        sel = vmask[lo:hi]
        out[t] = x[lo:hi][sel].mean(dim=0)
    return out


def smooth_rot(R: torch.Tensor, rows: torch.Tensor, radius: int) -> torch.Tensor:
    """One-step Karcher local mean of a (K,3,3) rotation chain over valid rows within
    ±radius. Drift-free (each pose averaged in its own tangent space)."""
    if radius <= 0 or len(rows) == 0:
        return R.clone()
    out = R.clone()
    K = R.shape[0]
    vmask = torch.zeros(K, dtype=torch.bool, device=R.device)
    vmask[rows] = True
    for t in rows.tolist():
        lo, hi = max(0, t - radius), min(K, t + radius + 1)
        sel = vmask[lo:hi].nonzero(as_tuple=True)[0] + lo
        rel = R[t].transpose(-1, -2).unsqueeze(0) @ R[sel]
        out[t] = R[t] @ axis_angle_to_matrix(so3_log(rel).mean(dim=0))
    return out


def apply_temporal_filter(states: torch.Tensor, valid: torch.Tensor, radius_hands: int = 0, radius_obj: int = 0) -> torch.Tensor:
    """Smooth the rigid channels of (K,207) recovered states: hand orient (SO(3)) + cam_t
    per valid side, object rot + trans. Pose/articulation channels are untouched (the
    rebuild applies only rigid corrections; articulation_filter.py handles the fingers).

    Per-entity radii because the jitter is per-entity: the extracted object chain is
    already smoother than ground truth (the optimizer's DCT correction absorbs the SAM3D
    re-anchoring), while the hands are jerkier than ground truth; radius 1 brings the
    hands to the ground-truth bandwidth."""
    if radius_hands <= 0 and radius_obj <= 0:
        return states.clone()
    out = states.clone()
    for col, sl, radius in ((0, LEFT, radius_hands), (1, RIGHT, radius_hands), (2, OBJ, radius_obj)):
        rows = valid[:, col].nonzero(as_tuple=True)[0]
        if len(rows) < 3 or radius <= 0:
            continue
        b = sl.start
        R = rotation_6d_to_matrix(states[rows, b + HAND_ORIENT.start : b + HAND_ORIENT.stop])
        R_full = torch.eye(3, device=states.device).repeat(states.shape[0], 1, 1)
        R_full[rows] = R
        out[:, b : b + 6][rows] = matrix_to_rotation_6d(smooth_rot(R_full, rows, radius)[rows])
        out[:, b + 6 : b + 9] = smooth_trans(states[:, b + 6 : b + 9], rows, radius)
    return out
