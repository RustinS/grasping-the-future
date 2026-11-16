"""Differentiable rotation conversions between axis-angle, rotation matrix, and 6D representation.

Uses the COLUMNS convention from Zhou et al. (CVPR 2019): 6D = [col0, col1] of the rotation matrix.
Note: pytorch3d uses a ROWS convention; cross-validation requires transposing.
"""

import torch
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_axis_angle as _pt3d_matrix_to_axis_angle
from torch import Tensor


def axis_angle_to_matrix(axis_angle: Tensor) -> Tensor:
    """Convert axis-angle rotations to rotation matrices via Rodrigues formula.

    Uses the unnormalized skew-symmetric formulation to avoid dividing by theta.
    Handles theta=0 with Taylor expansion for NaN-free gradients.

    Args:
        axis_angle: (*, 3) axis-angle vectors where direction is axis and magnitude is angle in radians.

    Returns:
        (*, 3, 3) rotation matrices.
    """
    batch_shape = axis_angle.shape[:-1]
    dtype, device = axis_angle.dtype, axis_angle.device

    # theta^2 for branch selection; safe_theta avoids sqrt(0) gradient = inf
    theta_sq = (axis_angle * axis_angle).sum(dim=-1, keepdim=True)  # (*, 1)
    safe_theta_sq = theta_sq.clamp(min=1e-12)
    safe_theta = safe_theta_sq.sqrt()  # (*, 1) — never zero, safe for division and sin/cos grads

    # Rodrigues coefficients: c1 = sin(theta)/theta, c2 = (1 - cos(theta))/theta^2
    sin_theta = torch.sin(safe_theta)
    cos_theta = torch.cos(safe_theta)

    small = (theta_sq < 1e-6).squeeze(-1)  # (*) bool

    c1_exact = sin_theta / safe_theta
    c1_taylor = 1.0 - theta_sq / 6.0
    c1 = torch.where(small.unsqueeze(-1), c1_taylor, c1_exact)  # (*, 1)

    c2_exact = (1.0 - cos_theta) / safe_theta_sq
    c2_taylor = 0.5 - theta_sq / 24.0
    c2 = torch.where(small.unsqueeze(-1), c2_taylor, c2_exact)  # (*, 1)

    # Build unnormalized skew-symmetric matrix K from raw axis_angle
    rx = axis_angle[..., 0]
    ry = axis_angle[..., 1]
    rz = axis_angle[..., 2]
    zeros = torch.zeros_like(rx)

    K = torch.stack([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=-1).reshape(batch_shape + (3, 3))

    # R = I + c1 * K + c2 * K @ K
    K_sq = K @ K
    eye = torch.eye(3, dtype=dtype, device=device).expand(batch_shape + (3, 3))
    R = eye + c1.unsqueeze(-1) * K + c2.unsqueeze(-1) * K_sq

    return R


def matrix_to_rotation_6d(matrix: Tensor) -> Tensor:
    """Extract 6D rotation representation from rotation matrices.

    Uses COLUMNS convention (Zhou et al. CVPR 2019): extracts first two columns
    and flattens as [col0, col1].

    Args:
        matrix: (*, 3, 3) rotation matrices.

    Returns:
        (*, 6) 6D rotation representation.
    """
    batch_shape = matrix.shape[:-2]
    # matrix[..., :, :2] extracts columns 0 and 1 -> (*, 3, 2)
    # Transpose to (*, 2, 3) so flatten gives [col0(3), col1(3)]
    return matrix[..., :, :2].transpose(-1, -2).clone().reshape(batch_shape + (6,))


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """Convert 6D rotation representation to rotation matrices via Gram-Schmidt.

    Uses COLUMNS convention: 6D = [col0, col1]. Reconstructs rotation matrix
    by orthogonalizing column vectors via Gram-Schmidt, recovering the third
    column via cross product.

    Clamps normalization at 1e-6 to prevent extreme gradients from near-zero or
    near-parallel noised 6D vectors during flow matching.

    Args:
        d6: (*, 6) 6D rotation representation.

    Returns:
        (*, 3, 3) rotation matrices.
    """
    a1, a2 = d6[..., :3], d6[..., 3:]

    # Normalize first column with the same gradient-safe floor used below.
    b1 = F.normalize(a1, dim=-1, eps=1e-6)

    # Gram-Schmidt: remove component of a2 along b1
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = a2 - dot * b1

    # The 1e-6 norm floor bounds the gradient for near-parallel noised inputs.
    b2_norm = b2.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    b2 = b2 / b2_norm

    # Third column via cross product
    b3 = torch.cross(b1, b2, dim=-1)

    # Stack as columns (dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def axis_angle_to_6d(axis_angle: Tensor) -> Tensor:
    """Convert axis-angle to 6D rotation representation.

    Args:
        axis_angle: (*, 3) axis-angle vectors.

    Returns:
        (*, 6) 6D rotation representation.
    """
    return matrix_to_rotation_6d(axis_angle_to_matrix(axis_angle))


def rotation_6d_to_axis_angle(d6: Tensor) -> Tensor:
    """Convert 6D rotation representation to axis-angle.

    Uses pytorch3d's matrix_to_axis_angle for the matrix->axis-angle step.
    This function is differentiable and is also used by the MANO natural-pose
    prior.  Callers that compare against a reference pose must account for the
    axis-angle branch at pi (the prior does so with a reference-aware
    equivalent-representation lift).

    Args:
        d6: (*, 6) 6D rotation representation.

    Returns:
        (*, 3) axis-angle vectors.
    """
    return _pt3d_matrix_to_axis_angle(rotation_6d_to_matrix(d6))
