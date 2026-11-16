"""Differentiable gross-contortion safety terms for MANO articulation.

MANO maps arbitrary local SO(3) joint rotations to a valid skinned mesh; it is
not a biomechanics engine and does not impose joint limits.  This module
expresses the conservative limits of the extraction's stage-2 articulation
filter in a form that can also be evaluated on decoded model predictions:

* axial twist: 25 degrees at MCP/CMC joints, 15 degrees at PIP/DIP joints;
* total swing: 135 degrees (a deliberately loose, gross-failure safety cap).

The limits are expressed around each shaped MANO hand's rest-bone axis.  They
are less prescriptive than a low-dimensional PCA pose prior and are not a
complete physiological model.  Fitted motion-capture poses occasionally cross
the twist limits, so the training loss acts only on the excess above the clean
target's own violation: obvious decoded contortions are penalized while unusual
real grasps are not pulled toward a canonical pose.
"""

from __future__ import annotations

import math

import torch
from pytorch3d.transforms import matrix_to_quaternion
from torch import Tensor

# MANO pose parameter k corresponds to skeleton joint k + 1.  In MANO's native
# pose order each finger occupies three consecutive joints, so 0,3,... are the
# MCP/CMC joints.
MANO_MCP_MASK = torch.tensor([j % 3 == 0 for j in range(15)], dtype=torch.bool)

# Fingertip vertices used to define the rest axis of terminal MANO joints.  The
# keys are native MANO skeleton indices (pose index + 1).
_END_TIP_VERT = {3: 317, 6: 444, 9: 673, 12: 556, 15: 745}


def mano_rest_bone_axes(mano_layer, betas: Tensor) -> Tensor:
    """Return shaped right-model rest-bone axes for each MANO pose joint.

    Args:
        mano_layer: ``manopth.ManoLayer`` instance used by the model.
        betas: ``(..., 10)`` MANO shape coefficients.

    Returns:
        ``(..., 15, 3)`` unit axes.  Left hands deliberately use the same axes:
        this project stores left articulation in right-model coordinates and
        mirrors only the FK output.
    """

    if betas.shape[-1] != 10:
        raise ValueError(f"MANO betas must end in 10 values, got {tuple(betas.shape)}")
    lead = betas.shape[:-1]
    flat = betas.reshape(-1, 10)
    dtype, device = flat.dtype, flat.device

    shapedirs = mano_layer.th_shapedirs.to(device=device, dtype=dtype)
    template = mano_layer.th_v_template.to(device=device, dtype=dtype).reshape(778, 3)
    reg = mano_layer.th_J_regressor.to(device=device, dtype=dtype)
    if reg.layout != torch.strided:
        reg = reg.to_dense()

    verts = template.unsqueeze(0) + torch.einsum("vck,bk->bvc", shapedirs, flat)
    joints = torch.einsum("jv,bvc->bjc", reg, verts)

    axes = []
    for pose_idx in range(15):
        joint_idx = pose_idx + 1
        child = verts[:, _END_TIP_VERT[joint_idx]] if joint_idx in _END_TIP_VERT else joints[:, joint_idx + 1]
        axis = child - joints[:, joint_idx]
        axes.append(torch.nn.functional.normalize(axis, dim=-1, eps=1e-8))
    return torch.stack(axes, dim=1).reshape(lead + (15, 3))


def swing_twist_angles(rotation: Tensor, bone_axes: Tensor) -> tuple[Tensor, Tensor]:
    """Decompose local rotations into signed twist and unsigned swing angles.

    ``rotation`` ends in ``(15,3,3)`` and ``bone_axes`` is broadcastable to
    ``rotation.shape[:-2] + (3,)``.  Returned angles have shape
    ``rotation.shape[:-2]`` and are in radians.
    """

    if rotation.shape[-3:] != (15, 3, 3):
        raise ValueError(f"expected (...,15,3,3) MANO rotations, got {tuple(rotation.shape)}")
    axes = torch.nn.functional.normalize(bone_axes.to(device=rotation.device, dtype=rotation.dtype), dim=-1, eps=1e-8)
    q = matrix_to_quaternion(rotation)  # (...,15,4), real-first
    # Canonicalize the quaternion double cover so signed twist lies in [-pi,pi].
    q = q * torch.where(q[..., :1] < 0, -torch.ones_like(q[..., :1]), torch.ones_like(q[..., :1]))
    real, imag = q[..., 0], q[..., 1:]
    axial = (imag * axes).sum(dim=-1)
    perpendicular = imag - axial.unsqueeze(-1) * axes
    twist_den = torch.sqrt((real * real + axial * axial).clamp(min=1e-12))
    twist = 2.0 * torch.atan2(axial, real.clamp(min=0.0))
    swing = 2.0 * torch.atan2(perpendicular.norm(dim=-1), twist_den)
    return twist, swing


def anatomy_violation_terms(
    rotation: Tensor,
    bone_axes: Tensor,
    *,
    twist_mcp_deg: float = 25.0,
    twist_pip_deg: float = 15.0,
    swing_deg: float = 135.0,
    violation_tolerance_deg: float = 0.1,
) -> dict[str, Tensor]:
    """Return per-joint conservative MANO safety-limit violation terms.

    ``energy`` is dimensionless: squared excess normalized by the corresponding
    limit.  The detached positive masks make an exactly-on-limit pose stationary
    (no non-zero subgradient at the boundary).
    """

    twist, swing = swing_twist_angles(rotation, bone_axes)
    mcp = MANO_MCP_MASK.to(device=rotation.device)
    twist_limit = torch.where(
        mcp,
        torch.full((15,), math.radians(twist_mcp_deg), device=rotation.device, dtype=rotation.dtype),
        torch.full((15,), math.radians(twist_pip_deg), device=rotation.device, dtype=rotation.dtype),
    )
    swing_limit = torch.full((15,), math.radians(swing_deg), device=rotation.device, dtype=rotation.dtype)

    dt = twist.abs() - twist_limit
    ds = swing - swing_limit
    twist_excess = dt * (dt > 0).detach().to(dt.dtype)
    swing_excess = ds * (ds > 0).detach().to(ds.dtype)
    energy = (twist_excess / twist_limit).square() + (swing_excess / swing_limit).square()
    # The energy uses the exact excess; the violation flags ignore excess below
    # violation_tolerance_deg, the floating-point residue of rotation round trips.
    violation_tol = math.radians(violation_tolerance_deg)
    return {
        "twist_rad": twist,
        "swing_rad": swing,
        "twist_excess_rad": twist_excess,
        "swing_excess_rad": swing_excess,
        "twist_violation": twist_excess > violation_tol,
        "swing_violation": swing_excess > violation_tol,
        "energy": energy,
    }


def excess_over_reference(pred_energy: Tensor, reference_energy: Tensor) -> Tensor:
    """Positive safety-limit energy above a detached clean-reference trajectory."""

    delta = pred_energy - reference_energy.detach()
    return delta * (delta > 0).detach().to(delta.dtype)
