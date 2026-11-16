"""Differentiable natural-pose prior from the MANO training distribution.

MANO's forward model is a skinned kinematic hand, not a biomechanics engine:
``use_pca=False`` accepts any collection of 15 local SO(3) rotations.  The
distributed MANO asset nevertheless contains a useful statistical prior:

* ``hands_mean`` is the mean 45-D axis-angle articulation;
* each row of ``hands_components`` is an orthogonal PCA direction scaled by
  that component's standard deviation;
* ``hands_coeffs`` are the corresponding standardized training coefficients.

This module recovers those standardized coefficients from either the project's
15x6-D joint rotations or MANO's native 45-D axis-angle pose.  A coefficient
square therefore has a common scale across modes, including the small-variance
tail modes that often manifest as implausible finger contortions.

The training loss (:meth:`excess`) penalizes component-wise prior energy only
when it exceeds the clean reference trajectory.  Consequently an unusual real
grasp is a stationary zero-loss target and is never projected toward MANO's
mean pose.  This is a soft prior, not joint clipping or a hard PCA-subspace
projection.

Left-hand convention is intentionally identical to right: this project stores
both hands' articulation in the MANO_RIGHT parameter frame and mirrors only the
left FK output geometry.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Literal

import numpy as np
import torch
from manopth.manolayer import ManoLayer
from torch import Tensor, nn

from gtf.model.repr.rotation import rotation_6d_to_axis_angle

PoseRepresentation = Literal["axis_angle", "rotation_6d"]
Reduction = Literal["none", "mean", "sum"]


def _reduce_components(value: Tensor, reduction: Reduction) -> Tensor:
    if reduction == "none":
        return value
    if reduction == "mean":
        return value.mean(dim=-1)
    if reduction == "sum":
        return value.sum(dim=-1)
    raise ValueError(f"unknown component reduction {reduction!r}")


class MANONaturalPosePrior(nn.Module):
    """MANO's standardized 45-component articulation prior.

    Built from the asset data (``smpl_data``) of an existing ``ManoLayer``, whose
    ``hands_mean`` does not depend on the layer's ``flat_hand_mean`` setting. The
    module owns only small constant buffers, not a second FK layer.
    """

    n_components = 45

    def __init__(self, *, mano_layer: ManoLayer) -> None:
        super().__init__()
        data = mano_layer.smpl_data
        mean = torch.as_tensor(np.asarray(data["hands_mean"]).copy(), dtype=torch.float32).reshape(self.n_components)
        components = torch.as_tensor(np.asarray(data["hands_components"]).copy(), dtype=torch.float32)
        if components.shape != (self.n_components, self.n_components):
            raise ValueError(f"expected MANO hands_components (45,45), got {tuple(components.shape)}")

        norm_sq = components.square().sum(dim=-1)
        if bool((norm_sq <= 1e-12).any()):
            raise ValueError("MANO pose prior contains a zero-length PCA component")
        # MANO stores orthogonal directions with variance folded into the row
        # norms.  Validate that contract instead of silently using an invalid
        # diagonal inverse if a different asset format is supplied.
        gram = components @ components.T
        off_diag = gram - torch.diag_embed(torch.diagonal(gram))
        relative_off_diag = off_diag.abs().max() / norm_sq.max()
        if not torch.isfinite(relative_off_diag) or float(relative_off_diag) > 1e-5:
            raise ValueError(f"MANO hands_components rows are not orthogonal (relative residual={float(relative_off_diag):.3g})")

        self.register_buffer("pose_mean", mean)
        self.register_buffer("pose_components", components)
        self.register_buffer("component_norm_sq", norm_sq)

        # Quantiles of the mean squared coefficient over the asset's training
        # coefficients (around one; the empirical distribution is non-Gaussian).
        # Diagnostic buffers stored with the checkpoint; the loss does not use them.
        coeffs = torch.as_tensor(np.asarray(data["hands_coeffs"]).copy(), dtype=torch.float32)
        if coeffs.ndim != 2 or coeffs.shape[1] != self.n_components:
            raise ValueError(f"expected MANO hands_coeffs (N,45), got {tuple(coeffs.shape)}")
        quantile_levels = torch.tensor([0.50, 0.90, 0.95, 0.99], dtype=torch.float32)
        reference_scores = coeffs.square().mean(dim=-1)
        self.register_buffer("reference_quantile_levels", quantile_levels)
        self.register_buffer("reference_score_quantiles", torch.quantile(reference_scores, quantile_levels))

    @classmethod
    def from_mano_helper(cls, mano_helper: nn.Module) -> MANONaturalPosePrior:
        """Construct from the project's existing ``MANOHelper`` instance."""

        if not hasattr(mano_helper, "mano_layer"):
            raise TypeError("mano_helper must expose its ManoLayer as .mano_layer")
        return cls(mano_layer=mano_helper.mano_layer)

    @staticmethod
    def _validate_side(side: str) -> None:
        if side not in {"left", "right"}:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    def to_axis_angle(self, pose: Tensor, *, representation: PoseRepresentation) -> Tensor:
        """Convert ``(...,45)/(..,15,3)`` or ``(...,90)/(..,15,6)`` to ``(...,45)``."""

        if representation == "axis_angle":
            if pose.shape[-2:] == (15, 3):
                return pose.reshape(pose.shape[:-2] + (self.n_components,))
            if pose.shape[-1:] == (self.n_components,):
                return pose
            raise ValueError(f"axis-angle MANO pose must end in (45,) or (15,3), got {tuple(pose.shape)}")

        if representation == "rotation_6d":
            if pose.shape[-2:] == (15, 6):
                per_joint = pose
                lead = pose.shape[:-2]
            elif pose.shape[-1:] == (15 * 6,):
                lead = pose.shape[:-1]
                per_joint = pose.reshape(lead + (15, 6))
            else:
                raise ValueError(f"6-D MANO pose must end in (90,) or (15,6), got {tuple(pose.shape)}")
            return rotation_6d_to_axis_angle(per_joint).reshape(lead + (self.n_components,))

        raise ValueError(f"unknown MANO pose representation {representation!r}")

    def _coefficients_from_axis_angle(self, axis_angle: Tensor) -> Tensor:
        """Invert MANO's variance-scaled PCA synthesis for an axis-angle pose: standardized ``(...,45)``.

        The smallest MANO PCA direction has std 0.036 rad, so bf16 arithmetic in the inverse can
        materially quantize its standardized coefficient; this runs in fp32 under mixed precision.
        """

        with torch.autocast(device_type=axis_angle.device.type, enabled=False) if axis_angle.device.type in {"cpu", "cuda"} else nullcontext():
            axis_angle = axis_angle.float() if axis_angle.dtype in {torch.float16, torch.bfloat16} else axis_angle
            mean = self.pose_mean.to(device=axis_angle.device, dtype=axis_angle.dtype)
            components = self.pose_components.to(device=axis_angle.device, dtype=axis_angle.dtype)
            norm_sq = self.component_norm_sq.to(device=axis_angle.device, dtype=axis_angle.dtype)
            delta = axis_angle - mean
            # Rows of components are orthogonal and include component std.  This
            # is exactly the inverse of manopth's coeff @ components + hands_mean.
            return torch.einsum("...d,kd->...k", delta, components) / norm_sq

    def _reference_aligned_axis_angle(
        self,
        pred_pose: Tensor,
        reference_pose: Tensor,
        *,
        representation: PoseRepresentation,
    ) -> tuple[Tensor, Tensor]:
        """Return principal reference AA and the nearest equivalent predicted AA.

        The principal SO(3)->axis-angle logarithm jumps from ``+pi*a`` to
        ``-pi*a`` at the branch cut even though the underlying rotations are
        adjacent.  For each finger joint independently, lift the predicted
        principal representative by ``k*2*pi`` along its axis, ``k in
        {0,-1,+1}``, choosing the representative nearest the clean reference.
        This is a local supervised comparison, not a combinatorial pose search.
        """

        pred_flat = self.to_axis_angle(pred_pose, representation=representation)
        reference_flat = self.to_axis_angle(reference_pose, representation=representation)
        pred_aa = pred_flat.reshape(pred_flat.shape[:-1] + (15, 3))
        reference_aa = reference_flat.reshape(reference_flat.shape[:-1] + (15, 3))
        # The 6-D path yields principal representatives for both tensors;
        # axis-angle inputs keep their supplied convention, and the lift below
        # still handles +/-2pi-separated predictions.

        theta = pred_aa.norm(dim=-1, keepdim=True)
        axis = pred_aa / theta.clamp(min=1e-8)
        # Put k=0 first so exact/tiny-angle ties keep the principal branch.
        k = pred_aa.new_tensor([0.0, -1.0, 1.0])
        candidates = pred_aa.unsqueeze(-2) + (2.0 * torch.pi) * k.view(*(1,) * (pred_aa.ndim - 1), 3, 1) * axis.unsqueeze(-2)
        distances = (candidates - reference_aa.unsqueeze(-2)).square().sum(dim=-1)
        selected = distances.detach().argmin(dim=-1, keepdim=True)
        aligned = candidates.gather(-2, selected.unsqueeze(-1).expand(*selected.shape, 3)).squeeze(-2)
        return aligned.reshape(aligned.shape[:-2] + (self.n_components,)), reference_aa.reshape(reference_aa.shape[:-2] + (self.n_components,))

    def excess(
        self,
        pred_pose: Tensor,
        reference_pose: Tensor,
        *,
        representation: PoseRepresentation,
        side: str = "right",
        reduction: Reduction = "mean",
        reference_floor: float = 1.0,
    ) -> Tensor:
        """Return component-wise prior energy newly introduced above reference.

        The positive mask and reference energy are detached.  At
        ``pred_pose == reference_pose`` both loss and gradient are exactly zero;
        a prediction that is more likely than the reference is also unpenalized.
        Component-wise comparison prevents one improved finger mode from
        cancelling a newly implausible mode elsewhere.

        Each excess is divided by ``max(reference_energy, reference_floor**2)``.
        Valid source conventions of the mixed corpus span a wide range of
        absolute MANO energies, so an unnormalized excess would give
        high-energy sources much larger gradients for the same fractional
        deterioration.  A floor of one corresponds to one standardized MANO
        coefficient and keeps the natural region well-conditioned.
        """

        if reference_floor <= 0:
            raise ValueError(f"reference_floor must be positive, got {reference_floor}")

        self._validate_side(side)
        context = torch.autocast(device_type=pred_pose.device.type, enabled=False) if pred_pose.device.type in {"cpu", "cuda"} else nullcontext()
        with context:
            work_pred = pred_pose.float() if pred_pose.dtype in {torch.float16, torch.bfloat16} else pred_pose
            work_reference = reference_pose.float() if reference_pose.dtype in {torch.float16, torch.bfloat16} else reference_pose
            pred_aa, reference_aa = self._reference_aligned_axis_angle(work_pred, work_reference, representation=representation)
            pred_energy = self._coefficients_from_axis_angle(pred_aa).square()
            reference_energy = self._coefficients_from_axis_angle(reference_aa).square().detach()
        delta = pred_energy - reference_energy
        positive_excess = delta * (delta > 0).detach().to(delta.dtype)
        positive_excess = positive_excess / reference_energy.clamp(min=reference_floor * reference_floor)
        return _reduce_components(positive_excess, reduction)
