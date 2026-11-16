"""Differentiable MANO forward kinematics wrapper for training-loop use.

Wraps manopth.ManoLayer to accept 6D rotation inputs and output vertices,
joints, and fingertip positions in meters. Conventions match gtf/utils/canonical.py
(mano_fk_verts/joints) and the extraction optimizer's MANO forward pass.

Always uses MANO_RIGHT.pkl. Left hands are computed by mirroring the
x-coordinate after FK (M = diag(-1, 1, 1)).
"""

import torch
import torch.nn as nn
from manopth.manolayer import ManoLayer
from torch import Tensor

from gtf.paths import MANO_DIR
from gtf.model.repr.rotation import rotation_6d_to_axis_angle

# Fingertip joint indices after manopth's internal reordering
# Corresponds to: thumb tip, index tip, middle tip, ring tip, pinky tip
FINGERTIP_INDICES = [4, 8, 12, 16, 20]


class MANOHelper(nn.Module):
    """Differentiable MANO FK: 6D rotation inputs → vertices, joints, fingertips.

    Uses MANO_RIGHT.pkl for both hands. Left hand = right-hand FK with x-mirrored output.
    cam_t is added manually after FK (not passed to manopth) to ensure correct gradients.
    Output is in meters, camera space.
    """

    def __init__(self, mano_root: str = MANO_DIR):
        super().__init__()
        self.mano_layer = ManoLayer(
            mano_root=mano_root,
            side="right",
            use_pca=False,
            ncomps=45,
            flat_hand_mean=True,
        )
        # Register faces as buffers (not parameters — no gradient)
        faces = self.mano_layer.th_faces.long()
        self.register_buffer("faces", faces)
        self.register_buffer("faces_left", faces[:, [0, 2, 1]])  # reversed winding for left hand

        # The MANO layer is never trained.
        for param in self.mano_layer.parameters():
            param.requires_grad = False

    def forward(
        self,
        orient_6d: Tensor,
        pose_6d: Tensor,
        betas: Tensor,
        cam_t: Tensor,
        side: str = "right",
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run differentiable MANO forward kinematics.

        Args:
            orient_6d: (B, 6) global orientation in 6D rotation representation.
            pose_6d: (B, 90) finger pose in 6D rotation (15 joints × 6D).
            betas: (B, 10) shape parameters.
            cam_t: (B, 3) wrist translation in meters.
            side: 'right' or 'left'.

        Returns:
            verts: (B, 778, 3) hand vertices in meters, camera space.
            joints: (B, 21, 3) hand joints in meters, camera space.
            fingertips: (B, 5, 3) fingertip positions in meters.
        """
        # 1. Convert orient 6D → axis-angle
        orient_aa = rotation_6d_to_axis_angle(orient_6d)  # (B, 3)

        # 2. Convert pose 6D → axis-angle (per-joint)
        B = pose_6d.shape[0]
        pose_6d_per_joint = pose_6d.reshape(B, 15, 6)  # (B, 15, 6)
        pose_aa_per_joint = rotation_6d_to_axis_angle(pose_6d_per_joint)  # (B, 15, 3)
        pose_aa = pose_aa_per_joint.reshape(B, 45)  # (B, 45)

        # 3. Run manopth — NO cam_t (translation is added below)
        full_pose = torch.cat([orient_aa, pose_aa], dim=-1)  # (B, 48)
        verts_mm, joints_mm = self.mano_layer(full_pose, th_betas=betas)

        # 4. mm → meters
        verts = verts_mm / 1000.0
        joints = joints_mm / 1000.0

        # 5. Add translation manually — gradient-safe
        verts = verts + cam_t.unsqueeze(1)
        joints = joints + cam_t.unsqueeze(1)

        # 6. Left hand: negate x-coordinate
        # Out-of-place to avoid autograd in-place error
        if side == "left":
            verts = torch.cat([-verts[..., :1], verts[..., 1:]], dim=-1)
            joints = torch.cat([-joints[..., :1], joints[..., 1:]], dim=-1)

        # 7. Extract fingertips
        fingertips = joints[:, FINGERTIP_INDICES, :]  # (B, 5, 3)

        return verts, joints, fingertips
