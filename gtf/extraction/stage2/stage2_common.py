"""Shared state layout + utilities for the stage-2 trajectory refinement chain (refine_trajectory.py).

The stage-2 denoiser (denoiser_model.py, checkpoints/denoiser/v0.pt) was trained on ground-truth trajectories
(DexYCB + HOT3D) corrupted with a model of extraction noise.

State layout: one 207-D vector per frame, raw camera-frame channels (meters / 6D rotations),
same per-entity layout as gtf/model/repr/tokenizer.py:

    [ left_hand(99) | right_hand(99) | obj(9) ]
    hand: [orient_6d(6) | cam_t(3) | pose_6d(90)]
    obj:  [rot_6d(6) | trans(3)]

Not gravity-aligned and not normalized with the forecasting model's stats: extraction has no
gravity direction when stage 2 runs, so the denoiser lives in the raw camera frame with its own
normalization stats (stored in the checkpoint). It operates at the extraction keyframe rate (~10 Hz).
"""

from pathlib import Path

import torch

from gtf.model.repr.rotation import axis_angle_to_6d, axis_angle_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix
from gtf.utils import RESULTS_DIR as _SHARED_RESULTS_DIR

RESULTS_DIR = Path(_SHARED_RESULTS_DIR)

WINDOW_T = 50  # denoiser window in keyframes (longer segments use overlapping windows)

STATE_DIM = 207

# Channel slices into the packed 207-D state
LEFT = slice(0, 99)
RIGHT = slice(99, 198)
OBJ = slice(198, 207)
HAND_ORIENT = slice(0, 6)  # within a hand's 99
HAND_CAM_T = slice(6, 9)
OBJ_ROT = slice(0, 6)  # within obj's 9
OBJ_TRANS = slice(6, 9)
# Absolute translation slices (for centering)
TRANS_SLICES = [slice(6, 9), slice(105, 108), slice(204, 207)]  # left cam_t, right cam_t, obj trans


def mid_valid_frame(frame_valid: torch.Tensor) -> torch.Tensor:
    """(B,T) bool → (B,) index of the middle valid frame (origin anchor)."""
    fv = frame_valid.bool()
    counts = fv.sum(dim=1)
    # index of the (count//2)-th valid frame
    csum = fv.cumsum(dim=1)
    target = (counts // 2 + 1).clamp(min=1).unsqueeze(1)
    return (csum < target).sum(dim=1).clamp(max=fv.shape[1] - 1)


def compute_origin(states: torch.Tensor, frame_valid: torch.Tensor) -> torch.Tensor:
    """Centering origin = object position at the middle valid frame, computed from the
    (possibly noisy) states. Object only, not a hand/object midpoint: corrupted hand
    positions can be far off, and the object translation carries no scale corruption, so
    it is a robust anchor available identically at train and application time."""
    mid = mid_valid_frame(frame_valid)
    return states[torch.arange(states.shape[0], device=states.device), mid][:, TRANS_SLICES[2]]


def center_states(states: torch.Tensor, origin: torch.Tensor, sign: float = -1.0) -> torch.Tensor:
    """Add sign*origin to every absolute-translation slice. sign=-1 centers, sign=+1 undoes."""
    out = states.clone()
    for sl in TRANS_SLICES:
        out[..., sl] = out[..., sl] + sign * origin.unsqueeze(1)
    return out


__all__ = [
    "RESULTS_DIR",
    "WINDOW_T",
    "STATE_DIM",
    "LEFT",
    "RIGHT",
    "OBJ",
    "HAND_ORIENT",
    "HAND_CAM_T",
    "OBJ_ROT",
    "OBJ_TRANS",
    "TRANS_SLICES",
    "mid_valid_frame",
    "compute_origin",
    "center_states",
    "axis_angle_to_6d",
    "axis_angle_to_matrix",
    "matrix_to_rotation_6d",
    "rotation_6d_to_matrix",
]
