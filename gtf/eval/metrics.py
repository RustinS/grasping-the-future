"""Per-sample evaluation metrics: translation ADE/FDE of the object and both wrists.

Every metric is computed over the entity's predicted frames (masked, present and valid) and
returned in millimeters. A metric is omitted when the entity has no predicted frame.
"""

import torch
from torch import Tensor


def translation_ade(pred: Tensor, gt: Tensor, mask: Tensor) -> Tensor:
    """Average displacement error: mean L2 over the masked frames.

    Args:
        pred: (T, 3) predicted translations in meters.
        gt: (T, 3) ground-truth translations in meters.
        mask: (T,) bool, frames to evaluate.

    Returns:
        Scalar in millimeters. NaN if no frame is masked.
    """
    if not mask.any():
        return torch.tensor(float("nan"), device=pred.device)
    dists = (pred[mask] - gt[mask]).norm(dim=-1)
    return dists.mean() * 1000.0


def translation_fde(pred: Tensor, gt: Tensor, mask: Tensor) -> Tensor:
    """Final displacement error: L2 at the last masked frame.

    Args:
        pred: (T, 3) predicted translations in meters.
        gt: (T, 3) ground-truth translations in meters.
        mask: (T,) bool.

    Returns:
        Scalar in millimeters. NaN if no frame is masked.
    """
    if not mask.any():
        return torch.tensor(float("nan"), device=pred.device)
    last_idx = mask.nonzero(as_tuple=False)[-1].item()
    return (pred[last_idx] - gt[last_idx]).norm() * 1000.0


def compute_all_metrics(
    pred_decoded: dict[str, Tensor],
    gt_decoded: dict[str, Tensor],
    obs_mask: Tensor,
    presence_mask: Tensor,
    valid_frames: Tensor,
) -> dict[str, float]:
    """Translation ADE/FDE for one sample of one window (unbatched).

    Args:
        pred_decoded: decoded prediction (axis-angle format) with obj_translation and
            hand_{left,right}_cam_t of shape (T, 3).
        gt_decoded: ground truth in the same format.
        obs_mask: (T, 5) bool, True = observed.
        presence_mask: (T, 5) bool.
        valid_frames: (T,) bool.

    Returns:
        Flat dict with obj_{ade,fde}_mm and hand_{left,right}_{ade,fde}_mm.
    """
    eval_mask = ~obs_mask & presence_mask & valid_frames.unsqueeze(-1)  # (T, 5)
    metrics = {}

    obj_mask = eval_mask[:, 2]
    if obj_mask.any():
        metrics["obj_ade_mm"] = translation_ade(pred_decoded["obj_translation"], gt_decoded["obj_translation"], obj_mask).item()
        metrics["obj_fde_mm"] = translation_fde(pred_decoded["obj_translation"], gt_decoded["obj_translation"], obj_mask).item()

    for side, slot in (("left", 0), ("right", 1)):
        hand_mask = eval_mask[:, slot]
        if not hand_mask.any():
            continue
        cam_t_key = f"hand_{side}_cam_t"
        metrics[f"hand_{side}_ade_mm"] = translation_ade(pred_decoded[cam_t_key], gt_decoded[cam_t_key], hand_mask).item()
        metrics[f"hand_{side}_fde_mm"] = translation_fde(pred_decoded[cam_t_key], gt_decoded[cam_t_key], hand_mask).item()

    return metrics
