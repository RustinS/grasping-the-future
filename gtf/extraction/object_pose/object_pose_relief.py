"""Depth-relief term for the object bundle adjustment (upgraded chain; chains.ObjectBA).

2D tracks of a small, far object barely constrain its out-of-plane tilt. The relief term asks, in every frame, that the
BA points' camera depths match the MoGe depth sampled at their visible track pixels inside the object mask, after
removing that frame's mean offset. Only the relief (tilt and shape) enters, so the term is blind to MoGe's per-frame
depth scale and offset. The upgraded chain uses it without the first-order rotation smoothness (`rotation_smooth=False`),
and its optimizer keeps the BA rotation (w_data_obj_rot 30, w_hold_obj_rot 0). With relief_weight 0 and rotation_smooth
True the adjuster behaves as ObjectBundleAdjuster.
"""

import numpy as np
import torch

from gtf.extraction.object_pose.bundle_adjustment import ObjectBundleAdjuster


class ReliefBundleAdjuster(ObjectBundleAdjuster):
    def __init__(self, tracks, visibility, depth_results, cam_to_world, ref_frame_idx, *, object_masks_all, relief_weight, relief_delta=0.15, rotation_smooth=True, **kw):
        self._depth_results = depth_results
        self._masks_all = np.asarray(object_masks_all)
        self.relief_weight = float(relief_weight)
        self.relief_delta = float(relief_delta)
        self.rotation_smooth = bool(rotation_smooth)
        self._relief = None
        super().__init__(tracks, visibility, depth_results, cam_to_world, ref_frame_idx, **kw)

    def _relief_targets(self):
        """(T,N) MoGe depth at each track pixel, and where it is usable (visible, inside the object mask, valid depth)."""
        if self._relief is not None and self._relief[0].shape[1] == self.N:
            return self._relief
        tracks, vis = self.tracks.cpu().numpy(), self.visibility.cpu().numpy()
        target = np.zeros((self.T, self.N), np.float32)
        valid = np.zeros((self.T, self.N), bool)
        for t in range(self.T):
            depth = np.asarray(self._depth_results[t]["depth"])
            depth_ok = np.asarray(self._depth_results[t]["mask"]).astype(bool)
            mask = self._masks_all[t] > 0
            if mask.shape != depth.shape:
                continue
            H, W = depth.shape
            ix = np.clip(np.rint(tracks[t, :, 0]).astype(int), 0, W - 1)
            iy = np.clip(np.rint(tracks[t, :, 1]).astype(int), 0, H - 1)
            target[t] = depth[iy, ix]
            valid[t] = vis[t] & depth_ok[iy, ix] & mask[iy, ix] & np.isfinite(depth[iy, ix]) & (depth[iy, ix] > 0.01)
        valid &= valid.sum(1, keepdims=True) >= 8  # a tilt needs several points
        self._relief = (torch.tensor(target, device=self.device), torch.tensor(valid, device=self.device))
        return self._relief

    def _relief_loss(self, rot_vecs, trans_vecs, obj_points):
        target, valid = self._relief_targets()
        w = valid.float() * self.track_weights[None, :]
        n = w.sum(1, keepdim=True).clamp(min=1e-6)
        r = self._camera_depths(rot_vecs, trans_vecs, obj_points) - target
        r = (r - (r * w).sum(1, keepdim=True) / n) / (self.relief_delta * self.object_scale)  # the frame's mean offset removed
        a = r.abs()
        huber = torch.where(a < 1.0, 0.5 * r.square(), a - 0.5)
        return (huber * w).sum() / w.sum().clamp(min=1)

    def _compute_loss(self, rot_vecs, trans_vecs, obj_points, lambda_smooth, **kw):
        loss = super()._compute_loss(rot_vecs, trans_vecs, obj_points, lambda_smooth if self.rotation_smooth else 0.0, **kw)
        if self.relief_weight > 0:
            loss = loss + self.relief_weight * self._relief_loss(rot_vecs, trans_vecs, obj_points)
        return loss
