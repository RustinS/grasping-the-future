"""Bundle adjustment of the object pose over the point tracks, depth and camera poses."""

import cv2
import numpy as np
import roma
import torch
from gtf.utils import rprint as print


def _robust_limit(values, floor, clamp_max=None):
    """Compute median + 3*MAD threshold, clamped to [floor, clamp_max]."""
    values = values[np.isfinite(values)]
    if values.size == 0:
        return floor
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    if mad < 1e-6:
        mad = float(np.std(values))
    limit = max(floor, med + 3.0 * mad)
    if clamp_max is not None:
        limit = min(limit, clamp_max)
    return limit


class ObjectBundleAdjuster:
    """Joint optimization of object shape and per-frame 6D pose."""

    def __init__(
        self,
        tracks,
        visibility,
        depth_results,
        cam_to_world,
        ref_frame_idx,
        device="cuda",
        boundary_dists=None,
    ):
        self.device = torch.device(device)
        self.T, self.N = tracks.shape[:2]
        self.ref = ref_frame_idx

        self.tracks = torch.tensor(tracks, dtype=torch.float32, device=self.device)
        self.visibility = torch.tensor(visibility, dtype=torch.bool, device=self.device)
        self.cam_to_world = torch.tensor(cam_to_world, dtype=torch.float32, device=self.device)

        self.intrinsics = torch.stack([torch.tensor(dr["intrinsics_px"], dtype=torch.float32, device=self.device) for dr in depth_results])

        if boundary_dists is not None:
            bd = torch.tensor(boundary_dists, dtype=torch.float32, device=self.device)
            median_bd = bd.median().clamp(min=1.0)
            self.track_weights = torch.clamp(bd / (median_bd * 0.5), 0.1, 1.0)
        else:
            self.track_weights = torch.ones(self.N, dtype=torch.float32, device=self.device)

        self._init_object_points(depth_results)
        self._prune_spatial_outliers()
        self._init_poses()

    def _init_object_points(self, depth_results):
        """Lift tracked points to 3D at reference frame."""
        K_ref = depth_results[self.ref]["intrinsics_px"]
        depth_ref = depth_results[self.ref]["depth"]
        H, W = depth_ref.shape
        fx, fy = K_ref[0, 0], K_ref[1, 1]
        cx, cy = K_ref[0, 2], K_ref[1, 2]

        tracks_ref = self.tracks[self.ref].cpu().numpy()
        # A track pixel without valid depth takes the median valid depth of the
        # whole image (1.0 when there is none).
        sampled_depth = np.empty(self.N, dtype=np.float32)
        for i in range(self.N):
            x, y = tracks_ref[i]
            iy = np.clip(int(round(y)), 0, H - 1)
            ix = np.clip(int(round(x)), 0, W - 1)
            d = depth_ref[iy, ix]
            if not (d > 0.01 and np.isfinite(d)):
                valid_depths = depth_ref[depth_ref > 0.01]
                d = float(np.median(valid_depths)) if len(valid_depths) > 0 else 1.0
            sampled_depth[i] = d

        pts_cam = []
        for i in range(self.N):
            x, y = tracks_ref[i]
            d = float(sampled_depth[i])
            pts_cam.append([(x - cx) * d / fx, (y - cy) * d / fy, d])

        pts_cam = np.array(pts_cam, dtype=np.float32)

        c2w_ref = self.cam_to_world[self.ref].cpu().numpy()
        pts_world = (c2w_ref[:3, :3] @ pts_cam.T).T + c2w_ref[:3, 3]

        self.centroid = pts_world.mean(axis=0)
        self.object_points = torch.tensor(pts_world - self.centroid, dtype=torch.float32, device=self.device)
        centered_norm = np.linalg.norm(pts_world - self.centroid, axis=-1)
        self.object_scale = float(max(np.median(centered_norm), 1e-3))

    def _prune_spatial_outliers(self):
        """Remove 3D points far from the median (depth/spatial outliers)."""
        pts = self.object_points.cpu().numpy()
        median = np.median(pts, axis=0)
        dists = np.linalg.norm(pts - median, axis=1)
        mad = np.median(dists)
        if mad < 1e-6:
            mad = np.std(dists)
        threshold = max(mad * 4.0, 0.01)
        keep = dists < threshold
        if keep.sum() < 10:
            return
        n_pruned = self.N - keep.sum()
        if n_pruned > 0:
            keep_t = torch.tensor(keep, device=self.device)
            self.object_points = self.object_points[keep_t]
            self.tracks = self.tracks[:, keep_t]
            self.visibility = self.visibility[:, keep_t]
            self.track_weights = self.track_weights[keep_t]
            self.N = self.object_points.shape[0]
            pts_kept = self.object_points.cpu().numpy()
            new_center = pts_kept.mean(axis=0)
            self.object_points -= torch.tensor(new_center, dtype=torch.float32, device=self.device)
            self.centroid = self.centroid + new_center
            print(f"  Pruned {n_pruned} spatial outliers, {self.N} points remain")

    def _project_points(self, rot_vecs, trans_vecs, obj_points):
        """Project object points to pixel coordinates. Returns projected (T,N,2) and per-point dist (T,N)."""
        R_obj = roma.rotvec_to_rotmat(rot_vecs)
        pts_world = torch.einsum("tij,nj->tni", R_obj, obj_points) + trans_vecs[:, None, :]
        w2c = torch.inverse(self.cam_to_world)
        pts_cam = torch.einsum("tij,tnj->tni", w2c[:, :3, :3], pts_world) + w2c[:, :3, 3:4].transpose(1, 2)
        z = pts_cam[:, :, 2:3].clamp(min=0.01)
        pts_norm = pts_cam[:, :, :2] / z

        fx, fy = self.intrinsics[:, 0, 0], self.intrinsics[:, 1, 1]
        cx, cy = self.intrinsics[:, 0, 2], self.intrinsics[:, 1, 2]
        projected = torch.stack(
            [
                pts_norm[:, :, 0] * fx[:, None] + cx[:, None],
                pts_norm[:, :, 1] * fy[:, None] + cy[:, None],
            ],
            dim=-1,
        )

        dist = (projected - self.tracks).norm(dim=-1)
        return projected, dist

    def _camera_depths(self, rot_vecs, trans_vecs, obj_points):
        rotations = roma.rotvec_to_rotmat(rot_vecs)
        points_world = torch.einsum("tij,nj->tni", rotations, obj_points) + trans_vecs[:, None, :]
        world_to_camera = torch.inverse(self.cam_to_world)
        points_camera = torch.einsum("tij,tnj->tni", world_to_camera[:, :3, :3], points_world) + world_to_camera[:, :3, 3:4].transpose(1, 2)
        return points_camera[:, :, 2]

    def _prune_by_reprojection(self, rot_vecs, trans_vecs, obj_points, threshold_px=10.0):
        """Prune points with persistently high reprojection error after Phase 1."""
        with torch.no_grad():
            _, dist = self._project_points(rot_vecs, trans_vecs, obj_points)
            dist_np = dist.cpu().numpy()
            vis_np = self.visibility.cpu().numpy().astype(bool)
            valid = vis_np.sum(axis=0) > 0

            median_err = np.full(self.N, np.inf, dtype=np.float32)
            p90_err = np.full(self.N, np.inf, dtype=np.float32)
            outlier_frac = np.ones(self.N, dtype=np.float32)
            for i in np.where(valid)[0]:
                errs = dist_np[vis_np[:, i], i]
                median_err[i] = float(np.median(errs))
                p90_err[i] = float(np.percentile(errs, 90))
                outlier_frac[i] = float((errs > threshold_px).mean())

            median_limit = _robust_limit(
                median_err[valid],
                floor=0.6 * threshold_px,
                clamp_max=2.0 * threshold_px,
            )
            p90_limit = _robust_limit(p90_err[valid], floor=1.5 * threshold_px, clamp_max=3.0 * threshold_px)
            outlier_limit = _robust_limit(outlier_frac[valid], floor=0.35, clamp_max=0.75)

            keep_np = valid & (median_err <= median_limit) & (p90_err <= p90_limit) & (outlier_frac <= outlier_limit)
            keep = torch.from_numpy(keep_np).to(self.device)
            if keep.sum() < 10:
                return rot_vecs, trans_vecs, obj_points
            n_pruned = self.N - keep.sum().item()
            if n_pruned > 0:
                self.tracks = self.tracks[:, keep]
                self.visibility = self.visibility[:, keep]
                self.track_weights = self.track_weights[keep]
                self.N = keep.sum().item()
                obj_points = obj_points[keep].detach().clone()
                if hasattr(self, "object_points_prior"):
                    self.object_points_prior = self.object_points_prior[keep]
                print(f"  Pruned {n_pruned} high-error points (median<={median_limit:.1f}px, p90<={p90_limit:.1f}px, outlier_frac<={outlier_limit:.2f}), {self.N} remain")
        return rot_vecs, trans_vecs, obj_points

    def _init_poses(self):
        """Initialize per-frame object poses with PnP (EPnP RANSAC + LM refinement) on the visible tracks.

        A frame without a PnP solution keeps the identity rotation at the centroid. Coverage is
        logged, with a warning when fewer than half of the evidence frames are initialized: from
        such a start phase 1 sees large residuals in the linear part of the Huber loss and can
        diverge.
        """
        self.rot_vecs = torch.zeros(self.T, 3, dtype=torch.float32, device=self.device)
        self.trans_vecs = torch.tensor(np.tile(self.centroid, (self.T, 1)), dtype=torch.float32, device=self.device)

        pts_local = self.object_points.cpu().numpy().astype(np.float64)
        n_init = 0
        n_evidence = 0

        for t in range(self.T):
            if t == self.ref:
                continue

            vis_mask = self.visibility[t].cpu().numpy()
            if vis_mask.sum() < 4:
                continue
            n_evidence += 1

            pts_2d = self.tracks[t, vis_mask].cpu().numpy().astype(np.float64)
            pts_3d = pts_local[vis_mask]
            K = self.intrinsics[t].cpu().numpy().astype(np.float64)
            c2w = self.cam_to_world[t].cpu().numpy()
            dist_coeffs = np.zeros(4, dtype=np.float64)

            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_3d,
                pts_2d,
                K[:3, :3],
                dist_coeffs,
                iterationsCount=500,
                reprojectionError=6.0,
                flags=cv2.SOLVEPNP_EPNP,
            )

            if not (success and inliers is not None and len(inliers) >= 4):
                continue

            inlier_idx = inliers[:, 0]
            rvec, tvec = cv2.solvePnPRefineLM(
                pts_3d[inlier_idx],
                pts_2d[inlier_idx],
                K[:3, :3],
                dist_coeffs,
                rvec,
                tvec,
            )

            R_cam_obj, _ = cv2.Rodrigues(rvec)
            T_cam_obj = np.eye(4, dtype=np.float64)
            T_cam_obj[:3, :3] = R_cam_obj
            T_cam_obj[:3, 3] = tvec.ravel()
            T_obj = c2w @ T_cam_obj

            rot_vec_t = roma.rotmat_to_rotvec(torch.tensor(T_obj[:3, :3].astype(np.float32)).unsqueeze(0))[0]
            self.rot_vecs[t] = rot_vec_t.to(self.device)
            self.trans_vecs[t] = torch.tensor(T_obj[:3, 3], dtype=torch.float32, device=self.device)
            n_init += 1

        print(f"  PnP init: {n_init}/{n_evidence} evidence frames initialized ({self.T} total)")
        if n_evidence >= 5 and n_init < 0.5 * n_evidence:
            print(f"  WARNING: PnP init coverage {n_init}/{n_evidence} < 50% — BA starts near identity/centroid and may diverge")

    def optimize(self, phase1_iters=200, phase2_iters=500):
        """Run two-phase bundle adjustment.

        Phase 1: optimize poses only (fix points)
        Phase 2: joint optimization of poses + points
        """
        rot_vecs = self.rot_vecs.clone().requires_grad_(True)
        trans_vecs = self.trans_vecs.clone().requires_grad_(True)
        obj_points = self.object_points.clone().requires_grad_(False)

        lambda_smooth = 0.1
        lambda_rigid = 2.0

        print(f"  BA Phase 1: optimizing poses ({phase1_iters} iters)...")
        optimizer = torch.optim.Adam([rot_vecs, trans_vecs], lr=1e-2)
        for _ in range(phase1_iters):
            optimizer.zero_grad()
            loss = self._compute_loss(rot_vecs, trans_vecs, obj_points, lambda_smooth)
            loss.backward()
            rot_vecs.grad[self.ref] = 0
            trans_vecs.grad[self.ref] = 0
            optimizer.step()

        rot_vecs, trans_vecs, obj_points = self._prune_by_reprojection(rot_vecs, trans_vecs, obj_points, threshold_px=10.0)
        rot_vecs = rot_vecs.detach().clone().requires_grad_(True)
        trans_vecs = trans_vecs.detach().clone().requires_grad_(True)
        self.object_points_prior = obj_points.detach().clone()
        prior_scale = (
            torch.linalg.norm(
                self.object_points_prior - self.object_points_prior.mean(dim=0, keepdim=True),
                dim=-1,
            )
            .median()
            .clamp(min=1e-3)
        )

        print(f"  BA Phase 2: joint optimization ({phase2_iters} iters)...")
        obj_points = obj_points.clone().requires_grad_(True)
        optimizer = torch.optim.Adam(
            [
                {"params": [rot_vecs, trans_vecs], "lr": 5e-3},
                {"params": [obj_points], "lr": 5e-3},
            ]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase2_iters)

        for _ in range(phase2_iters):
            optimizer.zero_grad()
            loss = self._compute_loss(
                rot_vecs,
                trans_vecs,
                obj_points,
                lambda_smooth,
                obj_points_prior=self.object_points_prior,
                lambda_rigid=lambda_rigid,
                rigid_scale=prior_scale,
            )
            loss.backward()
            rot_vecs.grad[self.ref] = 0
            trans_vecs.grad[self.ref] = 0
            optimizer.step()
            scheduler.step()

        self.rot_vecs = rot_vecs.detach()
        self.trans_vecs = trans_vecs.detach()
        self.object_points = obj_points.detach()

        reproj_errors = self._compute_per_frame_errors()
        n_ev = int(np.isfinite(reproj_errors).sum())
        if n_ev:
            print(
                f"  BA done: mean reproj error = {np.nanmean(reproj_errors):.2f}px, median = {np.nanmedian(reproj_errors):.2f}px ({n_ev}/{len(reproj_errors)} evidence frames)"
            )
        else:
            print(f"  BA done: no evidence frames (all {len(reproj_errors)} frames < 4 visible tracks) — trajectory unconstrained")
        return reproj_errors

    def _compute_loss(
        self,
        rot_vecs,
        trans_vecs,
        obj_points,
        lambda_smooth,
        obj_points_prior=None,
        lambda_rigid=0.0,
        rigid_scale=None,
    ):
        """Compute BA loss: reprojection + temporal smoothness + rigidity prior."""
        _, dist = self._project_points(rot_vecs, trans_vecs, obj_points)
        delta = 2.0
        huber = torch.where(dist < delta, 0.5 * dist**2, delta * (dist - 0.5 * delta))

        vis_float = self.visibility.float()
        w = self.track_weights[None, :]
        reproj_loss = (huber * vis_float * w).sum() / (vis_float * w).sum().clamp(min=1)

        if self.T > 1:
            R_obj = roma.rotvec_to_rotmat(rot_vecs)
            R_diff = torch.bmm(R_obj[:-1].transpose(1, 2), R_obj[1:])
            smooth_loss = roma.rotmat_to_rotvec(R_diff).pow(2).sum(dim=-1).mean()
        else:
            smooth_loss = torch.tensor(0.0, device=self.device)

        rigid_loss = torch.tensor(0.0, device=self.device)
        if obj_points_prior is not None and lambda_rigid > 0.0:
            scale = rigid_scale if rigid_scale is not None else torch.tensor(1.0, device=self.device)
            rigid_loss = ((obj_points - obj_points_prior) / scale).pow(2).sum(dim=-1).mean()

        return reproj_loss + lambda_smooth * smooth_loss + lambda_rigid * rigid_loss

    def _compute_per_frame_errors(self, min_vis=4):
        """Mean reprojection error per frame; frames with < min_vis visible tracks are NaN.

        NaN keeps frames without evidence out of the downstream 12 px median acceptance
        (optimizer_common._load_ba_camera_poses, which uses np.nanmedian).
        """
        with torch.no_grad():
            _, dist = self._project_points(self.rot_vecs, self.trans_vecs, self.object_points)
            vis_float = self.visibility.float()
            n_vis = vis_float.sum(dim=1)
            per_frame = (dist * vis_float).sum(dim=1) / n_vis.clamp(min=1)
            per_frame[n_vis < min_vis] = float("nan")
            return per_frame.cpu().numpy()

    def get_results(self):
        """Return optimized poses and points."""
        R_obj = roma.rotvec_to_rotmat(self.rot_vecs).detach().cpu().numpy()
        t_obj = self.trans_vecs.detach().cpu().numpy()
        pts = self.object_points.detach().cpu().numpy()
        return R_obj, t_obj, pts
