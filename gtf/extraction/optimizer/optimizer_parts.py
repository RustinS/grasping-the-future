"""Setup and geometry methods of MANOPoseOptimizer (mixin classes; the class itself is in pose_optimizer.py)."""

import numpy as np
import torch
import torch.nn.functional as F
import roma
import warp as wp
from scipy.spatial import cKDTree
import kaolin
from gtf.extraction.optimizer.contact_depth import align_hand_depth, order_evidence
from gtf.extraction.optimizer.optimizer_common import (
    OptimizerGeometryError,
    SILHOUETTE_DIRS,
    SILHOUETTE_N_DIRS,
    _is_warp_kernel_build_error,
    _uniform_subset,
    _validate_corrected_hand_depth,
    _warp_query_closest_winding,
    winding_sign,
)
from gtf.extraction.optimizer.scale_anchors import _FOLLOW_LOCAL_GAP


class OptimizerSetupMixin:
    """Initial state and targets of MANOPoseOptimizer: joint limits, grasp / carry / contact targets, smoothed initial poses, depth corrections."""

    def _first_valid_fk(self, side):
        """MANO FK on the first valid frame → (verts, joints) each (V/J, 3) on CPU."""
        t = torch.where(getattr(self, f"valid_{side}"))[0][0].item()
        with torch.no_grad():
            verts, joints = self.mano_fwd[side](
                getattr(self, f"global_orient_init_{side}")[t : t + 1].to(self.device),
                getattr(self, f"hand_pose_init_{side}")[t : t + 1].to(self.device),
                getattr(self, f"betas_const_{side}").unsqueeze(0).to(self.device),
                getattr(self, f"cam_t_init_{side}")[t : t + 1].to(self.device),
            )
        return verts[0].cpu(), joints[0].cpu()

    def _init_fingertip_indices(self):
        """Find fingertip vertex indices from the MANO mesh topology."""
        for side in self.active_sides:
            verts, joints = self._first_valid_fk(side)
            tip_vert_ids = [(verts - joints[ji]).pow(2).sum(-1).topk(20, largest=False)[1] for ji in [4, 8, 12, 16, 20]]
            self.register_buffer(f"fingertip_verts_{side}", torch.cat(tip_vert_ids))
            self.register_buffer(f"thumb_tip_verts_{side}", tip_vert_ids[0].clone())

    def _init_physics_vertex_weights(self):
        """Downweight unreliable thumb-tip vertices in collision losses."""
        for side in self.active_sides:
            w = torch.ones(778, dtype=torch.float32)
            w[torch.unique(getattr(self, f"thumb_tip_verts_{side}").cpu())] = self.thumb_vertex_weight
            self.register_buffer(f"physics_vertex_weights_{side}", w)

    def _init_joint_limits(self):
        """Soft anatomical joint-angle norm limits. Thumb excluded (inf/0)."""
        max_norm = torch.full((15,), float("inf"), dtype=torch.float32)
        stiffness = torch.zeros(15, dtype=torch.float32)
        # Fingers 1-4 (skip thumb): MCP, PIP, DIP limits and stiffness
        idx = torch.arange(3, 15)
        max_norm[idx] = torch.tensor([1.4, 1.8, 1.5]).repeat(4)
        stiffness[idx] = torch.tensor([1.0, 1.5, 2.5]).repeat(4)
        self.register_buffer("joint_max_norm", max_norm)
        self.register_buffer("joint_stiffness", stiffness)

    def _init_axis_regularization(self):
        """Precompute per-frame reference axes for axis regularization.

        Penalizes perpendicular (axis-changing) components of finger pose deltas
        more than along-axis (flexion-changing) components, preventing unnatural
        splay and twist.
        """
        perp_w = torch.ones(15, dtype=torch.float32)
        perp_w[[1, 2, 4, 5, 7, 8, 10, 11, 13, 14]] = 15.0  # PIP/DIP hinge joints
        perp_w[[3, 6, 9, 12]] = 7.0  # MCP
        perp_w[[0]] = 4.0  # Thumb CMC
        self.register_buffer("axis_perp_weight", perp_w)

        for side in self.active_sides:
            hp_init = getattr(self, f"hand_pose_init_{side}")
            joints_init = hp_init.view(self.T, 15, 3)
            angles_init = joints_init.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            axes_init = joints_init / angles_init
            axis_gate = (angles_init.squeeze(-1) > 0.05).float()
            self.register_buffer(f"ref_axes_{side}", axes_init)
            self.register_buffer(f"ref_axis_gate_{side}", axis_gate)

    @staticmethod
    def _enclosure_metrics(obj_verts_np, hand_verts_np, thumb_idx, non_thumb_idx, palm_idx):
        """Compute hand-object enclosure metrics via cKDTree."""
        tree = cKDTree(obj_verts_np)
        out = {}
        for name, idx in [("thumb", thumb_idx), ("fingers", non_thumb_idx), ("palm", palm_idx)]:
            pts = hand_verts_np[idx]
            _, nn = tree.query(pts, k=1)
            out[f"{name}_dz"] = float((pts[:, 2] - obj_verts_np[nn, 2]).mean())
            out[f"{name}_dist"] = float(np.linalg.norm(pts - obj_verts_np[nn], axis=1).mean())
        return out

    def _compute_thumb_region(self, side):
        """Compute extended thumb region (tip + 4 neighbor rings) for physics masking."""
        thumb_tip = getattr(self, f"thumb_tip_verts_{side}").cpu().numpy()
        faces = self.mano_fwd[side].hand_faces.long().cpu().numpy()
        adj = [set() for _ in range(778)]
        for a, b, c in faces:
            adj[a].update([b, c])
            adj[b].update([a, c])
            adj[c].update([a, b])
        region = set(thumb_tip.tolist())
        frontier = region.copy()
        for _ in range(4):
            frontier = {n for v in frontier for n in adj[v]} - region
            region |= frontier
        return np.array(sorted(region), dtype=np.int64)

    def _compute_hidden_thumb_mask(self, side, thumb_tip, non_thumb_tip, palm_idx):
        """Detect frames where thumb is hidden behind the object."""
        hand_v, _ = self.get_hand_verts_and_joints(side)
        hand_np = hand_v.detach().cpu().numpy()
        obj_np = self.obj_verts_init.cpu().numpy()
        valid = getattr(self, f"valid_{side}").cpu().numpy()
        hidden = np.zeros(self.T, dtype=bool)
        for t in range(self.T):
            if not valid[t]:
                continue
            m = self._enclosure_metrics(obj_np[t], hand_np[t], thumb_tip, non_thumb_tip, palm_idx)
            hidden[t] = m["thumb_dz"] < -0.003 and m["fingers_dz"] < -0.010 and m["palm_dz"] < -0.010 and m["fingers_dist"] < 0.050 and m["palm_dist"] < 0.060
        return hidden

    def _init_thumb_neutralization(self):
        """Detect hidden thumb, set target/hidden masks, rewrite fingertip verts, zero thumb physics."""
        for side in self.active_sides:
            target = self.frame_indices.cpu().numpy() <= self.start_frame_max
            self.register_buffer(f"target_mask_{side}", torch.tensor(target, dtype=torch.bool, device=self.device))

            thumb_tip = getattr(self, f"thumb_tip_verts_{side}").cpu().numpy()
            tip_all = getattr(self, f"fingertip_verts_{side}").cpu().numpy()
            non_thumb_tip = np.array([i for i in tip_all if i not in set(thumb_tip.tolist())], dtype=np.int64)
            palm_idx = getattr(self, f"palm_verts_{side}").cpu().numpy()

            thumb_region = self._compute_thumb_region(side)

            raw_hidden = self._compute_hidden_thumb_mask(side, thumb_tip, non_thumb_tip, palm_idx)
            hidden = raw_hidden & target
            setattr(self, f"fingertip_verts_{side}", torch.tensor(non_thumb_tip, dtype=torch.long, device=self.device))

            self.register_buffer(f"thumb_hidden_mask_{side}", torch.tensor(hidden, dtype=torch.bool, device=self.device))
            self.register_buffer(f"thumb_region_{side}", torch.tensor(thumb_region, dtype=torch.long, device=self.device))

            vertex_w = getattr(self, f"physics_vertex_weights_{side}").clone().to(self.device)
            vertex_w[torch.tensor(thumb_region, dtype=torch.long, device=self.device)] = 0.0
            setattr(self, f"physics_vertex_weights_{side}", vertex_w)

            if hidden.any():
                print(f"    {side}: thumb hidden on {int(hidden.sum())}/{int(target.sum())} start-window frames")

    def _init_shaft_palm_regions(self):
        """Find finger shaft and palm-side grasp vertex groups per hand."""
        shaft_joints = [(1, 2), (5, 6), (9, 10), (13, 14), (17, 18)]
        n_verts_per_shaft = 25
        n_palm_verts = 60

        for side in self.active_sides:
            verts, joints = self._first_valid_fk(side)
            shaft_vert_ids = []
            for mcp_j, pip_j in shaft_joints:
                midpoint = (joints[mcp_j] + joints[pip_j]) / 2
                dists = (verts - midpoint).pow(2).sum(-1)
                _, closest = dists.topk(n_verts_per_shaft, largest=False)
                shaft_vert_ids.append(closest)
            self.register_buffer(f"shaft_verts_{side}", torch.cat(shaft_vert_ids))

            palm_c, _, _, palm_n = self._palm_frame(joints.unsqueeze(0))
            palm_dist = (verts - palm_c[0]).pow(2).sum(-1)
            _, candidates = palm_dist.topk(150, largest=False)

            hand_faces = self.mano_fwd[side].hand_faces.long().cpu()
            v0 = verts[hand_faces[:, 0]]
            v1 = verts[hand_faces[:, 1]]
            v2 = verts[hand_faces[:, 2]]
            face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
            vert_normals = torch.zeros_like(verts)
            for fi in range(3):
                vert_normals.index_add_(0, hand_faces[:, fi], face_normals)
            vert_normals = F.normalize(vert_normals, dim=-1)

            facing_idx = candidates[(vert_normals[candidates] * palm_n[0]).sum(-1) < -0.1]
            if len(facing_idx) >= n_palm_verts:
                _, top_k = palm_dist[facing_idx].topk(n_palm_verts, largest=False)
                palm_vert_ids = facing_idx[top_k]
            else:
                palm_vert_ids = candidates[:n_palm_verts]
            self.register_buffer(f"palm_verts_{side}", palm_vert_ids)

    def _init_shaft_palm_contact_targets(self):
        """Mark shaft/palm vertices that should stay close to the object surface."""
        obj_faces = self.obj_faces.to(self.device)
        with torch.no_grad():
            obj_fv = self.get_obj_verts().detach().to(self.device)[:, obj_faces].contiguous()

            for side in self.active_sides:
                valid = getattr(self, f"valid_{side}").to(self.device)
                for region, thresh in [("shaft", 0.015), ("palm", 0.02)]:
                    idx = getattr(self, f"{region}_verts_{side}").to(self.device)
                    contact_mask = torch.zeros(self.T, idx.numel(), dtype=torch.bool, device=self.device)
                    if valid.any():
                        hand_v = self.get_hand_verts(side)[:, idx].contiguous()
                        dist_sq, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(hand_v[valid], obj_fv[valid])
                        dist = torch.sqrt(dist_sq + 1e-10)
                        contact_mask[valid] = dist < thresh
                    self._set_buffer(f"{region}_contact_mask_{side}", contact_mask.cpu())

    def _init_grasp_contact_anchors(self):
        """Carry forward object-local contact anchors from the last reliable grasp frame."""
        with torch.no_grad():
            obj_v = self.get_obj_verts().detach().to(self.device)
        max_anchors = 16
        min_good_anchors = 6
        min_conf_update = 0.55

        with torch.no_grad():
            for side in self.active_sides:
                valid = getattr(self, f"valid_{side}").to(self.device)
                frame_conf = getattr(self, f"wrist_confidence_{side}").to(self.device)
                hand_v = self.get_hand_verts(side).to(self.device)

                tip_idx = getattr(self, f"fingertip_verts_{side}").cpu().numpy()
                shaft_idx = getattr(self, f"shaft_verts_{side}").cpu().numpy()
                palm_idx = getattr(self, f"palm_verts_{side}").cpu().numpy()
                tip_mask = getattr(self, f"tip_contact_mask_{side}").cpu().numpy()
                shaft_mask = getattr(self, f"shaft_contact_mask_{side}").cpu().numpy()
                palm_mask = getattr(self, f"palm_contact_mask_{side}").cpu().numpy()

                anchor_hand = torch.zeros(self.T, max_anchors, dtype=torch.long, device=self.device)
                anchor_obj = torch.zeros(self.T, max_anchors, dtype=torch.long, device=self.device)
                anchor_offset = torch.zeros(self.T, max_anchors, 3, dtype=torch.float32, device=self.device)
                anchor_weight = torch.zeros(self.T, max_anchors, dtype=torch.float32, device=self.device)
                anchor_valid = torch.zeros(self.T, max_anchors, dtype=torch.bool, device=self.device)

                current = None
                n_updates = 0
                for t in range(self.T):
                    if valid[t]:
                        region_specs = [
                            (_uniform_subset(np.unique(tip_idx[tip_mask[t]]), 6), 1.0),
                            (_uniform_subset(np.unique(shaft_idx[shaft_mask[t]]), 5), 0.75),
                            (_uniform_subset(np.unique(palm_idx[palm_mask[t]]), 5), 1.0),
                        ]
                        seen = {}
                        for verts_np, w in region_specs:
                            for vid in verts_np.tolist():
                                seen[vid] = max(seen.get(vid, 0.0), w)

                        if len(seen) >= min_good_anchors and float(frame_conf[t]) >= min_conf_update:
                            hand_ids = _uniform_subset(np.array(sorted(seen.keys()), dtype=np.int64), max_anchors)
                            hand_ids_t = torch.from_numpy(hand_ids).to(self.device)
                            reg_w_t = torch.tensor([seen[int(vid)] for vid in hand_ids], dtype=torch.float32, device=self.device)
                            dist = torch.cdist(hand_v[t : t + 1, hand_ids_t], obj_v[t : t + 1]).squeeze(0)
                            obj_ids_t = dist.argmin(dim=1)
                            offsets_t = hand_v[t, hand_ids_t] - obj_v[t, obj_ids_t]
                            current = (hand_ids_t, obj_ids_t, offsets_t, reg_w_t)
                            n_updates += 1

                    if current is None or not valid[t]:
                        continue

                    hand_ids_t, obj_ids_t, offsets_t, reg_w_t = current
                    n = hand_ids_t.numel()
                    anchor_hand[t, :n] = hand_ids_t
                    anchor_obj[t, :n] = obj_ids_t
                    anchor_offset[t, :n] = offsets_t
                    anchor_weight[t, :n] = reg_w_t
                    anchor_valid[t, :n] = True

                if n_updates > 0:
                    print(f"    {side}: persistent grasp anchors refreshed on {n_updates} frames")
                self._set_buffer(f"grasp_anchor_hand_idx_{side}", anchor_hand.cpu())
                self._set_buffer(f"grasp_anchor_obj_idx_{side}", anchor_obj.cpu())
                self._set_buffer(f"grasp_anchor_offset_{side}", anchor_offset.cpu())
                self._set_buffer(f"grasp_anchor_weight_{side}", anchor_weight.cpu())
                self._set_buffer(f"grasp_anchor_valid_{side}", anchor_valid.cpu())

    def _smooth_cam_t_init(self):
        """Detect hand-tracker instability and derive a composite confidence signal."""
        for side in self.active_sides:
            ct = getattr(self, f"cam_t_init_{side}").clone()
            valid = getattr(self, f"valid_{side}")

            valid_idx = torch.where(valid)[0]
            if len(valid_idx) < 3:
                self.register_buffer(f"wrist_confidence_{side}", torch.ones(self.T, dtype=torch.float32))
                continue

            valid_times = self.frame_indices[valid_idx]
            pair_dt = (valid_times[1:] - valid_times[:-1]).clamp(min=1.0)

            ct_valid = ct[valid_idx]
            z_vel = (ct_valid[1:, 2] - ct_valid[:-1, 2]).abs() / pair_dt
            median_vel = z_vel.median().item()
            threshold = max(median_vel * 5, 0.005)

            outlier_mask = torch.zeros(self.T, dtype=torch.bool)
            for i in range(len(z_vel)):
                if z_vel[i] > threshold:
                    outlier_mask[valid_idx[i + 1]] = True

            n_outliers = outlier_mask.sum().item()
            if n_outliers > 0:
                print(f"    {side}: smoothing {n_outliers} depth-jump frames")
                for t in torch.where(outlier_mask)[0]:
                    t = t.item()
                    prev_t = next((i for i in range(t - 1, -1, -1) if valid[i] and not outlier_mask[i]), None)
                    next_t = next((i for i in range(t + 1, self.T) if valid[i] and not outlier_mask[i]), None)
                    if prev_t is not None and next_t is not None:
                        alpha = (t - prev_t) / (next_t - prev_t)
                        ct[t] = ct[prev_t] * (1 - alpha) + ct[next_t] * alpha
                    elif prev_t is not None:
                        ct[t] = ct[prev_t]
                    elif next_t is not None:
                        ct[t] = ct[next_t]

                setattr(self, f"cam_t_init_{side}", ct)

            ct_valid = ct[valid_idx]
            hp_valid = getattr(self, f"hand_pose_init_{side}")[valid_idx]
            go_valid = getattr(self, f"global_orient_init_{side}")[valid_idx]
            scale_valid = getattr(self, f"scale_z_init_{side}")[valid_idx].clamp(min=1e-6)

            xy_vel = (ct_valid[1:, :2] - ct_valid[:-1, :2]).norm(dim=-1) / pair_dt
            z_vel = (ct_valid[1:, 2] - ct_valid[:-1, 2]).abs() / pair_dt
            pose_vel = (hp_valid[1:] - hp_valid[:-1]).view(-1, 15, 3).norm(dim=-1).mean(dim=-1) / pair_dt
            orient_vel = (go_valid[1:] - go_valid[:-1]).norm(dim=-1) / pair_dt
            scale_vel = (torch.log(scale_valid[1:]) - torch.log(scale_valid[:-1])).abs() / pair_dt

            pair_conf = torch.ones_like(pair_dt)
            for metric, min_scale in [
                (xy_vel, 0.002),
                (z_vel, 0.0015),
                (pose_vel, 0.04),
                (orient_vel, 0.02),
                (scale_vel, 0.01),
            ]:
                med = metric.median()
                mad = (metric - med).abs().median()
                onset = torch.maximum(med * 4.0, med + 4.0 * mad)
                scale = torch.maximum(metric.new_tensor(min_scale), 4.0 * mad)
                excess = torch.relu(metric - onset)
                pair_conf = torch.minimum(pair_conf, torch.exp(-((excess / scale.clamp(min=min_scale)).pow(2))))

            confidence = torch.ones(self.T, dtype=torch.float32)
            confidence[outlier_mask] = 0.2
            confidence[valid_idx[:-1]] = torch.minimum(confidence[valid_idx[:-1]], pair_conf.cpu())
            confidence[valid_idx[1:]] = torch.minimum(confidence[valid_idx[1:]], pair_conf.cpu())
            # This shared confidence is not floored: the hold/carry/anchor weights are derived from
            # it too, and a floor would strengthen those lagging priors during fast motion (the
            # object then tracks its mask worse). The evidence floor applies to the data terms
            # only, in forward().
            confidence = confidence.clamp(min=0.05, max=1.0)
            n_low = int((confidence[valid] < 0.5).sum().item())
            if n_low > 0:
                print(f"    {side}: low-confidence tracker frames {n_low}/{int(valid.sum().item())}")
            self.register_buffer(f"wrist_confidence_{side}", confidence)

    def _smooth_obj_poses_init(self):
        """Detect and smooth outliers in SAM3D object pose trajectories."""
        if self.T < 3:
            with torch.no_grad():
                self.obj_verts_init.copy_(self._compute_obj_base())
            return

        kernel = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float32)

        for name in ("obj_trans_init", "obj_scale_init"):
            traj = getattr(self, name).clone()
            vel = (traj[1:] - traj[:-1]).abs()
            for d in range(traj.shape[1]):
                med = vel[:, d].median().item()
                thresh = max(med * 5, 1e-4)
                for i in range(len(vel)):
                    if vel[i, d] > thresh:
                        t = i + 1
                        prev_t = next((j for j in range(t - 1, -1, -1) if j == 0 or vel[min(j, len(vel) - 1), d] <= thresh), None)
                        next_t = next((j for j in range(t + 1, self.T) if j == self.T - 1 or vel[min(j - 1, len(vel) - 1), d] <= thresh), None)
                        if prev_t is not None and next_t is not None and prev_t != next_t:
                            alpha = (t - prev_t) / (next_t - prev_t)
                            traj[t, d] = traj[prev_t, d] * (1 - alpha) + traj[next_t, d] * alpha
                        elif prev_t is not None:
                            traj[t, d] = traj[prev_t, d]
                        elif next_t is not None:
                            traj[t, d] = traj[next_t, d]

            kern = kernel.view(1, 1, 3).expand(traj.shape[1], -1, -1)
            s = traj.T.unsqueeze(0)
            for _ in range(3):
                s = F.conv1d(F.pad(s, (1, 1), mode="replicate"), kern, groups=traj.shape[1])
            setattr(self, name, s.squeeze(0).T.contiguous())

        quats = roma.rotvec_to_unitquat(self.obj_rotvec_init)
        angles = []
        for i in range(len(quats) - 1):
            dot = torch.dot(quats[i], quats[i + 1]).clamp(-1, 1).abs()
            angles.append(2 * torch.acos(dot))
        angles = torch.stack(angles)
        med_angle = angles.median().item()
        angle_thresh = max(med_angle * 5, 0.05)
        for i in range(len(angles)):
            if angles[i] > angle_thresh:
                t = i + 1
                prev_t = next((j for j in range(t - 1, -1, -1) if j == 0 or (j > 0 and angles[j - 1] <= angle_thresh)), None)
                next_t = next((j for j in range(t + 1, self.T) if j == self.T - 1 or angles[min(j - 1, len(angles) - 1)] <= angle_thresh), None)
                if prev_t is not None and next_t is not None and prev_t != next_t:
                    alpha = (t - prev_t) / (next_t - prev_t)
                    q0, q1 = quats[prev_t], quats[next_t]
                    if torch.dot(q0, q1) < 0:
                        q1 = -q1
                    quats[t] = F.normalize(q0 * (1 - alpha) + q1 * alpha, dim=0)
                elif prev_t is not None:
                    quats[t] = quats[prev_t]
                elif next_t is not None:
                    quats[t] = quats[next_t]
        for _ in range(3):
            smoothed = quats.clone()
            for i in range(1, len(quats) - 1):
                q_prev = quats[i - 1]
                q_curr = quats[i]
                q_next = quats[i + 1]
                if torch.dot(q_prev, q_curr) < 0:
                    q_curr = -q_curr
                if torch.dot(q_curr, q_next) < 0:
                    q_next = -q_next
                smoothed[i] = F.normalize(0.25 * q_prev + 0.5 * q_curr + 0.25 * q_next, dim=0)
            quats = smoothed
        self.obj_rotvec_init.copy_(roma.unitquat_to_rotvec(quats))
        with torch.no_grad():
            self.obj_verts_init.copy_(self._compute_obj_base())

    def _init_wrist_rays(self):
        """Unit viewing ray of each frame's initial hand centroid in the WiLoR camera (wrist_prior_mode="ray").
        A cam_t move along this ray keeps the hand centroid's pixel (fixed hand scale mode)."""
        for side in self.active_sides:
            with torch.no_grad():
                hp = getattr(self, f"hand_pose_init_{side}").to(self.device)
                go = getattr(self, f"global_orient_init_{side}").to(self.device)
                ct = getattr(self, f"cam_t_init_{side}").to(self.device)
                betas = getattr(self, f"betas_const_{side}").to(self.device).unsqueeze(0).expand(self.T, -1)
                v, _ = self.mano_fwd[side](go, hp, betas, ct)
                c = v.mean(dim=1).float().cpu()
            n = c.norm(dim=-1, keepdim=True)
            ray = torch.where(n > 1e-6, c / n.clamp(min=1e-6), torch.tensor([0.0, 0.0, 1.0]))
            self.register_buffer(f"wrist_ray_{side}", ray)

    def _correct_depth_offset(self):
        """Pre-optimization depth correction: remove each hand's median depth offset between its centroid and the object's
        (hand_depth_correction="z": a cam_t z shift; "ray": along each frame's viewing ray, pixels kept)."""
        with torch.no_grad():
            obj_cz = self.obj_verts_init.to(self.device).mean(dim=1)[:, 2].cpu()
            for side in self.active_sides:
                valid = getattr(self, f"valid_{side}").cpu()
                if not valid.any():
                    continue
                hand_v, _ = self.get_hand_verts_and_joints(side)
                hand_cz = hand_v.mean(dim=1)[:, 2].cpu()
                z_offset = (hand_cz[valid] - obj_cz[valid]).median().item()
                if abs(z_offset) < 0.005:
                    continue
                calib = getattr(self, f"calib_{side}").cpu()
                scale_z_vals = calib[valid, 0]
                median_scale = scale_z_vals.median().item()
                if median_scale < 0.01:
                    continue
                if self.hand_depth_correction == "ray":
                    # pixel-preserving: scale this side's depth calibration per frame so the hand centroid moves
                    # by exactly -z_offset along its own viewing ray (calib[:, 0] = scale_z; x/z, y/z unchanged)
                    new_z = hand_cz[valid] - z_offset
                    if float(new_z.min()) < 0.03:
                        raise OptimizerGeometryError(f"{side}: ray depth correction {z_offset * 1000:.0f}mm would put the hand at {float(new_z.min()) * 1000:.0f}mm")
                    c_new = getattr(self, f"calib_{side}").clone()
                    c_new[valid, 0] = c_new[valid, 0] * (new_z / hand_cz[valid]).to(c_new.dtype)
                    setattr(self, f"calib_{side}", c_new)
                    print(f"    {side}: depth correction {z_offset * 1000:.1f}mm along the viewing rays")
                    continue
                cam_t_correction = z_offset / max(median_scale, 0.01)
                # `_calibrate_points` is a perspective map. Moving even one valid
                # MANO center through the camera plane makes x/z and y/z explode to
                # hundreds of kilometers while remaining finite. Reject this pose
                # source before mutating parameters so process_one_segment can retry
                # the complete sam3d_c mesh+pose unit.
                _validate_corrected_hand_depth(hand_cz[valid], scale_z_vals, cam_t_correction, side)
                ct = getattr(self, f"cam_t_init_{side}")
                ct[:, 2] -= cam_t_correction
                setattr(self, f"cam_t_init_{side}", ct)
                print(f"    {side}: depth correction {z_offset * 1000:.1f}mm (cam_t_z -= {cam_t_correction:.4f})")

    def _align_contact_depth(self, frames, touch_gap=_FOLLOW_LOCAL_GAP):
        """contact_depth="ray": per keyframe on which the image shows the hand on the object (load_segment "img_gap" <=
        touch_gap), move the hand along its own viewing ray onto the object surface (contact_depth.align_hand_depth: palm
        side on the surface, not inside, on the side the visible depth order allows, smooth over time, tapered to zero
        away from contact). Applied to the depth calibration (calib[:, 0]), so pixels are unchanged and the optimizer's
        data terms (WiLoR camera) do not see it. Needed because the contact targets below are seeded from proximity: a
        hand that starts > 2 cm from the object in depth never gets any and stays off it."""
        with torch.no_grad():
            obj_v = self.get_obj_verts().detach().float().cpu().numpy()
        faces = self.obj_faces.cpu().numpy()
        for side in self.active_sides:
            valid = getattr(self, f"valid_{side}").cpu().numpy()
            gaps = [f["hands"][side].get("img_gap") if side in f["hands"] else None for f in frames]
            touch = np.array([g is not None and g <= touch_gap for g in gaps]) & valid
            if not touch.any():
                print(f"    {side}: contact depth (ray): the image never shows this hand on the object")
                continue
            with torch.no_grad():
                hand_v, _ = self.get_hand_verts_and_joints(side)
            hand_v = hand_v.cpu().numpy()
            palm = np.unique(np.concatenate([getattr(self, f"{k}_verts_{side}").cpu().numpy() for k in ("fingertip", "shaft", "palm")]))
            order = [None] * self.T
            for t in np.flatnonzero(touch):
                f = frames[t]
                if f.get("obj_mask") is not None and f.get("hand_mask") is not None and f.get("K_moge") is not None:
                    order[t] = order_evidence(hand_v[t], obj_v[t], faces, f["K_moge"], f["depth_shape"], f["obj_mask"], f["hand_mask"])
            shift, info = align_hand_depth(hand_v, touch, obj_v, faces, palm, order=order)
            setattr(self, f"contact_depth_shift_{side}", shift)
            head = (
                f"    {side}: contact depth (ray): {info['n_touch']} touching keyframes (hand-object gap before {info['gap_before_mm']:.0f} mm median), "
                f"{info['n_evidence']} with palm-side contact in reach, depth order from the masks on {info['order_frames']}"
            )
            if not np.abs(shift).max() > 0:
                print(f"{head}: nothing to align")
                continue
            c = np.linalg.norm(hand_v.mean(1), axis=1)
            k = np.where(valid & (c > 1e-6), (c + shift) / np.maximum(c, 1e-6), 1.0)
            c_new = getattr(self, f"calib_{side}").clone()
            c_new[:, 0] = c_new[:, 0] * torch.from_numpy(k).to(c_new.dtype)
            setattr(self, f"calib_{side}", c_new)
            print(f"{head}: {info['n_aligned']} aligned, |shift| median {info['median_abs_mm']:.0f} mm, max {info['max_abs_mm']:.0f} mm")

    def _init_grasp_reference(self):
        """Select grasp reference from best-enclosure non-hidden frame."""
        with torch.no_grad():
            obj_c = self.obj_verts_init.to(self.device).mean(dim=1)
            for side in self.active_sides:
                valid = getattr(self, f"valid_{side}")
                conf = getattr(self, f"wrist_confidence_{side}")
                target = getattr(self, f"target_mask_{side}")
                hidden = getattr(self, f"thumb_hidden_mask_{side}")

                hand_v, hand_j = self.get_hand_verts_and_joints(side)
                palm_c, palm_u, palm_v, palm_n = self._palm_frame(hand_j)

                tip_idx = getattr(self, f"fingertip_verts_{side}").cpu().numpy()
                palm_idx = getattr(self, f"palm_verts_{side}").cpu().numpy()
                obj_np = self.obj_verts_init.cpu().numpy()
                hand_np = hand_v.cpu().numpy()

                best_t, best_score = None, float("inf")
                for t in range(self.T):
                    if not valid[t] or conf[t] < 0.6 or target[t] or hidden[t]:
                        continue
                    tree = cKDTree(obj_np[t])
                    _, nn_f = tree.query(hand_np[t, tip_idx], k=1)
                    _, nn_p = tree.query(hand_np[t, palm_idx], k=1)
                    d_f = float(np.linalg.norm(hand_np[t, tip_idx] - obj_np[t, nn_f], axis=1).mean())
                    d_p = float(np.linalg.norm(hand_np[t, palm_idx] - obj_np[t, nn_p], axis=1).mean())
                    score = d_f + d_p
                    if score < best_score:
                        best_score, best_t = score, t

                if best_t is None:
                    for t in range(self.T):
                        if valid[t] and conf[t] >= 0.6 and not target[t]:
                            best_t = t
                            break
                if best_t is None:
                    best_t = torch.where(valid)[0][0].item()

                print(f"    {side}: grasp reference frame {int(self.frame_indices[best_t].item())} (t={best_t})")

                ref_n = palm_n[best_t]
                ref_rel = obj_c[best_t] - palm_c[best_t]
                if torch.dot(ref_rel, ref_n) < 0:
                    ref_n = -ref_n

                ref_frame = torch.stack([palm_u[best_t], palm_v[best_t], ref_n])
                ref_coords = ref_frame @ ref_rel
                self.register_buffer(f"grasp_ref_normal_{side}", ref_n.cpu())
                self.register_buffer(f"grasp_ref_coords_{side}", ref_coords.cpu())

    def _init_tip_contact_targets(self):
        """Mark fingertip vertices that should stay close to the object surface."""
        obj_faces = self.obj_faces.to(self.device)
        with torch.no_grad():
            obj_fv = self.get_obj_verts().detach().to(self.device)[:, obj_faces].contiguous()

            for side in self.active_sides:
                valid = getattr(self, f"valid_{side}").to(self.device)
                tip_idx = getattr(self, f"fingertip_verts_{side}").to(self.device)
                tip_contact = torch.zeros(self.T, tip_idx.numel(), dtype=torch.bool, device=self.device)
                if valid.any():
                    hand_v = self.get_hand_verts(side)[:, tip_idx].contiguous()
                    dist_sq, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(
                        hand_v[valid],
                        obj_fv[valid],
                    )
                    dist = torch.sqrt(dist_sq + 1e-10)
                    tip_contact[valid] = dist < self.tip_activation_threshold
                self._set_buffer(f"tip_contact_mask_{side}", tip_contact.cpu())

    def _init_obj_mask_targets(self, frames):
        """Register per-keyframe object-mask targets for 2D silhouette alignment."""
        z2, z4, z_sdf = np.zeros(2, dtype=np.float32), np.zeros(4, dtype=np.float32), np.zeros((180, 320), dtype=np.float32)
        z_sup = np.zeros(SILHOUETTE_N_DIRS, dtype=np.float32)
        mask_valid, mask_centroid, mask_bbox, mask_sdf, mask_support, intr = [], [], [], [], [], []
        for f in frames:
            mt = f.get("obj_mask_target") or {}
            mask_valid.append(bool(mt.get("valid", False)))
            mask_centroid.append(mt.get("centroid", z2))
            mask_bbox.append(mt.get("bbox", z4))
            mask_sdf.append(mt.get("sdf", z_sdf))
            mask_support.append(mt.get("support", z_sup))
            K, (H, W) = f["K_moge"], f["depth_shape"]
            nw, nh = max(W - 1, 1), max(H - 1, 1)
            intr.append(np.array([K[0, 0] / nw, K[1, 1] / nh, K[0, 2] / nw, K[1, 2] / nh], dtype=np.float32))

        self.register_buffer("obj_mask_valid", torch.tensor(mask_valid, dtype=torch.bool))
        self.register_buffer("obj_mask_centroid", torch.tensor(np.stack(mask_centroid), dtype=torch.float32))
        self.register_buffer("obj_mask_bbox", torch.tensor(np.stack(mask_bbox), dtype=torch.float32))
        self.register_buffer("obj_mask_sdf", torch.tensor(np.stack(mask_sdf)[:, None], dtype=torch.float32))
        self.register_buffer("obj_mask_support", torch.tensor(np.stack(mask_support), dtype=torch.float32))
        self.register_buffer("obj_proj_intr", torch.tensor(np.stack(intr), dtype=torch.float32))
        self.register_buffer("obj_support_dirs", torch.tensor(SILHOUETTE_DIRS, dtype=torch.float32))

        n_obj = self.obj_verts_init.shape[1]
        self.register_buffer("obj_silhouette_idx", torch.tensor(np.linspace(0, n_obj - 1, min(2048, n_obj), dtype=np.int64), dtype=torch.long))
        self.silhouette_bbox_tau = self.silhouette_support_tau = 4.0 / 1280.0

    def _init_carry_targets(self):
        """Compute carry regularization targets for low-confidence bad-enclosure frames."""
        # Disable autocast explicitly: an earlier in-process stage (e.g. SAM 3D) can
        # leave autocast set to bfloat16, which turns einsum outputs into bfloat16 and
        # breaks the `.numpy()` conversion below.
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            for side in self.active_sides:
                valid = getattr(self, f"valid_{side}").cpu().numpy()
                target = getattr(self, f"target_mask_{side}").cpu().numpy()
                conf = getattr(self, f"wrist_confidence_{side}").cpu().numpy()

                non_thumb_tip = getattr(self, f"fingertip_verts_{side}").cpu().numpy()
                shaft_idx = getattr(self, f"shaft_verts_{side}").cpu().numpy()
                palm_idx = getattr(self, f"palm_verts_{side}").cpu().numpy()
                thumb_tip = getattr(self, f"thumb_tip_verts_{side}").cpu().numpy()
                carry_idx = np.unique(np.concatenate([non_thumb_tip, shaft_idx])).astype(np.int64)

                hand_v, hand_j = self.get_hand_verts_and_joints(side)
                palm_c, palm_u, palm_v, palm_n = self._palm_frame(hand_j)
                carry_coords = self._points_in_palm(hand_v[:, carry_idx], palm_c, palm_u, palm_v, palm_n).detach().cpu().numpy()

                hand_np = hand_v.detach().cpu().numpy()
                obj_np = self.get_obj_verts().detach().cpu().numpy()

                _zero_m = {"thumb_dz": 0, "fingers_dz": 0, "palm_dz": 0, "thumb_dist": 0, "fingers_dist": 0, "palm_dist": 0}
                metrics = [self._enclosure_metrics(obj_np[t], hand_np[t], thumb_tip, non_thumb_tip, palm_idx) if valid[t] else _zero_m for t in range(self.T)]

                bad_enclosure = np.array(
                    [m["fingers_dist"] > 0.050 or m["palm_dist"] > 0.060 or m["fingers_dz"] > 0.006 or m["palm_dz"] > 0.020 for m in metrics],
                    dtype=bool,
                )

                trusted = valid & (~target) & (conf >= self.carry_conf_high) & (~bad_enclosure)
                carry_mask = valid & (~target) & (conf < self.carry_conf_low) & bad_enclosure
                frame_ids = self.frame_indices.detach().cpu().numpy()

                target_coords = np.zeros((self.T, len(carry_idx), 3), dtype=np.float32)
                carry_weight = np.zeros(self.T, dtype=np.float32)
                usable = np.zeros(self.T, dtype=bool)

                for t in np.where(carry_mask)[0]:
                    prev_idx = next((i for i in range(t - 1, -1, -1) if trusted[i]), None)
                    next_idx = next((i for i in range(t + 1, self.T) if trusted[i]), None)
                    if prev_idx is None and next_idx is None:
                        continue
                    if prev_idx is None:
                        target_coords[t] = carry_coords[next_idx]
                    elif next_idx is None:
                        target_coords[t] = carry_coords[prev_idx]
                    else:
                        alpha = float(frame_ids[t] - frame_ids[prev_idx]) / float(frame_ids[next_idx] - frame_ids[prev_idx])
                        target_coords[t] = (1.0 - alpha) * carry_coords[prev_idx] + alpha * carry_coords[next_idx]
                    severity = max(metrics[t]["fingers_dist"] / 0.05, metrics[t]["palm_dist"] / 0.06, 1.0)
                    carry_weight[t] = np.clip((1.0 - conf[t]) * severity, 0.0, 1.5)
                    usable[t] = True

                self.register_buffer(f"carry_idx_{side}", torch.tensor(carry_idx, dtype=torch.long, device=self.device))
                self.register_buffer(f"carry_mask_{side}", torch.tensor(usable, dtype=torch.bool, device=self.device))
                self.register_buffer(f"carry_target_coords_{side}", torch.tensor(target_coords, dtype=torch.float32, device=self.device))
                self.register_buffer(f"carry_weight_{side}", torch.tensor(carry_weight, dtype=torch.float32, device=self.device))

                n_carry = int(usable.sum())
                if n_carry > 0:
                    print(f"    {side}: carry regularization on {n_carry} frames")


class OptimizerGeometryMixin:
    """Geometric queries of MANOPoseOptimizer: point-to-mesh distance with the winding sign, projections, soft image statistics, carry terms."""

    def _carry_losses(self, side):
        """Compute carry finger loss for low-confidence frames."""
        carry_mask = getattr(self, f"carry_mask_{side}")
        if not self._static_branch(f"carry_{side}", carry_mask):
            return torch.zeros((), device=self.device)

        hand_v, hand_j = self.get_hand_verts_and_joints(side)
        palm_c, palm_u, palm_v, palm_n = self._palm_frame(hand_j)
        carry_idx = getattr(self, f"carry_idx_{side}")
        target_coords = getattr(self, f"carry_target_coords_{side}")
        frame_idx = getattr(self, f"_carry_idx_frames_{side}")
        carry_weight = getattr(self, f"carry_weight_{side}").index_select(0, frame_idx)

        current = self._points_in_palm(hand_v[:, carry_idx], palm_c, palm_u, palm_v, palm_n)
        err = F.smooth_l1_loss(
            current.index_select(0, frame_idx),
            target_coords.index_select(0, frame_idx),
            beta=0.01,
            reduction="none",
        )
        return (err * carry_weight[:, None, None]).mean()

    def _mask_hidden_thumb_delta(self, side, d_hp):
        """Zero thumb pose deltas on start-window hidden-thumb frames."""
        hidden = getattr(self, f"thumb_hidden_mask_{side}", None)
        if hidden is not None and self._static_branch(f"hidden_thumb_{side}", hidden):
            d_hp = d_hp.clone()
            hidden_idx = getattr(self, f"_hidden_thumb_idx_{side}", None)
            if hidden_idx is None:
                hidden_idx = torch.where(hidden.to(d_hp.device))[0]
            d_hp[hidden_idx, 36:45] = 0.0
        return d_hp

    def _init_warp_bvh(self):
        """Build a static canonical-space BVH for repeated hand-object distances."""
        self._warp_bvh_enabled = False
        self._warp_query_cache = {}
        self._warp_query_verified = set()
        if not str(self.device).startswith("cuda"):
            return
        scale = self.obj_scale_init.detach()
        scale_spread = (scale.max(dim=1).values - scale.min(dim=1).values).abs().max()
        scale_mean = scale.abs().mean().clamp(min=1e-8)
        if float(scale_spread / scale_mean) > 1e-5:
            print("    Warp BVH disabled: non-uniform object scale")
            return

        wp.init()
        canonical_centered = (self.canonical_verts - self.canonical_verts.mean(dim=0)).contiguous()
        self.register_buffer("_warp_canonical_centered", canonical_centered)
        self.register_buffer("_warp_canonical_pinv", torch.linalg.pinv(canonical_centered))
        self.register_buffer("_warp_faces_i32", self.obj_faces.to(torch.int32).contiguous())
        self._warp_mesh = wp.Mesh(
            points=wp.from_torch(self._warp_canonical_centered, dtype=wp.vec3),
            indices=wp.from_torch(self._warp_faces_i32.flatten()),
            support_winding_number=True,
        )
        self._warp_bvh_enabled = True

    def _kaolin_winding_point_to_mesh_distance(self, points, object_vertices):
        """Correct, differentiable fallback when a Warp query cannot be trusted."""

        face_vertices = object_vertices[:, self.obj_faces].contiguous()
        distance_sq, face_idx, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(points, face_vertices)
        sign = winding_sign(points.detach(), object_vertices.detach(), self.obj_faces)
        return distance_sq, face_idx, sign

    def _warp_point_to_mesh_distance(self, side, points, object_vertices):
        """BVH nearest surface with a differentiable selected-triangle distance."""
        build_error = None
        with torch.no_grad():
            object_detached = object_vertices.detach()
            centers = object_detached.mean(dim=1, keepdim=True)
            centered = object_detached - centers
            transforms = torch.einsum("iv,tvj->tij", self._warp_canonical_pinv, centered)
            inverse = torch.linalg.inv(transforms)
            local = torch.bmm(points.detach() - centers, inverse).reshape(-1, 3).contiguous()
            count = local.shape[0]
            cached = self._warp_query_cache.get(side)
            if cached is None or cached[0].numel() != count:
                cached = (
                    torch.full((count,), -1, dtype=torch.int32, device=points.device),
                    torch.zeros(count, dtype=torch.float32, device=points.device),
                    torch.zeros(count, dtype=torch.float32, device=points.device),
                    torch.ones(count, dtype=torch.float32, device=points.device),
                )
                self._warp_query_cache[side] = cached
            face_flat, bary_u, bary_v, sign_flat = cached
            # Cached outputs must never retain a prior launch's valid face when
            # the current query produces no result or compilation aborts.
            face_flat.fill_(-1)
            bary_u.zero_()
            bary_v.zero_()
            sign_flat.fill_(1.0)
            warp_device = wp.device_from_torch(points.device)
            warp_stream = wp.stream_from_torch(torch.cuda.current_stream(points.device))
            try:
                with wp.ScopedStream(warp_stream):
                    wp.launch(
                        _warp_query_closest_winding,
                        dim=count,
                        inputs=[self._warp_mesh.id, local],
                        outputs=[face_flat, bary_u, bary_v, sign_flat],
                        device=warp_device,
                    )
            except Exception as error:
                if not _is_warp_kernel_build_error(error):
                    raise
                build_error = error

        if build_error is not None:
            self._warp_bvh_enabled = False
            print(f"    Warp BVH disabled after kernel build failure; using Kaolin distance fallback: {build_error}")
            return self._kaolin_winding_point_to_mesh_distance(points, object_vertices)

        face_idx = face_flat.to(torch.long).reshape(points.shape[:2])
        valid_face = (face_idx >= 0) & (face_idx < self.obj_faces.shape[0])
        if side not in self._warp_query_verified:
            invalid_count = int((~valid_face).sum().item())
            if invalid_count:
                self._warp_bvh_enabled = False
                print(f"    Warp BVH disabled after {invalid_count} invalid closest-face queries; using Kaolin fallback")
                return self._kaolin_winding_point_to_mesh_distance(points, object_vertices)
            self._warp_query_verified.add(side)
        safe_face_idx = face_idx.clamp(0, self.obj_faces.shape[0] - 1)
        query_faces = self.obj_faces[safe_face_idx]
        frame_idx = torch.arange(points.shape[0], device=points.device)[:, None, None]
        triangles = object_vertices[frame_idx, query_faces]
        barycentric = torch.stack([bary_u, bary_v, 1.0 - bary_u - bary_v], dim=-1)
        barycentric = barycentric.reshape(*points.shape[:2], 3)
        closest = (triangles * barycentric[..., None]).sum(dim=2)
        distance_sq = (points - closest).pow(2).sum(dim=-1)
        # No-query rows are impossible for finite points against a healthy BVH,
        # but masking them keeps a later anomalous query from reaching CUDA with
        # an out-of-range index.  One meter squared is safely outside contact.
        distance_sq = torch.where(valid_face, distance_sq, torch.ones_like(distance_sq))
        sign = torch.where(
            valid_face & (sign_flat.reshape(points.shape[:2]) < 0),
            -torch.ones_like(distance_sq),
            torch.ones_like(distance_sq),
        )
        return distance_sq, safe_face_idx, sign

    def _project_points_normalized(self, pts):
        """Project camera-space points to normalized image coordinates [0, 1]."""
        intr = self.obj_proj_intr.to(pts.device)
        fx, fy, cx, cy = [intr[:, i : i + 1] for i in range(4)]
        x, y, z = pts.unbind(-1)
        z_safe = z.clamp(min=1e-6)
        uv = torch.stack(
            [
                fx * (x / z_safe) + cx,
                fy * (y / z_safe) + cy,
            ],
            dim=-1,
        )
        valid = z > 1e-6
        return uv, valid

    def _soft_bbox(self, uv, valid):
        """Differentiable bbox approximation from projected points."""
        tau = self.silhouette_bbox_tau
        neg_inf = torch.full_like(uv[..., 0], -1e6)
        x, y = uv.unbind(-1)
        x_max = tau * torch.logsumexp(torch.where(valid, x / tau, neg_inf), dim=1)
        y_max = tau * torch.logsumexp(torch.where(valid, y / tau, neg_inf), dim=1)
        x_min = -tau * torch.logsumexp(torch.where(valid, -x / tau, neg_inf), dim=1)
        y_min = -tau * torch.logsumexp(torch.where(valid, -y / tau, neg_inf), dim=1)
        return torch.stack([x_min, y_min, x_max, y_max], dim=-1)

    def _soft_support(self, uv, valid):
        """Differentiable support function over multiple silhouette directions."""
        tau = self.silhouette_support_tau
        dirs = self.obj_support_dirs.to(uv.device)
        proj = torch.einsum("bnc,kc->bnk", uv, dirs)
        neg_inf = torch.full_like(proj, -1e6)
        return tau * torch.logsumexp(torch.where(valid.unsqueeze(-1), proj / tau, neg_inf), dim=1)

    def _weighted_mean(self, values, weights=None):
        if weights is None:
            return values.mean()
        weights = weights.to(values.device, dtype=values.dtype)
        return (values * weights).sum() / weights.sum().clamp(min=1e-6)

    def _weighted_vertex_mean(self, per_vertex, vertex_weights, frame_weights=None):
        weights = vertex_weights.to(per_vertex.device).unsqueeze(0).expand_as(per_vertex)
        if frame_weights is not None:
            weights = weights * frame_weights.to(per_vertex.device, dtype=per_vertex.dtype).unsqueeze(-1)
        return (per_vertex * weights).sum() / weights.sum().clamp(min=1.0)

    def _img_gate(self, side, valid_idx):
        """(n_valid,) image contact weights, or None when the gate is off."""
        if not self.image_contact_gate:
            return None
        w = getattr(self, f"img_contact_w_{side}")
        return w.index_select(0, valid_idx.to(w.device))

    def _region_pen_and_surface(self, losses, region, side, signed_dist, frame_conf, margin, surface_conf=None):
        """Pen + deep_pen + surface losses for a hand-vertex region (shaft/palm). surface_conf: frame weights for the
        surface (attracting) term only; None = frame_conf."""
        idx = getattr(self, f"{region}_verts_{side}").to(signed_dist.device)
        region_signed = signed_dist[:, idx]
        pen = torch.relu(margin - region_signed)
        losses[f"{region}_pen_{side}"] = self._weighted_mean(F.smooth_l1_loss(pen, torch.zeros_like(pen), beta=margin, reduction="none"), frame_conf[:, None])
        losses[f"deep_{region}_pen_{side}"] = self._weighted_mean(torch.relu(pen - margin).pow(2) / margin, frame_conf[:, None])
        valid_idx = getattr(self, f"_valid_idx_{side}")
        contact = getattr(self, f"{region}_contact_mask_{side}").index_select(0, valid_idx)
        if self._static_branch(f"{region}_contact_{side}", contact):
            surface = F.smooth_l1_loss(region_signed, torch.full_like(region_signed, margin), beta=margin, reduction="none")
            losses[f"{region}_surface_{side}"] = self._weighted_mean(surface, contact.float() * (frame_conf if surface_conf is None else surface_conf)[:, None])
