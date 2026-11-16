"""MANOPoseOptimizer: the parameters of both hands and the object over a segment, and the loss terms."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import roma
import kaolin
from gtf.extraction.optimizer import hand_evidence
from gtf.extraction.optimizer.optimizer_common import MANOForward, OptimizerGeometryError, TemporalBasis, _robust_segment_beta, _sanitize_object_faces, winding_sign
from gtf.extraction.optimizer.optimizer_parts import OptimizerGeometryMixin, OptimizerSetupMixin


class MANOPoseOptimizer(OptimizerSetupMixin, OptimizerGeometryMixin, nn.Module):
    """Articulated MANO + object pose optimizer.

    Optimizes per-frame:
      - Hand: global_orient (3), cam_t (3), hand_pose (45) = 51 DOF
      - Object: rotation (3) + translation (3) = 6 DOF
    """

    def __init__(
        self,
        frames,
        obj_faces,
        canonical_verts_raw,
        contact_threshold=0.008,
        min_separation=0.002,
        deep_pen_threshold=0.005,
        max_wrist_rot=0.08,
        max_wrist_trans=0.05,
        thumb_vertex_weight=0.25,
        grasp_side_margin=0.005,
        start_frame_max=59,
        carry_conf_low=0.5,
        carry_conf_high=0.8,
        conf_floor=0.3,
        decouple_depth=False,
        hand_depth_correction="z",
        wrist_prior_mode="camera",
        image_contact_gate=False,
        image_carry_gate=False,
        contact_depth="off",
        device="cuda",
    ):
        super().__init__()
        self.device = device
        # contact_depth="ray" (upgraded chain): before the contact targets are seeded, slide each hand along its own
        # viewing ray onto the object on the keyframes where the image shows it on the object (_align_contact_depth).
        if contact_depth not in ("off", "ray"):
            raise ValueError(f"bad contact_depth={contact_depth!r}")
        self.contact_depth = contact_depth
        # Image-faithful hand placement (upgraded chain): hand_depth_correction="ray" moves the pre-optimization
        # hand->object depth correction along each frame's viewing ray (the default z-only cam_t shift slides
        # the projection radially); wrist_prior_mode="ray" makes the wrist-translation prior cheap along the
        # ray through the hand and expensive sideways (the default xy/z split is in the WiLoR camera, where z
        # moves are radial image slides).
        if hand_depth_correction not in ("z", "ray") or wrist_prior_mode not in ("camera", "ray"):
            raise ValueError(f"bad hand_depth_correction={hand_depth_correction!r} / wrist_prior_mode={wrist_prior_mode!r}")
        self.hand_depth_correction = hand_depth_correction
        self.wrist_prior_mode = wrist_prior_mode
        # image_contact_gate (upgraded chain): the attracting contact terms (grasp hold, grasp anchors,
        # tip/shaft/palm surface, tip closure, contact, attract) act on a hand-keyframe only as far as the image shows
        # the hand touching the object mask (load_segment "img_gap" -> hand_evidence.contact_weight); penetration terms unchanged.
        self.image_contact_gate = bool(image_contact_gate)
        # image_carry_gate (upgraded chain): the hand-object hold / grasp-hold terms act on a keyframe pair only as far
        # as the image shows the hand moving with the object (hand_evidence.comotion_weight).
        self.image_carry_gate = bool(image_carry_gate)
        self.T = len(frames)
        # Recorded in optimized.npz: hands are placed rigidly at canonical MANO size (get_hand_verts_and_joints), and the
        # penetration sign is the winding number.
        self.hand_scale_mode = "fixed"
        self.pen_sign_mode = "winding"
        # One-directional contact decoupling (scale_anchor="contact_follow"): the object was
        # placed at the hand's existing depth in load_segment, so the hand←object depth drag
        # (_correct_depth_offset) must be skipped — re-aligning the hand to the object would
        # undo the one-directionality and drag the hand to the object's depth error.
        self.decouple_depth = decouple_depth
        self.conf_floor = conf_floor
        self.contact_threshold = contact_threshold
        self.min_separation = min_separation
        self.deep_pen_threshold = deep_pen_threshold
        self.thumb_vertex_weight = thumb_vertex_weight
        self.grasp_side_margin = grasp_side_margin
        self.carry_conf_low = carry_conf_low
        self.carry_conf_high = carry_conf_high
        self.start_frame_max = start_frame_max
        self.max_wrist_rot = max_wrist_rot
        self.max_wrist_trans = max_wrist_trans
        self.tip_contact_threshold = min(contact_threshold, 0.005)
        self.tip_surface_margin = 0.0015
        self.tip_activation_threshold = max(self.tip_contact_threshold * 2.0, 0.01)

        self.active_sides = sorted({s for f in frames for s in f["hands"]})
        self.mano_fwd = nn.ModuleDict({side: MANOForward(side, device) for side in self.active_sides})

        obj_verts_list = [f["obj_verts"] for f in frames]
        n_obj = obj_verts_list[0].shape[0]
        canonical_for_faces = np.asarray(canonical_verts_raw, dtype=np.float64)
        if canonical_for_faces.shape != (n_obj, 3):
            raise ValueError(f"canonical object vertices do not match the tracked vertex roster: canonical={canonical_for_faces.shape}, tracked={(n_obj, 3)}")
        obj_faces, removed_faces = _sanitize_object_faces(canonical_for_faces, obj_faces)
        if removed_faces:
            print(f"    Removed {removed_faces} degenerate object face(s) before distance queries")
        stacked_obj_verts = np.stack(obj_verts_list)
        if not np.isfinite(stacked_obj_verts).all():
            raise OptimizerGeometryError("tracked object vertices contain non-finite coordinates")
        self.register_buffer(
            "obj_verts_init",
            torch.tensor(stacked_obj_verts, dtype=torch.float32),
        )
        self.register_buffer("obj_faces", torch.tensor(obj_faces, dtype=torch.long))

        self.register_buffer("canonical_verts", torch.tensor(canonical_verts_raw, dtype=torch.float32))
        self.register_buffer("obj_flip", torch.diag(torch.tensor([-1.0, -1.0, 1.0])))

        obj_R_list = np.stack([f["obj_R_sam3d"] for f in frames])
        obj_t_list = np.stack([f["obj_t_sam3d"] for f in frames])
        obj_s_list = np.stack([f["obj_s_sam3d"] for f in frames])
        obj_R_t = torch.tensor(obj_R_list, dtype=torch.float32)
        obj_quat = roma.rotmat_to_unitquat(obj_R_t)
        for i in range(1, len(obj_quat)):
            if torch.dot(obj_quat[i], obj_quat[i - 1]) < 0:
                obj_quat[i] = -obj_quat[i]
        obj_rotvec = roma.unitquat_to_rotvec(obj_quat).numpy()
        self.register_buffer("obj_rotvec_init", torch.tensor(obj_rotvec, dtype=torch.float32))
        self.register_buffer("obj_trans_init", torch.tensor(obj_t_list, dtype=torch.float32))
        self.register_buffer("obj_scale_init", torch.tensor(obj_s_list, dtype=torch.float32))

        obj_n_basis = max(2, self.T // 8)
        self.obj_rot_basis = TemporalBasis(self.T, 3, n_basis=obj_n_basis, device=device)
        self.obj_trans_basis = TemporalBasis(self.T, 3, n_basis=obj_n_basis, device=device)
        # Global object scale refinement: the SAM 3D scale is a noisy single-frame estimate, and an
        # oversized mesh causes grasp penetration that no pen loss can fix. exp(0)=1 at init; clamped
        # to ±0.18 (~±20%) in _run_phase.
        self.obj_log_scale = nn.Parameter(torch.zeros(1, device=device))

        _zero = {"hand_pose": 45, "global_orient": 3, "cam_t": 3}
        for side in self.active_sides:
            init = {k: [] for k in _zero}
            valid = []
            calib = np.zeros((self.T, 9), dtype=np.float32)
            scale_z = np.ones(self.T, dtype=np.float32)
            betas_valid = []

            for t, f in enumerate(frames):
                if side in f["hands"]:
                    hd = f["hands"][side]
                    for k in _zero:
                        init[k].append(hd[k])
                    betas_valid.append(hd["betas"])
                    valid.append(True)
                    scale_z[t] = float(hd["scale_z"])
                    sz, K_h, isz, K_m = hd["scale_z"], hd["intr"], hd["img_size"], hd["K_moge"]
                    H_m, W_m = hd["depth_shape"]
                    W_h, H_h = float(isz[0]), float(isz[1])
                    calib[t] = [
                        sz,
                        K_h[0, 0] * W_m / W_h,
                        K_h[1, 1] * H_m / H_h,
                        K_h[0, 2] * W_m / W_h,
                        K_h[1, 2] * H_m / H_h,
                        K_m[0, 0],
                        K_m[1, 1],
                        K_m[0, 2],
                        K_m[1, 2],
                    ]
                else:
                    for k, dim in _zero.items():
                        init[k].append(np.zeros(dim, dtype=np.float32))
                    valid.append(False)

            for k in _zero:
                self.register_buffer(f"{k}_init_{side}", torch.tensor(np.stack(init[k]), dtype=torch.float32))
            self.register_buffer(f"betas_const_{side}", torch.tensor(_robust_segment_beta(betas_valid), dtype=torch.float32))
            self.register_buffer(f"valid_{side}", torch.tensor(valid, dtype=torch.bool))
            self.register_buffer(f"calib_{side}", torch.from_numpy(calib))
            self.register_buffer(f"scale_z_init_{side}", torch.from_numpy(scale_z))

            setattr(self, f"d_hand_pose_basis_{side}", TemporalBasis(self.T, 45, n_basis=max(2, self.T // 4), device=device))
            setattr(self, f"d_global_orient_{side}", nn.Parameter(torch.zeros(self.T, 3, device=device)))
            setattr(self, f"d_cam_t_{side}", nn.Parameter(torch.zeros(self.T, 3, device=device)))

        fi = torch.tensor([f["frame_index"] for f in frames], dtype=torch.float32)
        self.register_buffer("frame_indices", fi)
        self.register_buffer("frame_gaps", torch.diff(fi))

        for side in self.active_sides:
            sz = getattr(self, f"scale_z_init_{side}")[getattr(self, f"valid_{side}")]
            if sz.numel():
                print(f"    {side}: rigid hand placement (canonical MANO size; depth scale_z med {sz.median():.2f}, range {sz.min():.2f}-{sz.max():.2f})")

        self._init_fingertip_indices()
        self._init_shaft_palm_regions()
        self._smooth_cam_t_init()
        if self.image_carry_gate:
            oc = [f.get("obj_img_c") for f in frames]
            for side in self.active_sides:
                hc = [f["hands"][side].get("img_hand_c") if side in f["hands"] else None for f in frames]
                w = hand_evidence.comotion_weight(hc, oc)
                self.register_buffer(f"img_comove_w_{side}", torch.tensor(w, dtype=torch.float32))
                print(f"    {side}: image carry gate, {int((w < 0.5).sum())}/{len(w)} keyframe pairs released")
        if self.image_contact_gate:
            for side in self.active_sides:
                gaps = [f["hands"][side].get("img_gap") if side in f["hands"] else None for f in frames]
                w = hand_evidence.contact_weight(gaps)
                self.register_buffer(f"img_contact_w_{side}", torch.tensor(w, dtype=torch.float32))
                v = getattr(self, f"valid_{side}").cpu().numpy()
                print(
                    f"    {side}: image contact gate, mean weight {w[v].mean() if v.any() else float('nan'):.2f} ({int((w[v] < 0.5).sum())}/{int(v.sum())} keyframes released)"
                )
        self._init_physics_vertex_weights()
        self._init_joint_limits()
        self._init_axis_regularization()
        self._smooth_obj_poses_init()
        if not self.decouple_depth:
            self._correct_depth_offset()
        if self.contact_depth == "ray":
            self._align_contact_depth(frames)
        if self.wrist_prior_mode == "ray":
            self._init_wrist_rays()
        self._init_thumb_neutralization()
        self._init_grasp_reference()
        self._init_tip_contact_targets()
        self._init_shaft_palm_contact_targets()
        self._init_grasp_contact_anchors()
        self._update_grasp_active()
        self._init_obj_mask_targets(frames)
        self._init_carry_targets()

        self.to(device)
        self._init_warp_bvh()
        self._refresh_static_branch_flags()

    def _set_buffer(self, name, tensor):
        """Register a buffer, or overwrite it in place if it already exists (for refresh)."""
        if name in self._buffers:
            self._buffers[name] = tensor.to(self._buffers[name].device)
        else:
            self.register_buffer(name, tensor)

    def refresh_contact_targets(self):
        """Recompute contact masks + grasp anchors from the current optimized geometry.

        At init these are derived from the raw (noisy) poses; after a rigid-alignment
        phase the geometry has moved, so the surface-adherence and anchor losses would
        otherwise keep pulling toward stale init-time contact locations.
        """
        self._init_tip_contact_targets()
        self._init_shaft_palm_contact_targets()
        self._init_grasp_contact_anchors()
        self._update_grasp_active()
        self._refresh_static_branch_flags()

    def _refresh_static_branch_flags(self):
        """Cache CUDA-mask branches that only change on a contact refresh."""
        flags = {"sil_valid": bool(self.obj_mask_valid.any().item())}
        for side in self.active_sides:
            valid = getattr(self, f"valid_{side}")
            both = valid[:-1] & valid[1:] if self.T > 1 else valid[:0]
            triple = valid[:-2] & valid[1:-1] & valid[2:] if self.T > 2 else valid[:0]
            grasp_active = getattr(self, f"grasp_active_{side}")
            hold_rot = both & grasp_active[:-1] & grasp_active[1:] if self.T > 1 else valid[:0]
            self._set_buffer(f"_valid_idx_{side}", torch.where(valid)[0])
            self._set_buffer(f"_both_idx_{side}", torch.where(both)[0])
            self._set_buffer(f"_triple_idx_{side}", torch.where(triple)[0])
            self._set_buffer(f"_hold_rot_idx_{side}", torch.where(hold_rot)[0])
            self._set_buffer(
                f"_hidden_thumb_idx_{side}",
                torch.where(getattr(self, f"thumb_hidden_mask_{side}"))[0],
            )
            self._set_buffer(
                f"_carry_idx_frames_{side}",
                torch.where(getattr(self, f"carry_mask_{side}"))[0],
            )
            flags[f"valid_{side}"] = bool(valid.any().item())
            flags[f"both_{side}"] = bool((valid[:-1] & valid[1:]).any().item()) if self.T > 1 else False
            flags[f"triple_{side}"] = bool((valid[:-2] & valid[1:-1] & valid[2:]).any().item()) if self.T > 2 else False
            flags[f"anchor_{side}"] = bool(getattr(self, f"grasp_anchor_valid_{side}")[valid].any().item())
            flags[f"tip_contact_{side}"] = bool(getattr(self, f"tip_contact_mask_{side}")[valid].any().item())
            flags[f"grasp_active_{side}"] = bool(grasp_active[valid].any().item())
            flags[f"hidden_thumb_{side}"] = bool(getattr(self, f"thumb_hidden_mask_{side}").any().item())
            flags[f"carry_{side}"] = bool(getattr(self, f"carry_mask_{side}").any().item())
            flags[f"hold_rot_{side}"] = bool((hold_rot).any().item()) if self.T > 1 else False
            for region in ("shaft", "palm"):
                flags[f"{region}_contact_{side}"] = bool(getattr(self, f"{region}_contact_mask_{side}")[valid].any().item())
        self._static_branch_flags = flags

    def _static_branch(self, key, tensor):
        flags = getattr(self, "_static_branch_flags", None)
        if flags is not None and key in flags:
            return flags[key]
        return bool(tensor.any().item())

    def _update_grasp_active(self):
        """Per-frame grasp indicator: enough hand vertices in contact with the object."""
        for side in self.active_sides:
            valid = getattr(self, f"valid_{side}")
            n_contact = (
                getattr(self, f"tip_contact_mask_{side}").sum(-1)
                + getattr(self, f"shaft_contact_mask_{side}").sum(-1).to(getattr(self, f"tip_contact_mask_{side}").device)
                + getattr(self, f"palm_contact_mask_{side}").sum(-1).to(getattr(self, f"tip_contact_mask_{side}").device)
            )
            active = (n_contact >= 4) & valid.to(n_contact.device)
            self._set_buffer(f"grasp_active_{side}", active)
            n_act = int(active.sum())
            if n_act > 0:
                print(f"    {side}: grasp active on {n_act}/{int(valid.sum())} frames")

    def _compute_obj_base(self):
        """Canonical mesh → base object verts via smoothed SAM3D transforms."""
        R_base = roma.rotvec_to_rotmat(self.obj_rotvec_init)  # (T, 3, 3)
        scale = self.obj_scale_init * torch.exp(self.obj_log_scale).to(self.obj_scale_init.device)
        scaled = self.canonical_verts.unsqueeze(0) * scale.unsqueeze(1)  # (T, V, 3)
        rotated = torch.bmm(scaled, R_base)  # (T, V, 3) — row-vector multiplication
        base = (rotated + self.obj_trans_init.unsqueeze(1)) * self.obj_flip.diagonal()
        return base

    def _calibrate_points(self, side, pts_cam):
        """Project hand-stage (WiLoR) camera-space points into MoGe camera space (calib: scale_z + both intrinsics)."""
        c = getattr(self, f"calib_{side}").to(pts_cam.device)
        s, fx_hn, fy_hn, cx_hn, cy_hn, fx_m, fy_m, cx_m, cy_m = [c[:, i : i + 1] for i in range(9)]
        frame_valid = getattr(self, f"valid_{side}").to(pts_cam.device).unsqueeze(-1)

        one = torch.ones_like(s)
        zero = torch.zeros_like(cx_hn)
        s = torch.where(frame_valid, s, one)
        fx_hn = torch.where(frame_valid, fx_hn, one)
        fy_hn = torch.where(frame_valid, fy_hn, one)
        cx_hn = torch.where(frame_valid, cx_hn, zero)
        cy_hn = torch.where(frame_valid, cy_hn, zero)
        fx_m = torch.where(frame_valid, fx_m, one).clamp(min=1e-6)
        fy_m = torch.where(frame_valid, fy_m, one).clamp(min=1e-6)
        cx_m = torch.where(frame_valid, cx_m, zero)
        cy_m = torch.where(frame_valid, cy_m, zero)

        x_h, y_h, z_h = pts_cam.unbind(-1)
        z_safe = z_h.clamp(min=1e-8)
        z_m = z_h * s
        x_m = z_m * (fx_hn * x_h / z_safe + cx_hn - cx_m) / fx_m
        y_m = z_m * (fy_hn * y_h / z_safe + cy_hn - cy_m) / fy_m
        pts_m = torch.stack([x_m, y_m, z_m], dim=-1)
        return torch.where(frame_valid.unsqueeze(1), pts_m, torch.zeros_like(pts_m))

    def get_hand_verts_and_joints(self, side):
        """MANO FK + calibration → verts/joints in MoGe camera space.

        Rigid placement: the projective remap (_calibrate_points: z_m = z_h * scale_z, pixels
        preserved) is applied to the hand centroid only and the whole FK hand is translated
        there. The remap is a per-frame linear map whose local size distortion is
        scale_z * f_hn / f_m in xy and scale_z in depth; applied per point it would put the
        depth error into the hand's size (and squash it along depth), which breaks contact in
        the canonical export (the export re-runs FK at canonical scale). Translating keeps the
        hand at canonical MANO size and exactly rigid, with the centroid where the per-point
        remap puts it (the linear map commutes with the mean).
        """
        d_hp = self._mask_hidden_thumb_delta(side, getattr(self, f"d_hand_pose_basis_{side}")())
        hp = getattr(self, f"hand_pose_init_{side}").to(self.device) + d_hp
        go = getattr(self, f"global_orient_init_{side}").to(self.device) + getattr(self, f"d_global_orient_{side}")
        ct = getattr(self, f"cam_t_init_{side}").to(self.device) + getattr(self, f"d_cam_t_{side}")
        betas = getattr(self, f"betas_const_{side}").to(self.device).unsqueeze(0).expand(self.T, -1)
        verts_cam, joints_cam = self.mano_fwd[side](go, hp, betas, ct)
        anchor = verts_cam.mean(dim=1, keepdim=True)  # (T, 1, 3)
        delta = self._calibrate_points(side, anchor) - anchor
        frame_valid = getattr(self, f"valid_{side}").to(verts_cam.device).view(-1, 1, 1)
        verts = torch.where(frame_valid, verts_cam + delta, torch.zeros_like(verts_cam))
        joints = torch.where(frame_valid, joints_cam + delta, torch.zeros_like(joints_cam))
        # Defensive fp32 cast — downstream .numpy() conversions reject bfloat16/half.
        return verts.float(), joints.float()

    def _palm_frame(self, joints):
        """Return palm center and an orthonormal palm frame (center, u, v, normal)."""
        w, idx, mid, pnk = joints[:, 0], joints[:, 5], joints[:, 9], joints[:, 17]
        center = torch.stack([w, idx, mid, pnk], dim=1).mean(dim=1)
        u = F.normalize(idx - pnk, dim=-1, eps=1e-6)
        v_raw = mid - w
        v = F.normalize(v_raw - (v_raw * u).sum(-1, keepdim=True) * u, dim=-1, eps=1e-6)
        return center, u, v, F.normalize(torch.cross(u, v, dim=-1), dim=-1, eps=1e-6)

    def _points_in_palm(self, pts, palm_c, palm_u, palm_v, palm_n):
        """Transform (T, N, 3) points into palm-frame (u, v, n) coordinates."""
        rel = pts - palm_c.unsqueeze(1)
        return torch.einsum("tnd,tfd->tnf", rel, torch.stack([palm_u, palm_v, palm_n], dim=1))

    def get_hand_verts(self, side):
        """MANO FK + calibration → (T, 778, 3) in MoGe camera space."""
        verts, _ = self.get_hand_verts_and_joints(side)
        return verts

    def get_obj_verts(self):
        """Smoothed base + DCT corrections → final object verts."""
        rot_corr = self.obj_rot_basis()
        trans_corr = self.obj_trans_basis()
        # Buffers live on CPU until the end of __init__ while the DCT bases are already
        # on the target device; align so init-time contact-target computation works.
        base = self._compute_obj_base().to(rot_corr.device)
        R_corr = roma.rotvec_to_rotmat(rot_corr)
        centroid = base.mean(dim=1, keepdim=True)
        centered = base - centroid
        rotated = torch.einsum("tij,tvj->tvi", R_corr, centered)
        return rotated + centroid + trans_corr.unsqueeze(1)

    def forward(self):
        """Compute all losses."""
        losses = {}
        include_object_losses = getattr(self, "_include_object_losses", True)
        loss_sides = getattr(self, "_loss_sides", None)
        obj_v = self.get_obj_verts()
        obj_c = obj_v.mean(dim=1)
        # Effective canonical→camera object rotation (base SAM3D chain + DCT correction),
        # used by the in-hand rotation hold. Any consistent convention works (the loss is
        # conjugation-invariant), but this is the exact one from _compute_obj_base/get_obj_verts.
        R_base_init = roma.rotvec_to_rotmat(self.obj_rotvec_init)
        R_obj_eff = torch.matmul(roma.rotvec_to_rotmat(self.obj_rot_basis()), torch.matmul(self.obj_flip.unsqueeze(0), R_base_init.transpose(1, 2)))

        if include_object_losses:
            losses["data_obj_rot"] = self.obj_rot_basis().pow(2).mean()
            losses["data_obj_trans"] = self.obj_trans_basis().pow(2).mean()
            losses["data_obj_scale"] = self.obj_log_scale.pow(2).sum()

        sil_valid = self.obj_mask_valid
        if include_object_losses and self._static_branch("sil_valid", sil_valid):
            obj_uv, obj_uv_valid = self._project_points_normalized(obj_v[:, self.obj_silhouette_idx])
            sil_frames = sil_valid & obj_uv_valid.any(dim=1)
            if sil_frames.any():
                uv_sel = obj_uv[sil_frames]
                uv_valid_sel = obj_uv_valid[sil_frames]

                grid = torch.where(uv_valid_sel.unsqueeze(-1), uv_sel.mul(2.0).sub(1.0), torch.zeros_like(uv_sel))
                sdf = (
                    F.grid_sample(
                        self.obj_mask_sdf[sil_frames].to(obj_v.device),
                        grid.unsqueeze(1),
                        mode="bilinear",
                        padding_mode="border",
                        align_corners=True,
                    )
                    .squeeze(1)
                    .squeeze(1)
                )

                outside = torch.relu(sdf)
                losses["sil_obj_sdf"] = (
                    F.smooth_l1_loss(outside, torch.zeros_like(outside), beta=0.01, reduction="none") * uv_valid_sel.float()
                ).sum() / uv_valid_sel.float().sum().clamp(min=1.0)

                uv_center = (uv_sel * uv_valid_sel.unsqueeze(-1).float()).sum(dim=1) / uv_valid_sel.float().sum(dim=1, keepdim=True).clamp(min=1.0)
                losses["sil_obj_centroid"] = (uv_center - self.obj_mask_centroid[sil_frames].to(obj_v.device)).pow(2).mean()

                pred_bbox = self._soft_bbox(uv_sel, uv_valid_sel)
                losses["sil_obj_bbox"] = (pred_bbox - self.obj_mask_bbox[sil_frames].to(obj_v.device)).pow(2).mean()

                pred_support = self._soft_support(uv_sel, uv_valid_sel)
                losses["sil_obj_support"] = (pred_support - self.obj_mask_support[sil_frames].to(obj_v.device)).pow(2).mean()

        gap_w = 1.0 / self.frame_gaps.clamp(min=1.0) if self.T > 1 else None

        if include_object_losses and self.T > 1:
            obj_vel = obj_c[1:] - obj_c[:-1]
            losses["smooth_obj_vel"] = (obj_vel.pow(2).sum(-1) * gap_w).mean()
            if self.T > 2:
                losses["smooth_obj_acc"] = (obj_vel[1:] - obj_vel[:-1]).pow(2).mean()
            # Penalize the effective object angular velocity (base chain + correction), not the variation of the
            # correction alone: the correction must be free to counter-rotate high-frequency SAM 3D rotation jitter.
            R_rel = torch.matmul(R_obj_eff[:-1].transpose(1, 2), R_obj_eff[1:])
            losses["smooth_obj_rot"] = (roma.rotmat_to_rotvec(R_rel).pow(2).sum(-1) * gap_w).mean()

        active_loss_sides = self.active_sides if loss_sides is None else loss_sides
        for side in active_loss_sides:
            valid = getattr(self, f"valid_{side}")
            valid_idx = getattr(self, f"_valid_idx_{side}")
            d_hp = self._mask_hidden_thumb_delta(side, getattr(self, f"d_hand_pose_basis_{side}")())
            d_go = getattr(self, f"d_global_orient_{side}")
            d_ct = getattr(self, f"d_cam_t_{side}")
            wrist_conf = getattr(self, f"wrist_confidence_{side}").to(d_ct.device)
            hp_init = getattr(self, f"hand_pose_init_{side}").to(d_hp.device)
            hp = hp_init + d_hp
            joint_angles = hp.view(self.T, 15, 3).norm(dim=-1)
            joint_angles_init = hp_init.view(self.T, 15, 3).norm(dim=-1)
            if self._static_branch(f"valid_{side}", valid):
                conf_valid = wrist_conf.index_select(0, valid_idx)
                # Evidence floor (conf_floor) for the data terms: the velocity-MAD confidence also flags genuine fast
                # motion as unreliable, and without a floor the hold/carry priors take over on those frames and the
                # hand detaches from the video hand.
                conf_data = wrist_conf.clamp(min=self.conf_floor).index_select(0, valid_idx)
                losses[f"data_{side}_pose"] = self._weighted_mean(d_hp.index_select(0, valid_idx).pow(2).mean(-1), conf_data)
                losses[f"data_{side}_orient"] = self._weighted_mean(d_go.index_select(0, valid_idx).pow(2).mean(-1), conf_data)
                losses[f"data_{side}_trans"] = self._weighted_mean(d_ct.index_select(0, valid_idx).pow(2).mean(-1), conf_data)

                ct_err = d_ct.index_select(0, valid_idx)
                if self.wrist_prior_mode == "ray":  # sideways (image-moving) part strong, along-ray (depth) part weak
                    u = getattr(self, f"wrist_ray_{side}").to(ct_err.device).index_select(0, valid_idx)
                    par = (ct_err * u).sum(-1, keepdim=True)
                    perp = ct_err - par * u
                    ct_xy_prior = F.smooth_l1_loss(perp, torch.zeros_like(perp), beta=0.002, reduction="none").sum(-1)
                    ct_z_prior = F.smooth_l1_loss(par, torch.zeros_like(par), beta=0.005, reduction="none").sum(-1)
                else:
                    ct_xy_prior = F.smooth_l1_loss(ct_err[:, :2], torch.zeros_like(ct_err[:, :2]), beta=0.002, reduction="none").sum(-1)
                    ct_z_prior = F.smooth_l1_loss(ct_err[:, 2:], torch.zeros_like(ct_err[:, 2:]), beta=0.005, reduction="none").sum(-1)
                losses[f"wrist_trans_prior_{side}"] = self._weighted_mean(ct_xy_prior + ct_z_prior * 0.3, conf_data)

                go_err = d_go.index_select(0, valid_idx)
                go_prior = F.smooth_l1_loss(go_err, torch.zeros_like(go_err), beta=0.01, reduction="none").sum(-1)
                losses[f"wrist_orient_prior_{side}"] = self._weighted_mean(go_prior, conf_data)

                low_conf = 0.25 + 0.75 * (1.0 - conf_valid)
                bend_angle = F.smooth_l1_loss(
                    joint_angles.index_select(0, valid_idx),
                    joint_angles_init.index_select(0, valid_idx),
                    beta=0.12,
                    reduction="none",
                ).mean(-1)
                losses[f"bend_angle_{side}"] = self._weighted_mean(bend_angle, low_conf)

                exceed = torch.relu(joint_angles.index_select(0, valid_idx) - self.joint_max_norm.to(hp.device))
                limit_loss = (exceed * self.joint_stiffness.to(hp.device)).pow(2).mean(-1)
                losses[f"joint_limit_{side}"] = self._weighted_mean(limit_loss, conf_valid)

                ref_axes = getattr(self, f"ref_axes_{side}").to(hp.device)
                axis_gate = getattr(self, f"ref_axis_gate_{side}").to(hp.device)
                d_hp_joints = d_hp.view(self.T, 15, 3)
                d_along = (d_hp_joints * ref_axes).sum(-1, keepdim=True) * ref_axes
                d_perp = d_hp_joints - d_along
                d_perp_sq = d_perp.pow(2).sum(-1)
                weighted_perp = d_perp_sq * self.axis_perp_weight.to(hp.device) * axis_gate
                losses[f"axis_reg_{side}"] = self._weighted_mean(weighted_perp.index_select(0, valid_idx).mean(-1), conf_valid)

            hand_v, hand_j = self.get_hand_verts_and_joints(side)
            hand_c = hand_v.mean(dim=1)
            palm_c, palm_u, palm_v, palm_n = self._palm_frame(hand_j)
            # Proper palm rotation (rows u, v, u×v — before the ref-normal sign flip below,
            # which would make the frame improper and discontinuous across frames).
            palm_rot = torch.stack([palm_u, palm_v, palm_n], dim=1)
            ref_n = getattr(self, f"grasp_ref_normal_{side}").to(hand_v.device)
            ref_coords = getattr(self, f"grasp_ref_coords_{side}").to(hand_v.device)
            flip = torch.where((palm_n * ref_n.unsqueeze(0)).sum(-1, keepdim=True) < 0, -1.0, 1.0)
            palm_n = palm_n * flip
            rel_obj = obj_c - palm_c
            palm_coords = torch.einsum("td,tfd->tf", rel_obj, torch.stack([palm_u, palm_v, palm_n], dim=1))
            if self._static_branch(f"valid_{side}", valid):
                conf_valid = wrist_conf.index_select(0, valid_idx)
                g_img = self._img_gate(side, valid_idx)
                if g_img is not None:
                    conf_valid = conf_valid * g_img.to(conf_valid.device)
                if self.image_carry_gate and self.T > 1:  # frame weight = the weaker of its two adjacent pairs
                    w_cm = getattr(self, f"img_comove_w_{side}").to(conf_valid.device)
                    w_fr = torch.minimum(torch.cat([w_cm[:1], w_cm]), torch.cat([w_cm, w_cm[-1:]]))
                    conf_valid = conf_valid * w_fr.index_select(0, valid_idx.to(w_fr.device))
                palm_valid = palm_coords.index_select(0, valid_idx)
                grasp_err = (palm_valid - ref_coords.unsqueeze(0)).pow(2).mean(-1)
                losses[f"grasp_{side}"] = self._weighted_mean(grasp_err, conf_valid)
                losses[f"grasp_side_{side}"] = self._weighted_mean(
                    torch.relu(self.grasp_side_margin - palm_valid[:, 2]).pow(2),
                    conf_valid,
                )
                losses[f"grasp_depth_{side}"] = self._weighted_mean((palm_valid[:, 2] - ref_coords[2]).pow(2), conf_valid)

            if self.T > 1:
                both = valid[:-1] & valid[1:]
                both_idx = getattr(self, f"_both_idx_{side}")
                pair_conf = torch.minimum(wrist_conf[:-1], wrist_conf[1:])
                hold_pair_conf = pair_conf  # hand-object coupling terms; gated by image contact evidence when enabled
                if self.image_contact_gate:
                    w_img = getattr(self, f"img_contact_w_{side}").to(pair_conf.device)
                    hold_pair_conf = pair_conf * torch.minimum(w_img[:-1], w_img[1:])
                if self.image_carry_gate:
                    hold_pair_conf = hold_pair_conf * getattr(self, f"img_comove_w_{side}").to(pair_conf.device)
                rel_world_vel = rel_obj[1:] - rel_obj[:-1]
                rel_vel = palm_coords[1:] - palm_coords[:-1]
                if self._static_branch(f"both_{side}", both):
                    losses[f"hold_world_{side}_vel"] = self._weighted_mean(
                        rel_world_vel.index_select(0, both_idx).pow(2).sum(-1) * gap_w.index_select(0, both_idx),
                        hold_pair_conf.index_select(0, both_idx),
                    )
                    losses[f"hold_{side}_vel"] = self._weighted_mean(
                        rel_vel.index_select(0, both_idx).pow(2).sum(-1) * gap_w.index_select(0, both_idx),
                        hold_pair_conf.index_select(0, both_idx),
                    )

                # In-hand rotation hold: while grasped, the object's orientation in the palm
                # frame should be ~constant. Position-only hold terms above are blind to the
                # object spinning about its own center — the dominant SAM3D noise mode.
                grasp_active = getattr(self, f"grasp_active_{side}").to(obj_v.device)
                hold_rot_pairs = both & grasp_active[:-1] & grasp_active[1:]
                if self._static_branch(f"hold_rot_{side}", hold_rot_pairs):
                    hold_rot_idx = getattr(self, f"_hold_rot_idx_{side}")
                    R_rel = torch.matmul(palm_rot, R_obj_eff)
                    ang_rel = roma.rotmat_to_rotvec(torch.matmul(R_rel[:-1].transpose(1, 2), R_rel[1:]))
                    losses[f"hold_obj_rot_{side}"] = self._weighted_mean(
                        ang_rel.pow(2).sum(-1).index_select(0, hold_rot_idx) * gap_w.index_select(0, hold_rot_idx),
                        hold_pair_conf.index_select(0, hold_rot_idx),
                    )
                vel = hand_c[1:] - hand_c[:-1]
                if self._static_branch(f"both_{side}", both):
                    losses[f"smooth_{side}_vel"] = self._weighted_mean(
                        vel.index_select(0, both_idx).pow(2).sum(-1) * gap_w.index_select(0, both_idx),
                        pair_conf.index_select(0, both_idx),
                    )
                    losses[f"smooth_{side}_joints"] = self._weighted_mean(
                        (hp[1:] - hp[:-1]).index_select(0, both_idx).pow(2).mean(-1),
                        pair_conf.index_select(0, both_idx),
                    )

                    bend_stability = 1.0 + 4.0 * (1.0 - pair_conf.index_select(0, both_idx))
                    bend_smooth = (joint_angles[1:] - joint_angles[:-1]).index_select(0, both_idx).pow(2).mean(-1) * gap_w.index_select(0, both_idx) * bend_stability
                    losses[f"bend_smooth_{side}"] = bend_smooth.mean()

                    go = getattr(self, f"global_orient_init_{side}") + d_go
                    losses[f"smooth_{side}_orient"] = self._weighted_mean(
                        (go[1:] - go[:-1]).index_select(0, both_idx).pow(2).sum(-1) * gap_w.index_select(0, both_idx),
                        pair_conf.index_select(0, both_idx),
                    )

                    vert_vel = (hand_v[1:] - hand_v[:-1]).index_select(0, both_idx)
                    losses[f"smooth_{side}_verts"] = self._weighted_mean(
                        vert_vel.pow(2).sum(-1).mean(-1) * gap_w.index_select(0, both_idx),
                        pair_conf.index_select(0, both_idx),
                    )

                if self.T > 2:
                    triple = valid[:-2] & valid[1:-1] & valid[2:]
                    triple_idx = getattr(self, f"_triple_idx_{side}")
                    triple_conf = torch.minimum(torch.minimum(wrist_conf[:-2], wrist_conf[1:-1]), wrist_conf[2:])
                    hold_triple_conf = triple_conf
                    if self.image_contact_gate:
                        w_img = getattr(self, f"img_contact_w_{side}").to(triple_conf.device)
                        hold_triple_conf = triple_conf * torch.minimum(torch.minimum(w_img[:-2], w_img[1:-1]), w_img[2:])
                    if self.image_carry_gate:
                        w_cm = getattr(self, f"img_comove_w_{side}").to(triple_conf.device)
                        hold_triple_conf = hold_triple_conf * torch.minimum(w_cm[:-1], w_cm[1:])
                    if self._static_branch(f"triple_{side}", triple):
                        rel_world_acc = rel_world_vel[1:] - rel_world_vel[:-1]
                        rel_acc = rel_vel[1:] - rel_vel[:-1]
                        hand_acc = vel[1:] - vel[:-1]
                        losses[f"hold_world_{side}_acc"] = self._weighted_mean(
                            rel_world_acc.index_select(0, triple_idx).pow(2).mean(-1), hold_triple_conf.index_select(0, triple_idx)
                        )
                        losses[f"hold_{side}_acc"] = self._weighted_mean(rel_acc.index_select(0, triple_idx).pow(2).mean(-1), hold_triple_conf.index_select(0, triple_idx))
                        losses[f"smooth_{side}_acc"] = self._weighted_mean(hand_acc.index_select(0, triple_idx).pow(2).mean(-1), triple_conf.index_select(0, triple_idx))

            if not self._static_branch(f"valid_{side}", valid):
                continue
            frame_conf = wrist_conf.index_select(0, valid_idx)
            g_img = self._img_gate(side, valid_idx)
            att_conf = frame_conf if g_img is None else frame_conf * g_img.to(frame_conf.device)  # attracting terms only
            hv = hand_v.index_select(0, valid_idx).contiguous()
            obj_valid = obj_v.index_select(0, valid_idx)
            fv = obj_valid[:, self.obj_faces].contiguous()

            anchor_valid = getattr(self, f"grasp_anchor_valid_{side}").index_select(0, valid_idx)
            if self._static_branch(f"anchor_{side}", anchor_valid):
                hand_anchor_idx = getattr(self, f"grasp_anchor_hand_idx_{side}").index_select(0, valid_idx)
                obj_anchor_idx = getattr(self, f"grasp_anchor_obj_idx_{side}").index_select(0, valid_idx)
                anchor_offset = getattr(self, f"grasp_anchor_offset_{side}").index_select(0, valid_idx)
                anchor_weight = getattr(self, f"grasp_anchor_weight_{side}").index_select(0, valid_idx)
                gather_idx_h = hand_anchor_idx.unsqueeze(-1).expand(-1, -1, 3)
                gather_idx_o = obj_anchor_idx.unsqueeze(-1).expand(-1, -1, 3)
                hand_anchor = torch.gather(hv, 1, gather_idx_h)
                obj_anchor = torch.gather(obj_valid, 1, gather_idx_o)
                target_anchor = obj_anchor + anchor_offset
                anchor_err = F.smooth_l1_loss(hand_anchor, target_anchor, beta=0.004, reduction="none").sum(-1)
                anchor_frame_w = 0.2 + 0.5 * (1.0 - frame_conf)
                if g_img is not None:
                    anchor_frame_w = anchor_frame_w * g_img.to(anchor_frame_w.device)
                losses[f"grasp_anchor_{side}"] = self._weighted_mean(
                    anchor_err,
                    anchor_valid.float() * anchor_weight * anchor_frame_w[:, None],
                )

            if self._warp_bvh_enabled:
                dist_sq, face_idx, sign = self._warp_point_to_mesh_distance(side, hv, obj_valid)
            else:  # kaolin distance, sign by winding number
                dist_sq, face_idx, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(hv, fv)
                sign = winding_sign(hv.detach(), obj_valid.detach(), self.obj_faces)
            dist = torch.sqrt(dist_sq + 1e-10)
            signed_dist = dist * sign
            vertex_weights = getattr(self, f"physics_vertex_weights_{side}").to(hv.device)
            pen_depth = torch.relu(self.min_separation - signed_dist)
            pen_loss = F.smooth_l1_loss(
                pen_depth,
                torch.zeros_like(pen_depth),
                beta=self.contact_threshold,
                reduction="none",
            )
            losses[f"pen_{side}"] = self._weighted_vertex_mean(pen_loss, vertex_weights, frame_weights=frame_conf)

            deep_excess = torch.relu(pen_depth - self.deep_pen_threshold)
            deep_pen_loss = deep_excess.pow(2) / self.deep_pen_threshold
            losses[f"deep_pen_{side}"] = self._weighted_vertex_mean(deep_pen_loss, vertex_weights, frame_weights=frame_conf)

            tip_idx = getattr(self, f"fingertip_verts_{side}").to(hv.device)
            tip_signed = signed_dist[:, tip_idx]
            tip_pen_depth = torch.relu(self.min_separation - tip_signed)
            tip_pen_loss = F.smooth_l1_loss(
                tip_pen_depth,
                torch.zeros_like(tip_pen_depth),
                beta=self.tip_contact_threshold,
                reduction="none",
            )
            losses[f"tip_pen_{side}"] = self._weighted_mean(tip_pen_loss, frame_conf[:, None])

            tip_contact_mask = getattr(self, f"tip_contact_mask_{side}").index_select(0, valid_idx)
            if self._static_branch(f"tip_contact_{side}", tip_contact_mask):
                tip_surface = F.smooth_l1_loss(
                    tip_signed,
                    torch.full_like(tip_signed, self.tip_surface_margin),
                    beta=self.tip_contact_threshold,
                    reduction="none",
                )
                losses[f"tip_surface_{side}"] = self._weighted_mean(tip_surface, tip_contact_mask.float() * att_conf[:, None])

            # Grasp closure: on grasp-active frames, pull fingertips hovering near the object
            # (up to 6 cm) onto its surface. The hand estimator reports flat/splayed fingers when they are
            # occluded by the grasped object, and the pose priors keep them flat; tip_surface
            # only acts on tips already in contact, so nothing else closes the grip.
            ga_v = getattr(self, f"grasp_active_{side}").index_select(0, valid_idx)
            if self._static_branch(f"grasp_active_{side}", ga_v):
                close_w = torch.sigmoid((0.06 - tip_signed) / 0.01) * torch.sigmoid((tip_signed - self.min_separation) / 0.003)
                close_w = close_w * ga_v[:, None].float() * att_conf[:, None]
                close_err = torch.relu(tip_signed - self.tip_surface_margin)
                losses[f"tip_close_{side}"] = (close_err * close_w).sum() / close_w.sum().clamp(min=1.0)

            self._region_pen_and_surface(losses, "shaft", side, signed_dist, frame_conf, 0.003, surface_conf=None if g_img is None else att_conf)
            self._region_pen_and_surface(losses, "palm", side, signed_dist, frame_conf, 0.004, surface_conf=None if g_img is None else att_conf)

            contact_weight = torch.exp(-(signed_dist - self.min_separation).pow(2) / (2 * self.contact_threshold**2))
            contact_raw = (signed_dist - self.min_separation).abs()
            eff_w = contact_weight * vertex_weights.unsqueeze(0) * att_conf[:, None]
            losses[f"contact_{side}"] = (contact_raw * eff_w).sum() / eff_w.sum().clamp(min=1.0)

            attract_range = 0.05
            attract_weight = torch.sigmoid((attract_range - signed_dist) / 0.005) * torch.sigmoid((signed_dist - self.min_separation) / 0.003)
            attract_raw = (signed_dist - self.min_separation).pow(2)
            eff_w_a = attract_weight * vertex_weights.unsqueeze(0) * att_conf[:, None]
            losses[f"attract_{side}"] = (attract_raw * eff_w_a).sum() / eff_w_a.sum().clamp(min=1.0)

        return losses
