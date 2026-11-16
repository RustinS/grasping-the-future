"""Loss weights and the phased optimization schedule."""

import time
import numpy as np
import torch
from gtf.extraction.optimizer.pose_optimizer import MANOPoseOptimizer


DEFAULT_WEIGHTS = {
    "data_rot": 1.0,
    "data_trans": 1.0,
    "data_obj_rot": 3.0,
    # The smoothed SAM 3D translation path follows the SAM 3 mask; without a firm anchor the
    # bidirectional contact losses (pen/attract/anchors/tip_close) drag the object toward the
    # hand during fast motion, where the hand evidence is down-weighted.
    "data_obj_trans": 40.0,
    "data_obj_scale": 30.0,
    "data_pose": 20.0,
    "data_orient": 1.0,
    "smooth_vel": 5.0,
    "smooth_acc": 10.0,
    "smooth_rot": 5.0,
    "smooth_verts": 5.0,
    "smooth_orient": 3.0,
    "smooth_joints": 18.0,
    # Hand-object coupling: strong enough to suppress relative jitter, weak enough that the
    # per-frame evidence (data/sil terms) wins during genuine fast motion. Much stronger
    # coupling drags the object off the SAM 3 mask whenever the hand evidence is
    # down-weighted (fast-motion frames), a visible lag behind the video.
    "hold_world_vel": 100.0,
    "hold_world_acc": 200.0,
    "hold_vel": 70.0,
    "hold_acc": 140.0,
    "hold_obj_rot": 10.0,
    "sil_obj_sdf": 120.0,
    "sil_obj_centroid": 80.0,
    "sil_obj_bbox": 80.0,
    "sil_obj_support": 100.0,
    "grasp_side": 40.0,
    "grasp_depth": 110.0,
    "grasp": 45.0,
    "grasp_anchor": 80.0,
    "pen": 200.0,
    "deep_pen": 500.0,
    "deep_shaft_pen": 600.0,
    "deep_palm_pen": 600.0,
    "tip_pen": 300.0,
    "tip_surface": 120.0,
    "tip_close": 80.0,
    "shaft_pen": 250.0,
    "shaft_surface": 40.0,
    "palm_pen": 250.0,
    "palm_surface": 75.0,
    "contact": 50.0,
    "attract": 20.0,
    "wrist_trans_prior": 15.0,
    "wrist_orient_prior": 10.0,
    "bend_angle": 42.0,
    "bend_smooth": 38.0,
    "joint_limit": 50.0,
    "axis_reg": 90.0,
    "carry_finger": 120.0,
}


def _weight_for(key, weights):
    """Find best-matching weight by token overlap (most-specific match wins)."""
    matches = [(len(wk.split("_")), w) for wk, w in weights.items() if all(t in key for t in wk.split("_"))]
    return max(matches)[1] if matches else 1.0


def _require_finite_optimizer_loss(total, phase, components=None):
    """Abort before Adam can turn one invalid loss into persistent NaN state."""

    if bool(torch.isfinite(total).all().item()):
        return
    bad = []
    if components is not None:
        bad = [name for name, value in components.items() if not bool(torch.isfinite(value).all().item())]
    detail = f"; non-finite components={','.join(sorted(bad))}" if bad else ""
    raise FloatingPointError(f"non-finite optimizer loss in {phase}{detail}")


def _init_gpu_best(params):
    params = list(params)
    device = params[0].device
    return torch.full((), float("inf"), device=device), [parameter.detach().clone() for parameter in params]


@torch.no_grad()
def _update_gpu_best(candidate_loss, best_loss, best_state, params):
    improved = candidate_loss.detach() < best_loss
    best_loss.copy_(torch.where(improved, candidate_loss.detach(), best_loss))
    for saved, parameter in zip(best_state, params):
        saved.copy_(torch.where(improved, parameter.detach(), saved))


@torch.no_grad()
def _restore_gpu_best(best_loss, best_state, params):
    if not bool(torch.isfinite(best_loss).item()):
        return
    for parameter, saved in zip(params, best_state):
        parameter.copy_(saved)


def _run_phase(model, param_groups, eta_min, n_iters, weights, t0, log_every, wrist_freeze=None):
    """Run one optimization phase with Adam + cosine annealing.

    wrist_freeze: optional dict {side: (go_frozen, ct_frozen, target_mask)} to enforce
    start-window wrist freeze after each step.
    """
    param_groups = [g for g in ({**g, "params": list(g["params"])} for g in param_groups if g.get("params")) if g["params"]]
    if n_iters <= 0 or not param_groups:
        return

    opt = torch.optim.Adam(param_groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_iters, eta_min=eta_min)
    all_params = [p for g in param_groups for p in g["params"]]
    best_loss, best_state = _init_gpu_best(all_params)
    high_dim = [p for p in all_params if p.requires_grad and p.numel() > p.shape[0] * 10]
    low_dim = [p for p in all_params if p.requires_grad and p.numel() <= p.shape[0] * 10]

    for it in range(n_iters):
        opt.zero_grad()
        losses = model()
        total = sum(_weight_for(k, weights) * v for k, v in losses.items())
        _require_finite_optimizer_loss(total, "joint phase", losses)
        if not (torch.is_tensor(total) and total.requires_grad and total.grad_fn is not None):
            # No loss depends on this phase's params (e.g. a fingers-only phase whose finger losses are all gated off),
            # so backward() would raise. total is deterministic in the params, so this holds on every iteration.
            print("      [phase] no grad-coupled loss for these params — skipping phase")
            break
        candidate_loss = total.detach()
        total.backward()
        if high_dim:
            torch.nn.utils.clip_grad_norm_(high_dim, 1.0)
        if low_dim:
            torch.nn.utils.clip_grad_norm_(low_dim, 0.5)
        opt.step()
        with torch.no_grad():
            model.obj_log_scale.clamp_(-0.18, 0.18)
            for side in model.active_sides:
                getattr(model, f"d_global_orient_{side}").clamp_(-model.max_wrist_rot, model.max_wrist_rot)
                getattr(model, f"d_cam_t_{side}").clamp_(-model.max_wrist_trans, model.max_wrist_trans)
            if wrist_freeze:
                for side, (go_frozen, ct_frozen, target) in wrist_freeze.items():
                    go = getattr(model, f"d_global_orient_{side}")
                    ct = getattr(model, f"d_cam_t_{side}")
                    go[target] = go_frozen[target]
                    ct[target] = ct_frozen[target]
            _update_gpu_best(candidate_loss, best_loss, best_state, all_params)
        sched.step()
        if (it + 1) % log_every == 0 or it == 0 or it == n_iters - 1:
            val = total.item()
            parts = " ".join(f"{k}={v.item():.5f}(x{_weight_for(k, weights)})" for k, v in sorted(losses.items()))
            print(f"    [{it + 1:4d}/{n_iters}] total={val:.5f} | {parts}")
            print(f"      {time.time() - t0:.1f}s, lr={sched.get_last_lr()[0]:.6f}")

    _restore_gpu_best(best_loss, best_state, all_params)


def _run_phase0_start_window(model, t0, n_iters=160):
    """Phase 0: Start-window wrist solve — only adjust wrist on target-mask frames."""
    weights_p0 = {
        "grasp": 12.0,
        "grasp_side": 35.0,
        "grasp_depth": 45.0,
        "grasp_anchor": 260.0,
        "shaft_surface": 110.0,
        "palm_surface": 120.0,
        "shaft_pen": 250.0,
        "palm_pen": 250.0,
        "contact": 90.0,
        "attract": 35.0,
        "wrist_trans_prior": 1.5,
        "wrist_orient_prior": 1.0,
        "data_orient": 0.1,
        "data_trans": 0.1,
        "data_pose": 0.0,
    }

    sides = [side for side in model.active_sides if bool(getattr(model, f"target_mask_{side}").any().item())]
    if not sides:
        return

    model._include_object_losses = False
    model._loss_sides = tuple(sides)
    _set_requires_grad(model, obj=False, fingers=False, wrist=False)
    state = {}
    go_params, ct_params = [], []
    for side in sides:
        wrist_go = getattr(model, f"d_global_orient_{side}")
        wrist_ct = getattr(model, f"d_cam_t_{side}")
        target = getattr(model, f"target_mask_{side}").to(model.device)
        wrist_go.requires_grad = True
        wrist_ct.requires_grad = True
        active = [wrist_go, wrist_ct]
        state[side] = {
            "go": wrist_go,
            "ct": wrist_ct,
            "target": target,
            "params": active,
            "best": _init_gpu_best(active),
        }
        go_params.append(wrist_go)
        ct_params.append(wrist_ct)

    opt = torch.optim.Adam([{"params": go_params, "lr": 7e-4}, {"params": ct_params, "lr": 1.2e-3}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_iters, eta_min=1e-5)
    for it in range(n_iters):
        opt.zero_grad()
        losses = model()
        side_totals = {
            side: sum(
                _weight_for(key, weights_p0) * value
                for key, value in losses.items()
                if f"_{side}" in key and "_obj_" not in key and not key.startswith("sil_obj") and not key.startswith("smooth_obj")
            )
            for side in sides
        }
        total = sum(side_totals.values())
        _require_finite_optimizer_loss(total, "start-window phase", side_totals)
        total.backward()
        for side in sides:
            torch.nn.utils.clip_grad_norm_(state[side]["params"], 0.25)
        opt.step()
        with torch.no_grad():
            for side in sides:
                wrist_go = state[side]["go"]
                wrist_ct = state[side]["ct"]
                target = state[side]["target"]
                wrist_go[~target] = 0.0
                wrist_ct[~target] = 0.0
                wrist_go[target] = wrist_go[target].clamp(-0.45, 0.45)
                wrist_ct[target, :2] = wrist_ct[target, :2].clamp(-0.05, 0.05)
                wrist_ct[target, 2] = wrist_ct[target, 2].clamp(-0.08, 0.08)
                best_loss, best_state = state[side]["best"]
                _update_gpu_best(side_totals[side], best_loss, best_state, state[side]["params"])
        sched.step()
        if (it + 1) % 20 == 0 or it == 0 or it == n_iters - 1:
            values = " ".join(f"{side}={side_totals[side].item():.5f}" for side in sides)
            print(f"    [{it + 1:4d}/{n_iters}] {values} lr={sched.get_last_lr()[0]:.6f}  ({time.time() - t0:.1f}s)")

    for side in sides:
        best_loss, best_state = state[side]["best"]
        _restore_gpu_best(best_loss, best_state, state[side]["params"])
        wrist_go, target = state[side]["go"], state[side]["target"]
        print(f"    {side}: Phase 0 done, go range [{float(wrist_go[target].min()):.4f}, {float(wrist_go[target].max()):.4f}]")

    model._include_object_losses = True
    model._loss_sides = None


def _run_phase4_carry_refine(model, weights, t0, n_iters=180):
    """Phase 4: Carry refinement — regularize low-confidence frames toward trusted targets."""
    carry_w_finger = weights["carry_finger"]

    sides = [side for side in model.active_sides if bool(getattr(model, f"carry_mask_{side}").any().item())]
    if not sides:
        return

    model._include_object_losses = False
    model._loss_sides = tuple(sides)
    _set_requires_grad(model, obj=False, fingers=False, wrist=False)
    state = {}
    finger_groups, go_params, ct_params = [], [], []
    for side in sides:
        fingers = [getattr(model, f"d_hand_pose_basis_{side}").coeffs]
        wrist_go = getattr(model, f"d_global_orient_{side}")
        wrist_ct = getattr(model, f"d_cam_t_{side}")
        carry_mask = getattr(model, f"carry_mask_{side}").to(model.device)
        active = fingers + [wrist_go, wrist_ct]
        for parameter in active:
            parameter.requires_grad = True
        state[side] = {
            "fingers": fingers,
            "go": wrist_go,
            "ct": wrist_ct,
            "mask": carry_mask,
            "go_anchor": wrist_go.detach().clone(),
            "ct_anchor": wrist_ct.detach().clone(),
            "params": active,
            "best": _init_gpu_best(active),
        }
        finger_groups.extend(fingers)
        go_params.append(wrist_go)
        ct_params.append(wrist_ct)

    opt = torch.optim.Adam(
        [
            {"params": finger_groups, "lr": 3e-4},
            {"params": go_params, "lr": 1.5e-4},
            {"params": ct_params, "lr": 2.5e-4},
        ],
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_iters, eta_min=1e-5)
    for it in range(n_iters):
        opt.zero_grad()
        losses = model()
        side_totals = {
            side: sum(_weight_for(key, weights) * value for key, value in losses.items() if f"_{side}" in key) + carry_w_finger * model._carry_losses(side) for side in sides
        }
        total = sum(side_totals.values())
        _require_finite_optimizer_loss(total, "carry phase", side_totals)
        total.backward()
        for side in sides:
            torch.nn.utils.clip_grad_norm_(state[side]["fingers"], 0.6)
            torch.nn.utils.clip_grad_norm_([state[side]["go"], state[side]["ct"]], 0.25)
        opt.step()
        with torch.no_grad():
            for side in sides:
                wrist_go, wrist_ct = state[side]["go"], state[side]["ct"]
                carry_mask = state[side]["mask"]
                wrist_go[~carry_mask] = state[side]["go_anchor"][~carry_mask]
                wrist_ct[~carry_mask] = state[side]["ct_anchor"][~carry_mask]
                wrist_go[carry_mask] = wrist_go[carry_mask].clamp(-0.35, 0.35)
                wrist_ct[carry_mask, :2] = wrist_ct[carry_mask, :2].clamp(-0.04, 0.04)
                wrist_ct[carry_mask, 2] = wrist_ct[carry_mask, 2].clamp(-0.05, 0.05)
                best_loss, best_state = state[side]["best"]
                _update_gpu_best(side_totals[side], best_loss, best_state, state[side]["params"])
        sched.step()
        if (it + 1) % 20 == 0 or it == 0 or it == n_iters - 1:
            values = " ".join(f"{side}={side_totals[side].item():.5f}" for side in sides)
            print(f"    [{it + 1:4d}/{n_iters}] {values} lr={sched.get_last_lr()[0]:.6f}  ({time.time() - t0:.1f}s)")

    for side in sides:
        best_loss, best_state = state[side]["best"]
        _restore_gpu_best(best_loss, best_state, state[side]["params"])
        print(f"    {side}: Phase 4 carry-refine done, {int(state[side]['mask'].sum())} frames refined")

    model._include_object_losses = True
    model._loss_sides = None


def _split_phase_iters(n_iters):
    if n_iters <= 1:
        return 0, 0, n_iters
    if n_iters == 2:
        return 1, 0, 1
    rigid = max(1, int(round(n_iters * 0.3)))
    fingers = max(1, int(round(n_iters * 0.2)))
    return rigid, fingers, n_iters - rigid - fingers


def _resolve_phase_iters(n_iters):
    """Return the phase 0..4 iteration counts: 160/150/75/250/100 for the default 500-iteration schedule, otherwise
    160, a 30/20/50% split of n_iters over phases 1-3, and 180."""
    rigid, fingers, joint = _split_phase_iters(n_iters)
    if n_iters == 500:
        return 160, 150, 75, 250, 100
    return 160, rigid, fingers, joint, 180


def _set_requires_grad(model, obj=False, fingers=False, wrist=False):
    model.obj_rot_basis.coeffs.requires_grad = obj
    model.obj_trans_basis.coeffs.requires_grad = obj
    model.obj_log_scale.requires_grad = obj
    for side in model.active_sides:
        getattr(model, f"d_hand_pose_basis_{side}").coeffs.requires_grad = fingers
        getattr(model, f"d_global_orient_{side}").requires_grad = wrist
        getattr(model, f"d_cam_t_{side}").requires_grad = wrist


def optimize(frames, obj_faces, canonical_verts_raw, config=None, anchor_frame_index=None, device="cuda"):
    """Build MANOPoseOptimizer on the loaded keyframes and run the phased schedule (phase 0: start-window wrist solve;
    1: rigid object + wrists; 2: fingers; 3: everything; 4: carry refinement). Returns the optimized geometry, the
    parametric hand state and the final losses."""
    config = config or {}
    weights = {k: config.get(f"w_{k}", v) for k, v in DEFAULT_WEIGHTS.items()}
    n_iters = config.get("n_iters", 500)

    print(f"\n  Building MANO optimizer ({len(frames)} frames)...")
    model = MANOPoseOptimizer(
        frames,
        obj_faces,
        canonical_verts_raw=canonical_verts_raw,
        contact_threshold=config.get("contact_threshold", 0.008),
        min_separation=config.get("min_separation", 0.002),
        deep_pen_threshold=config.get("deep_pen_threshold", 0.005),
        max_wrist_rot=config.get("max_wrist_rot", 0.08),
        max_wrist_trans=config.get("max_wrist_trans", 0.05),
        thumb_vertex_weight=config.get("thumb_vertex_weight", 0.25),
        grasp_side_margin=config.get("grasp_side_margin", 0.005),
        start_frame_max=config.get("start_frame_max", 59),
        carry_conf_low=config.get("carry_conf_low", 0.5),
        carry_conf_high=config.get("carry_conf_high", 0.8),
        conf_floor=config.get("conf_floor", 0.3),
        # set by load_segment: True only when the object was placed at the hand's depth → skip the hand←object drag
        decouple_depth=config.get("decouple_depth", False),
        hand_depth_correction=config.get("hand_depth_correction", "z"),
        wrist_prior_mode=config.get("wrist_prior_mode", "camera"),
        image_contact_gate=config.get("image_contact_gate", False),
        image_carry_gate=config.get("image_carry_gate", False),
        contact_depth=config.get("contact_depth", "off"),
        device=device,
    )
    print(f"  Active sides: {model.active_sides}")
    print(f"  Weights: {weights}")

    obj_params = [model.obj_rot_basis.coeffs, model.obj_trans_basis.coeffs, model.obj_log_scale]
    wrist_params = []
    finger_params = []
    for side in model.active_sides:
        finger_params.append(getattr(model, f"d_hand_pose_basis_{side}").coeffs)
        wrist_params.append(getattr(model, f"d_global_orient_{side}"))
        wrist_params.append(getattr(model, f"d_cam_t_{side}"))

    n_phase0, n_phase_rigid, n_phase_fingers, n_phase_all, n_phase4 = _resolve_phase_iters(n_iters)
    t0 = time.time()

    print(f"\n  Phase 0: start-window wrist solve ({n_phase0} iters)")
    _run_phase0_start_window(model, t0, n_iters=n_phase0)

    wrist_freeze = {}
    with torch.no_grad():
        for side in model.active_sides:
            target = getattr(model, f"target_mask_{side}").to(model.device)
            if target.any():
                go_frozen = getattr(model, f"d_global_orient_{side}").detach().clone()
                ct_frozen = getattr(model, f"d_cam_t_{side}").detach().clone()
                wrist_freeze[side] = (go_frozen, ct_frozen, target)

    print(f"\n  Phase 1: rigid object + wrist ({n_phase_rigid} iters)")
    _set_requires_grad(model, obj=True, fingers=False, wrist=True)
    _run_phase(
        model,
        [
            {"params": obj_params, "lr": 3e-4},
            {"params": wrist_params, "lr": 4e-4},
        ],
        5e-5,
        n_phase_rigid,
        weights,
        t0,
        50,
        wrist_freeze=wrist_freeze,
    )

    print("\n  Refreshing contact targets from rigid-aligned geometry")
    model.refresh_contact_targets()

    print(f"\n  Phase 2: fingers only ({n_phase_fingers} iters, lr=4e-4)")
    model._include_object_losses = False
    model._loss_sides = None
    _set_requires_grad(model, obj=False, fingers=True, wrist=False)
    _run_phase(model, [{"params": finger_params, "lr": 4e-4}], 2e-4, n_phase_fingers, weights, t0, 100, wrist_freeze=wrist_freeze)

    print(f"\n  Phase 3: object + fingers + wrist ({n_phase_all} iters)")
    model._include_object_losses = True
    _set_requires_grad(model, obj=True, fingers=True, wrist=True)

    param_groups = [
        {"params": finger_params, "lr": 4e-4},
        {"params": obj_params, "lr": 1e-4},
        {"params": wrist_params, "lr": 2e-4},
    ]
    _run_phase(model, param_groups, 5e-5, n_phase_all, weights, t0, 50, wrist_freeze=wrist_freeze)

    print("\n  Refreshing contact targets from jointly-optimized geometry")
    model.refresh_contact_targets()
    model._init_carry_targets()

    print(f"\n  Phase 4: carry refinement ({n_phase4} iters)")
    _run_phase4_carry_refine(model, weights, t0, n_iters=n_phase4)

    with torch.no_grad():
        # Final forward pass: capture per-component and weighted total losses
        final_losses = model()
        loss_dict = {k: float(v.item()) for k, v in final_losses.items()}
        loss_dict["_weighted_total"] = sum(_weight_for(k, weights) * v.item() for k, v in final_losses.items())

        scale_corr = float(torch.exp(model.obj_log_scale).item())
        loss_dict["_obj_scale_correction"] = scale_corr
        if abs(scale_corr - 1.0) > 0.005:
            print(f"  Object scale refined: x{scale_corr:.3f}")

        # Extract hand parametric state (as in get_hand_verts_and_joints)
        hands_parametric = {}
        for side in model.active_sides:
            d_hp = model._mask_hidden_thumb_delta(side, getattr(model, f"d_hand_pose_basis_{side}")())
            hp = getattr(model, f"hand_pose_init_{side}").to(model.device) + d_hp
            hands_parametric[side] = {
                "hand_pose": hp.cpu().numpy().astype(np.float32),
                "betas": getattr(model, f"betas_const_{side}").cpu().numpy().astype(np.float32),
            }

        return {
            "frame_indices": [f["frame_index"] for f in frames],
            "obj_verts_optimized": model.get_obj_verts().cpu().numpy(),
            "hands": {
                side: {
                    "verts_optimized": model.get_hand_verts(side).cpu().numpy(),
                    "valid_mask": getattr(model, f"valid_{side}").cpu().numpy(),
                }
                for side in model.active_sides
            },
            "hands_parametric": hands_parametric,
            "canonical_verts": model.canonical_verts.cpu().numpy().astype(np.float32),
            "obj_faces": model.obj_faces.cpu().numpy(),
            # Fold the learned scale correction in so export_canonical's Procrustes
            # (canonical * obj_scale_init vs obj_verts) stays scale-consistent.
            "obj_scale_init": (model.obj_scale_init.cpu() * scale_corr).numpy().astype(np.float32),
            "anchor_frame_index": anchor_frame_index,
            "hand_scale_mode": model.hand_scale_mode,
            "pen_sign_mode": model.pen_sign_mode,
            "optimization_losses": loss_dict,
        }
