#!/usr/bin/env python3
"""Light, video-faithful contact projection at canonical export.

Resolves residual hand-object penetration that the optimizer's silhouette/depth
terms cannot constrain, without moving the hand off its video-observed position:
a translation-only (rigid) per-frame hand correction, capped to a few-mm envelope,
applied only where penetration exceeds a few-mm margin. The WiLoR finger pose is
left untouched (only cam_t moves), so the correction cannot deform the grasp; it
only slides the rigid hand a few mm out of the object surface.

Mechanism (a differentiable object SDF, `object_sdf.py`):
  - build ONE object SDF grid from the scaled-canonical mesh;
  - per valid frame, transform the (real, anchor-frame) hand verts into the object
    canonical frame via the export Procrustes R/t, query the SDF (negative = inside);
  - optimize a per-frame 3-DOF delta to drive penetration below `margin_mm`, with an
    L2 anchor to zero + a temporal-smoothness term, then hard-cap |delta| to
    `envelope_mm`;
  - write the delta into cam_t (M-conjugated for the left hand, whose cam_t lives in
    mirrored right-model coords: the left-hand reflection convention).

export_canonical runs it when the object Procrustes RMSE is under 2 mm (a rigid object
chain). It assumes a correctly scaled object: on an over-scaled object the observed grasps
penetrate the inflated mesh, and a penetration term would fight the video evidence. With
the scale right, the remaining penetration is a real error and a few-mm rigid slide
resolves it without disturbing the trajectory.
"""

import numpy as np

# M = diag(-1, 1, 1): left-hand cam_t lives in mirrored right-model coords, so a
# real-frame translation delta maps to M @ delta in cam_t space.
_M_LEFT = np.array([-1.0, 1.0, 1.0], dtype=np.float32)


def project_contact_export(
    hand_verts_anchor,
    hand_valid,
    hand_ct_canonical,
    canonical_verts_cv,
    obj_faces,
    anchor_scale,
    obj_rotation,
    obj_translation,
    *,
    device="cuda",
    margin_mm=2.0,
    envelope_mm=8.0,
    trigger_mm=3.0,
    n_iters=150,
    lr=3e-3,
    lambda_anchor=2.0,
    lambda_temporal=10.0,
):
    """Nudge hand cam_t (in place) by a capped rigid delta to resolve penetration.

    Args mirror export_canonical's anchor-frame quantities. `hand_ct_canonical[side]`
    (T,3) is modified in place; the saved hand verts (FK-reconstructed from cam_t)
    inherit the correction. Returns a per-side report dict.
    """
    import torch
    import torch.nn.functional as F
    from scipy.spatial.transform import Rotation

    from gtf.extraction.export.object_sdf import compute_object_sdf_grid, query_sdf

    report = {"margin_mm": margin_mm, "envelope_mm": envelope_mm}
    if device == "cuda" and not torch.cuda.is_available():
        # Without CUDA the projection runs on the CPU (skipping it would change the exported
        # hands); CPU Adam on (n,778,3) points takes seconds.
        device = "cpu"
        report["device_fallback"] = "cpu"

    # Object SDF in the scaled-canonical frame (built once; the export reconstructs
    # obj_verts[t] = source @ R_mat[t].T + t, source = canonical_verts_cv * anchor_scale).
    try:
        sdf_grid, grid_info = compute_object_sdf_grid(
            np.ascontiguousarray(canonical_verts_cv, dtype=np.float64),
            np.ascontiguousarray(obj_faces),
            np.asarray(anchor_scale, dtype=np.float64).reshape(-1),
            grid_resolution=64,
            padding=0.03,
        )
    except Exception as e:  # noqa: BLE001
        report["skipped"] = f"sdf_build_failed: {type(e).__name__}: {e}"
        return report
    sdf_grid = sdf_grid.to(device)
    R_all = Rotation.from_rotvec(np.asarray(obj_rotation, dtype=np.float64)).as_matrix()

    for side in ("left", "right"):
        valid = np.asarray(hand_valid[side], dtype=bool)
        rows = np.flatnonzero(valid)
        if len(rows) < 1:
            continue
        hv = torch.tensor(hand_verts_anchor[side][rows], dtype=torch.float32, device=device)  # (n,778,3) real frame
        Rt = torch.tensor(R_all[rows], dtype=torch.float32, device=device)  # (n,3,3)
        ot = torch.tensor(obj_translation[rows], dtype=torch.float32, device=device)  # (n,3)
        margin = margin_mm / 1000.0

        def pen_sdf(delta):  # delta (n,3) real-frame -> (n,778) signed dist (neg=inside)
            hin = torch.einsum("nvj,njk->nvk", hv + delta.unsqueeze(1) - ot.unsqueeze(1), Rt)
            return query_sdf(hin.reshape(-1, 3), sdf_grid, grid_info).reshape(len(rows), -1)

        with torch.no_grad():
            pen_before = F.relu(-pen_sdf(torch.zeros(len(rows), 3, device=device))).max(dim=1).values  # (n,)
        if float(pen_before.max()) * 1000 < trigger_mm:
            report[side] = {"applied": False, "n": int(len(rows)), "pen_before_max_mm": float(pen_before.max()) * 1000}
            continue

        delta = torch.zeros(len(rows), 3, device=device, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=lr)
        cap = envelope_mm / 1000.0
        for _ in range(n_iters):
            opt.zero_grad()
            s = pen_sdf(delta)
            l_pen = F.relu(-s - margin).pow(2).sum(dim=1).mean()
            l_anchor = lambda_anchor * delta.pow(2).sum(dim=1).mean()
            l_temporal = lambda_temporal * (delta[1:] - delta[:-1]).pow(2).sum(dim=1).mean() if len(rows) > 1 else 0.0
            (l_pen + l_anchor + l_temporal).backward()
            opt.step()
            with torch.no_grad():  # project onto the envelope each step
                nrm = delta.norm(dim=1, keepdim=True).clamp(min=1e-9)
                delta.mul_(nrm.clamp(max=cap) / nrm)

        with torch.no_grad():
            d = delta.detach()
            pen_after = F.relu(-pen_sdf(d)).max(dim=1).values
            dnp = d.cpu().numpy().astype(np.float32)

        # write into cam_t (M-conjugate for the left hand's mirrored coords)
        ct_delta = dnp * _M_LEFT if side == "left" else dnp
        hand_ct_canonical[side][rows] += ct_delta
        dnorm = np.linalg.norm(dnp, axis=1)
        report[side] = {
            "applied": True,
            "n": int(len(rows)),
            "pen_before_max_mm": float(pen_before.max()) * 1000,
            "pen_after_max_mm": float(pen_after.max()) * 1000,
            "pen_before_med_mm": float(pen_before.median()) * 1000,
            "pen_after_med_mm": float(pen_after.median()) * 1000,
            "delta_max_mm": float(dnorm.max()) * 1000,
            "delta_med_mm": float(np.median(dnorm)) * 1000,
        }
    return report
