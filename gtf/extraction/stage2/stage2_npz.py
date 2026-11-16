#!/usr/bin/env python3
"""optimized.npz <-> packed stage-2 state codec (recover_states / rebuild_npz).

Recovery uses transport semantics: the optimizer saves verts + 45-D pose + betas but not
orient/cam_t. Procrustes vs the zero-orient FK gives the global orientation R_k;
cam_t_k := wrist_k - FK_joint0(R_k, pose, betas) with wrist_k = J-regressor row 0 on the
saved verts. Left hands are un-mirrored first (x -> -x) -> right-model coords. The rebuild
applies the rigid correction to the saved verts about the wrist, so their shape passes
through untouched and the identity path (no-op correction) reproduces the input exactly by
construction. The object is an anchor-relative absolute rotation chain + centroid
translation, rebuilt by a per-frame rigid correction that preserves per-keyframe non-rigid
structure (see rebuild_npz).
"""

import numpy as np
import torch

from gtf.extraction.stage2.stage2_common import (
    HAND_CAM_T,
    HAND_ORIENT,
    LEFT,
    OBJ,
    OBJ_ROT,
    OBJ_TRANS,
    RIGHT,
    STATE_DIM,
    axis_angle_to_6d,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)


def procrustes_torch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Batched column-convention rotation R with B_centered ≈ R @ A_centered (points as columns).

    A, B: (K, N, 3). Returns (K, 3, 3)."""
    A0 = A - A.mean(dim=1, keepdim=True)
    B0 = B - B.mean(dim=1, keepdim=True)
    H = A0.transpose(1, 2) @ B0  # (K,3,3)
    U, _, Vt = torch.linalg.svd(H)
    det = torch.linalg.det(U @ Vt)
    D = torch.eye(3, device=A.device).repeat(len(A), 1, 1)
    D[:, 2, 2] = torch.sign(det)
    R_row = U @ D @ Vt  # row conv: B0 ≈ A0 @ R_row
    return R_row.transpose(1, 2)


ID6 = torch.tensor([1.0, 0, 0, 0, 1.0, 0])


@torch.no_grad()
def recover_states(opt: dict, mano, device) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """optimized.npz dict → (K,207) raw states + (K,3) entity-valid + rebuild context.

    Hand semantics are wrist-anchored transport (see module docstring): the recovered
    (orient, cam_t) satisfy wrist(saved verts) = FK_joint0(orient, pose, betas) + cam_t,
    the relation the training canonical obeys, so the denoiser sees in-convention
    channels and the identity rebuild is exact. The Procrustes residual between the saved
    verts and FK of the saved pose is reported as a diagnostic, not asserted."""
    K = len(opt["frame_indices"])
    states = torch.zeros(K, STATE_DIM, device=device)
    valid = torch.zeros(K, 3, dtype=torch.bool, device=device)
    ctx = {"K": K, "recover_resid_mm": {}}
    w_reg = mano.mano_layer.th_J_regressor[0].float().to(device)  # (778,) wrist = row 0

    for side, col, hs in (("left", 0, LEFT), ("right", 1, RIGHT)):
        # single-hand npz: a missing side is all-invalid
        if any(f"hand_{side}_{k}" not in opt for k in ("valid", "pose", "verts", "betas")):
            continue
        hv = torch.from_numpy(opt[f"hand_{side}_valid"].astype(bool)).to(device)
        if hv.sum() < 1:
            continue
        rows = hv.nonzero(as_tuple=True)[0]
        verts = torch.from_numpy(opt[f"hand_{side}_verts"]).float().to(device)[rows]
        if side == "left":
            verts = verts * torch.tensor([-1.0, 1, 1], device=device)  # un-mirror → right-model coords
        pose_aa = torch.from_numpy(opt[f"hand_{side}_pose"]).float().to(device)[rows].reshape(-1, 15, 3)
        pose_6d = axis_angle_to_6d(pose_aa).reshape(-1, 90)
        betas = torch.from_numpy(opt[f"hand_{side}_betas"]).float().to(device).unsqueeze(0).expand(len(rows), -1)
        zeros3 = torch.zeros(len(rows), 3, device=device)
        v0, _, _ = mano(ID6.to(device).expand(len(rows), -1), pose_6d, betas, zeros3, side="right")
        R = procrustes_torch(v0, verts)  # column conv == MANO global orient (scale-invariant)
        orient_6d = matrix_to_rotation_6d(R)
        v_fk, j_fk, _ = mano(orient_6d, pose_6d, betas, zeros3, side="right")
        wrist = torch.einsum("v,kvc->kc", w_reg, verts)
        cam_t = wrist - j_fk[:, 0]
        # diagnostic: how far saved verts are from a pure similarity of FK(saved pose)
        fit_t = (verts - v_fk).mean(dim=1)
        ctx["recover_resid_mm"][side] = (verts - v_fk - fit_t.unsqueeze(1)).norm(dim=-1).median().item() * 1000
        h = torch.zeros(K, 99, device=device)
        h[rows] = torch.cat([orient_6d, cam_t, pose_6d], dim=-1)
        states[:, hs] = h
        valid[:, col] = hv
        ctx[f"hand_{side}"] = {"rows": rows, "R": R, "wrist": wrist, "pose_6d": pose_6d, "betas": betas}

    # object: anchor-relative absolute rotation chain + centroid translation
    overts = torch.from_numpy(opt["obj_verts"]).float().to(device)  # (K,V,3)
    cents = overts.mean(dim=1)
    anchor_fi = int(opt["anchor_frame_index"]) if "anchor_frame_index" in opt else int(opt["frame_indices"][0])
    anchor_pos = int(np.argmin(np.abs(np.asarray(opt["frame_indices"]) - anchor_fi)))
    R_abs = procrustes_torch(overts[anchor_pos].unsqueeze(0).expand(K, -1, -1), overts)
    states[:, OBJ.start + OBJ_ROT.start : OBJ.start + OBJ_ROT.stop] = matrix_to_rotation_6d(R_abs)
    states[:, OBJ.start + OBJ_TRANS.start : OBJ.start + OBJ_TRANS.stop] = cents
    valid[:, 2] = True
    # rebuild-exactness: transporting anchor verts by the recovered chain must reproduce obj_verts
    rebuilt = (overts[anchor_pos] - cents[anchor_pos]).unsqueeze(0) @ R_abs.transpose(1, 2) + cents.unsqueeze(1)
    ctx["obj_rigid_resid_mm"] = (rebuilt - overts).norm(dim=-1).max().item() * 1000
    ctx["anchor_pos"] = anchor_pos
    ctx["obj_R"] = R_abs
    ctx["obj_cent"] = cents
    return states, valid, ctx


@torch.no_grad()
def rebuild_npz(opt: dict, states_dn: torch.Tensor, valid: torch.Tensor, ctx: dict, mano, device) -> dict:
    """Corrected states → optimized.npz-schema dict via transport.

    Hands: rigid correction about the wrist, verts_dn = ΔR·(verts − wrist) + wrist_dn with
    ΔR = R_dn·R_recoveredᵀ and wrist_dn = cam_t_dn + FK_joint0(orient_dn, original pose);
    the saved hand shape passes through untouched, and a no-op correction reproduces the
    input exactly.

    Object: per-frame rigid correction about each frame's own centroid,
    verts_dn[k] = ΔR_k·(verts[k] − cent_k) + t_dn_k with ΔR_k = R_dn_k·R_recovered_kᵀ.
    Preserves the per-keyframe SAM 3D scale (non-rigid structure) of obj_verts; identity
    corrections pass the input through exactly regardless of rigidity."""
    out = {k: v for k, v in opt.items()}
    mirror = torch.tensor([-1.0, 1, 1], device=device)
    for side, col, hs in (("left", 0, LEFT), ("right", 1, RIGHT)):
        hc = ctx.get(f"hand_{side}")
        if hc is None:
            continue
        rows = hc["rows"]
        h = states_dn[rows][:, hs]
        zeros3 = torch.zeros(len(rows), 3, device=device)
        _, j_fk_dn, _ = mano(h[:, HAND_ORIENT], hc["pose_6d"], hc["betas"], zeros3, side="right")
        wrist_dn = h[:, HAND_CAM_T] + j_fk_dn[:, 0]
        dR = rotation_6d_to_matrix(h[:, HAND_ORIENT]) @ hc["R"].transpose(1, 2)
        verts_all = torch.from_numpy(opt[f"hand_{side}_verts"]).float().to(device).clone()
        v_r = verts_all[rows]
        if side == "left":
            v_r = v_r * mirror
        v_dn = (v_r - hc["wrist"].unsqueeze(1)) @ dR.transpose(1, 2) + wrist_dn.unsqueeze(1)
        if side == "left":
            v_dn = v_dn * mirror
        verts_all[rows] = v_dn
        out[f"hand_{side}_verts"] = verts_all.cpu().numpy().astype(np.float32)
    o = states_dn[:, OBJ]
    R_dn = rotation_6d_to_matrix(o[:, OBJ_ROT])
    t_dn = o[:, OBJ_TRANS]
    overts = torch.from_numpy(opt["obj_verts"]).float().to(device)
    dR = R_dn @ ctx["obj_R"].transpose(1, 2)
    rebuilt = (overts - ctx["obj_cent"].unsqueeze(1)) @ dR.transpose(1, 2) + t_dn.unsqueeze(1)
    out["obj_verts"] = rebuilt.cpu().numpy().astype(np.float32)
    return out
