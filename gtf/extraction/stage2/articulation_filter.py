"""Stage-2 articulation filter: swing-twist clamp + Karcher smoothing of the finger joints.

The stage-2 temporal filter (temporal_filter.apply_temporal_filter) smooths only the rigid
hand channels (wrist orient + cam_t); finger articulation passes through it untouched, so
WiLoR's per-frame finger jitter and the optimizer's occasional unphysical axial twist would
survive into the training canonical. This module adds the articulation half:

  clamp  (swing-twist)  Decompose each joint's relative rotation about its rest bone axis
         into swing + twist; clamp twist tightly (MCP 25 / PIP-DIP 15 deg: fingers barely
         rotate axially) and swing generously (135 deg: a safety bound against broken splays).

  smooth (per-joint SO(3) Karcher, radius 1)  The articulation analog of temporal_filter's
         drift-free wrist Karcher. Radius 1 matches the fingertip jitter of ground-truth hands.

Delta semantics (to compose with the rigid transport): rebuild_npz transports the saved hand
verts (it does not re-FK from pose), and the npz holds no wrist orientation or translation.
So only the change in articulation, FK(pose_new) - FK(pose_orig), is applied, rotated into the
camera frame by the per-frame rigid placement recovered (Procrustes) from the saved verts.
pose_new == pose_orig leaves the input verts unchanged. Left hand: the manopth FK x-mirrors
the output, and the Procrustes runs in that same mirrored frame, so the delta is consistent
(no separate M-conjugation needed).

Entry point: apply_articulation_filter(out, mano, device, ...) mutates the rebuilt npz dict
in place (hand_{side}_verts + hand_{side}_pose) and returns a per-side report.
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation

# pose dim k (0..14) == MANO joint k+1. finger-end joints -> fingertip vertex (manopth verts).
_END_TIP = {3: 317, 6: 444, 9: 673, 12: 556, 15: 745}
_IS_MCP = np.array([j % 3 == 0 for j in range(15)])  # pose idx 0,3,6,9,12 are the MCPs


# --------------------------------------------------------------------------- FK (manopth, aa)
def _fk_canonical(layer, pose_aa, betas, side, device):
    """(K,45) axis-angle -> (K,778,3) canonical-placed verts (ID orient, no cam_t), meters.
    x-mirrors the left hand — matches MANOHelper / optimizer_common.MANOForward exactly."""
    K = pose_aa.shape[0]
    full = torch.cat([torch.zeros(K, 3, device=device), torch.as_tensor(pose_aa, dtype=torch.float32, device=device)], dim=-1)
    be = torch.as_tensor(np.asarray(betas, np.float32), device=device).reshape(1, 10).expand(K, -1)
    with torch.no_grad():
        v, _ = layer(full, th_betas=be)
    v = v / 1000.0
    if side == "left":
        v = torch.cat([-v[..., :1], v[..., 1:]], dim=-1)
    return v.cpu().numpy().astype(np.float64)


def _bone_axes(layer, betas, device):
    """(15,3) unit rest bone (twist) axes for the given betas (right-model frame; identical for
    the left hand because the pose dims are right-model dims, the M-mirror is on the output)."""
    Jreg = layer.th_J_regressor.to_dense().cpu().numpy() if hasattr(layer.th_J_regressor, "to_dense") else layer.th_J_regressor.cpu().numpy()
    v = _fk_canonical(layer, np.zeros((1, 45), np.float32), betas, "right", device)[0]
    J = Jreg @ v  # (16,3)
    axes = np.zeros((15, 3), np.float32)
    for k in range(15):
        jt = k + 1
        child = v[_END_TIP[jt]] if jt in _END_TIP else J[jt + 1]
        b = child - J[jt]
        axes[k] = b / max(np.linalg.norm(b), 1e-9)
    return axes


def _procrustes(src, dst):
    """Best rigid R (dst ~= src @ R.T + t), per (V,3) set; rotation only (det +1)."""
    sc, dc = src.mean(0), dst.mean(0)
    H = (src - sc).T @ (dst - dc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


# --------------------------------------------------------------------------- swing-twist clamp
def _swing_twist(q_xyzw, b):
    """quats (N,4 xyzw) about unit axis b -> (twist_angle_signed (N,), swing_rotvec (N,3)).
    q = swing * twist."""
    q = q_xyzw.copy()
    q[q[:, 3] < 0] *= -1.0  # canonicalize w>=0 (double cover) so twist angle in (-pi,pi)
    v, w = q[:, :3], q[:, 3]
    dproj = v @ b
    norm = np.sqrt(dproj * dproj + w * w)
    twist = np.zeros_like(q)
    deg = norm < 1e-7  # pi rotation about an axis perpendicular to b -> twist undefined
    twist[~deg, :3] = (dproj[~deg, None] * b[None]) / norm[~deg, None]
    twist[~deg, 3] = w[~deg] / norm[~deg]
    twist[deg] = np.array([0, 0, 0, 1.0])
    ang = 2.0 * np.arctan2(dproj, w)
    ang[deg] = 0.0
    swing = (Rotation.from_quat(q) * Rotation.from_quat(twist).inv()).as_rotvec()
    return ang, swing


def clamp_swing_twist(pose_aa, bone_axes, twist_mcp_deg=25.0, twist_pip_deg=15.0, swing_deg=135.0):
    """Clamp a (T,45) axis-angle MANO pose. Returns (pose_clamped, stats)."""
    pose_aa = np.asarray(pose_aa, np.float32)
    out = pose_aa.copy()
    tw_lim = np.where(_IS_MCP, twist_mcp_deg, twist_pip_deg)
    sw_lim = np.deg2rad(swing_deg)
    n_tw = n_sw = 0
    for k in range(15):
        q = Rotation.from_rotvec(pose_aa[:, 3 * k : 3 * k + 3]).as_quat()
        b = bone_axes[k]
        ang, sw = _swing_twist(q, b)
        sw_ang = np.linalg.norm(sw, axis=1)
        lim = np.deg2rad(tw_lim[k])
        n_tw += int((np.abs(ang) > lim).sum())
        n_sw += int((sw_ang > sw_lim).sum())
        ang_c = np.clip(ang, -lim, lim)
        scale = np.where(sw_ang > sw_lim, sw_lim / np.maximum(sw_ang, 1e-9), 1.0)
        R_tw = Rotation.from_rotvec(ang_c[:, None] * b[None])
        R_sw = Rotation.from_rotvec(sw * scale[:, None])
        out[:, 3 * k : 3 * k + 3] = (R_sw * R_tw).as_rotvec().astype(np.float32)
    return out, {"n_twist_viol": n_tw, "n_swing_viol": n_sw, "joint_frames": int(pose_aa.shape[0]) * 15}


# --------------------------------------------------------------------------- Karcher smoothing
def _karcher_step(rotvecs, rows_set, radius):
    R = Rotation.from_rotvec(rotvecs)
    out = rotvecs.copy()
    for ti in rows_set:
        lo, hi = max(0, ti - radius), min(len(rotvecs), ti + radius + 1)
        sel = [j for j in range(lo, hi) if j in rows_set]
        rel = (R[ti].inv() * R[sel]).as_rotvec()
        out[ti] = (R[ti] * Rotation.from_rotvec(rel.mean(0))).as_rotvec()
    return out


def smooth_articulation_karcher(pose_aa, valid, radius):
    """Per-joint one-step SO(3) Karcher smoothing of (T,45) over valid frames (drift-free local
    mean — constant-velocity articulation preserved; white finger jitter averaged away)."""
    if radius <= 0:
        return np.asarray(pose_aa, np.float32).copy()
    rows = set(np.flatnonzero(np.asarray(valid).astype(bool)).tolist())
    out = np.asarray(pose_aa, np.float32).copy()
    for j in range(15):
        out[:, 3 * j : 3 * j + 3] = _karcher_step(out[:, 3 * j : 3 * j + 3], rows, radius)
    return out


# --------------------------------------------------------------------------- public entry
def apply_articulation_filter(out, mano, device, *, smooth_radius=1, twist_mcp_deg=25.0, twist_pip_deg=15.0, swing_deg=135.0, min_valid=5):
    """Clamp + Karcher-smooth the finger articulation of the rebuilt npz `out` (mutated in
    place: hand_{side}_verts + hand_{side}_pose). Order: clamp then smooth. Returns a report.

    mano: a MANOHelper (uses its .mano_layer for FK + J_regressor)."""
    layer = mano.mano_layer
    report = {"smooth_radius": smooth_radius, "sides": {}}
    for side in ("left", "right"):
        vk, pk, bk, mk = f"hand_{side}_verts", f"hand_{side}_pose", f"hand_{side}_betas", f"hand_{side}_valid"
        if vk not in out or out[mk].astype(bool).sum() < min_valid:
            continue
        valid = out[mk].astype(bool)
        rows = np.flatnonzero(valid)
        pose = out[pk].astype(np.float32)
        betas = out[bk].astype(np.float32)
        verts = out[vk].astype(np.float32)
        axes = _bone_axes(layer, betas, device)

        pose_new = pose.copy()
        rep = {"n_valid": int(valid.sum())}
        pc, cst = clamp_swing_twist(pose_new[rows], axes, twist_mcp_deg, twist_pip_deg, swing_deg)
        pose_new[rows] = pc
        rep["twist_viol"] = cst["n_twist_viol"]
        rep["swing_viol"] = cst["n_swing_viol"]
        if smooth_radius > 0:
            pose_new = smooth_articulation_karcher(pose_new, valid, smooth_radius)

        # delta-replace verts on valid frames only (Procrustes placement from saved verts)
        tmpl_o = _fk_canonical(layer, pose[rows], betas, side, device)
        tmpl_n = _fk_canonical(layer, pose_new[rows], betas, side, device)
        d_can = tmpl_n - tmpl_o
        vv = verts.astype(np.float64)
        for ii, r in enumerate(rows):
            R = _procrustes(tmpl_o[ii], vv[r])
            vv[r] = vv[r] + d_can[ii] @ R.T
        moved = float(np.linalg.norm((vv[rows] - verts[rows].astype(np.float64)), axis=-1).mean() * 1000)
        out[vk] = vv.astype(np.float32)
        out[pk] = pose_new.astype(np.float32)
        rep["vert_delta_mean_mm"] = round(moved, 3)
        report["sides"][side] = rep
    return report
