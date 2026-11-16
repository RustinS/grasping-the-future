"""Smoother fitter for the exported hand + object trajectory (export step 7b, chains with smooth=True).

A batch MAP smoother (the fixed-interval Kalman / RTS smoother written as one least-squares fit, so it can carry priors
and couplings): the per-keyframe hand MANO parameters (wrist orientation, 45 finger joint rotations, translation) and the
object pose (rotation, translation) are re-fitted to

  data      stay near the exported values, with anisotropic position noise: a keyframe constrains position across the
            image (sigma_lat = sig_lat x depth) much more than along its own viewing ray (ray_ratio x sigma_lat), since
            single-view depth is where the noise is; rotations / finger angles: sig_rot, sig_pose
  smooth    white-acceleration process: sum |a_t|^2 dt over consecutive valid keyframes, on the exact keyframe times,
            for wrist position, wrist and object rotation (angular acceleration on SO(3)), finger angles, object centroid
  grasp     on grasp keyframes (hand within contact_m of the object for >= min_run keyframes) the wrist position and
            orientation in the object frame are smoothed too, so smoothing does not make a held object slide in the hand
  prior     MANO natural-pose prior, component-wise excess over the input pose (gtf/model/repr/mano_pose_prior.py
            semantics): a pose may not become less natural than the one it started from

With per-keyframe cameras (cam_E), the smoothness terms act on the trajectories as each keyframe's own camera sees them
(the parameters stay in the anchor frame): the per-keyframe cameras jitter, and smoothing the anchor-frame trajectory
would remove that camera noise from the world track only to put it back, reversed, on screen.
Strengths are per channel (s_*). Every strength defaults to 0 (off); the export passes CALIBRATED.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import numpy as np

DEFAULTS = dict(
    sig_lat=0.008,  # lateral position noise / depth (~1% of the image width at f = 1.25 W)
    ray_ratio=4.0,  # along-ray / lateral position noise
    sig_rot_hand=0.05,  # rad
    sig_rot_obj=0.03,  # rad
    sig_pose=0.08,  # rad per finger component
    s_hand_pos=0.0,  # smoothing strengths (0 = off); calibrated values: CALIBRATED below
    s_hand_rot=0.0,
    s_pose=0.0,
    s_obj_pos=0.0,
    s_obj_rot=0.0,
    s_grasp=0.0,
    w_prior=1.0,
    contact_m=0.02,
    min_run=3,
    iters=60,
)


# Strengths used by the export (chains with smooth=True): they bring the jitter of the hand depth, fingers, hand and
# object down to the level of real motion; much stronger settings damp real motion.
CALIBRATED = dict(s_hand_pos=10.0, s_hand_rot=1.0, s_pose=0.25, s_obj_pos=10.0, s_obj_rot=0.3, s_grasp=10.0)


@dataclass
class SmoothInputs:
    times: np.ndarray  # (T,) seconds
    obj_rotation: np.ndarray  # (T,3) rotvec: verts = R @ src + t
    obj_translation: np.ndarray  # (T,3)
    obj_source: np.ndarray  # (V,3) canonical object verts x scale
    hands: dict = field(default_factory=dict)  # side -> dict(valid (T,), orient (T,3), pose (T,45), cam_t (T,3), betas (10,))
    cam_centers: np.ndarray | None = None  # (T,3) camera centres in the same (anchor) frame; None = camera at the origin
    cam_E: np.ndarray | None = None  # (T,4,4) anchor frame -> each keyframe's own camera; None = smooth in the anchor frame


def _mano_layer(mano_dir):
    from gtf.utils.canonical import _import_manolayer

    ManoLayer = _import_manolayer()
    return ManoLayer(side="right", use_pca=False, ncomps=45, flat_hand_mean=True, mano_root=str(mano_dir))


class _Prior:
    """MANO natural-pose prior: standardized PCA coefficient energy, component-wise excess over a reference pose."""

    def __init__(self, mano_dir, torch):
        with open(f"{mano_dir}/MANO_RIGHT.pkl", "rb") as fh:
            d = pickle.load(fh, encoding="latin1")
        self.mean = torch.tensor(np.asarray(d["hands_mean"]), dtype=torch.float64)
        self.comp = torch.tensor(np.asarray(d["hands_components"]), dtype=torch.float64)
        self.nsq = self.comp.square().sum(-1)

    def energy(self, pose):
        return ((pose - self.mean) @ self.comp.T / self.nsq).square()

    def excess(self, pose, ref):
        e, r = self.energy(pose), self.energy(ref).detach()
        d = e - r
        return (d * (d > 0).to(d.dtype) / r.clamp(min=1.0)).mean(-1)


def _accel_penalty(x, t, mask):
    """sum |a|^2 * dt over interior keyframes of runs of valid keyframes; x (T,D) tensor, t (T,) tensor, mask (T,) bool."""
    if x.shape[0] < 3:
        return x.sum() * 0
    dt0, dt1 = (t[1:-1] - t[:-2]).clamp(min=1e-3), (t[2:] - t[1:-1]).clamp(min=1e-3)
    v0 = (x[1:-1] - x[:-2]) / dt0[:, None]
    v1 = (x[2:] - x[1:-1]) / dt1[:, None]
    dtm = 0.5 * (dt0 + dt1)
    a = (v1 - v0) / dtm[:, None]
    m = (mask[:-2] & mask[1:-1] & mask[2:]).to(x.dtype)
    return (a.square().sum(-1) * dtm * m).sum()


def _ang_accel_penalty(R, t, mask, roma):
    """angular acceleration on SO(3): body angular velocity log(R_t^T R_t+1) / dt, then its acceleration."""
    if R.shape[0] < 3:
        return R.sum() * 0
    dt = (t[1:] - t[:-1]).clamp(min=1e-3)
    w = roma.rotmat_to_rotvec(R[:-1].transpose(1, 2) @ R[1:]) / dt[:, None]
    dtm = 0.5 * (dt[1:] + dt[:-1])
    a = (w[1:] - w[:-1]) / dtm[:, None]
    m = (mask[:-2] & mask[1:-1] & mask[2:]).to(R.dtype)
    return (a.square().sum(-1) * dtm * m).sum()


def _grasp_mask(hand_v, valid, obj_v, contact_m, min_run):
    """(T,) bool: keyframes in runs (>= min_run) where the hand is within contact_m of the object"""
    T = len(valid)
    near = np.zeros(T, bool)
    for t in range(T):
        if valid[t]:
            h = hand_v[t][::8]
            o = obj_v[t][:: max(1, len(obj_v[t]) // 1500)]
            near[t] = float(np.sqrt(((h[:, None] - o[None]) ** 2).sum(-1)).min()) < contact_m
    out = np.zeros(T, bool)
    s = None
    for t, v in enumerate(list(near) + [False]):
        if v and s is None:
            s = t
        if not v and s is not None:
            if t - s >= min_run:
                out[s:t] = True
            s = None
    return out


def smooth(inp: SmoothInputs, mano_dir, cfg=None, device="cpu"):
    """Returns (dict with obj_rotation, obj_translation, hands {side: orient, pose, cam_t}, report)."""
    import roma
    import torch

    c = dict(DEFAULTS)
    c.update(cfg or {})
    dt = torch.float64
    T = len(inp.times)
    t = torch.tensor(inp.times, dtype=dt, device=device)
    layer = _mano_layer(mano_dir).to(device).double()
    prior = _Prior(mano_dir, torch)
    prior.mean, prior.comp, prior.nsq = prior.mean.to(device), prior.comp.to(device), prior.nsq.to(device)
    C = torch.zeros(T, 3, dtype=dt, device=device) if inp.cam_centers is None else torch.tensor(inp.cam_centers, dtype=dt, device=device)
    mirror = torch.tensor([-1.0, 1.0, 1.0], dtype=dt, device=device)

    # ---- object
    R_o0 = roma.rotvec_to_rotmat(torch.tensor(inp.obj_rotation, dtype=dt, device=device))
    t_o0 = torch.tensor(inp.obj_translation, dtype=dt, device=device)
    src = torch.tensor(inp.obj_source, dtype=dt, device=device)
    m_src = src.mean(0)
    obj_ok = torch.ones(T, dtype=torch.bool, device=device)
    d_or = torch.zeros(T, 3, dtype=dt, device=device, requires_grad=True)
    d_ot = torch.zeros(T, 3, dtype=dt, device=device, requires_grad=True)
    c_o0 = (R_o0 @ m_src) + t_o0

    # ---- hands
    H = {}
    obj_v_np = np.einsum("tij,vj->tvi", R_o0.cpu().numpy(), inp.obj_source) + inp.obj_translation[:, None]
    for side, h in inp.hands.items():
        valid = np.asarray(h["valid"], bool)
        if valid.sum() < 3:
            continue
        go0 = torch.tensor(h["orient"], dtype=dt, device=device)
        th0 = torch.tensor(h["pose"], dtype=dt, device=device)
        ct0 = torch.tensor(h["cam_t"], dtype=dt, device=device)
        be = torch.tensor(h["betas"], dtype=dt, device=device)[None].expand(T, -1)
        with torch.no_grad():
            v0, j0 = layer(torch.cat([go0, th0], 1), be)
        v0, j0 = v0 / 1000 + ct0[:, None], j0 / 1000 + ct0[:, None]
        if side == "left":
            v0, j0 = v0 * mirror, j0 * mirror
        vw = v0.cpu().numpy()
        grasp = _grasp_mask(vw, valid, obj_v_np, c["contact_m"], c["min_run"]) if c["s_grasp"] > 0 else np.zeros(T, bool)
        H[side] = dict(
            valid=torch.tensor(valid, device=device),
            grasp=torch.tensor(grasp & valid, device=device),
            R0=roma.rotvec_to_rotmat(go0),
            th0=th0,
            ct0=ct0,
            be=be,
            w0=j0[:, 0],
            d_r=torch.zeros(T, 3, dtype=dt, device=device, requires_grad=True),
            d_th=torch.zeros(T, 45, dtype=dt, device=device, requires_grad=True),
            d_ct=torch.zeros(T, 3, dtype=dt, device=device, requires_grad=True),
        )

    def aniso(p, p0, valid):
        e = p - p0
        ray = p0 - C
        depth = ray.norm(dim=-1).clamp(min=0.05)
        u = ray / depth[:, None]
        par = (e * u).sum(-1)
        perp = e - par[:, None] * u
        sl = c["sig_lat"] * depth
        r = perp.square().sum(-1) / sl.square() + par.square() / (c["ray_ratio"] * sl).square()
        return (r * valid.to(dt)).sum()

    cam_frame = inp.cam_E is not None
    if cam_frame:
        E_R = torch.tensor(inp.cam_E[:, :3, :3], dtype=dt, device=device)
        E_t = torch.tensor(inp.cam_E[:, :3, 3], dtype=dt, device=device)
    Mm = torch.diag(mirror)

    def to_s(x):  # positions (T,3) as smoothed: own camera or anchor frame
        return (E_R @ x[..., None])[..., 0] + E_t if cam_frame else x

    def rot_s(R):
        return E_R @ R if cam_frame else R

    def evaluate():
        R_o = R_o0 @ roma.rotvec_to_rotmat(d_or)
        t_o = t_o0 + d_ot
        c_o = (R_o @ m_src) + t_o
        L = {}
        L["obj_data"] = aniso(c_o, c_o0, obj_ok) + (d_or.square().sum(-1) / c["sig_rot_obj"] ** 2).sum()
        L["obj_smooth"] = c["s_obj_pos"] * _accel_penalty(to_s(c_o), t, obj_ok) + c["s_obj_rot"] * _ang_accel_penalty(rot_s(R_o), t, obj_ok, roma)
        outs = {}
        for side, h in H.items():
            R = h["R0"] @ roma.rotvec_to_rotmat(h["d_r"])
            go = roma.rotmat_to_rotvec(R)
            th = h["th0"] + h["d_th"]
            ct = h["ct0"] + h["d_ct"]
            _, j = layer(torch.cat([go, th], 1), h["be"])
            w = j[:, 0] / 1000 + ct
            if side == "left":
                w = w * mirror
            v = h["valid"]
            vf = v.to(dt)
            L[f"{side}_data"] = (
                aniso(w, h["w0"], v) + (h["d_r"].square().sum(-1) / c["sig_rot_hand"] ** 2 * vf).sum() + (h["d_th"].square().sum(-1) / c["sig_pose"] ** 2 * vf).sum()
            )
            R_w = R if side == "right" else Mm @ R @ Mm  # wrist orientation in the anchor frame (left params are mirrored)
            L[f"{side}_smooth"] = (
                c["s_hand_pos"] * _accel_penalty(to_s(w), t, v) + c["s_hand_rot"] * _ang_accel_penalty(rot_s(R_w), t, v, roma) + c["s_pose"] * _accel_penalty(th, t, v)
            )
            if c["w_prior"] > 0:
                L[f"{side}_prior"] = c["w_prior"] * (prior.excess(th, h["th0"]) * vf).sum()
            if c["s_grasp"] > 0 and bool(h["grasp"].any()):
                # wrist position and orientation in the object frame (left in mirrored coordinates: mirror the object too)
                Ro = R_o if side == "right" else Mm @ R_o @ Mm
                co = c_o if side == "right" else c_o * mirror
                wp = w if side == "right" else w * mirror
                rel = (Ro.transpose(1, 2) @ (wp - co)[..., None])[..., 0]
                L[f"{side}_grasp"] = c["s_grasp"] * (_accel_penalty(rel, t, h["grasp"]) + 0.1 * _ang_accel_penalty(Ro.transpose(1, 2) @ R, t, h["grasp"], roma))
            outs[side] = (go, th, ct)
        return L, R_o, t_o, outs

    params = [d_or, d_ot] + [p for h in H.values() for p in (h["d_r"], h["d_th"], h["d_ct"])]
    with torch.no_grad():
        L0, *_ = evaluate()
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=c["iters"], history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        L, *_ = evaluate()
        tot = sum(L.values())
        tot.backward()
        return tot

    if any(c[k] > 0 for k in ("s_hand_pos", "s_hand_rot", "s_pose", "s_obj_pos", "s_obj_rot", "s_grasp")):
        opt.step(closure)
    with torch.no_grad():
        L1, R_o, t_o, outs = evaluate()
    res = {
        "obj_rotation": roma.rotmat_to_rotvec(R_o).cpu().numpy().astype(np.float32),
        "obj_translation": t_o.cpu().numpy().astype(np.float32),
        "hands": {},
    }
    rep = {"loss_before": {k: float(v) for k, v in L0.items()}, "loss_after": {k: float(v) for k, v in L1.items()}}
    for side, (go, th, ct) in outs.items():
        v = H[side]["valid"].cpu().numpy()
        go_n, th_n, ct_n = go.cpu().numpy(), th.cpu().numpy(), ct.cpu().numpy()
        go_n[~v], th_n[~v], ct_n[~v] = inp.hands[side]["orient"][~v], inp.hands[side]["pose"][~v], inp.hands[side]["cam_t"][~v]
        res["hands"][side] = dict(orient=go_n.astype(np.float32), pose=th_n.astype(np.float32), cam_t=ct_n.astype(np.float32))
        rep[f"{side}_grasp_kf"] = int(H[side]["grasp"].sum())
    rep["cfg"] = {k: c[k] for k in DEFAULTS}
    return res, rep
