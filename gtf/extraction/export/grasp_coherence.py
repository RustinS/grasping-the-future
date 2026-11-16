#!/usr/bin/env python3
"""Grasp coherence (export step 3b): remove the excess hand-object relative-pose drift during a grasp.

During a grasp the hand+object are one rigid body, so the relative pose T_hand^-1·T_obj should
be constant up to slow in-hand adjustment. The pipeline estimates hand (WiLoR per-frame) and
object (SAM3D+BA) semi-independently, so independent per-frame noise makes the relative pose
wobble: the hand drifts on the object.

Method (drift-free, evidence-respecting):
  - detect grasp runs (hand-object min vertex distance < contact_m, length >= min_run);
  - express the hand pose in the object body frame (Q = R_obj_rel^T R_hand_rel, p = offset),
    constant for a rigid grasp; smooth it (one-step Karcher on SO(3) + MA on translation) over a
    short window (not to a constant: real grasps have in-hand motion);
  - the smoothing correction is the wobble to remove; only its time-varying part is applied (the
    smoothing keeps the mean rotation, and the translation correction has its mean subtracted)
    → the average absolute hand pose is preserved (no depth re-anchoring, the grasp's constant
    offset T_grasp_0 untouched: that one is set by image evidence and by contact_projection);
  - apply the mean-removed rigid correction to the hand verts (rotation about the hand centroid
    + centroid shift), tapered at run boundaries. Object + finger articulation untouched (the
    correction is a rigid transform of the whole hand → grasp aperture preserved).

Object is the reference because its BA rotation chain is the smoother channel; the mean-removal
makes the op a pure de-wobbling, so even imperfect object relative-motion only contributes its
time-varying error (small), never a constant re-anchor.

    uv run python -m gtf.extraction.export.grasp_coherence   # synthetic self-test
"""

import numpy as np


def _procrustes_R(A, B):
    """Rotation R s.t. centered A @ R ≈ centered B (no scale, no reflection)."""
    A0, B0 = A - A.mean(0), B - B.mean(0)
    U, _, Vt = np.linalg.svd(A0.T @ B0)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return U @ np.diag([1.0, 1.0, d]) @ Vt


def _log_so3(R):
    """SO(3) -> so(3) axis-angle vector (3,)."""
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    if th < 1e-7:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (th / (2 * np.sin(th)))


def _exp_so3(w):
    """so(3) -> SO(3)."""
    th = np.linalg.norm(w)
    if th < 1e-7:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def _karcher_smooth(Rs, radius):
    """One-step local Karcher smoothing of a rotation sequence (constant-velocity preserved
    exactly in the interior, drift-free). Rs: (n,3,3). Returns smoothed (n,3,3)."""
    n = len(Rs)
    out = np.empty_like(Rs)
    for t in range(n):
        lo, hi = max(0, t - radius), min(n, t + radius + 1)
        tang = np.mean([_log_so3(Rs[t].T @ Rs[j]) for j in range(lo, hi)], axis=0)
        out[t] = Rs[t] @ _exp_so3(tang)
    return out


def _ma(x, radius):
    """Unmasked moving average along axis 0 (window 2*radius+1, truncated at the ends)."""
    n = len(x)
    out = np.empty_like(x)
    for t in range(n):
        lo, hi = max(0, t - radius), min(n, t + radius + 1)
        out[t] = x[lo:hi].mean(0)
    return out


def detect_grasp_runs(hand_v, hand_valid, obj_v, contact_m=0.03, min_run=4, hand_stride=15, obj_max=300):
    """Frame indices grouped into contiguous grasp runs (min hand-obj vertex dist < contact_m)."""
    n = len(hand_v)
    ostride = max(1, obj_v.shape[1] // obj_max)
    contact = np.zeros(n, dtype=bool)
    for t in range(n):
        if not hand_valid[t]:
            continue
        d = np.linalg.norm(hand_v[t][::hand_stride][:, None] - obj_v[t][::ostride][None], axis=2).min()
        contact[t] = d < contact_m
    runs, i = [], 0
    while i < n:
        if contact[i]:
            j = i
            while j < n and contact[j]:
                j += 1
            if j - i >= min_run:
                runs.append(np.arange(i, j))
            i = j
        else:
            i += 1
    return runs


def _taper(n, taper):
    """1.0 in the interior, linearly ramping 0->1 over the first/last `taper` frames."""
    w = np.ones(n)
    k = min(taper, n // 2)
    if k > 0:
        ramp = (np.arange(k) + 1) / (k + 1)
        w[:k] = ramp
        w[-k:] = ramp[::-1]
    return w


def cohere_grasp_run(hand_v, obj_v, rot_radius=2, trans_radius=0, taper=2):
    """De-wobble one grasp run. hand_v/obj_v: (n,Vh,3)/(n,Vo,3) over the run (camera frame).
    Returns corrected hand_v (same shape). Object untouched. Articulation preserved (rigid).

    Rotation-only by default (trans_radius=0): the excess relative drift during a grasp is
    rotational, and smoothing the relative translation as well would over-rigidify the grasp."""
    n = len(hand_v)
    if n < 3:
        return hand_v.copy()
    anchor = n // 2
    Rh = np.stack([_procrustes_R(hand_v[anchor], hand_v[t]) for t in range(n)])  # hand rot rel anchor
    Ro = np.stack([_procrustes_R(obj_v[anchor], obj_v[t]) for t in range(n)])  # object rot rel anchor
    ch = hand_v.mean(1)  # (n,3)
    co = obj_v.mean(1)
    # hand pose in the OBJECT body frame (constant for a rigid grasp)
    Q = np.stack([Ro[t].T @ Rh[t] for t in range(n)])  # (n,3,3)
    p = np.stack([Ro[t].T @ (ch[t] - co[t]) for t in range(n)])  # (n,3)
    Q_sm = _karcher_smooth(Q, rot_radius) if rot_radius > 0 else Q
    p_sm = _ma(p, trans_radius) if trans_radius > 0 else p
    # target hand absolute-rel pose = riding the smoothed relative pose on the object reference.
    # Rotation needs no mean-removal: at the run anchor Rh_new==Rh and the MA/Karcher preserve
    # the means, so the gross motion + average grasp orientation are preserved; only the
    # relative-rotation wobble changes (Q->Q_sm). Translation is mean-corrected (preserve the
    # average absolute hand position → no depth re-anchoring; the constant grasp offset stays).
    Rh_new = np.stack([Ro[t] @ Q_sm[t] for t in range(n)])
    ch_new = np.stack([co[t] + Ro[t] @ p_sm[t] for t in range(n)])
    ch_new = ch_new - (ch_new.mean(0) - ch.mean(0))  # preserve average absolute position
    wt = _taper(n, taper)
    out = np.empty_like(hand_v)
    for t in range(n):
        # right-multiply rigid delta on centered verts: centered @ D gives Procrustes rot Rh@D,
        # so D = Rh^T Rh_new lands the hand at the smoothed orientation. Tapered in log-space.
        D = _exp_so3(_log_so3(Rh[t].T @ Rh_new[t]) * wt[t])
        dct = (ch_new[t] - ch[t]) * wt[t]
        out[t] = (hand_v[t] - ch[t]) @ D + ch[t] + dct
    return out


def apply_grasp_coherence(hand_v, hand_valid, obj_v, contact_m=0.03, rot_radius=2, trans_radius=0, taper=2, min_run=4):
    """Apply grasp-coherence to a full hand-vertex sequence. Returns (corrected_hand_v, report)."""
    out = hand_v.copy()
    runs = detect_grasp_runs(hand_v, hand_valid, obj_v, contact_m=contact_m, min_run=min_run)
    for r in runs:
        out[r] = cohere_grasp_run(hand_v[r], obj_v[r], rot_radius=rot_radius, trans_radius=trans_radius, taper=taper)
    return out, {"n_runs": len(runs), "grasp_frames": int(sum(len(r) for r in runs))}


# ----------------------------------------------------------------------------- synthetic test
def _selftest():
    rng = np.random.default_rng(0)
    n = 40
    # a rigid hand + object moving together: shared body rotation (slow) + translation
    Vh, Vo = 60, 80
    hand0 = rng.normal(scale=0.04, size=(Vh, 3))  # canonical hand cloud
    obj0 = rng.normal(scale=0.05, size=(Vo, 3))
    grasp_off = np.array([0.06, 0.0, 0.0])  # hand sits 6cm from object center
    hand_v, obj_v = [], []
    for t in range(n):
        ang = 0.9 * t * np.pi / 180  # slow shared rotation 0.9 deg/frame about y
        Rb = _exp_so3(np.array([0, ang, 0]))
        tb = np.array([0.002 * t, 0, 0.6])  # slow shared translation
        # small real in-hand adjustment (slow)
        rin = _exp_so3(np.array([0, 0, 0.3 * np.sin(t * 0.3) * np.pi / 180]))
        oc = obj0 @ Rb.T + tb
        hc = (hand0 @ rin.T + grasp_off) @ Rb.T + tb
        obj_v.append(oc)
        hand_v.append(hc)
    hand_v, obj_v = np.stack(hand_v), np.stack(obj_v)

    # inject high-frequency relative wobble onto the hand (the extraction artifact)
    noisy = hand_v.copy()
    for t in range(n):
        wob = _exp_so3(rng.normal(scale=8 * np.pi / 180, size=3))  # ~8deg/frame jitter
        toff = rng.normal(scale=0.02, size=3)  # 20mm jitter
        c = noisy[t].mean(0)
        noisy[t] = (noisy[t] - c) @ wob.T + c + toff

    def rel_drift(hv):
        Rh = [_procrustes_R(hv[n // 2], hv[t]) for t in range(n)]
        Ro = [_procrustes_R(obj_v[n // 2], obj_v[t]) for t in range(n)]
        ch, co = hv.mean(1), obj_v.mean(1)
        Q = [Ro[t].T @ Rh[t] for t in range(n)]
        p = np.stack([Ro[t].T @ (ch[t] - co[t]) for t in range(n)])
        qd = np.array([np.degrees(np.linalg.norm(_log_so3(Q[t] @ Q[n // 2].T))) for t in range(n)])
        return qd.std(), np.linalg.norm(p.std(0)) * 1000

    clean_r, clean_t = rel_drift(hand_v)
    noisy_r, noisy_t = rel_drift(noisy)
    corr, rep = apply_grasp_coherence(noisy, np.ones(n, bool), obj_v, contact_m=0.5, rot_radius=2, trans_radius=2, min_run=4)
    fix_r, fix_t = rel_drift(corr)
    # rotation-only (trans_radius=0) must leave the relative-translation drift ~unchanged
    rot_only, _ = apply_grasp_coherence(noisy, np.ones(n, bool), obj_v, contact_m=0.5, rot_radius=2, trans_radius=0, min_run=4)
    _, rot_only_t = rel_drift(rot_only)
    assert abs(rot_only_t - noisy_t) < 0.20 * noisy_t, f"trans_radius=0 changed translation drift: {noisy_t}->{rot_only_t}"
    # absolute mean preserved (no re-anchoring)
    mean_shift = np.linalg.norm(corr.mean((0, 1)) - noisy.mean((0, 1))) * 1000

    print(f"rel-ROT drift deg:   clean(GT) {clean_r:.1f} | noisy {noisy_r:.1f} | corrected {fix_r:.1f}")
    print(f"rel-TRANS drift mm:  clean(GT) {clean_t:.1f} | noisy {noisy_t:.1f} | corrected {fix_t:.1f}")
    print(f"abs-mean shift (must be ~0): {mean_shift:.2f} mm   | runs={rep['n_runs']} grasp_frames={rep['grasp_frames']}")
    assert fix_r < noisy_r * 0.6, f"rotation drift not removed: {noisy_r}->{fix_r}"
    assert fix_t < noisy_t * 0.7, f"translation drift not removed: {noisy_t}->{fix_t}"
    assert fix_r > clean_r * 0.5, f"over-smoothed below GT floor: {fix_r} vs GT {clean_r}"
    assert mean_shift < 2.0, f"absolute mean not preserved (re-anchored): {mean_shift}mm"
    # articulation preserved: pairwise hand-vertex distances unchanged (rigid correction)
    d_before = np.linalg.norm(noisy[5][:10][:, None] - noisy[5][:10][None], axis=2)
    d_after = np.linalg.norm(corr[5][:10][:, None] - corr[5][:10][None], axis=2)
    assert np.allclose(d_before, d_after, atol=1e-6), "articulation changed (correction not rigid)"
    print("articulation preserved (rigid): OK")
    print("\ngrasp-coherence self-tests pass")


if __name__ == "__main__":
    _selftest()
