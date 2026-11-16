#!/usr/bin/env python3
"""Depth-from-size stage-2 repair (refine_trajectory.py step 5).

Fixes scale-coupled object depth wander: the optimized chain carries a per-keyframe object
scale (obj_scale_init; the 3D size can vary several-fold within one segment, and size and depth
are correlated), so scale and depth co-wander in the silhouette's null direction (s/z
projective ambiguity) and no per-frame image evidence can see it. Two atomic components:

  A. Rigidify: scale every keyframe's object about its own centroid to the median size.
     A rigid object is unconditionally correct; this converts the hidden scale artifact
     into visible mask/sil ratio evidence.
  B. Ratio-flatten: apparent size is the one depth signal the video constrains over time
     (area ~ footprint/z^2; the reprojected sil shares the footprint, so
     r = mask_area/sil_area ~ (z_claimed/z_true)^2). Iterated local-linear kernel fit over
     the keyframe index -> smooth along-ray translation c_k with mean(log c) = 0 (the absolute
     depth level is unobservable and stays untouched). For a rigid object z*sqrt(area) is
     constant under any genuine motion, so B cannot fight real carries.

Guards:
  - min_size_fold=1.35 scope gate: B on already-rigid chains converts footprint/mask noise
    into depth error -> rigid chains abstain.
  - reduce-only: the correction must shrink the claimed depth log-variance (rejects masks
    that "breathe", e.g. on transparent glass).
  - reliability sampling: occlusion < 0.5 (hand-hull), border frames only when mask+sil
    clip together, 0.25-of-median size floor, winsorize, min samples/span.
  - hands are rigidly carried via a per-run anchor vertex (nearest object vertex at the
    run middle): the touch point is preserved exactly through rescale + depth move;
    linear taper outside runs; zero-filled invalid frames stay zero.

Runs on the stage-2 chain before the export's object refiner (called at the end of
refine_trajectory.refine_one_segment; the refined chain's ratio<->depth relation is
confounded by the refiner).
"""

from pathlib import Path

import numpy as np

from gtf.extraction.gates.interaction_metrics import _proj2d, _silhouette, detect_grasp_runs


def _border_touch(m, rim=2):
    return bool(m[:rim].any() or m[-rim:].any() or m[:, :rim].any() or m[:, -rim:].any())


def _kernel_smooth(x_all, x_s, y_s, w_s, bandwidth):
    """Local-linear Gaussian kernel regression of (x_s, y_s, w_s) evaluated at x_all.
    Local-linear (not Nadaraya-Watson) because the signal is a trend: NW's boundary/trend
    bias under-recovers the slow whole-segment drift the repair targets."""
    x_all = np.asarray(x_all, np.float64)
    dx = x_all[:, None] - np.asarray(x_s, np.float64)[None]
    kw = np.exp(-0.5 * (dx / max(bandwidth, 1e-9)) ** 2) * np.asarray(w_s, np.float64)[None]
    y = np.asarray(y_s, np.float64)[None]
    s0, s1, s2 = kw.sum(1), (kw * dx).sum(1), (kw * dx * dx).sum(1)
    t0, t1 = (kw * y).sum(1), (kw * dx * y).sum(1)
    den = s0 * s2 - s1 * s1
    nw = t0 / np.maximum(s0, 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        ll = (s2 * t0 - s1 * t1) / den
    return np.where(np.abs(den) > 1e-9 * np.maximum(s0, 1e-12) ** 2, ll, nw)


def measure_ratio_series(d, masks, K_list, frame_indices):
    """Per-keyframe (log mask/sil area ratio, weight) + centroid z. A keyframe is a sample when
    the mask has >= 30 px, the projected sil >= 100 px and the hands' hulls cover < half the
    sil. A border-touching mask counts only when the sil touches the border too (a truncated
    mask against a full sil biases the ratio), at weight 0.5. Samples whose sqrt(mask area) is
    below 0.25 of the median are dropped."""
    obj_v = np.asarray(d["obj_verts"], np.float64)
    faces = np.asarray(d["obj_faces"]) if "obj_faces" in d else None
    ki = np.asarray(frame_indices, int)
    H, W = int(masks.shape[1]), int(masks.shape[2])
    T = len(ki)

    hands = {}
    for side in ("left", "right"):
        vk, hk = f"hand_{side}_valid", f"hand_{side}_verts"
        if vk in d and hk in d:
            hands[side] = (np.asarray(d[hk], np.float64), np.asarray(d[vk]).astype(bool))

    log_r = np.full(T, np.nan)
    weight = np.zeros(T)
    sqrt_ma = np.full(T, np.nan)
    z = obj_v[:, :, 2].mean(axis=1)
    if faces is None:
        return log_r, weight, z

    import cv2

    for idx in range(T):
        fi = int(ki[idx])
        if fi < 0 or fi >= len(masks):
            continue
        mk = masks[fi].astype(bool)
        ma = float(mk.sum())
        if ma < 30:
            continue
        K = K_list[idx]
        sil = _silhouette(obj_v[idx], faces, K, H, W)
        if sil is None:
            continue
        sa = float(sil.sum())
        if sa < 100:
            continue
        hand_sil = np.zeros((H, W), np.uint8)
        for _side, (hv, hval) in hands.items():
            if hval[idx]:
                p, ok = _proj2d(hv[idx], K)
                if ok.sum() >= 3:
                    hull = cv2.convexHull(np.clip(p[ok], [-1e4, -1e4], [1e4, 1e4]).astype(np.int32))
                    cv2.fillConvexPoly(hand_sil, hull, 1)
        occ = float((sil.astype(bool) & hand_sil.astype(bool)).sum()) / sa
        if occ >= 0.5:
            continue
        mb, sb = _border_touch(mk), _border_touch(sil.astype(bool))
        if mb and not sb:
            continue  # truncated mask vs full sil: ratio biased by the truncation amount
        log_r[idx] = np.log(ma / sa)
        weight[idx] = 0.5 if mb else 1.0
        sqrt_ma[idx] = np.sqrt(ma)

    got = np.isfinite(sqrt_ma)
    if got.any():
        floor = 0.25 * np.median(sqrt_ma[got])
        bad = got & (sqrt_ma < floor)
        log_r[bad] = np.nan
        weight[bad] = 0.0
    return log_r, weight, z


def fit_depth_correction(
    log_r,
    weight,
    z,
    fit_axis,
    bandwidth_frac=0.02,
    min_bandwidth_steps=3.0,
    cap=2.5,
    shrink=1.0,
    min_samples=8,
    min_span_frac=0.45,
    min_var_reduction=0.02,
):
    """Per-keyframe along-ray factor c (mean log c = 0) from the measured ratio series.
    Returns (c, report). c = ones + reason on abstention."""
    ki = np.asarray(fit_axis, np.float64)
    T = len(ki)
    c = np.ones(T)
    ok = (weight > 0) & np.isfinite(log_r)
    rep = {"applied": False, "reason": None, "n_reliable": int(ok.sum())}
    if ok.sum() < min_samples:
        rep["reason"] = "too_few_reliable_samples"
        return c, rep
    span = ki[-1] - ki[0]
    s_span = ki[ok].max() - ki[ok].min()
    rep["span_frac"] = float(s_span / max(span, 1e-9))
    if span <= 0 or s_span < min_span_frac * span:
        rep["reason"] = "reliable_span_too_short"
        return c, rep

    lr = log_r.copy()
    med = float(np.median(lr[ok]))
    lr[ok] = np.clip(lr[ok], med - np.log(4.0), med + np.log(4.0))

    steps = np.diff(ki[ok])
    med_step = float(np.median(steps)) if len(steps) else 1.0
    bw = max(min_bandwidth_steps * med_step, bandwidth_frac * span)
    rep["bandwidth"] = float(bw)
    l_hat = _kernel_smooth(ki, ki[ok], lr[ok], weight[ok], bw)
    log_c = 0.5 * (l_hat.mean() - l_hat)  # mean(log c) = 0 -> depth LEVEL preserved
    log_c = np.clip(shrink * log_c, -np.log(cap), np.log(cap))

    zv = np.asarray(z, np.float64)
    z_ok = np.isfinite(zv) & (zv > 0.01)
    rep["max_abs_logc"] = float(np.abs(log_c).max())
    if z_ok.sum() >= 3:
        s_before = float(np.std(np.log(zv[z_ok])))
        s_after = float(np.std(np.log(zv[z_ok]) + log_c[z_ok]))
        rep["std_logz_before"], rep["std_logz_after"] = s_before, s_after
        if s_after > (1.0 - min_var_reduction) * s_before:
            rep["reason"] = "would_not_reduce_depth_motion"
            return c, rep
    rep["applied"] = True
    return np.exp(log_c), rep


def measure_size_series(obj_verts):
    """Per-keyframe object 3D size (RMS distance to centroid) + the centroid-rigidify
    factor c_A = median_size / size_k. A rigid object has c_A == 1 everywhere."""
    ov = np.asarray(obj_verts, np.float64)
    cent = ov.mean(axis=1)
    size = np.sqrt(((ov - cent[:, None, :]) ** 2).sum(-1).mean(1))
    s_bar = float(np.median(size))
    c_a = s_bar / np.maximum(size, 1e-9)
    return c_a, size, s_bar


def apply_object_repair(obj_verts, c_a, c_b):
    """obj'' = c_B * centroid + c_A * (obj - centroid): centroid-rigidify by c_A (size fix,
    centroid unchanged) + along-ray centroid translation by c_B (depth fix)."""
    ov = np.asarray(obj_verts, np.float64)
    cent = ov.mean(axis=1)
    return (np.asarray(c_b, np.float64)[:, None] * cent)[:, None, :] + np.asarray(c_a, np.float64)[:, None, None] * (ov - cent[:, None, :])


def carry_hands(d, obj_before, obj_after, contact_m=0.05, min_run=3, taper_kf=2):
    """Rigidly carry in-contact hands by the repair displacement of a per-run anchor vertex
    (the object vertex nearest the hand at the run's middle keyframe): the touch point is
    preserved exactly through both rescale and depth move. Linear taper outside runs;
    zero-filled invalid hand frames stay exactly zero (mask-zero invariant)."""
    T = len(obj_before)
    out = {"carry_w": {}}
    for side in ("left", "right"):
        vk, hk = f"hand_{side}_valid", f"hand_{side}_verts"
        if vk not in d or hk not in d:
            continue
        hv = np.asarray(d[hk], np.float64).copy()
        val = np.asarray(d[vk]).astype(bool)
        runs = detect_grasp_runs(hv, val, obj_before, contact_m=contact_m, min_run=min_run)
        w = np.zeros(T)
        delta = np.zeros((T, 3))
        for r in runs:
            mid = r[len(r) // 2]
            dm = np.linalg.norm(obj_before[mid][None] - hv[mid][::15][:, None], axis=2)
            v_star = int(dm.min(axis=0).argmin())
            run_delta = obj_after[:, v_star] - obj_before[:, v_star]
            lo, hi = int(r[0]), int(r[-1])
            w[r] = 1.0
            delta[r] = run_delta[r]
            for k in range(1, taper_kf + 1):
                fade = 1.0 - k / (taper_kf + 1.0)
                for t in (lo - k, hi + k):
                    if 0 <= t < T and fade > w[t]:
                        w[t] = fade
                        delta[t] = run_delta[t]
        w = w * val  # never move zero-filled invalid frames
        hv += (w[:, None] * delta)[:, None, :]
        out[hk] = hv
        out["carry_w"][side] = w
        out[f"n_runs_{side}"] = len(runs)
    return out


def repair_segment_chain(d, masks, K_list, frame_indices, contact_m=0.05, n_iter=4, cap=3.0, min_size_fold=1.35, **fit_kw):
    """Full repair on one chain dict. Returns (d_fixed, report). d_fixed is a shallow copy
    of d with obj_verts / carried hand verts replaced; report carries c_a/c_b + diagnostics.

    Atomic: if B's first pass abstains, A is reverted too. The composed correction is
    applied once to the original chain (exact carry, one taper); B is fit over the keyframe
    index (smooth across video-frame gaps) and iterated (kernel attenuation compensation),
    pass 1 gated strictly, later passes refine; the composed total re-checks reduce-only."""
    fi = np.asarray(frame_indices)
    T = len(fi)
    obj0 = np.asarray(d["obj_verts"], np.float64)
    c_a, size, s_bar = measure_size_series(obj0)
    size_fold = float(np.percentile(size, 95) / max(np.percentile(size, 5), 1e-9))
    if size_fold < min_size_fold:
        return dict(d), {"applied": False, "reason": "no_scale_artifact", "size_fold": size_fold, "c_a": np.ones(T), "c_b": np.ones(T)}

    log_c_b = np.zeros(T)
    work_obj = apply_object_repair(obj0, c_a, np.ones(T))
    d_work = dict(d)
    rep = None
    fit_axis = np.arange(T, dtype=np.float64)
    strict = dict(fit_kw)
    relaxed = dict(fit_kw)
    relaxed["min_var_reduction"] = -0.05
    for it in range(max(1, n_iter)):
        d_work["obj_verts"] = work_obj
        log_r, weight, z = measure_ratio_series(d_work, masks, K_list, frame_indices)
        c_it, rep_it = fit_depth_correction(log_r, weight, z, fit_axis, cap=cap, **(strict if it == 0 else relaxed))
        if it == 0:
            rep = rep_it
            if not rep_it["applied"]:
                rep.update({"c_a": np.ones(T), "c_b": np.ones(T), "size_fold": size_fold})
                return dict(d), rep
        elif not rep_it["applied"]:
            break
        log_c_b += np.log(c_it)
        rep["n_iters_applied"] = it + 1
        if it + 1 < max(1, n_iter):
            cent_w = work_obj.mean(axis=1)
            work_obj = work_obj + ((c_it - 1.0)[:, None] * cent_w)[:, None, :]

    log_c_b = np.clip(log_c_b, -np.log(cap), np.log(cap))
    rep["size_fold"] = size_fold
    rep["c_a"], rep["c_b"] = c_a, np.exp(log_c_b)
    rep["max_abs_logca"] = float(np.abs(np.log(c_a)).max())
    rep["max_abs_logcb"] = float(np.abs(log_c_b).max())
    z0 = obj0[:, :, 2].mean(axis=1)
    z_ok = np.isfinite(z0) & (z0 > 0.01)
    if z_ok.sum() >= 3:
        s_b, s_a = float(np.std(np.log(z0[z_ok]))), float(np.std(np.log(z0[z_ok]) + log_c_b[z_ok]))
        rep["std_logz_before"], rep["std_logz_after"] = s_b, s_a
        if s_a > s_b:
            rep["applied"] = False
            rep["reason"] = "composed_would_not_reduce_depth_motion"
            rep["c_a"], rep["c_b"] = np.ones(T), np.ones(T)
            return dict(d), rep

    d_fixed = dict(d)
    obj_fixed = apply_object_repair(obj0, c_a, np.exp(log_c_b))
    d_fixed["obj_verts"] = obj_fixed
    ch = carry_hands(d, obj0, obj_fixed, contact_m=contact_m)
    for side in ("left", "right"):
        hk = f"hand_{side}_verts"
        if hk in ch:
            d_fixed[hk] = ch[hk]
    cent_move = np.linalg.norm(obj_fixed.mean(axis=1) - obj0.mean(axis=1), axis=1)
    rep["delta_mm_max"] = float(cent_move.max() * 1000.0)
    rep["carry_frames"] = {s: int((w > 0).sum()) for s, w in ch["carry_w"].items()}
    return d_fixed, rep


def load_masks_and_K(seg_dir, frame_indices):
    """(masks, K_list) for the repair from a segment dir; None if inputs missing."""
    seg_dir = Path(seg_dir)
    mask_p = seg_dir / "SAM3" / "masks.npz"
    if not mask_p.exists():
        return None
    with np.load(mask_p, allow_pickle=False) as archive:
        masks = archive["masks"]

    from gtf.utils import load_moge_intrinsics

    fi = np.asarray(frame_indices, int)
    K_default = load_moge_intrinsics(str(seg_dir), int(fi[0]))
    if K_default is None:
        for f in fi[1:10]:
            K_default = load_moge_intrinsics(str(seg_dir), int(f))
            if K_default is not None:
                break
    if K_default is None:
        return None
    K_list = []
    for f in fi:
        Kf = load_moge_intrinsics(str(seg_dir), int(f))
        K_list.append(Kf if Kf is not None else K_default)
    return masks, K_list


def repair_stage2_inplace(out, seg_dir):
    """Entry point called by refine_trajectory: repair the in-memory stage-2 dict `out` (obj_verts +
    hand_*_verts replaced in place when applied). Returns a JSON-safe report block."""
    inputs = load_masks_and_K(seg_dir, out["frame_indices"])
    if inputs is None:
        return {"applied": False, "reason": "no_masks_or_K"}
    masks, K_list = inputs
    d_fixed, rep = repair_segment_chain(out, masks, K_list, out["frame_indices"])
    if rep.get("applied"):
        out["obj_verts"] = d_fixed["obj_verts"].astype(out["obj_verts"].dtype)
        for side in ("left", "right"):
            hk = f"hand_{side}_verts"
            if hk in d_fixed and hk in out:
                out[hk] = d_fixed[hk].astype(out[hk].dtype)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rep.items() if k not in ("c_a", "c_b") and not isinstance(v, np.ndarray)}
