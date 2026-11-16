#!/usr/bin/env python3
"""Interaction-fidelity metrics for the quality gate: video-grounded, image-space, depth-invariant.

They catch the HOI failures the physics/image gates (1-13) are structurally blind to: utensil or
false-positive "hand" tracks and ghost (off-object) contact (e.g. a stirring spoon tracked as a
hand, a clip without manipulation, a grasp that does not touch the object).

  contact_cover  fraction of the 3D-contact hand verts (within eps of the object) whose 2D
                 projection lands inside the object mask: interaction consistency (the optimizer
                 never jointly fits the hand to the mask). low = ghost manipulation.  → gate 15
  rot_jerk_deg   hand<->object 2D relative-rotation high-freq jerk over the grasp run: the image-
                 space counterpart of the 3D relative-rotation drift. high = utensil/unstable track. → gate 14
  obj_centroid   projected-object centroid vs the mask centroid (%W) — object 2D placement.  soft
  obj_orient_deg projected-object 2nd-moment PCA axis vs the mask (elongated only).         soft/obs
  hand_drift_pctW optimized hand 2D centroid vs the WiLoR image-fit hand (the optimizer's own
                 evidence): hand off the video.  soft

The two hard gates are `rot_jerk` and `contact_cover` (`hand_drift`/`obj_orient` over-flag good
pours/grasps, so they are soft signals only). `interaction_fidelity_score` folds the soft axes into
a continuous [floor,1] sample weight for gate-passing segments (build_manifest → dataset
extractor_score). Pure numpy/cv2 (torch only lazily, for the WiLoR hand).
"""

import numpy as np

from gtf.extraction.export.grasp_coherence import detect_grasp_runs


def contact_cover(hand_v, obj_v, K, mask, eps=0.015):
    """Fraction of the 3D-contact hand verts (within eps of the object) projecting inside the mask.
    Returns None if no 3D contact this frame (defer to a coarser interaction check)."""
    hd = np.asarray(hand_v, np.float64)
    ov = np.asarray(obj_v, np.float64)
    ov = ov[:: max(1, len(ov) // 400)]
    dmin = np.sqrt(((hd[:, None] - ov[None]) ** 2).sum(-1)).min(1)
    cverts = hd[(dmin < eps) & (hd[:, 2] > 1e-6)]
    if len(cverts) < 3:
        return None
    p = np.stack([K[0, 0] * cverts[:, 0] / cverts[:, 2] + K[0, 2], K[1, 1] * cverts[:, 1] / cverts[:, 2] + K[1, 2]], 1)
    u = np.round(p[:, 0]).astype(int)
    v = np.round(p[:, 1]).astype(int)
    H, W = mask.shape
    ib = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if ib.sum() == 0:
        return 0.0
    return float(mask[v[ib], u[ib]].astype(bool).mean())


def _proj2d(v, K):
    """(N,3) camera-frame verts -> (N,2) pixels (NaN where z<=0) + ok mask."""
    v = np.asarray(v, np.float64)
    z = v[:, 2]
    ok = z > 1e-6
    zc = np.where(ok, z, 1.0)
    p = np.empty((len(v), 2))
    p[:, 0] = K[0, 0] * v[:, 0] / zc + K[0, 2]
    p[:, 1] = K[1, 1] * v[:, 1] / zc + K[1, 2]
    p[~ok] = np.nan
    return p, ok


def _rot2d(a, b):
    """Apparent 2D rotation (deg) aligning corresponded point set a -> b (rigid, no scale)."""
    a = a - a.mean(0)
    b = b - b.mean(0)
    M = b.T @ a  # 2x2
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U = U.copy()
        U[:, -1] *= -1
        R = U @ Vt
    return np.degrees(np.arctan2(R[1, 0], R[0, 0]))


def temporal_jerk_2d(hand_v, obj_v, K_list, valid, contact_m=0.04, min_run=4):
    """2D relative-rotation high-freq jerk over grasp runs. hand_v/obj_v: (n,Vh/Vo,3) camera frame;
    K_list: per-frame (3,3); valid: (n,) hand validity. Returns None if no run."""
    runs = detect_grasp_runs(hand_v, valid, obj_v, contact_m=contact_m, min_run=min_run)
    if not runs:
        return None
    ostride = max(1, obj_v.shape[1] // 400)
    rot_jerk, wts = [], []
    for r in runs:
        if len(r) < 4:
            continue
        dth_h, dth_o = [], []
        prev_h = prev_o = prev_mh = prev_mo = None
        for k, t in enumerate(r):
            ph, mh = _proj2d(hand_v[t], K_list[t])
            po, mo = _proj2d(obj_v[t][::ostride], K_list[t])
            if k > 0:
                if prev_h is not None and mh.sum() >= 8 and mo.sum() >= 8:
                    hm, om = mh & prev_mh, mo & prev_mo
                    if hm.sum() >= 8 and om.sum() >= 8:
                        dth_h.append(_rot2d(prev_h[hm], ph[hm]))
                        dth_o.append(_rot2d(prev_o[om], po[om]))
                    else:
                        dth_h.append(np.nan)
                        dth_o.append(np.nan)
                else:
                    dth_h.append(np.nan)
                    dth_o.append(np.nan)
            prev_h, prev_o, prev_mh, prev_mo = ph, po, mh, mo
        dphi = np.asarray(dth_h) - np.asarray(dth_o)
        dphi = dphi[np.isfinite(dphi)]
        if len(dphi) >= 2:
            rot_jerk.append(float(np.sqrt(np.mean(np.diff(dphi) ** 2))))
            wts.append(len(r))
    if not rot_jerk:
        return None
    w = np.asarray(wts, np.float64)
    return {"rot_jerk_deg": float(np.average(rot_jerk, weights=w)), "grasp_frames": int(sum(wts))}


# ── soft axes (object 2D placement/orientation + hand video-fidelity) ────────
def _silhouette(verts, faces, K, H, W):
    """Filled projected-object silhouette (one cv2.fillPoly call over all triangles; CPU, no pyrender)."""
    import cv2

    v = np.asarray(verts, np.float64)
    if len(v) < 3:
        return None
    z = v[:, 2].clip(1e-6)
    pts = np.clip(np.stack([K[0, 0] * v[:, 0] / z + K[0, 2], K[1, 1] * v[:, 1] / z + K[1, 2]], 1), [-1e4, -1e4], [1e4, 1e4])
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, pts[np.asarray(faces, np.int64)].astype(np.int32), 1)
    return m


def region_moments(mask):
    """centroid (cx,cy), principal-axis angle [0,180), axis-ratio (>=1) from 2nd moments."""
    ys, xs = np.nonzero(np.asarray(mask, bool))
    if len(xs) < 10:
        return None
    cx, cy = float(xs.mean()), float(ys.mean())
    x, y = xs - cx, ys - cy
    cov = np.array([[(x * x).mean(), (x * y).mean()], [(x * y).mean(), (y * y).mean()]])
    w, V = np.linalg.eigh(cov)
    major = V[:, -1]
    ang = float(np.degrees(np.arctan2(major[1], major[0])) % 180.0)
    ratio = float(np.sqrt(max(w[-1], 1e-9) / max(w[0], 1e-9)))
    return (cx, cy), ang, ratio


def _obj_orient_centroid(verts, faces, K, mask, min_ratio=1.25):
    """(orient_err_deg | nan, centroid_off_pctW) of the projected object vs the mask. depth/scale-free."""
    H, W = mask.shape
    sil = _silhouette(verts, faces, K, H, W)
    if sil is None or int(sil.sum()) < 30:
        return None
    rm, rs = region_moments(mask), region_moments(sil)
    if rm is None or rs is None:
        return None
    (mcx, mcy), ma, mr = rm
    (scx, scy), sa, sr = rs
    cent = float(np.hypot(mcx - scx, mcy - scy) / W * 100.0)
    d = abs(ma - sa) % 180.0
    orient = float(min(d, 180.0 - d)) if (mr > min_ratio and sr > min_ratio) else float("nan")
    return orient, cent


def _wilor_centroids(seg_dir, side, mask_hw):
    """{video_frame_idx: WiLoR image-fit hand 2D centroid in mask-res px}: the optimizer's own
    image evidence ('where the video shows the hand'), read from the WiLoR output in
    hands/. torch is lazy."""
    from pathlib import Path

    sd = Path(seg_dir)
    ptf = sd / "hands" / f"{side}.pt"
    npzf = sd / "hands" / f"{side}.npz"
    if not ptf.is_file() or not npzf.is_file():
        return {}
    try:
        import torch

        d = torch.load(ptf, map_location="cpu", weights_only=False)
        fn = np.load(npzf, allow_pickle=True)["frame_names"]
    except Exception:  # noqa: BLE001
        return {}
    ch, intr, isz = d["cHands"].numpy(), d["intr"].numpy(), d["img_size"].numpy()
    H, W = mask_hw
    out = {}
    for i, name in enumerate(fn):
        try:
            vf = int(str(name))
        except Exception:  # noqa: BLE001
            continue
        v = ch[i]
        v = v[v[:, 2] > 1e-6]
        if len(v) < 3:
            continue
        K = intr[i]
        p = np.stack([K[0, 0] * v[:, 0] / v[:, 2] + K[0, 2], K[1, 1] * v[:, 1] / v[:, 2] + K[1, 2]], 1)
        sx, sy = W / float(isz[i][0]), H / float(isz[i][1])
        out[vf] = np.array([p[:, 0].mean() * sx, p[:, 1].mean() * sy])
    return out


def interaction_fidelity_metrics(d, masks, K_list, frame_indices, seg_dir=None):
    """All interaction-fidelity metrics from the optimized npz `d` (the gated chain), the SAM3
    masks, and the per-keyframe intrinsics K_list (aligned to frame_indices).

    Gates: contact_cover (mean over valid (side,frame)), rot_jerk (grasp-frame-weighted over sides).
    Soft:  obj_orient/obj_centroid (projected obj vs mask), hand_drift (vs WiLoR; needs seg_dir).
    Any metric is None when unmeasurable; all pure-CPU."""
    obj_v = np.asarray(d["obj_verts"], np.float64)
    faces = np.asarray(d["obj_faces"]) if "obj_faces" in d else None
    ki = np.asarray(frame_indices, int)
    W = int(masks.shape[2])
    wil = {s: (_wilor_centroids(seg_dir, s, (int(masks.shape[1]), W)) if seg_dir is not None else {}) for s in ("left", "right")}

    cc, oo, oc, hd = [], [], [], []
    for idx, fi in enumerate(ki):
        if fi >= len(masks):
            continue
        mk = masks[fi].astype(bool)
        if mk.sum() < 30:
            continue
        K = K_list[idx]
        if faces is not None:
            r = _obj_orient_centroid(obj_v[idx], faces, K, mk)
            if r is not None:
                oo.append(r[0])
                oc.append(r[1])
        for side in ("left", "right"):
            vk = f"hand_{side}_valid"
            if vk not in d or not bool(d[vk][idx]):
                continue
            c = contact_cover(d[f"hand_{side}_verts"][idx], obj_v[idx], K, mk)
            if c is not None:
                cc.append(c)
            wc = wil[side].get(int(fi))
            if wc is not None:
                p, ok = _proj2d(d[f"hand_{side}_verts"][idx], K)
                if ok.sum() >= 3:
                    hd.append(float(np.hypot(*(np.nanmean(p[ok], 0) - wc)) / W * 100.0))

    rj, wts = [], []
    for side in ("left", "right"):
        vk = f"hand_{side}_valid"
        if vk not in d:
            continue
        v = np.asarray(d[vk]).astype(bool)
        if v.sum() < 4:
            continue
        tc = temporal_jerk_2d(np.asarray(d[f"hand_{side}_verts"], np.float64), obj_v, K_list, v)
        if tc is not None:
            rj.append(tc["rot_jerk_deg"])
            wts.append(tc["grasp_frames"])

    def _m(a):
        a = [x for x in a if not np.isnan(x)]
        return float(np.mean(a)) if a else None

    return {
        "contact_cover": float(np.mean(cc)) if cc else None,
        "rot_jerk": float(np.average(rj, weights=wts)) if rj else None,
        "obj_orient": _m(oo),
        "obj_centroid": _m(oc),
        "hand_drift": float(np.mean(hd)) if hd else None,
    }


def contact_cover_metric(d, masks, K_list, frame_indices):
    """The ``contact_cover`` member of ``interaction_fidelity_metrics`` alone (same loop and reduction).

    quality_gate evaluates the full bundle on the objref chain and takes gate 15 from this
    value on the pre-objref chain.
    """

    obj_v = np.asarray(d["obj_verts"], np.float64)
    ki = np.asarray(frame_indices, int)
    cc = []
    for idx, fi in enumerate(ki):
        if fi >= len(masks):
            continue
        mk = masks[fi].astype(bool)
        if mk.sum() < 30:
            continue
        K = K_list[idx]
        for side in ("left", "right"):
            valid_key = f"hand_{side}_valid"
            if valid_key not in d or not bool(d[valid_key][idx]):
                continue
            value = contact_cover(d[f"hand_{side}_verts"][idx], obj_v[idx], K, mk)
            if value is not None:
                cc.append(value)
    return float(np.mean(cc)) if cc else None


# ── motion-consistency metrics (video-grounded motion faithfulness) ────────
def _med3(x):
    """3-tap median smoothing along axis 0 (edge-padded): removes single-frame spikes."""
    x = np.asarray(x, np.float64)
    if len(x) < 3:
        return x
    pad = np.concatenate([x[:1], x, x[-1:]])
    return np.median(np.stack([pad[:-2], pad[1:-1], pad[2:]]), axis=0)


def motion_consistency_metrics(d, masks, K_list, frame_indices, contact_m=0.05):
    """Does the extracted trajectory move the way the video shows? The per-frame gates
    (sil_iou / contact_cover / obj_centroid) are structurally blind to motion-level failures:
    object depth wander (per-keyframe MoGe depth noise can walk the object far along the view
    ray while the video shows it still; depth is invisible to per-frame silhouettes, but its
    trend is not: apparent size ~ 1/z), in-plane ghost wobble (extraction 2D path uncorrelated
    with the observed mask path), and no-interaction drift (untouched objects travelling).

    All image-space, per-keyframe, pure numpy/cv2 on the gated chain `d` (objref sidecar when
    present, the same chain as gates 14/15). Returns dict (None fields = unmeasurable, defer):

      area_ratio_drift  p90/p10 of per-keyframe (mask_area / projected-object-sil area) on
                        low-occlusion frames above the relative-size floor (border frames are
                        kept, see the inline comment). A constant offset (absolute scale error) is
                        invisible by design; drift means the extraction's apparent-size trend
                        (depth + rotation footprint) disagrees with the video.
      motion2d_ratio    projected-object 2D path length / mask-centroid 2D path length
      motion2d_corr     Pearson corr of per-step 2D displacement vectors (ext vs mask)
      sil_path_pctW / mask_path_pctW   the two smoothed path lengths (%W — gate trigger context)
      depth_claim_x     p95/p5 of object centroid depth (claimed depth excursion, fold)
      size_obs_x        p95/p5 of sqrt(mask area) on non-border frames (observed excursion, fold)
      contact_frac_time fraction of keyframes with some valid hand within contact_m of the object
      move_nocontact_m  3D object path length over steps with no hand contact (ghost motion)
    """
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

    # per-keyframe records
    ratio_rows = []  # (mask_area / sil_area) on clean frames
    cent_rows = []  # (sil_cx, sil_cy, mask_cx, mask_cy)
    z_rows, size_rows = [], []
    contact = np.zeros(T, dtype=bool)
    ostride = max(1, obj_v.shape[1] // 300)
    for idx in range(T):
        fi = int(ki[idx])
        # 3D contact (independent of masks)
        for side, (hv, hval) in hands.items():
            if hval[idx] and not contact[idx]:
                dist = np.linalg.norm(hv[idx][::15][:, None] - obj_v[idx][::ostride][None], axis=2).min()
                contact[idx] = bool(dist < contact_m)
        if fi >= len(masks):
            continue
        mk = masks[fi].astype(bool)
        ma = float(mk.sum())
        if ma < 30 or faces is None:
            continue
        K = K_list[idx]
        sil = _silhouette(obj_v[idx], faces, K, H, W)
        if sil is None:
            continue
        sa = float(sil.sum())
        if sa < 100:
            continue
        # hand occlusion via convex fill of the projected hand verts (cheap; filter-only)
        hand_sil = np.zeros((H, W), np.uint8)
        for side, (hv, hval) in hands.items():
            if hval[idx]:
                p, ok = _proj2d(hv[idx], K)
                if ok.sum() >= 3:
                    import cv2

                    hull = cv2.convexHull(np.clip(p[ok], [-1e4, -1e4], [1e4, 1e4]).astype(np.int32))
                    cv2.fillConvexPoly(hand_sil, hull, 1)
        occ = float((sil.astype(bool) & hand_sil.astype(bool)).sum()) / sa
        m_border = bool(mk[:2].any() or mk[-2:].any() or mk[:, :2].any() or mk[:, -2:].any())
        ys, xs = np.nonzero(mk)
        sys_, sxs = np.nonzero(sil)
        if occ < 0.6:
            cent_rows.append((float(sxs.mean()), float(sys_.mean()), float(xs.mean()), float(ys.mean())))
        # Border frames are deliberately kept in the ratio series: when the tracking is right,
        # mask and rendered sil clip at the same canvas edge together (ratio stable), and
        # segments with a depth blow-up spend many frames border-touching — filtering them would
        # remove exactly the diagnostic frames. Entering/exiting slivers are rejected by the
        # relative-size floor below instead.
        if occ < 0.5:
            ratio_rows.append((ma / sa, np.sqrt(ma)))
        if not m_border:
            size_rows.append(np.sqrt(ma))
        z_rows.append(float(obj_v[idx][:, 2].mean()))

    out = {
        "area_ratio_drift": None,
        "motion2d_ratio": None,
        "motion2d_corr": None,
        "sil_path_pctW": None,
        "mask_path_pctW": None,
        "depth_claim_x": None,
        "size_obs_x": None,
        "contact_frac_time": float(contact.mean()) if T else None,
        "move_nocontact_m": None,
    }
    c3 = obj_v.mean(axis=1)
    step3 = np.linalg.norm(np.diff(_med3(c3), axis=0), axis=1)
    nc = (~contact[:-1]) & (~contact[1:])
    out["move_nocontact_m"] = float(step3[nc].sum())

    if len(ratio_rows) >= 8:
        rr = np.asarray(ratio_rows)
        # relative-size floor: entering/exiting mask slivers fake huge ratio swings
        keep = rr[:, 1] >= 0.25 * np.median(rr[:, 1])
        if keep.sum() >= 8:
            r = _med3(rr[keep, 0])
            out["area_ratio_drift"] = float(np.percentile(r, 90) / max(np.percentile(r, 10), 1e-9))
    if len(cent_rows) >= 8:
        c = np.asarray(cent_rows)
        sc, mc = _med3(c[:, :2]), _med3(c[:, 2:])
        ds, dm = np.diff(sc, axis=0), np.diff(mc, axis=0)
        sp = float(np.linalg.norm(ds, axis=1).sum()) / W * 100.0
        mp = float(np.linalg.norm(dm, axis=1).sum()) / W * 100.0
        out["sil_path_pctW"], out["mask_path_pctW"] = sp, mp
        out["motion2d_ratio"] = float(sp / max(mp, 1e-6))
        if ds.std() > 1e-9 and dm.std() > 1e-9:
            out["motion2d_corr"] = float(np.corrcoef(ds.ravel(), dm.ravel())[0, 1])
    if len(z_rows) >= 8:
        z = _med3(np.asarray(z_rows))
        if z.min() > 0.01:
            out["depth_claim_x"] = float(np.percentile(z, 95) / max(np.percentile(z, 5), 1e-9))
    if len(size_rows) >= 8:
        s = _med3(np.asarray(size_rows))
        out["size_obs_x"] = float(np.percentile(s, 95) / max(np.percentile(s, 5), 1e-9))
    return out


def mask_hand_coverage(masks, hands, faces_by_side, frame_indices=None, stride=3, min_mask_px=50):
    """Median fraction of the SAM3 object mask covered by the projected MANO hand silhouette.

    The mask-is-the-hand detector (gate 20): agnostic pointing can bind to the hand itself,
    and the resulting hand-mask "object" is self-consistently wrong through every downstream
    stage (reconstruction, pose and contact all track the hand), so it can pass all prior gates.
    Genuine objects have low coverage (a few percent of the mask); a hand taken as the object
    covers a large fraction.

    masks: (T,mh,mw) SAM3 masks. hands: the WiLoR output in hands/ (gtf.utils.load_hand_data dict). faces_by_side:
    {"left": faces, "right": faces} (gtf.utils.load_mano_faces). frame_indices: restrict to
    these frames (the gates' exported-window set); default = all hand-tracked frames.
    Returns None when unmeasurable (no hands / no usable frames); the gate then defers.
    """
    if not hands or masks is None:
        return None
    T, mh, mw = masks.shape
    if frame_indices is None:
        frames = sorted({f for h in hands.values() for f in h["frame_to_idx"]})
    else:
        frames = sorted(int(f) for f in frame_indices)
    frames = [f for f in frames if 0 <= f < T][::stride]
    fracs = []
    for f in frames:
        m = masks[f] > 0
        area = int(m.sum())
        if area < min_mask_px:
            continue
        sil = None
        for side, h in hands.items():
            idx = h["frame_to_idx"].get(f)
            valid_frames = h.get("valid_frames")
            if idx is None or (valid_frames is not None and f not in valid_frames):
                continue
            v = h["verts"][idx]
            K = np.array(h["intr"][idx] if h["intr"].ndim == 3 else h["intr"], np.float64)
            W, H = (h["img_size"][idx] if h["img_size"].ndim == 2 else h["img_size"])[:2]
            Ks = K.copy()
            Ks[0] *= mw / float(W)
            Ks[1] *= mh / float(H)
            s = _silhouette(v, faces_by_side[side], Ks, mh, mw)
            if s is not None:
                sil = s if sil is None else (sil | s)
        if sil is None or int(sil.sum()) < min_mask_px:
            continue
        fracs.append(float((m & (sil > 0)).sum()) / area)
    if not fracs:
        return None
    return float(np.median(fracs))


def _ramp(x, good, bad):
    """1.0 at `good`, 0.0 at `bad`, linear between, clipped. Handles either direction (good</>bad)."""
    if x is None or bad == good:
        return None
    return float(np.clip((bad - x) / (bad - good), 0.0, 1.0))


def interaction_fidelity_score(contact_cover=None, rot_jerk=None, hand_drift=None, obj_centroid=None, floor=0.7, motion2d_corr=None, area_ratio_drift=None):
    """Continuous [floor, 1] soft sample weight for a gate-passing segment: down-weights the
    borderline-but-passing segments (the model's confidence channel, via build_manifest → extractor_score).
    Weighted mean of the per-axis goodness (the two gate axes weighted above the soft axes,
    which over-flag); missing axes are skipped; floored so survivors stay usable. 1.0 = clean.
    motion2d_corr / area_ratio_drift: gentle down-weight for passing segments with imperfect
    motion (damped depth, transparent-object mask breathing) below gate 16/17."""
    comps = [
        (0.35, _ramp(contact_cover, 0.40, 0.08)),  # high good; 0 at the moderate gate threshold
        (0.35, _ramp(rot_jerk, 3.5, 6.0)),  # low good; 0 at the moderate gate threshold
        (0.15, _ramp(hand_drift, 5.0, 15.0)),  # low good (%W)
        (0.15, _ramp(obj_centroid, 4.0, 12.0)),  # low good (%W)
        (0.15, _ramp(motion2d_corr, 0.85, 0.2)),  # high good
        (0.15, _ramp(area_ratio_drift, 1.5, 3.5)),  # low good
    ]
    comps = [(w, s) for w, s in comps if s is not None]
    if not comps:
        return 1.0
    g = sum(w * s for w, s in comps) / sum(w for w, _ in comps)
    return float(floor + (1.0 - floor) * g)
