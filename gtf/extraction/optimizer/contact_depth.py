"""Depth coherence between a hand and the object it holds (upgraded chain: follow_depth="local", contact_depth="ray").

A monocular video fixes where the hand and the object are in the image, not how far each is from the camera: the hand's
depth comes from its apparent size, the object's from the scene depth and its own (unknown) size, and the two disagree by
several centimeters. With a size-based hand depth and the object following the hand, nothing re-aligns them per frame, so
the object can float in front of or behind the hand that the video shows holding it; a move along the hand's viewing ray
closes that gap and keeps the hand where the image shows it.

Pure numpy (+ igl for exact signed distances, cv2 for a small z-buffer):

  object_depth_at_hand   the object's depth where the hand is in the image, for the clip-level object scale. contact_follow
                         scales the object about the camera so that it sits at the hand's depth; pairing the hand with the
                         object's centroid is wrong for long objects held at one end (the object shrinks and floats
                         centimeters off the hand). Optimizer setting `follow_depth=local`.
  order_evidence         which of hand and object the video shows in front, per hand vertex: a vertex on a visible-hand
                         pixel must not be behind the object's front surface there, a vertex on a visible-object pixel must
                         not be in front of it. This is what tells the front of a thick object from its back.
  local_follow_depth     object_depth_at_hand made order-aware: a hand seen in front of the object is paired with the
                         object's front surface, a hand hidden behind it with its back surface.
  align_hand_depth       per keyframe on which the image shows the hand on the object: the shift of the hand along its own
                         viewing ray (pixels unchanged; a 3 cm move at 60 cm changes its drawn size by 5%) that puts its
                         palm side on the object surface without going inside it and without contradicting the visible
                         depth order, chosen jointly over time (Viterbi over a shift grid, so the depth neither jumps nor
                         flips between the front and the back of the object), tapered to zero away from contact.
                         Optimizer setting `contact_depth=ray`: applied to the hand depth calibration before the optimizer
                         seeds its contact targets (they are seeded from proximity, so a hand that starts > 2 cm from the
                         object never gets any).
"""

import numpy as np

TAU = 0.03  # m: palm-side vertices farther than this from the surface count as "not in contact" (cost saturates)
CLOSE = 0.008  # m: a palm-side vertex this near the surface is "on" it
MIN_CLOSE = 5  # palm-side vertices that must be able to get on the surface for a keyframe to say anything about depth
PEN_FREE = 0.004  # m of penetration that costs nothing (soft tissue, mesh noise)
PEN_CAP = 0.05
W_PEN = 4.0  # mean depth inside the object (all sampled vertices)
W_PEN_MAX = 6.0  # deepest vertex inside the object: a hand is pressed on, never buried, to bring more of the palm close
W_VIS = 3.0  # share of the hand's vertices that contradict the visible depth order
VIS_FRONT_TOL = 0.02  # m: a vertex on a visible-hand pixel may be this far behind the object's front surface (hand thickness)
VIS_BACK_TOL = 0.005
W_PRIOR = 0.15  # cost of the largest allowed shift relative to "no palm vertex in contact" (= 1)
W_NO_CONTACT = 1.0  # a shift at which fewer than MIN_CLOSE palm-side vertices are on the surface, on a keyframe where some
#                     shift reaches that: the image shows contact, so a fingertip grasp (contact term only ~0.05 below "none")
#                     must still beat staying 10 cm away
W_STEP = 0.5  # cost of a 1 cm shift change between consecutive keyframes
GAP_CAP = 6  # keyframes: touching keyframes farther apart than this are coupled as if this far apart
HAND_HALF = 0.015  # m: a hand lying on a surface has its center this far from it
SIL_MIN_PRECISION = 0.6  # order_evidence: share of the mesh silhouette around the hand that must be object or hand in the video


def _signed_distance(points, verts, faces):
    import igl

    P = np.ascontiguousarray(points, dtype=np.float64)
    V = np.ascontiguousarray(verts, dtype=np.float64)
    F = np.ascontiguousarray(faces, dtype=np.int64)
    d2, _, _ = igl.point_mesh_squared_distance(P, V, F)
    w = igl.fast_winding_number(V, F, P)
    return np.where(w > 0.5, -1.0, 1.0) * np.sqrt(np.maximum(d2, 0.0))


def object_depth_at_hand(obj_verts, K_obj, hand_uv, min_verts=12, radius_frac=0.5, nearest_frac=0.04):
    """Depth (camera z) of the object where the hand is in the image.

    obj_verts (V,3) camera coordinates; K_obj 3x3 intrinsics of that camera in the pixel units of hand_uv; hand_uv (J,2)
    the hand's joints in the same image. Mean z of the object vertices that project within radius_frac x the hand's image
    extent of the hand's image center (front and back of the object there -> its local middle; align_hand_depth then
    settles on which side of it the hand is); when fewer than min_verts do, of the nearest_frac of vertices closest to the
    hand in the image (a hand that touches the object's edge). Returns (z, n_inside) or (None, 0)."""
    V = np.asarray(obj_verts, dtype=np.float64)
    uv = np.asarray(hand_uv, dtype=np.float64)
    ok = V[:, 2] > 1e-6
    if int(ok.sum()) < 3 or len(uv) < 3:
        return None, 0
    V = V[ok]
    K = np.asarray(K_obj, dtype=np.float64)
    pu = V[:, 0] / V[:, 2] * K[0, 0] + K[0, 2]
    pv = V[:, 1] / V[:, 2] * K[1, 1] + K[1, 2]
    c = uv.mean(0)
    ext = float(np.hypot(np.ptp(uv[:, 0]), np.ptp(uv[:, 1])))
    d = np.hypot(pu - c[0], pv - c[1])
    inside = d <= radius_frac * max(ext, 1.0)
    n_in = int(inside.sum())
    if n_in < min_verts:
        k = max(min_verts, int(round(nearest_frac * len(V))))
        inside = np.zeros(len(V), bool)
        inside[np.argsort(d)[: min(k, len(V))]] = True
    return float(V[inside, 2].mean()), n_in


def front_depth(verts, faces, K, hw, width=256, roi=None, far=False):
    """Nearest-surface depth of a mesh per pixel (painter's z-buffer at `width` px): (Z (h,width) float32, inf where the
    mesh is not drawn; scale = low-res px per K px). roi = (u0, v0, u1, v1) in K pixels: only faces touching it are drawn.
    far=True: the farthest surface instead (the back of the object)."""
    import cv2

    H, W = int(hw[0]), int(hw[1])
    s = float(width) / max(W, 1)
    h = max(1, int(round(H * s)))
    Z = np.full((h, int(width)), np.inf, np.float32)
    V = np.asarray(verts, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    z = V[:, 2]
    F = F[(z[F] > 1e-6).all(1)]
    if not len(F):
        return Z, s
    zs = np.where(z > 1e-6, z, 1.0)
    uv = np.stack([V[:, 0] / zs * K[0, 0] + K[0, 2], V[:, 1] / zs * K[1, 1] + K[1, 2]], 1) * s
    P = uv[F]
    lo, hi = P.min(1), P.max(1)
    box = (0.0, 0.0, float(width), float(h)) if roi is None else tuple(float(x) * s for x in roi)
    keep = (hi[:, 0] >= box[0]) & (lo[:, 0] <= box[2]) & (hi[:, 1] >= box[1]) & (lo[:, 1] <= box[3])
    P, zf = np.round(P[keep]).astype(np.int32), z[F[keep]].mean(1)
    for i in np.argsort(zf if far else -zf):  # far to near: the nearest surface is painted last (far=True: the reverse)
        cv2.fillConvexPoly(Z, P[i], float(zf[i]))
    return Z, s


def order_evidence(hand_verts, obj_verts, faces, K, hw, obj_mask, hand_mask, min_precision=SIL_MIN_PRECISION):
    """Visible depth order at the hand's vertices, or None without both masks or where the object mesh cannot be trusted.

    The cue reads the object's extent under the hand from the mesh (the video cannot show it), so it is only as good as
    the mesh's fit: around the hand (its image box + 5% of the width) at least min_precision of the mesh's silhouette must
    lie on object or hand pixels of the video; otherwise None ("precision" is returned with the evidence). Without this
    check a flat object whose mesh covers twice its mask puts a hand resting on its rim "in front of the object" by the
    object's whole depth.

    hand_verts / obj_verts in one camera with intrinsics K for an image of hw = (H, W); obj_mask = the object's visible
    pixels (SAM 3 on the video), hand_mask = the visible hand pixels (SAM 3 "hand"), any resolution. Per hand vertex:
    zf / zb (depth of the object's front / back surface at the vertex's pixel, inf off its silhouette), on_hand (the pixel
    shows a hand), on_obj (the pixel shows the object and no hand)."""
    if obj_mask is None or hand_mask is None:
        return None
    Hv = np.asarray(hand_verts, dtype=np.float64)
    if (Hv[:, 2] <= 1e-6).any():
        return None
    K = np.asarray(K, dtype=np.float64)
    H, W = int(hw[0]), int(hw[1])
    u = Hv[:, 0] / Hv[:, 2] * K[0, 0] + K[0, 2]
    v = Hv[:, 1] / Hv[:, 2] * K[1, 1] + K[1, 2]
    pad = 0.05 * W
    roi = (u.min() - pad, v.min() - pad, u.max() + pad, v.max() + pad)
    Z, s = front_depth(obj_verts, faces, K, (H, W), roi=roi)
    Zb, _ = front_depth(obj_verts, faces, K, (H, W), roi=roi, far=True)
    ui, vi = np.round(u * s).astype(int), np.round(v * s).astype(int)
    inb = (ui >= 0) & (ui < Z.shape[1]) & (vi >= 0) & (vi < Z.shape[0])
    zf, zb = np.full(len(Hv), np.inf), np.full(len(Hv), np.inf)
    zf[inb] = Z[vi[inb], ui[inb]]
    zb[inb] = Zb[vi[inb], ui[inb]]
    # does the mesh silhouette around the hand agree with the video? (low-res pixel centers inside the roi)
    yy, xx = np.nonzero(np.isfinite(Z))
    kx, ky = (xx + 0.5) / s, (yy + 0.5) / s
    box = (kx >= roi[0]) & (kx <= roi[2]) & (ky >= roi[1]) & (ky <= roi[3])
    if int(box.sum()) < 20:
        precision = 1.0  # (almost) no mesh near the hand: nothing to contradict
    else:
        seen = np.zeros(int(box.sum()), bool)
        for m in (obj_mask, hand_mask):
            m = np.asarray(m)
            mx = np.clip((kx[box] * m.shape[1] / W).astype(int), 0, m.shape[1] - 1)
            my = np.clip((ky[box] * m.shape[0] / H).astype(int), 0, m.shape[0] - 1)
            seen |= m[my, mx].astype(bool)
        precision = float(seen.mean())
    if precision < min_precision:
        return None

    def sample(m):
        m = np.asarray(m)
        x = np.clip(np.round(u * m.shape[1] / W).astype(int), 0, m.shape[1] - 1)
        y = np.clip(np.round(v * m.shape[0] / H).astype(int), 0, m.shape[0] - 1)
        return m[y, x].astype(bool) & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    on_hand = sample(hand_mask)
    return dict(zf=zf, zb=zb, on_hand=on_hand, on_obj=sample(obj_mask) & ~on_hand, precision=precision)


def local_follow_depth(obj_verts, faces, K, hw, hand_px, hand_uv, obj_mask=None, hand_mask=None, min_px=30):
    """The pair that ties the object's clip-level depth to one hand on one keyframe: (z_obj, offset, kind), meaning "the
    object depth z_obj belongs at the hand's center depth + offset".

    hand_px (N,2) the hand's vertices and hand_uv (J,2) its joints in the pixels of K (image hw = (H, W)). With both masks
    (order_evidence): a hand the video shows in front of the object (its pixels are hand pixels inside the object's
    silhouette) is paired with the object's front surface there, HAND_HALF behind the hand's center; a hand hidden behind
    it (its pixels show the object) with the back surface, HAND_HALF in front. Otherwise (a wrapped or edge grasp, or no
    hand mask) with the object's middle where the hand is (object_depth_at_hand). (None, 0, "none") without object there."""
    z_mid, _ = object_depth_at_hand(obj_verts, K, hand_uv)
    if z_mid is None:
        return None, 0.0, "none"
    px = np.asarray(hand_px, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    rays = np.stack([(px[:, 0] - K[0, 2]) / K[0, 0], (px[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(px))], 1)
    od = order_evidence(rays, obj_verts, faces, K, hw, obj_mask, hand_mask)
    if od is None:
        return z_mid, 0.0, "middle"
    fin = np.isfinite(od["zf"])
    vis, hid = od["on_hand"] & fin, od["on_obj"] & fin
    if vis.sum() >= min_px and vis.sum() >= 2 * hid.sum():
        return float(np.median(od["zf"][vis])), HAND_HALF, "front"
    if hid.sum() >= min_px and hid.sum() >= 2 * vis.sum():
        return float(np.median(od["zb"][hid])), -HAND_HALF, "back"
    return z_mid, 0.0, "middle"


def ray_contact_cost(hand, palm_idx, obj, faces, shifts, order=None, extra_stride=8):
    """Cost of holding the hand at each shift (meters along its centroid's viewing ray).

    contact = share of the palm side that is not on the surface (truncated distance of the palm-side vertices); pen =
    depth inside the object beyond PEN_FREE (mean and deepest, palm side and every extra_stride-th vertex); vis = share of
    all vertices that contradict the visible depth order (order_evidence; 0 without it). cost = contact + pen + vis: low
    where the hand lies on the object on the side the video shows, high in free space, inside it, and on the wrong side.
    Returns dict(cost (S,), n_close (S,) palm-side vertices on the surface, mind (S,) closest signed distance)."""
    hand = np.asarray(hand, dtype=np.float64)
    shifts = np.asarray(shifts, dtype=np.float64)
    c = hand.mean(0)
    r = c / max(float(np.linalg.norm(c)), 1e-9)
    palm_idx = np.asarray(palm_idx, dtype=np.int64)
    idx = np.concatenate([palm_idx, np.arange(0, len(hand), extra_stride)])
    P = hand[idx]
    S = len(shifts)
    sd = _signed_distance((P[None] + shifts[:, None, None] * r[None, None]).reshape(-1, 3), obj, faces).reshape(S, len(idx))
    pal = sd[:, : len(palm_idx)]
    contact = np.minimum(np.abs(pal), TAU).mean(1) / TAU
    pen = np.clip(-sd - PEN_FREE, 0.0, PEN_CAP) / PEN_CAP
    cost = contact + W_PEN * pen.mean(1) + W_PEN_MAX * pen.max(1)
    if order is not None:
        z = hand[None, :, 2] + shifts[:, None] * r[2]
        fin = np.isfinite(order["zf"])[None]
        zf = np.where(np.isfinite(order["zf"]), order["zf"], 0.0)[None]
        hidden_hand = order["on_hand"][None] & fin & (z > zf + VIS_FRONT_TOL)  # a visible part of the hand would be behind the object's front
        hidden_obj = order["on_obj"][None] & fin & (z < zf - VIS_BACK_TOL)  # the hand would cover object pixels the video shows
        cost = cost + W_VIS * (hidden_hand.sum(1) + hidden_obj.sum(1)) / float(hand.shape[0])
    return dict(cost=cost, n_close=(np.abs(pal) <= CLOSE).sum(1), mind=sd.min(1))


def _viterbi(cost, step_cost):
    """cost (T,S); step_cost (T-1,S,S). Minimum-cost state path."""
    T, S = cost.shape
    acc = cost[0].copy()
    back = np.zeros((T, S), np.int64)
    for t in range(1, T):
        m = acc[:, None] + step_cost[t - 1]
        back[t] = m.argmin(0)
        acc = m.min(0) + cost[t]
    path = np.zeros(T, np.int64)
    path[-1] = int(acc.argmin())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def align_hand_depth(hand_verts, touch, obj_verts, faces, palm_idx, order=None, max_shift=None, max_rel=0.25, step=0.01, max_gap=3, taper=4, smooth=1.5):
    """Per-keyframe shift (m, + = away from the camera) of a hand along its viewing ray onto the object it touches.

    hand_verts (T,Nh,3), obj_verts (T,Nv,3) in one camera per keyframe; touch (T,) bool = the image shows this hand on the
    object on that keyframe (and both are valid); order = per keyframe an order_evidence dict or None. All touching
    keyframes are solved jointly (ray_contact_cost + a penalty for not being on the surface where some shift reaches it +
    a small prior to stay put + a penalty on changes of the hand's depth
    relative to the object, which weakens with the distance between keyframes), refined on a 4x finer grid, and that
    relative depth is smoothed over `smooth` keyframes: in contact the hand follows the object's depth. A keyframe on
    which fewer than MIN_CLOSE palm-side vertices can reach the surface at any allowed shift says nothing about depth (a
    sideways miss or a graze): it follows its neighbors; a run of touching keyframes (gaps <= max_gap) without any
    evidence is left alone. Outside runs the shift decays linearly to zero over `taper` keyframes. |shift| <=
    min(max_shift, max_rel x depth); max_shift=None: max_rel x the median depth of the touching keyframes, within
    0.10-0.60 m (the depth error of a hand grows with its distance, so a fixed bound stops far hands short of the
    object), with the grid step widened to keep at most ~60 states. Returns (shift (T,), info dict)."""
    hand_verts = np.asarray(hand_verts, dtype=np.float64)
    obj_verts = np.asarray(obj_verts, dtype=np.float64)
    T = len(hand_verts)
    touch = np.asarray(touch, bool)
    shift = np.zeros(T)
    info = dict(n_touch=int(touch.sum()), n_evidence=0, n_aligned=0, median_abs_mm=0.0, max_abs_mm=0.0, gap_before_mm=None, order_frames=0)
    idx = np.flatnonzero(touch)
    if len(idx) == 0:
        return shift, info
    depth = np.linalg.norm(hand_verts.mean(1), axis=1)
    if max_shift is None:
        max_shift = float(np.clip(max_rel * np.median(depth[idx]), 0.10, 0.60))
        step = max(step, max_shift / 30.0)
    n_side = int(np.ceil(max_shift / step - 1e-9))
    grid = np.arange(-n_side, n_side + 1) * step  # symmetric, 0 on the grid
    zero = n_side
    n = len(idx)
    C = np.zeros((n, len(grid)))
    ev = np.zeros(n, bool)
    gap0 = np.zeros(n)
    lim = np.minimum(max_shift, max_rel * depth[idx])
    for i, t in enumerate(idx):
        od = order[t] if order is not None else None
        info["order_frames"] += int(od is not None)
        r = ray_contact_cost(hand_verts[t], palm_idx, obj_verts[t], faces, grid, order=od)
        out = np.abs(grid) > lim[i] + 1e-9
        gap0[i] = max(float(r["mind"][zero]), 0.0)
        ev[i] = bool((r["n_close"][~out] >= MIN_CLOSE).any())
        c = r["cost"] + W_NO_CONTACT * (r["n_close"] < MIN_CLOSE) + W_PRIOR * np.abs(grid) / max_shift if ev[i] else np.zeros(len(grid))
        c[out] = 1e6
        C[i] = c
    info.update(n_evidence=int(ev.sum()), gap_before_mm=float(np.median(gap0) * 1000))
    if not ev.any():
        return shift, info
    # what must be steady over time is the hand's depth relative to the object (rel = hand - object along the ray, before
    # the shift), not the shift: a hand on an object follows the object's depth, whatever its own size depth does
    r_hat = hand_verts[idx].mean(1) / np.maximum(depth[idx, None], 1e-9)
    rel = depth[idx] - (r_hat * obj_verts[idx].mean(1)).sum(1)
    dg = grid[:, None] - grid[None]
    tr = (
        np.stack([W_STEP * np.abs(dg + (rel[i] - rel[i + 1])) / 0.01 / min(max(1, int(idx[i + 1] - idx[i])), GAP_CAP) for i in range(n - 1)])
        if n > 1
        else np.zeros((0, len(grid), len(grid)))
    )
    path = _viterbi(C, tr) if n > 1 else np.array([int(C[0].argmin())])
    sol = grid[path].astype(np.float64)
    fine = np.arange(-step, step + 1e-9, step / 4.0)
    for i, t in enumerate(idx):
        if not ev[i]:
            continue
        r = ray_contact_cost(hand_verts[t], palm_idx, obj_verts[t], faces, sol[i] + fine, order=order[t] if order is not None else None)
        c = r["cost"] + W_NO_CONTACT * (r["n_close"] < MIN_CLOSE) + W_STEP * np.abs(fine) / 0.01 * 0.25
        sol[i] = float(np.clip(sol[i] + fine[int(c.argmin())], -lim[i], lim[i]))
    # runs of touching keyframes; a run without any evidence is left alone
    run_id = np.concatenate([[0], np.cumsum(np.diff(idx) > max_gap)])
    solved = np.zeros(T, bool)
    for rid in np.unique(run_id):
        m = run_id == rid
        if not ev[m].any():
            continue
        s, ti = sol[m], idx[m]
        if smooth > 0 and len(s) > 2:  # the per-keyframe refinement must not put its own jitter on the hand-object depth
            w = np.exp(-0.5 * ((ti[:, None] - ti[None]) / smooth) ** 2)
            s = np.clip((w * (s + rel[m])[None]).sum(1) / w.sum(1) - rel[m], -lim[m], lim[m])
        shift[ti] = s
        solved[ti] = True
    st = np.flatnonzero(solved)
    out = np.zeros(T)
    for t in range(T):
        if solved[t]:
            out[t] = shift[t]
            continue
        lo, hi = st[st < t], st[st > t]
        a = (int(lo[-1]), shift[lo[-1]]) if len(lo) else None
        b = (int(hi[0]), shift[hi[0]]) if len(hi) else None
        if a and b and b[0] - a[0] <= max_gap + 1:
            out[t] = a[1] + (b[1] - a[1]) * (t - a[0]) / (b[0] - a[0])
        else:
            va = a[1] * max(0.0, 1.0 - (t - a[0]) / (taper + 1.0)) if a else 0.0
            vb = b[1] * max(0.0, 1.0 - (b[0] - t) / (taper + 1.0)) if b else 0.0
            out[t] = va if abs(va) >= abs(vb) else vb
    if solved.any():
        info.update(n_aligned=int(solved.sum()), median_abs_mm=float(np.median(np.abs(out[solved])) * 1000), max_abs_mm=float(np.abs(out[solved]).max() * 1000))
    return out, info
