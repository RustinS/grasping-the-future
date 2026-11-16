"""Object scale and depth anchors of the optimizer: the MoGe metric scale and the hand-contact anchor."""

import cv2
import numpy as np
from gtf.utils import load_moge_depth


def _metric_scale_ratio(obj_verts_cv, depth_map, K_m, mask, min_pix=80):
    """Multiplicative object-scale correction anchoring the SAM3D recon to metric depth.

    A single view leaves the scale and depth of a SAM 3D mesh ambiguous: the silhouette fixes
    only its angular size, so an over-scaled mesh placed too far matches the image. MoGe metric
    depth at the segmented pixels supplies the distance (angular size x distance = metric size).
    The lateral (xy) metric extents are compared, since the visible MoGe front surface and the
    full mesh share the silhouette's width and height (the z extent is not comparable: MoGe sees
    only the front). The correction c = size_moge / size_mesh scales the placed object about the
    camera origin (size and depth x c), which leaves its projection unchanged. c inherits MoGe's
    depth-scale error (X, Y are proportional to z). Returns None when unreliable (too few pixels
    or a degenerate extent)."""
    if mask is None or depth_map is None:
        return None
    H, W = depth_map.shape[:2]
    m = mask
    if m.shape[:2] != (H, W):
        m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    valid = (m > 0) & np.isfinite(depth_map) & (depth_map > 1e-3)
    if int(valid.sum()) < min_pix:
        return None
    ys, xs = np.nonzero(valid)
    z = depth_map[ys, xs].astype(np.float64)
    fx, fy, cx, cy = K_m[0, 0], K_m[1, 1], K_m[0, 2], K_m[1, 2]
    X = (xs - cx) * z / fx
    Y = (ys - cy) * z / fy

    def _rng(a):
        return float(np.percentile(a, 95) - np.percentile(a, 5))

    e_moge = float(np.hypot(_rng(X), _rng(Y)))
    v = np.asarray(obj_verts_cv, dtype=np.float64)
    e_mesh = float(np.hypot(_rng(v[:, 0]), _rng(v[:, 1])))
    if e_mesh < 1e-4 or e_moge < 1e-4:
        return None
    return e_moge / e_mesh


def _pctl_range(a, lo=5, hi=95):
    a = np.asarray(a, dtype=np.float64)
    return float(np.percentile(a, hi) - np.percentile(a, lo)) if len(a) > 4 else (float(a.max() - a.min()) if len(a) else 0.0)


# Contact keyframes of the hand anchor (scale_anchor="contact_follow").
# The hand's image footprint must be >= 10% of the image width: a small / far hand gives an unreliable
# footprint depth, and it is where MoGe also tends to mis-estimate the focal.
_ANCHOR_MIN_HANDFRAC = 0.10
_ANCHOR_MAX_CONTACT = 0.35  # hand-keypoint centroid within this (normalized) of the object mask
_ANCHOR_MIN_FRAMES = 3  # need >= this many reliable contact keyframes
_ANCHOR_MIN_FOV = 70.0  # the hand anchor applies on a wide FoV only (on a narrow FoV the MoGe anchor is used)


def _hand_depth_anchor(joints_3d, K_h, img_size_h, depth_map, K_m, mask):
    """Per-keyframe evidence of one hand for the contact anchor, using the canonical-size hand as a ruler.

    Returns (s_depth, hand_frac, contact_dist, z_impl) or None. z_impl = f_m * hand_size / P_hand is
    the depth at which a hand of the joints' metric lateral size produces its image footprint P_hand
    through the MoGe focal f_m, and s_depth = z_moge_at_hand / z_impl. hand_frac is the footprint as a
    fraction of the image width (small / far hands are gated out at the call site) and contact_dist
    the normalized image distance between the hand keypoints and the object mask centroid.
    """
    if mask is None or joints_3d is None or depth_map is None:
        return None
    H_m, W_m = depth_map.shape[:2]
    W_h, H_h = float(img_size_h[0]), float(img_size_h[1])
    if W_h < 1 or H_h < 1:
        return None
    j = np.asarray(joints_3d, dtype=np.float64)
    z = j[:, 2]
    vz = z > 1e-6
    if int(vz.sum()) < 8:
        return None
    fx_hn, fy_hn = K_h[0, 0] * W_m / W_h, K_h[1, 1] * H_m / H_h
    cx_hn, cy_hn = K_h[0, 2] * W_m / W_h, K_h[1, 2] * H_m / H_h
    u = fx_hn * j[vz, 0] / z[vz] + cx_hn
    v = fy_hn * j[vz, 1] / z[vz] + cy_hn
    ib = (u >= 0) & (u < W_m) & (v >= 0) & (v < H_m)
    if int(ib.sum()) < 5:
        return None
    ui = np.clip(np.round(u[ib]).astype(int), 0, W_m - 1)
    vi = np.clip(np.round(v[ib]).astype(int), 0, H_m - 1)
    zm = depth_map[vi, ui]
    zm = zm[zm > 1e-3]
    if len(zm) < 5:
        return None
    # bimodal-gap filter (some joints project onto distant background) — keep the near cluster
    zm = np.sort(zm)
    gaps = np.diff(zm)
    if len(gaps):
        gi = int(np.argmax(gaps))
        rng = zm[-1] - zm[0]
        if rng > 0 and gaps[gi] / rng > 0.3 and gi >= 2 and (len(zm) - gi - 1) >= 2:
            zm = zm[: gi + 1]
    z_moge = float(np.median(zm))
    p_hand = float(np.hypot(_pctl_range(u), _pctl_range(v)))  # image footprint (pixels, MoGe image)
    hand_size = float(np.hypot(_pctl_range(j[vz, 0]), _pctl_range(j[vz, 1])))  # metric lateral size
    if p_hand < 1e-3 or hand_size < 1e-4:
        return None
    # z_impl uses the MoGe camera focal K_m, not the WiLoR focal that projected the joints: the
    # footprint is in the MoGe image, so z_impl is comparable to z_moge. With the WiLoR focal it
    # would cancel against the projection and reduce to the raw WiLoR joint depth.
    f_m = float(K_m[0, 0])
    z_impl = f_m * hand_size / p_hand
    if not (0.1 < z_impl < 3.0):
        return None
    ys, xs = np.nonzero(mask > 0)
    if len(xs) < 30:
        return None
    contact = float(np.hypot(float(u.mean()) / W_m - float(xs.mean()) / mask.shape[1], float(v.mean()) / H_m - float(ys.mean()) / mask.shape[0]))
    return z_moge / z_impl, p_hand / W_m, contact, z_impl


def _segment_depth_loader(segment_dir):
    """Return a segment-scoped, lazy MoGe depth loader.

    ``load_segment`` reads the same depth once per valid hand to calibrate MANO and once more
    for the object path. The depth archives are compressed, so caching avoids decompressing
    every array several times. Only this segment's requested frames are kept, and the closure
    (with its arrays) is released when ``load_segment`` returns. Consumers treat the arrays as
    read-only.
    """

    cache = {}

    def load(frame_index):
        frame_index = int(frame_index)
        if frame_index not in cache:
            cache[frame_index] = load_moge_depth(segment_dir, frame_index)
        return cache[frame_index]

    return load


_CONTACT_CLIP = (0.5, 1.5)  # bound the contact-anchor object scale
_FOLLOW_LOCAL_GAP = 0.05  # follow_depth="local": WiLoR hand within this (x its image size) of the object mask = on the object
_FOLLOW_LOCAL_CLIP = (0.2, 3.0)  # follow_depth="local": bound of the object scale (as size_fallback="hands")


def _resolve_contact_anchor(contact_pairs, cam_fov_deg=None, clip=None):
    """Tie the object's absolute scale / depth to the hand's depth at contact, over the whole trajectory.

    The object's absolute scale and depth are one degree of freedom: the silhouette ties size to
    depth and SAM 3D supplies the relative trajectory. At contact the object sits at the hand's
    depth, so over all contact keyframes c = argmin_c sum (c z_obj - z_hand)^2 =
    sum(z_obj z_hand) / sum(z_obj^2), an over-determined fit. Placing the object at the hand's depth
    (instead of moving the hand to the object's) leaves the hand where the video puts it.
    contact_pairs = [(z_obj, z_hand), ...]. Returns (c, info); c=None on a narrow FoV or with too
    few contact keyframes."""
    if cam_fov_deg is not None and cam_fov_deg < _ANCHOR_MIN_FOV:
        return (
            None,
            f"no-op (narrow FoV {cam_fov_deg:.0f}° < {_ANCHOR_MIN_FOV:.0f}°: the MoGe placement is reliable there)",
        )
    if len(contact_pairs) < _ANCHOR_MIN_FRAMES:
        return None, f"no-op ({len(contact_pairs)}/{_ANCHOR_MIN_FRAMES} reliable contact keyframes)"
    zo = np.array([p[0] for p in contact_pairs], dtype=np.float64)
    zh = np.array([p[1] for p in contact_pairs], dtype=np.float64)
    # Trimmed least squares: one bad contact frame would swing a plain fit, so per-frame ratios more
    # than 25% off the median are dropped first.
    ratios = zh / np.where(zo > 1e-9, zo, 1.0)
    med = float(np.median(ratios))
    keep = np.abs(ratios - med) <= 0.25 * max(med, 1e-6)
    if int(keep.sum()) >= _ANCHOR_MIN_FRAMES:
        zo, zh = zo[keep], zh[keep]
    denom = float(np.sum(zo * zo))
    if denom < 1e-9:
        return None, "no-op (degenerate z_obj)"
    raw = float(np.sum(zo * zh) / denom)
    c = float(np.clip(raw, *(clip or _CONTACT_CLIP)))
    return c, f"c={c:.3f} (trimmed-LS over {int(keep.sum())}/{len(contact_pairs)} contact keyframes, raw {raw:.3f})"


def _moge_scale(moge_ratios):
    """Median of the per-keyframe MoGe ratios, clipped to [0.5, 1.3]. Returns (c, info); (None, None) below 3 keyframes."""
    if len(moge_ratios) < 3:
        return None, None
    raw = float(np.median(moge_ratios))
    c = float(np.clip(raw, 0.5, 1.3))
    return c, f"c={c:.3f} (raw median {raw:.3f} over {len(moge_ratios)} keyframes)"


def _resolve_anchor(scale_anchor, moge_ratios, contact_pairs, cam_fov_deg=None, contact_clip=None):
    """The object-scale correction of an anchor mode. Returns (c, info); c=None: no correction.

    "moge": the MoGe metric scale (_moge_scale). "contact_follow": the hand-contact anchor
    (_resolve_contact_anchor); where it abstains (narrow FoV, too few contacts) the MoGe anchor.
    """
    if scale_anchor == "contact_follow":
        c, info = _resolve_contact_anchor(contact_pairs, cam_fov_deg=cam_fov_deg, clip=contact_clip)
        if c is not None:
            return c, info
        c_m, info_m = _moge_scale(moge_ratios)
        return c_m, f"FALLBACK→moge ({info}): {info_m}"
    if scale_anchor == "moge":
        return _moge_scale(moge_ratios)
    raise ValueError(f"unknown scale_anchor {scale_anchor!r}")
