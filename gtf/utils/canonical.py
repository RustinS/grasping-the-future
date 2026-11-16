"""Shared utilities for canonical export and dataset ingestion.

Provides rotation conversions, rigid Procrustes alignment, coordinate frame
transforms, gravity estimation, canonical.npz validation, MANO helpers and mesh I/O.
"""

import os
import zipfile

import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Rotation conversions
# ---------------------------------------------------------------------------


def rotmat_to_aa(R):
    """Rotation matrices to axis-angle vectors.

    Args:
        R: (*, 3, 3) rotation matrices.
    Returns:
        (*, 3) axis-angle vectors.
    """
    shape = R.shape[:-2]
    return Rotation.from_matrix(R.reshape(-1, 3, 3)).as_rotvec().reshape(*shape, 3).astype(R.dtype)


# ---------------------------------------------------------------------------
# Procrustes alignment
# ---------------------------------------------------------------------------


def rigid_procrustes(source, target, weights=None):
    """Rigid Procrustes alignment (rotation + translation, no scaling).

    Finds R, t minimizing  sum_i w_i ||R @ source_i + t - target_i||^2.

    Args:
        source: (N, 3) source points.
        target: (N, 3) target points.
        weights: optional (N,) non-negative weights. Default: uniform.
    Returns:
        R: (3, 3) rotation matrix (det = +1).
        t: (3,) translation vector.
        rmse: float, root mean square error after alignment.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    N = source.shape[0]

    if weights is None:
        w = np.ones(N, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)

    w_sum = w.sum()
    mu_s = (w[:, None] * source).sum(axis=0) / w_sum
    mu_t = (w[:, None] * target).sum(axis=0) / w_sum

    X = source - mu_s
    Y = target - mu_t

    H = (w[:, None] * X).T @ Y  # (3, 3) cross-covariance
    U, _S, Vt = np.linalg.svd(H)

    # Correct reflection: ensure det(R) = +1
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T

    t = mu_t - R @ mu_s

    residuals = (R @ source.T).T + t - target
    rmse = float(np.sqrt(np.mean(np.sum(residuals**2, axis=-1))))

    return R.astype(np.float32), t.astype(np.float32), rmse


# ---------------------------------------------------------------------------
# Coordinate frame transforms
# ---------------------------------------------------------------------------


def transform_to_anchor_frame(verts, cam_to_world, frame_indices, anchor_frame_index):
    """Transform per-frame vertices from per-frame camera space to anchor camera frame.

    Args:
        verts: (T_kf, V, 3) vertices in per-frame camera space.
        cam_to_world: (T_video, 4, 4) dense camera-to-world transforms (all video frames).
        frame_indices: (T_kf,) int array of video frame numbers for each keyframe.
        anchor_frame_index: int, video frame number of the anchor frame.
    Returns:
        (T_kf, V, 3) vertices in anchor camera frame.
    Raises:
        ValueError: if anchor_frame_index >= len(cam_to_world).
    """
    T_kf = verts.shape[0]
    T_video = cam_to_world.shape[0]

    if anchor_frame_index >= T_video:
        raise ValueError(f"anchor_frame_index {anchor_frame_index} >= cam_to_world length {T_video}")

    c2w_anchor = cam_to_world[anchor_frame_index].astype(np.float64)  # (4, 4)
    w2c_anchor = np.linalg.inv(c2w_anchor)  # (4, 4)

    result = np.empty_like(verts)
    for t in range(T_kf):
        fi = int(frame_indices[t])
        if fi >= T_video:
            # Frame beyond camera pose range — identity fallback
            result[t] = verts[t]
            continue

        c2w_t = cam_to_world[fi].astype(np.float64)  # (4, 4)
        T_anchor_from_t = w2c_anchor @ c2w_t  # (4, 4)

        R = T_anchor_from_t[:3, :3]  # (3, 3)
        tvec = T_anchor_from_t[:3, 3]  # (3,)

        result[t] = (verts[t].astype(np.float64) @ R.T + tvec).astype(verts.dtype)

    return result


# ---------------------------------------------------------------------------
# Gravity estimation
# ---------------------------------------------------------------------------
#
# Both estimators return ``(gravity_direction (3,), support_plane_height)`` in
# camera space (OpenCV axes), gravity pointing down, or None.
#
# ``estimate_gravity`` fits the single largest plane of one depth map by RANSAC
# and orients its normal by camera Y (the support normal points to -Y, up in the
# image). The motion-capture loaders use it.
#
# ``estimate_gravity_support`` is the extraction pipeline's estimator. It pools
# plane hypotheses from several depth frames expressed in one anchor camera frame:
#
#   * sequential RANSAC extracts several planes per frame;
#   * each normal is oriented by visibility: an observed surface faces the
#     camera, so its outward normal satisfies ``n . (cam_center - centroid) > 0``,
#     and for a surface seen from above (floor, table, counter) that is the up
#     direction;
#   * a plane is kept only if the tracked hand / object sit above it at a
#     plausible height, and a plane the object comes down onto ranks first;
#   * it returns None when no plane passes.


def estimate_gravity(depth_map, intrinsics, n_ransac=500, inlier_thresh=0.02, min_inliers_frac=0.1):
    """Estimate gravity direction and support plane from a depth map via RANSAC.

    Unprojects depth to 3D, fits a dominant plane, returns the plane normal
    (oriented downward) and the plane height.

    Args:
        depth_map: (H, W) float depth in meters.
        intrinsics: (3, 3) camera intrinsic matrix.
        n_ransac: number of RANSAC iterations.
        inlier_thresh: inlier distance threshold in meters.
        min_inliers_frac: minimum fraction of valid points that must be inliers.
    Returns:
        Tuple of (gravity_direction (3,), support_plane_height float) or None.
    """
    H, W = depth_map.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    valid = (depth_map > 0.1) & (depth_map < 10.0) & np.isfinite(depth_map)
    z = depth_map[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    pts = np.stack([x, y, z], axis=-1)  # (N, 3)

    N = pts.shape[0]
    if N < 100:
        return None

    min_inliers = int(N * min_inliers_frac)
    best_inliers = 0
    best_normal = None
    best_d = None

    rng = np.random.RandomState(42)
    for _ in range(n_ransac):
        idx = rng.choice(N, 3, replace=False)
        p0, p1, p2 = pts[idx[0]], pts[idx[1]], pts[idx[2]]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal = normal / norm
        d = -np.dot(normal, p0)

        dists = np.abs(pts @ normal + d)
        n_inliers = int(np.sum(dists < inlier_thresh))

        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_normal = normal
            best_d = d

    if best_inliers < min_inliers or best_normal is None:
        return None

    # Orient normal: in camera space, gravity is roughly +Y (downward in image)
    # The support plane normal should point upward (against gravity)
    if best_normal[1] > 0:
        best_normal = -best_normal
        best_d = -best_d

    # gravity_direction points downward (opposite to plane normal)
    gravity_direction = -best_normal
    support_plane_height = -best_d  # height of the plane along normal direction

    return gravity_direction.astype(np.float32), float(support_plane_height)


# metadata.json ``gravity_source`` of an extracted segment with support-plane gravity.
SUPPORT_PLANE_SOURCE = "support_plane_consensus"

SUPPORT_PLANE_PARAMS = {
    "n_ransac": 600,  # RANSAC trials per plane hypothesis
    "inlier_thresh": 0.02,  # m, plane inlier band (as in estimate_gravity)
    "n_planes": 4,  # sequential hypotheses extracted per frame
    "min_plane_frac": 0.04,  # a hypothesis must hold >=4% of the frame's points
    "max_points": 40000,  # deterministic subsample per frame
    "merge_deg": 12.0,  # cross-frame hypothesis clustering tolerance
    "min_frames_frac": 0.5,  # a direction must appear in >=50% of the frames
    "visibility_margin": 0.10,  # |cos| between normal and view ray; below it visibility does not fix the sign
    "below_tol": 0.05,  # m, tolerated interaction mass below the plane
    "min_above_frac": 0.90,  # fraction of interaction points that must be above
    "max_interaction_h": 1.80,  # m, plausible interaction height above a support surface
    "contact_max_h": 0.15,  # m, "the object rests on it" bonus threshold
    "contact_pct": 5.0,  # percentile of per-frame object heights used for the contact test
    "min_scene_above_frac": 0.60,  # a support surface has most of the scene above it
    # Camera-orientation prior. Geometry alone cannot always separate a ground
    # plane from a large fronto-parallel surface. A hand-manipulation camera is
    # held roughly level (small roll about the optical axis) and does not look
    # steeply up at the interaction; the bounds are loose enough for a top-down shot.
    "prior_max_roll_deg": 45.0,  # |atan2(g_x, g_y)|, only tested when the horizontal part is meaningful
    "prior_min_gxy": 0.15,  # below this, gravity is ~along the optical axis and roll is undefined
    "prior_max_up_pitch_deg": 30.0,  # how far the camera may look upward
    # Gravity within ~37 deg of the optical axis means the camera looks steeply
    # down at the support surface. A fronto-parallel backdrop (wall, counter
    # front, the subject's body) gives the same picture: a large plane faces the
    # camera, the whole scene sits in front of it and the interaction is a few
    # centimeters off it. One depth stream cannot separate the two, so such a
    # candidate is refused. 0.80 is 53 deg of down-pitch.
    "prior_max_abs_gz": 0.80,
}


def _unproject_depth_points(depth_map, intrinsics, max_points=None, rng_seed=0, min_depth=0.1, max_depth=10.0):
    """Unproject a depth map to a (N, 3) camera-space point cloud.

    Deterministic: the optional subsample uses a seeded RandomState, so the same
    inputs always give the same points.

    Args:
        depth_map: (H, W) float depth in meters.
        intrinsics: (3, 3) camera intrinsic matrix.
        max_points: optional cap; points are subsampled without replacement.
        rng_seed: seed for the subsample.
        min_depth / max_depth: validity band in meters.
    Returns:
        (N, 3) float64 points, camera space (OpenCV: +X right, +Y down, +Z forward).
    """
    depth_map = np.asarray(depth_map)
    H, W = depth_map.shape
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])

    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    valid = (depth_map > min_depth) & (depth_map < max_depth) & np.isfinite(depth_map)
    z = depth_map[valid].astype(np.float64)
    x = (u[valid].astype(np.float64) - cx) * z / fx
    y = (v[valid].astype(np.float64) - cy) * z / fy
    pts = np.stack([x, y, z], axis=-1)

    if max_points is not None and pts.shape[0] > max_points:
        idx = np.random.RandomState(rng_seed).choice(pts.shape[0], max_points, replace=False)
        idx.sort()
        pts = pts[idx]
    return pts


def _fit_plane_ransac(pts, n_ransac, inlier_thresh, rng):
    """Single max-inlier plane by RANSAC. Returns (normal, d, inlier_mask) or None."""
    N = pts.shape[0]
    if N < 3:
        return None
    best_n, best_d, best_cnt = None, None, 0
    for _ in range(n_ransac):
        idx = rng.choice(N, 3, replace=False)
        p0, p1, p2 = pts[idx[0]], pts[idx[1]], pts[idx[2]]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal = normal / norm
        d = -float(normal @ p0)
        cnt = int(np.count_nonzero(np.abs(pts @ normal + d) < inlier_thresh))
        if cnt > best_cnt:
            best_cnt, best_n, best_d = cnt, normal, d
    if best_n is None or best_cnt < 3:
        return None

    # Least-squares refit on the consensus set (RANSAC's 3-point normal is noisy)
    mask = np.abs(pts @ best_n + best_d) < inlier_thresh
    P = pts[mask]
    c = P.mean(axis=0)
    _u, _s, vt = np.linalg.svd(P - c, full_matrices=False)
    normal = vt[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    d = -float(normal @ c)
    mask = np.abs(pts @ normal + d) < inlier_thresh
    return normal, d, mask


def _plane_hypotheses(pts, n_planes, n_ransac, inlier_thresh, min_plane_frac, rng_seed=42):
    """Sequential RANSAC: extract up to n_planes planes, removing inliers each round.

    Returns a list of dicts with keys ``normal`` (3,), ``d``, ``frac``, ``centroid`` (3,).
    """
    rng = np.random.RandomState(rng_seed)
    remaining = np.arange(pts.shape[0])
    total = pts.shape[0]
    out = []
    for _ in range(n_planes):
        if remaining.size < max(3, int(total * min_plane_frac)):
            break
        fit = _fit_plane_ransac(pts[remaining], n_ransac, inlier_thresh, rng)
        if fit is None:
            break
        normal, d, mask = fit
        frac = float(mask.sum()) / float(total)
        if frac < min_plane_frac:
            break
        inlier_pts = pts[remaining][mask]
        out.append(
            {
                "normal": normal.astype(np.float64),
                "d": float(d),
                "frac": frac,
                "centroid": inlier_pts.mean(axis=0),
            }
        )
        remaining = remaining[~mask]
    return out


def _orient_by_visibility(normal, d, centroid, cam_center):
    """Flip a plane normal so it faces the camera that observed it.

    A surface is only visible from the side its outward normal points to, so the
    outward normal of an observed patch satisfies ``n . (cam_center - centroid) > 0``.
    For a support surface seen from above (floor, table, counter) the outward
    normal IS the up direction, which makes this a sign rule with no gravity
    prior in it.

    Returns (normal, d, margin) where ``margin`` is the |cosine| between the
    normal and the view ray -- near 0 the plane is seen edge-on and the sign is
    not determined by visibility.
    """
    view = np.asarray(cam_center, dtype=np.float64) - np.asarray(centroid, dtype=np.float64)
    vn = np.linalg.norm(view)
    if vn < 1e-9:
        return np.asarray(normal, dtype=np.float64), float(d), 0.0
    cos = float(np.asarray(normal, dtype=np.float64) @ (view / vn))
    if cos < 0.0:
        return -np.asarray(normal, dtype=np.float64), -float(d), abs(cos)
    return np.asarray(normal, dtype=np.float64), float(d), abs(cos)


def estimate_gravity_support(
    frames,
    hand_points=None,
    object_points=None,
    return_report=False,
):
    """Support-plane gravity estimation (see the note above :func:`estimate_gravity`).

    Pools plane hypotheses from one or more depth frames expressed in a single
    anchor camera frame, orients each by visibility, and selects the hypothesis
    the tracked interaction rests on. Returns ``None`` when no hypothesis is
    supported.

    Args:
        frames: list of dicts, one per depth frame, with keys
            ``depth`` (H, W) meters and ``intrinsics`` (3, 3); optional
            ``R`` (3, 3) / ``t`` (3,) mapping that frame's camera space into the
            anchor frame (identity when omitted -- correct for a static camera).
        hand_points: optional list (per frame index, any length) of (N, 3) hand
            points in the anchor frame, or a single (N, 3) array. Frames with no
            valid hand may be None/empty.
        object_points: same, for the manipulated object. The contact test uses
            the object when present, else the hand.
        return_report: also return a JSON-safe diagnostic dict.
    Returns:
        ``(gravity_direction (3,) float32, support_plane_height float)`` -- the
        same contract as :func:`estimate_gravity`, gravity pointing DOWN -- or
        ``None``.  With ``return_report`` the return is
        ``(result_or_None, report_dict)``.
    """
    p = SUPPORT_PLANE_PARAMS

    report = {"n_frames": 0, "n_hypotheses": 0, "n_clusters": 0, "rejects": [], "accepted": None, "source": SUPPORT_PLANE_SOURCE}

    def _fin(result):
        return (result, report) if return_report else result

    if not frames:
        report["rejects"].append("no_frames")
        return _fin(None)

    def _pts_for(seq, i):
        if seq is None:
            return None
        if isinstance(seq, np.ndarray):
            return seq if seq.size else None
        if i < len(seq) and seq[i] is not None and len(seq[i]):
            return np.asarray(seq[i], dtype=np.float64).reshape(-1, 3)
        return None

    # --- 1. per-frame hypotheses, lifted into the anchor frame ---------------
    hyps = []
    scene_chunks = []
    for i, fr in enumerate(frames):
        depth = fr.get("depth")
        K = fr.get("intrinsics")
        if depth is None or K is None:
            continue
        pts = _unproject_depth_points(depth, K, max_points=p["max_points"], rng_seed=42 + i)
        if pts.shape[0] < 100:
            continue
        R = fr.get("R")
        t = fr.get("t")
        if R is not None:
            R = np.asarray(R, dtype=np.float64)
            t = np.zeros(3) if t is None else np.asarray(t, dtype=np.float64)
            pts_a = pts @ R.T + t
            cam_center = t
        else:
            pts_a = pts
            cam_center = np.zeros(3)
        scene_chunks.append(pts_a)
        for h in _plane_hypotheses(pts_a, p["n_planes"], p["n_ransac"], p["inlier_thresh"], p["min_plane_frac"], rng_seed=42 + i):
            n, d, margin = _orient_by_visibility(h["normal"], h["d"], h["centroid"], cam_center)
            hyps.append({"normal": n, "d": d, "frac": h["frac"], "margin": margin, "frame": i})
        report["n_frames"] += 1

    report["n_hypotheses"] = len(hyps)
    if not hyps:
        report["rejects"].append("no_plane_hypotheses")
        return _fin(None)

    scene = np.concatenate(scene_chunks, axis=0) if scene_chunks else np.zeros((0, 3))

    # --- 2. cluster hypotheses across frames by direction --------------------
    # Offsets are kept per member: a floor and the table above it are parallel,
    # so they belong to the same direction cluster but are different surfaces,
    # and the support tests must be evaluated against each surface separately.
    cos_tol = np.cos(np.radians(p["merge_deg"]))
    hyps.sort(key=lambda h: -h["frac"])
    clusters = []
    for h in hyps:
        for c in clusters:
            if float(c["normal"] @ h["normal"]) >= cos_tol:
                c["members"].append(h)
                w = np.array([m["frac"] for m in c["members"]])
                N = np.stack([m["normal"] for m in c["members"]])
                nm = (w[:, None] * N).sum(axis=0)
                c["normal"] = nm / (np.linalg.norm(nm) + 1e-12)
                break
        else:
            clusters.append({"normal": h["normal"].copy(), "members": [h]})
    report["n_clusters"] = len(clusters)

    # --- 3. gate + score each cluster ---------------------------------------
    n_frames = max(1, report["n_frames"])
    hand_all, obj_all, obj_by_frame = [], [], []
    for i in range(max(n_frames, len(frames))):
        hp = _pts_for(hand_points, i)
        op = _pts_for(object_points, i)
        if hp is not None:
            hand_all.append(hp)
        if op is not None:
            obj_all.append(op)
            obj_by_frame.append(op)
        elif hp is not None:
            obj_by_frame.append(hp)  # contact test falls back to the hand
    hand_all = np.concatenate(hand_all, axis=0) if hand_all else np.zeros((0, 3))
    obj_all = np.concatenate(obj_all, axis=0) if obj_all else np.zeros((0, 3))
    interaction = np.concatenate([a for a in (hand_all, obj_all) if a.size], axis=0) if (hand_all.size or obj_all.size) else np.zeros((0, 3))

    max_roll = np.radians(p["prior_max_roll_deg"])
    min_gz = -np.sin(np.radians(p["prior_max_up_pitch_deg"]))

    scored = []
    for ci, c in enumerate(clusters):
        n = c["normal"]
        base = {
            "cluster": ci,
            "normal": [float(x) for x in n],
            "mass": float(sum(m["frac"] for m in c["members"]) / n_frames),
            "n_frames": len({m["frame"] for m in c["members"]}),
            "visibility_margin": float(max(m["margin"] for m in c["members"])),
        }
        dir_fails = []
        if base["mass"] < p["min_plane_frac"]:
            dir_fails.append("mass")
        if base["n_frames"] < max(1, int(np.ceil(p["min_frames_frac"] * n_frames))):
            dir_fails.append("frames")
        if base["visibility_margin"] < p["visibility_margin"]:
            dir_fails.append("edge_on")  # sign not determined by visibility
        g = -n  # candidate gravity (down) in camera space
        base["implied_roll_deg"] = float(np.degrees(np.arctan2(g[0], g[1])))
        base["implied_g_z"] = float(g[2])
        if float(np.hypot(g[0], g[1])) >= p["prior_min_gxy"] and abs(np.arctan2(g[0], g[1])) > max_roll:
            dir_fails.append("prior_roll")
        if g[2] < min_gz:
            dir_fails.append("prior_looks_up")
        if abs(g[2]) > p["prior_max_abs_gz"]:
            dir_fails.append("prior_optical_axis")
        if dir_fails:
            base["fails"] = dir_fails
            report["rejects"].append(base)
            continue

        # Evaluate every distinct surface offset in this direction and keep the
        # best-supported one (floor vs table-top vs shelf all share a normal).
        best_member = None
        member_fails = []
        for m in c["members"]:
            d = m["d"]
            info = dict(base, d=float(d))
            fails = []
            if scene.size:
                h_scene = scene @ n + d
                info["scene_above_frac"] = float(np.mean(h_scene > -p["below_tol"]))
                if info["scene_above_frac"] < p["min_scene_above_frac"]:
                    fails.append("scene_below")
            if interaction.size:
                h_int = interaction @ n + d
                info["above_frac"] = float(np.mean(h_int > -p["below_tol"]))
                info["median_h"] = float(np.median(h_int))
                if info["above_frac"] < p["min_above_frac"]:
                    fails.append("interaction_below")
                if info["median_h"] > p["max_interaction_h"]:
                    fails.append("too_high")
                if obj_by_frame:
                    info["min_contact_h"] = float(np.min([float(np.percentile(op @ n + d, p["contact_pct"])) for op in obj_by_frame]))
            else:
                fails.append("no_interaction_points")
            info["fails"] = fails
            if fails:
                member_fails.append(info)
                continue
            # Rank supported surfaces: the interaction should sit low over the
            # surface it uses, the surface should carry mass, and a surface the
            # object rests on gets double weight.
            clearance = max(0.0, info.get("median_h", 0.0))
            rests = info.get("min_contact_h", 1e9) <= p["contact_max_h"]
            info["rests_on"] = bool(rests)
            info["score"] = float(info["mass"] * info.get("scene_above_frac", 1.0) ** 2 * np.exp(-clearance / 1.0) * (2.0 if rests else 1.0))
            if best_member is None or info["score"] > best_member["score"]:
                best_member = info
        if best_member is None:
            base["fails"] = ["no_supported_surface"]
            base["member_fails"] = sorted({f for mf in member_fails for f in mf["fails"]})
            base["best_member_stats"] = min(member_fails, key=lambda mf: len(mf["fails"])) if member_fails else None
            report["rejects"].append(base)
            continue
        scored.append(best_member)

    if not scored:
        return _fin(None)

    best = max(scored, key=lambda s: s["score"])
    n = np.asarray(best["normal"], dtype=np.float64)
    gravity_direction = (-n / (np.linalg.norm(n) + 1e-12)).astype(np.float32)
    support_plane_height = float(-best["d"])
    best["candidates"] = len(scored)
    report["accepted"] = best
    return _fin((gravity_direction, support_plane_height))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# Required keys for canonical.npz (both sides always present)
_REQUIRED_KEYS = {
    "frame_indices",
    "fps",
    "obj_canonical_verts",
    "obj_faces",
    "obj_rotation",
    "obj_translation",
    "obj_scale",
    "obj_verts",
}

_PER_SIDE_KEYS = {"orient", "pose", "cam_t", "betas", "valid", "verts"}


def _npz_array_meta(path, key):
    with zipfile.ZipFile(path) as zf:
        name = f"{key}.npy"
        with zf.open(name) as f:
            version = np.lib.format.read_magic(f)
            shape, _, dtype = np.lib.format._read_array_header(f, version)
    return tuple(shape), np.dtype(dtype)


def validate_canonical_npz(path, full_scan=True):
    """Validate a canonical.npz file for schema compliance.

    Args:
        path: path to canonical.npz file.
        full_scan: run the checks that read the large arrays (NaN/Inf, zeroed invalid hand
            frames, extrinsics values); with False, shapes and dtypes come from the array headers.
    Returns:
        Tuple of (is_valid: bool, errors: list[str]).
    """
    errors = []

    if not os.path.exists(path):
        return False, [f"File not found: {path}"]

    try:
        d = np.load(path, allow_pickle=False)
    except Exception as e:
        return False, [f"Failed to load: {e}"]

    keys = set(d.files)

    # Check required keys
    for k in _REQUIRED_KEYS:
        if k not in keys:
            errors.append(f"Missing required key: {k}")

    # Check both sides present
    for side in ("left", "right"):
        for suffix in _PER_SIDE_KEYS:
            k = f"hand_{side}_{suffix}"
            if k not in keys:
                errors.append(f"Missing hand key: {k}")

    if errors:
        return False, errors

    def _shape_dtype(k):
        if full_scan:
            arr = d[k]
            return arr.shape, arr.dtype
        return _npz_array_meta(path, k)

    # Shape consistency
    T = _shape_dtype("frame_indices")[0][0]
    V_obj = _shape_dtype("obj_canonical_verts")[0][0]

    shape_checks = {
        "frame_indices": (T,),
        "obj_rotation": (T, 3),
        "obj_translation": (T, 3),
        "obj_scale": (3,),
        "obj_verts": (T, V_obj, 3),
        "obj_canonical_verts": (V_obj, 3),
    }
    for k, expected in shape_checks.items():
        shape, _ = _shape_dtype(k)
        if shape != expected:
            errors.append(f"{k}: expected shape {expected}, got {shape}")

    for side in ("left", "right"):
        prefix = f"hand_{side}"
        shape, _ = _shape_dtype(f"{prefix}_orient")
        if shape != (T, 3):
            errors.append(f"{prefix}_orient: expected ({T}, 3), got {shape}")
        shape, _ = _shape_dtype(f"{prefix}_pose")
        if shape != (T, 45):
            errors.append(f"{prefix}_pose: expected ({T}, 45), got {shape}")
        shape, _ = _shape_dtype(f"{prefix}_cam_t")
        if shape != (T, 3):
            errors.append(f"{prefix}_cam_t: expected ({T}, 3), got {shape}")
        shape, _ = _shape_dtype(f"{prefix}_betas")
        if shape != (10,):
            errors.append(f"{prefix}_betas: expected (10,), got {shape}")
        shape, _ = _shape_dtype(f"{prefix}_valid")
        if shape != (T,):
            errors.append(f"{prefix}_valid: expected ({T},), got {shape}")
        shape, _ = _shape_dtype(f"{prefix}_verts")
        if shape[0] != T or shape[2] != 3:
            errors.append(f"{prefix}_verts: expected ({T}, 778, 3), got {shape}")

    # Dtype checks
    _expected_dtypes = {
        "frame_indices": "i",  # integer
        "fps": "f",  # float
        "obj_canonical_verts": "f",
        "obj_faces": "i",
        "obj_rotation": "f",
        "obj_translation": "f",
        "obj_scale": "f",
        "obj_verts": "f",
    }
    for side in ("left", "right"):
        _expected_dtypes[f"hand_{side}_orient"] = "f"
        _expected_dtypes[f"hand_{side}_pose"] = "f"
        _expected_dtypes[f"hand_{side}_cam_t"] = "f"
        _expected_dtypes[f"hand_{side}_betas"] = "f"
        _expected_dtypes[f"hand_{side}_verts"] = "f"
        _expected_dtypes[f"hand_{side}_valid"] = "b"  # bool
    for k, expected_kind in _expected_dtypes.items():
        if k in keys:
            _, dtype = _shape_dtype(k)
            if dtype.kind != expected_kind:
                errors.append(f"{k}: expected dtype kind '{expected_kind}', got '{dtype}'")

    # Value / temporal checks that the dtype-kind checks can't see.
    if "frame_indices" in keys:
        _fi = np.asarray(d["frame_indices"])
        if _fi.ndim != 1:
            errors.append(f"frame_indices: expected 1-D, got {_fi.shape}")
        elif _fi.shape[0] >= 2 and not (np.diff(_fi) > 0).all():
            errors.append("frame_indices: not strictly increasing")
    if "fps" in keys:
        _fps = float(np.asarray(d["fps"]).reshape(-1)[0])
        if not (_fps > 0):
            errors.append(f"fps: expected > 0, got {_fps}")
    if "source_frame_indices" in keys:
        _sfi = np.asarray(d["source_frame_indices"])
        if _sfi.shape[0] != T:
            errors.append(f"source_frame_indices: expected length {T}, got {_sfi.shape}")

    # NaN / Inf checks on all float arrays
    if full_scan:
        for k in d.files:
            arr = d[k]
            if arr.dtype.kind == "f":
                if np.any(np.isnan(arr)):
                    errors.append(f"{k}: contains NaN")
                if np.any(np.isinf(arr)):
                    errors.append(f"{k}: contains Inf")

    # Invalid frame consistency: if valid=False, all hand params/verts must be zero
    if full_scan:
        for side in ("left", "right"):
            prefix = f"hand_{side}"
            valid = d[f"{prefix}_valid"]
            invalid = ~valid
            if np.any(invalid):
                for suffix in ("orient", "pose", "cam_t", "verts"):
                    arr = d[f"{prefix}_{suffix}"]
                    if np.any(arr[invalid] != 0):
                        errors.append(f"{prefix}_{suffix}: nonzero values on invalid frames")

    # Scene fields: each is independently optional
    if "gravity_direction" in keys:
        shape, dtype = _shape_dtype("gravity_direction")
        if shape != (3,):
            errors.append(f"gravity_direction: expected (3,), got {shape}")
        if dtype.kind != "f":
            errors.append(f"gravity_direction: expected float dtype, got {dtype}")
    if "support_plane_height" in keys:
        shape, dtype = _shape_dtype("support_plane_height")
        if len(shape) != 0:
            errors.append(f"support_plane_height: expected scalar, got shape {shape}")
        if dtype.kind != "f":
            errors.append(f"support_plane_height: expected float dtype, got {dtype}")

    # Camera intrinsics: optional, but if present must be 3x3 float with positive fx, fy
    if "camera_K" in keys:
        shape, dtype = _shape_dtype("camera_K")
        if shape != (3, 3):
            errors.append(f"camera_K: expected (3,3), got {shape}")
        elif dtype.kind != "f":
            errors.append(f"camera_K: expected float dtype, got {dtype}")
        else:
            K = d["camera_K"]
            if K[0, 0] <= 0 or K[1, 1] <= 0:
                errors.append(f"camera_K: fx={K[0, 0]}, fy={K[1, 1]} must be positive")

    # Optional present-anchor fields. If any world pose field is present, require
    # the complete set so training can re-anchor without dataset-specific logic.
    present_anchor_fields = {
        "camera_extrinsics",
        "hand_left_orient_world",
        "hand_left_cam_t_world",
        "hand_right_orient_world",
        "hand_right_cam_t_world",
        "obj_rotation_world",
        "obj_translation_world",
    }
    if keys & present_anchor_fields:
        missing = sorted(present_anchor_fields - keys)
        if missing:
            errors.append(f"present-anchor fields incomplete; missing {missing}")
        else:
            shape, dtype = _shape_dtype("camera_extrinsics")
            if shape not in ((T, 4, 4), (4, 4)):
                errors.append(f"camera_extrinsics: expected ({T},4,4) or (4,4), got {shape}")
            elif dtype.kind != "f":
                errors.append(f"camera_extrinsics: expected float dtype, got {dtype}")
            elif full_scan:
                E = d["camera_extrinsics"]
                E_check = E[None] if E.shape == (4, 4) else E
                if not np.allclose(E_check[:, 3, :], np.array([0.0, 0.0, 0.0, 1.0], dtype=E_check.dtype), atol=1e-4):
                    errors.append("camera_extrinsics: last row must be [0,0,0,1]")
                det = np.linalg.det(E_check[:, :3, :3])
                if not np.all(np.isfinite(det)) or np.max(np.abs(det - 1.0)) > 1e-2:
                    errors.append("camera_extrinsics: rotation determinant must be near +1")

            for side in ("left", "right"):
                for suffix in ("orient_world", "cam_t_world"):
                    k = f"hand_{side}_{suffix}"
                    shape, dtype = _shape_dtype(k)
                    if shape != (T, 3):
                        errors.append(f"{k}: expected ({T}, 3), got {shape}")
                    if dtype.kind != "f":
                        errors.append(f"{k}: expected float dtype, got {dtype}")

            for k in ("obj_rotation_world", "obj_translation_world"):
                shape, dtype = _shape_dtype(k)
                if shape != (T, 3):
                    errors.append(f"{k}: expected ({T}, 3), got {shape}")
                if dtype.kind != "f":
                    errors.append(f"{k}: expected float dtype, got {dtype}")

            if "gravity_world" in keys:
                shape, dtype = _shape_dtype("gravity_world")
                if shape != (3,):
                    errors.append(f"gravity_world: expected (3,), got {shape}")
                if dtype.kind != "f":
                    errors.append(f"gravity_world: expected float dtype, got {dtype}")

    # Valid mask: at least one side should have valid frames (sanity)
    any_valid = False
    for side in ("left", "right"):
        if np.any(d[f"hand_{side}_valid"]):
            any_valid = True
    if not any_valid:
        errors.append("WARNING: no valid hand frames for either side")

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# MANO PCA and mirroring
# ---------------------------------------------------------------------------

_mano_cache = {}


def _preload_torch_cuda_deps():
    """Preload the PyTorch wheel's libnvJitLink when present.

    An older system libnvJitLink found ahead of the wheel's CUDA libraries makes
    ``import torch`` fail, even for CPU-only MANO FK.
    """
    import ctypes
    import sys

    py = f"python{sys.version_info.major}.{sys.version_info.minor}"
    lib = os.path.join(sys.prefix, "lib", py, "site-packages", "nvidia", "nvjitlink", "lib", "libnvJitLink.so.12")
    if os.path.exists(lib):
        try:
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


def _load_mano_pca(mano_model_path):
    """Load and cache MANO PCA components and hand mean from pkl."""
    path_str = str(mano_model_path)
    if path_str not in _mano_cache:
        import pickle

        with open(path_str, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        _mano_cache[path_str] = {
            "hands_components": np.array(data["hands_components"], dtype=np.float32),  # (45, 45)
            "hands_mean": np.array(data["hands_mean"], dtype=np.float32),  # (45,)
        }
    return _mano_cache[path_str]


def mano_pca_to_aa(pca_coeffs, n_comps, mano_model_path):
    """Expand MANO PCA coefficients to full 45D axis-angle finger pose.

    Args:
        pca_coeffs: (*, n_comps) PCA coefficients.
        n_comps: number of PCA components (e.g. 15 for HOT3D, 45 for DexYCB).
        mano_model_path: path to MANO_RIGHT.pkl.
    Returns:
        (*, 45) full finger pose in axis-angle.
    """
    cache = _load_mano_pca(mano_model_path)
    components = cache["hands_components"][:n_comps]  # (n_comps, 45)
    mean = cache["hands_mean"]  # (45,)
    pca_coeffs = np.asarray(pca_coeffs, dtype=np.float32)
    shape = pca_coeffs.shape[:-1]
    flat = pca_coeffs.reshape(-1, n_comps)
    aa_flat = flat @ components + mean  # (N, 45)
    return aa_flat.reshape(*shape, 45)


def mirror_left_hand_params(orient_aa, pose_aa, cam_t):
    """Convert native left-MANO params to right-model x-mirror convention.

    Applies M @ R @ M similarity transform (M = diag(-1,1,1)) to orient
    and each of 15 joint rotation matrices, and negates cam_t x-component.

    Args:
        orient_aa: (T, 3) global orient in left-MANO convention.
        pose_aa: (T, 45) finger pose in left-MANO convention.
        cam_t: (T, 3) translation in camera space.
    Returns:
        Tuple of (orient_aa_right, pose_aa_right, cam_t_right).
    """
    M = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)

    # Orient: (T, 3) → rotmat → mirror → aa
    orient_mat = Rotation.from_rotvec(orient_aa).as_matrix().astype(np.float32)
    orient_mirrored = np.einsum("ij,...jk,kl->...il", M, orient_mat, M)
    orient_out = Rotation.from_matrix(orient_mirrored).as_rotvec().astype(np.float32)

    # Pose: (T, 45) → (T, 15, 3) → rotmats → mirror → aa
    T = pose_aa.shape[0]
    pose_rotvec = pose_aa.reshape(T, 15, 3)
    pose_mat = Rotation.from_rotvec(pose_rotvec.reshape(-1, 3)).as_matrix().astype(np.float32).reshape(T, 15, 3, 3)
    pose_mirrored = np.einsum("ij,...jk,kl->...il", M, pose_mat, M)
    pose_out = Rotation.from_matrix(pose_mirrored.reshape(-1, 3, 3)).as_rotvec().astype(np.float32).reshape(T, 45)

    # Translation: negate x
    cam_t_out = cam_t.copy()
    cam_t_out[:, 0] = -cam_t_out[:, 0]

    return orient_out, pose_out, cam_t_out


def _import_manolayer():
    """Import manopth's ManoLayer lazily (torch is only needed by the callers that run MANO)."""
    from manopth.manolayer import ManoLayer

    return ManoLayer


def mano_fk_verts(orient_aa, pose_aa, betas, cam_t, side, mano_dir):
    """Run MANO forward kinematics to reconstruct per-frame vertices.

    Always uses MANO_RIGHT.pkl. For left hands, x-mirrors the output.

    Args:
        orient_aa: (T, 3) global orientation axis-angle.
        pose_aa: (T, 45) finger pose axis-angle.
        betas: (10,) shape parameters.
        cam_t: (T, 3) translation.
        side: 'left' or 'right'.
        mano_dir: path to directory containing MANO_RIGHT.pkl.
    Returns:
        (T, 778, 3) float32 vertices in camera space.
    """
    _preload_torch_cuda_deps()
    import torch

    ManoLayer = _import_manolayer()

    T = orient_aa.shape[0]
    mano_layer = ManoLayer(side="right", use_pca=False, ncomps=45, flat_hand_mean=True, mano_root=str(mano_dir))

    go = torch.from_numpy(orient_aa).float()
    hp = torch.from_numpy(pose_aa).float()
    bt = torch.from_numpy(betas).float().unsqueeze(0).expand(T, -1)
    ct = torch.from_numpy(cam_t).float()  # manopth expects meters for translation

    with torch.no_grad():
        full_pose = torch.cat([go, hp], dim=1)  # (T, 48)
        verts, _ = mano_layer(full_pose, bt, ct)
        verts = verts.numpy().astype(np.float32) / 1000.0  # manopth outputs in mm → meters

    if side == "left":
        verts[:, :, 0] *= -1

    return verts


def mano_fk_joints(orient_aa, pose_aa, betas, cam_t, side, mano_dir):
    """Run MANO forward kinematics to get 21 joint positions in camera space.

    Always uses MANO_RIGHT.pkl. For left hands, x-mirrors the output.

    Args:
        orient_aa: (T, 3) global orientation axis-angle.
        pose_aa: (T, 45) finger pose axis-angle.
        betas: (10,) shape parameters.
        cam_t: (T, 3) translation.
        side: 'left' or 'right'.
        mano_dir: path to directory containing MANO_RIGHT.pkl.
    Returns:
        (T, 21, 3) float32 joints in camera space.
    """
    _preload_torch_cuda_deps()
    import torch

    ManoLayer = _import_manolayer()

    T = orient_aa.shape[0]
    mano_layer = ManoLayer(side="right", use_pca=False, ncomps=45, flat_hand_mean=True, mano_root=str(mano_dir))

    go = torch.from_numpy(orient_aa).float()
    hp = torch.from_numpy(pose_aa).float()
    bt = torch.from_numpy(betas).float().unsqueeze(0).expand(T, -1)
    ct = torch.from_numpy(cam_t).float()  # manopth expects meters for translation

    with torch.no_grad():
        full_pose = torch.cat([go, hp], dim=1)  # (T, 48)
        _, joints = mano_layer(full_pose, bt, ct)
        joints = joints.numpy().astype(np.float32) / 1000.0  # manopth outputs in mm → meters

    if side == "left":
        joints[:, :, 0] *= -1

    return joints


_J0_CACHE: dict = {}


def mano_root_joint(betas, side, mano_dir):
    """Zero-pose MANO root-joint position j0(betas) in native-side model space, meters.

    Computed with the RIGHT model (only MANO_RIGHT.pkl is required); the left root
    uses the mirror identity j0_left = M @ j0_right (M = diag(-1,1,1)) — exact within
    the repo's right-model-x-mirror left-hand convention.
    """
    betas = np.asarray(betas, dtype=np.float32).reshape(10)
    key = (betas.tobytes(), side, str(mano_dir))
    if key not in _J0_CACHE:
        joints = mano_fk_joints(
            np.zeros((1, 3), dtype=np.float32),
            np.zeros((1, 45), dtype=np.float32),
            betas,
            np.zeros((1, 3), dtype=np.float32),
            "right",
            str(mano_dir),
        )
        j0 = joints[0, 0].astype(np.float32)
        if side == "left":
            j0 = j0 * np.array([-1.0, 1.0, 1.0], dtype=np.float32)
        _J0_CACHE[key] = j0
    return _J0_CACHE[key].copy()


def root_trans_to_manopth_cam_t(trans, betas, side, mano_dir):
    """Convert a root-joint-position MANO translation to manopth cam_t semantics.

    TACO ``hand_trans`` / OakInk2 ``__tsl`` place the root joint at ``trans``
    (manotorch center_idx=0 semantics): verts = R @ (v - j0) + trans, while
    manopth FK places the root at j0(betas) + cam_t:
    verts = R @ (v - j0) + j0 + cam_t. Equating root positions:

        cam_t = trans - j0(betas)      (independent of the orientation)

    Apply in the native frame (native side, before mirror_left_hand_params and
    any world->camera transform).

    Args:
        trans: (T, 3) source translation (root-joint position).
        betas: (10,) shape parameters.
        side: 'left' or 'right' (native side of the source params).
        mano_dir: path to the directory containing MANO_RIGHT.pkl.
    Returns:
        (T, 3) float32 cam_t in manopth semantics.
    """
    trans = np.asarray(trans, dtype=np.float32).reshape(-1, 3)
    j0 = mano_root_joint(betas, side, mano_dir)
    return (trans - j0).astype(np.float32)


# ---------------------------------------------------------------------------
# Mesh I/O
# ---------------------------------------------------------------------------


def load_mesh(path):
    """Load a mesh file, returning vertices and faces.

    Args:
        path: path to mesh file (.obj, .ply, .glb, .off).
    Returns:
        Tuple of (verts (V, 3) float64, faces (F, 3) int64).
    """
    import trimesh

    mesh = trimesh.load(path, force="mesh", process=False)
    verts = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int64)
    return verts, faces
