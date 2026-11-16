#!/usr/bin/env python3
"""Object-pose refinement: re-pose the object mesh to its SAM 3 masks.

The SAM 3D mesh is kept as it is; only its 6DoF pose is re-fitted against the SAM 3 mask (with the MoGe intrinsics).

Per keyframe the refiner finds the rigid 6DoF delta (rotation about the object centroid + translation) that maximizes
the silhouette IoU with the SAM 3 mask, then makes it safe for a training trajectory:
  - temporal smoothing of the per-keyframe deltas (Karcher mean on SO(3), moving average on translation) so the
    per-frame IoU fit does not inject relative-pose jitter (the `rot_jerk` gate);
  - an envelope cap (max rotation / translation) so a bad mask cannot fling the object;
  - an accept-only-if-improves gate (the smoothed delta must raise the IoU by a margin, else the original pose is
    kept): do-no-harm.
The silhouette IoU does not see depth along the viewing ray, so the refiner corrects the in-plane / apparent-scale pose
error and leaves the depth of the chain as it is (depth is not observable in the video either).

    uv run python -m gtf.extraction.export.object_pose_refine   # synthetic self-test
"""

import hashlib
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import scipy
from scipy.optimize import minimize

from gtf.extraction.chains import ObjectRefine
from gtf.extraction.export.grasp_coherence import _exp_so3, _karcher_smooth, _log_so3, _ma, detect_grasp_runs

# Per-keyframe Powell seeds are independent → run them in threads (cv2.fillPoly + numpy release
# the GIL; fillPoly is most of the objective's cost). Results are reduced in seed order, so the
# output equals the serial loop's. OBJREF_SEED_WORKERS=1 runs the seeds serially.
_SEED_WORKERS = max(1, int(os.environ.get("OBJREF_SEED_WORKERS", "4")))
_PROJECTED_CACHE_MAX_BYTES = 64 << 20
OBJREF_CHECKPOINT_FORMAT = "gtf-object-refine-checkpoint-v1"
_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _project_vertices_i32(verts, K):
    """Project vertices exactly as OpenCV's integer polygon rasterizer sees them."""

    v = np.asarray(verts, np.float64)
    if len(v) < 3:
        return None
    z = v[:, 2].clip(1e-6)
    pts = np.clip(np.stack([K[0, 0] * v[:, 0] / z + K[0, 2], K[1, 1] * v[:, 1] / z + K[1, 2]], 1), [-1e4, -1e4], [1e4, 1e4])
    # Casting before the face gather equals ``pts[faces].astype(np.int32)`` elementwise.
    # The explicit int32 array is the exact, piecewise-constant input of fillPoly
    # (and the objective cache key in _refine_keyframe).
    return np.ascontiguousarray(pts, dtype=np.int32)


def _silhouette_from_projected(projected, faces, H, W, front=False):
    """Rasterized silhouette. front=False: all triangles in one fillPoly call. front=True: the camera-facing triangles
    only; the facing side is the winding with the larger projected area (robust to a flipped mesh)."""
    m = np.zeros((H, W), np.uint8)
    tri = projected[faces]
    if front:
        d1, d2 = (tri[:, 1] - tri[:, 0]).astype(np.int64), (tri[:, 2] - tri[:, 0]).astype(np.int64)
        a = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
        tri = tri[a > 0] if a[a > 0].sum() >= -a[a < 0].sum() else tri[a < 0]
    cv2.fillPoly(m, tri, 1)
    return m


def _silhouette(verts, faces, K, H, W, front=False):
    projected = _project_vertices_i32(verts, K)
    if projected is None:
        return None
    return _silhouette_from_projected(projected, np.asarray(faces, np.int64), H, W, front=front)


def _iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def _refine_keyframe(
    verts,
    faces,
    K,
    mask,
    maxdim=200,
    seeds=None,
    *,
    basis=None,
    front=False,
):
    """Independent 6DoF silhouette-IoU re-pose of one keyframe's object mesh. Returns (delta6,
    iou_before, iou_after_indep). delta6 = [axis-angle(3), translation(3)] about the centroid.
    `seeds` warm-starts the multi-start (pass the previous keyframe's delta to keep the trajectory
    in one rotational basin — symmetric objects otherwise alias into different basins per frame).
    `basis` (k,6) with orthonormal rows restricts the fit to delta6 = q @ basis (q in R^k), e.g. the image-plane slide
    of ObjectRefine.fit="lateral"; None = the full 6DoF fit. `front`: silhouette mode (_silhouette_from_projected)."""
    H0, W0 = mask.shape
    f = maxdim / max(H0, W0)
    ms = cv2.resize(mask.astype(np.uint8), (max(1, round(W0 * f)), max(1, round(H0 * f))), interpolation=cv2.INTER_NEAREST).astype(bool)
    H, W = ms.shape
    Ks = K.copy()
    Ks[:2] *= f
    c = verts.mean(0)
    vc = verts - c
    faces_i = np.ascontiguousarray(np.asarray(faces, np.int64))  # converted once, not per evaluation
    # Powell probes a continuous 6DoF pose, but the objective is an integer-pixel
    # raster and therefore piecewise constant: many evaluations repeat a projected
    # int32 vertex array already seen. Keying by the complete byte representation is
    # collision-free: a hit is the exact same fillPoly input, target mask, and IoU.
    # The cap bounds transient memory for pathological optimizer traces; stopping
    # insertion never changes the objective, it only gives up acceleration for later
    # novel projections.
    projected_objectives = {}
    projected_cache_bytes = 0

    def neg_iou(p):
        nonlocal projected_cache_bytes
        vp = vc @ _exp_so3(p[:3]).T + c + p[3:6]
        projected = _project_vertices_i32(vp, Ks)
        if projected is None:
            return 1.0
        key = projected.tobytes(order="C")
        cached = projected_objectives.get(key)
        if cached is not None:
            return cached
        value = -_iou(_silhouette_from_projected(projected, faces_i, H, W, front=front), ms)
        if projected_cache_bytes + len(key) <= _PROJECTED_CACHE_MAX_BYTES:
            # Multiple seed threads may race on the same miss.  Both compute the
            # same scalar; setdefault makes their eventual shared value benign.
            inserted_value = value
            value = projected_objectives.setdefault(key, inserted_value)
            if value is inserted_value:
                projected_cache_bytes += len(key)
        return value

    if seeds is None:
        seeds = [np.zeros(6), np.array([0, 0.4, 0, 0, 0, 0.0]), np.array([0, -0.4, 0, 0, 0, 0.0])]
    i0 = -neg_iou(np.zeros(6))  # current-pose IoU (always at identity, the reference)
    # exact-duplicate seeds collapse to one Powell run (deterministic: same seed → same result);
    # first-occurrence order preserved so the in-order reduce below matches the serial semantics.
    uniq, seen = [], set()
    for s_ in seeds:
        a = np.asarray(s_, np.float64)
        key = a.tobytes()
        if key not in seen:
            seen.add(key)
            uniq.append(a)

    if basis is not None:
        B = np.asarray(basis, np.float64)
        full = neg_iou
        neg_iou_q = lambda q: full(q @ B)  # noqa: E731
        uniq_q, seen_q = [], set()
        for s_ in uniq:
            q = B @ s_
            if q.tobytes() not in seen_q:
                seen_q.add(q.tobytes())
                uniq_q.append(q)
        uniq = uniq_q

    def run_seed(seed):
        f_ = neg_iou if basis is None else neg_iou_q
        r = minimize(f_, seed, method="Powell", options={"maxiter": 60, "xtol": 1e-3, "ftol": 1e-3})
        if basis is not None:
            r.x = np.asarray(r.x, np.float64) @ B
        return r

    if _SEED_WORKERS > 1 and len(uniq) > 1:
        with ThreadPoolExecutor(max_workers=min(len(uniq), _SEED_WORKERS)) as ex:
            results = list(ex.map(run_seed, uniq))
    else:
        results = [run_seed(s_) for s_ in uniq]
    best, bp = i0, np.zeros(6)
    for r in results:  # in seed order — identical tie-breaking to the serial loop
        if -r.fun > best:
            best, bp = -r.fun, r.x
    return bp, float(i0), float(best)


def _soft_apply(
    obj_verts,
    obj_faces,
    K_list,
    masks,
    kf_indices,
    Rs_sm,
    ts_sm,
    accepted,
    has,
    i_before,
    rep,
    soft_r,
    hand_verts,
    hand_valid,
    contact_m,
    min_run,
    run_guard=False,
    front=False,
):
    """ObjectRefine.soft_accept path: weight w_i = moving average of (accepted or in a grasp run with mask evidence), the
    delta applied as (exp(w log R), w t). Hands in their grasp runs are carried by the same weighted delta, so the
    hand-object relative pose is preserved exactly there (as in the hard path, without on/off steps). run_guard: a
    grasp run is forced only if the correction does not lower its mean mask IoU (do-no-harm per run)."""
    n = len(obj_verts)
    obj0 = np.asarray(obj_verts, np.float64)
    grasp = np.zeros(n, bool)
    side_run = {}
    if hand_verts is not None and hand_valid is not None:
        for side, hv in hand_verts.items():
            hvalid = np.asarray(hand_valid[side]).astype(bool)
            ir = np.zeros(n, bool)
            for r in detect_grasp_runs(np.asarray(hv, np.float64), hvalid, obj0, contact_m=contact_m, min_run=min_run):
                ir[r] = True
            side_run[side] = ir & hvalid
            grasp |= side_run[side]
    if run_guard and grasp.any():
        for s0, s1 in _true_runs(grasp):
            ib, ia = [], []
            for i in range(s0, s1):
                if not has[i]:
                    continue
                mk = masks[int(kf_indices[i])].astype(bool)
                Kf = np.asarray(K_list[i], np.float64)
                sb = _silhouette(obj0[i], obj_faces, Kf, mk.shape[0], mk.shape[1], front=front)
                sa = _silhouette(_apply_delta(obj0[i], Rs_sm[i], ts_sm[i]), obj_faces, Kf, mk.shape[0], mk.shape[1], front=front)
                ib.append(_iou(sb, mk) if sb is not None else 0.0)
                ia.append(_iou(sa, mk) if sa is not None else 0.0)
            if ib and np.mean(ia) < np.mean(ib):
                grasp[s0:s1] = False
                for side in side_run:
                    side_run[side][s0:s1] = False
                rep["n_runs_guarded"] = rep.get("n_runs_guarded", 0) + 1
    a = (accepted | (grasp & has)).astype(np.float64)
    w = _ma(a[:, None], soft_r)[:, 0]
    out = obj0.copy()
    Rw, tw = np.repeat(np.eye(3)[None], n, axis=0), np.zeros((n, 3))
    i_after = np.array(i_before, copy=True)
    for i in range(n):
        if w[i] <= 1e-6:
            continue
        Rw[i] = _exp_so3(w[i] * _log_so3(Rs_sm[i]))
        tw[i] = w[i] * ts_sm[i]
        out[i] = _apply_delta(obj0[i], Rw[i], tw[i])
        if has[i]:
            mk = masks[int(kf_indices[i])].astype(bool)
            sil = _silhouette(out[i], obj_faces, np.asarray(K_list[i], np.float64), mk.shape[0], mk.shape[1], front=front)
            i_after[i] = _iou(sil, mk) if sil is not None else 0.0
    rep["iou_after"] = float(np.nanmean(i_after[has])) if has.any() else float("nan")
    rep["soft_accept_radius"] = int(soft_r)
    rep["soft_weight_mean"] = float(w.mean())
    if side_run:
        carried, n_carry = {}, 0
        for side, hv in hand_verts.items():
            hv = np.asarray(hv, np.float64)
            hout = hv.copy()
            for i in range(n):
                if side_run[side][i] and w[i] > 1e-6:
                    oc = obj0[i].mean(0)
                    hout[i] = (hv[i] - oc) @ Rw[i].T + oc + tw[i]
                    n_carry += 1
            carried[side] = hout.astype(hv.dtype if hasattr(hv, "dtype") else np.float32)
        rep["hand_verts"] = carried
        rep["n_carried_kf"] = int(n_carry)
        rep["n_grasp_kf"] = int(grasp.sum())
    return out.astype(obj_verts.dtype if hasattr(obj_verts, "dtype") else np.float32), rep


def _apply_delta(verts, rot, trans):
    c = verts.mean(0)
    return (verts - c) @ rot.T + c + trans


def _true_runs(mask):
    """Return half-open contiguous runs of a one-dimensional boolean mask."""
    mask = np.asarray(mask, bool)
    if mask.ndim != 1:
        raise ValueError(f"expected a 1-D run mask, got {mask.shape}")
    padded = np.pad(mask.astype(np.int8), (1, 1))
    edges = np.diff(padded)
    return [(int(start), int(end)) for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def _run_rows(mask):
    """JSON-safe half-open ranges for a boolean frame mask."""
    return [[start, end] for start, end in _true_runs(mask)]


def _hash_array(digest, label, value):
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(label.encode("utf-8") + b"\0")
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))


def _checkpoint_input_sha256(
    obj_verts,
    obj_faces,
    K_list,
    masks,
    kf_indices,
    *,
    max_rot_deg,
    max_trans_m,
    min_mask_px,
    fit_dof="6dof",
    front_silhouette=False,
):
    """Bind resumable optimizer state to every value that affects it."""

    digest = hashlib.sha256()
    contract = {
        "format": OBJREF_CHECKPOINT_FORMAT,
        "algorithm": "warm-started-powell-seeds-in-order-v1",
        # Checkpoints live beside the segment outputs and can outlive the code
        # that wrote them. Binding this file's source and the numerical libraries
        # that run Powell / rasterization makes any code or dependency change
        # recompute instead of resuming.
        "implementation_sha256": _IMPLEMENTATION_SHA256,
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "scipy_version": scipy.__version__,
        "max_rot_deg": float(max_rot_deg).hex(),
        "max_trans_m": float(max_trans_m).hex(),
        "min_mask_px": int(min_mask_px),
    }
    if fit_dof != "6dof":  # only non-default options enter the contract
        contract["fit_dof"] = str(fit_dof)
    if front_silhouette:
        contract["front_silhouette"] = True
    digest.update(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode())
    _hash_array(digest, "obj_verts", obj_verts)
    _hash_array(digest, "obj_faces", obj_faces)
    _hash_array(digest, "K_list", K_list)
    _hash_array(digest, "kf_indices", kf_indices)
    # Only referenced masks can affect the optimizer. Hash them in roster order,
    # including duplicate frame references, so the binding matches the exact loop.
    for row, frame_index in enumerate(np.asarray(kf_indices).tolist()):
        fi = int(frame_index)
        if fi < len(masks):
            _hash_array(digest, f"mask[{row}]@{fi}", masks[fi])
    return digest.hexdigest()


def _checkpoint_state_sha256(next_index, deltas, i_before, i_indep, has, prev):
    digest = hashlib.sha256()
    digest.update(str(int(next_index)).encode("ascii") + b"\0")
    for label, value in (
        ("deltas", deltas),
        ("iou_before", i_before),
        ("iou_independent", i_indep),
        ("has_mask", has),
        ("previous_delta", prev),
    ):
        _hash_array(digest, label, value)
    return digest.hexdigest()


def _write_refine_checkpoint(path, input_sha256, next_index, deltas, i_before, i_indep, has, prev):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state_sha256 = _checkpoint_state_sha256(next_index, deltas, i_before, i_indep, has, prev)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(
                stream,
                format=np.str_(OBJREF_CHECKPOINT_FORMAT),
                input_sha256=np.str_(input_sha256),
                state_sha256=np.str_(state_sha256),
                next_index=np.int64(next_index),
                deltas=deltas,
                iou_before=i_before,
                iou_independent=i_indep,
                has_mask=has,
                previous_delta=prev,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _load_refine_checkpoint(path, input_sha256, n):
    try:
        with np.load(path, allow_pickle=False) as payload:
            required = {
                "format",
                "input_sha256",
                "state_sha256",
                "next_index",
                "deltas",
                "iou_before",
                "iou_independent",
                "has_mask",
                "previous_delta",
            }
            if set(payload.files) != required:
                return None
            if str(payload["format"].item()) != OBJREF_CHECKPOINT_FORMAT or str(payload["input_sha256"].item()) != input_sha256:
                return None
            next_index = int(payload["next_index"].item())
            deltas = payload["deltas"]
            i_before = payload["iou_before"]
            i_indep = payload["iou_independent"]
            has = payload["has_mask"]
            prev = payload["previous_delta"]
            if (
                not 0 <= next_index <= n
                or deltas.shape != (n, 6)
                or i_before.shape != (n,)
                or i_indep.shape != (n,)
                or has.shape != (n,)
                or prev.shape != (6,)
                or deltas.dtype != np.float64
                or i_before.dtype != np.float64
                or i_indep.dtype != np.float64
                or has.dtype != np.bool_
                or prev.dtype != np.float64
            ):
                return None
            expected_state = _checkpoint_state_sha256(next_index, deltas, i_before, i_indep, has, prev)
            if str(payload["state_sha256"].item()) != expected_state:
                return None
            return next_index, deltas, i_before, i_indep, has, prev
    # This file is only an acceleration hint. Any ordinary decoder, schema, CRC,
    # dtype, or validation failure must become a cache miss; letting it escape
    # would reach export_canonical's "refiner skipped" fallback and silently
    # export without the refinement. BaseException subclasses still interrupt.
    except Exception:  # noqa: BLE001
        return None


def _fit_basis(fit_dof, verts):
    """Orthonormal (k,6) rows spanning the allowed delta6 = [axis-angle, translation]; None = full 6DoF. "lateral": the
    two translations perpendicular to the viewing ray through the object centroid."""
    if fit_dof == "6dof":
        return None
    d = verts.mean(0)
    d = d / max(np.linalg.norm(d), 1e-9)  # viewing ray through the centroid (camera at the origin)
    up = np.array([0.0, 1.0, 0.0]) if abs(d[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(d, up)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    L = np.zeros((2, 6))
    L[0, 3:], L[1, 3:] = e1, e2
    return L


def compute_refine_deltas(
    obj_verts,
    obj_faces,
    K_list,
    masks,
    kf_indices,
    *,
    max_rot_deg=35.0,
    max_trans_m=0.30,
    min_mask_px=30,
    checkpoint_path=None,
    checkpoint_every=8,
    fit_dof="6dof",
    front_silhouette=False,
):
    """Per-keyframe warm-started silhouette-IoU re-pose. Returns (deltas (n,6), i_before (n,), i_indep (n,), has (n,)).

    fit_dof (chains.ObjectRefine.fit): "6dof" the full fit; "lateral" translation in the plane perpendicular to the
    viewing ray through the object centroid (depth and rotation held: the object slides on screen but neither it nor a
    carried hand changes apparent size). Discarding the rotation of a 6DoF fit afterwards would keep a translation that
    was only right together with that rotation, so the constraint has to be in the fit itself.
    front_silhouette: silhouette mode (_silhouette_from_projected)."""
    if fit_dof not in ("6dof", "lateral"):
        raise ValueError(f"unknown fit_dof {fit_dof!r}")
    n = len(obj_verts)
    if checkpoint_path is not None and checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    deltas = np.zeros((n, 6))
    i_before = np.full(n, np.nan)
    i_indep = np.full(n, np.nan)
    has = np.zeros(n, bool)
    max_rot = np.radians(max_rot_deg)
    prev = np.zeros(6)  # warm-start seed (previous keyframe's delta — keeps the chain coherent)
    start_index = 0
    checkpoint_input = None
    checkpoint_enabled = checkpoint_path is not None
    if checkpoint_enabled:
        checkpoint_input = _checkpoint_input_sha256(
            obj_verts,
            obj_faces,
            K_list,
            masks,
            kf_indices,
            max_rot_deg=max_rot_deg,
            max_trans_m=max_trans_m,
            min_mask_px=min_mask_px,
            fit_dof=fit_dof,
            front_silhouette=front_silhouette,
        )
        restored = _load_refine_checkpoint(checkpoint_path, checkpoint_input, n)
        if restored is not None:
            start_index, deltas, i_before, i_indep, has, prev = restored
    for i in range(start_index, n):
        fi = int(kf_indices[i])
        if fi < len(masks):
            mk = masks[fi].astype(bool)
            if mk.sum() >= min_mask_px:
                seeds = [np.zeros(6), prev, prev + np.array([0, 0.3, 0, 0, 0, 0.0]), prev + np.array([0, -0.3, 0, 0, 0, 0.0])]
                basis = _fit_basis(fit_dof, np.asarray(obj_verts[i], np.float64))
                p, i0, i1 = _refine_keyframe(
                    np.asarray(obj_verts[i], np.float64), obj_faces, np.asarray(K_list[i], np.float64), mk, seeds=seeds, basis=basis, front=front_silhouette
                )
                ang = np.linalg.norm(p[:3])  # envelope cap (a bad mask can't fling the object)
                if ang > max_rot:
                    p[:3] *= max_rot / ang
                tn = np.linalg.norm(p[3:6])
                if tn > max_trans_m:
                    p[3:6] *= max_trans_m / tn
                deltas[i] = p
                prev = p
                i_before[i], i_indep[i] = i0, i1
                has[i] = True
        next_index = i + 1
        if checkpoint_enabled and (next_index % checkpoint_every == 0 or next_index == n):
            try:
                _write_refine_checkpoint(
                    checkpoint_path,
                    checkpoint_input,
                    next_index,
                    deltas,
                    i_before,
                    i_indep,
                    has,
                    prev,
                )
            except OSError as error:
                # A full/temporarily unavailable cache filesystem must not alter
                # the quality path. Continue the exact computation, forfeiting
                # only resume acceleration for this process.
                print(
                    f"WARNING: object-refine checkpoint disabled: {error}",
                    file=sys.stderr,
                )
                checkpoint_enabled = False
    return deltas, i_before, i_indep, has


def refine_object_trajectory(
    obj_verts,
    obj_faces,
    K_list,
    masks,
    kf_indices,
    *,
    iou_margin=0.03,
    max_rot_deg=35.0,
    max_trans_m=0.30,
    smooth_radius=2,
    min_mask_px=30,
    hand_verts=None,
    hand_valid=None,
    contact_m=0.04,
    min_run=4,
    checkpoint_path=None,
    checkpoint_every=8,
    options=None,
):
    """Refine the object 6DoF chain to the SAM3 masks (video-faithful), temporally smoothed +
    accept-gated. `options` (chains.ObjectRefine; None = the default ObjectRefine): what the fit may change, whether its
    rotation is applied, the translation smoothing radius, soft acceptance, the per-run guard and the silhouette mode.
    obj_verts: (n,V,3) camera frame; K_list: per-keyframe (3,3); masks: (T,H,W); kf_indices: (n,) frame index of each
    keyframe. Returns (refined_obj_verts, report).

    Carrying the hand: re-posing the object alone breaks the hand-object grasp (the object moves to
    its mask, off the co-placed hand, and contact / rot_jerk, both relative-pose properties, get
    worse). If `hand_verts`/`hand_valid` (dicts side->array) are given, the object's rigid delta is
    also applied to the hand on contact-run keyframes: the same rigid map about the object centroid,
    so the hand-object relative pose is preserved exactly (contact + rot_jerk untouched) while the
    grasp as a whole moves to align the object with its mask. Every frame of a grasp run with mask
    evidence takes the smoothed object delta, including frames whose own accept gate rejected it.
    The carried hands are returned in report["hand_verts"] (dict side->array)."""
    if options is None:
        options = ObjectRefine()
    n = len(obj_verts)
    deltas, i_before, i_indep, has = compute_refine_deltas(
        obj_verts,
        obj_faces,
        K_list,
        masks,
        kf_indices,
        max_rot_deg=max_rot_deg,
        max_trans_m=max_trans_m,
        min_mask_px=min_mask_px,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
        fit_dof=options.fit,
        front_silhouette=options.front_silhouette,
    )

    # temporal smoothing of the per-keyframe deltas (rotation via Karcher, translation via MA)
    Rs = np.stack([_exp_so3(deltas[i, :3]) for i in range(n)])
    Rs_sm = _karcher_smooth(Rs, smooth_radius) if smooth_radius > 0 else Rs
    ts_sm = _ma(deltas[:, 3:6], smooth_radius) if smooth_radius > 0 else deltas[:, 3:6]
    # ObjectRefine options: drop the rotation correction, use its own translation smoothing radius, and (soft_accept,
    # below) replace the per-frame accept gate with a smoothed weight.
    if not options.rotation:
        Rs_sm = np.repeat(np.eye(3)[None], n, axis=0)
    if options.trans_radius is not None:
        ts_sm = _ma(deltas[:, 3:6], options.trans_radius) if options.trans_radius > 0 else deltas[:, 3:6]
    front = options.front_silhouette

    out = np.array(obj_verts, np.float64).copy()
    accepted = np.zeros(n, bool)
    i_after = np.array(i_before, copy=True)
    for i in range(n):
        if not has[i]:
            continue
        cand = _apply_delta(out[i], Rs_sm[i], ts_sm[i])
        fi = int(kf_indices[i])
        mk = masks[fi].astype(bool)
        s = _silhouette(cand, obj_faces, np.asarray(K_list[i], np.float64), mk.shape[0], mk.shape[1], front=front)
        ic = _iou(s, mk) if s is not None else 0.0
        # accept the smoothed delta only if it still improves IoU by the margin (do-no-harm)
        if ic >= i_before[i] + iou_margin:
            out[i] = cand
            accepted[i] = True
            i_after[i] = ic
    rep = {
        "n_kf": int(has.sum()),
        "n_accepted": int(accepted.sum()),
        "iou_before": float(np.nanmean(i_before[has])) if has.any() else float("nan"),
        "iou_indep": float(np.nanmean(i_indep[has])) if has.any() else float("nan"),
        "iou_after": float(np.nanmean(i_after[has])) if has.any() else float("nan"),
        "rot_delta_deg_med": float(np.degrees(np.median([np.linalg.norm(deltas[i, :3]) for i in range(n) if accepted[i]]))) if accepted.any() else 0.0,
    }

    if options.soft_accept > 0:
        return _soft_apply(
            obj_verts,
            obj_faces,
            K_list,
            masks,
            kf_indices,
            Rs_sm,
            ts_sm,
            accepted,
            has,
            i_before,
            rep,
            options.soft_accept,
            hand_verts,
            hand_valid,
            contact_m,
            min_run,
            run_guard=options.run_guard,
            front=front,
        )

    # Carry the hand, run-consistent: each grasp run moves as one rigid body. Within a run the object
    # takes its smoothed delta on every frame with mask evidence (overriding the per-frame accept gate,
    # which keeps the object trajectory coherent across the run) and the hand is carried by the same
    # rigid map about the object centroid, so the hand-object relative pose is preserved exactly across
    # the run (contact + rot_jerk untouched). A per-frame carry would jitter at accept boundaries.
    # Outside runs the object keeps the per-frame accept decision and the hand is untouched.
    if hand_verts is not None and hand_valid is not None:
        obj0 = np.asarray(obj_verts, np.float64)
        side_run = {}
        grasp = np.zeros(n, bool)
        for side, hv in hand_verts.items():
            hvalid = np.asarray(hand_valid[side]).astype(bool)
            runs = detect_grasp_runs(np.asarray(hv, np.float64), hvalid, obj0, contact_m=contact_m, min_run=min_run)
            ir = np.zeros(n, bool)
            for r in runs:
                ir[r] = True
            side_run[side] = ir & hvalid & has
            grasp |= side_run[side]
        grasp_runs = _true_runs(grasp)
        n_force = 0
        for i in range(n):
            if grasp[i] and not accepted[i]:
                out[i] = _apply_delta(obj0[i], Rs_sm[i], ts_sm[i])
                n_force += 1
        # carry each hand by the same delta on its own grasp-run frames
        carried, n_carry = {}, 0
        for side, hv in hand_verts.items():
            hv = np.asarray(hv, np.float64)
            hout = hv.copy()
            for i in range(n):
                if side_run[side][i]:
                    oc = obj0[i].mean(0)
                    hout[i] = (hv[i] - oc) @ Rs_sm[i].T + oc + ts_sm[i]
                    n_carry += 1
            carried[side] = hout.astype(hv.dtype if hasattr(hv, "dtype") else np.float32)
        rep["hand_verts"] = carried
        rep["n_carried_kf"] = int(n_carry)
        rep["n_forced_obj"] = int(n_force)
        rep["n_grasp_runs"] = len(grasp_runs)
        rep["n_grasp_kf"] = int(grasp.sum())
        # Half-open keyframe ranges (after the interaction-window crop) of the grasp runs and of each
        # side's carried frames; export_canonical stores them as JSON in the objref sidecar.
        rep["grasp_runs_half_open"] = [list(run) for run in grasp_runs]
        rep["carried_runs_half_open_by_side"] = {side: _run_rows(side_run[side]) for side in sorted(side_run)}

    return out.astype(obj_verts.dtype if hasattr(obj_verts, "dtype") else np.float32), rep


# ----------------------------------------------------------------------------- synthetic test
def _selftest():
    n = 12
    K = np.array([[600.0, 0, 256], [0, 600.0, 256], [0, 0, 1.0]])
    import trimesh

    # a dense anisotropic ellipsoid (1280 faces): dense so the face-raster silhouette fills solid
    # (a sparse 12-face box projects to slivers), anisotropic so rotation changes the silhouette.
    ico = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    bverts = np.asarray(ico.vertices) * np.array([0.07, 0.045, 0.03])
    bfaces = np.asarray(ico.faces)
    true_verts, masks, Ks, kfi = [], [], [], []
    Tt = np.zeros((n, 512, 512), np.uint8)
    for t in range(n):
        # object slides across frames (translation only): the pose error is constant, so the
        # correct per-keyframe delta is constant => temporal smoothing is a true no-op here.
        tt = np.array([0.004 * t - 0.02, 0.002 * t, 0.6])
        gtv = bverts + tt
        true_verts.append(gtv)
        mk = _silhouette(gtv, bfaces, K, 512, 512)
        Tt[t] = mk
        masks.append(mk)
        Ks.append(K)
        kfi.append(t)
    Tt = Tt.astype(bool)
    # extraction = the true mesh mis-posed by a wrong rotation+translation (the pose error).
    # Mild enough to keep partial mask overlap (~0.35 IoU); a near-zero-overlap start has no
    # silhouette gradient.
    Rwrong = _exp_so3(np.array([0.05, 0.18, 0.03]))
    twrong = np.array([0.012, -0.008, 0.0])
    ext_verts = np.stack([_apply_delta(np.asarray(true_verts[t]), Rwrong, twrong) for t in range(n)])

    def mean_iou(vv):
        return float(np.mean([_iou(_silhouette(vv[t], bfaces, K, 512, 512), Tt[t]) for t in range(n)]))

    iou_ext = mean_iou(ext_verts)
    refined, rep = refine_object_trajectory(ext_verts, bfaces, Ks, Tt, np.array(kfi))
    iou_ref = mean_iou(refined)
    print(f"object pose refine: mean obj_IoU {iou_ext:.2f} -> {iou_ref:.2f}  (accepted {rep['n_accepted']}/{rep['n_kf']}, rot_delta {rep['rot_delta_deg_med']:.0f}deg)")
    assert iou_ref > iou_ext + 0.15, f"refine did not recover the pose: {iou_ext}->{iou_ref}"

    # already-good pose -> near no-op (no spurious large deltas, IoU not hurt)
    refined2, rep2 = refine_object_trajectory(np.stack(true_verts), bfaces, Ks, Tt, np.array(kfi))
    iou_good = mean_iou(refined2)
    assert iou_good > 0.9, f"refine hurt an already-good pose: {iou_good}"
    print(f"already-good pose: IoU stays {iou_good:.2f} (accepted {rep2['n_accepted']}/{rep2['n_kf']}, do-no-harm)")
    print("\nobject-pose-refine self-tests pass")


if __name__ == "__main__":
    _selftest()
