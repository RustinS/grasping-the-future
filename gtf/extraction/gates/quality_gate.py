#!/usr/bin/env python3
"""Quality gate: filter sequences by 21 physics + video-alignment + interaction-fidelity metrics.

Computes per-segment quality metrics and pass/fail verdicts (thresholds in gate_thresholds.py).
Saves per-segment quality.json and an aggregated quality_summary.csv.

Usage:
    python -m gtf.extraction.gates.quality_gate                         # all segments, moderate thresholds
    python -m gtf.extraction.gates.quality_gate --level loose            # loose thresholds
    python -m gtf.extraction.gates.quality_gate --segment "seg_name"     # single segment
    python -m gtf.extraction.gates.quality_gate --force                  # recompute even if quality.json exists

Gates (21 total):
    1. vol_cv          – object shape stability (rigid body must not deform)
    2. acc_max         – object translational smoothness (no teleporting)
    3. ang_acc_max     – object rotational smoothness (no sudden flips)
    4. sil_iou         – rendered silhouette vs SAM3 mask agreement
    5. area_ratio      – rendered vs mask area on low-occlusion frames
    6. rel_pose_max    – hand-object relative pose stability in object frame
    7. iou_trend       – SAM3D IoU trend over time (no progressive drift)
    8. hand_rate       – best hand detection rate
    9. hand_scale      – hand metric size vs canonical MANO, and its drift over the segment
                         (a wrong depth scale gives wrong hand sizes and wrong contact)
   10. interaction     – some hand must approach the object at some frame (object motion
                         alone does not count)
   11. scene_cut       – no shot boundary inside the segment (after a mid-segment cut the
                         per-frame stages keep succeeding on a different scene)
   12. penetration     – worst-side p90 hand-into-object penetration depth, sign by
                         generalized winding number (hoi_contact.py)
   13. obj_shape       – wrong 3D model: self-baked RGB re-projection consistency low and
                         SAM3 mask border-truncation high (a reconstruction from a truncated
                         view, e.g. a glass as a squat disc, can still match the mask)
   14. rot_jerk        – hand<->object 2D relative-rotation jerk over the grasp (non-rigid
                         grasps: a utensil, a false-positive hand; interaction_metrics.py)
   15. contact_cover   – 3D-contact hand verts must project inside the object mask
                         (no off-object "ghost" contact; interaction_metrics.py)
   16. motion2d        – in-plane motion agreement: the projected object's 2D path must agree
                         with the SAM3 mask's path (phantom motion, missing motion, wobble)
   17. size_consistency – apparent-size trend agreement (conjunction): mask/sil area-ratio
                         drift high, a large claimed depth excursion and mediocre in-plane
                         coherence = the object wandering along the view ray (per-keyframe
                         depth noise; invisible to every per-frame gate)
   18. interaction_sustain – contact must be sustained (fraction of keyframes with a valid
                         hand near the object); gate 10 needs only a single approach frame
   19. obj_size        – hand-manipulable object scale: the placed object's diagonal must be
                         < obj_size_max_m (SAM3 can track a person-scale region, e.g. a garment)
   20. obj_is_hand     – the mask is the hand (pointing can bind to the hand itself): the median
                         fraction of the SAM3 mask covered by the projected MANO hand silhouette
                         must be < mask_in_hand_max
   21. obj_reground    – re-grounding verifier (reground.py): fresh Molmo2 evidence at spread
                         window frames vs the tracked SAM3 mask, produced by the pipeline's
                         `reground` step for gate-passing segments. Fails when Molmo2 mostly
                         refuses "the object held in the hand" (nothing is held) or when its
                         held-points land far from the tracked mask (something else is held).
                         Axes None (no evidence yet) -> the gate defers, so CPU-only gating
                         works without it.

All gates are window-scoped: when the export cropped to an interaction window (or the segment
is a multi-window {seg}__iwN child), every gate measures only that window's keyframes; the
window is the segment (the objref sidecar's frame set, else the canonical's interaction_crop).
"""

import argparse
import csv
import json
import os
import tempfile
import time

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

from gtf.utils import (
    RESULTS_DIR,
    load_hand_data,
    load_mano_faces,
    load_moge_intrinsics,
    load_object_mesh,
)
from gtf.utils.video_timing import file_sha256 as _sha256_file

from gtf.extraction.gates.gate_thresholds import IMAGE_METRICS, THRESHOLDS, evaluate_gates  # noqa: E402
from gtf.extraction.gates.gate_geometry import _angular_velocity, _camera_rotation_speed, _continuous_quaternions, _depth_aware_area_ratios  # noqa: E402
from gtf.extraction.gates.gate_geometry import _exported_window, _hand_scale_metrics, _interaction_metrics, _obj_rgb_consistency  # noqa: E402
from gtf.extraction.gates.gate_geometry import _procrustes_rotations, _render_silhouette, _resolve_area_ratio, _scope_sam3d_frames, _subset_chain_to_window  # noqa: E402


def compute_metrics(
    seg_name,
    opt_subdir="optimized_mano",
    results_dir=None,
    *,
    image_metrics_only=False,
):
    """Compute quality metrics for a segment. Returns dict or None.

    opt_subdir: "optimized_stage2" to gate the stage-2-refined trajectory
    (gtf/extraction/stage2/refine_trajectory.py) instead of the raw optimizer output.

    ``image_metrics_only``: the raw-extraction pass of ``process_one``. It returns only
    ``IMAGE_METRICS`` (computed as usual) and skips the physics, cut, contact, motion and
    re-ground metrics, which that pass does not use.

    Window scope: when the export cropped to an interaction window (or the segment is a
    multi-window child), all gates (physics, image, hand_rate, scene_cut, iou_trend) measure
    only that window's keyframes: the window is the segment; out-of-window evidence belongs
    to a different (or no) trajectory."""
    # results_dir defaults to RESULTS_DIR; another root gates a separate results
    # tree without changing this module's global RESULTS_DIR.
    results_root = os.fspath(RESULTS_DIR if results_dir is None else results_dir)
    seg_dir = os.path.join(results_root, seg_name)
    opt_path = os.path.join(seg_dir, opt_subdir, "optimized.npz")
    mask_path = os.path.join(seg_dir, "SAM3", "masks.npz")

    if not all(os.path.exists(p) for p in [opt_path, mask_path]):
        return None

    d = {k: v for k, v in np.load(opt_path).items()}
    # The optimizer records which SAM3D subdir supplied mesh + poses ("sam3d" or the
    # "sam3d_c" BA fallback); the gate's mesh/metadata reads must follow it, else the
    # canonical-verts <-> faces correspondence (gates 4/5/13) silently breaks.
    pose_subdir = str(d.get("pose_source", "sam3d"))
    sam3d_path = os.path.join(seg_dir, pose_subdir, "metadata.json")
    if not os.path.exists(sam3d_path):
        return None
    window = _exported_window(seg_dir)
    window_scoped = False
    if window is not None:
        d, window_scoped = _subset_chain_to_window(d, window[0], window[1])
    sam3d_meta = json.load(open(sam3d_path))
    obj_verts = d["obj_verts"]  # (T, V, 3) in MoGe camera space
    frame_indices = d["frame_indices"]
    T = len(obj_verts)
    if T < 3:
        return None

    # ── Gate 1: vol_cv ──────────────────────────────────────────────────
    obj_range = obj_verts.max(axis=1) - obj_verts.min(axis=1)
    vol = np.prod(obj_range, axis=1)
    vol_cv = float(vol.std() / vol.mean()) if vol.mean() > 0 else 999.0

    # ── Gate 2: acc_max ─────────────────────────────────────────────────
    centroids = obj_verts.mean(axis=1)
    vel = np.diff(centroids, axis=0)
    acc = np.linalg.norm(np.diff(vel, axis=0), axis=1)
    acc_max = float(acc.max())

    # ── Gate 3: ang_acc_max ─────────────────────────────────────────────
    rot_mats = _procrustes_rotations(obj_verts)
    cont_quats = _continuous_quaternions(rot_mats)
    ang_vel = _angular_velocity(cont_quats)
    ang_acc = np.abs(np.diff(ang_vel))
    ang_acc_max = float(ang_acc.max()) if len(ang_acc) > 0 else 0.0

    # ── Gate 8: hand_rate ───────────────────────────────────────────────
    best_hand = 0.0
    for side in ("left", "right"):
        k = f"hand_{side}_valid"
        if k in d:
            best_hand = max(best_hand, float(d[k].sum() / len(d[k])))

    # ── Gate 6: rel_pose_max ────────────────────────────────────────────
    obj_centroids = obj_verts.mean(axis=1)
    obj_diag = float(np.linalg.norm(obj_verts[0].max(axis=0) - obj_verts[0].min(axis=0)))
    ref = obj_verts[0] - obj_centroids[0]
    rel_pose_max = 999.0

    for side in ("left", "right"):
        hv_key, valid_key = f"hand_{side}_verts", f"hand_{side}_valid"
        if hv_key not in d or valid_key not in d:
            continue
        hand_verts = d[hv_key]
        hand_valid = d[valid_key]
        valid_idx = np.where(hand_valid)[0]
        if len(valid_idx) < 3:
            continue

        hand_centroids = hand_verts.mean(axis=1)
        rel_in_obj = []
        for t in valid_idx:
            centered = obj_verts[t] - obj_centroids[t]
            H_mat = ref.T @ centered
            U, S, Vt = np.linalg.svd(H_mat)
            det = np.linalg.det(Vt.T @ U.T)
            R = Vt.T @ np.diag([1, 1, det]) @ U.T
            rel_in_obj.append(R.T @ (hand_centroids[t] - obj_centroids[t]))
        rel_in_obj = np.array(rel_in_obj)
        rel_vel = np.diff(rel_in_obj, axis=0)
        gaps = np.maximum(np.diff(valid_idx), 1)
        rel_speed = np.linalg.norm(rel_vel, axis=1) / gaps
        val = float(rel_speed.max() / obj_diag) if obj_diag > 0 else 999.0
        rel_pose_max = min(rel_pose_max, val)

    # ── Gate 7: iou_trend ───────────────────────────────────────────────
    frames = _scope_sam3d_frames(sam3d_meta.get("frames", []), window, window_scoped)
    if window_scoped and not frames:
        # SAM 3D rows outside the window are not evidence for it; no quality result then.
        return None
    ious = [f["pose"]["iou"] for f in frames if f.get("status") == "ok" and "pose" in f and "iou" in f["pose"]]
    if len(ious) >= 10:
        iou_trend = float(np.mean(ious[-5:]) - np.mean(ious[:5]))
    elif len(ious) >= 4:
        half = len(ious) // 2
        iou_trend = float(np.mean(ious[half:]) - np.mean(ious[:half]))
    else:
        iou_trend = 0.0

    # ── Gates 4 & 5: sil_iou, area_ratio (rendering) ───────────────────
    anchor_idx = sam3d_meta.get("anchor_frame_index", 0)
    anchor_mesh = load_object_mesh(seg_dir, anchor_idx, subdir=pose_subdir)
    if anchor_mesh is None:
        anchor_mesh = load_object_mesh(seg_dir, frames[0]["frame_index"], subdir=pose_subdir)
    if anchor_mesh is None:
        return None
    obj_faces = np.array(anchor_mesh.faces)

    mano_fr, mano_fl = load_mano_faces()
    mano_faces_map = {"right": mano_fr, "left": mano_fl}

    K_default = load_moge_intrinsics(seg_dir, frame_indices[0])
    if K_default is None:
        for fi in frame_indices[1:10]:
            K_default = load_moge_intrinsics(seg_dir, fi)
            if K_default is not None:
                break
    if K_default is None:
        return None

    sam3_masks = np.load(mask_path)["masks"]
    H_img, W_img = sam3_masks.shape[1], sam3_masks.shape[2]

    raw_ious = []
    area_ratios = []

    # ── obj_shape observability: RGB appearance + border truncation ─────
    # Self-baked appearance consistency: sample each anchor-visible vertex's color
    # from the anchor video frame (SAM3D exports carry no real vertex colors,
    # with_texture_baking=False, so they are baked here), then check that a correct
    # shape+pose re-projects those colors onto same-colored video pixels at far
    # keyframes. Wrong reconstructions scramble the correspondence. Per-vertex
    # Pearson correlation; lighting never enters (video vs video). Low-texture
    # objects (baked std ~0) are uninformative → None. Border truncation counts
    # keyframes whose SAM3 mask touches the image border: a reconstruction from a
    # partially visible object can get the shape wrong (a glass as a squat disc).
    border_touch = []

    for t_idx in range(T):
        fi = frame_indices[t_idx]
        if fi >= len(sam3_masks):
            continue

        sam3_mask = sam3_masks[fi].astype(bool)
        K_frame = load_moge_intrinsics(seg_dir, fi)
        if K_frame is None:
            K_frame = K_default

        # Render object silhouette
        obj_sil = _render_silhouette(obj_verts[t_idx], obj_faces, K_frame, H_img, W_img)

        # Render hand silhouettes
        hand_sil = np.zeros((H_img, W_img), dtype=bool)
        for side in ("left", "right"):
            hv_key, valid_key = f"hand_{side}_verts", f"hand_{side}_valid"
            if hv_key in d and valid_key in d and d[valid_key][t_idx]:
                h_sil = _render_silhouette(d[hv_key][t_idx], mano_faces_map[side], K_frame, H_img, W_img)
                hand_sil |= h_sil

        # Raw IoU
        inter = (obj_sil & sam3_mask).sum()
        union = (obj_sil | sam3_mask).sum()
        raw_ious.append(float(inter / union) if union > 0 else 0.0)

        # Area ratio on low-occlusion frames
        obj_total = obj_sil.sum()
        occluded = (obj_sil & hand_sil).sum()
        occ_ratio = occluded / obj_total if obj_total > 0 else 0
        if occ_ratio < 0.2 and obj_total > 100:
            area_ratios.append(float(sam3_mask.sum() / obj_total))

        # obj_shape inputs
        if sam3_mask.any():
            border_touch.append(bool(sam3_mask[:2, :].any() or sam3_mask[-2:, :].any() or sam3_mask[:, :2].any() or sam3_mask[:, -2:].any()))

    sil_iou = float(np.mean(raw_ious)) if raw_ious else 0.0
    # The projected-overlap ratios when any frame qualifies; otherwise a second,
    # depth-aware rendering pass (gate_geometry).
    area_ratio = _resolve_area_ratio(
        area_ratios,
        lambda: _depth_aware_area_ratios(
            seg_dir,
            d,
            obj_verts,
            obj_faces,
            frame_indices,
            sam3_masks,
            mano_faces_map,
            K_default,
            H_img,
            W_img,
        ),
    )
    hand_occluders = [
        (
            d[f"hand_{side}_verts"],
            mano_faces_map[side],
            d[f"hand_{side}_valid"].astype(bool),
        )
        for side in ("left", "right")
        if f"hand_{side}_verts" in d and f"hand_{side}_valid" in d
    ]
    obj_rgb_corr = _obj_rgb_consistency(
        seg_dir,
        obj_verts,
        obj_faces,
        frame_indices,
        anchor_idx,
        K_default,
        H_img,
        W_img,
        hands=hand_occluders,
    )
    mask_border_frac = float(np.mean(border_touch)) if border_touch else None
    sil_iou_p10 = float(np.percentile(raw_ious, 10)) if raw_ious else 0.0

    if image_metrics_only:
        min_hand_obj_dist, obj_motion_range = _interaction_metrics(d)
        return {
            "sil_iou": round(sil_iou, 4),
            "area_ratio": round(area_ratio, 4) if area_ratio is not None else None,
            "iou_trend": round(iou_trend, 4),
            "min_hand_obj_dist": (round(min_hand_obj_dist, 4) if min_hand_obj_dist is not None else None),
            "obj_motion_range": round(obj_motion_range, 4),
            "obj_rgb_corr": (round(obj_rgb_corr, 4) if obj_rgb_corr is not None else None),
            "mask_border_frac": (round(mask_border_frac, 4) if mask_border_frac is not None else None),
            "sil_iou_p10": round(sil_iou_p10, 4),
        }

    # ── Gate 9: hand_scale ──────────────────────────────────────────────
    hand_scale_med, hand_scale_drift = _hand_scale_metrics(d)

    # ── Gate 10: interaction ────────────────────────────────────────────
    min_hand_obj_dist, obj_motion_range = _interaction_metrics(d)

    # ── Gate 11: scene_cut ──────────────────────────────────────────────
    n_scene_cuts = None
    video_path = os.path.join(seg_dir, "video.mp4")
    if os.path.exists(video_path):
        from gtf.extraction.frontend.detect_cuts import detect_scene_cuts

        cuts, _, _ = detect_scene_cuts(video_path)
        if window_scoped:
            # A cut is an interval boundary, not per-row evidence.  A cut on a
            # skipped video frame between two exact-roster rows still corrupts
            # their trajectory and therefore remains in scope.
            cuts = [c for c in cuts if window[0] <= c <= window[1]]
        n_scene_cuts = len(cuts)

    # ── Gate 12: penetration (winding-number sign, hoi_contact.py) ─────
    from gtf.extraction.gates.hoi_contact import segment_contact_metrics

    pen = segment_contact_metrics(d)
    worst = pen.get("worst")
    contact_frac_max = max((m["contact_frac"] for s, m in pen.items() if s != "worst"), default=None)

    # ── Observability (not a gate): camera rotation speed ──────────────
    # The whole chain assumes a quasi-static camera (states live in per-frame camera
    # coords with no re-expression). Recorded so camera motion can be related to gate
    # failures.
    cam_rot_speed = _camera_rotation_speed(seg_dir, video_path)

    # ── Gates 14 & 15: interaction fidelity (contact_cover, rot_jerk) ────
    # Video-grounded interaction checks the physics/image gates are structurally blind to:
    # ghost (off-object) contact and non-rigid/utensil grasp wobble (interaction_metrics.py;
    # e.g. a stirring spoon tracked as a hand passes the other gates). Computed on the gated
    # chain `d`, so not an IMAGE_METRICS entry (no swap to the raw extraction).
    from gtf.extraction.gates.interaction_metrics import contact_cover_metric, interaction_fidelity_metrics
    from gtf.extraction.gates.reground import load_reground_scores  # gate 21

    # Gate 14 (rot_jerk) sees the object-pose-refined chain via the sidecar `objref_chain.npz`
    # that export_canonical's refiner persists (the refiner runs once, in export). Gate 15's
    # contact_cover is taken from the pre-objref chain below. Falls back to `d` when the
    # sidecar is absent.
    # The sidecar may be cropped to the interaction window, so its own frame_indices and
    # valid masks are taken too: gates 14-18 then measure exactly the window the canonical trains on.
    d_if = d
    sidecar_p = os.path.join(seg_dir, opt_subdir, "objref_chain.npz")
    if os.path.exists(sidecar_p):
        sc = np.load(sidecar_p)
        d_if = dict(d)
        for k in (
            "obj_verts",
            "hand_left_verts",
            "hand_right_verts",
            "hand_left_valid",
            "hand_right_valid",
            "frame_indices",
        ):
            if k in sc.files:
                d_if[k] = sc[k]
    objref_sidecar_used = d_if is not d
    objref_sidecar_sha256 = _sha256_file(sidecar_p) if objref_sidecar_used else None
    fi_if = np.asarray(d_if["frame_indices"])
    K_list_if = []
    for fi in fi_if:
        Kf = load_moge_intrinsics(seg_dir, fi)
        K_list_if.append(Kf if Kf is not None else K_default)
    if_metrics = interaction_fidelity_metrics(d_if, sam3_masks, K_list_if, fi_if, seg_dir=seg_dir)

    # Gate 15's contact_cover reads the pre-objref chain, so it scores the optimizer's contact and
    # not the contact runs in which the refiner carries the hand with the object.
    # rot_jerk and gates 16-18 keep the sidecar view; the sidecar contact is recorded as
    # `contact_cover_objref` for observability.
    contact_cover_objref = if_metrics["contact_cover"]
    if d_if is not d:
        fi_raw = np.asarray(d["frame_indices"])
        K_list_raw = []
        for fi in fi_raw:
            Kf = load_moge_intrinsics(seg_dir, fi)
            K_list_raw.append(Kf if Kf is not None else K_default)
        if_metrics = dict(if_metrics)
        if_metrics["contact_cover"] = contact_cover_metric(d, sam3_masks, K_list_raw, fi_raw)

    # ── Gates 16-18: motion consistency (motion2d, size_consistency, interaction_sustain) ──
    # Video-grounded motion-level checks: the per-frame gates are blind to phantom/missing
    # in-plane motion, object depth wander (a size-trend disagreement), and gesture-only clips
    # whose single approach frame satisfies gate 10. Same chain as 14/15.
    from gtf.extraction.gates.interaction_metrics import motion_consistency_metrics

    mc_metrics = motion_consistency_metrics(d_if, sam3_masks, K_list_if, fi_if)

    # ── Gate 20: the mask is the hand (obj_is_hand) ─────────────────────────
    # Agnostic pointing can bind to the hand itself; the hand-mask "object" is then
    # self-consistently wrong through recon/pose/contact and passes every prior gate.
    # Median SAM3-mask coverage by the projected MANO hand silhouette over the same
    # window frames the other interaction gates measure.
    from gtf.extraction.gates.interaction_metrics import mask_hand_coverage

    mask_in_hand = None
    try:
        hands_packed = load_hand_data(seg_dir)
        if hands_packed:
            mano_fr, mano_fl = load_mano_faces()
            mask_in_hand = mask_hand_coverage(
                sam3_masks,
                hands_packed,
                {"right": mano_fr, "left": mano_fl},
                frame_indices=fi_if,
            )
    except Exception:
        mask_in_hand = None

    result = {
        "segment": seg_name,
        "n_frames": T,
        "window_scoped": bool(window_scoped),
        "window_frames": [int(window[0]), int(window[1])] if window_scoped else None,
        "objref_sidecar_used": objref_sidecar_used,
        "objref_sidecar_sha256": objref_sidecar_sha256,
        "vol_cv": round(vol_cv, 4),
        "acc_max": round(acc_max, 4),
        "ang_acc_max": round(ang_acc_max, 2),
        "sil_iou": round(sil_iou, 4),
        "area_ratio": round(area_ratio, 4) if area_ratio is not None else None,
        "rel_pose_max": round(rel_pose_max, 4),
        "obj_diag_m": round(obj_diag, 4),
        "iou_trend": round(iou_trend, 4),
        "hand_rate": round(best_hand, 4),
        "hand_scale_med": round(hand_scale_med, 4) if hand_scale_med is not None else None,
        "hand_scale_drift": round(hand_scale_drift, 4) if hand_scale_drift is not None else None,
        "min_hand_obj_dist": round(min_hand_obj_dist, 4) if min_hand_obj_dist is not None else None,
        "obj_motion_range": round(obj_motion_range, 4),
        "n_scene_cuts": n_scene_cuts,
        "cam_rot_speed": round(cam_rot_speed, 2) if cam_rot_speed is not None else None,
        "pen_p90_mm": worst["pen_p90_mm"] if worst else None,
        "pen_max_mm": worst["pen_max_mm"] if worst else None,
        "pen_side": worst["side"] if worst else None,
        "contact_frac_max": contact_frac_max,
        "obj_rgb_corr": round(obj_rgb_corr, 4) if obj_rgb_corr is not None else None,
        "mask_border_frac": round(mask_border_frac, 4) if mask_border_frac is not None else None,
        "sil_iou_p10": round(sil_iou_p10, 4),
        "contact_cover": round(if_metrics["contact_cover"], 4) if if_metrics["contact_cover"] is not None else None,
        "contact_cover_objref": round(contact_cover_objref, 4) if contact_cover_objref is not None else None,
        "rot_jerk": round(if_metrics["rot_jerk"], 4) if if_metrics["rot_jerk"] is not None else None,
        "mask_in_hand": round(mask_in_hand, 4) if mask_in_hand is not None else None,
        # soft interaction-fidelity axes (not gates: they over-flag good pours/grasps; fed to the
        # continuous interaction_fidelity_score sample weight in build_manifest)
        "obj_orient": round(if_metrics["obj_orient"], 4) if if_metrics["obj_orient"] is not None else None,
        "obj_centroid": round(if_metrics["obj_centroid"], 4) if if_metrics["obj_centroid"] is not None else None,
        "hand_drift": round(if_metrics["hand_drift"], 4) if if_metrics["hand_drift"] is not None else None,
        # motion-consistency metrics (gates 16-18 + observability)
        **{k: (round(v, 4) if v is not None else None) for k, v in mc_metrics.items()},
        # re-grounding verifier axes (gate 21): from {seg}/reground/reground.json when the
        # orchestrator's reground step has run; all None (the gate defers) otherwise
        **load_reground_scores(seg_dir),
    }
    return result


def _atomic_json(path, payload):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.tmp-", dir=directory)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def process_one(seg_name, thresh, force=False, opt_subdir="optimized_mano", results_dir=None):
    """Process one segment: compute metrics, evaluate gates, save quality.json."""
    results_root = os.fspath(RESULTS_DIR if results_dir is None else results_dir)
    out_dir = os.path.join(results_root, seg_name, "quality")
    out_path = os.path.join(out_dir, "quality.json")
    seg_dir = os.path.join(results_root, seg_name)
    canonical_path = os.path.join(seg_dir, "canonical", "canonical.npz")
    reground_path = os.path.join(seg_dir, "reground", "reground.json")
    stale_existing = False

    if not force and os.path.exists(out_path):
        with open(out_path) as f:
            result = json.load(f)
        if not os.path.isfile(canonical_path):
            raise RuntimeError(f"existing quality cannot bind missing canonical: {canonical_path}")
        canonical_sha = _sha256_file(canonical_path)
        try:
            with np.load(canonical_path, allow_pickle=False) as canonical:
                if "frame_indices" not in canonical.files:
                    raise RuntimeError(f"existing quality canonical has no frame_indices: {canonical_path}")
                n_frames = int(np.asarray(canonical["frame_indices"]).size)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"cannot validate existing quality canonical: {canonical_path}") from exc
        reground_sha = _sha256_file(reground_path) if os.path.isfile(reground_path) else None
        if (
            result.get("segment") == seg_name
            and result.get("n_frames") == n_frames
            and isinstance(result.get("window_scoped"), bool)
            and result.get("threshold_level") == thresh.get("_level", "custom")
            and result.get("opt_subdir") == opt_subdir
            and result.get("canonical_sha256") == canonical_sha
            and "reground_sha256" in result
            and result.get("reground_sha256") == reground_sha
        ):
            return result
        # The inputs changed: recompute instead of reusing metrics of another trajectory.
        stale_existing = True

    metrics = compute_metrics(seg_name, opt_subdir=opt_subdir, results_dir=results_root)
    if metrics is None:
        if stale_existing:
            raise RuntimeError("existing quality is not bound to the current re-ground/canonical inputs and a full recomputation is unavailable")
        return None
    if opt_subdir != "optimized_mano":
        raw = compute_metrics(
            seg_name,
            opt_subdir="optimized_mano",
            results_dir=results_root,
            image_metrics_only=True,
        )
        if raw is not None:
            for k in IMAGE_METRICS:
                metrics[k + "_refined"] = metrics[k]
                metrics[k] = raw[k]

    gates, failed = evaluate_gates(metrics, thresh)
    result = {
        **metrics,
        "pass": len(failed) == 0,
        "failed_gates": failed,
        "gates": gates,
        "threshold_level": thresh.get("_level", "custom"),
        "opt_subdir": opt_subdir,
    }

    if not os.path.isfile(canonical_path):
        raise RuntimeError(f"quality output cannot bind missing canonical: {canonical_path}")
    result["canonical_sha256"] = _sha256_file(canonical_path)
    result["reground_sha256"] = _sha256_file(reground_path) if os.path.isfile(reground_path) else None

    _atomic_json(out_path, result)

    return result


def mark_reground_unverified(
    seg_name,
    *,
    status,
    support_frames=None,
    results_dir=None,
):
    """Fail an existing quality row whose fresh re-grounding evidence is missing.

    The metric axes stay ``None`` (no refusal rate or distance is invented), but the
    segment must not pass without the evidence: the ``obj_reground`` gate is set to
    failed, the evidence status is recorded and the re-ground digest refreshed.
    """

    if status not in {
        "collection_failed",
        "insufficient_support",
        "quality_recompute_failed",
    }:
        raise ValueError(f"unsupported re-ground evidence status: {status!r}")
    results_root = os.fspath(RESULTS_DIR if results_dir is None else results_dir)
    seg_dir = os.path.join(results_root, seg_name)
    quality_path = os.path.join(seg_dir, "quality", "quality.json")
    if not os.path.isfile(quality_path):
        raise RuntimeError(f"cannot fail-close missing quality row for {seg_name}: {quality_path}")
    with open(quality_path) as handle:
        result = json.load(handle)
    gates = result.get("gates")
    if not isinstance(gates, dict):
        raise RuntimeError(f"quality row has no gate map: {quality_path}")
    gates["obj_reground"] = False
    failed = result.get("failed_gates")
    if not isinstance(failed, list) or any(not isinstance(item, str) for item in failed):
        raise RuntimeError(f"quality row has a malformed failure roster: {quality_path}")
    result["failed_gates"] = sorted(set(failed) | {"obj_reground"})
    result["pass"] = False
    result["reground_evidence_status"] = status
    result["reground_support_frames"] = int(support_frames) if support_frames is not None else None
    reground_path = os.path.join(seg_dir, "reground", "reground.json")
    result["reground_sha256"] = _sha256_file(reground_path) if os.path.isfile(reground_path) else None
    _atomic_json(quality_path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m gtf.extraction.gates.quality_gate", description="Quality gate for hand-object sequences")
    parser.add_argument("--segment", type=str, help="Process a single segment")
    parser.add_argument("--level", choices=["loose", "moderate", "strict"], default="moderate")
    parser.add_argument("--force", action="store_true", help="Recompute even if quality.json exists")
    parser.add_argument(
        "--results-dir",
        default=RESULTS_DIR,
        help="Results root to read and write (default: RESULTS_DIR)",
    )
    args = parser.parse_args(argv)
    results_root = os.fspath(args.results_dir)

    thresh = {**THRESHOLDS[args.level], "_level": args.level}

    if args.segment:
        segments = [args.segment]
    else:
        segments = sorted(
            s
            for s in os.listdir(results_root)
            if os.path.isdir(os.path.join(results_root, s)) and os.path.exists(os.path.join(results_root, s, "optimized_mano", "optimized.npz"))
        )

    print(f"Quality gate: {len(segments)} segments, level={args.level}")
    print(f"Thresholds: {json.dumps({k: v for k, v in thresh.items() if k != '_level'}, indent=2)}\n")

    results = []
    skipped = []
    t0 = time.time()
    for i, seg in enumerate(segments):
        # Match the orchestrator's chain selection: gate the stage-2-refined
        # trajectory (what the model trains on) whenever it exists.
        subdir = "optimized_stage2" if os.path.exists(os.path.join(results_root, seg, "optimized_stage2", "optimized.npz")) else "optimized_mano"
        r = process_one(seg, thresh, force=args.force, opt_subdir=subdir, results_dir=results_root)
        if r is None:
            print(f"  [{i + 1}/{len(segments)}] {seg}: SKIP (missing data)")
            skipped.append(seg)
            continue

        status = "PASS" if r["pass"] else f"FAIL ({', '.join(r['failed_gates'])})"
        print(f"  [{i + 1}/{len(segments)}] {seg}: {status}")
        results.append(r)

    elapsed = time.time() - t0

    # ── Summary ─────────────────────────────────────────────────────────
    n_pass = sum(1 for r in results if r["pass"])
    n_fail = len(results) - n_pass
    print(f"\nDone in {elapsed:.1f}s. {n_pass}/{len(results)} pass, {n_fail} rejected.")

    # A run over all segments skips incomplete ones, but an explicitly requested
    # segment must get fresh metrics: returning success without them could leave a
    # stale quality result in place.
    if args.segment and skipped:
        raise SystemExit("quality gate skipped a requested segment with missing data: " + ", ".join(skipped))

    # Per-gate rejection count
    gate_counts = {}
    for r in results:
        for g in r["failed_gates"]:
            gate_counts[g] = gate_counts.get(g, 0) + 1
    if gate_counts:
        print("\nPer-gate rejections:")
        for g, c in sorted(gate_counts.items(), key=lambda x: -x[1]):
            print(f"  {g:20s}: {c}")

    # ── Write summary CSV ───────────────────────────────────────────────
    csv_path = os.path.join(results_root, "quality_summary.csv")
    fields = [
        "segment",
        "pass",
        "failed_gates",
        "n_frames",
        "vol_cv",
        "acc_max",
        "ang_acc_max",
        "sil_iou",
        "area_ratio",
        "rel_pose_max",
        "iou_trend",
        "hand_rate",
        "hand_scale_med",
        "hand_scale_drift",
        "min_hand_obj_dist",
        "obj_motion_range",
        "n_scene_cuts",
        "cam_rot_speed",
        "pen_p90_mm",
        "pen_max_mm",
        "pen_side",
        "contact_frac_max",
        "obj_rgb_corr",
        "mask_border_frac",
        "sil_iou_p10",
        "threshold_level",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(results, key=lambda x: x["segment"]):
            row = {**r, "failed_gates": ";".join(r["failed_gates"])}
            writer.writerow(row)
    print(f"\nSummary CSV: {csv_path}")


if __name__ == "__main__":
    main()
