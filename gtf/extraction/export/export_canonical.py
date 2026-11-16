#!/usr/bin/env python3
# ruff: noqa: E402
"""Export canonical.npz from optimized MANO results.

Reads optimized.npz (stage-2 refined when present), crops it to the interaction window, refines the object pose against
the SAM 3 masks, transforms everything to the anchor camera frame, recovers MANO / object parameters by Procrustes,
estimates gravity and writes canonical/canonical.npz + metadata.json (plus one `{segment}__iwN` child per extra
interaction window).

Usage:
    uv run python -m gtf.extraction.export.export_canonical --segment "segment_name"
    uv run python -m gtf.extraction.export.export_canonical --force --object cup
"""

import argparse
import json
import os
import traceback

import cv2
import numpy as np
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation

from gtf.paths import MANO_DIR

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from gtf.utils import (
    RESULTS_DIR,
    is_action_valid,
    load_moge_depth,
    load_moge_intrinsics,
    get_segment_object,
    rprint as print,
)  # noqa: E402
from gtf.utils.canonical import (
    SUPPORT_PLANE_SOURCE,
    rigid_procrustes,
    transform_to_anchor_frame,
    estimate_gravity_support,
    validate_canonical_npz,
)  # noqa: E402
from gtf.utils.npz import savez_compressed_level  # noqa: E402

from gtf.extraction import chains  # noqa: E402
from gtf.utils.video_timing import (  # noqa: E402
    FRAME_TIME_CONVENTION,
    array_sha256,
    probe_selected_video_timing,
    validate_exact_timing_arrays,
)

from gtf.extraction.export.canonical_records import CANONICAL_TRANSACTION_FORMAT, CANONICAL_TRANSACTION_MARKER, INTERACTION_WINDOW_CHILDREN_FIELD  # noqa: E402
from gtf.extraction.export.canonical_records import _file_sha256, _json_safe  # noqa: E402
from gtf.extraction.export.canonical_records import _retire_unexpected_interaction_window_children, _write_json_atomic  # noqa: E402
from gtf.extraction.export.canonical_records import declared_interaction_window_children, outputs_complete  # noqa: E402, F401
from gtf.extraction.export.export_windows import _estimate_camera_poses, _interaction_windows, _quality_hard_bad  # noqa: E402


def _make_window_child(parent_name, child_name, first_frame, last_frame, window_index):
    """Create a child segment dir for an extra interaction window: symlink every upstream
    artifact from the parent (the extraction is shared, only the exported window differs),
    give the child a real optimized_stage2/ dir (so its export writes its own objref sidecar
    without clobbering the parent's) and a root metadata.json carrying the window, which
    process_one_segment's step 1a honors instead of re-detecting. Idempotent."""
    parent_dir = os.path.join(str(RESULTS_DIR), parent_name)
    child_dir = os.path.join(str(RESULTS_DIR), child_name)
    os.makedirs(child_dir, exist_ok=True)
    for item in (
        "video.mp4",
        "video_full.mp4",
        "valid_action",
        "SAM3",
        "sam3d",
        "sam3d_c",
        "molmo2",
        "object_pose",
        "optimized_mano",
        "hands",
        "wilor",
        "cut_check.json",
    ):
        src = os.path.join(parent_dir, item)
        dst = os.path.join(child_dir, item)
        if os.path.exists(src) and not os.path.lexists(dst):
            os.symlink(os.path.join("..", parent_name, item), dst)
    for sub in ("optimized_stage2",):
        src_dir = os.path.join(parent_dir, sub)
        if not os.path.isdir(src_dir):
            continue
        dst_dir = os.path.join(child_dir, sub)
        os.makedirs(dst_dir, exist_ok=True)
        for f in ("optimized.npz", "refine_report.json"):
            src = os.path.join(src_dir, f)
            dst = os.path.join(dst_dir, f)
            if os.path.exists(src) and not os.path.lexists(dst):
                os.symlink(os.path.join("..", "..", parent_name, sub, f), dst)
        # the child's export writes its own objref_chain.npz here; drop a stale one
        stale = os.path.join(dst_dir, "objref_chain.npz")
        if os.path.lexists(stale) and os.path.islink(stale):
            os.unlink(stale)
    meta = {}
    pmeta_p = os.path.join(parent_dir, "metadata.json")
    if os.path.exists(pmeta_p):
        with open(pmeta_p) as f:
            meta = json.load(f)
    meta["interaction_window"] = {
        "parent": parent_name,
        "index": int(window_index),
        "first_frame": int(first_frame),
        "last_frame": int(last_frame),
    }
    _write_json_atomic(os.path.join(child_dir, "metadata.json"), meta)
    return child_dir


def process_one_segment(segment_name, opt_subdir="optimized_mano", *, chain=None):
    """Export canonical.npz for a single segment. Returns True on success.

    opt_subdir: which optimized.npz to export — "optimized_stage2" when the stage-2
    trajectory refinement (gtf/extraction/stage2/refine_trajectory.py) has run; its corrections are
    vert-level, so the per-frame Procrustes below carries them into the canonical params.
    chain: the extraction chain (chains.py; None = $EXTRACTION_CHAIN, else production): its object-refiner options and
    whether the smoother fitter (step 7b) runs."""
    if opt_subdir not in {"optimized_mano", "optimized_stage2"}:
        raise ValueError(f"unsupported optimized source: {opt_subdir}")
    chain = chains.get(chain)
    segment_dir = os.path.join(str(RESULTS_DIR), segment_name)
    opt_path = os.path.join(segment_dir, opt_subdir, "optimized.npz")
    out_dir = os.path.join(segment_dir, "canonical")
    out_path = os.path.join(out_dir, "canonical.npz")
    meta_out_path = os.path.join(out_dir, "metadata.json")
    transaction_path = os.path.join(out_dir, CANONICAL_TRANSACTION_MARKER)
    Path(transaction_path).unlink(missing_ok=True)

    # ---- 1. Load optimized.npz ----
    optimized_sha256 = _file_sha256(opt_path)
    with np.load(opt_path, allow_pickle=True) as d_npz:
        d = {k: d_npz[k] for k in d_npz.files}

    anchor_frame_index = int(d["anchor_frame_index"])
    canonical_verts_raw = d["canonical_verts"]
    obj_faces = d["obj_faces"]

    # ---- 1a. Interaction-window crop ----
    # The object chain is only meaningful while the object is tracked (mask present) and the
    # interaction is live: before pickup, after set-down or before the object enters the frame, the
    # per-keyframe SAM 3D + MoGe chain wanders (object motion with no hand near). The export
    # is cropped to the contact-and-visible window (padded to keep approach / release context,
    # clamped to the mask-visible extent); no crop when the window is too short (the
    # interaction_sustain gate decides then) or when it would trim less than ~0.8 s. The crop runs
    # before the object-pose refiner, so the objref sidecar and the canonical stay index-aligned
    # (quality_gate reads both).
    # Multiple windows: a clip can hold several disconnected interactions (pick, set down, pick
    # again). Each merged contact run of >= 8 core keyframes is a window. The primary window (most
    # contact keyframes) is exported here as canonical/; every extra window becomes a child segment
    # {segment}__iwN (symlinked upstream artifacts, its own canonical) exported at step 11. A child
    # directory carries "interaction_window" in its root metadata.json, used here instead of
    # re-detection.
    seg_meta = {}
    root_meta_path = os.path.join(segment_dir, "metadata.json")
    if os.path.exists(root_meta_path):
        with open(root_meta_path) as f:
            seg_meta = json.load(f)
    window_child = seg_meta.get("interaction_window")  # set on __iwN child dirs
    interaction_crop = None
    extra_windows = []
    masks_p = os.path.join(segment_dir, "SAM3", "masks.npz")
    # masks.npz is small on disk but inflates to hundreds of MiB; window discovery and the object
    # refiner share one decompressed copy.
    sam3_masks_cache = None
    if os.path.exists(masks_p):
        fps_probe = 30.0
        vp_probe = os.path.join(segment_dir, "video.mp4")
        if os.path.exists(vp_probe):
            cap = cv2.VideoCapture(vp_probe)
            if cap.get(cv2.CAP_PROP_FPS) > 0:
                fps_probe = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
        win = None
        fi_all = np.asarray(d["frame_indices"], np.int64)
        if window_child is not None:
            sel = np.where((fi_all >= int(window_child["first_frame"])) & (fi_all <= int(window_child["last_frame"])))[0]
            if len(sel) >= 8:
                win = (int(sel[0]), int(sel[-1]))
            else:
                print(f"  SKIP: interaction_window {window_child['first_frame']}..{window_child['last_frame']} covers <8 keyframes")
                return False
        else:
            try:
                with np.load(masks_p, allow_pickle=False) as masks_archive:
                    sam3_masks_cache = masks_archive["masks"]
                # Quality windows: chain-corrupt keyframes leave the window cores, split them and
                # clamp their padding, so clean sub-trajectories of a segment whose whole span fails
                # the temporal gates still export as __iwN children.
                qb = _quality_hard_bad(d)
                if qb.any():
                    print(f"  Quality-window mask: {int(qb.sum())}/{len(qb)} chain-corrupt keyframes excised from window cores")
                wins = _interaction_windows(d, sam3_masks_cache, fps_probe, quality_bad=qb)
            except Exception as e:  # noqa: BLE001
                wins = []
                print(f"  Interaction-crop check SKIPPED ({type(e).__name__}: {e})")
            if wins:
                primary = max(range(len(wins)), key=lambda j: wins[j]["core_kf"])
                win = (wins[primary]["i0"], wins[primary]["i1"])
                extra_windows = [(int(fi_all[w["i0"]]), int(fi_all[w["i1"]])) for j, w in enumerate(wins) if j != primary]
                if extra_windows:
                    print(f"  Multi-window: {len(wins)} interaction windows (primary core_kf={wins[primary]['core_kf']}; {len(extra_windows)} extra -> child segments)")
        if win is not None:
            i0, i1 = win
            T_before = len(d["frame_indices"])
            per_frame = ["frame_indices", "obj_verts", "obj_scale_init"] + [f"hand_{s}_{f}" for s in ("left", "right") for f in ("verts", "valid", "pose")]
            for k in per_frame:
                if k in d and getattr(d[k], "ndim", 0) >= 1 and len(d[k]) == T_before:
                    d[k] = d[k][i0 : i1 + 1]
            interaction_crop = {
                "applied": True,
                "kf_before": int(T_before),
                "kf_kept": int(i1 - i0 + 1),
                "first_frame": int(d["frame_indices"][0]),
                "last_frame": int(d["frame_indices"][-1]),
            }
            if window_child is not None:
                interaction_crop["window_index"] = int(window_child["index"])
                interaction_crop["window_parent"] = window_child["parent"]
            print(
                f"  Interaction crop: keyframes {T_before} -> {interaction_crop['kf_kept']} (video frames {interaction_crop['first_frame']}..{interaction_crop['last_frame']})"
            )

    frame_indices = d["frame_indices"]
    obj_verts = d["obj_verts"]
    obj_scale_init = d["obj_scale_init"]
    T = len(frame_indices)

    # ---- 1b. Object-pose refinement ----
    # Re-poses the object mesh per keyframe to maximize its silhouette IoU with the SAM 3 mask (with
    # each frame's MoGe intrinsics) and carries the grasping hand with it: a whole grasp run moves as
    # one rigid body, so the hand-object relative pose (contact, rotation jerk) is preserved. It runs
    # on the per-frame camera verts (before the anchor transform) so the silhouette projects
    # correctly; the refined object and carried hand verts then flow through the transform and the
    # Procrustes fits below. Skipped when the SAM 3 masks or MoGe depth are unavailable.
    object_pose_refined = False
    refined_hands_pf = None
    obj_refine_report = {}
    if os.path.exists(masks_p):
        try:
            from gtf.extraction.export.object_pose_refine import refine_object_trajectory

            if sam3_masks_cache is None:
                with np.load(masks_p, allow_pickle=False) as masks_archive:
                    sam3_masks_cache = masks_archive["masks"]
            masks_r = sam3_masks_cache
            Himg, Wimg = int(masks_r.shape[1]), int(masks_r.shape[2])
            K_list = []
            for fi in frame_indices:
                Km = None
                if int(fi) < len(masks_r):
                    Km = load_moge_intrinsics(segment_dir, int(fi))
                if Km is None:
                    Km = np.array(
                        [
                            [1.25 * Wimg, 0, Wimg / 2],
                            [0, 1.25 * Wimg, Himg / 2],
                            [0, 0, 1.0],
                        ]
                    )
                K_list.append(np.asarray(Km, np.float64))
            hands_pf = {s: d[f"hand_{s}_verts"].astype(np.float64) for s in ("left", "right")}
            valid_pf = {s: np.asarray(d[f"hand_{s}_valid"]).astype(bool) for s in ("left", "right")}
            obj_verts, obj_refine_report = refine_object_trajectory(
                obj_verts.astype(np.float64),
                obj_faces,
                K_list,
                masks_r,
                frame_indices,
                hand_verts=hands_pf,
                hand_valid=valid_pf,
                checkpoint_path=os.path.join(
                    segment_dir,
                    ".objref_deltas_checkpoint.npz",
                ),
                options=chain.object_refine,
            )
            refined_hands_pf = obj_refine_report.pop("hand_verts", None)
            object_pose_refined = True
            # The refined per-frame chain, so quality_gate's interaction metrics (gates 14/15:
            # contact_cover, rot_jerk) see the refinement without re-running the refiner.
            sidecar = {
                "obj_verts": obj_verts.astype(np.float32),
                "frame_indices": frame_indices,
                "obj_faces": obj_faces,
            }
            for s in ("left", "right"):
                hv = refined_hands_pf[s] if refined_hands_pf is not None else d[f"hand_{s}_verts"]
                sidecar[f"hand_{s}_verts"] = np.asarray(hv, np.float32)
                sidecar[f"hand_{s}_valid"] = valid_pf[s]
            sidecar["grasp_carry_report_json"] = np.str_(
                json.dumps(
                    _json_safe(obj_refine_report),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
            savez_compressed_level(
                os.path.join(segment_dir, opt_subdir, "objref_chain.npz"),
                compresslevel=6,
                allow_pickle=False,
                **sidecar,
            )
            print(
                f"  Object-pose refine: accepted {obj_refine_report['n_accepted']}/{obj_refine_report['n_kf']} kf, carried {obj_refine_report.get('n_carried_kf', 0)} hand-kf, forced {obj_refine_report.get('n_forced_obj', 0)}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"  Object-pose refine SKIPPED ({type(e).__name__}: {e})")

    # ---- 2. Camera poses ----
    cam_to_world = _estimate_camera_poses(segment_dir)
    camera_aligned = cam_to_world is not None

    if not camera_aligned:
        print("  WARNING: No camera poses available, assuming static camera")

    # ---- 3. Transform to anchor frame ----
    if camera_aligned:
        obj_verts_anchor = transform_to_anchor_frame(obj_verts, cam_to_world, frame_indices, anchor_frame_index)
    else:
        obj_verts_anchor = obj_verts.copy()

    hand_verts_anchor = {}
    hand_valid = {}
    for side in ("left", "right"):
        hv_key = f"hand_{side}_verts"
        val_key = f"hand_{side}_valid"
        hand_valid[side] = d[val_key]
        # the object refiner's carried hand verts (they keep the grasp during contact) when present
        hv_src = refined_hands_pf[side] if refined_hands_pf is not None else d[hv_key]
        if camera_aligned:
            hv = transform_to_anchor_frame(hv_src, cam_to_world, frame_indices, anchor_frame_index)
        else:
            hv = np.asarray(hv_src).copy()
        # Re-zero invalid frames (zeros transformed by camera pose become nonzero)
        hv[~hand_valid[side]] = 0.0
        hand_verts_anchor[side] = hv

    # ---- 3b. Grasp coherence ----
    # Hand (WiLoR, per frame) and object (SAM 3D / BA) are estimated semi-independently, so their
    # relative pose wobbles during a grasp where it should be rigid. The hand's relative rotation in
    # the object body frame is smoothed and applied as a rigid, mean-preserving hand correction in the
    # anchor frame, before the hand Procrustes, so the Procrustes recovers consistent (orient, cam_t)
    # from the corrected verts (including the left-hand mirror). Rotation only.
    from gtf.extraction.export.grasp_coherence import apply_grasp_coherence

    gc_report = {}
    for side in ("left", "right"):
        if int(np.asarray(hand_valid[side]).sum()) >= 4:
            hand_verts_anchor[side], gc_report[side] = apply_grasp_coherence(
                hand_verts_anchor[side],
                hand_valid[side],
                obj_verts_anchor,
                contact_m=0.04,
                rot_radius=3,
                trans_radius=0,
                min_run=4,
            )
    fired = {s: r for s, r in gc_report.items() if r.get("n_runs")}
    if fired:
        print(f"  Grasp-coherence (rot-only): {fired}")

    # ---- 4. Hand Procrustes ----
    from manopth.manolayer import ManoLayer

    mano_layer = ManoLayer(mano_root=MANO_DIR, side="right", use_pca=False, ncomps=45, flat_hand_mean=True)

    hand_go_canonical = {}
    hand_ct_canonical = {}
    hand_procrustes_rmse = {}

    for side in ("left", "right"):
        pose_key = f"hand_{side}_pose"
        hand_pose = d[pose_key]
        betas = d[f"hand_{side}_betas"]
        valid = hand_valid[side]

        go_out = np.zeros((T, 3), dtype=np.float32)
        ct_out = np.zeros((T, 3), dtype=np.float32)
        rmse_out = np.zeros(T, dtype=np.float32)

        if not np.any(valid):
            hand_go_canonical[side] = go_out
            hand_ct_canonical[side] = ct_out
            hand_procrustes_rmse[side] = rmse_out
            continue

        betas_t = torch.tensor(betas, dtype=torch.float32).unsqueeze(0)

        for t in range(T):
            if not valid[t]:
                continue

            hp_t = torch.tensor(hand_pose[t : t + 1], dtype=torch.float32)

            # Step A: MANO FK with go=0 to get local-frame shape for Procrustes rotation
            full_pose_local = torch.cat([torch.zeros(1, 3), hp_t], dim=-1)
            v_local, _ = mano_layer(full_pose_local, th_betas=betas_t)
            v_local = v_local[0].numpy() / 1000.0  # (778, 3) meters

            target = hand_verts_anchor[side][t]  # (778, 3) in anchor frame

            # Left hand: un-mirror the target before Procrustes (the optimizer runs the right MANO
            # model and x-flips its output for the left hand)
            if side == "left":
                target = target * np.array([-1, 1, 1], dtype=np.float32)

            # Procrustes gives us the correct rotation
            R, _t_proc, _ = rigid_procrustes(v_local, target)
            go_vec = Rotation.from_matrix(R).as_rotvec().astype(np.float32)

            # Step B: MANO FK with derived go to get properly-rotated verts
            # (MANO LBS rotates around root joint, so R @ v_local != MANO(go, hp, betas))
            full_pose_go = torch.cat([torch.tensor(go_vec[None], dtype=torch.float32), hp_t], dim=-1)
            v_go, _ = mano_layer(full_pose_go, th_betas=betas_t)
            v_go = v_go[0].numpy() / 1000.0  # (778, 3) meters

            # Step C: cam_t = optimal translation from FK output to target
            ct_vec = (target - v_go).mean(axis=0).astype(np.float32)

            # Compute final RMSE
            rmse = float(np.sqrt(np.mean(np.sum((v_go + ct_vec - target) ** 2, axis=-1))))

            go_out[t] = go_vec
            ct_out[t] = ct_vec
            rmse_out[t] = rmse

        valid_rmse = rmse_out[valid]
        if len(valid_rmse) > 0:
            print(f"  Hand {side} Procrustes RMSE: mean={np.mean(valid_rmse) * 1000:.2f}mm, max={np.max(valid_rmse) * 1000:.2f}mm")

        hand_go_canonical[side] = go_out
        hand_ct_canonical[side] = ct_out
        hand_procrustes_rmse[side] = rmse_out

    # ---- 5. Object Procrustes ----
    canonical_verts_cv = canonical_verts_raw * np.array([-1, -1, 1], dtype=np.float32)

    # Anchor scale: obj_scale_init at keyframe matching anchor_frame_index
    anchor_kf_idx = None
    for t_idx, fi in enumerate(frame_indices):
        if int(fi) == anchor_frame_index:
            anchor_kf_idx = t_idx
            break
    if anchor_kf_idx is None:
        anchor_kf_idx = int(np.argmin(np.abs(frame_indices.astype(np.int64) - anchor_frame_index)))
        print(f"  WARNING: anchor_frame_index {anchor_frame_index} not in frame_indices, using nearest keyframe idx={anchor_kf_idx} (frame {frame_indices[anchor_kf_idx]})")

    anchor_scale = obj_scale_init[anchor_kf_idx]
    source_mesh = canonical_verts_cv * anchor_scale

    obj_rotation = np.zeros((T, 3), dtype=np.float32)
    obj_translation = np.zeros((T, 3), dtype=np.float32)
    obj_procrustes_rmse = np.zeros(T, dtype=np.float32)

    for t in range(T):
        R, t_vec, rmse = rigid_procrustes(source_mesh, obj_verts_anchor[t])
        obj_rotation[t] = Rotation.from_matrix(R).as_rotvec().astype(np.float32)
        obj_translation[t] = t_vec
        obj_procrustes_rmse[t] = rmse

    mean_obj_rmse = float(obj_procrustes_rmse.mean())
    max_obj_rmse = float(obj_procrustes_rmse.max())
    print(f"  Object Procrustes RMSE: mean={mean_obj_rmse * 1000:.2f}mm, max={max_obj_rmse * 1000:.2f}mm")
    if max_obj_rmse > 0.002:
        print(f"  WARNING: Object Procrustes max RMSE > 2mm ({max_obj_rmse * 1000:.2f}mm)")

    # ---- 5b. Contact projection ----
    # Envelope-capped rigid hand nudge that resolves residual penetration without moving the hand
    # off its video-observed position. Runs only on rigid object chains (Procrustes RMSE < 2 mm).
    # Modifies hand_ct_canonical in place; the FK-reconstructed verts below inherit the correction.
    if max_obj_rmse < 0.002:
        from gtf.extraction.export.contact_projection import project_contact_export

        cp_report = project_contact_export(
            hand_verts_anchor,
            hand_valid,
            hand_ct_canonical,
            canonical_verts_cv,
            obj_faces,
            anchor_scale,
            obj_rotation,
            obj_translation,
        )
        print(f"  Contact projection: {cp_report}")

    # ---- 6. Gravity ----
    # Support-plane estimator (gtf/utils/canonical.py): plane hypotheses pooled over the exported keyframes
    # (lifted into the anchor camera), sign from surface visibility, the plane the tracked hand / object
    # rests over; it abstains (has_gravity=False) when no plane is supported.
    gravity_direction = None
    support_plane_height = None
    has_gravity = False
    has_support_plane = False
    gravity_source = None
    has_scene = os.path.exists(os.path.join(segment_dir, "video.mp4"))

    gframes, ghands, gobjs = [], [], []
    n_gsel = 8  # keyframes lifted into the anchor camera
    w2c_anchor = None
    if camera_aligned and anchor_frame_index < len(cam_to_world):
        w2c_anchor = np.linalg.inv(cam_to_world[anchor_frame_index].astype(np.float64))
    for t in np.unique(np.linspace(0, T - 1, min(n_gsel, T)).astype(int)):
        fi = int(frame_indices[t])
        dm, K = load_moge_depth(segment_dir, fi)
        if dm is None or K is None:
            continue
        R_t = t_t = None
        if w2c_anchor is not None and fi < len(cam_to_world):
            Ta = w2c_anchor @ cam_to_world[fi].astype(np.float64)
            R_t, t_t = Ta[:3, :3], Ta[:3, 3]
        gframes.append({"depth": dm, "intrinsics": K, "R": R_t, "t": t_t})
        # canonical/anchor-frame hand verts are true camera-space points
        # (the left mirror is already undone), so no M-conjugation applies.
        hp = [hand_verts_anchor[s][t] for s in ("left", "right") if hand_valid[s][t]]
        ghands.append(np.concatenate(hp, axis=0).astype(np.float64) if hp else None)
        gobjs.append(np.asarray(obj_verts_anchor[t], np.float64))
    result, gravity_report = estimate_gravity_support(gframes, hand_points=ghands, object_points=gobjs, return_report=True)
    if result is not None:
        gravity_direction, support_plane_height = result
        has_gravity = True
        has_support_plane = True
        gravity_source = SUPPORT_PLANE_SOURCE
    else:
        print(f"  Gravity: no support plane found ({len(gframes)} frames, {gravity_report.get('n_clusters', 0)} candidate directions) — has_gravity=False")

    # ---- 7. FPS ----
    fps = 30.0
    video_path = os.path.join(segment_dir, "video.mp4")
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        fps_detected = cap.get(cv2.CAP_PROP_FPS)
        if fps_detected > 0:
            fps = fps_detected
        cap.release()

    # ---- 7b. Smoother fitter (chains with smooth=True; gtf/extraction/export/trajectory_smoother.py) ----
    # Re-fits the final hand MANO parameters and object pose with anisotropic data terms, smoothness on each keyframe's
    # own-camera trajectory, grasp coupling and the MANO natural-pose prior; the verts below are rebuilt from the
    # smoothed parameters. Strengths: trajectory_smoother.CALIBRATED.
    smooth_report = None
    if chain.smooth:
        from gtf.extraction.export import trajectory_smoother as tsm

        if camera_aligned:
            c2w_a = cam_to_world[anchor_frame_index].astype(np.float64)
            cam_E = np.stack([np.linalg.inv(cam_to_world[min(int(f_), len(cam_to_world) - 1)].astype(np.float64)) @ c2w_a for f_ in frame_indices])
            centres = np.stack([np.linalg.inv(e)[:3, 3] for e in cam_E])
        else:
            cam_E, centres = np.tile(np.eye(4), (T, 1, 1)), None
        hands_in = {}
        for side in ("left", "right"):
            if int(np.sum(hand_valid[side])) >= 3:
                hands_in[side] = dict(
                    valid=np.asarray(hand_valid[side], bool),
                    orient=hand_go_canonical[side],
                    pose=d[f"hand_{side}_pose"].astype(np.float32),
                    cam_t=hand_ct_canonical[side],
                    betas=d[f"hand_{side}_betas"],
                )
        sres, smooth_report = tsm.smooth(
            tsm.SmoothInputs(
                times=np.asarray(frame_indices, np.float64) / float(fps),
                obj_rotation=obj_rotation,
                obj_translation=obj_translation,
                obj_source=np.asarray(source_mesh, np.float64),
                hands=hands_in,
                cam_centers=centres,
                cam_E=cam_E,
            ),
            MANO_DIR,
            dict(tsm.CALIBRATED),
        )
        obj_rotation, obj_translation = sres["obj_rotation"], sres["obj_translation"]
        for side, hres in sres["hands"].items():
            hand_go_canonical[side], hand_ct_canonical[side] = hres["orient"], hres["cam_t"]
            d[f"hand_{side}_pose"] = hres["pose"]
        print(f"  Smoother fitter: loss {sum(smooth_report['loss_before'].values()):.1f} -> {sum(smooth_report['loss_after'].values()):.1f}")

    # ---- 8. Write canonical.npz ----
    os.makedirs(out_dir, exist_ok=True)

    # Reconstruct verts from params for self-consistency (params are the source of truth,
    # verts are derived for validation/visualization and must match exactly)
    obj_verts_reconstructed = np.zeros_like(obj_verts_anchor)
    for t in range(T):
        R_mat = Rotation.from_rotvec(obj_rotation[t]).as_matrix()
        obj_verts_reconstructed[t] = (R_mat @ source_mesh.T).T + obj_translation[t]

    save_dict = {
        "frame_indices": frame_indices,
        "fps": np.float32(fps),
        "obj_canonical_verts": canonical_verts_cv,
        "obj_faces": obj_faces,
        "obj_rotation": obj_rotation,
        "obj_translation": obj_translation,
        "obj_scale": anchor_scale,
        "obj_verts": obj_verts_reconstructed.astype(np.float32),
    }

    for side in ("left", "right"):
        valid = hand_valid[side]
        # Zero out hand_pose on invalid frames for consistency with zero verts/orient/cam_t
        hand_pose = d[f"hand_{side}_pose"].copy()
        hand_pose[~valid] = 0.0

        # Reconstruct hand verts from params for self-consistency
        go = hand_go_canonical[side]
        ct = hand_ct_canonical[side]
        betas_side = d[f"hand_{side}_betas"]
        betas_t = torch.tensor(betas_side, dtype=torch.float32).unsqueeze(0)
        hand_verts_reconstructed = np.zeros_like(hand_verts_anchor[side])
        for t in range(T):
            if not valid[t]:
                continue
            hp_t = torch.tensor(hand_pose[t : t + 1], dtype=torch.float32)
            go_t = torch.tensor(go[t : t + 1], dtype=torch.float32)
            fp = torch.cat([go_t, hp_t], dim=-1)
            v, _ = mano_layer(fp, th_betas=betas_t)
            v = (v[0] / 1000.0 + torch.tensor(ct[t], dtype=torch.float32)).numpy()
            if side == "left":
                v = v * np.array([-1, 1, 1], dtype=np.float32)
            hand_verts_reconstructed[t] = v

        save_dict[f"hand_{side}_valid"] = valid
        save_dict[f"hand_{side}_verts"] = hand_verts_reconstructed.astype(np.float32)
        save_dict[f"hand_{side}_orient"] = go
        save_dict[f"hand_{side}_pose"] = hand_pose.astype(np.float32)
        save_dict[f"hand_{side}_cam_t"] = ct
        save_dict[f"hand_{side}_betas"] = betas_side

    if has_gravity and gravity_direction is not None:
        save_dict["gravity_direction"] = gravity_direction
    if has_support_plane and support_plane_height is not None:
        save_dict["support_plane_height"] = np.float32(support_plane_height)

    # Exact physical time comes from the decoded video frame roster, after the interaction-window
    # crop has fixed the canonical frame identities. ``fps`` is a nominal scalar only.
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"cannot author exact canonical timing without video: {video_path}")
    frame_times_s, timing_provenance = probe_selected_video_timing(video_path, frame_indices)
    save_dict["frame_times_s"] = frame_times_s
    save_dict["frame_time_convention"] = np.asarray(FRAME_TIME_CONVENTION)

    if _file_sha256(opt_path) != optimized_sha256:
        raise RuntimeError("optimized input changed during canonical export; refusing to publish a mixed-generation canonical")

    savez_compressed_level(
        out_path,
        compresslevel=6,
        allow_pickle=False,
        **save_dict,
    )

    # ---- 9. Write metadata.json ----
    is_bimanual = np.any(hand_valid["left"]) and np.any(hand_valid["right"])
    obj_name = get_segment_object(segment_name) or "unknown"
    interaction_window_children = [f"{segment_name}__iw{index}" for index in range(1, len(extra_windows) + 1)] if window_child is None else []

    # Camera motion metric
    camera_motion_m = 0.0
    if camera_aligned and cam_to_world is not None:
        positions = cam_to_world[frame_indices, :3, 3]
        camera_motion_m = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())

    metadata = {
        "source_dataset": "action100m",
        "segment_id": segment_name,
        "confidence_tier": "extracted",
        "has_depth": True,
        "has_scene": has_scene,
        "has_gravity": has_gravity,
        "gravity_source": gravity_source,
        "gravity_estimate": _json_safe(gravity_report.get("accepted")) if gravity_report else None,
        "has_support_plane": has_support_plane,
        "has_native_hand_pose": False,
        "has_native_object_pose": False,
        "object_is_rigid": True,
        "primary_object_id": obj_name,
        "primary_object_category": obj_name,
        "bimanual": bool(is_bimanual),
        "viewpoint": "exocentric",
        "hand_source": "wilor_optimized",
        "object_source": "sam3d_optimized",
        "depth_source": "moge_v2",
        "mano_convention": "right_model_x_mirror",
        "camera_aligned": camera_aligned,
        "camera_motion_m": round(camera_motion_m, 4),
        "obj_procrustes_rmse_mean_mm": round(mean_obj_rmse * 1000, 3),
        "obj_procrustes_rmse_max_mm": round(max_obj_rmse * 1000, 3),
        "stage2_refined": opt_subdir != "optimized_mano",
        "object_pose_refined": bool(object_pose_refined),
        "object_refine": _json_safe(obj_refine_report) if object_pose_refined else None,
        "chain": chain.name,
        "smoother": _json_safe(smooth_report) if smooth_report is not None else None,
        "interaction_crop": interaction_crop,
        "interaction_window": window_child,
        INTERACTION_WINDOW_CHILDREN_FIELD: interaction_window_children,
        "optimized_subdir": opt_subdir,
        "optimized_sha256": optimized_sha256,
        "canonical_transaction_format": CANONICAL_TRANSACTION_FORMAT,
        **timing_provenance,
    }

    _write_json_atomic(meta_out_path, metadata)

    # ---- 10. Validate ----
    is_valid, errors = validate_canonical_npz(out_path)
    if not is_valid:
        print("  VALIDATION FAILED:")
        for e in errors:
            print(f"    {e}")
        return False
    try:
        with np.load(out_path, allow_pickle=False) as canonical:
            validated_indices, validated_times = validate_exact_timing_arrays(
                canonical["frame_indices"],
                canonical["frame_times_s"],
                canonical["frame_time_convention"],
            )
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"  EXACT TIMING VALIDATION FAILED: {exc}")
        return False

    frame_roster_sha = array_sha256(validated_indices, "<i8")
    frame_times_sha = array_sha256(validated_times, "<f8")
    if frame_times_sha != timing_provenance["selected_frame_times_sha256"]:
        print("  EXACT TIMING VALIDATION FAILED: selected timing digest changed")
        return False

    print(f"  canonical.npz written ({out_path})")

    # ---- 11. Extra interaction windows -> child segments ----
    # Each extra disjoint interaction window is its own training trajectory: build a symlink-farm
    # child dir and export it through this same function (its root metadata carries the window, so
    # step 1a crops to it: no re-detection, no recursion past depth 1). The parent transaction is
    # withheld unless every declared child commits. Downstream stages consume only this declared
    # roster, never an ambient {segment}__iw* glob.
    children_ok = True
    if interaction_window_children:
        for j, ((f0, f1), child_name) in enumerate(zip(extra_windows, interaction_window_children, strict=True), start=1):
            try:
                _make_window_child(segment_name, child_name, f0, f1, j)
                print(f"  Window child {child_name} (video frames {f0}..{f1}):")
                ok_child = process_one_segment(child_name, opt_subdir=opt_subdir, chain=chain)
                if not ok_child:
                    children_ok = False
                    print(f"  Window child {child_name} export FAILED (parent transaction withheld)")
            except Exception as e:  # noqa: BLE001
                children_ok = False
                print(f"  Window child {child_name} ERROR ({type(e).__name__}: {e}); parent transaction withheld")
    if not children_ok:
        print("  INCOMPLETE: one or more interaction-window children failed")
        return False
    if _file_sha256(opt_path) != optimized_sha256:
        print("  INCOMPLETE: optimized input changed before canonical transaction commit")
        return False
    try:
        _retire_unexpected_interaction_window_children(segment_dir, interaction_window_children)
    except OSError as exc:
        print(f"  INCOMPLETE: could not retire stale interaction-window children ({type(exc).__name__}: {exc})")
        return False
    if _file_sha256(opt_path) != optimized_sha256:
        print("  INCOMPLETE: optimized input changed while committing child roster")
        return False
    _write_json_atomic(
        transaction_path,
        {
            "format": CANONICAL_TRANSACTION_FORMAT,
            "segment_id": segment_name,
            "canonical_sha256": _file_sha256(out_path),
            "metadata_sha256": _file_sha256(meta_out_path),
            "video_sha256": timing_provenance["video_sha256"],
            "optimized_subdir": opt_subdir,
            "optimized_sha256": optimized_sha256,
            "frame_roster_sha256": frame_roster_sha,
            "frame_times_sha256": frame_times_sha,
            INTERACTION_WINDOW_CHILDREN_FIELD: interaction_window_children,
        },
    )
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(prog="python -m gtf.extraction.export.export_canonical", description="Export canonical.npz from optimized MANO")
    parser.add_argument("--segment", type=str, help="Process a specific segment")
    parser.add_argument("--max-videos", type=int, default=0, help="Max segments (0=all)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--object", type=str, default=None, help="Filter by validated object")
    chains.add_argument(parser)
    args = parser.parse_args()

    if args.segment:
        names = [args.segment]
    else:
        names = []
        for name in sorted(os.listdir(str(RESULTS_DIR))):
            opt = os.path.join(str(RESULTS_DIR), name, "optimized_mano", "optimized.npz")
            if os.path.exists(opt):
                names.append(name)

    if args.object:
        names = [n for n in names if is_action_valid(n, args.object)]

    if args.max_videos > 0:
        names = names[: args.max_videos]

    succeeded = 0
    skipped = 0
    failed = 0

    for i, name in enumerate(names):
        out = os.path.join(str(RESULTS_DIR), name, "canonical", "canonical.npz")
        if not args.force and os.path.exists(out):
            skipped += 1
            continue

        print(f"\n[{i + 1}/{len(names)}] {name}")
        try:
            # the stage-2 trajectory when present, as in the orchestrator and the quality_gate CLI
            sub = "optimized_stage2" if os.path.exists(os.path.join(str(RESULTS_DIR), name, "optimized_stage2", "optimized.npz")) else "optimized_mano"
            ok = process_one_segment(name, opt_subdir=sub, chain=chains.claim(os.path.join(str(RESULTS_DIR), name), args.chain))
            if ok:
                succeeded += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\nDone: {succeeded} exported, {skipped} skipped, {failed} failed (total {len(names)})")


if __name__ == "__main__":
    main()
