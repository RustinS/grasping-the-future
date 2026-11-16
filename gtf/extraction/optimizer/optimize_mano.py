#!/usr/bin/env python3
"""MANO-parameterized hand-object pose optimization.

Optimizes full MANO articulation (51 DOF/hand) + object 6-DOF pose,
enabling per-finger adjustments to resolve penetration.

Inputs: the hand stage's MANO parameters (WiLoR, in `hands/`), SAM 3D meshes, the object-pose chain and MoGe depth.

Usage:
    uv run python -m gtf.extraction.optimizer.optimize_mano --segment SEGMENT_NAME
    uv run python -m gtf.extraction.optimizer.optimize_mano --segment=SEGMENT_NAME --chain upgraded   # "=" form for names starting with '-'
"""

import argparse
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import json
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

from gtf.utils import (  # noqa: E402
    load_sam3d_metadata,
    load_mano_faces,
    load_object_mesh,
    quaternion_to_matrix,
    GLB_TO_RAW,
    compute_scale_z,
    smooth_scales,
    opencv_to_opengl,
    render_overlay,
    render_3d_view,
    letterbox,
    is_action_valid,
    reject_action,
    RESULTS_DIR,
    SEGMENTS_DIR,
    PANEL_W,
    PANEL_H,
    HAND_COLOR,
    HAND_COLOR_LEFT,
)
from gtf.utils.npz import savez_compressed_level  # noqa: E402
from gtf.utils.cli_settings import Setting, add_settings, apply_settings  # noqa: E402

from gtf.extraction import chains  # noqa: E402
from gtf.extraction.optimizer import hand_evidence  # noqa: E402
from gtf.extraction.optimizer.contact_depth import local_follow_depth  # noqa: E402

from gtf.extraction.optimizer.optimizer_common import OptimizerGeometryError, OptimizerTerminalReject, _build_mask_target, _load_ba_camera_poses  # noqa: E402
from gtf.extraction.optimizer.optimizer_common import MANOForward, load_hand_mano  # noqa: E402, F401  (MANOForward: used by the hand stage through this module)
from gtf.extraction.optimizer.scale_anchors import _ANCHOR_MAX_CONTACT, _ANCHOR_MIN_FRAMES, _ANCHOR_MIN_HANDFRAC, _FOLLOW_LOCAL_CLIP, _FOLLOW_LOCAL_GAP  # noqa: E402
from gtf.extraction.optimizer.scale_anchors import _hand_depth_anchor, _metric_scale_ratio, _resolve_anchor, _segment_depth_loader  # noqa: E402
from gtf.extraction.optimizer.optimizer_schedule import optimize  # noqa: E402

MAX_OPTIMIZER_CAMERA_ABS_M = 20.0
SIZE_SCENE_RANGE = (12.0, 18.0)  # hand_depth_source="size_scene": median hand extent (% of the image width) over which "size" takes over


def load_segment(
    segment_name,
    scale_anchor="moge",
    force_sam3dc=False,
    hand_depth_source="moge",
    size_fallback="moge",
    hand_mask_veto=False,
    image_contact_gate=False,
    image_carry_gate=False,
    follow_depth="centroid",
    contact_evidence=False,
):
    """Load per-keyframe hand MANO + object data calibrated to MoGe space.

    Returns (frames, obj_faces, canonical_verts_raw, anchor_frame_index, decouple_depth, pose_source).

    Object pose: the per-keyframe rotations come from the bundle-adjusted object track (object_pose/), anchored at the
    SAM 3D anchor keyframe; the translations are SAM 3D's per-keyframe fits (they follow the mask centroid). When the BA
    chain is missing or rejected and a complete sam3d_c/ (temporal SAM 3D) exists, the anchor mesh and the per-keyframe
    poses are both loaded from sam3d_c as one unit; pose_source names the SAM 3D folder used ("sam3d" | "sam3d_c").
    force_sam3dc uses that unit even when the BA passed (the retry after the primary geometry fails a safety check).

    The object mesh is scaled about the camera (silhouette kept) by one clip-level factor from the anchor (`scale_anchor`).

    Chain options (chains.py; every default is the production behavior, the upgraded chain sets the value in brackets):

    hand_depth_source ["size_scene"]: "moge" = MoGe depth at the hand. "size" = the depth at which the canonical-size MANO
    has the apparent size WiLoR measured (scale_z = f_moge / f_wilor), so pixels and apparent size both match the video.
    "size_scene" = "size" for the clip's scale, with the scene (MoGe) depth per frame where the hand is small in the image,
    where WiLoR's own depth is unstable (people standing back): per hand, a log-space blend of
    "size" and k x the "moge" scale, k = the clip median of size / moge; the weight of "size" goes from 0 to 1 as the
    median hand extent goes from SIZE_SCENE_RANGE[0] to [1] (% of the image width), so large hands keep "size".

    scale_anchor ["contact_follow"] + follow_depth ["local"]: "moge" scales the object to the MoGe metric depth at its
    mask (_metric_scale_ratio). "contact_follow" ties the object's depth to the hand's (the object follows the hand, the
    hand is not moved), falling back to "moge" on a narrow FoV or with too few contact keyframes. follow_depth: which
    object depth is paired with the hand's. "centroid" pairs the hand with the object's center, which is wrong for a
    long object held at one end (the object shrinks and ends up off the hand). "local" pairs every hand the image shows on
    the object (WiLoR hand within 5% of its size of the SAM 3 mask) with the object's depth where that hand is
    (contact_depth.local_follow_depth: front surface under a hand seen in front, back surface in front of a hidden hand
    [needs hand_masks/masks.npz], else the middle); with fewer than 3 such keyframes the centroid pairs are used.

    size_fallback ["hands"]: with a size-based hand depth and too few contact keyframes, "moge" falls back to the MoGe
    object scale and the optimizer then drags the hands to that depth, undoing the size-based depth; "hands" scales the
    object to the median hand / object depth ratio over every keyframe with a hand and skips the drag.

    hand_mask_veto [on]: drop WiLoR detections with no RGB hand under them (hand_evidence.mask_veto; masks from
    run_hand_masks.py). image_contact_gate [on]: record the image gap between WiLoR's hand and the object mask per
    hand-keyframe ("img_gap") for the optimizer's contact gate. image_carry_gate [on]: record the hand's and the object
    mask's image centroids ("img_hand_c", "obj_img_c") for its carry gate. contact_evidence [on, with contact_depth="ray"]:
    each frame also carries the video's masks ("obj_mask", "hand_mask") for the visible depth order used by
    MANOPoseOptimizer._align_contact_depth.
    """
    segment_dir = os.path.join(RESULTS_DIR, segment_name)
    if scale_anchor not in ("moge", "contact_follow"):
        raise ValueError(f"unknown scale_anchor {scale_anchor!r}")

    # BA availability decides the per-keyframe pose source, so check it before loading the SAM 3D metadata: BA rejected
    # or missing + complete sam3d_c/ -> the temporal fallback (mesh and poses from sam3d_c as one unit; a sam3d mesh with
    # sam3d_c poses would break the vertex correspondence of everything downstream of canonical_verts).
    c_dir = os.path.join(segment_dir, "sam3d_c")
    c_complete = os.path.exists(os.path.join(c_dir, "metadata.json")) and os.path.exists(os.path.join(c_dir, ".sam3d_c_done"))
    if force_sam3dc:
        if not c_complete:
            raise FileNotFoundError(f"No complete sam3d_c geometry fallback for {segment_name}")
        ba = None
        pose_subdir = "sam3d_c"
        print("  Object pose chain: forced sam3d_c optimizer-geometry fallback")
    else:
        ba = _load_ba_camera_poses(segment_dir)
        pose_subdir = "sam3d"
        if ba is None and c_complete:
            pose_subdir = "sam3d_c"
            print("  Object pose chain: BA unavailable/rejected -> sam3d_c temporal fallback")

    sam3d_meta = load_sam3d_metadata(segment_dir, subdir=pose_subdir)
    if sam3d_meta is None:
        raise FileNotFoundError(f"No SAM3D metadata for {segment_name}")

    keyframe_indices = sam3d_meta["keyframe_indices"]
    anchor_frame_index = sam3d_meta.get("anchor_frame_index")
    frame_entries = {e["frame_index"]: e for e in sam3d_meta["frames"]}
    masks = None
    masks_path = os.path.join(segment_dir, "SAM3", "masks.npz")
    if os.path.exists(masks_path):
        masks = np.load(masks_path)["masks"]

    hand_params = {side: data for side in ("left", "right") if (data := load_hand_mano(segment_dir, side)) is not None}
    if not hand_params:
        raise ValueError("No MANO parameters found — re-run the hand stage (WiLoR)")
    hm = None
    if hand_mask_veto or contact_evidence or follow_depth == "local":
        from gtf.extraction.hands.run_hand_masks import load_hand_masks

        hm = load_hand_masks(segment_dir)
    if hand_mask_veto:
        if hm is None:
            print("  hand-mask veto: no hand_masks/masks.npz, skipped")
        else:
            vet = hand_evidence.mask_veto(hand_params, hm)
            print("  hand-mask veto: " + (", ".join(f"{sd} {len(v)} frames" for sd, v in vet.items()) or "SAM 3 sees too few hands in this clip, nothing vetoed"))

    # Cached per-frame MoGe depth, shared by the hand calibration and the object path. The cache is local to this load
    # and released before optimization starts, which bounds host memory to one segment per optimizer process.
    raw_moge_depth = _segment_depth_loader(segment_dir)

    size_like = hand_depth_source in ("size", "size_scene")
    smoothed_scales = {}
    for side, hd in hand_params.items():
        raw_scales = {}
        scene_scales, hand_extent = {}, []  # size_scene only
        for fi in keyframe_indices:
            if fi not in hd["frame_to_idx"] or fi not in hd["valid_frames"]:
                raw_scales[fi] = None
                continue
            depth_map, K_m = raw_moge_depth(fi)
            if depth_map is None:
                raw_scales[fi] = None
                continue
            ai = hd["frame_to_idx"][fi]
            if size_like:
                fx_hn = float(hd["intr"][ai][0, 0]) * depth_map.shape[1] / float(hd["img_size"][ai][0])
                raw_scales[fi] = float(K_m[0, 0]) / fx_hn if fx_hn > 1e-6 else None
                if hand_depth_source == "size_scene":
                    scene_scales[fi] = compute_scale_z(hd["joints"][ai], depth_map, hd["intr"][ai], hd["img_size"][ai], K_m)
                    hand_extent.append(hand_evidence.image_extent_pct(hd["verts"][ai], hd["intr"][ai], hd["img_size"][ai]))
            elif hand_depth_source == "moge":
                raw_scales[fi] = compute_scale_z(hd["joints"][ai], depth_map, hd["intr"][ai], hd["img_size"][ai], K_m)
            else:
                raise ValueError(f"unknown hand_depth_source {hand_depth_source!r}")
        if hand_depth_source == "size_scene":
            raw_scales, ss_info = hand_evidence.size_scene_scales(raw_scales, scene_scales, hand_extent, SIZE_SCENE_RANGE)
            print(f"  size_scene hand depth ({side}): {ss_info}")
        smoothed_scales[side] = smooth_scales(raw_scales)

    mesh_frame = anchor_frame_index if anchor_frame_index is not None else keyframe_indices[0]
    anchor_mesh = load_object_mesh(segment_dir, mesh_frame, subdir=pose_subdir)
    if anchor_mesh is None and pose_subdir == "sam3d_c":
        if force_sam3dc:
            raise FileNotFoundError(f"Forced sam3d_c fallback has no anchor mesh for frame {mesh_frame}")
        # No anchor mesh in sam3d_c -> the whole fallback reverts to sam3d (poses and mesh must come from one
        # reconstruction: independent diffusion samples have different canonical orientations).
        print("  sam3d_c fallback reverted: no anchor mesh in sam3d_c/ — using raw sam3d poses")
        pose_subdir = "sam3d"
        sam3d_meta = load_sam3d_metadata(segment_dir)
        keyframe_indices = sam3d_meta["keyframe_indices"]
        anchor_frame_index = sam3d_meta.get("anchor_frame_index")
        frame_entries = {e["frame_index"]: e for e in sam3d_meta["frames"]}
        mesh_frame = anchor_frame_index if anchor_frame_index is not None else keyframe_indices[0]
        anchor_mesh = load_object_mesh(segment_dir, mesh_frame)
    if anchor_mesh is None and anchor_frame_index is not None:
        mesh_frame = keyframe_indices[0]
        anchor_mesh = load_object_mesh(segment_dir, mesh_frame, subdir=pose_subdir)
    if anchor_mesh is None:
        raise FileNotFoundError(f"No object mesh found (tried frame {mesh_frame})")

    canonical_verts_raw = np.array(anchor_mesh.vertices, dtype=np.float64) @ GLB_TO_RAW
    shared_obj_faces = np.array(anchor_mesh.faces, dtype=np.int64)
    vc = getattr(anchor_mesh.visual, "vertex_colors", None)
    anchor_colors = np.array(vc[:, :4], dtype=np.float32) / 255.0 if vc is not None else np.full((len(canonical_verts_raw), 4), 0.7, dtype=np.float32)

    # Rebuild the per-keyframe rotation chain from the BA texture-track rotations, anchored at the SAM 3D anchor keyframe
    # (the most reliable mask / pose). The SAM 3D per-keyframe rotations are independent single-frame estimates that roll
    # in the hand; the BA track carries the temporal rotation evidence. The SAM 3D centroid path is kept.
    R_ba_cam = ba[0] if ba is not None else None
    ba_M_anchor = None
    F_flip = np.diag([-1.0, -1.0, 1.0])
    if R_ba_cam is not None:
        anchor_entry = frame_entries.get(mesh_frame)
        if anchor_entry is not None and anchor_entry.get("status") == "ok" and anchor_entry.get("pose") is not None and mesh_frame < len(R_ba_cam):
            ap = anchor_entry["pose"]
            R_anchor = quaternion_to_matrix(np.array(ap["rotation"][0], dtype=np.float64))
            ba_M_anchor = F_flip @ R_anchor.T  # canonical -> camera at anchor (column convention)
            ba_R_anchor_cam_T = R_ba_cam[mesh_frame].T
            canonical_centroid = canonical_verts_raw.mean(axis=0)
            print(f"  Object pose chain: BA texture-track (anchor frame {mesh_frame}, rotation)")
    if ba_M_anchor is None:
        print(f"  Object pose chain: {pose_subdir} per-keyframe (BA unavailable/rejected)")

    frames = []
    moge_ratios = []
    contact_pairs = []  # (z_obj_centroid, z_hand) per contact keyframe (contact_follow)
    follow_all = []  # (z_obj_centroid, mean calibrated hand depth) per keyframe with a hand — size_fallback="hands"
    local_pairs = []  # follow_depth="local": (object depth where the hand is in the image, that hand's depth) per touching hand-keyframe
    local_kinds = {}  # how each pair was made: object front (hand seen in front of it) / back (hand hidden behind it) / middle
    if follow_depth not in ("centroid", "local"):
        raise ValueError(f"unknown follow_depth {follow_depth!r}")
    if size_fallback not in ("moge", "hands"):
        raise ValueError(f"unknown size_fallback {size_fallback!r}")
    cam_fov_deg = None  # camera horizontal FoV (gates the contact anchor)

    for frame_idx in keyframe_indices:
        depth_map, K_m = raw_moge_depth(frame_idx)

        frame_entry = frame_entries.get(frame_idx)
        if frame_entry is None or frame_entry.get("status") != "ok" or frame_entry.get("pose") is None:
            continue
        pose_data = frame_entry["pose"]

        R = quaternion_to_matrix(np.array(pose_data["rotation"][0], dtype=np.float64))
        t_vec = np.array(pose_data["translation"][0], dtype=np.float64)
        s_vec = np.array(pose_data["scale"][0], dtype=np.float64)

        if ba_M_anchor is not None:
            bi = min(frame_idx, len(R_ba_cam) - 1)
            Q = R_ba_cam[bi] @ ba_R_anchor_cam_T  # camera-frame relative rotation anchor -> t
            R_new = (Q @ ba_M_anchor).T @ F_flip  # back to SAM3D row-vector convention
            # Keep the SAM 3D camera-space centroid path under the BA rotation
            t_vec = t_vec + (canonical_centroid * s_vec) @ R - (canonical_centroid * s_vec) @ R_new
            R = R_new

        obj_verts_cv = (((canonical_verts_raw * s_vec) @ R + t_vec) * np.array([-1, -1, 1], dtype=np.float64)).astype(np.float32)

        if depth_map is None:
            continue

        hand_data = {}
        for side, hd in hand_params.items():
            if frame_idx not in hd["frame_to_idx"] or frame_idx not in hd["valid_frames"]:
                continue
            ai = hd["frame_to_idx"][frame_idx]
            s_scale = smoothed_scales.get(side, {}).get(frame_idx)
            if s_scale is None:
                continue
            hand_data[side] = {
                "hand_pose": hd["hand_pose"][ai],
                "global_orient": hd["global_orient"][ai],
                "betas": hd["betas"][ai],
                "cam_t": hd["cam_t"][ai],
                "intr": hd["intr"][ai],
                "img_size": hd["img_size"][ai],
                "scale_z": s_scale,
                "depth_shape": depth_map.shape,
                "K_moge": K_m,
            }
            if image_contact_gate:
                mk_ = masks[frame_idx] if masks is not None and frame_idx < len(masks) else None
                hand_data[side]["img_gap"] = hand_evidence.image_gap(hd["verts"][ai], hd["intr"][ai], hd["img_size"][ai], mk_)
            if image_carry_gate:
                mk_ = masks[frame_idx] if masks is not None and frame_idx < len(masks) else None
                if mk_ is not None:
                    u_, ok_ = hand_evidence.project_hand(hd["verts"][ai], hd["intr"][ai], hd["img_size"][ai], mk_.shape)
                    hand_data[side]["img_hand_c"] = None if u_ is None or int(ok_.sum()) < 50 else (u_[ok_].mean(0) / mk_.shape[1]).tolist()

        mk = masks[frame_idx] if masks is not None and frame_idx < len(masks) else None
        if cam_fov_deg is None and K_m is not None:
            fx = float(K_m[0, 0])
            if fx > 1.0:
                cam_fov_deg = float(np.degrees(2.0 * np.arctan(depth_map.shape[1] / (2.0 * fx))))
        # contact_follow also collects the MoGe ratios: it falls back to the MoGe anchor on a narrow FoV (where MoGe is
        # reliable and following the hand would transfer the hand's own depth error).
        r = _metric_scale_ratio(obj_verts_cv, depth_map, K_m, mk)
        if r is not None:
            moge_ratios.append(r)
        if scale_anchor == "contact_follow":
            # One-directional decoupling: the object's single scale / depth degree of freedom is tied to the hand's
            # calibrated placement depth z_calib = mean(joint_z) * scale_z (where the hand already sits). The object is
            # scaled about the camera to land at the hand's depth, so the object follows the hand and the hand is never
            # dragged (the optimizer skips _correct_depth_offset, decouple_depth). On a wide FoV the MoGe object depth is
            # more biased than the hand's; a narrow FoV falls back to the MoGe anchor.
            z_obj = float(np.mean(obj_verts_cv[:, 2]))
            if size_like and size_fallback == "hands" and z_obj > 1e-3:
                zs = []
                for _side, hd in hand_params.items():
                    s_scale = smoothed_scales.get(_side, {}).get(frame_idx)
                    if frame_idx not in hd["frame_to_idx"] or frame_idx not in hd["valid_frames"] or s_scale is None:
                        continue
                    jz = np.asarray(hd["joints"][hd["frame_to_idx"][frame_idx]], dtype=np.float64)[:, 2]
                    jz = jz[jz > 1e-6]
                    if len(jz) >= 8:
                        zs.append(float(np.mean(jz)) * float(s_scale))
                if zs:
                    follow_all.append((z_obj, float(np.mean(zs))))
            if follow_depth == "local":
                # image-evidenced pairs: (object depth where the hand is, that hand's depth) per hand on the object
                for _side, hd in hand_params.items():
                    s_scale = smoothed_scales.get(_side, {}).get(frame_idx)
                    if frame_idx not in hd["frame_to_idx"] or frame_idx not in hd["valid_frames"] or s_scale is None or K_m is None:
                        continue
                    ai = hd["frame_to_idx"][frame_idx]
                    g = hand_evidence.image_gap(hd["verts"][ai], hd["intr"][ai], hd["img_size"][ai], mk)
                    j = np.asarray(hd["joints"][ai], dtype=np.float64)
                    j = j[j[:, 2] > 1e-6]
                    if g is None or g > _FOLLOW_LOCAL_GAP or len(j) < 8:
                        continue
                    K_h, (W_h, H_h) = hd["intr"][ai], (float(hd["img_size"][ai][0]), float(hd["img_size"][ai][1]))
                    H_m, W_m = depth_map.shape[:2]
                    to_m = np.array([W_m / W_h, H_m / H_h])
                    uv = np.stack([K_h[0, 0] * j[:, 0] / j[:, 2] + K_h[0, 2], K_h[1, 1] * j[:, 1] / j[:, 2] + K_h[1, 2]], 1) * to_m
                    hv_ = np.asarray(hd["verts"][ai], dtype=np.float64)
                    hv_ = hv_[hv_[:, 2] > 1e-6]
                    px = np.stack([K_h[0, 0] * hv_[:, 0] / hv_[:, 2] + K_h[0, 2], K_h[1, 1] * hv_[:, 1] / hv_[:, 2] + K_h[1, 2]], 1) * to_m
                    z_loc, off, kind = local_follow_depth(obj_verts_cv, shared_obj_faces, K_m, (H_m, W_m), px, uv, mk, hm.get(int(frame_idx)) if hm else None)
                    z_hand = float(np.mean(j[:, 2])) * float(s_scale) + off
                    if z_loc is not None and z_loc > 1e-3 and z_hand > 1e-3:
                        local_pairs.append((z_loc, z_hand))
                        local_kinds[kind] = local_kinds.get(kind, 0) + 1
            best = None  # (1-contact, z_hand_calib)
            for _side, hd in hand_params.items():
                if frame_idx not in hd["frame_to_idx"] or frame_idx not in hd["valid_frames"]:
                    continue
                s_scale = smoothed_scales.get(_side, {}).get(frame_idx)
                if s_scale is None:
                    continue
                ai = hd["frame_to_idx"][frame_idx]
                res = _hand_depth_anchor(hd["joints"][ai], hd["intr"][ai], hd["img_size"][ai], depth_map, K_m, mk)
                if res is None:
                    continue
                _s_kf, hfrac, contact, _z_impl = res
                if hfrac < _ANCHOR_MIN_HANDFRAC or contact >= _ANCHOR_MAX_CONTACT:
                    continue
                jz = np.asarray(hd["joints"][ai], dtype=np.float64)[:, 2]
                jz = jz[jz > 1e-6]
                if len(jz) < 8:
                    continue
                z_hand_calib = float(np.mean(jz)) * float(s_scale)
                if 1e-3 < z_hand_calib and (best is None or (1.0 - contact) > best[0]):
                    best = (1.0 - contact, z_hand_calib)
            if best is not None and z_obj > 1e-3:
                contact_pairs.append((z_obj, best[1]))

        frames.append(
            {
                "frame_index": frame_idx,
                "hands": hand_data,
                "obj_verts": obj_verts_cv,
                "obj_faces": shared_obj_faces,
                "obj_colors": anchor_colors,
                "obj_R_sam3d": R.astype(np.float32),
                "obj_t_sam3d": t_vec.astype(np.float32),
                "obj_s_sam3d": s_vec.astype(np.float32),
                "K_moge": K_m,
                "depth_shape": depth_map.shape,
                "obj_mask_target": _build_mask_target(masks[frame_idx] if masks is not None and frame_idx < len(masks) else None),
                "obj_img_c": (
                    (np.argwhere(masks[frame_idx])[:, ::-1].mean(0) / masks[frame_idx].shape[1]).tolist()
                    if image_carry_gate and masks is not None and frame_idx < len(masks) and int(np.count_nonzero(masks[frame_idx])) >= 30
                    else None
                ),
            }
        )
        if contact_evidence:
            frames[-1]["obj_mask"] = masks[frame_idx] if masks is not None and frame_idx < len(masks) else None
            frames[-1]["hand_mask"] = hm.get(int(frame_idx)) if hm else None

    if not frames:
        raise ValueError(f"No valid object keyframes with depth for {segment_name}")
    if not any(f["hands"] for f in frames):
        raise OptimizerTerminalReject(f"optimizer_no_hand_object_overlap: no valid hand observations overlap the object keyframes for {segment_name}")

    # Object-scale correction: scale the placed object about the camera origin by the anchor's factor (silhouette kept,
    # see _metric_scale_ratio). The (verts, scale, translation) triple is scaled together so the optimizer's data target
    # and its canonical x scale @ R + t rebuild stay consistent.
    # decouple_depth is True only when the object was placed at the hand's depth (the contact_follow anchor fired, or
    # size_fallback="hands"): the optimizer then skips _correct_depth_offset (the hand <- object drag). On the MoGe
    # fallback the object is at the MoGe depth and the drag still runs (contact_follow then behaves like "moge").
    # With a size-based hand depth the hand depth is not MoGe's, so the narrow-FoV gate (which keeps the MoGe hand depth
    # error out of the object) does not apply: the object always follows the hand.
    use_local = follow_depth == "local" and scale_anchor == "contact_follow" and len(local_pairs) >= _ANCHOR_MIN_FRAMES
    # local pairs: the object may follow the hand as far as size_fallback="hands" lets it (0.2-3x), not only 0.5-1.5x: the
    # narrower bound stops the object short of the hand for small hands (whose size depth is too far) and close-ups
    c, info = _resolve_anchor(
        scale_anchor,
        moge_ratios,
        local_pairs if use_local else contact_pairs,
        cam_fov_deg=None if size_like else cam_fov_deg,
        contact_clip=_FOLLOW_LOCAL_CLIP if use_local else None,
    )
    if use_local and info:
        info = f"{info}; follow_depth=local (object depth at the hand, {len(local_pairs)} touching hand-keyframes: {', '.join(f'{k} {v}' for k, v in sorted(local_kinds.items()))}; centroid pairs {len(contact_pairs)})"
    elif follow_depth == "local" and scale_anchor == "contact_follow" and info:
        info = f"{info}; follow_depth=local had {len(local_pairs)} touching hand-keyframes (< {_ANCHOR_MIN_FRAMES}): centroid pairs used"
    hands_follow = False
    if size_like and size_fallback == "hands" and scale_anchor == "contact_follow" and info and "FALLBACK" in info and len(follow_all) >= 3:
        ratios = np.array([zh / zo for zo, zh in follow_all], dtype=np.float64)
        raw = float(np.median(ratios))
        c = float(np.clip(raw, 0.2, 3.0))
        info = f"{info}; size_fallback=hands: c={c:.3f} (median hand/object depth {raw:.3f} over {len(follow_all)} keyframes)"
        hands_follow = True
    decouple_depth = False
    if c is not None:
        if abs(c - 1.0) > 0.02:
            for f in frames:
                f["obj_verts"] = (f["obj_verts"] * c).astype(np.float32)
                f["obj_t_sam3d"] = (f["obj_t_sam3d"] * c).astype(np.float32)
                f["obj_s_sam3d"] = (f["obj_s_sam3d"] * c).astype(np.float32)
        decouple_depth = (scale_anchor == "contact_follow" and "FALLBACK" not in info) or hands_follow
        print(f"  Object-scale correction [{scale_anchor}]: {info}")

    anchor_fi = anchor_frame_index if anchor_frame_index is not None else keyframe_indices[0]
    return frames, shared_obj_faces, canonical_verts_raw, anchor_fi, decouple_depth, pose_subdir


def render_panel_video(segment_name, frames, results):
    """Render the 2x2 panel video: frame, overlay, 3D front view, 3D top view."""
    segment_dir = os.path.join(RESULTS_DIR, segment_name)
    video_path = os.path.join(segment_dir, "video.mp4")
    out_dir = os.path.join(segment_dir, "optimized_mano")
    out_path = os.path.join(out_dir, "vis_panel.mp4")

    if not os.path.exists(video_path):
        print("  No video.mp4")
        return None

    cap = cv2.VideoCapture(video_path)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    faces_right, faces_left = load_mano_faces()
    hf = {"right": faces_right, "left": faces_left}

    obj_opt = results["obj_verts_optimized"]
    active = list(results["hands"].keys())
    panels = []
    n = len(frames)

    print(f"\n  Rendering panel video ({n} frames)...")

    for i, fd in enumerate(frames):
        fi = fd["frame_index"]
        if fi >= total:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, bgr = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        K = fd.get("K_moge")
        if K is not None:
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            W_r, H_r = fd["depth_shape"][1], fd["depth_shape"][0]
        else:
            fx = fy = 1100.0
            cx, cy = vid_w / 2.0, vid_h / 2.0
            W_r, H_r = vid_w, vid_h

        hand_gl = []
        for side in active:
            hdata = results["hands"][side]
            if not hdata["valid_mask"][i]:
                continue
            v_gl = opencv_to_opengl(hdata["verts_optimized"][i])
            color = np.array([*(HAND_COLOR if side == "right" else HAND_COLOR_LEFT), 1.0], dtype=np.float32)
            hand_gl.append((v_gl, hf[side], color))

        obj_gl = (opencv_to_opengl(obj_opt[i]), fd["obj_faces"], fd["obj_colors"])
        all_gl = hand_gl + [obj_gl]

        p_tl = letterbox(rgb, PANEL_W, PANEL_H)
        overlay = render_overlay(rgb, hand_gl, obj_gl, fx, fy, cx, cy, W_r, H_r)
        p_tr = letterbox(overlay, PANEL_W, PANEL_H)

        av = np.concatenate([m[0] for m in all_gl if len(m[0]) > 0])
        c = av.mean(0)
        e = np.abs(av - c).max() * 1.5 + 0.01

        cam_front = np.eye(4, dtype=np.float32)
        cam_front[:3, 3] = [c[0], c[1], c[2] + e * 2]
        p_bl = render_3d_view(all_gl, cam_front, PANEL_W, PANEL_H)

        cam_top = np.eye(4, dtype=np.float32)
        cam_top[:3, :3] = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
        cam_top[:3, 3] = [c[0], c[1] + e * 2, c[2]]
        p_br = render_3d_view(all_gl, cam_top, PANEL_W, PANEL_H)

        panels.append(np.concatenate([np.concatenate([p_tl, p_tr], axis=1), np.concatenate([p_bl, p_br], axis=1)], axis=0))

        if (i + 1) % 20 == 0 or i == n - 1:
            print(f"    [{i + 1}/{n}]")

    cap.release()

    if not panels:
        return None

    import av

    os.makedirs(out_dir, exist_ok=True)
    container = av.open(out_path, mode="w")
    stream = container.add_stream("libx264", rate=10)
    stream.width = PANEL_W * 2
    stream.height = PANEL_H * 2
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "23", "preset": "fast", "movflags": "+faststart"}
    for p in panels:
        for pkt in stream.encode(av.VideoFrame.from_ndarray(p, format="rgb24")):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()
    print(f"  Video: {out_path} ({len(panels)} frames)")
    return out_path


def process_one_segment(segment_name, config, device="cuda"):
    """Optimize one segment and write optimized_mano/optimized.npz; on a geometry failure of the primary object pose
    chain, retry once with the sam3d_c fallback. Raises OptimizerTerminalReject for segments that cannot yield a window."""
    print(f"Segment: {segment_name}")

    def _load_optimizer_inputs(force_sam3dc=False):
        loaded = load_segment(
            segment_name,
            scale_anchor=config.get("scale_anchor", "moge"),
            force_sam3dc=force_sam3dc,
            hand_depth_source=config.get("hand_depth_source", "moge"),
            size_fallback=config.get("size_fallback", "moge"),
            hand_mask_veto=config.get("hand_mask_veto", False),
            image_contact_gate=config.get("image_contact_gate", False) or config.get("contact_depth", "off") != "off",
            image_carry_gate=config.get("image_carry_gate", False),
            follow_depth=config.get("follow_depth", "centroid"),
            contact_evidence=config.get("contact_depth", "off") != "off",
        )
        if len(loaded[0]) < 2:
            raise OptimizerTerminalReject(f"optimizer_insufficient_keyframes: {len(loaded[0])} valid frame(s)")
        print(f"Loaded {len(loaded[0])} valid keyframes")
        return loaded

    (
        frames,
        obj_faces,
        canonical_verts_raw,
        anchor_frame_index,
        decouple_depth,
        pose_source,
    ) = _load_optimizer_inputs()

    run_config = {**config, "decouple_depth": decouple_depth}
    try:
        results = optimize(
            frames,
            obj_faces,
            canonical_verts_raw,
            run_config,
            anchor_frame_index=anchor_frame_index,
            device=device,
        )
    except OptimizerGeometryError as primary_error:
        if pose_source != "sam3d":
            raise OptimizerTerminalReject(f"optimizer_unsafe_geometry_after_fallback: {primary_error}") from primary_error
        print(f"  Primary optimizer geometry rejected: {primary_error}; retrying the complete sam3d_c fallback")
        # The failed constructor may have allocated CUDA tensors before the
        # physical check. Clear traceback locals and cached blocks before retry.
        traceback.clear_frames(primary_error.__traceback__)
        import gc

        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        try:
            (
                frames,
                obj_faces,
                canonical_verts_raw,
                anchor_frame_index,
                decouple_depth,
                pose_source,
            ) = _load_optimizer_inputs(force_sam3dc=True)
        except FileNotFoundError as fallback_error:
            raise OptimizerTerminalReject(f"optimizer_unsafe_geometry_no_fallback: {primary_error}") from fallback_error
        run_config = {**config, "decouple_depth": decouple_depth}
        try:
            results = optimize(
                frames,
                obj_faces,
                canonical_verts_raw,
                run_config,
                anchor_frame_index=anchor_frame_index,
                device=device,
            )
        except OptimizerGeometryError as fallback_error:
            raise OptimizerTerminalReject(f"optimizer_unsafe_geometry_after_fallback: {fallback_error}") from fallback_error

    numeric_results = {
        "obj_verts_optimized": results["obj_verts_optimized"],
        "canonical_verts": results["canonical_verts"],
        "obj_scale_init": results["obj_scale_init"],
    }
    for side, hand in results["hands"].items():
        numeric_results[f"hand_{side}_verts"] = hand["verts_optimized"]
    for side, hand in results["hands_parametric"].items():
        numeric_results[f"hand_{side}_pose"] = hand["hand_pose"]
        numeric_results[f"hand_{side}_betas"] = hand["betas"]
    bad_results = [name for name, value in numeric_results.items() if not np.isfinite(np.asarray(value)).all()]
    bad_losses = [name for name, value in results["optimization_losses"].items() if not np.isfinite(value)]
    if bad_results or bad_losses:
        raise FloatingPointError(f"refusing to save non-finite optimizer output: arrays={bad_results}, losses={bad_losses}")

    # Only camera-space geometry is bounded in meters. Canonical coordinates,
    # MANO pose coefficients, betas, and scale parameters are different units.
    camera_results = {
        "obj_verts_optimized": results["obj_verts_optimized"],
        **{f"hand_{side}_verts": hand["verts_optimized"] for side, hand in results["hands"].items()},
    }
    implausible_results = [
        name for name, value in camera_results.items() if np.asarray(value).size and float(np.abs(np.asarray(value, dtype=np.float64)).max()) > MAX_OPTIMIZER_CAMERA_ABS_M
    ]
    if implausible_results:
        raise OptimizerTerminalReject(f"optimizer_implausible_camera_geometry: refusing to save arrays outside +/-{MAX_OPTIMIZER_CAMERA_ABS_M:g}m: {implausible_results}")

    out_dir = os.path.join(RESULTS_DIR, segment_name, "optimized_mano")
    os.makedirs(out_dir, exist_ok=True)
    T = len(results["frame_indices"])
    save_dict = {
        "frame_indices": np.array(results["frame_indices"]),
        "obj_verts": results["obj_verts_optimized"],
    }
    for side, hd in results["hands"].items():
        save_dict[f"hand_{side}_verts"] = hd["verts_optimized"]
        save_dict[f"hand_{side}_valid"] = hd["valid_mask"]

    # Parametric state: write both sides, zero-fill absent sides for uniform schema
    for side in ("left", "right"):
        if side in results["hands_parametric"]:
            hp = results["hands_parametric"][side]
            save_dict[f"hand_{side}_pose"] = hp["hand_pose"]
            save_dict[f"hand_{side}_betas"] = hp["betas"]
        else:
            save_dict[f"hand_{side}_pose"] = np.zeros((T, 45), dtype=np.float32)
            save_dict[f"hand_{side}_betas"] = np.zeros(10, dtype=np.float32)
            if f"hand_{side}_verts" not in save_dict:
                save_dict[f"hand_{side}_verts"] = np.zeros((T, 778, 3), dtype=np.float32)
                save_dict[f"hand_{side}_valid"] = np.zeros(T, dtype=bool)
    save_dict["canonical_verts"] = results["canonical_verts"]
    save_dict["obj_faces"] = results["obj_faces"]
    save_dict["obj_scale_init"] = results["obj_scale_init"]
    save_dict["anchor_frame_index"] = np.int64(results["anchor_frame_index"])
    save_dict["hand_scale_mode"] = np.array(results["hand_scale_mode"])
    save_dict["pen_sign_mode"] = np.array(results["pen_sign_mode"])
    # which SAM 3D folder supplied the anchor mesh + per-keyframe poses ("sam3d" | "sam3d_c"):
    # the quality gate reads its mesh from the same folder (vertex correspondence)
    save_dict["pose_source"] = np.array(pose_source)
    save_dict["optimization_losses_json"] = np.array(json.dumps(results["optimization_losses"], allow_nan=False))

    savez_compressed_level(
        os.path.join(out_dir, "optimized.npz"),
        compresslevel=6,
        allow_pickle=True,
        **save_dict,
    )

    if config.get("visualize", False):
        render_panel_video(segment_name, frames, results)
    return True


# Settings without a flag of their own; change one with --set KEY=VALUE (gtf/utils/cli_settings.py).
SETTINGS = {
    "contact_threshold": Setting(0.008, float, "Hand-object contact distance scale in meters (penetration Huber width, contact-term falloff)"),
    "min_separation": Setting(0.002, float, "Minimum hand-object separation in meters"),
    "deep_pen_threshold": Setting(0.005, float, "Penetration depth beyond which quadratic tail activates"),
    "max_wrist_rot": Setting(0.08, float, "Clamp wrist orientation deltas in radians"),
    "max_wrist_trans": Setting(0.05, float, "Clamp wrist translation deltas in meters"),
    "thumb_vertex_weight": Setting(0.25, float, "Relative collision weight for thumb tip vertices"),
    "grasp_side_margin": Setting(0.005, float, "Minimum palm-normal distance for the object center"),
    "start_frame_max": Setting(59, int, "Last frame index for start-window wrist solve"),
    "carry_conf_low": Setting(0.5, float, "Confidence threshold below which carry regularization activates"),
    "carry_conf_high": Setting(0.8, float, "Confidence threshold for trusted frames in carry"),
    "w_hold_obj_rot": Setting(None, float, "Override in-hand object rotation hold weight (0 disables)"),
    "w_data_obj_rot": Setting(None, float, "Override the weight tying the object rotation to its BA chain (default 3; the upgraded chain uses 30)"),
    "conf_floor": Setting(0.3, float, "Floor for the wrist tracker confidence in the hand data terms (0.05 = no floor)"),
    "hand_depth_source": Setting(
        None,
        str,
        "hand depth: MoGe depth at the hand, depth from its apparent size, or size_scene = size for the clip and the scene depth per frame where the hand is small in the image",
        choices=["moge", "size", "size_scene"],
    ),
    "scale_anchor": Setting(None, str, "object scale/depth anchor: MoGe metric depth, or the object follows the hand (load_segment)", choices=["moge", "contact_follow"]),
    "follow_depth": Setting(
        None, str, "contact_follow: tie the hand's depth to the object's centroid, or to the object's depth where the hand is in the image", choices=["centroid", "local"]
    ),
    "size_fallback": Setting(None, str, "size hand depth with too few contacts: MoGe object scale + hand drag, or the object follows the hands", choices=["moge", "hands"]),
    "hand_depth_correction": Setting(
        None, str, "pre-optimization hand->object depth correction: cam_t z shift, or along each frame's viewing ray (keeps pixels)", choices=["z", "ray"]
    ),
    "wrist_prior": Setting(None, str, "wrist-translation prior split: camera xy/z, or sideways/along the hand's viewing ray", choices=["camera", "ray"]),
    "contact_depth": Setting(
        None,
        str,
        "ray = before contact seeding, slide each hand along its viewing ray onto the object where the image shows contact (contact_depth.py)",
        choices=["off", "ray"],
    ),
    "hand_mask_veto": Setting(None, bool, "drop WiLoR detections with no RGB hand under them (needs hand_masks/masks.npz from run_hand_masks.py)"),
    "image_contact_gate": Setting(None, bool, "contact/grasp attraction only where the image shows the hand touching the object mask"),
    "image_carry_gate": Setting(None, bool, "hand-object hold terms only where the image shows the hand moving with the object"),
}


def main():
    parser = argparse.ArgumentParser(prog="python -m gtf.extraction.optimizer.optimize_mano", description="MANO-parameterized hand-object optimization")
    parser.add_argument("--segment", type=str, help="Process a specific segment")
    parser.add_argument("--max-videos", type=int, default=0, help="Max segments to process (0=all)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--object", type=str, default=None, help="Only process segments validated for this object (auto-detected if omitted)")
    parser.add_argument("--n-iters", type=int, default=500, help="Optimizer iterations (500 = the default phase schedule)")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Render optimized_mano/vis_panel.mp4 (disabled by default)",
    )
    # Chain options: the defaults come from --chain (chains.py; production when nothing is given); a --set of one of them overrides it.
    chains.add_argument(parser)
    parser.add_argument("--device", type=str, default="cuda")
    add_settings(parser, SETTINGS)
    args = parser.parse_args()
    apply_settings(parser, args, SETTINGS)

    config = {
        k: getattr(args, k)
        for k in (
            "n_iters",
            "contact_threshold",
            "min_separation",
            "deep_pen_threshold",
            "max_wrist_rot",
            "max_wrist_trans",
            "thumb_vertex_weight",
            "grasp_side_margin",
            "start_frame_max",
            "carry_conf_low",
            "carry_conf_high",
        )
    }
    config["conf_floor"] = args.conf_floor
    config["visualize"] = args.visualize
    chain = chains.get(args.chain)
    config.update(chain.optimizer)
    explicit = {
        "hand_depth_source": args.hand_depth_source,
        "scale_anchor": args.scale_anchor,
        "follow_depth": args.follow_depth,
        "size_fallback": args.size_fallback,
        "hand_depth_correction": args.hand_depth_correction,
        "wrist_prior_mode": args.wrist_prior,
        "contact_depth": args.contact_depth,
        "hand_mask_veto": args.hand_mask_veto,
        "image_contact_gate": args.image_contact_gate,
        "image_carry_gate": args.image_carry_gate,
        "w_hold_obj_rot": args.w_hold_obj_rot,
        "w_data_obj_rot": args.w_data_obj_rot,
    }
    config.update({k: v for k, v in explicit.items() if v is not None})

    if args.segment:
        names = [args.segment]
    else:
        with open(os.path.join(SEGMENTS_DIR, "manifest.json")) as f:
            manifest = json.load(f)
        names = [os.path.splitext(e["file"])[0] for e in manifest["succeeded"]]

    if args.max_videos > 0:
        names = names[: args.max_videos]

    n_before = len(names)
    names = [n for n in names if is_action_valid(n, args.object)]
    _label = f" ({args.object})" if args.object else ""
    print(f"Action validation filter{_label}: {len(names)}/{n_before} valid")

    valid_names = []
    for name in names:
        seg_dir = os.path.join(RESULTS_DIR, name)
        if args.force:
            Path(seg_dir, "optimized_mano", "optimized.npz").unlink(missing_ok=True)
        checks = {"hands": os.path.isdir(os.path.join(seg_dir, "hands")), "sam3d": os.path.exists(os.path.join(seg_dir, "sam3d", "metadata.json"))}
        missing = [k for k, v in checks.items() if not v]
        if missing:
            print(f"Skipping {name}: missing {', '.join(missing)}")
        elif not args.force and os.path.exists(os.path.join(seg_dir, "optimized_mano", "optimized.npz")):
            print(f"Skipping {name}: output exists (use --force to overwrite)")
        else:
            valid_names.append(name)

    print(f"Processing {len(valid_names)} segments")
    succeeded = 0

    for i, name in enumerate(valid_names):
        print(f"\n[{i + 1}/{len(valid_names)}] {name}")
        try:
            chains.claim(os.path.join(RESULTS_DIR, name), chain)
            ok = process_one_segment(name, config, args.device)
            if ok:
                succeeded += 1
        except OptimizerTerminalReject as e:
            reject_action(name, str(e), args.object)
            print(f"  TERMINAL REJECT: {e}")
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()

    print(f"\nDone: {succeeded}/{len(valid_names)} segments succeeded")
    if succeeded != len(valid_names):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
