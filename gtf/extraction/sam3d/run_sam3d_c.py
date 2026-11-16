#!/usr/bin/env python3
"""SAM3D with a temporal prior: dual-init pose estimation using the previous keyframe's pose.

The temporal fallback for object rotation: the pipeline runs it only on segments whose
object bundle adjustment is missing or rejected, and the optimizer then reads its poses.

Reconstructs the object once from the best anchor frame (same as run_sam3d.py),
then for all other keyframes runs dual-init pose estimation:
  - Init A: previous keyframe's refined pose as starting point
  - Init B: Stage 1's per-frame estimate (what run_sam3d.py uses)
Both are refined against the frozen anchor Gaussians; the higher IoU wins, with a
rotation-aware margin in favor of Init A (see run_dual_init_frame). Processes
bidirectionally from the anchor (forward to the end, backward to the start).

Results go to sam3d_c/ next to sam3d/, which is left unchanged.

Usage:
    python -m gtf.extraction.sam3d.run_sam3d_c
    python -m gtf.extraction.sam3d.run_sam3d_c --max-videos 1 --force --visualize

Outputs go to results/{segment_name}/sam3d_c/:
    - frame_{NNNN}/splat.ply   (anchor + fallback frames only)
    - frame_{NNNN}/mesh.glb    (anchor + fallback frames only)
    - frame_{NNNN}/pose.json   (all frames, with init_a_iou/init_b_iou/winner)
    - metadata.json
    - .sam3d_c_done            (marker, only if all keyframes succeed)
    - vis_overlay.mp4          (optional overlay video)
"""

# ---- Environment setup (must come before any sam3d imports) ----
import os
import sys

if "CUDA_HOME" not in os.environ and os.path.isfile(os.path.join("/usr/local/cuda", "bin", "nvcc")):
    os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ["LIDRA_SKIP_INIT"] = "true"

try:
    import flash_attn  # noqa: F401
except ImportError:
    os.environ.setdefault("ATTN_BACKEND", "xformers")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")

from gtf.paths import REPO_ROOT

ROOT = str(REPO_ROOT)
SAM3D_REPO = os.path.join(ROOT, "third-party", "sam-3d-objects")
SAM3D_NOTEBOOK = os.path.join(SAM3D_REPO, "notebook")

for p in (SAM3D_REPO, SAM3D_NOTEBOOK):
    if p not in sys.path:
        sys.path.insert(0, p)

import argparse  # noqa: E402
import json  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from decord import VideoReader, cpu  # noqa: E402

from pathlib import Path  # noqa: E402

from gtf.utils import load_manifest, rprint as print  # noqa: E402
from gtf.extraction.sam3d.run_sam3d import RESULTS_DIR, _process_segment_independent, segment_name  # noqa: E402
from gtf.extraction.sam3d.sam3d_frames import _extract_pose_dict, _load_frame, _save_overlay_video, _save_segment_metadata, process_frame, render_overlay_frame  # noqa: E402
from gtf.extraction.sam3d.sam3d_inference import build_sam3d_inference, run_stage1_pose  # noqa: E402
from gtf.extraction.sam3d.sam3d_keyframes import rank_anchor_candidates, select_keyframes  # noqa: E402

MARKER_FILE_C = ".sam3d_c_done"
DEFAULT_TIEBREAKER_MARGIN = 0.05
# Both candidates' IoUs below this (scaled by the frame's mask area relative to the anchor's): full reconstruction.
POSE_ONLY_IOU_THRESHOLD = 0.3


def outputs_complete_c(sam3d_c_dir):
    """Validate the final marker and temporal keyframe roster."""

    sam3d_c_dir = Path(sam3d_c_dir)
    try:
        if (sam3d_c_dir / MARKER_FILE_C).read_text() != "done\n":
            return False
        metadata = json.loads((sam3d_c_dir / "metadata.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(metadata, dict)
        and isinstance(metadata.get("num_keyframes_succeeded"), int)
        and metadata["num_keyframes_succeeded"] > 0
        and metadata.get("num_keyframes_failed") == 0
        and metadata.get("temporal_prior") is True
    )


def write_done_marker_c(sam3d_c_dir):
    marker = Path(sam3d_c_dir) / MARKER_FILE_C
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as stream:
            stream.write("done\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, marker)
    finally:
        temporary.unlink(missing_ok=True)


def run_post_opt_with_pose(pipeline, frozen_gs, intrinsics, ss_return_dict, ss_input_dict, pose):
    """Run post-optimization starting from a specific pose. Returns (None, 0.0) on failure."""
    from copy import deepcopy

    pose_dict = {**ss_return_dict, "rotation": pose["rotation"], "translation": pose["translation"], "scale": pose["scale"]}
    try:
        refined = pipeline.run_post_optimization_GS(
            deepcopy(frozen_gs),
            intrinsics,
            pose_dict,
            ss_input_dict,
            backend="gsplat",
        )
        iou = refined.get("iou", 0.0)
        if hasattr(iou, "item"):
            iou = iou.item()
        return refined, iou
    except Exception as e:
        print(f"    WARNING: post-opt failed for init: {e}")
        return None, 0.0


ROTATION_MARGIN_LAMBDA = 0.35  # extra IoU margin per 180 deg of rotation away from the previous pose (run_dual_init_frame)


def _quat_angle_deg(qa, qb):
    """Angle (deg) between two quaternion rotations (sign-invariant)."""
    a = qa.reshape(-1)[:4].double()
    b = qb.reshape(-1)[:4].double()
    a = a / a.norm().clamp(min=1e-9)
    b = b / b.norm().clamp(min=1e-9)
    d = torch.abs((a * b).sum()).clamp(max=1.0)
    return float(torch.rad2deg(2 * torch.acos(d)))


def run_dual_init_frame(pipeline, rgba, frozen_gs, prev_pose, seed=42, stage1_steps=20, tiebreaker_margin=DEFAULT_TIEBREAKER_MARGIN, rotation_lambda=ROTATION_MARGIN_LAMBDA):
    """Dual-init pose estimation: prev_pose vs stage1_pose, rotation-aware IoU winner rule.

    Both candidates are refined by layout post-optimization against the frozen anchor
    Gaussians. The fresh Stage 1 candidate must beat the temporal candidate by a margin
    that grows with its rotational distance from the previous pose:
        accept stage1 iff iou_b > iou_a + margin + rotation_lambda * (rot_dist/180)
    On silhouette-symmetric objects a ~180-deg flipped pose fits the mask as well as the
    temporal pose, so a flat margin lets such flips through; with the rotation term a
    180-deg flip pays +rotation_lambda IoU, while a genuine recovery after track loss (the
    prev-pose refinement collapsed) still wins on its large IoU gap. rotation_lambda=0
    gives the flat margin rule."""
    ss_return_dict, ss_input_dict, pointmap_dict = run_stage1_pose(pipeline, rgba, seed, stage1_steps)
    intrinsics = pointmap_dict["intrinsics"]

    stage1_pose = {k: ss_return_dict[k].clone() for k in ("rotation", "translation", "scale")}

    refined_a, iou_a = run_post_opt_with_pose(pipeline, frozen_gs, intrinsics, ss_return_dict, ss_input_dict, prev_pose)
    refined_b, iou_b = run_post_opt_with_pose(pipeline, frozen_gs, intrinsics, ss_return_dict, ss_input_dict, stage1_pose)

    if refined_a is None and refined_b is None:
        winner = "stage1"
        result = {**ss_return_dict}
    elif refined_a is None:
        winner = "stage1"
        result = {**ss_return_dict, **refined_b}
    elif refined_b is None:
        winner = "prev_pose"
        result = {**ss_return_dict, **refined_a}
    else:
        rot_dist = _quat_angle_deg(refined_b["rotation"], prev_pose["rotation"])
        eff_margin = tiebreaker_margin + rotation_lambda * (rot_dist / 180.0)
        if iou_a >= iou_b - eff_margin:
            winner = "prev_pose"
            result = {**ss_return_dict, **refined_a}
        else:
            winner = "stage1"
            result = {**ss_return_dict, **refined_b}
        result["rot_dist_b_prev_deg"] = rot_dist

    result["intrinsics"] = intrinsics
    result["init_a_iou"] = iou_a
    result["init_b_iou"] = iou_b
    result["winner"] = winner

    return result


def extract_pose_tensors(output):
    """Extract rotation/translation/scale as GPU tensors from an output dict."""
    return {k: output[k].detach().clone() for k in ("rotation", "translation", "scale")}


def save_pose_json(output, frame_dir, extra_fields=None):
    """Serialize pose + optional extra fields to pose.json. Returns pose dict."""
    os.makedirs(frame_dir, exist_ok=True)
    pose = _extract_pose_dict(output, extra_keys=("iou", "iou_before_optim", "optim_accepted", "init_a_iou", "init_b_iou", "winner"))
    if extra_fields:
        pose.update(extra_fields)
    with open(os.path.join(frame_dir, "pose.json"), "w") as f:
        json.dump(pose, f, indent=2)
    return pose


def clean_stale_artifacts(frame_dir):
    """Remove stale splat.ply and mesh.glb from a frame directory."""
    for stale in ("splat.ply", "mesh.glb"):
        stale_path = os.path.join(frame_dir, stale)
        if os.path.exists(stale_path):
            os.remove(stale_path)


def process_segment_temporal(
    inference_obj,
    name,
    seg_dir,
    sam3d_c_dir,
    select_fps=10.0,
    seed=42,
    visualize=False,
    tiebreaker_margin=DEFAULT_TIEBREAKER_MARGIN,
):
    """Process one segment with temporal prior: bidirectional dual-init from anchor.

    Returns (num_succeeded, num_failed).
    """
    video_path = os.path.join(seg_dir, "video.mp4")
    sam3_masks_path = os.path.join(seg_dir, "SAM3", "masks.npz")

    masks_data = np.load(sam3_masks_path)
    masks = masks_data["masks"]
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
    total_frames = len(vr)
    video_fps = vr.get_avg_fps()
    print(f"  Video: {total_frames} frames at {video_fps:.1f} fps")

    keyframe_indices = select_keyframes(masks, video_fps=video_fps, select_fps=select_fps)
    if not keyframe_indices:
        print("  WARNING: no valid keyframes found (all masks empty)")
        return 0, 0

    print(f"  Selected {len(keyframe_indices)} keyframes: {keyframe_indices}")
    os.makedirs(sam3d_c_dir, exist_ok=True)
    anchor_candidates = rank_anchor_candidates(masks, keyframe_indices)

    results = {}
    overlays = []
    overlay_frame_indices = []
    num_succeeded = 0
    num_failed = 0
    pipeline = inference_obj._pipeline

    frozen_gs = None
    frozen_glb = None
    anchor_idx = None
    anchor_output = None

    for candidate in anchor_candidates:
        loaded = _load_frame(vr, masks, candidate, total_frames)
        if loaded is None:
            continue
        frame_rgb, frame_mask, mask_pixels = loaded
        frame_dir = os.path.join(sam3d_c_dir, f"frame_{candidate:04d}")
        print(f"  Anchor candidate frame {candidate}: mask pixels={mask_pixels}")
        try:
            pose, overlay, output = process_frame(
                inference_obj,
                frame_rgb,
                frame_mask,
                frame_dir,
                seed=seed,
                visualize=visualize,
            )
            frozen_gs = output["gaussian"][0]
            frozen_glb = output.get("glb")
            anchor_idx = candidate
            anchor_output = output
            results[candidate] = {
                "frame_index": int(candidate),
                "status": "ok",
                "mode": "anchor",
                "pose": pose,
                "mask_pixels": mask_pixels,
            }
            if overlay is not None:
                overlays.append(overlay)
                overlay_frame_indices.append(int(candidate))
            num_succeeded += 1
            print(f"  Anchor frame {candidate} succeeded")
            break
        except Exception as e:
            print(f"  Anchor candidate {candidate} failed: {e}")
            print(traceback.format_exc(), no_extra=True)
            continue
        finally:
            torch.cuda.empty_cache()

    if frozen_gs is None:
        print("  WARNING: all anchor candidates failed, falling back to per-frame independent mode")
        return _process_segment_independent(
            inference_obj,
            vr,
            masks,
            total_frames,
            keyframe_indices,
            sam3d_c_dir,
            select_fps,
            seed,
            visualize,
            None,  # no MoGe keyframe cache in the temporal variant
        )

    anchor_pose = extract_pose_tensors(anchor_output)
    anchor_mask_pixels = results[anchor_idx]["mask_pixels"]

    # Split keyframes into forward (after anchor) and backward (before anchor)
    forward_frames = [idx for idx in keyframe_indices if idx > anchor_idx]
    backward_frames = [idx for idx in keyframe_indices if idx < anchor_idx]
    backward_frames.reverse()  # process from anchor outward

    for pass_name, pass_frames in [
        ("forward", forward_frames),
        ("backward", backward_frames),
    ]:
        if not pass_frames:
            continue
        print(f"  {pass_name.capitalize()} pass: {len(pass_frames)} frames")
        prev_pose = {k: v.clone() for k, v in anchor_pose.items()}

        for frame_idx in pass_frames:
            loaded = _load_frame(vr, masks, frame_idx, total_frames)
            if loaded is None:
                print(f"  WARNING: frame {frame_idx} >= total {total_frames}, skipping")
                results[frame_idx] = {
                    "frame_index": int(frame_idx),
                    "status": "failed",
                    "error": f"frame_idx {frame_idx} >= total_frames {total_frames}",
                }
                num_failed += 1
                continue

            frame_rgb, frame_mask, mask_pixels = loaded
            frame_dir = os.path.join(sam3d_c_dir, f"frame_{frame_idx:04d}")
            kf_num = sum(1 for k in keyframe_indices if k <= frame_idx)
            print(f"  [{kf_num}/{len(keyframe_indices)}] Frame {frame_idx} ({pass_name}): mask pixels={mask_pixels}")

            mode = "temporal"
            try:
                os.makedirs(frame_dir, exist_ok=True)
                clean_stale_artifacts(frame_dir)

                rgba = np.concatenate(
                    [
                        frame_rgb[..., :3],
                        (frame_mask.astype(np.uint8) * 255)[..., None],
                    ],
                    axis=-1,
                )
                output = run_dual_init_frame(
                    pipeline,
                    rgba,
                    frozen_gs,
                    prev_pose,
                    seed=seed,
                    tiebreaker_margin=tiebreaker_margin,
                )

                output["gaussian"] = [frozen_gs]
                output["glb"] = frozen_glb

                iou_a = output["init_a_iou"]
                iou_b = output["init_b_iou"]
                winner = output["winner"]

                # Scale the threshold by the mask area ratio, so occlusion alone does not trigger the fallback
                mask_ratio = mask_pixels / anchor_mask_pixels
                adjusted_threshold = POSE_ONLY_IOU_THRESHOLD * mask_ratio

                print(
                    f"    Init A (prev_pose) IoU={iou_a:.3f}, "
                    f"Init B (stage1) IoU={iou_b:.3f} → winner={winner} "
                    f"(mask_ratio={mask_ratio:.2f}, threshold={adjusted_threshold:.3f})"
                )

                if iou_a < adjusted_threshold and iou_b < adjusted_threshold:
                    print(f"    Both IoUs below adjusted threshold {adjusted_threshold:.3f}, falling back to full reconstruction + chain reset")
                    pose, overlay, full_output = process_frame(
                        inference_obj,
                        frame_rgb,
                        frame_mask,
                        frame_dir,
                        seed=seed,
                        visualize=visualize,
                    )
                    mode = "fallback"
                    prev_pose = extract_pose_tensors(full_output)
                    results[frame_idx] = {
                        "frame_index": int(frame_idx),
                        "status": "ok",
                        "mode": mode,
                        "init_a_iou": iou_a,
                        "init_b_iou": iou_b,
                        "mask_ratio": round(mask_ratio, 4),
                        "adjusted_threshold": round(adjusted_threshold, 4),
                        "pose": pose,
                        "mask_pixels": mask_pixels,
                    }
                else:
                    pose = save_pose_json(output, frame_dir)
                    overlay = render_overlay_frame(output, frame_rgb, output["intrinsics"]) if visualize else None
                    prev_pose = extract_pose_tensors(output)
                    results[frame_idx] = {
                        "frame_index": int(frame_idx),
                        "status": "ok",
                        "mode": mode,
                        "init_a_iou": iou_a,
                        "init_b_iou": iou_b,
                        "winner": winner,
                        "mask_ratio": round(mask_ratio, 4),
                        "adjusted_threshold": round(adjusted_threshold, 4),
                        "pose": pose,
                        "mask_pixels": mask_pixels,
                    }

            except Exception as e:
                print(f"  Dual-init failed for frame {frame_idx}: {e}")
                print(traceback.format_exc(), no_extra=True)
                try:
                    pose, overlay, full_output = process_frame(
                        inference_obj,
                        frame_rgb,
                        frame_mask,
                        frame_dir,
                        seed=seed,
                        visualize=visualize,
                    )
                    mode = "fallback"
                    prev_pose = extract_pose_tensors(full_output)
                    results[frame_idx] = {
                        "frame_index": int(frame_idx),
                        "status": "ok",
                        "mode": mode,
                        "pose": pose,
                        "mask_pixels": mask_pixels,
                    }
                except Exception as e2:
                    print(f"  ERROR on frame {frame_idx} (fallback also failed): {e2}")
                    print(traceback.format_exc(), no_extra=True)
                    if visualize:
                        overlays.append(np.concatenate([frame_rgb, frame_rgb, frame_rgb], axis=1))
                        overlay_frame_indices.append(int(frame_idx))
                    results[frame_idx] = {
                        "frame_index": int(frame_idx),
                        "status": "failed",
                        "error": str(e2),
                        "mask_pixels": mask_pixels,
                    }
                    num_failed += 1
                    torch.cuda.empty_cache()
                    continue

            if overlay is not None:
                overlays.append(overlay)
                overlay_frame_indices.append(int(frame_idx))
            if results[frame_idx]["status"] == "ok":
                num_succeeded += 1
            torch.cuda.empty_cache()

    del vr, masks, masks_data

    vis_overlay_path, overlay_frame_indices, num_overlay_frames = _save_overlay_video(overlays, overlay_frame_indices, sam3d_c_dir, select_fps)
    results_list = [results[idx] for idx in keyframe_indices if idx in results]
    n_temporal = sum(1 for r in results_list if r.get("status") == "ok" and r.get("mode") == "temporal")
    n_fallback = sum(1 for r in results_list if r.get("status") == "ok" and r.get("mode") == "fallback")
    n_prev_wins = sum(1 for r in results_list if r.get("winner") == "prev_pose")
    n_stage1_wins = sum(1 for r in results_list if r.get("winner") == "stage1")

    print(f"  Summary: anchor=1, temporal={n_temporal} (prev_pose={n_prev_wins}, stage1={n_stage1_wins}), fallback={n_fallback}, failed={num_failed}")

    _save_segment_metadata(
        sam3d_c_dir,
        results_list,
        total_frames,
        video_fps,
        select_fps,
        keyframe_indices,
        num_succeeded,
        num_failed,
        vis_overlay_path,
        overlay_frame_indices,
        num_overlay_frames,
        seed,
        anchor_frame_index=anchor_idx,
        extra_fields={"temporal_prior": True, "processing_order": "bidirectional", "tiebreaker_margin": tiebreaker_margin, "iou_threshold": POSE_ONLY_IOU_THRESHOLD},
    )
    return num_succeeded, num_failed


def main():
    parser = argparse.ArgumentParser(prog="python -m gtf.extraction.sam3d.run_sam3d_c", description="SAM3D with a temporal prior: dual-init pose estimation")
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Process only first N segments (0 = all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recompute even if outputs exist",
    )
    parser.add_argument(
        "--select-fps",
        type=float,
        default=10.0,
        help="Target FPS for keyframe selection (default: 10.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for SAM3D inference (default: 42)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for SAM3D model",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Path to SAM3D checkpoints directory",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate per-frame overlay videos (sam3d_c/vis_overlay.mp4)",
    )
    parser.add_argument(
        "--tiebreaker-margin",
        type=float,
        default=DEFAULT_TIEBREAKER_MARGIN,
        help=f"IoU margin for tiebreaker favoring prev_pose (default: {DEFAULT_TIEBREAKER_MARGIN})",
    )
    parser.add_argument(
        "--object",
        type=str,
        default=None,
        help="Only process segments validated for this object (auto-detected if omitted)",
    )
    args = parser.parse_args()

    entries = load_manifest()
    if args.max_videos > 0:
        entries = entries[: args.max_videos]
        print(f"Limited to first {args.max_videos} segments")

    names = [segment_name(e) for e in entries]

    from gtf.utils import is_action_valid

    total = len(names)
    valid_names = [n for n in names if is_action_valid(n, args.object)]
    _label = f" ({args.object})" if args.object else ""
    print(f"Action validation filter{_label}: {len(valid_names)}/{total} valid")
    names = valid_names

    inference_obj = build_sam3d_inference(checkpoint_dir=args.checkpoint_dir, compile_model=args.compile)

    N = len(names)
    done = skipped = failed = no_masks = 0

    for vi, name in enumerate(names):
        seg_dir = os.path.join(RESULTS_DIR, name)
        sam3d_c_dir = os.path.join(seg_dir, "sam3d_c")
        video_path = os.path.join(seg_dir, "video.mp4")
        sam3_masks_path = os.path.join(seg_dir, "SAM3", "masks.npz")

        if not os.path.exists(video_path):
            print(f"[{vi + 1}/{N}] SKIP (no video): {name}")
            failed += 1
            continue

        if not args.force and outputs_complete_c(sam3d_c_dir):
            skipped += 1
            continue

        if not os.path.exists(sam3_masks_path):
            print(f"[{vi + 1}/{N}] SKIP (no SAM3 masks): {name}")
            no_masks += 1
            continue

        print(f"\n[{vi + 1}/{N}] Processing: {name}")
        try:
            (Path(sam3d_c_dir) / MARKER_FILE_C).unlink(missing_ok=True)
            num_ok, num_fail = process_segment_temporal(
                inference_obj,
                name,
                seg_dir,
                sam3d_c_dir,
                select_fps=args.select_fps,
                seed=args.seed,
                visualize=args.visualize,
                tiebreaker_margin=args.tiebreaker_margin,
            )

            if num_fail == 0 and num_ok > 0:
                write_done_marker_c(sam3d_c_dir)
                done += 1
                print(f"  Done: {name} ({num_ok} keyframes)")
            elif num_ok > 0:
                print(f"  PARTIAL: {name} ({num_ok} ok, {num_fail} failed) — marker NOT written")
                failed += 1
            else:
                print(f"  FAILED: {name} (no keyframes succeeded)")
                failed += 1

            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR processing {name}: {e}")
            print(traceback.format_exc(), no_extra=True)
            failed += 1
            torch.cuda.empty_cache()
            continue

    print(f"\n[SAM3D-C] Done: {done} processed, {skipped} skipped, {no_masks} no masks, {failed} failed")


if __name__ == "__main__":
    main()
