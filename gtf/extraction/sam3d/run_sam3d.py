#!/usr/bin/env python3
"""Run SAM 3D Objects (3D reconstruction + pose estimation) on video segment keyframes.

Reads SAM3 masks from SAM3/masks.npz and selects keyframes at a target FPS rate
from frames with usable masks. The object is reconstructed once (Gaussian splat,
mesh and pose) from the best-ranked anchor keyframe; every other keyframe gets a
pose-only Stage 1 estimate (batched), with a full reconstruction as its fallback.
If no anchor candidate reconstructs, every keyframe is reconstructed independently.

Usage:
    python -m gtf.extraction.sam3d.run_sam3d
    python -m gtf.extraction.sam3d.run_sam3d --max-videos 2
    python -m gtf.extraction.sam3d.run_sam3d --select-fps 2
    python -m gtf.extraction.sam3d.run_sam3d --visualize

Outputs go to results/{segment_name}/sam3d/:
    - frame_{NNNN}/splat.ply   (Gaussian splat; anchor and full-reconstruction frames)
    - frame_{NNNN}/mesh.glb    (3D mesh; same frames)
    - frame_{NNNN}/pose.json   (rotation, translation, scale; same frames)
    - moge_keyframes.npz       (raw MoGe outputs for all successful keyframes)
    - metadata.json            (keyframe indices, per-frame status and pose)
    - .sam3d_done              (marker, only if all keyframes succeed)
    - vis_overlay.mp4          (optional: Original | Gaussian | Mesh overlay)
"""

# ---- Environment setup (must come before any sam3d imports) ----
import os
import sys

if "CUDA_HOME" not in os.environ:
    if os.path.isfile(os.path.join("/usr/local/cuda", "bin", "nvcc")):
        os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ["LIDRA_SKIP_INIT"] = "true"
# spconv's MaskImplicitGemm autotuner can find no kernel with torch 2.10 +
# spconv-cu126 2.3.8 ("can't find suitable algorithm for 0" in decode_slat).
# Force the untuned native algorithm through sam3d_objects' SPCONV_ALGO setting.
os.environ.setdefault("SPCONV_ALGO", "native")

# Auto-detect flash_attn; fall back to xformers if unavailable
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

from gtf.utils import RESULTS_DIR, load_manifest, rprint as print  # noqa: E402
from gtf.utils.video_timing import file_sha256  # noqa: E402
from gtf.extraction.sam3d.sam3d_keyframes import rank_anchor_candidates, select_keyframes  # noqa: E402
from gtf.extraction.sam3d.sam3d_inference import build_sam3d_inference  # noqa: E402
from gtf.extraction.sam3d.sam3d_frames import MogeCacheCollector, _compact_pose_tree, _load_frame, _save_overlay_video  # noqa: E402
from gtf.extraction.sam3d.sam3d_frames import _save_segment_metadata, process_frame, process_pose_frame, process_pose_frames_batch  # noqa: E402

MARKER_FILE = ".sam3d_done"
SAM3D_INPUT_BINDING_FORMAT = "sam3d-input-binding-v1"
POSE_STAGE1_STEPS = 10  # Stage 1 diffusion steps of the pose-only keyframes (the anchor reconstruction uses 20)


def segment_name(entry):
    return os.path.splitext(entry["file"])[0]


def _sam3d_input_binding(video_path, sam3_masks_path):
    """Content identity of the two mutable producer inputs SAM3D consumes."""

    return {
        "format": SAM3D_INPUT_BINDING_FORMAT,
        "video_sha256": file_sha256(video_path),
        "sam3_masks_sha256": file_sha256(sam3_masks_path),
    }


def outputs_complete(sam3d_dir, video_path=None, sam3_masks_path=None):
    """Validate the final marker and keyframe roster before resume skips."""

    sam3d_dir = Path(sam3d_dir)
    try:
        if (sam3d_dir / MARKER_FILE).read_text() != "done\n":
            return False
        metadata = json.loads((sam3d_dir / "metadata.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    complete = (
        isinstance(metadata, dict)
        and isinstance(metadata.get("num_keyframes_succeeded"), int)
        and metadata["num_keyframes_succeeded"] > 0
        and metadata.get("num_keyframes_failed") == 0
        and isinstance(metadata.get("keyframe_indices"), list)
    )
    if not complete:
        return False
    if video_path is None and sam3_masks_path is None:
        return True
    if video_path is None or sam3_masks_path is None:
        return False
    try:
        return metadata.get("input_binding") == _sam3d_input_binding(video_path, sam3_masks_path)
    except OSError:
        return False


def write_done_marker(sam3d_dir):
    marker = Path(sam3d_dir) / MARKER_FILE
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


def process_segment(
    inference_obj,
    name,
    seg_dir,
    sam3d_dir,
    select_fps=1.0,
    seed=42,
    visualize=False,
    pose_batch_size=8,
):
    """Process one segment: reconstruct once from the best frame, estimate the pose for all others.

    pose_batch_size: batched pose-only inference. A failed batch falls back to singleton
    inference with the same step count, and a failed singleton to a full reconstruction.

    Returns (num_succeeded, num_failed).
    """
    video_path = os.path.join(seg_dir, "video.mp4")
    sam3_masks_path = os.path.join(seg_dir, "SAM3", "masks.npz")
    input_binding = _sam3d_input_binding(video_path, sam3_masks_path)

    masks_data = np.load(sam3_masks_path)
    masks = masks_data["masks"]
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
    total_frames = len(vr)
    video_fps = vr.get_avg_fps()
    print(f"  Video: {total_frames} frames at {video_fps:.1f} fps")

    os.makedirs(sam3d_dir, exist_ok=True)

    keyframe_fields = {"keyframe_selection_mode": "select_fps_sam3d_bbox_v2"}
    keyframe_indices, keyframe_stats = select_keyframes(
        masks,
        video_fps=video_fps,
        select_fps=select_fps,
        total_frames=total_frames,
        return_stats=True,
    )
    keyframe_fields.update(keyframe_stats)
    if not keyframe_indices:
        print("  WARNING: no valid keyframes found (all masks empty)")
        return 0, 0

    print(f"  Selected {len(keyframe_indices)} keyframes: {keyframe_indices}")

    moge_collector = MogeCacheCollector(sam3d_dir)

    anchor_fields = {"anchor_selection_mode": "rank_all_keyframes_v1"}
    anchor_candidates = rank_anchor_candidates(masks, keyframe_indices)
    results = {}
    overlays = []
    overlay_frame_indices = []
    num_succeeded = 0
    num_failed = 0

    frozen_gs = None
    frozen_glb = None
    anchor_idx = None
    pipeline = inference_obj._pipeline

    for candidate in anchor_candidates:
        loaded = _load_frame(vr, masks, candidate, total_frames)
        if loaded is None:
            continue
        frame_rgb, frame_mask, mask_pixels = loaded
        frame_dir = os.path.join(sam3d_dir, f"frame_{candidate:04d}")
        print(f"  Anchor candidate frame {candidate}: mask pixels={mask_pixels}")
        try:
            pose, overlay, output = process_frame(
                inference_obj,
                frame_rgb,
                frame_mask,
                frame_dir,
                seed=seed,
                visualize=visualize,
                frame_index=candidate,
                moge_collector=moge_collector,
            )
            frozen_gs = output["gaussian"][0]
            frozen_glb = output.get("glb")
            anchor_idx = candidate
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
        print("  WARNING: all anchor candidates failed, falling back to per-frame mode")
        return _process_segment_independent(
            inference_obj,
            vr,
            masks,
            total_frames,
            keyframe_indices,
            sam3d_dir,
            select_fps,
            seed,
            visualize,
            moge_collector,
            {**keyframe_fields, **anchor_fields},
            video_path,
            sam3_masks_path,
            input_binding,
        )

    batched_poses = {}
    batched_mask_pixels = {}
    if pose_batch_size > 1 and not visualize:
        pose_indices = [index for index in keyframe_indices if index != anchor_idx]
        for offset in range(0, len(pose_indices), pose_batch_size):
            batch_indices = pose_indices[offset : offset + pose_batch_size]
            frame_items = []
            for frame_idx in batch_indices:
                loaded = _load_frame(vr, masks, frame_idx, total_frames)
                if loaded is None:
                    continue
                frame_rgb, frame_mask, mask_pixels = loaded
                frame_items.append(
                    {
                        "frame_index": frame_idx,
                        "frame_rgb": frame_rgb,
                        "frame_mask": frame_mask,
                        "mask_pixels": mask_pixels,
                        "frame_dir": os.path.join(sam3d_dir, f"frame_{frame_idx:04d}"),
                    }
                )
            if not frame_items:
                continue
            try:
                poses = process_pose_frames_batch(
                    pipeline,
                    frame_items,
                    seed=seed,
                    stage1_steps=POSE_STAGE1_STEPS,
                    moge_collector=moge_collector,
                )
                batched_poses.update((item["frame_index"], pose) for item, pose in zip(frame_items, poses))
                batched_mask_pixels.update((item["frame_index"], item["mask_pixels"]) for item in frame_items)
            except Exception as e:
                indices = [item["frame_index"] for item in frame_items]
                print(f"  Batched pose inference failed for {indices}: {e}; falling back to singleton")
                print(traceback.format_exc(), no_extra=True)
                # A failed batch may leave large, unusable allocations behind;
                # release those before the singleton recovery path.  Successful
                # batches deliberately retain allocator blocks for the next
                # same-shaped batch instead of synchronizing and reallocating.
                torch.cuda.empty_cache()

    for frame_idx in keyframe_indices:
        if frame_idx == anchor_idx:
            continue  # already processed

        loaded = None
        if frame_idx in batched_poses:
            # The batch builder already decoded/resized this frame and recorded
            # its mask population.  Pose publication needs only the CPU pose and
            # that scalar, so do not seek through the video a second time.
            frame_rgb = frame_mask = None
            mask_pixels = batched_mask_pixels[frame_idx]
        else:
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
        frame_dir = os.path.join(sam3d_dir, f"frame_{frame_idx:04d}")
        kf_num = sum(1 for k in keyframe_indices if k <= frame_idx)
        print(f"  [{kf_num}/{len(keyframe_indices)}] Frame {frame_idx} (pose-only): mask pixels={mask_pixels}")

        mode = "pose_only"
        try:
            if frame_idx in batched_poses:
                pose, overlay = batched_poses[frame_idx], None
                mode = f"pose_only_steps{POSE_STAGE1_STEPS}_batch{pose_batch_size}"
            else:
                pose, overlay = process_pose_frame(
                    pipeline,
                    frame_rgb,
                    frame_mask,
                    frame_dir,
                    frozen_gs,
                    frozen_glb,
                    seed=seed,
                    visualize=visualize,
                    stage1_steps=POSE_STAGE1_STEPS,
                    frame_index=frame_idx,
                    moge_collector=moge_collector,
                )

        except Exception as e:
            print(f"  Pose-only failed for frame {frame_idx}: {e}")
            print(traceback.format_exc(), no_extra=True)
            try:
                if loaded is None:
                    loaded = _load_frame(vr, masks, frame_idx, total_frames)
                    if loaded is None:
                        raise RuntimeError(f"frame_idx {frame_idx} >= total_frames {total_frames}")
                    frame_rgb, frame_mask, mask_pixels = loaded
                pose, overlay, _ = process_frame(
                    inference_obj,
                    frame_rgb,
                    frame_mask,
                    frame_dir,
                    seed=seed,
                    visualize=visualize,
                    frame_index=frame_idx,
                    moge_collector=moge_collector,
                )
                mode = "fallback"
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
        results[frame_idx] = {
            "frame_index": int(frame_idx),
            "status": "ok",
            "mode": mode,
            "pose": pose,
            "mask_pixels": mask_pixels,
        }
        num_succeeded += 1
        if frame_idx not in batched_poses or mode == "fallback":
            torch.cuda.empty_cache()

    del vr, masks, masks_data

    vis_overlay_path, overlay_frame_indices, num_overlay_frames = _save_overlay_video(overlays, overlay_frame_indices, sam3d_dir, select_fps)
    results_list = [results[idx] for idx in keyframe_indices if idx in results]
    if _sam3d_input_binding(video_path, sam3_masks_path) != input_binding:
        raise RuntimeError("SAM3D video or SAM3 masks changed during reconstruction; refusing to publish a mixed-generation result")
    _save_segment_metadata(
        sam3d_dir,
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
        extra_fields={
            "pose_batch_size": int(pose_batch_size),
            "pose_stage1_steps": POSE_STAGE1_STEPS,
            "pose_postopt": False,
            "compact_pose_tree": True,
            "input_binding": input_binding,
            **keyframe_fields,
            **anchor_fields,
        },
    )
    moge_collector.commit()
    _compact_pose_tree(sam3d_dir, results, anchor_idx)
    return num_succeeded, num_failed


def _process_segment_independent(
    inference_obj,
    vr,
    masks,
    total_frames,
    keyframe_indices,
    sam3d_dir,
    select_fps,
    seed,
    visualize,
    moge_collector,
    keyframe_fields=None,
    video_path=None,
    sam3_masks_path=None,
    input_binding=None,
):
    """Fallback when no anchor candidate reconstructs: run the full reconstruction on every keyframe independently."""
    results = []
    overlays = []
    overlay_frame_indices = []
    num_succeeded = 0
    num_failed = 0

    for fi, frame_idx in enumerate(keyframe_indices):
        loaded = _load_frame(vr, masks, frame_idx, total_frames)
        if loaded is None:
            print(f"  WARNING: frame {frame_idx} >= total {total_frames}, skipping")
            results.append(
                {
                    "frame_index": int(frame_idx),
                    "status": "failed",
                    "error": f"frame_idx {frame_idx} >= total_frames {total_frames}",
                }
            )
            num_failed += 1
            continue

        frame_rgb, frame_mask, mask_pixels = loaded
        frame_dir = os.path.join(sam3d_dir, f"frame_{frame_idx:04d}")
        print(f"  [{fi + 1}/{len(keyframe_indices)}] Frame {frame_idx}: mask pixels={mask_pixels}")

        try:
            pose, overlay, _ = process_frame(
                inference_obj,
                frame_rgb,
                frame_mask,
                frame_dir,
                seed=seed,
                visualize=visualize,
                frame_index=frame_idx,
                moge_collector=moge_collector,
            )
            if overlay is not None:
                overlays.append(overlay)
                overlay_frame_indices.append(int(frame_idx))
            results.append(
                {
                    "frame_index": int(frame_idx),
                    "status": "ok",
                    "mode": "independent",
                    "pose": pose,
                    "mask_pixels": mask_pixels,
                }
            )
            num_succeeded += 1

        except Exception as e:
            print(f"  ERROR on frame {frame_idx}: {e}")
            print(traceback.format_exc(), no_extra=True)
            if visualize:
                overlays.append(np.concatenate([frame_rgb, frame_rgb, frame_rgb], axis=1))
                overlay_frame_indices.append(int(frame_idx))
            results.append(
                {
                    "frame_index": int(frame_idx),
                    "status": "failed",
                    "error": str(e),
                    "mask_pixels": mask_pixels,
                }
            )
            num_failed += 1

        torch.cuda.empty_cache()

    video_fps = vr.get_avg_fps()
    del vr, masks

    vis_overlay_path, overlay_frame_indices, num_overlay_frames = _save_overlay_video(overlays, overlay_frame_indices, sam3d_dir, select_fps)
    if video_path is None or sam3_masks_path is None or input_binding is None or _sam3d_input_binding(video_path, sam3_masks_path) != input_binding:
        raise RuntimeError("SAM3D video or SAM3 masks changed during independent reconstruction; refusing to publish a mixed-generation result")
    _save_segment_metadata(
        sam3d_dir,
        results,
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
        tracking_mode=False,
        extra_fields={**(keyframe_fields or {}), "input_binding": input_binding},
    )
    if moge_collector is not None:
        moge_collector.commit()
    else:
        (Path(sam3d_dir) / "moge_keyframes.npz").unlink(missing_ok=True)
    return num_succeeded, num_failed


def main():
    parser = argparse.ArgumentParser(prog="python -m gtf.extraction.sam3d.run_sam3d", description="Run SAM 3D Objects (3D reconstruction + pose) on video segment keyframes")
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
        help="Path to SAM3D checkpoints directory (default: sam-3d-objects/checkpoints/hf)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate per-frame overlay videos (sam3d/vis_overlay.mp4)",
    )
    parser.add_argument(
        "--pose-batch-size",
        type=int,
        default=8,
        help="Pose-only Stage 1 batch size (default: 8; use 1 for singleton)",
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
        sam3d_dir = os.path.join(seg_dir, "sam3d")
        video_path = os.path.join(seg_dir, "video.mp4")
        sam3_masks_path = os.path.join(seg_dir, "SAM3", "masks.npz")

        if not os.path.exists(video_path):
            print(f"[{vi + 1}/{N}] SKIP (no video): {name}")
            failed += 1
            continue

        if not os.path.exists(sam3_masks_path):
            print(f"[{vi + 1}/{N}] SKIP (no SAM3 masks): {name}")
            no_masks += 1
            continue

        if not args.force and outputs_complete(sam3d_dir, video_path, sam3_masks_path):
            skipped += 1
            continue

        print(f"\n[{vi + 1}/{N}] Processing: {name}")
        try:
            (Path(sam3d_dir) / MARKER_FILE).unlink(missing_ok=True)
            num_ok, num_fail = process_segment(
                inference_obj,
                name,
                seg_dir,
                sam3d_dir,
                select_fps=args.select_fps,
                seed=args.seed,
                visualize=args.visualize,
                pose_batch_size=max(1, args.pose_batch_size),
            )

            if num_fail == 0 and num_ok > 0:
                write_done_marker(sam3d_dir)
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

    print(f"\n[SAM3D] Done: {done} processed, {skipped} skipped, {no_masks} no masks, {failed} failed")


if __name__ == "__main__":
    main()
