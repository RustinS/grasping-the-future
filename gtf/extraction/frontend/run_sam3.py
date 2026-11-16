#!/usr/bin/env python3
"""Run SAM3 video object segmentation using Molmo2 point prompts.

Prompts SAM3 with the Molmo2 points on the pointing frame (molmo2/points.json),
then propagates the mask forward and backward through the video.

Outputs go to results/{segment_name}/SAM3/:
    masks.npz, metadata.json, .sam3_done, vis.mp4 (optional)
"""

import argparse
import json
import math
import os
import tempfile
import traceback
import zipfile
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu

from gtf.utils import RESULTS_DIR, is_action_valid, load_manifest, rprint as print
from gtf.utils.npz import savez_compressed_level
from gtf.utils.video_timing import file_sha256

MARKER_FILE = ".sam3_done"


def outputs_complete(sam3_dir, video_path=None, points_path=None):
    """Validate the published SAM3 transaction before a resume skips it."""

    sam3_dir = Path(sam3_dir)
    masks_path = sam3_dir / "masks.npz"
    metadata_path = sam3_dir / "metadata.json"
    marker_path = sam3_dir / MARKER_FILE
    try:
        if marker_path.read_text() != "done\n":
            return False
        metadata = json.loads(metadata_path.read_text())
        with np.load(masks_path, allow_pickle=False) as archive:
            if archive.files != ["masks"]:
                return False
            mask_shape = list(archive["masks"].shape)
        shape = metadata.get("mask_shape")
        complete = (
            isinstance(shape, list)
            and len(shape) == 3
            and all(isinstance(value, int) and value > 0 for value in shape)
            and shape == mask_shape
            and metadata.get("total_frames") == shape[0]
        )
        if not complete:
            return False
        if video_path is not None:
            video_path = Path(video_path)
            if not video_path.is_file() or metadata.get("video_sha256") != file_sha256(video_path):
                return False
        if points_path is not None:
            points_path = Path(points_path)
            if not points_path.is_file() or metadata.get("points_sha256") != file_sha256(points_path):
                return False
        return True
    except (OSError, ValueError, EOFError, json.JSONDecodeError, zipfile.BadZipFile):
        return False


def _atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path, payload):
    _atomic_text(path, json.dumps(payload, indent=2) + "\n")


def write_done_marker(sam3_dir):
    _atomic_text(Path(sam3_dir) / MARKER_FILE, "done\n")


def build_predictor(checkpoint_path=None):
    """Build SAM3 tracker predictor (SAM2-compatible API for point prompts)."""
    from sam3.model_builder import build_sam3_video_model

    print("Building SAM3 video model...")
    sam3_model = build_sam3_video_model(checkpoint_path=checkpoint_path)
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone
    print("SAM3 tracker predictor ready")
    return predictor


def _prompt_payload(molmo_data):
    """Return (points normalized by the image size, labels, n_positive, n_negative).

    The Molmo2 points are positive prompts (label 1); there are no negative prompts.
    """

    width = molmo_data["image_width"]
    height = molmo_data["image_height"]
    if (
        not isinstance(width, (int, float))
        or isinstance(width, bool)
        or not isinstance(height, (int, float))
        or isinstance(height, bool)
        or not math.isfinite(float(width))
        or not math.isfinite(float(height))
        or float(width) <= 0
        or float(height) <= 0
    ):
        raise ValueError("SAM3 prompt image dimensions must be finite and positive")
    positives = molmo_data["points"]
    if not isinstance(positives, list):
        raise ValueError("SAM3 prompt points must be a list")
    rows = []
    for point in positives:
        x, y = point.get("pixel_x"), point.get("pixel_y")
        if (
            not isinstance(x, (int, float))
            or isinstance(x, bool)
            or not isinstance(y, (int, float))
            or isinstance(y, bool)
            or not math.isfinite(float(x))
            or not math.isfinite(float(y))
            or not 0 <= float(x) < float(width)
            or not 0 <= float(y) < float(height)
        ):
            raise ValueError("SAM3 prompt point lacks numeric pixel coordinates")
        rows.append([float(x) / width, float(y) / height])
    if not positives:
        raise ValueError("SAM3 requires at least one positive point")
    return rows, [1] * len(rows), len(positives), 0


def process_segment(predictor, seg_dir, sam3_dir):
    """Run SAM3 on one segment using Molmo2 points. Returns True on success."""
    video_path = os.path.join(seg_dir, "video.mp4")
    molmo_path = os.path.join(seg_dir, "molmo2", "points.json")

    with open(molmo_path) as f:
        molmo_data = json.load(f)

    video_sha256 = file_sha256(video_path)
    points_sha256 = file_sha256(molmo_path)
    if molmo_data.get("video_sha256") != video_sha256:
        raise ValueError("Molmo2 prompt is not bound to the current video; rerun Molmo2 before SAM3")

    frame_index = molmo_data["frame_index"]
    img_w, img_h = molmo_data["image_width"], molmo_data["image_height"]
    points_list = molmo_data["points"]

    if not points_list:
        print(f"  WARNING: no points in {molmo_path}")
        return False

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
    total_frames = len(vr)
    vid_h, vid_w = vr[0].shape[:2]
    del vr

    if not isinstance(frame_index, int) or isinstance(frame_index, bool) or not 0 <= frame_index < total_frames:
        raise ValueError(f"SAM3 prompt frame_index {frame_index!r} is outside 0:{total_frames}")

    if img_w != vid_w or img_h != vid_h:
        raise ValueError(f"SAM3 prompt size ({img_w}x{img_h}) != video ({vid_w}x{vid_h})")

    rel_points, point_labels, num_positive, num_negative = _prompt_payload(molmo_data)
    num_points = len(rel_points)
    print(f"  Prompting frame {frame_index}/{total_frames} with {num_positive} positive + {num_negative} negative point(s)")

    inference_state = predictor.init_state(video_path=video_path)
    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_index,
        obj_id=1,
        points=torch.tensor(rel_points, dtype=torch.float32),
        labels=torch.tensor(point_labels, dtype=torch.int32),
        clear_old_points=False,
    )

    # Propagate forward then backward from prompted frame
    video_segments = {}
    for reverse, extra in [(False, {"propagate_preflight": True}), (True, {})]:
        for frame_idx, obj_ids, _, video_res_masks, _ in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=frame_index,
            max_frame_num_to_track=total_frames,
            reverse=reverse,
            **extra,
        ):
            for i, oid in enumerate(obj_ids):
                if oid == 1:
                    video_segments[frame_idx] = (video_res_masks[i] > 0.0).squeeze(0).cpu().numpy()

    # Assemble masks
    first_mask = next(iter(video_segments.values()), None)
    mask_h, mask_w = first_mask.shape if first_mask is not None else (vid_h, vid_w)
    masks = np.zeros((total_frames, mask_h, mask_w), dtype=np.uint8)
    for fi, seg_mask in video_segments.items():
        if fi < total_frames:
            masks[fi] = seg_mask.astype(np.uint8)
    frames_with_masks = sum(1 for fi, m in video_segments.items() if fi < total_frames and m.any())

    print(f"  Masks: {frames_with_masks}/{total_frames} frames, shape ({total_frames}, {mask_h}, {mask_w})")

    os.makedirs(sam3_dir, exist_ok=True)
    savez_compressed_level(
        os.path.join(sam3_dir, "masks.npz"),
        compresslevel=6,
        allow_pickle=False,
        masks=masks,
    )

    meta = {
        "frame_index_prompted": frame_index,
        "num_points": num_points,
        "num_positive_points": num_positive,
        "num_negative_points": num_negative,
        "point_labels": point_labels,
        "rel_points": rel_points,
        "total_frames": total_frames,
        "frames_with_masks": frames_with_masks,
        "mask_shape": [total_frames, mask_h, mask_w],
        "video_sha256": video_sha256,
        "points_sha256": points_sha256,
    }
    if file_sha256(video_path) != video_sha256:
        raise RuntimeError("video changed during SAM3 inference")
    if file_sha256(molmo_path) != points_sha256:
        raise RuntimeError("Molmo2 points changed during SAM3 inference")
    _atomic_json(os.path.join(sam3_dir, "metadata.json"), meta)
    return True


def render_vis_video(video_path, sam3_dir):
    """Render overlay video showing SAM3 masks on original frames."""
    import imageio.v3 as iio

    masks = np.load(os.path.join(sam3_dir, "masks.npz"))["masks"]
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
    fps = vr.get_avg_fps()
    T = len(vr)

    alpha, color = 0.4, np.array([30, 144, 255], dtype=np.float64)
    vis_frames = []

    for start in range(0, T, 64):
        indices = list(range(start, min(start + 64, T)))
        batch = vr.get_batch(indices)
        # decord bridge may be set to "torch" by SAM3
        frames = batch.numpy() if hasattr(batch, "numpy") else batch.asnumpy()
        for i, idx in enumerate(indices):
            frame = frames[i].astype(np.float64)
            if idx < masks.shape[0]:
                mask = masks[idx] > 0
                frame[mask] = frame[mask] * (1 - alpha) + color * alpha
            vis_frames.append(frame.clip(0, 255).astype(np.uint8))
    del vr

    # Pad to even dimensions for H.264
    h, w = vis_frames[0].shape[:2]
    pad_h, pad_w = h % 2, w % 2
    if pad_h or pad_w:
        vis_frames = [np.pad(f, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant") for f in vis_frames]

    iio.imwrite(os.path.join(sam3_dir, "vis.mp4"), np.stack(vis_frames), fps=fps, codec="libx264")


def main():
    parser = argparse.ArgumentParser(prog="python -m gtf.extraction.frontend.run_sam3", description="Run SAM3 segmentation with Molmo2 point prompts")
    parser.add_argument("--max-videos", type=int, default=0, help="Process only first N segments (0 = all)")
    parser.add_argument("--force", action="store_true", help="Force recompute even if outputs exist")
    parser.add_argument("--visualize", action="store_true", help="Generate overlay visualization videos")
    parser.add_argument("--visualize-only", action="store_true", help="Only generate visualizations, skip inference")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to local SAM3 checkpoint")
    parser.add_argument("--object", type=str, default=None, help="Only process segments validated for this object (auto-detected if omitted)")
    args = parser.parse_args()
    if args.visualize_only:
        args.visualize = True

    entries = load_manifest()
    if args.max_videos > 0:
        entries = entries[: args.max_videos]
        print(f"Limited to first {args.max_videos} segments")

    names = [os.path.splitext(e["file"])[0] for e in entries]

    total = len(names)
    names = [n for n in names if is_action_valid(n, args.object)]
    _label = f" ({args.object})" if args.object else ""
    print(f"Action validation filter{_label}: {len(names)}/{total} valid")

    predictor = None if args.visualize_only else build_predictor(checkpoint_path=args.checkpoint)

    N = len(names)
    done = skipped = failed = no_points = 0
    vis_done = vis_skipped = 0

    for vi, name in enumerate(names):
        seg_dir = os.path.join(RESULTS_DIR, name)
        sam3_dir = os.path.join(seg_dir, "SAM3")
        video_path = os.path.join(seg_dir, "video.mp4")

        if not os.path.exists(video_path):
            print(f"[{vi + 1}/{N}] SKIP (no video): {name}")
            failed += 1
            continue

        # Inference
        if not args.visualize_only:
            molmo_path = os.path.join(seg_dir, "molmo2", "points.json")
            if not args.force and outputs_complete(sam3_dir, video_path, molmo_path):
                skipped += 1
            elif not os.path.exists(molmo_path):
                print(f"[{vi + 1}/{N}] SKIP (no molmo2 points): {name}")
                no_points += 1
            else:
                with open(molmo_path) as f:
                    molmo_check = json.load(f)
                if not molmo_check.get("points"):
                    print(f"[{vi + 1}/{N}] SKIP (empty points list): {name}")
                    no_points += 1
                    continue

                print(f"\n[{vi + 1}/{N}] Processing: {name}")
                try:
                    (Path(sam3_dir) / MARKER_FILE).unlink(missing_ok=True)
                    if process_segment(predictor, seg_dir, sam3_dir):
                        write_done_marker(sam3_dir)
                        done += 1
                        print(f"  Done: {name}")
                    else:
                        failed += 1
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"  ERROR processing {name}: {e}")
                    print(traceback.format_exc())
                    failed += 1
                    continue

        # Visualization
        if args.visualize:
            vis_path = os.path.join(sam3_dir, "vis.mp4")
            if not args.force and os.path.exists(vis_path) and os.path.getsize(vis_path) > 0:
                vis_skipped += 1
            elif os.path.exists(os.path.join(sam3_dir, "masks.npz")):
                print(f"  Rendering visualization: {name}")
                try:
                    render_vis_video(video_path, sam3_dir)
                    vis_done += 1
                    print(f"  Saved: {vis_path}")
                except Exception as e:
                    print(f"  ERROR rendering vis for {name}: {e}")
                    print(traceback.format_exc())

    if not args.visualize_only:
        print(f"\n[SAM3] Inference: {done} processed, {skipped} skipped, {no_points} no points, {failed} failed")
    if args.visualize:
        print(f"[SAM3] Visualization: {vis_done} rendered, {vis_skipped} skipped")


if __name__ == "__main__":
    main()
