#!/usr/bin/env python3
"""6D rigid object pose tracking pipeline.

Tracks dense 2D points on the object across the video, lifts them to 3D
via metric depth (MoGe v2 / MetricAnything), estimates camera poses, and
jointly optimizes object shape + per-frame 6D pose via bundle adjustment.

Pipeline stages:
  1. MoGe v2 depth estimation → metric depth + intrinsics per frame
  2. CoTracker3 dense tracking → 2D tracks + visibility on object
  3. Camera pose estimation   → per-frame cam-to-world (MegaSaM or PnP)
  4. Object bundle adjustment → joint {P_local, T_obj_t} optimization
                                (the upgraded chain adds a depth-relief term, object_pose_relief.py)
  5. Visualization            → 4-panel video (optional, --visualize)

Usage:
    uv run python -m gtf.extraction.object_pose.run_object_pose
    uv run python -m gtf.extraction.object_pose.run_object_pose --segment "name"
    uv run python -m gtf.extraction.object_pose.run_object_pose --segment "name" --chain upgraded
    uv run python -m gtf.extraction.object_pose.run_object_pose --max-videos 2 --no-megasam
    uv run python -m gtf.extraction.object_pose.run_object_pose --force
    uv run python -m gtf.extraction.object_pose.run_object_pose --visualize-only
"""

import argparse
import json
import os
import traceback
from pathlib import Path

import torch

from gtf.utils import is_action_valid
from gtf.utils import rprint as print

from gtf.extraction.object_pose.pose_cache import ROOT
from gtf.extraction.object_pose.pose_cache import ObjectPoseTerminalReject, _object_pose_done_matches  # noqa: F401  (imported from here by the orchestrator)
from gtf.extraction.object_pose.object_depth import DepthEstimator
from gtf.extraction.object_pose.point_tracking import clear_cotracker_cache, offload_cotracker_cache  # noqa: F401  (imported from here by the orchestrator)
from gtf.extraction.object_pose.camera_poses import run_camera_poses  # noqa: F401  (imported from here by export_windows.py)
from gtf.extraction.object_pose.pose_resume import object_pose_metadata_request_matches
from gtf.extraction.object_pose.pose_pipeline import DEFAULT_TRACKING_MODE, process_segment

SEGMENTS_DIR = os.path.join(ROOT, "segments")
RESULTS_DIR = os.path.join(ROOT, "results")


def main():
    parser = argparse.ArgumentParser(prog="python -m gtf.extraction.object_pose.run_object_pose", description="6D rigid object pose tracking pipeline")
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Process only first N segments (0 = all)",
    )
    parser.add_argument("--segment", type=str, default=None, help="Process only this segment name")
    parser.add_argument("--force", action="store_true", help="Force recompute even if outputs exist")
    parser.add_argument("--no-megasam", action="store_true", help="Force PnP fallback for camera poses")
    parser.add_argument(
        "--require-megasam",
        action="store_true",
        help="Fail instead of silently falling back to PnP when MegaSaM is unavailable",
    )
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Only re-render visualization from cached results",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Render object_pose/vis_pose.mp4 (disabled by default)",
    )
    parser.add_argument(
        "--cotracker-grid-size",
        type=int,
        default=30,
        help="Grid size for CoTracker3 point seeding (default: 30)",
    )
    parser.add_argument(
        "--tracking-mode",
        type=str,
        default=DEFAULT_TRACKING_MODE,
        choices=["full_bidir", "crop_bidir", "hybrid"],
        help="Object tracking mode (default: hybrid)",
    )
    parser.add_argument(
        "--object",
        type=str,
        default=None,
        help="Only process segments validated for this object (auto-detected if omitted)",
    )
    from gtf.extraction import chains

    chains.add_argument(parser)
    args = parser.parse_args()
    chain = chains.get(args.chain)

    manifest_path = os.path.join(SEGMENTS_DIR, "manifest.json")
    with open(manifest_path) as f:
        entries = json.load(f)["succeeded"]
    print(f"Loaded {len(entries)} succeeded segments from manifest")

    if args.max_videos > 0:
        entries = entries[: args.max_videos]
        print(f"Limited to first {args.max_videos} segments")

    names = [os.path.splitext(e["file"])[0] for e in entries]
    total = len(names)
    valid_names = [n for n in names if is_action_valid(n, args.object)]
    _label = f" ({args.object})" if args.object else ""
    print(f"Action validation filter{_label}: {len(valid_names)}/{total} valid")
    names = valid_names

    if args.segment:
        matched = [n for n in names if n == args.segment]
        if not matched:
            matched = [os.path.splitext(e["file"])[0] for e in entries if args.segment in os.path.splitext(e["file"])[0]]
            if not matched:
                print(f"Segment '{args.segment}' not found in manifest")
                return
        names = matched
        print(f"Processing segment: {names[0]}")

    depth_estimator = None
    if not args.visualize_only:
        depth_estimator = DepthEstimator()

    N = len(names)
    done = skipped = failed = 0

    for vi, name in enumerate(names):
        seg_dir = os.path.join(RESULTS_DIR, name)
        pose_dir = os.path.join(seg_dir, "object_pose")
        video_path = os.path.join(seg_dir, "video.mp4")

        if not os.path.exists(video_path):
            print(f"[{vi + 1}/{N}] SKIP (no video): {name}")
            failed += 1
            continue

        final_marker = os.path.join(pose_dir, ".object_pose_done")
        if not args.force and not args.visualize_only and os.path.exists(final_marker):
            metadata_path = os.path.join(pose_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path) as f:
                    meta = json.load(f)
                if object_pose_metadata_request_matches(
                    seg_dir,
                    pose_dir,
                    meta,
                    tracking_mode=args.tracking_mode,
                    cotracker_grid_size=args.cotracker_grid_size,
                    use_megasam=not args.no_megasam,
                    require_megasam=args.require_megasam,
                ) and (not args.visualize or (Path(pose_dir) / "vis_pose.mp4").is_file()):
                    skipped += 1
                    continue

        print(f"\n[{vi + 1}/{N}] Processing: {name}")
        try:
            chains.claim(seg_dir, chain)
            ok = process_segment(
                depth_estimator,
                name,
                seg_dir,
                pose_dir,
                force=args.force,
                use_megasam=not args.no_megasam,
                cotracker_grid_size=args.cotracker_grid_size,
                tracking_mode=args.tracking_mode,
                visualize_only=args.visualize_only,
                visualize=args.visualize,
                require_megasam=args.require_megasam,
                object_ba=chain.object_ba,
            )
            if ok:
                done += 1
                print(f"  Done: {name}")
            else:
                failed += 1
        except Exception as e:
            print(f"  ERROR processing {name}: {e}")
            print(traceback.format_exc())
            failed += 1

        torch.cuda.empty_cache()

    print(f"\n[Object Pose] {done} processed, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
