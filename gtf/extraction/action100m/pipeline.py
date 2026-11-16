#!/usr/bin/env python3
"""Find Action100M segments and run the full extraction pipeline on them.

Searches the Action100M parquet annotations for object-manipulation segments, cuts them out of
locally stored source videos (--video-dir, one file per YouTube id) and runs the pipeline:

    find → fetch → validate → cutcheck → hands → molmo2 → sam3 → sam3d
         → object_pose → sam3dc → optimize → refine → export_canonical
         → quality_gate → reground

From the object-pose bundle adjustment onward a run uses one of two chains (--chain production|upgraded,
gtf/extraction/chains.py).

Object-agnostic: omit --object to run on all videos. `find` then discovers an object per
segment (an LLM over the captions), and validation and pointing are object-agnostic ("is an
object manipulated?"; pointing falls back to "the object held in the hand" for generic nouns).
Omit --object on every later run as well: a specific --object rejects segments discovered
under a different noun.

Usage:
    python -m gtf.extraction.action100m.pipeline --object cup --video-dir /data/yt   # one object (keyword search)
    python -m gtf.extraction.action100m.pipeline --object cup --limit 10 --video-dir /data/yt
    python -m gtf.extraction.action100m.pipeline --video-dir /data/yt                # all videos (per-segment discovery)
    python -m gtf.extraction.action100m.pipeline --start-from sam3d --force
    python -m gtf.extraction.action100m.pipeline --start-from optimize --segment my_segment_name
    ACTION100M_RESULTS_DIR=/abs/results_upgraded python -m gtf.extraction.action100m.pipeline --segment NAME --chain upgraded
"""

import argparse
import json
import shutil
from pathlib import Path

from gtf.utils import rprint as print
from gtf.utils.cli_settings import Setting, add_settings, apply_settings
from gtf.extraction import chains
from gtf.paths import REPO_ROOT
from gtf.extraction.action100m.stage_support import RESULTS_DIR, SEGMENTS_DIR
from gtf.extraction.action100m.steps_intake import run_cutcheck_step, run_fetch_step, run_find_step, run_validate_step
from gtf.extraction.action100m.steps_perception import run_earlyexit_step, run_hands_step, run_molmo2_step, run_sam3_step, run_sam3d_step, run_sam3dc_step
from gtf.extraction.action100m.steps_object_pose import run_object_pose_step
from gtf.extraction.action100m.steps_fit import run_export_canonical_step, run_optimize_step, run_quality_gate_step, run_refine_step, run_reground_step

DATA_DIR = REPO_ROOT / "datasets" / "action100m"

PIPELINE_STEPS = [
    "find",
    "fetch",
    "validate",
    "cutcheck",
    "hands",
    "molmo2",
    "sam3",
    "sam3d",
    "object_pose",
    "sam3dc",
    "optimize",
    "refine",
    "export_canonical",
    "quality_gate",
    "reground",
]


# ---------------------------------------------------------------------------
# Segment discovery
# ---------------------------------------------------------------------------


def get_segment_names_from_manifest():
    """Load segment names from the fetch manifest (SEGMENTS_DIR/manifest.json)."""
    manifest_path = SEGMENTS_DIR / "manifest.json"
    if not manifest_path.exists():
        return []
    with open(manifest_path) as f:
        manifest = json.load(f)
    return [Path(e["file"]).stem for e in manifest["succeeded"]]


def get_segment_names_from_results():
    """Discover segment directories in the results directory, excluding multi-window export
    children: {seg}__iwN directories are symlink farms owned by their parent's export step, and
    running upstream steps on them would write through the symlinks into the parent."""
    if not RESULTS_DIR.exists():
        return []
    return sorted(
        d.name for d in RESULTS_DIR.iterdir() if d.is_dir() and not d.name.startswith("results_") and not d.name.startswith("quality") and "__iw" not in d.name
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# Settings without a flag of their own; change one with --set KEY=VALUE (gtf/utils/cli_settings.py).
SETTINGS = {
    "sam3d_pose_batch_size": Setting(8, int, "SAM3D pose-only Stage 1 batch size"),
    "all_fields": Setting(False, bool, "Find (keyword mode): search all text fields, not just the action fields"),
    "min_level": Setting(3, int, "Find (discovery mode): minimum node level"),
    "action_filter": Setting(None, str, "Filter by gpt_action_brief (substring)"),
    "top_actions": Setting(None, int, "Keep only segments from the N most frequent actions"),
    "min_duration": Setting(1.0, float, "Min segment duration in seconds"),
    "max_duration": Setting(None, float, "Max segment duration in seconds"),
    "unique_videos": Setting(False, bool, "Keep only first segment per video"),
    "limit_per_action": Setting(None, int, "Max segments per unique action"),
    "shuffle": Setting(False, bool, "Randomize order before dedup/limit"),
    "seed": Setting(None, int, "Seed for --shuffle"),
    "temperature": Setting(0.7, float, "Molmo2 sampling temperature"),
    "num_points": Setting(5, int, "Molmo2: unique points to collect"),
    "select_fps": Setting(10.0, float, "SAM3D: keyframe selection density"),
    "object_pose_visualize": Setting(False, bool, "Render the diagnostic object-pose video"),
    "cotracker_grid_size": Setting(30, int, "Object pose: CoTracker grid size"),
    "tracking_mode": Setting(None, str, "Object pose: tracking mode (full_bidir|crop_bidir|hybrid)"),
    "cut_min_keep_s": Setting(4.0, float, "Cut check: min usable pre-cut footage to truncate instead of reject"),
    "prepass_workers": Setting(5, int, "Concurrent CPU workers for cutcheck"),
    "early_exit_workers": Setting(10, int, "Concurrent read-only workers for the interaction early-exit proxy"),
    "n_iters": Setting(500, int, "Optimizer iterations; 500 selects the default phase schedule (160/150/75/250/100)"),
    "optimizer_visualize": Setting(False, bool, "Render optimized_mano/vis_panel.mp4"),
    "no_multishot": Setting(False, bool, "Do not harvest post-first-cut shots as {seg}__shotN child segments at cutcheck"),
}


def main():
    parser = argparse.ArgumentParser(prog="python -m gtf.extraction.action100m.pipeline", description="Load Action100M segments and run the full extraction pipeline")

    parser.add_argument(
        "--object",
        type=str,
        default=None,
        help="Target object (if omitted, uses LLM auto-discovery for find step, auto-detected from metadata for later steps)",
    )

    # Pipeline control
    parser.add_argument("--force", action="store_true", help="Recompute all steps even if outputs exist")
    parser.add_argument(
        "--start-from",
        choices=PIPELINE_STEPS,
        default="find",
        help="Start pipeline from this step (default: find)",
    )
    parser.add_argument(
        "--stop-after",
        choices=PIPELINE_STEPS,
        default=None,
        help="Stop the pipeline after this step (inclusive). The GPU/CPU split driver (scripts/run_split_extraction.sh) "
        "runs GPU stages with '--stop-after refine' and a CPU pool with '--start-from export_canonical'.",
    )
    chains.add_argument(parser)
    parser.add_argument(
        "--segment",
        action="append",
        default=None,
        help="Process only specific segment(s) (repeatable; skips find/fetch)",
    )
    parser.add_argument(
        "--segment-file",
        type=Path,
        default=None,
        help="Process segment names from a newline-delimited file",
    )
    parser.add_argument(
        "--stage-queue-db",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--stage-worker-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--early-exit",
        action="store_true",
        help="After sam3, reject segments whose hand never nears the object mask (a conservative interaction proxy) "
        "before the expensive sam3d/object_pose/optimize stages",
    )
    parser.add_argument("--early-exit-only", action="store_true", help=argparse.SUPPRESS)

    # Find step
    parser.add_argument(
        "--parquet-dir",
        type=str,
        default=str(DATA_DIR),
        help="Directory containing Action100M parquet files",
    )

    # Fetch step
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="Directory of the Action100M source videos, one file per YouTube id ({video_uid}.mp4/.mkv/.webm); required by the fetch step",
    )
    parser.add_argument("--limit", type=int, default=None, help="Global max segments to fetch")

    # Per-step params
    parser.add_argument("--no-megasam", action="store_true", help="Object pose: estimate the cameras with PnP instead of MegaSaM")
    parser.add_argument(
        "--camera-require-megasam",
        action="store_true",
        help="Object pose: fail instead of falling back to PnP cameras when MegaSaM is unavailable",
    )
    parser.add_argument(
        "--optimizer-workers",
        type=int,
        default=4,
        help="Concurrent optimizer processes per GPU (1 runs the optimizer in this process)",
    )
    parser.add_argument(
        "--reground-seed",
        type=int,
        default=42,
        help="Deterministic base seed for fresh Molmo2 re-grounding evidence",
    )
    parser.add_argument(
        "--quality-level",
        choices=["loose", "moderate", "strict"],
        default="moderate",
        help="Quality gate threshold level",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device for GPU steps")

    add_settings(parser, SETTINGS)
    args = parser.parse_args()
    apply_settings(parser, args, SETTINGS)

    start_idx = PIPELINE_STEPS.index(args.start_from)
    stop_idx = PIPELINE_STEPS.index(args.stop_after) if args.stop_after else len(PIPELINE_STEPS) - 1
    if stop_idx < start_idx:
        parser.error(f"--stop-after {args.stop_after} precedes --start-from {args.start_from}")
    active_steps = set(PIPELINE_STEPS[start_idx : stop_idx + 1])

    if args.segment and args.segment_file:
        parser.error("--segment and --segment-file are mutually exclusive")

    # Explicit segments skip discovery/fetch.
    if args.segment or args.segment_file:
        active_steps -= {"find", "fetch"}
        if args.segment_file:
            segment_names = [line.strip() for line in args.segment_file.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]
        else:
            segment_names = list(args.segment)
        print(f"Explicit segment mode: {len(segment_names)} segment(s)")
    else:
        segment_names = None  # will be populated after find/fetch

    step_num = 0
    total_steps = len(active_steps)

    # --- Find ---
    results_path = None
    if "find" in active_steps:
        step_num += 1
        print("=" * 60)
        print(f"STEP {step_num}/{total_steps}: Find segments in Action100M")
        print("=" * 60)
        results_path = run_find_step(args)
        if results_path is None:
            return

    # --- Fetch ---
    if "fetch" in active_steps:
        step_num += 1
        print("\n" + "=" * 60)
        print(f"STEP {step_num}/{total_steps}: Cut segments from the source videos")
        print("=" * 60)
        run_fetch_step(args, results_path)

    # Determine segment names for downstream steps
    if segment_names is None:
        segment_names = get_segment_names_from_manifest()
        if not segment_names:
            segment_names = get_segment_names_from_results()
        print(f"\n{len(segment_names)} segments available for extraction pipeline")

    if not segment_names:
        print("No segments to process.")
        return

    # Stage the results directories from the fetch manifest before validation: copy each clip into
    # RESULTS_DIR/{name}/ and write metadata.json with the object (from --object or the discovered
    # noun), so validation can resolve the object when --object is omitted. Only fills what is
    # missing; an existing directory is never overwritten.
    manifest_entries = {}
    manifest_path = SEGMENTS_DIR / "manifest.json"
    if manifest_path.exists():
        try:
            for e in json.loads(manifest_path.read_text()).get("succeeded", []):
                manifest_entries[Path(e["file"]).stem] = e
        except (json.JSONDecodeError, OSError):
            pass
    for name in segment_names:
        seg_dir = RESULTS_DIR / name
        seg_dir.mkdir(parents=True, exist_ok=True)
        entry = manifest_entries.get(name, {})
        meta_path = seg_dir / "metadata.json"
        if not meta_path.exists():
            meta = dict(entry)
            eff_obj = args.object or entry.get("object")
            if eff_obj:
                meta["object"] = eff_obj
            meta_path.write_text(json.dumps(meta, indent=2))
        video_dst = seg_dir / "video.mp4"
        if not video_dst.exists():
            video_src = SEGMENTS_DIR / entry.get("file", f"{name}.mp4")
            if video_src.exists():
                shutil.copy2(video_src, video_dst)

    if args.early_exit_only:
        run_earlyexit_step(segment_names, args)
        return

    # --- Validate ---
    if "validate" in active_steps:
        step_num += 1
        print("\n" + "=" * 60)
        print(f"STEP {step_num}/{total_steps}: Validate with Molmo2")
        print("=" * 60)
        run_validate_step(segment_names, args)

    # --- Cut check ---
    if "cutcheck" in active_steps:
        step_num += 1
        print("\n" + "=" * 60)
        print(f"STEP {step_num}/{total_steps}: Scene-cut check")
        print("=" * 60)
        shot_children = run_cutcheck_step(segment_names, args)
        if shot_children:
            # Harvested shots are full segments: validate them here (the validate step
            # already ran), then fold them into the rest of the pipeline run.
            print(f"\n  Validating {len(shot_children)} harvested shot children...")
            run_validate_step(shot_children, args)
            segment_names = list(segment_names) + shot_children

    # --- Hands (WiLoR) ---
    if "hands" in active_steps:
        step_num += 1
        print("\n" + "=" * 60)
        print(f"STEP {step_num}/{total_steps}: Hand tracking (WiLoR)")
        print("=" * 60)
        run_hands_step(segment_names, args)

    # --- Molmo2 ---
    if "molmo2" in active_steps:
        step_num += 1
        print("\n" + "=" * 60)
        print(f"STEP {step_num}/{total_steps}: Molmo2 object pointing")
        print("=" * 60)
        run_molmo2_step(segment_names, args)

    # --- SAM3 ---
    if "sam3" in active_steps:
        step_num += 1
        print("\n" + "=" * 60)
        print(f"STEP {step_num}/{total_steps}: SAM3 video segmentation")
        print("=" * 60)
        run_sam3_step(segment_names, args)

    # --- Early-exit interaction proxy (opt-in) ---
    if args.early_exit and {"sam3d", "object_pose", "optimize"} & active_steps:
        run_earlyexit_step(segment_names, args)

    # --- SAM3D ---
    if "sam3d" in active_steps:
        step_num += 1
        print("\n" + "=" * 60)
        print(f"STEP {step_num}/{total_steps}: SAM3D 3D reconstruction")
        print("=" * 60)
        run_sam3d_step(segment_names, args)

    # --- Object Pose ---
    if "object_pose" in active_steps:
        step_num += 1
        print("\n" + "=" * 60)
        print(f"STEP {step_num}/{total_steps}: Object pose estimation")
        print("=" * 60)
        run_object_pose_step(segment_names, args)

    # --- sam3d_c fallback (BA-rejected segments only) ---
    if "sam3dc" in active_steps:
        step_num += 1
        print("\n" + "=" * 60)
        print(f"STEP {step_num}/{total_steps}: sam3d_c temporal fallback")
        print("=" * 60)
        run_sam3dc_step(segment_names, args)

    # --- Optimize ---
    if "optimize" in active_steps:
        step_num += 1
        print("\n" + "=" * 60)
        print(f"STEP {step_num}/{total_steps}: MANO optimization")
        print("=" * 60)
        run_optimize_step(segment_names, args)

    # --- Refine (stage 2) ---
    if "refine" in active_steps:
        step_num += 1
        print("\n" + "=" * 60)
        print(f"STEP {step_num}/{total_steps}: Stage-2 trajectory refinement")
        print("=" * 60)
        run_refine_step(segment_names, args)

    # --- Export Canonical ---
    if "export_canonical" in active_steps:
        step_num += 1
        print("\n" + "=" * 60)
        print(f"STEP {step_num}/{total_steps}: Canonical export")
        print("=" * 60)
        run_export_canonical_step(segment_names, args)

    # --- Quality Gate ---
    if "quality_gate" in active_steps:
        step_num += 1
        print("\n" + "=" * 60)
        print(f"STEP {step_num}/{total_steps}: Quality gate")
        print("=" * 60)
        run_quality_gate_step(segment_names, args)

    # --- Re-grounding verifier (gate 21) ---
    if "reground" in active_steps:
        step_num += 1
        print("\n" + "=" * 60)
        print(f"STEP {step_num}/{total_steps}: Re-grounding verifier")
        print("=" * 60)
        run_reground_step(segment_names, args)

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
