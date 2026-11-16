"""Pipeline steps hands (WiLoR), molmo2 (pointing), sam3 (masks), the early exit, sam3d and its temporal fallback sam3dc."""

import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import torch
from gtf.utils import get_segment_object, is_action_valid, reject_action, rprint as print
from gtf.extraction.action100m.stage_support import (
    RESULTS_DIR,
    _drain_stage_queue,
    _raise_queued_stage_failure,
    _stage_work_items,
    _stale_vs_upstream,
)


# ---------------------------------------------------------------------------
# hands: WiLoR hand tracking
# ---------------------------------------------------------------------------


def run_hands_step(segment_names, args):
    """Run the hand stage (WiLoR) on validated segments, then reject the segments without hands.

    It writes the WiLoR output to `hands/` (`hands/{side}.pt|.npz`, `hands/det.pkl`, `hands/.done`),
    the format every later stage reads.
    """
    from gtf.extraction.hands.run_wilor_hands import run_wilor_hands_step

    run_wilor_hands_step(segment_names, args)
    reject_handless_segments(segment_names, args)


def reject_handless_segments(segment_names, args):
    """Reject segments whose hand stage kept no hand at all.

    Such a segment can never produce a canonical (the optimizer needs valid hand
    observations and every interaction gate needs hands), so it is rejected here through
    the valid_action mechanism instead of running the sam3d / object_pose / optimize stages.
    Rejects only on positive evidence of an empty hand stage (a WiLoR report with no kept
    side); a crashed or missing hand stage is retried instead.
    """
    n_nohands = 0
    for name in segment_names:
        seg_dir = RESULTS_DIR / name
        if not is_action_valid(name, args.object):
            continue
        rep_path = seg_dir / "hands" / "wilor_report.json"
        if not rep_path.exists():
            continue
        rep = json.loads(rep_path.read_text()).get("report", {})
        if any(rep.get(s, {}).get("kept") for s in ("left", "right")):
            continue
        reject_action(
            name,
            "no_hands: hand stage kept zero sides (early-exit before the GPU tail)",
        )
        print(f"  [hands] {name}: no tracked hands — REJECTED (skipping molmo2/sam3/sam3d/object_pose/optimize)")
        n_nohands += 1
    if n_nohands:
        print(f"  [hands] early-exited {n_nohands} handless segment(s)")


# ---------------------------------------------------------------------------
# molmo2: object pointing
# ---------------------------------------------------------------------------


def run_molmo2_step(segment_names, args):
    """Run Molmo2 object pointing on validated segments; a segment with no accepted point on any candidate frame is rejected."""
    from gtf.extraction.frontend.run_molmo2 import load_model, point_segment
    from gtf.extraction.frontend.run_molmo2 import outputs_complete as molmo2_complete

    valid_names = [n for n in segment_names if is_action_valid(n, args.object)]

    def _molmo2_complete(name):
        return molmo2_complete(
            RESULTS_DIR / name / "molmo2",
            RESULTS_DIR / name / "video.mp4",
        )

    if not valid_names:
        print("  No validated segments for Molmo2")
        return
    if not args.force and all(_molmo2_complete(n) for n in valid_names):
        print(f"  [Molmo2] all {len(valid_names)} segments already pointed — model load skipped")
        return

    processor, model = load_model()
    done = skipped = failed = rejected = no_object = 0

    for vi, name in enumerate(valid_names):
        obj = args.object or get_segment_object(name)
        if not obj:
            no_object += 1
            continue

        if not args.force and _molmo2_complete(name):
            skipped += 1
            continue

        print(f"  [{vi + 1}/{len(valid_names)}] Molmo2: {name} (object={obj})")
        try:
            status = point_segment(
                processor,
                model,
                name,
                obj,
                num_points=args.num_points,
                temperature=args.temperature,
            )
            if status == "done":
                done += 1
            elif status == "failed":
                reason = "molmo2_no_valid_grounding: exhausted candidate frames without an accepted object point"
                reject_action(name, reason, obj)
                print(f"    TERMINAL REJECT: {reason}")
                rejected += 1
            else:
                failed += 1
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"    ERROR: {e}")
            traceback.print_exc()
            failed += 1

    del processor, model
    torch.cuda.empty_cache()
    print(f"  [Molmo2] {done} done, {skipped} skipped, {rejected} rejected, {failed} failed")
    if no_object:
        print(f"  [Molmo2] {no_object} skipped (no object name)")


# ---------------------------------------------------------------------------
# sam3: video object masks
# ---------------------------------------------------------------------------


def run_sam3_step(segment_names, args):
    """Run SAM3 video segmentation of the pointed object (SAM3/masks.npz + .sam3_done)."""
    from gtf.extraction.frontend.run_sam3 import build_predictor, process_segment, write_done_marker
    from gtf.extraction.frontend.run_sam3 import outputs_complete as sam3_complete

    valid_names = [n for n in segment_names if is_action_valid(n, args.object)]

    def _sam3_stale(n):
        return _stale_vs_upstream(
            RESULTS_DIR / n / "SAM3" / "masks.npz",
            RESULTS_DIR / n / "video.mp4",
            RESULTS_DIR / n / "molmo2" / "points.json",
        )

    def _sam3_complete(n):
        return sam3_complete(
            str(RESULTS_DIR / n / "SAM3"),
            RESULTS_DIR / n / "video.mp4",
            RESULTS_DIR / n / "molmo2" / "points.json",
        )

    missing_upstream = [n for n in valid_names if not (RESULTS_DIR / n / "molmo2" / ".molmo2_done").exists()]
    queued = bool(getattr(args, "stage_queue_db", None))
    all_complete = all(not (RESULTS_DIR / n / "molmo2" / ".molmo2_done").exists() or (_sam3_complete(n) and not _sam3_stale(n)) for n in valid_names)
    # A standalone run skips segments without Molmo2 output. In queue mode a worker that claims
    # such a segment withholds its SAM3 marker and exits with the claim unacknowledged, so the
    # split driver's retry worker takes it; the other workers continue with other segments.
    if queued and missing_upstream:
        all_complete = False

    if not args.force and all_complete:
        print(f"  [SAM3] all {len(valid_names)} segments already segmented (or lack molmo2) — model load skipped")
        _drain_stage_queue("sam3", args)
        return
    predictor = build_predictor()
    done = skipped = failed = 0

    for vi, name in enumerate(_stage_work_items("sam3", valid_names, args)):
        seg_dir = str(RESULTS_DIR / name)
        sam3_dir = str(RESULTS_DIR / name / "SAM3")
        molmo_done = (RESULTS_DIR / name / "molmo2" / ".molmo2_done").exists()

        if not molmo_done:
            Path(sam3_dir, ".sam3_done").unlink(missing_ok=True)
            failed += 1
            _raise_queued_stage_failure(args, "sam3", name, "Molmo2 completion marker missing")
            continue

        if not args.force and _sam3_complete(name) and not _sam3_stale(name):
            skipped += 1
            continue

        print(f"  [{vi + 1}/{len(valid_names)}] SAM3: {name}")
        try:
            Path(sam3_dir, ".sam3_done").unlink(missing_ok=True)
            ok = process_segment(predictor, seg_dir, sam3_dir)
            if ok:
                write_done_marker(sam3_dir)
        except Exception as e:
            print(f"    ERROR: {e}")
            traceback.print_exc()
            failed += 1
            _raise_queued_stage_failure(args, "sam3", name, e)
            continue
        if ok:
            done += 1
        else:
            failed += 1
            _raise_queued_stage_failure(args, "sam3", name, "predictor returned no complete output")

    del predictor
    torch.cuda.empty_cache()
    print(f"  [SAM3] {done} done, {skipped} skipped, {failed} failed")


# ---------------------------------------------------------------------------
# Optional early exit (--early-exit), between SAM3 and SAM3D
# ---------------------------------------------------------------------------


def run_earlyexit_step(segment_names, args):
    """Reject segments whose tracked hand never comes near the SAM3 object mask (`--early-exit`).

    Runs after sam3 and before the expensive sam3d / object_pose / optimize stages. Such a
    segment cannot pass the interaction / interaction_sustain gates; the proxy thresholds
    (gtf/extraction/frontend/interaction_proxy.py) sit below the values of segments that pass
    the gates. Missing inputs, few hand frames and already-extracted segments are never
    rejected."""
    from gtf.extraction.frontend.interaction_proxy import should_early_exit

    def check_one(name):
        seg_dir = RESULTS_DIR / name
        if not is_action_valid(name, args.object):
            return None
        # never reject a segment whose optimizer already ran (re-runs / resumes)
        if (seg_dir / "optimized_mano" / "optimized.npz").exists():
            return None
        reject, prox = should_early_exit(seg_dir)
        return name, reject, prox

    workers = min(max(1, int(getattr(args, "early_exit_workers", 10))), len(segment_names))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(check_one, segment_names))
    else:
        rows = [check_one(name) for name in segment_names]
    checked = [row for row in rows if row is not None]
    n_checked = len(checked)
    n_rej = 0
    # Write rejections and logs in segment-list order: only the read-only proxy runs in
    # parallel, so the rejections are written deterministically.
    for name, reject, prox in checked:
        if not reject:
            continue
        reject_action(
            name,
            f"early_exit_proxy: hand never near object mask (prox_rate={prox['prox_rate']:.2f}, prox_run_s={prox['prox_run_s']:.1f})",
        )
        print(f"  [early-exit] {name}: prox_rate={prox['prox_rate']:.2f} run={prox['prox_run_s']:.1f}s — REJECTED pre-tail")
        n_rej += 1
    if n_checked:
        print(f"  [early-exit] {n_rej}/{n_checked} pre-tail segments rejected by the interaction proxy")


# ---------------------------------------------------------------------------
# sam3d: object reconstruction and keyframe poses
# ---------------------------------------------------------------------------


def run_sam3d_step(segment_names, args):
    """Run SAM3D: object mesh reconstruction and per-keyframe 6DoF pose (sam3d/ + .sam3d_done)."""
    from gtf.extraction.sam3d.run_sam3d import build_sam3d_inference, process_segment, write_done_marker
    from gtf.extraction.sam3d.run_sam3d import outputs_complete as sam3d_complete

    valid_names = [n for n in segment_names if is_action_valid(n, args.object)]

    def _sam3d_stale(n):
        return _stale_vs_upstream(
            RESULTS_DIR / n / "sam3d" / ".sam3d_done",
            RESULTS_DIR / n / "SAM3" / "masks.npz",
        )

    # Skip the model load when every segment is already done.
    if not args.force and all(
        not (RESULTS_DIR / n / "SAM3" / ".sam3_done").exists()
        or (
            not _sam3d_stale(n)
            and sam3d_complete(
                str(RESULTS_DIR / n / "sam3d"),
                RESULTS_DIR / n / "video.mp4",
                RESULTS_DIR / n / "SAM3" / "masks.npz",
            )
        )
        for n in valid_names
    ):
        print(f"  [SAM3D] all {len(valid_names)} segments already done (or lack SAM3) — model load skipped")
        _drain_stage_queue("sam3d", args)
        return
    inference_obj = build_sam3d_inference()
    done = skipped = rejected = failed = 0

    for vi, name in enumerate(_stage_work_items("sam3d", valid_names, args)):
        seg_dir = str(RESULTS_DIR / name)
        sam3d_dir = str(RESULTS_DIR / name / "sam3d")

        if not (RESULTS_DIR / name / "SAM3" / ".sam3_done").exists():
            failed += 1
            continue

        if (
            not args.force
            and not _sam3d_stale(name)
            and sam3d_complete(
                sam3d_dir,
                RESULTS_DIR / name / "video.mp4",
                RESULTS_DIR / name / "SAM3" / "masks.npz",
            )
        ):
            skipped += 1
            continue

        print(f"  [{vi + 1}/{len(valid_names)}] SAM3D: {name}")
        try:
            Path(sam3d_dir, ".sam3d_done").unlink(missing_ok=True)
            # Earlier stages in the same process (WiLoR / Molmo2 / SAM3) can leave a global bf16
            # autocast enabled, which breaks SAM3D decoding ("unsupported ScalarType BFloat16").
            with torch.amp.autocast("cuda", enabled=False):
                n_ok, n_fail = process_segment(
                    inference_obj,
                    name,
                    seg_dir,
                    sam3d_dir,
                    select_fps=args.select_fps,
                    seed=42,
                    visualize=False,
                    pose_batch_size=max(1, args.sam3d_pose_batch_size),
                )
            if n_ok > 0:
                # The done marker the optimize step waits for: written only when no keyframe
                # failed (process_segment itself never writes it)
                if n_fail == 0:
                    write_done_marker(sam3d_dir)
                    done += 1
                else:
                    print(f"    PARTIAL: {n_ok} ok, {n_fail} failed — .sam3d_done NOT written")
                    failed += 1
                    _raise_queued_stage_failure(
                        args,
                        "sam3d",
                        name,
                        f"partial reconstruction ({n_ok} ok, {n_fail} failed)",
                    )
            elif n_fail == 0:
                # (0, 0) means the SAM3 masks contain no usable object keyframe (for
                # example every mask is empty): re-running cannot change that, so the
                # segment is rejected.
                reason = "sam3d_no_valid_keyframes: all current SAM3 masks are empty"
                reject_action(name, reason, args.object)
                print(f"    TERMINAL REJECT: {reason}")
                rejected += 1
            else:
                failed += 1
                _raise_queued_stage_failure(
                    args,
                    "sam3d",
                    name,
                    f"all {n_fail} attempted keyframes failed reconstruction",
                )
        except Exception as e:
            print(f"    ERROR: {e}")
            traceback.print_exc()
            failed += 1
            _raise_queued_stage_failure(args, "sam3d", name, e)

    del inference_obj
    torch.cuda.empty_cache()
    print(f"  [SAM3D] {done} done, {skipped} skipped, {rejected} rejected, {failed} failed")


# ---------------------------------------------------------------------------
# sam3dc: temporal SAM3D fallback (segments whose bundle adjustment is rejected)
# ---------------------------------------------------------------------------


def _ba_fallback_needed(seg_dir):
    """True when the optimizer's BA rotation chain is unavailable or rejected, the condition under
    which load_segment uses sam3d_c (kept in sync with optimizer_common._load_ba_camera_poses:
    done marker, npz, and a NaN-median reprojection error <= 12 px over evidence frames; NaN
    entries are frames without evidence)."""
    import numpy as np

    op = seg_dir / "object_pose"
    if not (op / ".object_pose_done").exists() or not (op / "object_poses.npz").exists():
        return True
    try:
        reproj = np.load(op / "object_poses.npz")["reprojection_errors"]
    except Exception:
        return True
    if not np.isfinite(reproj).any():
        return True
    return float(np.nanmedian(reproj)) > 12.0


def run_sam3dc_step(segment_names, args):
    """Generate sam3d_c (temporal dual initialization, rotation-aware winner selection) for
    segments whose BA chain is unavailable or rejected: a rotation fallback close to the BA
    chain, much smoother than the raw per-keyframe SAM3D poses. The optimizer's load_segment
    uses sam3d_c automatically when the BA is rejected."""
    from gtf.extraction.sam3d import run_sam3d_c as rc

    valid_names = [n for n in segment_names if is_action_valid(n, args.object)]

    def _todo(n):
        seg_dir = RESULTS_DIR / n
        if not (seg_dir / "sam3d" / ".sam3d_done").exists() or not (seg_dir / "SAM3" / "masks.npz").exists():
            return False
        c_dir = seg_dir / "sam3d_c"
        if not args.force and rc.outputs_complete_c(c_dir) and not _stale_vs_upstream(c_dir / ".sam3d_c_done", seg_dir / "SAM3" / "masks.npz"):
            return False
        return _ba_fallback_needed(seg_dir)

    if not args.force and not any(_todo(n) for n in valid_names):
        print(f"  [sam3d_c] no BA-rejected segments need the fallback ({len(valid_names)} checked) — model load skipped")
        _drain_stage_queue("sam3dc", args)
        return

    inference_obj = rc.build_sam3d_inference(checkpoint_dir=None, compile_model=False)
    done = skipped = failed = 0
    for vi, name in enumerate(_stage_work_items("sam3dc", valid_names, args)):
        if not _todo(name):
            skipped += 1
            continue
        seg_dir = str(RESULTS_DIR / name)
        c_dir = os.path.join(seg_dir, "sam3d_c")
        print(f"  [{vi + 1}/{len(valid_names)}] sam3d_c fallback: {name}")
        try:
            Path(c_dir, ".sam3d_c_done").unlink(missing_ok=True)
            # fp32 guard against a leaked bf16 autocast (see run_sam3d_step)
            with torch.amp.autocast("cuda", enabled=False):
                n_ok, n_fail = rc.process_segment_temporal(inference_obj, name.split("/")[-1], seg_dir, c_dir)
            if n_fail == 0 and n_ok > 0:
                rc.write_done_marker_c(c_dir)
                done += 1
            else:
                print(f"    sam3d_c incomplete (ok={n_ok} fail={n_fail}) — marker withheld, optimize keeps the raw fallback")
                failed += 1
                _raise_queued_stage_failure(
                    args,
                    "sam3dc",
                    name,
                    f"temporal fallback incomplete ({n_ok} ok, {n_fail} failed)",
                )
        except Exception as e:
            print(f"    ERROR: {e}")
            traceback.print_exc()
            failed += 1
            _raise_queued_stage_failure(args, "sam3dc", name, e)
        torch.cuda.empty_cache()

    del inference_obj
    torch.cuda.empty_cache()
    print(f"  [sam3d_c] {done} generated, {skipped} skipped, {failed} failed")
