"""Pipeline steps optimize, refine, export_canonical, quality_gate and reground."""

import json
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
from gtf.paths import MANO_DIR
from gtf.utils import is_action_valid, reject_action, rprint as print
from gtf.extraction.action100m.stage_support import (
    RESULTS_DIR,
    _claim_chain,
    _cuda_context_is_unusable,
    _drain_stage_queue,
    _expand_declared_interaction_window_children,
    _optimized_output_complete,
    _quality_precondition_rejected,
    _raise_queued_stage_failure,
    _stage_work_items,
    _stale_vs_upstream,
    stage2_subdir,
)


# ---------------------------------------------------------------------------
# optimize: joint hand-object optimization
# ---------------------------------------------------------------------------

_HAND_MASK_PROCESSOR = None


def _ensure_hand_masks(name):
    """RGB hand masks (SAM 3 "hand" at the keyframes) for chains whose optimizer uses them; an existing file is kept."""
    global _HAND_MASK_PROCESSOR
    if (RESULTS_DIR / name / "hand_masks" / "masks.npz").exists():
        return
    from gtf.extraction.hands.run_hand_masks import segment_hand_masks

    if _HAND_MASK_PROCESSOR is None:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        _HAND_MASK_PROCESSOR = Sam3Processor(build_sam3_image_model(), confidence_threshold=0.4)
    n_kf, frac = segment_hand_masks(_HAND_MASK_PROCESSOR, str(RESULTS_DIR / name))
    print(f"    hand masks: {n_kf} keyframes, {100 * frac:.0f}% with a hand")


def run_optimize_step(segment_names, args):
    """Run the joint MANO hand-object optimization (optimized_mano/optimized.npz).

    Three modes: one segment at a time from the split driver's stage queue; several optimizer subprocesses per GPU
    (--optimizer-workers > 1); or in this process."""
    from gtf.extraction.optimizer.optimize_mano import OptimizerTerminalReject, process_one_segment

    valid_names = [n for n in segment_names if is_action_valid(n, args.object)]
    chain = _claim_chain(valid_names, args)
    config = {
        "n_iters": args.n_iters,
        "contact_threshold": 0.008,
        "min_separation": 0.002,
        "deep_pen_threshold": 0.005,
        "max_wrist_rot": 0.08,
        "max_wrist_trans": 0.05,
        "thumb_vertex_weight": 0.25,
        "grasp_side_margin": 0.005,
        "start_frame_max": 59,
        "carry_conf_low": 0.5,
        "carry_conf_high": 0.8,
        "visualize": getattr(args, "optimizer_visualize", False),
        **chain.optimizer,
    }
    done = skipped = failed = rejected = 0

    if getattr(args, "stage_queue_db", None):
        for vi, name in enumerate(_stage_work_items("optimize", valid_names, args)):
            out_path = RESULTS_DIR / name / "optimized_mano" / "optimized.npz"
            if not (RESULTS_DIR / name / "sam3d" / ".sam3d_done").exists():
                print(f"  Optimize: {name} SKIPPED - sam3d/.sam3d_done missing")
                failed += 1
                _raise_queued_stage_failure(args, "optimize", name, "sam3d/.sam3d_done missing")
                continue
            if (
                not args.force
                and _optimized_output_complete(out_path)
                and not _stale_vs_upstream(
                    out_path,
                    RESULTS_DIR / name / "sam3d" / ".sam3d_done",
                    RESULTS_DIR / name / "object_pose" / ".object_pose_done",
                )
            ):
                skipped += 1
                continue
            print(f"  [{vi + 1}] Optimize: {name}")
            try:
                out_path.unlink(missing_ok=True)
                if chain.hand_masks:
                    _ensure_hand_masks(name)
                # Earlier stages in the same process can leave a global bf16 autocast enabled;
                # the optimizer's torch.linalg.inv has no bf16 kernel, so force fp32.
                with torch.amp.autocast("cuda", enabled=False):
                    ok = process_one_segment(name, config, device=args.device)
                if str(args.device).startswith("cuda"):
                    # Surface asynchronous kernel failures while this queue item is unacknowledged.
                    torch.cuda.synchronize()
                if ok:
                    done += 1
                else:
                    failed += 1
                    _raise_queued_stage_failure(args, "optimize", name, "optimizer returned false")
            except OptimizerTerminalReject as error:
                reject_action(name, str(error), args.object)
                print(f"    TERMINAL REJECT: {error}")
                rejected += 1
            except Exception as error:
                print(f"    ERROR: {error}")
                traceback.print_exc()
                if _cuda_context_is_unusable(error):
                    print("    FATAL: CUDA context is unusable; exiting this worker so the stage queue can retry the segment in a fresh process")
                    raise
                failed += 1
                _raise_queued_stage_failure(args, "optimize", name, error)
        print(f"  [Optimize] {done} done, {skipped} skipped, {rejected} rejected, {failed} failed")
        return

    pending = []
    for vi, name in enumerate(valid_names):
        out_path = RESULTS_DIR / name / "optimized_mano" / "optimized.npz"

        if not (RESULTS_DIR / name / "sam3d" / ".sam3d_done").exists():
            print(f"  [{vi + 1}/{len(valid_names)}] Optimize: {name} SKIPPED — sam3d/.sam3d_done missing (run the sam3d step first)")
            failed += 1
            continue

        if (
            not args.force
            and _optimized_output_complete(out_path)
            and not _stale_vs_upstream(
                out_path,
                RESULTS_DIR / name / "sam3d" / ".sam3d_done",
                RESULTS_DIR / name / "object_pose" / ".object_pose_done",
            )
        ):
            skipped += 1
            continue

        pending.append((vi, name))

    if not pending:
        print(f"  [Optimize] {done} done, {skipped} skipped, {failed} failed")
        return

    optimizer_workers = max(1, int(getattr(args, "optimizer_workers", 4)))
    if optimizer_workers > 1 and len(pending) > 1:
        if chain.hand_masks:  # in the parent, one model, before the optimizer subprocesses take the GPU
            ready = []
            for vi, name in pending:
                try:
                    _ensure_hand_masks(name)
                    ready.append((vi, name))
                except Exception as error:
                    print(f"    ERROR: hand masks failed for {name}: {error}")
                    failed += 1
            pending = ready

        def run_subprocess(item):
            vi, name = item
            print(f"  [{vi + 1}/{len(valid_names)}] Optimize: {name}")
            output = RESULTS_DIR / name / "optimized_mano" / "optimized.npz"
            output.unlink(missing_ok=True)
            command = [
                sys.executable,
                "-u",
                "-m",
                "gtf.extraction.optimizer.optimize_mano",
                f"--segment={name}",
                "--force",
                "--n-iters",
                str(args.n_iters),
                "--device",
                args.device,
                "--chain",
                chain.name,
            ]
            if args.object:
                command.extend(["--object", args.object])
            if getattr(args, "optimizer_visualize", False):
                command.append("--visualize")
            result = subprocess.run(command)
            return name, result.returncode == 0 and _optimized_output_complete(output)

        print(f"  [Optimize] {len(pending)} segments with {min(optimizer_workers, len(pending))} workers on {args.device}")
        with ThreadPoolExecutor(max_workers=optimizer_workers) as executor:
            futures = {executor.submit(run_subprocess, item): item[1] for item in pending}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    _, ok = future.result()
                except Exception as error:
                    print(f"    ERROR: optimizer subprocess crashed for {name}: {error}")
                    failed += 1
                    continue
                if ok:
                    done += 1
                else:
                    print(f"    ERROR: optimizer subprocess failed for {name}")
                    failed += 1
        print(f"  [Optimize] {done} done, {skipped} skipped, {failed} failed")
        return

    for vi, name in pending:
        print(f"  [{vi + 1}/{len(valid_names)}] Optimize: {name}")
        try:
            (RESULTS_DIR / name / "optimized_mano" / "optimized.npz").unlink(missing_ok=True)
            if chain.hand_masks:
                _ensure_hand_masks(name)
            # fp32 guard against a leaked bf16 autocast (see the queue branch above)
            with torch.amp.autocast("cuda", enabled=False):
                ok = process_one_segment(name, config, device=args.device)
            if str(args.device).startswith("cuda"):
                torch.cuda.synchronize()
            if ok:
                done += 1
            else:
                failed += 1
        except OptimizerTerminalReject as e:
            reject_action(name, str(e), args.object)
            print(f"    TERMINAL REJECT: {e}")
            rejected += 1
        except Exception as e:
            print(f"    ERROR: {e}")
            traceback.print_exc()
            if _cuda_context_is_unusable(e):
                raise
            failed += 1

    print(f"  [Optimize] {done} done, {skipped} skipped, {rejected} rejected, {failed} failed")


# ---------------------------------------------------------------------------
# refine: stage-2 trajectory refinement
# ---------------------------------------------------------------------------


def run_refine_step(segment_names, args):
    """Stage-2 trajectory refinement into optimized_stage2/: hand-rigidity object-rotation bridge, learned
    per-segment translation bias (production chain), temporal and articulation filters, depth-from-size repair."""
    from gtf.extraction.stage2.refine_trajectory import (
        DEFAULT_CKPT,
        RefinePreconditionReject,
        load_denoiser,
        refine_one_segment,
    )

    from gtf.model.repr.mano import MANOHelper
    from gtf.utils.action100m_quality_precondition import (
        load_quality_precondition_reject,
        publish_quality_precondition_reject,
    )

    valid_names = [n for n in segment_names if is_action_valid(n, args.object)]
    if not valid_names:
        _drain_stage_queue("refine", args)
        print("  [Refine] no valid segments")
        return
    chain = _claim_chain(valid_names, args)
    device = torch.device(args.device)
    mano = MANOHelper(MANO_DIR).to(device)
    model = stats = None
    if chain.refine_bias:
        model, stats = load_denoiser(DEFAULT_CKPT, device)

    done = skipped = rejected = failed = 0
    for vi, name in enumerate(_stage_work_items("refine", valid_names, args)):
        existing_reject = load_quality_precondition_reject(
            RESULTS_DIR / name,
            expected_segment=name,
        )
        if existing_reject is not None:
            rejected += 1
            print(f"  [{vi + 1}/{len(valid_names)}] {name}: quality precondition already rejected ({existing_reject['reason']}, K={existing_reject['observed_keyframes']})")
            continue
        if not (RESULTS_DIR / name / "optimized_mano" / "optimized.npz").exists():
            failed += 1
            _raise_queued_stage_failure(args, "refine", name, "optimized_mano/optimized.npz missing")
            continue
        try:
            # A full run in one process can leak a global bf16 autocast from an earlier GPU stage;
            # refine's Procrustes / Karcher SVDs have no bf16 CUDA kernel, so force fp32.
            with torch.amp.autocast("cuda", enabled=False):
                r = refine_one_segment(
                    name,
                    mano=mano,
                    device=device,
                    model=model,
                    stats=stats,
                    force=args.force,
                )
            if r is None or "error" in r:
                failed += 1
                if r is not None:
                    print(f"    ERROR {name}: {r['error']}")
                _raise_queued_stage_failure(
                    args,
                    "refine",
                    name,
                    "refiner returned no result" if r is None else r["error"],
                )
            elif r.get("skipped"):
                skipped += 1
            else:
                done += 1
                hr, bi = r["handrigid"], r["bias"]
                bias_s = ", ".join(f"{e}: {sum(x * x for x in bi[e + '_mm']) ** 0.5:.0f}mm" for e in ("left", "right", "obj") if e + "_mm" in bi)
                print(f"  [{vi + 1}/{len(valid_names)}] {name}: hr={'on' if hr.get('applied') else 'off'} bias[{bias_s}]")
        except RefinePreconditionReject as e:
            record = publish_quality_precondition_reject(
                RESULTS_DIR / name,
                expected_segment=name,
            )
            rejected += 1
            print(
                f"    TERMINAL QUALITY PRECONDITION REJECT {name}: {record['reason']} ({record['observed_keyframes']}/{record['minimum_required_keyframes']} keyframes); {e}"
            )
        except Exception as e:
            print(f"    ERROR: {e}")
            traceback.print_exc()
            failed += 1
            _raise_queued_stage_failure(args, "refine", name, e)

    print(f"  [Refine] {done} done, {skipped} skipped, {rejected} precondition rejected, {failed} failed")


# ---------------------------------------------------------------------------
# export_canonical
# ---------------------------------------------------------------------------


def run_export_canonical_step(segment_names, args):
    """Export canonical/canonical.npz from the optimized trajectory (stage-2 refined when present).

    The export can also write `__iwN` interaction-window children; the chain decides the export options."""
    from gtf.extraction.export.export_canonical import outputs_complete, process_one_segment

    valid_names = [n for n in segment_names if is_action_valid(n, args.object) and not _quality_precondition_rejected(n)]
    chain = _claim_chain(valid_names, args)
    done = skipped = failed = 0

    for vi, name in enumerate(valid_names):
        opt_subdir = stage2_subdir(name)
        opt_path = RESULTS_DIR / name / opt_subdir / "optimized.npz"
        out_path = RESULTS_DIR / name / "canonical" / "canonical.npz"

        if not opt_path.exists():
            skipped += 1
            continue
        if not args.force and outputs_complete(RESULTS_DIR / name, expected_segment=name) and not _stale_vs_upstream(out_path, opt_path):
            skipped += 1
            continue

        print(f"  [{vi + 1}/{len(valid_names)}] Export canonical: {name} ({opt_subdir})")
        try:
            ok = process_one_segment(name, opt_subdir=opt_subdir, chain=chain)
            if ok:
                done += 1
            else:
                failed += 1
        except Exception as e:
            print(f"    ERROR: {e}")
            traceback.print_exc()
            failed += 1

    print(f"  [Export canonical] {done} done, {skipped} skipped, {failed} failed")


# ---------------------------------------------------------------------------
# quality_gate
# ---------------------------------------------------------------------------


def run_quality_gate_step(segment_names, args):
    """Run the quality metrics and gates (quality/quality.json; on the stage-2 refined trajectory when present).

    The gates are scoped to the exported window; the declared `__iwN` children are gated as segments of their own."""
    from gtf.extraction.gates.quality_gate import THRESHOLDS, process_one

    valid_names = [n for n in segment_names if is_action_valid(n, args.object) and not _quality_precondition_rejected(n)]
    # The `__iwN` interaction-window children are gated like any segment, but only the ones
    # the parent's canonical export declares.
    valid_names = _expand_declared_interaction_window_children(valid_names)
    thresh = {**THRESHOLDS[args.quality_level], "_level": args.quality_level}

    done = skipped = 0
    passed = rejected = 0

    for vi, name in enumerate(valid_names):
        opt_subdir = stage2_subdir(name)
        if not (RESULTS_DIR / name / opt_subdir / "optimized.npz").exists():
            skipped += 1
            continue

        r = process_one(name, thresh, force=args.force, opt_subdir=opt_subdir)
        if r is None:
            skipped += 1
            continue

        done += 1
        if r.get("pass", False):
            passed += 1
        else:
            rejected += 1

    print(f"  [Quality gate] {done} evaluated ({passed} passed, {rejected} rejected), {skipped} skipped")
    if getattr(args, "segment", None) and skipped:
        raise RuntimeError(f"the quality step skipped {skipped} explicitly requested segment(s)")


# ---------------------------------------------------------------------------
# reground: re-grounding verifier (gate 21)
# ---------------------------------------------------------------------------


def _patch_quality_with_reground(name, thresh):
    """Re-evaluate an existing quality.json with fresh reground metrics merged in, without
    recomputing the other metrics (the fallback when a full recompute is not possible)."""
    from gtf.extraction.gates.quality_gate import (
        _atomic_json,
        _sha256_file,
        evaluate_gates,
    )
    from gtf.extraction.gates.reground import load_reground_scores

    qp = RESULTS_DIR / name / "quality" / "quality.json"
    if not qp.exists():
        return None
    q = json.loads(qp.read_text())
    canonical_path = RESULTS_DIR / name / "canonical" / "canonical.npz"
    if not canonical_path.is_file():
        raise RuntimeError(f"cannot bind patched quality without canonical: {canonical_path}")
    canonical_sha = _sha256_file(canonical_path)
    with np.load(canonical_path, allow_pickle=False) as canonical:
        if "frame_indices" not in canonical.files:
            raise RuntimeError(f"cannot bind patched quality without frame_indices: {canonical_path}")
        n_frames = int(np.asarray(canonical["frame_indices"]).size)
    if q.get("canonical_sha256") != canonical_sha or q.get("segment") != name or q.get("n_frames") != n_frames or not isinstance(q.get("window_scoped"), bool):
        raise RuntimeError("cannot patch re-ground axes onto quality metrics that are not bound to the current canonical")
    q.update(load_reground_scores(RESULTS_DIR / name))
    q.pop("reground_evidence_status", None)
    q.pop("reground_support_frames", None)
    metrics = {k: v for k, v in q.items() if k not in ("pass", "failed_gates", "gates", "threshold_level", "opt_subdir")}
    gates, failed = evaluate_gates(metrics, thresh)
    q.update(
        {
            "pass": len(failed) == 0,
            "failed_gates": failed,
            "gates": gates,
            "threshold_level": thresh.get("_level", "custom"),
        }
    )
    reground_path = RESULTS_DIR / name / "reground" / "reground.json"
    q["canonical_sha256"] = canonical_sha
    q["reground_sha256"] = _sha256_file(reground_path) if reground_path.is_file() else None
    _atomic_json(qp, q)
    return q


def run_reground_step(segment_names, args):
    """Re-grounding verifier (gate 21): fresh Molmo2 evidence against the tracked mask for
    gate-passing segments only (a wrong tracked object is invisible to the other gates),
    then re-evaluate their quality gate.

    Re-evaluating quality runs on the CPU; collecting missing Molmo2 evidence needs a GPU
    (python -m gtf.extraction.action100m.pipeline --start-from reground)."""
    has_cuda = torch.cuda.is_available()
    from gtf.extraction.gates.quality_gate import THRESHOLDS, mark_reground_unverified, process_one
    from gtf.extraction.gates.reground import (
        collect_evidence,
        has_minimum_reground_support,
        materialized_quality_binds_current_reground,
        reground_evidence_needs_refresh,
        reground_support_frames,
    )

    valid_names = [n for n in segment_names if is_action_valid(n, args.object)]
    expanded = _expand_declared_interaction_window_children(valid_names)
    todo = []
    quality_retry = []
    for n in expanded:
        qp = RESULTS_DIR / n / "quality" / "quality.json"
        if not qp.exists():
            continue
        try:
            quality = json.loads(qp.read_text())
        except Exception:
            continue
        failed_gates = set(quality.get("failed_gates", []))
        if not quality.get("pass", False) and failed_gates != {"obj_reground"}:
            continue  # only fresh evidence can adjudicate a reground-only failure
        needs_evidence = reground_evidence_needs_refresh(RESULTS_DIR / n, force=args.force)
        if not needs_evidence:
            if not materialized_quality_binds_current_reground(RESULTS_DIR / n) or quality.get("reground_evidence_status") in {
                "collection_failed",
                "insufficient_support",
                "quality_recompute_failed",
            }:
                quality_retry.append(n)
            continue  # fresh, sufficiently-supported evidence is already bound
        todo.append(n)

    thresh = {**THRESHOLDS[args.quality_level], "_level": args.quality_level}

    def recompute_quality(name):
        result = process_one(
            name,
            thresh,
            force=True,
            opt_subdir=stage2_subdir(name),
        )
        if result is None:
            # Without the inputs of a full recompute, the stored metrics still take the fresh re-ground axes.
            result = _patch_quality_with_reground(name, thresh)
        if not isinstance(result, dict):
            raise RuntimeError("quality recompute returned no result")
        return result

    flipped = failed = 0
    for name in quality_retry:
        try:
            result = recompute_quality(name)
        except Exception as exc:  # noqa: BLE001
            print(f"    QUALITY-RETRY ERROR {name}: {exc}")
            mark_reground_unverified(
                name,
                status="quality_recompute_failed",
                results_dir=RESULTS_DIR,
            )
            failed += 1
            continue
        if not result.get("pass", False):
            flipped += 1

    if not todo:
        print("  [Reground] no gate-passing segments need evidence")
        if quality_retry:
            print(f"  [Reground] {len(quality_retry)} quality-only retries, {flipped} rejected, {failed} errors")
        if getattr(args, "segment", None) and failed:
            raise RuntimeError(f"the re-ground step failed for {failed} explicitly requested segment(s)")
        return

    if not has_cuda:
        print("  [Reground] no CUDA — quality-only retries completed; missing evidence deferred to a GPU pass")
        return

    from gtf.extraction.frontend.run_molmo2 import load_model

    processor, model = load_model()
    for vi, name in enumerate(todo):
        try:
            evidence = collect_evidence(
                processor,
                model,
                name,
                seed=getattr(args, "reground_seed", 42),
                results_dir=RESULTS_DIR,
            )
        except Exception as e:
            print(f"    ERROR {name}: {e}")
            mark_reground_unverified(
                name,
                status="collection_failed",
                results_dir=RESULTS_DIR,
            )
            failed += 1
            continue
        if not has_minimum_reground_support(evidence):
            support = len(reground_support_frames(evidence))
            print(f"    INSUFFICIENT {name}: fresh child-window evidence has fewer than 2 held-prompt frames")
            mark_reground_unverified(
                name,
                status="insufficient_support",
                support_frames=support,
                results_dir=RESULTS_DIR,
            )
            failed += 1
            continue
        try:
            r = recompute_quality(name)
        except Exception as exc:  # noqa: BLE001
            print(f"    QUALITY ERROR {name}: {exc}")
            mark_reground_unverified(
                name,
                status="quality_recompute_failed",
                results_dir=RESULTS_DIR,
            )
            failed += 1
            continue
        still = bool(r and r.get("pass", False))
        if not still:
            flipped += 1
        fg = ",".join(r.get("failed_gates", [])) if r else "?"
        print(f"  [{vi + 1}/{len(todo)}] {name}: {'PASS' if still else f'REJECTED ({fg})'}")
    print(f"  [Reground] {len(todo)} evidence runs + {len(quality_retry)} quality retries, {flipped} rejected, {failed} errors")
    if getattr(args, "segment", None) and failed:
        raise RuntimeError(f"the re-ground step failed for {failed} explicitly requested segment(s)")
