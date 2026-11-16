"""Pipeline step object_pose: the GPU prefix per segment and the deferred bundle adjustment."""

import gc
import json
import multiprocessing as mp
import os
import traceback
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import torch
from gtf.utils import is_action_valid, reject_action, rprint as print
from gtf.extraction import chains
from gtf.extraction.action100m.stage_support import RESULTS_DIR, _claim_chain, _drain_stage_queue, _raise_queued_stage_failure, _stage_work_items, _stale_vs_upstream


def _run_deferred_object_pose_ba(name, options):
    """Run the deferred bundle adjustment of one segment (its earlier object-pose stages are saved) in a child process.

    Raises unless the segment's object-pose done marker matches its metadata afterwards."""

    from gtf.extraction.object_pose.run_object_pose import _object_pose_done_matches, process_segment

    memory_fraction = float(os.environ.get("EXTRACTION_OBJECT_POSE_BA_MEMORY_FRACTION", "0.05"))
    if not 0.0 < memory_fraction <= 0.1:
        raise ValueError("EXTRACTION_OBJECT_POSE_BA_MEMORY_FRACTION must be in (0, 0.1]")
    if torch.cuda.is_available():
        # The BA allocates under 50 MiB. Capping this process's caching allocator at a
        # small fraction of GPU memory (default 0.05) keeps it from taking the memory that
        # CoTracker and MegaSaM use in the parent process.
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
    segment_dir = RESULTS_DIR / name
    pose_dir = segment_dir / "object_pose"
    ok = process_segment(
        None,
        name,
        str(segment_dir),
        str(pose_dir),
        force=False,
        **options,
    )
    if not ok or not _object_pose_done_matches(pose_dir / ".object_pose_done", pose_dir / "metadata.json"):
        raise RuntimeError(f"{name}: deferred BA did not publish exact completion")
    return name


def _object_pose_async_ba_workers(args):
    """Number of background BA processes (EXTRACTION_OBJECT_POSE_ASYNC_BA_WORKERS, 0 or 1).

    The split driver's retry worker (worker id ending in `_recovery`) always runs the BA in-line."""

    raw = os.environ.get("EXTRACTION_OBJECT_POSE_ASYNC_BA_WORKERS", "0")
    try:
        workers = int(raw)
    except ValueError as exc:
        raise ValueError("EXTRACTION_OBJECT_POSE_ASYNC_BA_WORKERS must be 0 or 1") from exc
    if workers not in {0, 1}:
        raise ValueError("EXTRACTION_OBJECT_POSE_ASYNC_BA_WORKERS must be 0 or 1")
    worker_id = str(getattr(args, "stage_worker_id", "") or "")
    if worker_id.endswith("_recovery"):
        return 0
    return workers


def _run_object_pose_step_async(
    valid_names,
    args,
    *,
    workers,
    is_stale,
    depth_estimator,
):
    """Object pose with the bundle adjustment in a background process.

    The GPU stages before the BA (depth, tracking, cameras) of the next segment overlap the low-memory BA of the
    previous one. A queue item is acknowledged only after its BA has finished; after the first failure no new
    segment is started, and in queue mode the failure is raised at the end."""

    from gtf.extraction.object_pose.run_object_pose import (
        DEFAULT_TRACKING_MODE,
        ObjectPoseTerminalReject,
        _object_pose_done_matches,
        offload_cotracker_cache,
        process_segment,
    )

    allowed = set(valid_names)
    queue = None
    local_names = iter(valid_names)
    if getattr(args, "stage_queue_db", None):
        from gtf.extraction.action100m.stage_queue import StageQueue

        queue = StageQueue(
            args.stage_queue_db,
            "object_pose",
            getattr(args, "stage_worker_id", None),
        )

    def claim_next():
        if queue is None:
            return next(local_names, None)
        while (name := queue.claim()) is not None:
            if name in allowed:
                return name
            queue.complete(name)
        return None

    def acknowledge(name):
        if queue is not None:
            queue.complete(name)

    options = {
        "use_megasam": not args.no_megasam,
        "cotracker_grid_size": args.cotracker_grid_size,
        "tracking_mode": args.tracking_mode or DEFAULT_TRACKING_MODE,
        "visualize": args.object_pose_visualize,
        "require_megasam": args.camera_require_megasam,
        "object_ba": chains.get(getattr(args, "chain", None)).object_ba,
    }
    pending = deque()
    done = skipped = failed = rejected = 0
    first_failure = None

    def finish_oldest():
        nonlocal done, failed, first_failure
        name, future = pending.popleft()
        try:
            completed_name = future.result()
            pose_dir = RESULTS_DIR / name / "object_pose"
            if completed_name != name or not _object_pose_done_matches(pose_dir / ".object_pose_done", pose_dir / "metadata.json"):
                raise RuntimeError(f"{name}: async BA returned without exact completion")
            acknowledge(name)
            done += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            if first_failure is None:
                first_failure = (name, exc)
            print(f"    ASYNC BA ERROR {name}: {exc}")
            traceback.print_exc()

    executor = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
    )
    try:
        index = 0
        while first_failure is None and (name := claim_next()) is not None:
            index += 1
            segment_dir = RESULTS_DIR / name
            pose_dir = segment_dir / "object_pose"
            done_marker = pose_dir / ".object_pose_done"
            if not (segment_dir / "SAM3" / ".sam3_done").exists():
                failed += 1
                acknowledge(name)
                continue
            if not args.force and done_marker.exists() and not is_stale(name):
                skipped += 1
                acknowledge(name)
                continue
            print(f"  [{index}/{len(valid_names)}] Object pose prefix: {name}")
            try:
                offload_cotracker_cache()
                ok = process_segment(
                    depth_estimator,
                    name,
                    str(segment_dir),
                    str(pose_dir),
                    force=args.force,
                    defer_bundle_adjustment=True,
                    **options,
                )
                if not ok:
                    raise RuntimeError("pose prefix returned false")
                pending.append((name, executor.submit(_run_deferred_object_pose_ba, name, options)))
            except ObjectPoseTerminalReject as exc:
                reject_action(name, str(exc), args.object)
                print(f"    TERMINAL REJECT: {exc}")
                rejected += 1
                acknowledge(name)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                first_failure = (name, exc)
                print(f"    ERROR: {exc}")
                traceback.print_exc()
            finally:
                depth_estimator.offload()
                offload_cotracker_cache()
                gc.collect()
                torch.cuda.empty_cache()

            # One queued BA plus one prefix in flight is enough to hide this
            # tail in the normal case while bounding retained queue claims.
            if len(pending) >= workers + 1:
                finish_oldest()

        while pending:
            finish_oldest()
    finally:
        executor.shutdown(wait=True, cancel_futures=False)
        if queue is not None:
            queue.close()

    print(f"  [Object pose async-BA] {done} done, {skipped} skipped, {rejected} rejected, {failed} failed")
    if first_failure is not None and getattr(args, "stage_queue_db", None):
        name, exc = first_failure
        raise RuntimeError(f"object_pose queue item {name} failed: {exc}") from exc


def run_object_pose_step(segment_names, args):
    """Run 6D object pose tracking (depth, tracking, camera poses, bundle adjustment)."""
    from gtf.extraction.object_pose.run_object_pose import (
        DEFAULT_TRACKING_MODE,
        DepthEstimator,
        ObjectPoseTerminalReject,
        _object_pose_done_matches,
        clear_cotracker_cache,
        offload_cotracker_cache,
        object_pose_metadata_request_matches,
        process_segment,
    )

    valid_names = [n for n in segment_names if is_action_valid(n, args.object)]
    chain = _claim_chain(valid_names, args)

    def _opose_stale(n):
        segment_dir = RESULTS_DIR / n
        pose_dir = segment_dir / "object_pose"
        metadata_path = pose_dir / "metadata.json"
        if args.object_pose_visualize and not (pose_dir / "vis_pose.mp4").is_file():
            return True
        if _stale_vs_upstream(
            pose_dir / ".object_pose_done",
            segment_dir / "SAM3" / "masks.npz",
        ):
            return True
        if not _object_pose_done_matches(pose_dir / ".object_pose_done", metadata_path):
            return True
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError):
            return True
        return not object_pose_metadata_request_matches(
            segment_dir,
            pose_dir,
            metadata,
            tracking_mode=args.tracking_mode or DEFAULT_TRACKING_MODE,
            cotracker_grid_size=args.cotracker_grid_size,
            use_megasam=not args.no_megasam,
            require_megasam=args.camera_require_megasam,
        )

    if not args.force and all(
        not (RESULTS_DIR / n / "SAM3" / ".sam3_done").exists()
        or (os.path.exists(os.path.join(str(RESULTS_DIR / n), "object_pose", ".object_pose_done")) and not _opose_stale(n))
        for n in valid_names
    ):
        print(f"  [ObjectPose] all {len(valid_names)} segments already done (or lack SAM3) — model load skipped")
        _drain_stage_queue("object_pose", args)
        return

    depth_estimator = DepthEstimator(device=args.device)
    async_ba_workers = _object_pose_async_ba_workers(args)
    if async_ba_workers:
        print(f"  [Object pose] bundle adjustment in {async_ba_workers} background worker(s)")
        try:
            _run_object_pose_step_async(
                valid_names,
                args,
                workers=async_ba_workers,
                is_stale=_opose_stale,
                depth_estimator=depth_estimator,
            )
        finally:
            del depth_estimator
            clear_cotracker_cache()
            torch.cuda.empty_cache()
        return
    done = skipped = failed = rejected = 0

    for vi, name in enumerate(_stage_work_items("object_pose", valid_names, args)):
        seg_dir = str(RESULTS_DIR / name)
        pose_dir = os.path.join(seg_dir, "object_pose")
        done_marker = os.path.join(pose_dir, ".object_pose_done")

        if not (RESULTS_DIR / name / "SAM3" / ".sam3_done").exists():
            failed += 1
            continue

        if not args.force and os.path.exists(done_marker) and not _opose_stale(name):
            skipped += 1
            continue

        print(f"  [{vi + 1}/{len(valid_names)}] Object pose: {name}")
        try:
            # An earlier segment may have left CoTracker resident, especially
            # after an exception. process_segment pages MoGe in only for depth
            # and offloads it again before tracking.
            offload_cotracker_cache()
            ok = process_segment(
                depth_estimator,
                name,
                seg_dir,
                pose_dir,
                # Only --force recomputes everything: process_segment checks each of its
                # cached intermediates (camera settings, evidence hash) itself and keeps
                # the depth and tracks that are still valid.
                force=args.force,
                use_megasam=not args.no_megasam,
                cotracker_grid_size=args.cotracker_grid_size,
                tracking_mode=args.tracking_mode or DEFAULT_TRACKING_MODE,
                visualize=args.object_pose_visualize,
                require_megasam=args.camera_require_megasam,
                object_ba=chain.object_ba,
            )
            if ok:
                done += 1
            else:
                failed += 1
                _raise_queued_stage_failure(args, "object_pose", name, "pose estimator returned false")
        except ObjectPoseTerminalReject as e:
            # A property of the source video, not a runtime failure: record it as a
            # rejection (the mechanism validation and the cut check use), so the
            # segment counts as finished and is not retried.
            reject_action(name, str(e), args.object)
            print(f"    TERMINAL REJECT: {e}")
            rejected += 1
        except Exception as e:
            print(f"    ERROR: {e}")
            traceback.print_exc()
            failed += 1
            _raise_queued_stage_failure(args, "object_pose", name, e)
        finally:
            depth_estimator.offload()
            offload_cotracker_cache()
            gc.collect()
            torch.cuda.empty_cache()

    del depth_estimator
    clear_cotracker_cache()
    torch.cuda.empty_cache()
    print(f"  [Object pose] {done} done, {skipped} skipped, {rejected} rejected, {failed} failed")
