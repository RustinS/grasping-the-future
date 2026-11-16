#!/usr/bin/env python3
"""CPU tail of the GPU/CPU split driver: export_canonical + quality_gate per segment, as soon as its trajectory is ready.

Started by scripts/run_split_extraction.sh. A segment is ready when the GPU block that contains it has finished refine
(a ready marker from gtf.extraction.batch.tail_blocks) or, at the end, when the whole GPU phase has succeeded. Each segment runs
`python -m gtf.extraction.action100m.pipeline --start-from export_canonical` in its own process group, without CUDA, with a deadline.
"""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from gtf.extraction.action100m.stage_queue import read_names
from gtf.extraction.batch.tail_blocks import _sha256_file, load_authorizations
from gtf.paths import REPO_ROOT

RESULTS = Path(os.environ.get("ACTION100M_RESULTS_DIR") or REPO_ROOT / "results")
PIPELINE_MODULE = "gtf.extraction.action100m.pipeline"
DEFAULT_SEGMENT_TIMEOUT_SECONDS = 90 * 60
DEFAULT_TERMINATE_GRACE_SECONDS = 30


class SegmentProcessTimeout(subprocess.TimeoutExpired):
    """A CPU-tail child exceeded its deadline and was stopped best-effort."""

    def __init__(self, cmd, timeout, *, pid, cleanup_complete):
        super().__init__(cmd, timeout)
        self.pid = pid
        self.cleanup_complete = cleanup_complete


class ProcessGroupRegistry:
    """Tracks every running CPU-tail child (each in its own process group) so a shutdown can stop them all."""

    def __init__(self, *, terminate_grace_seconds=DEFAULT_TERMINATE_GRACE_SECONDS):
        self._terminate_grace_seconds = terminate_grace_seconds
        self._lock = threading.Lock()
        self._processes = {}
        self._shutdown_started = False

    def register(self, process):
        with self._lock:
            if self._shutdown_started:
                return False
            self._processes[process.pid] = process
            return True

    def unregister(self, process):
        with self._lock:
            self._processes.pop(process.pid, None)

    def terminate_unregistered(self, process):
        return _terminate_process_group(
            process, grace_seconds=self._terminate_grace_seconds
        )

    def shutdown(self):
        # Mark shutdown before taking the snapshot. A child spawned concurrently
        # with parent TERM will then refuse registration and stop itself.
        with self._lock:
            self._shutdown_started = True
            processes = list(self._processes.values())
        return _terminate_process_groups(
            processes, grace_seconds=self._terminate_grace_seconds
        )


@contextmanager
def parent_shutdown_guard(registry):
    """On SIGINT / SIGTERM or an exception in this process, stop every child process group (bounded wait)."""

    previous_handlers = {}
    handling_signal = False

    def handle_signal(signal_number, _frame):
        nonlocal handling_signal
        if handling_signal:
            return
        handling_signal = True
        print(
            f"[cpu-tail] parent received signal {signal_number}; "
            "stopping isolated child process groups",
            flush=True,
        )
        registry.shutdown()
        raise SystemExit(128 + signal_number)

    for signal_number in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signal_number] = signal.getsignal(signal_number)
        signal.signal(signal_number, handle_signal)
    try:
        yield
    except BaseException:
        registry.shutdown()
        raise
    finally:
        for signal_number, previous_handler in previous_handlers.items():
            signal.signal(signal_number, previous_handler)


def positive_seconds(value):
    """Argparse type for finite, positive subprocess deadlines."""

    try:
        seconds = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("must be a number of seconds") from error
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return seconds


def _process_group_exists(process):
    # Reap the direct child whenever possible before probing its process group.
    process.poll()
    try:
        os.killpg(process.pid, 0)
        return True
    except ProcessLookupError:
        return False


def _signal_process_groups(processes, signal_number):
    for process in processes:
        if not _process_group_exists(process):
            continue
        try:
            os.killpg(process.pid, signal_number)
        except ProcessLookupError:
            pass


def _wait_for_process_groups(processes, timeout):
    deadline = time.monotonic() + timeout
    remaining = list(processes)
    while remaining:
        remaining = [process for process in remaining if _process_group_exists(process)]
        if not remaining or time.monotonic() >= deadline:
            break
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    # Popen owns only the group leader; reap it even when an uninterruptible
    # descendant keeps the group alive beyond the bounded shutdown wait.
    for process in processes:
        try:
            process.wait(timeout=0)
        except subprocess.TimeoutExpired:
            pass
    return remaining


def _terminate_process_groups(processes, *, grace_seconds):
    """Stop child process groups in parallel (SIGTERM, then SIGKILL), with one shared grace period; True when all stopped."""

    processes = list({process.pid: process for process in processes}.values())
    if not processes:
        return True
    _signal_process_groups(processes, signal.SIGTERM)
    remaining = _wait_for_process_groups(processes, grace_seconds)
    if remaining:
        _signal_process_groups(remaining, signal.SIGKILL)
        remaining = _wait_for_process_groups(remaining, grace_seconds)
    # A process in uninterruptible I/O can survive both signals: report the
    # failure to the caller instead of waiting for it.
    return not remaining


def _terminate_process_group(process, *, grace_seconds):
    """Stop one child process group (never this process's own group)."""

    return _terminate_process_groups([process], grace_seconds=grace_seconds)


def run_subprocess_with_timeout(
    command,
    *,
    cwd,
    env,
    stdout,
    timeout_seconds,
    terminate_grace_seconds=DEFAULT_TERMINATE_GRACE_SECONDS,
    registry=None,
):
    """Run one segment in its own process group with a bounded wait."""

    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    registered = registry is not None and registry.register(process)
    if registry is not None and not registered:
        registry.terminate_unregistered(process)
        raise RuntimeError("CPU-tail parent shutdown began during child launch")
    try:
        try:
            return process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            cleanup_complete = _terminate_process_group(
                process, grace_seconds=terminate_grace_seconds
            )
            raise SegmentProcessTimeout(
                command,
                timeout_seconds,
                pid=process.pid,
                cleanup_complete=cleanup_complete,
            ) from error
    finally:
        if registered:
            registry.unregister(process)


def safe_log_name(name):
    return name.replace("/", "_").replace("\\", "_")


def cpu_tail_environment(seed_workers):
    if seed_workers < 1:
        raise ValueError("seed worker count must be positive")
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["OBJREF_SEED_WORKERS"] = str(seed_workers)
    return environment


def _same_file_generation(path, before):
    after = path.stat()
    return (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) == (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )


def _refine_transaction_complete(optimized, refined, report_path, segment_name):
    """Whether refine_report.json describes exactly these two files.

    The report's `transaction` record must match the size, mtime and SHA-256 of the optimizer output (input) and
    the size and SHA-256 of the refined trajectory (output), and none of the three files may change during the check."""

    try:
        if (
            not optimized.is_file()
            or not refined.is_file()
            or not report_path.is_file()
        ):
            return False
        input_stat = optimized.stat()
        output_stat = refined.stat()
        report_stat = report_path.stat()
        if (
            output_stat.st_mtime_ns <= input_stat.st_mtime_ns
            or report_stat.st_mtime_ns < output_stat.st_mtime_ns
        ):
            return False
        report = json.loads(report_path.read_text(encoding="utf-8"))
        transaction = report.get("transaction")
        if not isinstance(transaction, dict):
            return False
        frame_count = transaction.get("frame_count")
        if (
            report.get("segment") != segment_name
            or not isinstance(frame_count, int)
            or frame_count <= 0
            or report.get("K") != frame_count
            or transaction.get("format") != "gtf.refine.v2"
            or transaction.get("input_size_bytes") != input_stat.st_size
            or transaction.get("input_mtime_ns") != input_stat.st_mtime_ns
            or transaction.get("output_size_bytes") != output_stat.st_size
            or transaction.get("input_sha256") != _sha256_file(optimized)
            or transaction.get("output_sha256") != _sha256_file(refined)
        ):
            return False
        # Catch atomic replacements or in-place rewrites during validation.
        return (
            _same_file_generation(optimized, input_stat)
            and _same_file_generation(refined, output_stat)
            and _same_file_generation(report_path, report_stat)
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def trajectory_is_fresh(candidate, segment, force=False, started_at=0.0):
    """Whether the trajectory file `candidate` is newer than every upstream GPU stage of the segment.

    A refined trajectory (optimized_stage2/) must also match its refine report and sit on a fresh optimizer output.
    With --force, a file older than this run's phase-start marker is stale."""
    if not candidate.exists():
        return False
    candidate_time = candidate.stat().st_mtime
    if force and candidate_time < started_at:
        return False
    if candidate.parent.name == "optimized_stage2":
        optimized = segment / "optimized_mano" / "optimized.npz"
        report_path = candidate.with_name("refine_report.json")
        if not optimized.exists() or not report_path.exists():
            return False
        if not _refine_transaction_complete(
            optimized, candidate, report_path, segment.name
        ):
            return False
        return candidate_time >= optimized.stat().st_mtime and trajectory_is_fresh(
            optimized,
            segment,
            force=force,
            started_at=started_at,
        )
    upstream = [
        segment / "SAM3" / ".sam3_done",
        segment / "sam3d" / ".sam3d_done",
        segment / "object_pose" / ".object_pose_done",
    ]
    # A missing upstream marker means the GPU stage has not run yet (the split driver runs
    # sam3d / object_pose for every segment), so the trajectory is not fresh.
    return all(
        path.exists() and candidate_time >= path.stat().st_mtime for path in upstream
    )


def main():
    parser = argparse.ArgumentParser(prog="python -m gtf.extraction.batch.cpu_tail_queue")
    parser.add_argument("--segment-file", type=Path, required=True)
    parser.add_argument("--phase-done", type=Path, required=True)
    parser.add_argument("--phase-success", type=Path, required=True)
    parser.add_argument("--phase-start", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--tail-plan", type=Path, required=True)
    parser.add_argument("--block-ready-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--seed-workers",
        type=int,
        default=2,
        help="Object-refiner seed threads per segment process (scheduling only; results do not depend on it)",
    )
    parser.add_argument(
        "--segment-timeout-seconds",
        type=positive_seconds,
        default=os.environ.get(
            "EXTRACTION_CPU_TAIL_TIMEOUT_SECONDS",
            str(DEFAULT_SEGMENT_TIMEOUT_SECONDS),
        ),
        help="Maximum wall time of one export/gate subprocess in seconds (default: 5400, or EXTRACTION_CPU_TAIL_TIMEOUT_SECONDS)",
    )
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.workers < 1 or args.seed_workers < 1:
        parser.error("--workers and --seed-workers must be positive")
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
    force = "--force" in extra
    started_at = args.phase_start.stat().st_mtime
    args.log_dir.mkdir(parents=True, exist_ok=True)
    pending = set(read_names(args.segment_file))
    running = {}
    completed = failed = skipped = 0
    process_registry = ProcessGroupRegistry()

    def run_one(name):
        command = [
            sys.executable,
            "-u",
            "-m",
            PIPELINE_MODULE,
            "--start-from",
            "export_canonical",
            f"--segment={name}",
            *extra,
        ]
        # Export workers run without CUDA and with a few object-refiner seed threads each; the
        # refiner reduces its seeds in a fixed order, so the thread count does not change results.
        environment = cpu_tail_environment(args.seed_workers)
        log_path = args.log_dir / f"cpu_{safe_log_name(name)}.log"
        with log_path.open("w") as log:
            try:
                returncode = run_subprocess_with_timeout(
                    command,
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=log,
                    timeout_seconds=args.segment_timeout_seconds,
                    registry=process_registry,
                )
            except SegmentProcessTimeout as error:
                cleanup = "stopped" if error.cleanup_complete else "still uninterruptible"
                message = f"[cpu-tail] ERROR: {name} exceeded the {args.segment_timeout_seconds:g}s segment deadline; child process group {error.pid} is {cleanup}."
                print(message, file=log, flush=True)
                print(message, flush=True)
                raise
        return returncode

    seen = set(pending)
    with ThreadPoolExecutor(
        max_workers=max(1, args.workers)
    ) as executor, parent_shutdown_guard(process_registry):
        while pending or running:
            gpu_done = args.phase_done.exists()
            gpu_succeeded = args.phase_success.exists()
            block_authorizations = load_authorizations(
                plan_path=args.tail_plan,
                ready_dir=args.block_ready_dir,
                attempt_id=args.attempt_id,
                segment_file=args.segment_file,
                phase_start=args.phase_start,
            )
            # the segment file can grow during the run: the driver appends the {seg}__shotN
            # children that cutcheck harvests
            for name in read_names(args.segment_file):
                if name not in seen:
                    seen.add(name)
                    pending.add(name)
            for name in list(pending):
                segment = RESULTS / name
                candidate = segment / "optimized_stage2" / "optimized.npz"
                # A block marker means no later GPU stage writes this segment; the global
                # success marker covers the rest at the end.
                authorization = block_authorizations.get(name)
                if authorization is None and gpu_succeeded:
                    authorization = args.phase_success
                ready = authorization is not None and trajectory_is_fresh(
                    candidate, segment, force, started_at
                )
                if ready:
                    pending.remove(name)
                    running[executor.submit(run_one, name)] = name
                elif gpu_done:
                    pending.remove(name)
                    skipped += 1

            for future in list(running):
                if not future.done():
                    continue
                name = running.pop(future)
                try:
                    returncode = future.result()
                except Exception as error:
                    print(f"[cpu-tail] {name} failed: {error}", flush=True)
                    failed += 1
                else:
                    if returncode == 0:
                        completed += 1
                    else:
                        print(
                            f"[cpu-tail] {name} exited {returncode}; see its log",
                            flush=True,
                        )
                        failed += 1

            if pending or running:
                time.sleep(2)

    print(
        f"[cpu-tail] complete={completed} failed={failed} skipped={skipped}",
        flush=True,
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
