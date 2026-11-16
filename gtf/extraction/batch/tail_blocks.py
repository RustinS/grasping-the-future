#!/usr/bin/env python3
"""Blocks of the GPU/CPU split driver: its segment list cut into blocks for the heavy GPU stages, with ready markers.

scripts/run_split_extraction.sh runs sam3d → object_pose → sam3dc → optimize → refine one block at a time and publishes
a ready marker after each block, so the CPU tail (cpu_tail_queue.py) can export and gate that block while the GPUs
work on the next one. Blocks keep a source segment together with its `__shotN` / `__iwN` children, and their sizes
decrease geometrically so the CPU tail finishes soon after the GPUs (a two-machine flow shop).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

CHILD_SUFFIX_RE = re.compile(r"__(?:(?:iw|shot)\d+)$")


def source_owner(segment_name: str) -> str:
    """The source segment a child belongs to: strips nested `__shotN` / `__iwN` suffixes."""

    owner = segment_name
    while True:
        parent = CHILD_SUFFIX_RE.sub("", owner)
        if parent == owner:
            return owner
        owner = parent


PLAN_FORMAT = "gtf.gpu-tail-block-plan.v2"
READY_FORMAT = "gtf.gpu-tail-block-ready.v1"
FLOW_SCHEDULE_KIND = "two-stage-geometric-lpt"


class TailBlockError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, value: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    _atomic_bytes(path, encoded)


def read_names(path: Path) -> list[str]:
    names: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        name = raw.strip()
        if not name or name.startswith("#"):
            continue
        if Path(name).name != name or name in {".", ".."} or "\0" in name:
            raise TailBlockError(f"unsafe segment name in {path}: {name!r}")
        if name in names:
            raise TailBlockError(f"duplicate segment in {path}: {name}")
        names.append(name)
    return names


def _group_weight(names: list[str], results_root: Path) -> int:
    """Work estimate of a group of segments: SAM3's ``total_frames`` per segment, 1 when missing.

    Blocks are planned right after SAM3, before object pose exists. The frame count is a common proxy
    for SAM3D, tracking, optimization and CPU-tail work and needs no video decode. Segments without it
    are normally front-end rejects and cheap. The value affects ordering only, never which stages run.
    """

    total = 0
    for name in names:
        segment = results_root / name
        try:
            metadata = json.loads((segment / "SAM3" / "metadata.json").read_text())
            frames = metadata.get("total_frames")
            if isinstance(frames, int) and not isinstance(frames, bool) and frames > 0:
                total += frames
                continue
        except (OSError, json.JSONDecodeError):
            pass
        total += 1
    return total


def validate_cpu_gpu_work_ratio(value: float | str) -> float:
    """The CPU/GPU work ratio as a float in (0, 1]; raises ValueError otherwise.

    The block schedule (flow_target_shares) assumes the GPU stages take more total time than the CPU tail.
    At a ratio of one all blocks are equal; a CPU-heavier run would need a different schedule.
    """

    try:
        ratio = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid CPU/GPU work ratio: {value!r}") from error
    if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ValueError(
            "CPU/GPU work ratio must be finite and in the interval (0, 1]"
        )
    return ratio


def flow_target_shares(block_count: int, cpu_gpu_work_ratio: float) -> list[float]:
    """Work shares of the blocks for a two-machine GPU -> CPU flow shop (divisible load).

    With GPU work ``G``, CPU work ``C`` and ``r = C/G <= 1``, the CPU never idles when
    ``C*x[i] = G*x[i+1]``, so consecutive shares decrease geometrically by ``r``. Normalized, these shares
    minimize the CPU finish time for a fixed block count. The fixed per-block model load time is not modeled:
    the ratio only decides how segments are distributed over the blocks.
    """

    if block_count < 1:
        raise ValueError("tail block count must be positive")
    ratio = validate_cpu_gpu_work_ratio(cpu_gpu_work_ratio)
    powers = [ratio**index for index in range(block_count)]
    denominator = math.fsum(powers)
    shares = [power / denominator for power in powers]
    if any(not math.isfinite(share) or share <= 0.0 for share in shares):
        raise ValueError("CPU/GPU ratio underflowed for the requested block count")
    return shares


def _partition_names_with_weights(
    names: list[str],
    block_count: int,
    *,
    results_root: Path,
    cpu_gpu_work_ratio: float,
) -> tuple[list[list[str]], list[int], list[float]]:
    """Deterministic blocks whose work weights follow the target shares, each source segment in one block with
    its children; returns (blocks, block weights, target shares)."""

    if block_count < 1:
        raise ValueError("tail block count must be positive")
    ratio = validate_cpu_gpu_work_ratio(cpu_gpu_work_ratio)
    if not names:
        return [], [], []
    groups: dict[str, list[str]] = defaultdict(list)
    positions = {name: index for index, name in enumerate(names)}
    for name in names:
        groups[source_owner(name)].append(name)
    count = min(block_count, len(groups))
    target_shares = flow_target_shares(count, ratio)
    blocks: list[list[str]] = [[] for _ in range(count)]
    totals = [0] * count
    weighted_groups = [
        (_group_weight(group, results_root), owner, group)
        for owner, group in groups.items()
    ]
    for weight, _owner, group in sorted(
        weighted_groups, key=lambda row: (-row[0], row[1])
    ):
        # Weighted longest-processing-time assignment against the target shares; the
        # owner grouping keeps each source segment and its children in one block.
        target = min(
            range(count),
            key=lambda index: (
                (totals[index] + weight) / target_shares[index],
                index,
            ),
        )
        blocks[target].extend(group)
        totals[target] += weight
    for block in blocks:
        block.sort(key=positions.__getitem__)
    return blocks, totals, target_shares


def create_plan(
    *,
    segment_file: Path,
    output_dir: Path,
    block_count: int,
    attempt_id: str,
    phase_start: Path,
    results_root: Path,
    cpu_gpu_work_ratio: float,
) -> tuple[Path, dict[str, Any]]:
    """Write block_NNN.segments files and plan.json into output_dir; returns (plan path, plan).

    The plan is bound to the attempt id, the segment list (SHA-256) and the phase-start marker (mtime)."""
    if not attempt_id or "\0" in attempt_id:
        raise ValueError("tail block attempt id must be non-empty")
    segment_file = Path(segment_file).resolve(strict=True)
    phase_start = Path(phase_start).resolve(strict=True)
    output_dir = Path(output_dir)
    names = read_names(segment_file)
    ratio = validate_cpu_gpu_work_ratio(cpu_gpu_work_ratio)
    blocks, block_weights, target_shares = _partition_names_with_weights(
        names,
        block_count,
        results_root=Path(results_root),
        cpu_gpu_work_ratio=ratio,
    )
    total_weight = sum(block_weights)
    actual_shares = (
        [weight / total_weight for weight in block_weights] if total_weight else []
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, block in enumerate(blocks):
        path = output_dir / f"block_{index:03d}.segments"
        _atomic_bytes(path, "".join(f"{name}\n" for name in block).encode())
        rows.append(
            {
                "index": index,
                "segment_file": path.name,
                "segment_file_sha256": _sha256_file(path),
                "segments": block,
                "work_weight": block_weights[index],
                "target_work_share": target_shares[index],
                "actual_work_share": actual_shares[index],
            }
        )
    plan = {
        "format": PLAN_FORMAT,
        "attempt_id": attempt_id,
        "roster_sha256": _sha256_file(segment_file),
        "phase_start_mtime_ns": phase_start.stat().st_mtime_ns,
        "requested_block_count": block_count,
        "block_count": len(rows),
        "schedule": {
            "kind": FLOW_SCHEDULE_KIND,
            "cpu_gpu_work_ratio": ratio,
            "weight_basis": "committed_sam3_total_frames_per_source_family_fallback_1",
            "total_work_weight": total_weight,
            "target_work_shares": target_shares,
            "actual_work_shares": actual_shares,
        },
        "blocks": rows,
    }
    plan_path = output_dir / "plan.json"
    _atomic_json(plan_path, plan)
    return plan_path, plan


def validate_plan(
    plan_path: Path,
    *,
    attempt_id: str,
    segment_file: Path,
    phase_start: Path,
) -> dict[str, Any]:
    """Load plan.json and check it against the attempt, the segment list, the phase-start marker and the block
    files; raises TailBlockError on any mismatch."""
    plan_path = Path(plan_path)
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TailBlockError(f"tail block plan is unreadable: {error}") from error
    if not isinstance(plan, dict):
        raise TailBlockError("tail block plan is not a JSON object")
    segment_file = Path(segment_file)
    phase_start = Path(phase_start)
    if (
        plan.get("format") != PLAN_FORMAT
        or plan.get("attempt_id") != attempt_id
        or plan.get("roster_sha256") != _sha256_file(segment_file)
        or plan.get("phase_start_mtime_ns") != phase_start.stat().st_mtime_ns
    ):
        raise TailBlockError("tail block plan attempt binding mismatch")
    blocks = plan.get("blocks")
    if not isinstance(blocks, list) or plan.get("block_count") != len(blocks):
        raise TailBlockError("tail block plan roster is malformed")
    schedule = plan.get("schedule")
    if not isinstance(schedule, dict) or schedule.get("kind") != FLOW_SCHEDULE_KIND:
        raise TailBlockError("tail block flow schedule is malformed")
    try:
        ratio = validate_cpu_gpu_work_ratio(schedule.get("cpu_gpu_work_ratio"))
        target_shares = flow_target_shares(len(blocks), ratio) if blocks else []
    except ValueError as error:
        raise TailBlockError(f"tail block flow schedule is malformed: {error}") from error
    if (
        schedule.get("weight_basis")
        != "committed_sam3_total_frames_per_source_family_fallback_1"
        or schedule.get("target_work_shares") != target_shares
    ):
        raise TailBlockError("tail block flow schedule is malformed")
    expected_names = read_names(segment_file)
    actual_names: list[str] = []
    actual_weights: list[int] = []
    owner_blocks: dict[str, int] = {}
    for index, row in enumerate(blocks):
        if not isinstance(row, dict) or row.get("index") != index:
            raise TailBlockError("tail block indexes are malformed")
        names = row.get("segments")
        filename = row.get("segment_file")
        if (
            not isinstance(names, list)
            or any(not isinstance(name, str) for name in names)
            or filename != f"block_{index:03d}.segments"
            or not isinstance(row.get("work_weight"), int)
            or isinstance(row.get("work_weight"), bool)
            or row["work_weight"] <= 0
            or row.get("target_work_share") != target_shares[index]
        ):
            raise TailBlockError(f"tail block {index} is malformed")
        block_path = plan_path.parent / filename
        if (
            row.get("segment_file_sha256") != _sha256_file(block_path)
            or read_names(block_path) != names
        ):
            raise TailBlockError(f"tail block {index} file binding mismatch")
        for name in names:
            owner = source_owner(name)
            prior = owner_blocks.setdefault(owner, index)
            if prior != index:
                raise TailBlockError(
                    f"tail block plan splits source family {owner} across blocks"
                )
        actual_names.extend(names)
        actual_weights.append(row["work_weight"])
    if len(actual_names) != len(set(actual_names)) or set(actual_names) != set(
        expected_names
    ):
        raise TailBlockError("tail blocks do not partition the exact shard roster")
    total_weight = sum(actual_weights)
    actual_shares = (
        [weight / total_weight for weight in actual_weights] if total_weight else []
    )
    if (
        schedule.get("total_work_weight") != total_weight
        or schedule.get("actual_work_shares") != actual_shares
        or any(
            row.get("actual_work_share") != actual_shares[index]
            for index, row in enumerate(blocks)
        )
    ):
        raise TailBlockError("tail block actual work shares are malformed")
    return plan


def publish_ready(
    *,
    plan_path: Path,
    ready_dir: Path,
    block_index: int,
    attempt_id: str,
    segment_file: Path,
    phase_start: Path,
) -> Path:
    """Write the ready marker of one block (the driver calls it after every GPU stage of the block has finished)."""
    plan = validate_plan(
        plan_path,
        attempt_id=attempt_id,
        segment_file=segment_file,
        phase_start=phase_start,
    )
    if not 0 <= block_index < len(plan["blocks"]):
        raise TailBlockError(f"tail block index is out of range: {block_index}")
    block = plan["blocks"][block_index]
    marker = {
        "format": READY_FORMAT,
        "attempt_id": attempt_id,
        "plan_sha256": _sha256_file(Path(plan_path)),
        "roster_sha256": plan["roster_sha256"],
        "phase_start_mtime_ns": plan["phase_start_mtime_ns"],
        "block_index": block_index,
        "block_count": len(plan["blocks"]),
        "segment_file_sha256": block["segment_file_sha256"],
        "segments": block["segments"],
    }
    path = Path(ready_dir) / f"block_{block_index:03d}.ready.json"
    _atomic_json(path, marker)
    return path


def load_authorizations(
    *,
    plan_path: Path,
    ready_dir: Path,
    attempt_id: str,
    segment_file: Path,
    phase_start: Path,
) -> dict[str, Path]:
    """Map every segment of a block with a valid ready marker to that marker's path.

    No later GPU stage writes those segments, so the CPU tail can process them. Returns {} before the plan exists.
    """

    plan_path = Path(plan_path)
    if not plan_path.is_file():
        return {}
    plan = validate_plan(
        plan_path,
        attempt_id=attempt_id,
        segment_file=segment_file,
        phase_start=phase_start,
    )
    plan_sha256 = _sha256_file(plan_path)
    authorizations: dict[str, Path] = {}
    ready_dir = Path(ready_dir)
    if not ready_dir.is_dir():
        return authorizations
    for path in sorted(ready_dir.glob("*.ready.json")):
        try:
            marker = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise TailBlockError(f"tail block marker is unreadable: {path}") from error
        index = marker.get("block_index") if isinstance(marker, dict) else None
        if not isinstance(index, int) or not 0 <= index < len(plan["blocks"]):
            raise TailBlockError(f"tail block marker index is malformed: {path}")
        block = plan["blocks"][index]
        if (
            path.name != f"block_{index:03d}.ready.json"
            or marker.get("format") != READY_FORMAT
            or marker.get("attempt_id") != attempt_id
            or marker.get("plan_sha256") != plan_sha256
            or marker.get("roster_sha256") != plan["roster_sha256"]
            or marker.get("phase_start_mtime_ns") != plan["phase_start_mtime_ns"]
            or marker.get("block_count") != len(plan["blocks"])
            or marker.get("segment_file_sha256") != block["segment_file_sha256"]
            or marker.get("segments") != block["segments"]
        ):
            raise TailBlockError(f"tail block marker binding mismatch: {path}")
        for name in block["segments"]:
            if name in authorizations:
                raise TailBlockError(f"duplicate tail authorization for {name}")
            authorizations[name] = path
    return authorizations


def main(argv: list[str] | None = None) -> int:
    def ratio_argument(value: str) -> float:
        try:
            return validate_cpu_gpu_work_ratio(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(str(error)) from error

    parser = argparse.ArgumentParser(prog="python -m gtf.extraction.batch.tail_blocks")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ratio_parser = subparsers.add_parser("validate-ratio")
    ratio_parser.add_argument("--cpu-gpu-work-ratio", type=ratio_argument, required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--segment-file", type=Path, required=True)
    plan_parser.add_argument("--output-dir", type=Path, required=True)
    plan_parser.add_argument("--blocks", type=int, required=True)
    plan_parser.add_argument("--attempt-id", required=True)
    plan_parser.add_argument("--phase-start", type=Path, required=True)
    plan_parser.add_argument("--results-root", type=Path, required=True)
    plan_parser.add_argument(
        "--cpu-gpu-work-ratio", type=ratio_argument, required=True
    )

    ready_parser = subparsers.add_parser("publish-ready")
    ready_parser.add_argument("--plan", type=Path, required=True)
    ready_parser.add_argument("--ready-dir", type=Path, required=True)
    ready_parser.add_argument("--block-index", type=int, required=True)
    ready_parser.add_argument("--attempt-id", required=True)
    ready_parser.add_argument("--segment-file", type=Path, required=True)
    ready_parser.add_argument("--phase-start", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "validate-ratio":
        print(format(args.cpu_gpu_work_ratio, ".15g"))
    elif args.command == "plan":
        path, plan = create_plan(
            segment_file=args.segment_file,
            output_dir=args.output_dir,
            block_count=args.blocks,
            attempt_id=args.attempt_id,
            phase_start=args.phase_start,
            results_root=args.results_root,
            cpu_gpu_work_ratio=args.cpu_gpu_work_ratio,
        )
        print(f"{path}\t{plan['block_count']}")
    else:
        path = publish_ready(
            plan_path=args.plan,
            ready_dir=args.ready_dir,
            block_index=args.block_index,
            attempt_id=args.attempt_id,
            segment_file=args.segment_file,
            phase_start=args.phase_start,
        )
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
