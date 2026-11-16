"""Evaluate a GTF checkpoint on the evaluation split (eval.split).

Usage:
    python -m gtf.eval.run_eval [--config CONFIG.yaml ...] eval.checkpoint_path=PATH [key=value ...]

    # Sharded: run every shard with the same arguments plus eval.num_shards=N eval.shard_idx=I,
    # then merge the shards of one output directory:
    python -m gtf.eval.run_eval --merge OUTPUT_DIR

The config is the GTFConfig defaults, then the YAML files in order, then the key=value
overrides. Results go to <eval.output_dir>/eval_results.json; a shard writes
eval_shard_<I>of<N>.json holding its per-window accumulators instead.
"""

import argparse
import glob
import json
import math
import os
from datetime import datetime

import torch
from omegaconf import OmegaConf

from gtf.paths import MANO_DIR
from gtf.config import GTFConfig


def _sanitize_for_json(obj):
    """Replace float NaN/Inf with None (json.dump writes them as invalid JSON literals)."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, torch.Tensor):
        return _sanitize_for_json(obj.item() if obj.numel() == 1 else obj.tolist())
    return obj


def _json_default(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    return str(obj)


def _print_summary(results: dict) -> None:
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY (mm)")
    print("=" * 70)
    for task, res in results.items():
        print(f"\n--- {task} ({res.get('n_windows_evaluated')} windows) ---")
        for proto in ("single", "joint_best_of_N"):
            for key, val in sorted(res.get(proto, {}).items()):
                if isinstance(val, float) and math.isfinite(val):
                    print(f"  {proto:16s} {key:24s} {val:10.2f}")


def merge_shards(shard_dir: str) -> str:
    """Merge the shards of one sharded run into <shard_dir>/eval_results.json.

    Refuses incomplete or duplicate shard sets, shards run with different configs, and merged
    window sets that are not exactly the expected windows.
    """
    from gtf.eval.evaluate import aggregate_task

    files = sorted(glob.glob(os.path.join(shard_dir, "eval_shard_*of*.json")))
    if not files:
        raise SystemExit(f"no eval_shard_*of*.json in {shard_dir}")
    blobs = []
    for path in files:
        with open(path) as f:
            blobs.append(json.load(f))
    n_shards = {b["num_shards"] for b in blobs}
    if len(n_shards) != 1:
        raise SystemExit(f"shards disagree on num_shards: {sorted(n_shards)}")
    n_shards = n_shards.pop()
    have = sorted(b["shard_idx"] for b in blobs)
    if have != list(range(n_shards)):
        raise SystemExit(f"incomplete or duplicate shards: have {have}, expected 0..{n_shards - 1}")

    def _protocol(cfg: dict) -> str:
        cfg = json.loads(json.dumps(cfg))
        cfg["eval"].pop("shard_idx", None)
        return json.dumps(cfg, sort_keys=True)

    if len({_protocol(b["config"]) for b in blobs}) != 1:
        raise SystemExit("shards were run with different configs")

    cfg = blobs[0]["config"]
    thresholds = [float(t) for t in cfg["eval"]["action_thresholds_mm"]]
    results = {}
    for task in cfg["eval"]["tasks"]:
        raws = [b["raw"][task] for b in blobs]
        merged = {"all_window_metrics": [], "act_rows": [], "gap_records": {}, "n_expected_windows": raws[0]["n_expected_windows"]}
        for raw in raws:
            if raw["n_expected_windows"] != merged["n_expected_windows"]:
                raise SystemExit(f"{task}: shards disagree on the expected window count")
            merged["all_window_metrics"].extend(raw["all_window_metrics"])
            merged["act_rows"].extend(raw["act_rows"])
            for ent, recs in raw["gap_records"].items():
                merged["gap_records"].setdefault(ent, []).extend(recs)
        ids = [(r["segment_id"], r["window_start"]) for r in merged["act_rows"]]
        if len(set(ids)) != len(ids):
            raise SystemExit(f"{task}: {len(ids) - len(set(ids))} windows appear in more than one shard")
        if len(ids) != merged["n_expected_windows"]:
            raise SystemExit(f"{task}: merged {len(ids)} windows, expected {merged['n_expected_windows']}")
        results[task] = aggregate_task(merged, thresholds)
        print(f"  {task}: merged {len(ids)} windows from {n_shards} shards")

    output = {"checkpoint": blobs[0]["checkpoint"], "timestamp": blobs[0]["timestamp"], "merged_from_shards": n_shards, "config": cfg, "results": results}
    out_path = os.path.join(shard_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(_sanitize_for_json(output), f, indent=2, default=_json_default)
    print(f"\nMerged results saved to {out_path}")
    _print_summary(results)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        prog="python -m gtf.eval.run_eval", description="GTF evaluation", epilog="Remaining arguments are OmegaConf overrides (e.g. eval.n_samples=20)"
    )
    parser.add_argument("--config", action="append", default=[], help="YAML config file; repeat to chain (later files override earlier ones)")
    parser.add_argument("--merge", metavar="DIR", help="merge the eval_shard_*of*.json files of DIR and exit")
    parser.add_argument("overrides", nargs="*", help="OmegaConf key=value overrides")
    args = parser.parse_args()
    if args.merge:
        merge_shards(args.merge)
        return

    torch.set_float32_matmul_precision("high")
    cfg = OmegaConf.merge(
        OmegaConf.structured(GTFConfig),
        *[OmegaConf.load(p) for p in args.config],
        OmegaConf.from_dotlist(args.overrides),
    )
    eval_cfg = cfg.eval
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_shards = max(1, int(eval_cfg.num_shards))
    shard_idx = int(eval_cfg.shard_idx)

    print(f"Checkpoint:  {eval_cfg.checkpoint_path}")
    print(f"Tasks:       {list(eval_cfg.tasks)}")
    print(f"Samples:     {eval_cfg.n_samples} x {eval_cfg.n_steps} Euler steps (amp={bool(eval_cfg.amp)})")
    print(f"Split:       {eval_cfg.split}, windows {eval_cfg.window_seconds}s / stride {eval_cfg.stride_seconds}s")
    if num_shards > 1:
        print(f"Shard:       {shard_idx} of {num_shards}")

    from gtf.eval.checkpoint import load_lit_module_from_checkpoint
    from gtf.eval.evaluate import run_evaluation

    lit = load_lit_module_from_checkpoint(cfg, device, mano_root=MANO_DIR)
    raw_by_task = {} if num_shards > 1 else None
    results = run_evaluation(lit.model, lit.stats, eval_cfg, cfg.data, MANO_DIR, device, raw_by_task=raw_by_task)

    output_dir = str(eval_cfg.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    header = {
        "checkpoint": str(eval_cfg.checkpoint_path),
        "timestamp": datetime.now().isoformat(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    if num_shards > 1:
        # Raw accumulators keep NaN as JSON NaN literals (read back as float NaN) for re-aggregation.
        path = os.path.join(output_dir, f"eval_shard_{shard_idx}of{num_shards}.json")
        with open(path, "w") as f:
            json.dump({**header, "shard_idx": shard_idx, "num_shards": num_shards, "raw": raw_by_task}, f, default=_json_default)
        print(f"\nShard {shard_idx}/{num_shards} saved to {path}; merge with: python -m gtf.eval.run_eval --merge {output_dir}")
        return

    path = os.path.join(output_dir, "eval_results.json")
    with open(path, "w") as f:
        json.dump(_sanitize_for_json({**header, "results": results}), f, indent=2, default=_json_default)
    print(f"\nResults saved to {path}")
    _print_summary(results)


if __name__ == "__main__":
    main()
