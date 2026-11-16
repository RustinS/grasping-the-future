#!/usr/bin/env python3
"""Persistent WiLoR runner for a shard of independent video jobs.

Runs inside WiLoR's own virtualenv. It calls the single-video runner (run_wilor.py)
once per job and caches the WiLoR model and the YOLO hand detector between jobs, so
both load once per shard. --jobs is a JSON list of {name, video, K, out[, batch_size]};
--status receives one {name, ok[, error]} row per finished job.
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np


# Runs in WiLoR's own virtualenv by file path: imports nothing of gtf (gtf/extraction/hands/ is three levels below the root).
ROOT = Path(__file__).resolve().parents[3]
WILOR_ROOT = ROOT / "third-party" / "wilor" / "WiLoR"
WILOR_RUNNER = Path(__file__).resolve().with_name("run_wilor.py")


def _load_runner():
    os.chdir(WILOR_ROOT)
    sys.path.insert(0, str(WILOR_ROOT))

    import ultralytics
    import wilor.models

    original_load_wilor = wilor.models.load_wilor
    original_yolo = ultralytics.YOLO
    cache = {"model": None, "detector": None}

    def cached_load_wilor(*args, **kwargs):
        if cache["model"] is None:
            cache["model"] = original_load_wilor(*args, **kwargs)
        return cache["model"]

    def cached_yolo(*args, **kwargs):
        if cache["detector"] is None:
            cache["detector"] = original_yolo(*args, **kwargs)
        return cache["detector"]

    wilor.models.load_wilor = cached_load_wilor
    ultralytics.YOLO = cached_yolo

    spec = importlib.util.spec_from_file_location("wilor_single_video", WILOR_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {WILOR_RUNNER}")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def _run_job(runner, job):
    output = Path(job["out"])
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.npz")
    temporary.unlink(missing_ok=True)
    sys.argv = [
        str(WILOR_RUNNER),
        "--video",
        job["video"],
        "--K",
        job["K"],
        "--out",
        str(temporary),
        "--batch-size",
        str(job.get("batch_size", 32)),
    ]
    try:
        runner.main()
        with np.load(temporary, allow_pickle=False) as result:
            if "n_frames" not in result.files or "K" not in result.files:
                raise RuntimeError("WiLoR output is incomplete")
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _write_status(path, rows):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(rows, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()

    jobs = json.loads(args.jobs.read_text())
    runner = _load_runner()
    rows = []
    for index, job in enumerate(jobs, start=1):
        print(f"[persistent_wilor] [{index}/{len(jobs)}] {job['name']}", flush=True)
        try:
            _run_job(runner, job)
            rows.append({"name": job["name"], "ok": True})
        except BaseException as error:
            if isinstance(error, KeyboardInterrupt):
                raise
            rows.append(
                {
                    "name": job["name"],
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(f"[persistent_wilor] FAILED {job['name']}: {error}", flush=True)
        _write_status(args.status, rows)

    if any(not row["ok"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
