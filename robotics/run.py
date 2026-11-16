"""Run preprocessing and evaluation for every episode of a prepared output folder on one node, then summarize.

    robotics/.venv/bin/python robotics/run.py [--out robotics/output] [--python robotics/.venv/bin/python] [--gpus 0 1 2 3]
                           [--workers-per-gpu 2] [--cpu-workers 8] [--threads 16] [--episodes ep000 ep001 ...]

Each step is a separate process in the SPIDER environment: preprocess.py on CPU (up to --cpu-workers at a time), then
evaluate.py --arm reference on a GPU slot, then, if the reference succeeded in at least one seed, evaluate.py --arm
forecast. A forecast trial may start as soon as one reference replay has succeeded. Completed steps leave receipts, so
rerunning the command resumes; a failed step is reported and not retried (logs in <out>/logs/). Prepare first
(prepare.py), which needs the project environment.
"""

import argparse
import collections
import os
import subprocess
import time
from pathlib import Path

from common import DEFAULT_OUT, HERE, qualification_path, read, reference_transferable, write

DEFAULT_PYTHON = HERE / ".venv/bin/python"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--python", type=Path, default=DEFAULT_PYTHON, help="Python of the SPIDER environment")
    p.add_argument("--gpus", type=int, nargs="+", default=[0])
    p.add_argument("--workers-per-gpu", type=int, default=2)
    p.add_argument("--cpu-workers", type=int, default=8)
    p.add_argument("--threads", type=int, default=16, help="CPU threads per grasp search")
    p.add_argument("--episodes", nargs="*", help="only these episodes (default: the whole cohort)")
    a = p.parse_args()
    out = a.out.resolve()
    episodes = [e["episode_id"] for e in read(out / "cohort.json")["episodes"]]
    if a.episodes:
        episodes = [ep for ep in episodes if ep in set(a.episodes)]
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    active = {}  # (step, episode) -> (process, log file, gpu)
    errors = {}

    def launch(key, gpu=None):
        step, ep = key
        env = dict(os.environ, MUJOCO_GL="disable", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2", CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu))
        if step == "preprocess":
            cmd = [str(a.python), "-u", str(HERE / "preprocess.py"), ep, "--out", str(out), "--threads", str(a.threads)]
        else:
            cmd = [str(a.python), "-u", str(HERE / "evaluate.py"), ep, "--arm", step, "--out", str(out)]
        f = (logs / f"{step}_{ep}.log").open("a")
        active[key] = (subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=HERE), f, gpu)
        print("LAUNCH", step, ep, "cpu" if gpu is None else f"gpu{gpu}", flush=True)

    def preprocessed(ep):
        s = out / "preprocess" / ep / "status.json"
        return read(s)["status"] if s.exists() else None

    while True:
        for key, (proc, f, _) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            f.close()
            del active[key]
            print("EXIT", *key, code, flush=True)
            if code:
                errors["/".join(key)] = dict(returncode=code, log=str(logs / f"{key[0]}_{key[1]}.log"))
        pre = {ep: preprocessed(ep) for ep in episodes}
        qualified = {ep for ep in episodes if qualification_path(out, ep).exists()}
        complete = {ep for ep in episodes if (out / "work/forecast" / ep / "completed.json").exists()}
        # CPU: preprocessing.
        # A receipt left at "running" by a killed process is started again.
        pending = [ep for ep in episodes if pre[ep] not in ("completed", "failed") and ("preprocess", ep) not in active and "preprocess/" + ep not in errors]
        for ep in pending[: a.cpu_workers - sum(k[0] == "preprocess" for k in active)]:
            launch(("preprocess", ep))
        # GPU: forecast trials of retained recordings first, then reference trials.
        todo = [("forecast", ep) for ep in episodes if ep not in complete and pre[ep] == "completed" and reference_transferable(out, ep)]
        todo += [("reference", ep) for ep in episodes if pre[ep] == "completed" and ep not in qualified]
        todo = [k for k in todo if k not in active and "/".join(k) not in errors]
        occupied = collections.Counter(g for _, _, g in active.values() if g is not None)
        free = [g for g in a.gpus for _ in range(a.workers_per_gpu - occupied[g])]
        for key, gpu in zip(todo, free):
            launch(key, gpu)
        write(
            out / "progress.json",
            dict(
                episodes=len(episodes),
                preprocessed=sum(s == "completed" for s in pre.values()),
                preprocess_failed=sum(s == "failed" for s in pre.values()),
                reference_qualified=len(qualified),
                retained=sum(read(qualification_path(out, ep))["retained"] for ep in qualified),
                forecast_completed=len(complete),
                active=[dict(step=k[0], episode=k[1], gpu=v[2]) for k, v in active.items()],
                errors=errors,
            ),
        )
        if not active:
            stuck = [ep for ep in episodes if pre[ep] not in ("completed", "failed") and "preprocess/" + ep not in errors]
            waiting = [ep for ep in episodes if pre[ep] == "completed" and (ep not in qualified or (read(qualification_path(out, ep))["retained"] and ep not in complete))]
            if not stuck and not [ep for ep in waiting if not any(k.endswith("/" + ep) for k in errors)]:
                break
        time.sleep(5)
    if errors:
        print("Steps with errors:", errors, flush=True)
    subprocess.run([str(a.python), str(HERE / "summarize.py"), "--out", str(out)], cwd=HERE, check=False)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
