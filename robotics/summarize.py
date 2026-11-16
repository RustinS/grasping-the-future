"""Step 4: success rates over the retained recordings (the three rows of the paper's table).

Runs in either environment (it needs NumPy).

    robotics/.venv/bin/python robotics/summarize.py [--out robotics/output]

Writes trials.csv (every forecast-arm execution of the retained recordings), qualification.csv (reference outcome per
recording) and results.json (rates, recording-bootstrap 95% intervals, per-source and per-mode breakdowns, and the
stricter all-three-reference-seeds filter), then prints the table.
"""

import argparse
import collections
import csv
from pathlib import Path

import numpy as np

from common import DEFAULT_OUT, LABELS, METHODS, SEEDS, SOURCES, baseline_dir, qualification_path, read, target, write


def fetch(out, e, method, arm="forecast"):
    ep = e["episode_id"]
    if method == "spider":
        paths = [target(out, arm, e, s) / "replay.json" for s in SEEDS]
    else:
        paths = [baseline_dir(out, method, arm, ep) / "replay.json"]
    rows = []
    for i, p in enumerate(paths):
        r = read(p) if p.exists() else dict(status="pending", success=False, complete=False, coverage=0.0)
        r.update(episode=ep, source=e["source"], mode=e["mode"], arm=arm, method=method, seed=SEEDS[i] if method == "spider" else None)
        rows.append(r)
    return rows


def aggregate(out, episodes):
    """Per method: successes / executions, and a bootstrap over recordings (seeds stay grouped)."""
    result = dict(n_recordings=len(episodes), methods={}, paired_differences={})
    rng = np.random.default_rng(0)
    n = len(episodes)
    samples = rng.integers(n, size=(10000, n)) if n else None
    values = {}
    for method in METHODS:
        grouped = [fetch(out, e, method) for e in episodes]
        rows = [r for g in grouped for r in g]
        pending = sum(r["status"] in ["pending", "running", "error", "failed"] for r in rows)
        v = np.array([np.mean([float(r["success"]) for r in g]) for g in grouped])
        values[method] = v
        result["methods"][method] = dict(
            n_recordings=n,
            n_executions=len(rows),
            n_success=sum(bool(r["success"]) for r in rows),
            n_pending=pending,
            success_rate=float(v.mean()) if n and not pending else None,
            success_rate_ci95=np.percentile(v[samples].mean(1), [2.5, 97.5]).tolist() if n and not pending else None,
            status_counts=dict(collections.Counter(r["status"] for r in rows)),
        )
    for baseline in METHODS[:2]:
        delta = values["spider"] - values[baseline]
        pending = result["methods"]["spider"]["n_pending"] + result["methods"][baseline]["n_pending"]
        result["paired_differences"][f"spider_minus_{baseline}"] = dict(
            difference=float(delta.mean()) if n and not pending else None,
            ci95=np.percentile(delta[samples].mean(1), [2.5, 97.5]).tolist() if n and not pending else None,
        )
    return result


def write_csv(path, rows):
    fields = sorted(set().union(*(r.keys() for r in rows))) if rows else ["episode"]
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main(out):
    episodes = read(out / "cohort.json")["episodes"]
    qualification = []
    pending = []
    for e in episodes:
        p = qualification_path(out, e["episode_id"])
        if not p.exists():
            pre = out / "preprocess" / e["episode_id"] / "status.json"
            if pre.exists() and read(pre)["status"] == "failed":
                # No robot reference could be built: the recording cannot be retained.
                qualification.append(dict(e, success_by_seed=[False] * len(SEEDS), retained=False, retained_all_seeds=False, preprocess_failed=True))
            else:
                pending.append(e["episode_id"])
            continue
        q = read(p)
        assert q["status"] == "complete"
        assert q["retained"] == any(q["success_by_seed"]) and q["retained_all_seeds"] == all(q["success_by_seed"])
        qualification.append(dict(e, **{k: v for k, v in q.items() if k != "episode"}))
    retained = [e for e in qualification if e["retained"]]
    strict = [e for e in qualification if e["retained_all_seeds"]]
    result = dict(
        status="partial",
        n_candidate_recordings=len(episodes),
        n_reference_qualified=len(qualification),
        n_pending_reference=len(pending),
        pending_reference=pending,
        n_retained=len(retained),
        aggregate=aggregate(out, retained),
        by_source={
            s: dict(n_candidate_recordings=sum(e["source"] == s for e in episodes), **aggregate(out, [e for e in retained if e["source"] == s]))
            for s in SOURCES
            if any(e["source"] == s for e in episodes)
        },
        by_mode={m: aggregate(out, [e for e in retained if e["mode"] == m]) for m in ["right", "left", "bimanual"]},
        sensitivity_all_reference_seeds=aggregate(out, strict),
        retained_episode_ids=[e["episode_id"] for e in retained],
    )
    if not pending and not any(x["n_pending"] for x in result["aggregate"]["methods"].values()):
        result["status"] = "complete"
    write(out / "results.json", result)
    write_csv(out / "trials.csv", [r for e in retained for m in METHODS for r in fetch(out, e, m)])
    write_csv(
        out / "qualification.csv",
        [{k: e[k] for k in ["episode_id", "source", "mode", "retained", "retained_all_seeds", "success_by_seed"]} for e in qualification],
    )
    print(f"Status: {result['status']}. {len(qualification)}/{len(episodes)} references qualified, {len(retained)} recordings retained.")
    for m in METHODS:
        g = result["aggregate"]["methods"][m]
        rate = f"{100 * g['n_success'] / g['n_executions']:.1f}%" if g["n_executions"] else "--"
        extra = f", {g['n_pending']} pending" if g["n_pending"] else ""
        print(f"{LABELS[m]:<24} {g['n_success']}/{g['n_executions']} ({rate}{extra})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    main(p.parse_args().out.resolve())
