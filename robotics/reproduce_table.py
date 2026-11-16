"""Print the paper's retargeting table from a per-trial CSV (standard library only).

    python robotics/reproduce_table.py [TRIALS_CSV] [--by-source]

TRIALS_CSV defaults to robotics/results/trials.csv (the paper's run); robotics/summarize.py writes the same file for a
new run (<out>/trials.csv). Each row is one execution on a retained recording: one per baseline, three (seeds 0, 1, 2)
for SPIDER. Needed columns: episode, method (native_ik / adapted_ik / spider), seed, success; source is used by
--by-source; rows with arm == reference are ignored.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

METHODS = ["native_ik", "adapted_ik", "spider"]
LABELS = {"native_ik": "Kinematic", "adapted_ik": "Kinematic + grasp prep", "spider": "Dynamic (SPIDER)"}
EXECUTIONS = {"native_ik": 1, "adapted_ik": 1, "spider": 3}


def table(rows):
    episodes = {r["episode"] for r in rows}
    out = {}
    for m in METHODS:
        rs = [r for r in rows if r["method"] == m]
        per_episode = defaultdict(list)
        for r in rs:
            per_episode[r["episode"]].append(r.get("seed", ""))
        # Every retained recording has exactly one execution per baseline and three distinct SPIDER seeds.
        assert set(per_episode) == episodes, f"{m}: not every retained recording has a trial"
        assert all(len(s) == EXECUTIONS[m] and len(set(s)) == len(s) for s in per_episode.values()), f"{m}: wrong number of executions"
        k = sum(r["success"] in ("True", "true", "1") for r in rs)
        out[m] = (k, len(rs))
    return len(episodes), out


def show(name, rows):
    n, result = table(rows)
    print(f"{name}: {n} retained recordings")
    print(f"  {'Retargeting method':<24} {'Successes':>10} {'Rate (%)':>9}")
    for m in METHODS:
        k, total = result[m]
        print(f"  {LABELS[m]:<24} {f'{k}/{total}':>10} {100 * k / total:>9.1f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trials", type=Path, nargs="?", default=Path(__file__).resolve().parent / "results/trials.csv")
    p.add_argument("--by-source", action="store_true")
    a = p.parse_args()
    with a.trials.open() as f:
        rows = [r for r in csv.DictReader(f) if r.get("arm", "forecast") != "reference"]
    show("All", rows)
    if a.by_source:
        for source in sorted({r["source"] for r in rows}):
            show(source, [r for r in rows if r["source"] == source])


if __name__ == "__main__":
    main()
