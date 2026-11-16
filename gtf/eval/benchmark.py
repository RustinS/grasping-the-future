r"""The paper's prediction benchmark: run a checkpoint on the benchmark cohorts and score Tables 1 and 2.

The benchmark has two populations of five-second validation windows (benchmark/mocap.json: 256 motion-capture
windows; benchmark/action100m.json: 75 internet-video windows extracted by the production chain) and seven tasks.
`run` samples N trajectories per window and task and saves every sample's translations; `tables` scores them:

  Table 1 (all tasks)   one sample, the best of N (the sample with the lowest translation error summed over the
                        predicted entities), the energy score of all N samples, and how often the reference lies
                        inside the N samples (it is no farther from the center than the farthest sample);
                        "end" columns score the last predicted frame of windows whose reference moves at least
                        50 mm (maximum excursion over the predicted span), "middle" columns the middle frame of
                        every gap with observed frames on both sides.
  Table 2 (forecasting) ADE over the predicted frames and FDE at the last one (windows moving >= 50 mm) for the
                        object and each wrist: one sample and the energy score of all N samples.

Energy score per window: mean_i |x_i - y| - 0.5 * mean_{i<j} |x_i - x_j| (mm). Aggregation: mean per dataset,
equal weight over the motion-capture datasets, then 0.5 motion capture + 0.5 internet video; hands average both
wrists. Table 1 uses the default scene caches (sensor depth where a dataset has it). Table 2 scores the
motion-capture population on scene caches built from estimated depth (gtf/datasets/scene/preprocess.py --depth
estimated): run that population into its own directory with the manifest of those caches and pass it to
`tables --table2-mocap-run`.

`run --forecasts` also writes <out>/<population>/future_prediction.pt, the first sample of every window with its
reference as full hand and object states, the input of the robot retargeting (robotics/prepare.py).

Usage:
  python -m gtf.eval.benchmark run --checkpoint CKPT --population mocap --out results/benchmark \
      [--manifest datasets/canonical_manifest.json] [--stats STATS (training checkpoints only)] \
      [--uni3d-cache datasets/uni3d_embeddings.pt] [--tasks future_prediction,...] [--forecasts] [eval.key=value ...]
  python -m gtf.eval.benchmark tables results/benchmark [--table2-mocap-run RUN_DIR] [--json tables.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from gtf.paths import REPO_ROOT

TASKS = ["future_prediction", "hand_to_object", "object_to_hand", "temporal_infilling", "goal_infilling", "goal_hand_only", "goal_object_only"]
POPULATIONS = ("mocap", "action100m")
SLOT = {"hand_left": 0, "hand_right": 1, "obj": 2}
# Table 1 columns: (task, entity group, scored frame)
TABLE1 = [
    ("future_prediction", "obj", "end"), ("future_prediction", "hands", "end"),
    ("hand_to_object", "obj", "end"), ("object_to_hand", "hands", "end"),
    ("temporal_infilling", "obj", "middle"), ("temporal_infilling", "hands", "middle"),
    ("goal_infilling", "obj", "middle"), ("goal_infilling", "hands", "middle"),
    ("goal_hand_only", "obj", "end"), ("goal_hand_only", "hands", "middle"),
    ("goal_object_only", "obj", "middle"), ("goal_object_only", "hands", "end"),
]
GROUPS = {"obj": ["obj"], "hands": ["hand_left", "hand_right"]}
MIN_EXCURSION_MM = 50.0


def _select_windows(dataset, windows):
    """Dataset indices of the cohort windows, identified by (source, segment_id, first video frame), in cohort order."""
    wanted = {(w["source"], w["segment_id"], int(w["first_frame"])): i for i, w in enumerate(windows)}
    segments = {k[:2] for k in wanted}
    found = [None] * len(windows)
    frame_indices = {}
    for index, (segment_index, start, _) in enumerate(dataset.windows):
        entry = dataset.segments[segment_index]
        key = (str(entry["source_dataset"]), str(entry["segment_id"]))
        if key not in segments:
            continue
        if segment_index not in frame_indices:
            with np.load(entry["canonical_path"], allow_pickle=False) as payload:
                frame_indices[segment_index] = payload["frame_indices"].copy()
        i = wanted.get(key + (int(frame_indices[segment_index][start]),))
        if i is not None:
            found[i] = index
    missing = [(windows[i]["source"], windows[i]["segment_id"], windows[i]["first_frame"]) for i, f in enumerate(found) if f is None]
    if missing:
        raise ValueError(f"{len(missing)} benchmark windows are not in the dataset, e.g. {missing[:3]}")
    return found


def run(args, overrides):
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from gtf.eval.checkpoint import config_from_checkpoint, load_lit_module_from_checkpoint
    from gtf.eval.evaluate import evaluate_task, shuffle_windows
    from gtf.data.dataset import GTFDataset, gtf_collate
    from gtf.paths import MANO_DIR

    cohort = json.loads((REPO_ROOT / "benchmark" / f"{args.population}.json").read_text())
    cfg = config_from_checkpoint(args.checkpoint)
    cfg = OmegaConf.merge(cfg, OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides))
    cfg.eval.checkpoint_path = str(args.checkpoint)
    cfg.data.manifest_path = args.manifest
    if args.stats:
        cfg.data.stats_path = args.stats
    cfg.data.uni3d_cache_path = args.uni3d_cache
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit = load_lit_module_from_checkpoint(cfg, device, mano_root=MANO_DIR)

    segment_ids = sorted({w["segment_id"] for w in cohort["windows"]})
    dataset = GTFDataset(
        manifest_path=args.manifest, window_size=int(cfg.eval.window_size), split="val", mano_root=MANO_DIR,
        segment_ids=segment_ids, window_seconds=float(cfg.eval.window_seconds), stride_seconds=float(cfg.eval.stride_seconds),
        uni3d_cache_path=args.uni3d_cache,
    )
    samples = [dataset[i] for i in _select_windows(dataset, cohort["windows"])]
    ordered = shuffle_windows(samples, cfg.eval)
    loader = DataLoader([ordered[i] for i in range(len(ordered))], batch_size=int(cfg.eval.batch_size), shuffle=False, num_workers=0, collate_fn=gtf_collate)

    out = Path(args.out) / args.population
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    for task in args.tasks:
        forecasts = [] if args.forecasts and task == "future_prediction" else None
        if (out / f"{task}.pkl").exists() and (forecasts is None or (out / f"{task}.pt").exists()):
            print(f"{args.population}/{task}: done already", flush=True)
            continue
        raw, records = {}, []
        result = evaluate_task(lit.model, loader, lit.stats, task, cfg.eval, device, raw_out=raw, samples_out=records, forecasts_out=forecasts)
        if len(records) != cohort["n_windows"]:
            raise RuntimeError(f"{task}: {len(records)} windows evaluated, expected {cohort['n_windows']}")
        (out / f"{task}.json").write_text(json.dumps(dict(result=result, act_rows=raw["act_rows"], gap_records=raw["gap_records"]), default=float))
        with open(out / f"{task}.pkl.tmp", "wb") as f:
            pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(out / f"{task}.pkl.tmp", out / f"{task}.pkl")
        if forecasts is not None:
            _save_forecasts(out / f"{task}.pt", task, forecasts, dataset)
        print(f"{args.population}/{task}: {len(records)} windows", flush=True)


def _save_forecasts(path, task, rows, dataset):
    """Write the forecast file of robotics/prepare.py: the evaluator's rows plus each window's object mesh."""
    import torch

    canonical = {str(e["segment_id"]): e["canonical_path"] for e in dataset.segments}
    meshes = {}
    for row in rows:
        sid = row["identity"][1]
        if sid not in meshes:
            with np.load(canonical[sid], allow_pickle=False) as npz:
                meshes[sid] = dict(
                    obj_canonical_verts=torch.from_numpy(npz["obj_canonical_verts"].astype(np.float32)),
                    obj_scale=torch.from_numpy(np.asarray(npz["obj_scale"], dtype=np.float32).reshape(3)),
                    obj_faces=torch.from_numpy(npz["obj_faces"].astype(np.int64)),
                )
        row.update(meshes[sid])
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"task": task, "rows": rows}, tmp)
    os.replace(tmp, path)


def _gap_runs(obs, pres):
    """Masked runs of one entity (same order and rules as the evaluator's gap records); bounded = an observed
    frame on both sides with the entity present from the entry anchor on."""
    T, out, t = len(obs), [], 0
    while t < T:
        if obs[t] or not pres[t]:
            t += 1
            continue
        s = t
        while t < T and not obs[t] and pres[t]:
            t += 1
        e = t - 1
        entry = next((r for r in range(s - 1, -1, -1) if obs[r] and pres[r]), -1)
        exit_ = e + 1 if (e + 1 < T and pres[e + 1] and obs[e + 1]) else -1
        if entry < 0 and exit_ < 0:
            continue
        frames = list(range(s, e + 1))
        bounded = entry >= 0 and bool(pres[entry:s + 1].all()) and exit_ >= 0
        out.append((frames[len(frames) // 2], bounded))
    return out


def _cloud(p, y):
    """p (N, 3) samples, y (3,) reference -> per-sample errors (N,), mean pairwise distance, reference inside (0/1)."""
    n = len(p)
    err = np.linalg.norm(p - y, axis=1)
    pair = np.linalg.norm(p[:, None] - p[None], axis=2)[np.triu_indices(n, 1)].mean() if n > 1 else 0.0
    z = np.vstack([p, y[None]])
    r = np.linalg.norm(z - (z.sum(0, keepdims=True) - z) / n, axis=1)
    return err, pair, float(r[-1] <= r[:-1].max())


def _table1_items(records, kind):
    """(source, entity, frame-scores) for every scored window frame of one task."""
    for rec in records:
        valid = rec["valid"].astype(bool)
        T = int(valid.sum())
        for ent, slot in SLOT.items():
            if ent not in rec:
                continue
            p, g = rec[ent]["pred"].astype(np.float64), rec[ent]["gt"].astype(np.float64)
            obs, pres = rec["obs"][:, slot].astype(bool), rec["presence"][:, slot].astype(bool)
            if kind == "end":
                m = ~obs & pres & valid
                ex = rec[ent]["excursion_mm"]
                if not m.any() or not (math.isfinite(ex) and ex >= MIN_EXCURSION_MM):
                    continue
                frames = [int(np.nonzero(m)[0][-1])]
            else:
                frames = [f for f, bounded in (_gap_runs(obs[:T], pres[:T]) if T >= 3 else []) if bounded]
            j = rec["joint_idx"] if rec["joint_idx"] is not None else 0
            for f in frames:
                err, pair, inside = _cloud(p[:, f], g[f])
                yield rec["source"], ent, dict(single=err[0], best=err[j], es=err.mean() - 0.5 * pair, inside=inside)


def _macro(rows):
    """rows: [(source, value)] -> mean of the per-source means."""
    by = defaultdict(list)
    for s, v in rows:
        by[s].append(v)
    return float(np.mean([np.mean(v) for v in by.values()])) if by else float("nan")


def table1(run_dir):
    cells = {}
    for task, group, kind in TABLE1:
        per_pop = {}
        for pop in POPULATIONS:
            records = pickle.load(open(run_dir / pop / f"{task}.pkl", "rb"))
            items = list(_table1_items(records, kind))
            per_pop[pop] = {}
            for q in ("single", "best", "es", "inside"):
                vals = [_macro([(s, v[q]) for s, e, v in items if e == ent]) for ent in GROUPS[group]]
                vals = [v for v in vals if math.isfinite(v)]
                scale = 100.0 if q == "inside" else 1000.0
                per_pop[pop][q] = scale * float(np.mean(vals)) if vals else float("nan")
        cells[f"{task}/{group}"] = {q: 0.5 * (per_pop["mocap"][q] + per_pop["action100m"][q]) for q in ("single", "best", "es", "inside")} | {"by_population": per_pop}
    return cells


def table2(run_dir, mocap_run_dir=None):
    rows = defaultdict(list)  # (entity, metric, protocol) -> [(population, source, value)]
    for pop in POPULATIONS:
        src = mocap_run_dir if pop == "mocap" and mocap_run_dir is not None else run_dir
        for rec in pickle.load(open(src / pop / "future_prediction.pkl", "rb")):
            valid = rec["valid"].astype(bool)
            for ent, slot in SLOT.items():
                if ent not in rec:
                    continue
                scored = ~rec["obs"][:, slot].astype(bool) & rec["presence"][:, slot].astype(bool) & valid
                if not scored.any():
                    continue
                p, y = rec[ent]["pred"][:, scored].astype(np.float64), rec[ent]["gt"][scored].astype(np.float64)
                n = len(p)
                iu = np.triu_indices(n, 1)
                ade = np.linalg.norm(p - y[None], axis=-1).mean(-1)
                ade_pair = np.linalg.norm(p[:, None] - p[None], axis=-1).mean(-1)[iu].mean() if n > 1 else 0.0
                rows[ent, "ADE", "single"].append((pop, rec["source"], ade[0]))
                rows[ent, "ADE", "es"].append((pop, rec["source"], ade.mean() - 0.5 * ade_pair))
                ex = rec[ent]["excursion_mm"]
                if math.isfinite(ex) and ex >= MIN_EXCURSION_MM:
                    fde = np.linalg.norm(p[:, -1] - y[-1], axis=-1)
                    fde_pair = np.linalg.norm(p[:, None, -1] - p[None, :, -1], axis=-1)[iu].mean() if n > 1 else 0.0
                    rows[ent, "FDE", "single"].append((pop, rec["source"], fde[0]))
                    rows[ent, "FDE", "es"].append((pop, rec["source"], fde.mean() - 0.5 * fde_pair))
    out = {}
    for (ent, metric, proto), vals in rows.items():
        pops = {pop: 1000.0 * _macro([(s, v) for p_, s, v in vals if p_ == pop]) for pop in POPULATIONS}
        out[f"{ent}/{metric}/{proto}"] = dict(combined=0.5 * (pops["mocap"] + pops["action100m"]), **pops)
    return out


def tables(args):
    run_dir = Path(args.run_dir)
    t1, t2 = table1(run_dir), table2(run_dir, Path(args.table2_mocap_run) if args.table2_mocap_run else None)
    print("Table 1 (mm; inside in %)")
    print(f"{'column':30s} {'one sample':>11s} {'best of N':>10s} {'energy':>8s} {'inside':>7s}")
    for task, group, kind in TABLE1:
        c = t1[f"{task}/{group}"]
        print(f"{task + ' / ' + group + ' (' + kind + ')':30s} {c['single']:11.1f} {c['best']:10.1f} {c['es']:8.1f} {c['inside']:7.0f}")
    print("\nTable 2, future prediction (mm)")
    for proto, label in (("es", "energy score"), ("single", "one sample")):
        cells = [t2.get(f"{ent}/{m}/{proto}", {}).get("combined", float("nan")) for ent in ("obj", "hand_left", "hand_right") for m in ("ADE", "FDE")]
        print(f"{label:14s} object ADE/FDE {cells[0]:6.1f} {cells[1]:6.1f}   left wrist {cells[2]:6.1f} {cells[3]:6.1f}   right wrist {cells[4]:6.1f} {cells[5]:6.1f}")
    if args.json:
        Path(args.json).write_text(json.dumps(dict(table1=t1, table2=t2), indent=1))


def main():
    parser = argparse.ArgumentParser(prog="python -m gtf.eval.benchmark", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="sample the benchmark windows of one population")
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--population", choices=POPULATIONS, required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--manifest", default="datasets/canonical_manifest.json")
    r.add_argument("--stats", default=None, help="normalization statistics of a training checkpoint (an exported one carries them)")
    r.add_argument("--uni3d-cache", default="datasets/uni3d_embeddings.pt")
    r.add_argument("--config", default=str(REPO_ROOT / "configs" / "benchmark.yaml"))
    r.add_argument("--tasks", default=",".join(TASKS), help="comma-separated subset of the seven tasks")
    r.add_argument("--forecasts", action="store_true", help="also write future_prediction.pt, the input of robotics/prepare.py")
    t = sub.add_parser("tables", help="score Tables 1 and 2 from a run directory holding both populations")
    t.add_argument("run_dir")
    t.add_argument("--table2-mocap-run", help="run directory of the motion-capture population on estimated-depth scene caches (Table 2)")
    t.add_argument("--json")
    args, overrides = parser.parse_known_args()
    if args.command == "run":
        args.tasks = args.tasks.split(",")
        unknown = sorted(set(args.tasks) - set(TASKS))
        if unknown:
            parser.error(f"unknown tasks {unknown}")
    run(args, overrides) if args.command == "run" else tables(args)


if __name__ == "__main__":
    main()
