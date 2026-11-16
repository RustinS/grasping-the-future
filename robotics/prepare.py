"""Step 1: freeze the cohort and export each forecast and its reference into SPIDER's MANO keypoint format.

Runs in the project environment (it needs torch, the forecast files and the project's MANO layer).

    python robotics/prepare.py --forecast FORECAST.pt [FORECAST.pt ...] [--out robotics/output]

`--forecast` takes the forecast files of `python -m gtf.eval.benchmark run --forecasts` (format: robotics/README.md). By default the
cohort is the paper's 170 windows (`robotics/cohort.json`); every listed window must be present in the forecast files.
`--all-eligible` instead takes every eligible window, one per segment.

For each episode and each arm (reference = recorded human motion, forecast = model output), this writes
    arms/<arm>/dataset/processed/forecast/mano/<mode>/<episode>/0/trajectory_keypoints.npz   SPIDER's input
    arms/<arm>/dataset/processed/forecast/mano/<mode>/<episode>/task_info.json
    arms/<arm>/dataset/processed/forecast/assets/objects/<episode>/visual.obj               object mesh
    arms/<arm>/episodes/<episode>/{generated.npz, export.json}                              joints for contact, timing
"""

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from common import (
    ARMS,
    DATASET,
    DEFAULT_COHORT,
    DEFAULT_MANO,
    DEFAULT_OUT,
    LEFT_PALM_OFFSET,
    MIRROR_X,
    PALM_OFFSET,
    RECIPE,
    REF_DT,
    REPO,
    ROBOT,
    SEEDS,
    SOURCES,
    SPIDER,
    SPIDER_COMMIT,
    continuous_quaternions,
    dataset_dir,
    export_dir,
    linear,
    read,
    sha,
    uniform_times,
    write,
)

SELECTION_SALT = "spider-expanded-20260925"


def catalog_row(row):
    """Eligibility of one window. Reads only the reference, the masks and the mesh, never the forecast."""
    n = int(row["valid_frames"].sum())
    presence = row["presence_mask"][:n, :3].numpy()
    pos = row["gt_decoded"]["obj_translation"][:n].numpy()
    times = row["frame_times_s"][:n].numpy()
    vertices = row["obj_canonical_verts"].numpy() * row["obj_scale"].numpy()
    valid = row["valid_frames"].numpy()
    reasons = []
    if n < 3 or not valid[:n].all() or valid[n:].any():
        reasons.append("short_or_noncontiguous")
    stable = n > 0 and bool((presence == presence[0]).all() and presence[0, 2] and presence[0, :2].any())
    if not stable:
        reasons.append("changing_hand_or_object_presence")
    motion = float(np.linalg.norm(pos - pos[:1], axis=1).max()) if n else 0.0
    if motion < 0.05:
        reasons.append("reference_excursion_below_50mm")
    if not np.isfinite(vertices).all() or np.ptp(vertices, axis=0).max() > 0.5:
        reasons.append("invalid_or_large_mesh")
    if not np.isfinite(pos).all() or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        reasons.append("invalid_timing_or_reference")
    sides = [s for s, i in [("right", 1), ("left", 0)] if n and presence[0, i]]
    mode = "bimanual" if len(sides) == 2 else sides[0] if sides else None
    obs = row["obs_mask"][:n, :3].numpy()[:, presence[0]].all(axis=1) if stable else np.zeros(n, dtype=bool)
    nobs = int(obs.sum())
    if not 0 < nobs < n or not obs[:nobs].all() or obs[nobs:].any():
        reasons.append("invalid_observed_prefix")
    source, segment, first = row["identity"]
    return dict(
        identity=[source, segment, int(first)],
        source=source,
        mode=mode,
        sides=sides,
        n_frames=n,
        observed_frames=nobs,
        duration_s=float(times[-1] - times[0]) if n else 0.0,
        reference_excursion_m=motion,
        eligible=not reasons,
        reasons=reasons,
    )


def load_forecasts(paths):
    import torch

    rows = {}
    files = []
    for i, path in enumerate(paths):
        data = torch.load(path, map_location="cpu", weights_only=False)
        if data.get("task", "future_prediction") != "future_prediction":
            raise ValueError(f"{path}: expected a future_prediction forecast file, found {data.get('task')}")
        files.append(dict(path=str(Path(path).resolve()), sha256=sha(path), n_rows=len(data["rows"])))
        for j, row in enumerate(data["rows"]):
            key = (row["identity"][0], row["identity"][1], int(row["identity"][2]))
            if key in rows:
                raise ValueError(f"window {key} appears twice in the forecast files")
            rows[key] = (i, j, row)
    return files, rows


def freeze(args):
    out = args.out
    if (out / "cohort.json").exists():
        print("Cohort already frozen:", out / "cohort.json")
        return
    files, rows = load_forecasts(args.forecast)
    catalog = {key: dict(catalog_row(row), forecast_file=i, forecast_row=j) for key, (i, j, row) in rows.items()}
    if args.all_eligible:
        # One window per segment, in a fixed hash order that depends only on the window identity.
        def order(r):
            return hashlib.sha256(json.dumps([SELECTION_SALT, r["identity"]], sort_keys=True).encode()).hexdigest()

        episodes, used = [], set()
        for source in SOURCES + sorted({r["source"] for r in catalog.values()} - set(SOURCES)):
            for r in sorted((r for r in catalog.values() if r["source"] == source and r["eligible"]), key=order):
                if (source, r["identity"][1]) in used:
                    continue
                used.add((source, r["identity"][1]))
                episodes.append(dict(r, episode_id=f"ep{len(episodes):03d}"))
    else:
        listed = read(args.cohort)["episodes"]
        episodes, missing = [], []
        for item in listed:
            key = (item["source"], item["segment_id"], int(item["first_frame"]))
            if key not in catalog:
                missing.append(key)
                continue
            r = catalog[key]
            if not r["eligible"] or r["mode"] != item["mode"]:
                raise ValueError(f"{item['episode_id']} {key}: not eligible in this forecast file ({r['reasons']}, mode {r['mode']})")
            episodes.append(dict(r, episode_id=item["episode_id"]))
        if missing:
            raise ValueError(f"{len(missing)} cohort windows are not in the forecast files, e.g. {missing[:3]}")
    protocol = dict(
        task="future_prediction",
        optimizer=RECIPE,
        simulator_seeds=SEEDS,
        spider_commit=SPIDER_COMMIT,
        eligibility="Stable nonempty hand/object presence, >=50mm reference object excursion, <=0.50m mesh extent, finite contiguous timestamps, contiguous observed prefix.",
        retained="A recording is retained iff the reference succeeds under SPIDER in at least one of seeds 0,1,2; every retargeting method is scored on the same retained recordings.",
        success="Complete independent dynamic replay AND mean future object position error <0.10m AND SO(3) error <0.50rad; observed prefix and padded endpoint excluded.",
    )
    write(out / "protocol.json", protocol)
    write(
        out / "cohort.json",
        dict(
            forecasts=files,
            episodes=episodes,
            n_episodes=len(episodes),
            counts=dict(Counter(e["source"] for e in episodes)),
            modes=dict(Counter(e["mode"] for e in episodes)),
            n_windows=len(catalog),
            n_eligible_windows=sum(r["eligible"] for r in catalog.values()),
        ),
    )
    print("FROZEN", len(episodes), dict(Counter(e["source"] for e in episodes)), dict(Counter(e["mode"] for e in episodes)), flush=True)


def export_episode(e, row, out, mano, torch, axis_angle_to_6d):
    import trimesh

    ep, n, nobs, mode = e["episode_id"], e["n_frames"], e["observed_frames"], e["mode"]
    real = "left" if mode == "left" else "right"  # the side that carries the real object in SPIDER's scene
    times = row["frame_times_s"][:n].double().numpy().copy()
    times -= times[0]
    dst = uniform_times(times)
    verts = row["obj_canonical_verts"].numpy() * row["obj_scale"].numpy()
    faces = row["obj_faces"].numpy()
    # World frame: rotate to +Z up, put the reference object's first position at the origin in x/y and its lowest
    # vertex on z=0. Both arms share this frame.
    Q = Rotation.from_euler("x", -90, degrees=True).as_matrix()
    p0 = row["gt_decoded"]["obj_translation"][0].numpy()
    r0 = Rotation.from_rotvec(row["gt_decoded"]["obj_rotation"][0].numpy())
    offset = -(Q @ p0)
    offset[2] = -((r0.apply(verts) + p0) @ Q.T)[:, 2].min()
    # The forecast must reproduce the observed prefix it was given.
    for field in ["obj_translation", "obj_rotation"] + [f"hand_{side}_{f}" for side in e["sides"] for f in ["orient", "pose", "cam_t"]]:
        torch.testing.assert_close(row["decoded"][field][:nobs], row["gt_decoded"][field][:nobs], atol=2e-5, rtol=2e-5)

    def resample(x):
        return Slerp(times, Rotation.from_rotvec(x))(np.clip(dst, times[0], times[-1]))

    def pose7(pos, rot):
        return np.c_[pos, continuous_quaternions(Rotation.from_matrix(rot).as_quat()[:, [3, 0, 1, 2]])]

    def ten(x):
        return torch.tensor(x, dtype=torch.float32)

    for arm in ARMS:
        if (export_dir(out, arm, ep) / "export.json").exists():
            continue
        decoded = row["gt_decoded" if arm == "reference" else "decoded"]
        obj = pose7(
            linear(decoded["obj_translation"][:n].numpy(), times, dst) @ Q.T + offset,
            Q @ resample(decoded["obj_rotation"][:n].numpy()).as_matrix(),
        )
        zero = np.zeros_like(obj)
        zero[:, 3] = 1
        kp = {f"qpos_obj_{s}": obj if s == real else zero.copy() for s in ["right", "left"]}
        generated = dict(times=dst, object_poses=obj, object_vertices=verts, object_faces=faces)
        for side in ["right", "left"]:
            tips = np.zeros((len(dst), 5, 7))
            tips[:, :, 3] = 1
            if side not in e["sides"]:
                kp[f"qpos_wrist_{side}"] = zero.copy()
                kp[f"qpos_finger_{side}"] = tips
                continue
            orient = resample(decoded[f"hand_{side}_orient"][:n].numpy())
            pose = decoded[f"hand_{side}_pose"][:n].numpy().reshape(n, 15, 3)
            pose = np.stack([resample(pose[:, j]).as_rotvec() for j in range(15)], axis=1)
            trans = linear(decoded[f"hand_{side}_cam_t"][:n].numpy(), times, dst)
            with torch.no_grad():
                _, joints, _ = mano(
                    axis_angle_to_6d(ten(orient.as_rotvec())),
                    axis_angle_to_6d(ten(pose)).reshape(len(dst), 90),
                    row[f"betas_{side}"][None].expand(len(dst), 10),
                    ten(trans),
                    side,
                )
            joints = joints.numpy() @ Q.T + offset
            R = orient.as_matrix()
            palm = Q @ (R @ PALM_OFFSET if side == "right" else MIRROR_X @ R @ MIRROR_X @ LEFT_PALM_OFFSET)
            kp[f"qpos_wrist_{side}"] = pose7(joints[:, 0], palm)
            tips[:, :, :3] = joints[:, [4, 8, 12, 16, 20]]
            kp[f"qpos_finger_{side}"] = tips
            generated[f"hand_{side}_joints"] = joints
        generated["hand_joints"] = generated[f"hand_{real}_joints"]
        root = dataset_dir(out, arm)
        folder = root / "processed" / DATASET / "mano" / mode / ep / "0"
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "trajectory_keypoints.npz", **kp)
        mesh = root / "processed" / DATASET / "assets/objects" / ep
        mesh.mkdir(parents=True, exist_ok=True)
        trimesh.Trimesh(verts, faces, process=False).export(mesh / "visual.obj")
        shutil.copytree(SPIDER / "spider/assets/robots" / ROBOT, root / "processed" / DATASET / "assets/robots" / ROBOT, dirs_exist_ok=True)
        info = dict(
            task=ep,
            dataset_name=DATASET,
            robot_type="mano",
            embodiment_type=mode,
            data_id=0,
            ref_dt=REF_DT,
            n_frames=len(dst),
            right_object_mesh_dir=str(mesh.relative_to(root)) if real == "right" else None,
            left_object_mesh_dir=str(mesh.relative_to(root)) if real == "left" else None,
            obj_first_frame_lowest_world_z=0.0,
        )
        info[f"{real}_obj_first_frame_xy"] = obj[0, :2].tolist()
        write(folder.parent / "task_info.json", info)
        dest = export_dir(out, arm, ep)
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest / "generated.npz", **generated)
        write(
            dest / "export.json",
            dict(
                episode_id=ep,
                arm=arm,
                identity=e["identity"],
                mode=mode,
                sides=e["sides"],
                n_frames=n,
                n_export_frames=len(dst),
                observed_end_s=float(times[nobs - 1]),
                generated_start_s=float(times[nobs]),
                world_rotation=Q.tolist(),
                world_offset=offset.tolist(),
                keypoints_sha256=sha(folder / "trajectory_keypoints.npz"),
                generated_sha256=sha(dest / "generated.npz"),
            ),
        )


def export(args):
    import torch

    sys.path.insert(0, str(REPO))  # the repository root, so the gtf package imports without being installed
    from gtf.model.repr.mano import MANOHelper
    from gtf.model.repr.rotation import axis_angle_to_6d

    out = args.out
    if not (SPIDER / "spider/assets/robots" / ROBOT).is_dir():
        raise SystemExit(f"SPIDER not found at {SPIDER}; run robotics/third_party.sh or set SPIDER_ROOT")
    if not (Path(args.mano_dir) / "MANO_RIGHT.pkl").exists():
        raise SystemExit(f"MANO_RIGHT.pkl not found in {args.mano_dir}")
    torch.set_num_threads(2)
    mano = MANOHelper(str(args.mano_dir)).eval()
    cohort = read(out / "cohort.json")
    files = cohort["forecasts"]
    forecasts = {}
    for i, f in enumerate(files):
        if sha(f["path"]) != f["sha256"]:
            raise ValueError(f"forecast file changed since the cohort was frozen: {f['path']}")
        forecasts[i] = torch.load(f["path"], map_location="cpu", weights_only=False)["rows"]
    for e in cohort["episodes"]:
        row = forecasts[e["forecast_file"]][e["forecast_row"]]
        assert [row["identity"][0], row["identity"][1], int(row["identity"][2])] == e["identity"]
        export_episode(e, row, out, mano, torch, axis_angle_to_6d)
        print("EXPORTED", e["episode_id"], e["source"], e["mode"], flush=True)
    write(out / "prepared.json", dict(status="complete", n_episodes=len(cohort["episodes"]), cohort_sha256=sha(out / "cohort.json")))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--forecast", type=Path, nargs="+", help="forecast file(s) in the format of robotics/README.md (needed to freeze the cohort)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cohort", type=Path, default=DEFAULT_COHORT, help="window list to use (default: the paper's 170 windows)")
    p.add_argument("--all-eligible", action="store_true", help="ignore --cohort; take every eligible window, one per segment")
    p.add_argument("--mano-dir", type=Path, default=DEFAULT_MANO, help="folder with MANO_RIGHT.pkl")
    p.add_argument("--stage", choices=["all", "cohort", "export"], default="all")
    args = p.parse_args()
    args.out = args.out.resolve()
    if args.stage in ["all", "cohort"]:
        if not (args.out / "cohort.json").exists() and not args.forecast:
            p.error("--forecast is required to freeze the cohort")
        freeze(args)
    if args.stage in ["all", "export"]:
        export(args)


if __name__ == "__main__":
    main()
