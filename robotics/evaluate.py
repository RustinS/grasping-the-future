"""Step 3: SPIDER trials and independent replays for one episode and one arm.

Runs in the SPIDER environment, on one GPU.

    robotics/.venv/bin/python robotics/evaluate.py EPISODE --arm reference|forecast [--out robotics/output]

For each simulator seed 0, 1, 2 this runs SPIDER's `examples/run_mjwp_fast.py` (recipe in common.RECIPE), then
replays the saved hand controls in a fresh one-world MuJoCo Warp environment: free object, gravity on, no object
actuation, no state resets, no replanning. The replay, not the optimizer's own record, is scored.

reference arm: the three seeds decide whether the recording is retained (>= 1 successful replay); written to
    qualification/<episode>.json.
forecast arm (retained recordings only): the three SPIDER seeds, plus one replay of each baseline:
    native_ik   SPIDER's IK joint targets on the original scene and initial state (the "Kinematic" row)
    adapted_ik  the grasp-prepared scene, initial state and controls given to SPIDER, without optimization
                (the "Kinematic + grasp prep" row)
"""

import argparse
import gc
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disable")
from common import (
    DATASET,
    DEFAULT_OUT,
    RECIPE,
    REF_DT,
    ROBOT,
    SEEDS,
    SIM_DT,
    SPIDER,
    baseline_dir,
    check_interpolation,
    contiguous_prefix_length,
    dataset_dir,
    episode,
    execution_reference,
    export_dir,
    measure,
    qualification_path,
    read,
    reference_transferable,
    sha,
    target,
    write,
)

sys.path.insert(0, str(SPIDER))
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from spider.config import Config, filter_config_fields, process_config  # noqa: E402
from spider.io import load_data  # noqa: E402
from spider.simulators.mjwp import load_env_params, setup_env, setup_mj_model, step_env  # noqa: E402

from preprocess import object_id  # noqa: E402

torch.set_num_threads(2)
TERMINAL = {"completed", "initialization_failed", "no_contiguous_controls", "partial_controls", "nonfinite_dynamics"}


def config(out, arm, e, seed=0, samples=None):
    base = OmegaConf.load(SPIDER / "examples/config/default.yaml")
    fast = OmegaConf.load(SPIDER / "examples/config/override/oakinkv2_fast.yaml")
    kwargs = filter_config_fields(OmegaConf.to_container(OmegaConf.merge(base, fast), resolve=True))
    kwargs.update(RECIPE)
    kwargs.update(
        dataset_dir=str(dataset_dir(out, arm)),
        dataset_name=DATASET,
        robot_type=ROBOT,
        embodiment_type=e["mode"],
        task=e["episode_id"],
        data_id=seed,
        seed=seed,
        ref_dt=REF_DT,
        sim_dt=SIM_DT,
        show_viewer=False,
        save_video=False,
        wait_on_finish=False,
        save_viser=False,
        save_rerun=False,
        viewer="mujoco",
        device="cuda:0",
    )
    if samples is not None:
        kwargs["num_samples"] = samples
    return Config(**kwargs)


def physics(out, arm, e, seed):
    """One SPIDER optimization. A failed initial-state sanity check is a (counted) failure, not an error."""
    from examples.run_mjwp_fast import main

    dst = target(out, arm, e, seed)
    dst.mkdir(parents=True, exist_ok=True)
    status = dst / "run.json"
    if status.exists() and read(status)["status"] in ["completed", "failed"]:
        return read(status)
    if seed:
        shutil.copy2(target(out, arm, e) / "trajectory_kinematic.npz", dst / "trajectory_kinematic.npz")
    receipt = dict(
        episode=e["episode_id"],
        arm=arm,
        seed=seed,
        status="running",
        mode=e["mode"],
        kinematic_sha256=sha(dst / "trajectory_kinematic.npz"),
        scene_sha256=sha(dst.parent / "scene.xml"),
        contact_guidance=False,
        num_dyn=1,
    )
    write(status, receipt)
    started = time.monotonic()
    try:
        main(config(out, arm, e, seed))
        artifact = dst / "trajectory_mjwp_fast.npz"
        if not artifact.exists():
            raise RuntimeError("SPIDER produced no trajectory")
        receipt.update(status="completed", trajectory_sha256=sha(artifact))
    except Exception as exc:
        physical = "Initial-state sanity check FAILED" in str(exc)
        receipt.update(
            status="failed" if physical else "error",
            error=f"{type(exc).__name__}: {exc}",
            physical_initialization_failure=physical,
            traceback=traceback.format_exc(),
        )
        if not physical:
            raise
    finally:
        receipt["wall_s"] = time.monotonic() - started
        write(status, receipt)
        gc.collect()
        torch.cuda.empty_cache()
    return receipt


def replay(out, arm, e, seed=0, method="spider"):
    """Integrate controls from the initial state in a fresh world and score the object against the arm's reference."""
    ep = e["episode_id"]
    dst = target(out, arm, e, seed)
    folder = dst if method == "spider" else baseline_dir(out, method, arm, ep)
    folder.mkdir(parents=True, exist_ok=True)
    status = folder / "replay.json"
    if status.exists() and read(status)["status"] in TERMINAL:
        return read(status)
    receipt = dict(
        episode=ep,
        arm=arm,
        seed=seed if method == "spider" else None,
        method=method,
        status="running",
        success=False,
        complete=False,
        coverage=0.0,
        pos_mean_m=None,
        rot_mean_rad=None,
    )
    started = time.monotonic()
    write(status, receipt)
    try:
        optimized = None
        if method == "spider":
            run = read(dst / "run.json")
            if run["status"] == "failed":
                assert run.get("physical_initialization_failure") or "Initial-state sanity check FAILED" in run.get("error", "")
                receipt.update(status="initialization_failed", reason=run["error"])
                return receipt
            assert run["status"] == "completed"
            saved = dst / "trajectory_mjwp_fast.npz"
            assert sha(saved) == run["trajectory_sha256"]
            assert sha(dst / "trajectory_kinematic.npz") == run["kinematic_sha256"]
            assert sha(dst.parent / "scene.xml") == run["scene_sha256"]
            raw = dict(np.load(saved))
            prefix = contiguous_prefix_length(raw["time"])
        cfg = process_config(config(out, arm, e, seed, samples=1))
        if method == "native_ik":
            cfg.model_path = str(dst.parent / "scene_original.xml")
            cfg.data_path = str(dst / "trajectory_kinematic_original.npz")
        assert cfg.num_dyn == 1 and not cfg.contact_guidance and cfg.sim_dt == SIM_DT
        model = setup_mj_model(cfg)
        body, adr = object_id(model, e["mode"])
        assert model.nu == cfg.nu and np.linalg.norm(model.opt.gravity) > 0
        kin = np.load(cfg.data_path)["qpos"]
        ref, times, unpadded = execution_reference(kin)
        export = read(export_dir(out, arm, ep) / "export.json")
        mask = (times >= export["generated_start_s"] - 1e-8) & unpadded
        refs = load_data(cfg, cfg.data_path)
        # SPIDER's own interpolation must agree with our scoring timeline up to float32 roundoff.
        native_ref = refs[0][1 : len(ref)].cpu().numpy()
        receipt["reference_interpolation_roundoff_check"] = check_interpolation(kin, ref[:-1], native_ref)
        if method == "spider":
            controls = raw["ctrl"].reshape(-1, cfg.nu)
            n = min(prefix, len(ref), len(controls))
            optimized = raw["qpos"].reshape(-1, cfg.nq)[:n]
        else:
            controls = refs[2][: len(ref)].cpu().numpy()
            n = len(ref)
        receipt.update(
            object_body=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body),
            object_qpos_address=adr,
            expected_steps=len(ref),
            scene_sha256=sha(cfg.model_path),
            kinematic_sha256=sha(cfg.data_path),
        )
        if not n:
            receipt.update(status="no_contiguous_controls", n_executed_steps=0)
            return receipt
        env = setup_env(cfg, refs)
        load_env_params(cfg, env, {"pair_margin": 0.0, "xy_offset": 0.0})
        qpos = []
        qvel = []
        for u in controls[:n]:
            step_env(cfg, env, torch.as_tensor(u, dtype=torch.float32, device=cfg.device))
            wp.synchronize()
            qp = wp.to_torch(env.data_wp.qpos)[0].cpu().numpy().copy()
            qv = wp.to_torch(env.data_wp.qvel)[0].cpu().numpy().copy()
            if not np.isfinite(qp).all() or not np.isfinite(qv).all():
                break
            qpos.append(qp)
            qvel.append(qv)
        qp = np.array(qpos).reshape(-1, cfg.nq)
        qv = np.array(qvel).reshape(-1, cfg.nv)
        executed = len(qp)
        artifact = folder / "trajectory_replay.npz"
        np.savez_compressed(artifact, qpos=qp, qvel=qv, ctrl=controls[:executed], time=np.arange(1, executed + 1) * SIM_DT)
        if executed:
            receipt.update(measure(qp[:, adr : adr + 7], ref[:, adr : adr + 7], mask, len(ref)))
        receipt.update(
            status="completed" if executed == len(ref) else "partial_controls" if executed == n else "nonfinite_dynamics",
            n_executed_steps=executed,
            trajectory_sha256=sha(artifact),
        )
        if optimized is not None and executed:
            errors = np.linalg.norm(qp[:, adr : adr + 3] - optimized[:executed, adr : adr + 3], axis=1)
            receipt.update(
                max_object_position_difference_from_saved_m=float(errors.max()),
                mean_object_position_difference_from_saved_m=float(errors.mean()),
            )
    except Exception as exc:
        receipt.update(status="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        receipt["wall_s"] = time.monotonic() - started
        write(status, receipt)
        gc.collect()
        torch.cuda.empty_cache()
    return receipt


def run(out, ep, arm):
    out = Path(out)
    folder = out / "work" / arm / ep
    folder.mkdir(parents=True, exist_ok=True)
    os.chdir(folder)
    pre = read(out / "preprocess" / ep / "status.json")
    assert pre["status"] == "completed", pre
    if arm == "forecast":
        assert reference_transferable(out, ep), "The reference of this recording has no successful replay"
    e = episode(out, ep)
    results = []
    for seed in SEEDS:
        physics(out, arm, e, seed)
        results.append(replay(out, arm, e, seed))
        assert results[-1]["status"] not in {"running", "error"}, results[-1]
        print("TRIAL", arm, ep, seed, results[-1]["status"], results[-1]["success"], flush=True)
    if arm == "reference":
        success = [bool(r["success"]) for r in results]
        write(qualification_path(out, ep), dict(episode=ep, status="complete", success_by_seed=success, retained=any(success), retained_all_seeds=all(success)))
    else:
        for method in ["native_ik", "adapted_ik"]:
            r = replay(out, arm, e, 0, method)
            print("BASELINE", method, ep, r["status"], r["success"], flush=True)
    write(folder / "completed.json", dict(episode=ep, arm=arm, status="complete"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("episode")
    p.add_argument("--arm", choices=["reference", "forecast"], default="reference")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = p.parse_args()
    run(a.out.resolve(), a.episode, a.arm)
