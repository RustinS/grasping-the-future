"""Step 2: SPIDER preprocessing, observed-state grasp preparation and the grasp adapter, for one episode.

Runs in the SPIDER environment, on CPU.

    robotics/.venv/bin/python robotics/preprocess.py EPISODE [--out robotics/output] [--threads 24]

1. SPIDER `decompose` (CoACD) and `generate_xml` build the XHand scene from the reference arm's object mesh; the
   forecast arm reuses the same scene. SPIDER `ik_fast` turns each arm's wrist and fingertip keypoints into an XHand
   joint trajectory (kept as trajectory_kinematic_original.npz and scene_original.xml: the "Kinematic" baseline).
2. Initial contact is decided from observed frames only: the object counts as held if, in >= 80% of the first (at
   most ten, and only observed) 50 Hz frames, at least two fingertips are within 3 cm of the object surface. Contact
   is lost after ten consecutive frames without it; the release then blends out over 15 frames.
3. For an initially held object, a fixed CEM search (30 iterations x 256 candidates, seed 0) finds a robot grasp and
   motor controls that hold the free object still for 3 s (<= 10 mm, <= 0.30 rad) from the observed initial state.
   The floor is lowered 0.30 m so it cannot support the object. Both arms share this initial grasp.
4. The adapter moves each arm's robot wrist rigidly with that arm's own object trajectory while its contact weight is
   1 and keeps the prepared grip (plus gravity feed-forward), then releases following that arm's own contact. Object
   targets are never changed. The result is trajectory_kinematic.npz + scene.xml: the input to SPIDER, and the
   "Kinematic + grasp prep" baseline when replayed as is.
"""

import argparse
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "disable")
from common import ARMS, DATASET, DEFAULT_OUT, ROBOT, SPIDER, dataset_dir, episode, export_dir, object_errors, read, sha, target, write

sys.path.insert(0, str(SPIDER))
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from mujoco import rollout  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
from spider.simulators.mjwp import setup_mj_model  # noqa: E402

# ---------------------------------------------------------------------------------------------------------------------
# Contact from observed frames
# ---------------------------------------------------------------------------------------------------------------------


def fingertip_gaps(generated):
    """(T, 5) distance of each fingertip to the nearest object-mesh vertex, in the object's frame."""
    obj = generated["object_poses"]
    rot = Rotation.from_quat(obj[:, [4, 5, 6, 3]]).as_matrix()
    tips = generated["hand_joints"][:, [4, 8, 12, 16, 20]]
    local = np.einsum("tji,tfj->tfi", rot, tips - obj[:, :3, None].transpose(0, 2, 1))
    return cKDTree(generated["object_vertices"]).query(local.reshape(-1, 3))[0].reshape(len(obj), 5)


def observed_contact(g, observed_end):
    gaps = fingertip_gaps(g)
    n = min(10, int(np.sum(g["times"] <= observed_end + 1e-8)))
    assert n >= 4, "Native initial IK smoothing must fit inside the observed prefix"
    near = (gaps < 0.03).sum(axis=1) >= 2
    held = bool(near[:n].mean() >= 0.8)
    weights = np.ones(len(near))
    lost = 0
    release = None
    if held:
        for i, contact in enumerate(near):
            lost = 0 if contact else lost + 1
            if lost >= 10:
                release = i - 9
                break
        if release is not None:
            weights[release:] = np.clip(1 - np.arange(len(near) - release) / 15, 0, 1)
    else:
        weights[:] = 0
    return held, weights, gaps, release, n


# ---------------------------------------------------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------------------------------------------------


def object_id(model, mode):
    """Body id and qpos address of the real (free, unactuated) object. Bimanual scenes also hold an inactive dummy."""
    side = "left" if mode == "left" else "right"
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_object")
    assert body >= 0
    joint = int(model.body_jntadr[body])
    adr = int(model.jnt_qposadr[joint])
    assert model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE and joint not in model.actuator_trnid[:, 0]
    assert model.body_mass[body] > 0 and model.body_gravcomp[body] == 0 and model.neq == 0
    return body, adr


def gravity_controls(model, qpos, mode, sides, weights):
    """Position targets with gravity feed-forward through the hand motors (and, weighted, the held object's load)."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    body, _ = object_id(model, mode)
    torque = data.qfrc_bias.copy()
    den = max(sum(weights), 1e-8)
    for side, w in zip(sides, weights):
        if w <= 0:
            continue
        palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_hand_link")
        jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, jacr, data.xipos[body], palm)
        torque -= jacp.T @ (model.body_mass[body] * model.opt.gravity) * (w / den)
    joints = model.actuator_trnid[:, 0]
    np.testing.assert_array_equal(model.jnt_qposadr[joints], np.arange(model.nu))
    ctrl = qpos[: model.nu] + torque[model.jnt_dofadr[joints]] / model.actuator_gainprm[:, 0]
    return np.clip(ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])


def hold(model, qpos, ctrl, mode):
    """Hold check: 3 s at 2 ms with fixed controls; the object must stay within 10 mm and 0.30 rad."""
    _, adr = object_id(model, mode)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.ctrl[:] = ctrl
    poses = []
    model.opt.timestep = 0.002
    for _ in range(1500):
        mujoco.mj_step(model, data)
        poses.append(data.qpos[adr : adr + 7].copy())
    pos, rot = object_errors(np.array(poses), np.repeat(qpos[None, adr : adr + 7], 1500, axis=0))
    return dict(
        max_position_m=float(pos.max()),
        max_rotation_rad=float(rot.max()),
        passed=bool(data.time >= 2.999 and np.isfinite(poses).all() and pos.max() <= 0.01 and rot.max() <= 0.3),
    )


def root_rotation(qpos):
    """Rotation of the XHand floating base from its three root angles."""
    return Rotation.from_euler("ZXY", qpos[..., 3:6] * [1, 1, -1])


def continuous_angles(rotations, initial):
    """Root angles for `rotations`, choosing at each frame the representation closest to the previous one."""
    values = []
    previous = np.asarray(initial) * [1, 1, -1]
    for a in rotations.as_euler("ZXY"):
        candidates = np.array([a, [a[0] + np.pi, np.pi - a[1], a[2] + np.pi]])
        candidates += 2 * np.pi * np.round((previous - candidates) / (2 * np.pi))
        best = candidates[np.linalg.norm(candidates - previous, axis=1).argmin()]
        values.append(best * [1, 1, -1])
        previous = best
    return np.array(values)


# ---------------------------------------------------------------------------------------------------------------------
# Grasp preparation (observed initial state only)
# ---------------------------------------------------------------------------------------------------------------------


def calibrate(out, e, model, q0, weights, threads):
    """CEM over hand pose offsets and finger control offsets that make the free object hold still for 3 s."""
    path = Path(out) / "grasp_init" / e["episode_id"]
    if (path / "grasp.npz").exists():
        return dict(np.load(path / "grasp.npz"))
    nu = model.nu
    _, adr = object_id(model, e["mode"])
    nh = len(e["sides"])
    h = nu // nh
    finger = np.concatenate([np.arange(k * h + 6, (k + 1) * h) for k in range(nh)])
    joints = model.actuator_trnid[:, 0]
    model.opt.timestep = 0.002
    ff = gravity_controls(model, q0, e["mode"], e["sides"], weights) - q0[:nu]
    # Per hand: root position (3, m), root angles (3, rad), finger joints; then finger control offsets.
    scale = np.r_[np.tile([0.012] * 3 + [0.15] * 3 + [0.22] * (h - 6), nh), [0.15] * len(finger)]
    bound = np.r_[np.tile([0.03] * 3 + [0.45] * 3 + [0.55] * (h - 6), nh), [0.5] * len(finger)]
    mean = np.zeros(len(scale))
    std = scale.copy()
    rng = np.random.default_rng(0)
    best = None
    history = []
    started = time.monotonic()
    data = [mujoco.MjData(model) for _ in range(threads)]
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    with rollout.Rollout(nthread=threads) as pool:
        for iteration in range(30):
            z = np.clip(mean + rng.normal(size=(256, len(scale))) * std, -bound, bound)
            z[0] = 0 if best is None else best["parameters"]
            qp = np.repeat(q0[None], len(z), axis=0)
            qp[:, :nu] = np.clip(qp[:, :nu] + z[:, :nu], model.jnt_range[joints, 0], model.jnt_range[joints, 1])
            ctrl = qp[:, :nu] + ff
            ctrl[:, finger] += z[:, nu:]
            ctrl = np.clip(ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            initial = np.zeros((len(z), nstate))
            initial[:, 1 : 1 + model.nq] = qp
            states, _ = pool.rollout(model, data, initial, ctrl[:, None], nstep=1500)
            obj = states[:, :, 1 + adr : 1 + adr + 7]
            pos = np.linalg.norm(obj[:, :, :3] - q0[adr : adr + 3], axis=-1)
            quat = obj[:, :, 3:]
            norm = np.linalg.norm(quat, axis=-1) * np.linalg.norm(q0[adr + 3 : adr + 7])
            cosine = np.abs(np.einsum("bti,i->bt", quat, q0[adr + 3 : adr + 7])) / np.maximum(norm, 1e-12)
            rot = 2 * np.arccos(np.clip(cosine, 0, 1))
            pm = pos.max(1)
            rm = rot.max(1)
            objective = np.maximum(pm / 0.01, rm / 0.3) + 0.2 * (pm / 0.01 + rm / 0.3) + 0.002 * np.mean((z / scale) ** 2, axis=1)
            objective[~np.isfinite(objective)] = 1e6
            objective[states[:, -1, 0] < 2.999] = 1e6
            i = int(objective.argmin())
            if best is None or objective[i] < best["objective"]:
                best = dict(objective=float(objective[i]), parameters=z[i].copy(), qpos=qp[i].copy(), ctrl=ctrl[i].copy())
            elite = z[np.argsort(objective)[:32]]
            mean = 0.3 * mean + 0.7 * elite.mean(0)
            std = np.maximum(0.3 * std + 0.7 * elite.std(0), scale * 0.04)
            history.append(dict(iteration=iteration, best=best["objective"], position_mm=float(pm[i] * 1000), rotation_rad=float(rm[i])))
            print("GRASP", e["episode_id"], iteration, round(best["objective"], 4), flush=True)
    validation = hold(model, best["qpos"], best["ctrl"], e["mode"])
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "grasp.npz", **{k: v for k, v in best.items() if k != "objective"})
    write(
        path / "result.json",
        dict(episode=e["episode_id"], budget=[30, 256], weights=list(weights), validation=validation, history=history, wall_s=time.monotonic() - started),
    )
    return best


def adapt(out, e, contacts, grasp, held):
    """Write each arm's SPIDER input: grasp-preserving hand reference + controls; object targets unchanged."""
    for arm in ARMS:
        dst = target(out, arm, e)
        raw = dict(np.load(dst / "trajectory_kinematic_original.npz"))
        q = raw["qpos"].copy()
        model = setup_mj_model(SimpleNamespace(model_path=str(dst.parent / "scene.xml"), sim_dt=0.002, embodiment_type=e["mode"]))
        nu = model.nu
        h = nu // len(e["sides"])
        _, adr = object_id(model, e["mode"])
        objects = q[:, nu:].copy()
        obj = q[:, adr : adr + 7]
        if held:
            init = grasp["qpos"]
            np.testing.assert_allclose(obj[0], init[adr : adr + 7], atol=2e-6, rtol=0)
            delta = Rotation.from_quat(obj[:, [4, 5, 6, 3]]) * Rotation.from_quat(obj[0, [4, 5, 6, 3]]).inv()
            weights = []
            for j, side in enumerate(e["sides"]):
                start = j * h
                # SPIDER's IK drops two leading frames and one trailing frame of the 50 Hz keypoints.
                w = contacts[arm][side]["w"][2:-1]
                weights.append(w)
                orig = q[:, start : start + h].copy()
                anchor_pos = obj[:, :3] + delta.apply(init[start : start + 3] - obj[0, :3])
                original_rot = root_rotation(orig)
                anchor_rot = delta * root_rotation(init[start : start + h])
                mixed = original_rot * Rotation.from_rotvec(w[:, None] * (original_rot.inv() * anchor_rot).as_rotvec())
                q[:, start : start + 3] = w[:, None] * anchor_pos + (1 - w[:, None]) * orig[:, :3]
                q[:, start + 3 : start + 6] = continuous_angles(mixed, init[start + 3 : start + 6])
                q[:, start + 6 : start + h] = w[:, None] * init[start + 6 : start + h] + (1 - w[:, None]) * orig[:, 6:h]
            q[0, :nu] = init[:nu]
            # The palm must move rigidly with the object wherever the weight is 1 (checked with the robot's own FK).
            md = mujoco.MjData(model)
            for side, w in zip(e["sides"], weights):
                palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_hand_link")
                md.qpos[:] = init
                mujoco.mj_forward(model, md)
                r0 = md.xmat[palm].reshape(3, 3).copy()
                for k in np.flatnonzero(w == 1):
                    md.qpos[:] = q[k]
                    mujoco.mj_forward(model, md)
                    np.testing.assert_allclose(md.xmat[palm].reshape(3, 3), delta[k].as_matrix() @ r0, atol=1e-7)
            initial_weights = [float(contacts["reference"][s]["held"]) for s in e["sides"]]
            residual = grasp["ctrl"] - gravity_controls(model, init, e["mode"], e["sides"], initial_weights)
            ctrl = []
            for k, qp in enumerate(q):
                w = [float(x[k]) for x in weights]
                # One hand carries the full payload above weight 0.5; two hands share it continuously.
                physical_weights = [float(w[0] > 0.5)] if len(w) == 1 else w
                u = gravity_controls(model, qp, e["mode"], e["sides"], physical_weights)
                for j, value in enumerate(w):
                    u[j * h : (j + 1) * h] += value * residual[j * h : (j + 1) * h]
                ctrl.append(u)
            raw["qpos"] = q
            raw["ctrl"] = np.clip(ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            raw["ctrl"][0] = grasp["ctrl"]
            for i in range(1, len(q)):
                mujoco.mj_differentiatePos(model, raw["qvel"][i], 0.02, q[i - 1], q[i])
            raw["qvel"][0] = raw["qvel"][1]
        np.testing.assert_array_equal(q[:, nu:], objects)
        np.savez_compressed(dst / "trajectory_kinematic.npz", **raw)
        write(
            dst / "preprocess.json",
            dict(
                status="complete",
                mode=e["mode"],
                kinematic_sha256=sha(dst / "trajectory_kinematic.npz"),
                original_kinematic_sha256=sha(dst / "trajectory_kinematic_original.npz"),
                scene_sha256=sha(dst.parent / "scene.xml"),
                object_qpos_address=adr,
                initially_held=held,
                contacts={s: dict(held=c["held"], release_frame=c["release"], observed_initial_frames=c["n_initial"]) for s, c in contacts[arm].items()},
            ),
        )


# ---------------------------------------------------------------------------------------------------------------------
# One episode
# ---------------------------------------------------------------------------------------------------------------------


def run(out, ep, threads=24):
    out = Path(out)
    status = out / "preprocess" / ep / "status.json"
    if status.exists() and read(status)["status"] in ["completed", "failed"]:
        return
    status.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(status.parent)
    e = episode(out, ep)
    receipt = dict(episode=ep, status="running", mode=e["mode"])
    write(status, receipt)
    started = time.monotonic()
    try:
        from spider.preprocess import decompose, generate_xml, ik_fast

        reference = target(out, "reference", e)
        for arm in ARMS:
            dst = target(out, arm, e)
            dst.mkdir(parents=True, exist_ok=True)
            kwargs = dict(dataset_dir=str(dataset_dir(out, arm)), dataset_name=DATASET, robot_type=ROBOT, embodiment_type=e["mode"], task=ep, data_id=0)
            if not (dst / "trajectory_kinematic_original.npz").exists():
                if arm == "reference":
                    decompose.main(**kwargs)
                    generate_xml.main(**kwargs, show_viewer=False)
                else:
                    # Same convex decomposition and scene for both arms.
                    rel = Path("processed") / DATASET / "assets/objects" / ep
                    shutil.copytree(dataset_dir(out, "reference") / rel, dataset_dir(out, arm) / rel, dirs_exist_ok=True)
                    for name in ["scene.xml", "scene_eq.xml", "task_info.json"]:
                        shutil.copy2(reference.parent / name, dst.parent / name)
                np.random.seed(0)
                n_frames = read(export_dir(out, arm, ep) / "export.json")["n_export_frames"]
                # An explicit end index: the default slice would drop the last frame.
                ik_fast.main(**kwargs, show_viewer=False, save_video=False, end_idx=n_frames, ref_dt=0.02)
                shutil.copy2(dst / "trajectory_kinematic.npz", dst / "trajectory_kinematic_original.npz")
                shutil.copy2(dst.parent / "scene.xml", dst.parent / "scene_original.xml")
        contacts = {}
        for arm in ARMS:
            meta = read(export_dir(out, arm, ep) / "export.json")
            g = dict(np.load(export_dir(out, arm, ep) / "generated.npz"))
            contacts[arm] = {}
            for side in e["sides"]:
                joints = g[f"hand_{side}_joints"]
                held, w, gaps, release, n_initial = observed_contact(dict(g, hand_joints=joints), meta["observed_end_s"])
                contacts[arm][side] = dict(held=held, w=w, release=release, n_initial=n_initial)
                if arm == "forecast":
                    assert held == contacts["reference"][side]["held"], "Observed contact differs between paired arms"
        held = any(c["held"] for c in contacts["reference"].values())
        grasp = None
        for arm in ARMS:
            dst = target(out, arm, e)
            if held:
                spec = mujoco.MjSpec.from_file(str(dst.parent / "scene_original.xml"))
                spec.geom("floor").pos[2] -= 0.3
                (dst.parent / "scene.xml").write_text(spec.to_xml())
            else:
                shutil.copy2(dst.parent / "scene_original.xml", dst.parent / "scene.xml")
        if held:
            model = setup_mj_model(SimpleNamespace(model_path=str(reference.parent / "scene.xml"), sim_dt=0.002, embodiment_type=e["mode"]))
            q0 = np.load(reference / "trajectory_kinematic_original.npz")["qpos"][0]
            grasp = calibrate(out, e, model, q0, [float(contacts["reference"][s]["held"]) for s in e["sides"]], threads)
        adapt(out, e, contacts, grasp, held)
        receipt.update(status="completed", initially_held=held)
    except Exception as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        receipt["wall_s"] = time.monotonic() - started
        write(status, receipt)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("episode")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--threads", type=int, default=24, help="CPU threads of the grasp search")
    a = p.parse_args()
    run(a.out.resolve(), a.episode, a.threads)
