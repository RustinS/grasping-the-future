"""Constants, output layout and shared numerics of the retargeting experiment.

Imported by every step. It needs only NumPy, so it loads in both the project environment (prepare,
summarize) and the SPIDER environment (preprocess, evaluate).
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SPIDER = Path(os.environ.get("SPIDER_ROOT", HERE / "spider")).resolve()
SPIDER_COMMIT = "44717007de41cbef7565dff7ff9f4453557a2d3d"
DEFAULT_OUT = HERE / "output"
DEFAULT_COHORT = HERE / "cohort.json"
DEFAULT_MANO = REPO / "third-party/mano"

DATASET = "forecast"  # SPIDER dataset name; it is only a directory name in SPIDER's layout
ROBOT = "xhand"
ARMS = ["reference", "forecast"]  # the recorded human future and the model's forecast of it
METHODS = ["native_ik", "adapted_ik", "spider"]
LABELS = {"native_ik": "Kinematic", "adapted_ik": "Kinematic + grasp prep", "spider": "Dynamic (SPIDER)"}
SEEDS = [0, 1, 2]
SOURCES = ["dexycb", "hot3d", "action100m", "ho3d", "h2o", "hocap", "oakink2", "taco", "show3d"]

REF_DT = 0.02  # SPIDER reference spacing (50 Hz)
SIM_DT = 0.01  # simulation and control step
POSITION_THRESHOLD_M = 0.1
ROTATION_THRESHOLD_RAD = 0.5
# SPIDER optimizer recipe on top of the upstream oakinkv2_fast preset.
RECIPE = dict(
    ctrl_dt=0.2,
    horizon=1.2,
    knot_dt=0.2,
    num_samples=1024,
    max_num_iterations=8,
    max_local_retries=1,
    max_revert_depth=1,
    num_dr=1,
    pair_margin_range=[0.0, 0.0],
    xy_offset_range=[0.0, 0.0],
)

# MANO canonical wrist frame -> XHand palm frame (right hand, as in SPIDER's oakinkv2 processor).
PALM_OFFSET = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
# Left hand: MANO left = x-mirrored right model; this maps the mirrored wrist frame to the left XHand palm frame.
LEFT_PALM_OFFSET = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0.0]])
MIRROR_X = np.diag([-1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------------------------------------------------
# Files and output layout
# ---------------------------------------------------------------------------------------------------------------------


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def arm_dir(out, arm):
    return Path(out) / "arms" / arm


def dataset_dir(out, arm):
    return arm_dir(out, arm) / "dataset"


def episode(out, ep):
    return next(e for e in read(Path(out) / "cohort.json")["episodes"] if e["episode_id"] == ep)


def target(out, arm, e, seed=0):
    """SPIDER's per-seed robot folder of one episode: .../processed/<dataset>/xhand/<mode>/<episode>/<seed>."""
    return dataset_dir(out, arm) / "processed" / DATASET / ROBOT / e["mode"] / e["episode_id"] / str(seed)


def export_dir(out, arm, ep):
    return arm_dir(out, arm) / "episodes" / ep


def qualification_path(out, ep):
    return Path(out) / "qualification" / f"{ep}.json"


def baseline_dir(out, method, arm, ep):
    return Path(out) / "baselines" / method / arm / ep


def reference_transferable(out, ep):
    """One completed successful reference replay already decides eligibility (all three seeds still run)."""
    q = qualification_path(out, ep)
    if q.exists():
        return bool(read(q)["retained"])
    e = episode(out, ep)
    for seed in SEEDS:
        p = target(out, "reference", e, seed) / "replay.json"
        if p.exists():
            r = read(p)
            if r.get("status") == "completed" and r.get("success") and r.get("complete"):
                return True
    return False


# ---------------------------------------------------------------------------------------------------------------------
# Resampling and rotations
# ---------------------------------------------------------------------------------------------------------------------


def linear(values, times, target):
    values = np.asarray(values)
    flat = values.reshape(len(times), -1)
    return np.stack([np.interp(target, times, col) for col in flat.T], axis=-1).reshape((len(target),) + values.shape[1:])


def continuous_quaternions(values):
    """Choose a continuous quaternion lift without changing any SO(3) pose.

    SPIDER averages and interpolates quaternion components, so antipodal representations must be unwrapped first.
    """
    values = np.asarray(values).copy()
    if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
        raise ValueError("expected finite (T, 4) quaternions")
    if np.any(np.linalg.norm(values, axis=1) < 1e-8):
        raise ValueError("zero quaternion")
    for i in range(1, len(values)):
        if np.dot(values[i - 1], values[i]) < 0:
            values[i] *= -1
    return values


def uniform_times(times, dt=REF_DT):
    """A uniform grid from the first to the last timestamp (no padded endpoint)."""
    times = np.asarray(times, dtype=np.float64)
    if len(times) < 2 or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError("timestamps must be finite and strictly increasing")
    return times[0] + np.arange(int(np.floor((times[-1] - times[0]) / dt + 1e-7)) + 1) * dt


# ---------------------------------------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------------------------------------


def object_errors(actual, reference):
    """Absolute positions and sign-invariant SO(3) geodesic errors of (T, 7) [xyz, wxyz] poses."""
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape or actual.ndim != 2 or actual.shape[1] != 7:
        raise ValueError("object poses must be matching (T, 7) wxyz arrays")
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("nonfinite object pose")
    qa, qr = actual[:, 3:], reference[:, 3:]
    na, nr = np.linalg.norm(qa, axis=-1), np.linalg.norm(qr, axis=-1)
    if np.any(na < 1e-8) or np.any(nr < 1e-8):
        raise ValueError("zero quaternion")
    cosine = np.abs(np.sum(qa * qr, axis=-1) / (na * nr))
    return np.linalg.norm(actual[:, :3] - reference[:, :3], axis=-1), 2 * np.arccos(np.clip(cosine, 0, 1))


def reference_grid(qpos, factor=2):
    """NumPy equivalent of SPIDER's interp(order=1, align_corners=True) from ref_dt to sim_dt."""
    coordinates = np.linspace(0, len(qpos) - 1, len(qpos) * factor)
    return np.stack([np.interp(coordinates, np.arange(len(qpos)), col) for col in qpos.T], axis=1)


def execution_reference(qpos):
    """The reference each simulated step is scored against, its times, and a mask without the padded endpoint.

    The state recorded after step i targets reference i+1 (run_mjwp_fast slices reference[s+1:s+horizon+1]). SPIDER's
    IK averages three frames and drops the first, so retained reference frames start at 2*ref_dt. The final repeated
    endpoint is optimizer padding and is not scored.
    """
    raw = reference_grid(qpos)
    times = 2 * REF_DT + np.linspace(0, (len(qpos) - 1) * REF_DT, len(raw))
    reference = np.concatenate([raw[1:], raw[-1:]])
    times = np.concatenate([times[1:], times[-1:] + SIM_DT])
    unpadded = np.ones(len(raw), dtype=bool)
    unpadded[-1] = False
    return reference, times, unpadded


def measure(qpos, reference, generated_mask, full_length):
    """Success = full-length replay AND mean future object error < 0.10 m and < 0.50 rad."""
    n = min(len(qpos), full_length)
    if n == 0 or len(reference) < full_length:
        raise ValueError("empty or short reference")
    complete = len(qpos) >= full_length
    pos, rot = object_errors(qpos[:n, -7:], reference[:n, -7:])
    mask = generated_mask[:n]
    if not mask.any():
        return dict(complete=complete, success=False, coverage=0.0, pos_mean_m=None, rot_mean_rad=None)
    return dict(
        complete=complete,
        success=bool(complete and pos[mask].mean() < POSITION_THRESHOLD_M and rot[mask].mean() < ROTATION_THRESHOLD_RAD),
        coverage=float(mask.sum() / generated_mask.sum()),
        n_generated_steps=int(mask.sum()),
        pos_mean_m=float(pos[mask].mean()),
        rot_mean_rad=float(rot[mask].mean()),
        pos_p95_m=float(np.percentile(pos[mask], 95)),
        pos_final_m=float(pos[mask][-1]) if complete else None,
        rot_final_rad=float(rot[mask][-1]) if complete else None,
    )


def contiguous_prefix_length(times):
    """Length of the saved optimizer output that is on the regular sim_dt timeline from the first step.

    When SPIDER exhausts its reverts it can leave a final chunk after a time gap; only the correctly timed prefix is
    replayed, and the trial then fails (it is never slid forward into the gap).
    """
    times = np.asarray(times).reshape(-1)
    good = np.isfinite(times) & np.isclose(times, SIM_DT * (np.arange(len(times)) + 1), atol=1e-6, rtol=0)
    bad = np.flatnonzero(~good)
    return int(bad[0]) if len(bad) else len(times)


def check_interpolation(source, expected, actual):
    """Check SPIDER's float32 reference interpolation against `reference_grid` within a roundoff bound.

    Sample coordinates are evaluated in float32: coordinate roundoff is O(T*eps), amplified by the largest adjacent
    source jump; value casting and interpolation add O(eps*|value|). This catches a shifted timeline without flagging
    representational rounding.
    """
    source = np.asarray(source, dtype=np.float64)
    bound = 8 * np.finfo(np.float32).eps * (len(source) * np.abs(np.diff(source, axis=0)).max(axis=0) + np.abs(source).max(axis=0) + 1.0)
    error = np.abs(np.asarray(expected, dtype=np.float64) - np.asarray(actual, dtype=np.float64))
    assert error.shape == np.asarray(expected).shape == np.asarray(actual).shape
    ratio = float((error / bound).max())
    assert ratio <= 1.0, dict(max_relative_to_roundoff_bound=ratio, max_error=float(error.max()), bound=bound.tolist())
    return dict(max_abs_difference=float(error.max()), max_relative_to_roundoff_bound=ratio)
