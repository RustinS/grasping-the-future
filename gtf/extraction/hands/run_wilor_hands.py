#!/usr/bin/env python3
"""WiLoR hand stage (the `hands` step of the Action100M pipeline, gtf.extraction.action100m.pipeline).

WiLoR is a single-shot hand detector + 3D MANO reconstruction that needs no person
detector, so it also works on close-up clips that show only hands.

The stage writes the WiLoR output in hands/ (`hands/{side}.{pt,npz}`
+ `hands/det.pkl` + `.done`), the format every later stage reads. Steps:
  1. WiLoR inference in its own virtualenv (subprocess) -> `wilor/wilor.npz`
     (detections, camera-space MANO verts, native MANO parameters). It runs with a
     FOV-matched focal f = 1.25 W (horizontal FOV about 43.6 deg); WiLoR's
     near-orthographic default focal (about 5000/256 * max(H, W)) breaks the optimizer's
     MoGe depth calibration, which projects hand joints through the stored intrinsics.
     EXTRACT_KNOWN_K replaces the focal with a calibrated camera's intrinsics.
  2. Confidence + temporal-persistence false-positive filter: drops sparse or flickering
     detections. A sustained moderate-confidence false positive on a hand-held object
     (e.g. a spoon tracked as a hand) passes; the downstream interaction / contact
     gates are there to catch it.
  3. MANO parameters (global_orient / hand_pose / betas / cam_t): WiLoR's native
     parameters converted to axis-angle (ingest_native) and run through optimize_mano's
     MANOForward (right model, flat_hand_mean=True, left hand = x-mirror of the right model).

Usage:
    python -m gtf.extraction.hands.run_wilor_hands --segment my_segment_name
    python -m gtf.extraction.hands.run_wilor_hands --segment my_seg --force --device cuda
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import pickle
import subprocess
import tempfile
import traceback
import zipfile
from pathlib import Path

import numpy as np

# torch is imported lazily inside ingest_native_matrices / write_hand_outputs so the false-positive
# filter (filter_detections / _longest_run / fov_matched_K) runs without torch or a GPU.

from gtf.paths import REPO_ROOT
from gtf.utils import RESULTS_DIR as _RESULTS_DIR

RESULTS_DIR = Path(_RESULTS_DIR)

WILOR_VENV_PY = REPO_ROOT / "third-party" / "wilor" / ".venv" / "bin" / "python"
# The two WiLoR runners run in WiLoR's own virtualenv, by file path; they import nothing of gtf.
WILOR_RUNNER = Path(__file__).resolve().with_name("run_wilor.py")
WILOR_PERSISTENT_RUNNER = Path(__file__).resolve().with_name("run_wilor_persistent.py")

# False-positive filter defaults (one setting for every video: real-hand tracks have
# sustained high-confidence runs and pass easily; the thresholds only remove sparse or
# low-confidence detections)
MIN_CONF = 0.3
MIN_FRAMES = 10  # a side needs at least this many surviving frames
MIN_RUN = 8  # ...and a longest near-contiguous run of at least this many
MAX_GAP = 3  # frames; gaps up to this don't break a run
HAND_FOV_FOCAL_RATIO = 1.25  # f = 1.25 W: f = 800 at W = 640, horizontal FOV about 43.6 deg


def fov_matched_K(W, H):
    f = HAND_FOV_FOCAL_RATIO * W
    return f"{f:.2f},{f:.2f},{W / 2:.2f},{H / 2:.2f}"


def resolve_K(W, H):
    """Camera intrinsics for the WiLoR runner.

    Default: the FOV-matched focal (no calibration needed). EXTRACT_KNOWN_K="fx,fy,cx,cy"
    supplies the true intrinsics of a calibrated camera instead.
    """
    known = os.environ.get("EXTRACT_KNOWN_K", "").strip()
    if not known:
        return fov_matched_K(W, H)
    parts = [float(x) for x in known.split(",")]
    if len(parts) != 4:
        raise ValueError(f"EXTRACT_KNOWN_K must be 'fx,fy,cx,cy', got {known!r}")
    return ",".join(f"{v:.6f}" for v in parts)


def _atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path, payload):
    _atomic_text(path, json.dumps(payload, indent=1, allow_nan=False) + "\n")


def _wilor_output_complete(path):
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "n_frames",
                "K",
                "left_frames",
                "left_verts",
                "right_frames",
                "right_verts",
            }
            return required.issubset(archive.files) and int(archive["n_frames"]) > 0
    except (OSError, ValueError, EOFError, zipfile.BadZipFile):
        return False


def hand_outputs_complete(segment_dir):
    """Check that the packed WiLoR output in hands/ is complete before a resume skips it.

    Markers: "wilor" (with wilor_report.json and the packed files of every written side) or
    "wilor:no_hands" (report with no written side).
    """

    hand_dir = Path(segment_dir) / "hands"
    try:
        marker = (hand_dir / ".done").read_text()
    except OSError:
        return False
    try:
        report = json.loads((hand_dir / "wilor_report.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(report, dict) or report.get("tracker") != "wilor":
        return False
    written = report.get("written")
    if marker == "wilor:no_hands\n":
        return written == []
    if marker != "wilor\n" or not isinstance(written, list) or not written:
        return False
    det = hand_dir / "det.pkl"
    if not det.is_file() or det.stat().st_size <= 0:
        return False
    for row in written:
        side = row.get("side") if isinstance(row, dict) else None
        if side not in {"left", "right"}:
            return False
        for suffix in ("pt", "npz"):
            payload = hand_dir / f"{side}.{suffix}"
            if not payload.is_file() or payload.stat().st_size <= 0:
                return False
    return True


def run_wilor_inference(segment_dir, force=False):
    """Subprocess WiLoR inference in its isolated venv -> wilor/wilor.npz."""
    import imageio.v2 as imageio

    seg = Path(segment_dir)
    video = seg / "video.mp4"
    if not video.exists():
        print(f"  [wilor] no video.mp4 in {seg}")
        return None
    out_npz = seg / "wilor" / "wilor.npz"
    if not force and _wilor_output_complete(out_npz):
        return out_npz
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    rdr = imageio.get_reader(str(video))
    H, W = rdr.get_data(0).shape[:2]
    rdr.close()
    K = resolve_K(W, H)

    if not WILOR_VENV_PY.exists():
        raise SystemExit(f"WiLoR venv missing: {WILOR_VENV_PY} (run third-party/wilor setup)")
    # the isolated venv ships its own CUDA libs; drop the main-venv LD_PRELOAD
    env = dict(os.environ)
    env.pop("LD_PRELOAD", None)
    cmd = [str(WILOR_VENV_PY), str(WILOR_RUNNER), "--video", str(video), "--K", K, "--out", str(out_npz)]
    print(f"  [wilor] inference (K f={float(K.split(',')[0]):.0f}, {W}x{H})")
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0 or not out_npz.exists():
        print(f"  [wilor] inference FAILED:\n{chr(10).join(r.stderr.splitlines()[-12:])}")
        return None
    return out_npz


def _wilor_inference_job(name, force=False):
    """Build one persistent-runner job, or None when inference is unnecessary."""
    import imageio.v2 as imageio

    seg = RESULTS_DIR / name
    video = seg / "video.mp4"
    output = seg / "wilor" / "wilor.npz"
    if not video.exists() or (not force and _wilor_output_complete(output)):
        return None
    reader = imageio.get_reader(str(video))
    H, W = reader.get_data(0).shape[:2]
    metadata = reader.get_meta_data()
    reader.close()
    # The frame-count hint feeds a deterministic greedy partition that balances the
    # resident worker processes. FFmpeg reads fps and duration from the container
    # header, so this does not decode the video.
    try:
        frame_count_hint = round(float(metadata["fps"]) * float(metadata["duration"]))
    except (KeyError, TypeError, ValueError, OverflowError):
        frame_count_hint = 1
    if not math.isfinite(frame_count_hint) or frame_count_hint < 1:
        frame_count_hint = 1
    output.parent.mkdir(parents=True, exist_ok=True)
    return {
        "name": name,
        "video": str(video),
        "K": resolve_K(W, H),
        "out": str(output),
        "batch_size": 32,
        "frame_count_hint": int(frame_count_hint),
    }


def _partition_wilor_jobs(jobs, worker_count):
    """Deterministically balance independent videos while preserving local order."""

    worker_count = min(max(1, int(worker_count)), len(jobs))
    partitions = [[] for _ in range(worker_count)]
    loads = [0] * worker_count
    indexed = list(enumerate(jobs))
    for original_index, job in sorted(
        indexed,
        key=lambda item: (-int(item[1].get("frame_count_hint", 1)), item[0]),
    ):
        worker = min(range(worker_count), key=lambda index: (loads[index], index))
        partitions[worker].append((original_index, job))
        loads[worker] += max(1, int(job.get("frame_count_hint", 1)))
    return [[job for _index, job in sorted(partition)] for partition in partitions]


def _read_wilor_worker_status(status_path, jobs):
    """Accept only unique, expected acknowledgements with complete output bytes."""

    expected = {job["name"]: job for job in jobs}
    try:
        rows = json.loads(Path(status_path).read_text())
        if not isinstance(rows, list):
            raise TypeError("status payload is not a list")
        names = [row.get("name") for row in rows if isinstance(row, dict)]
        if len(names) != len(rows) or len(names) != len(set(names)):
            raise ValueError("status contains malformed or duplicate names")
        if not set(names).issubset(expected):
            raise ValueError("status acknowledges an unassigned job")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        return set(), str(error)

    succeeded = {row["name"] for row in rows if row.get("ok") is True and _wilor_output_complete(expected[row["name"]]["out"])}
    return succeeded, None


def _run_wilor_worker(jobs, temporary, label, env):
    jobs_path = Path(temporary) / f"jobs_{label}.json"
    status_path = Path(temporary) / f"status_{label}.json"
    jobs_path.write_text(json.dumps(jobs))
    command = [
        str(WILOR_VENV_PY),
        str(WILOR_PERSISTENT_RUNNER),
        "--jobs",
        str(jobs_path),
        "--status",
        str(status_path),
    ]
    try:
        result = subprocess.run(command, env=env)
    except BaseException as error:
        if isinstance(error, KeyboardInterrupt):
            raise
        return set(), None, f"{type(error).__name__}: {error}"
    succeeded, status_error = _read_wilor_worker_status(status_path, jobs)
    return succeeded, result.returncode, status_error


def _wilor_worker_count():
    raw = os.environ.get("EXTRACTION_WILOR_WORKERS", "1")
    try:
        count = int(raw)
    except ValueError as error:
        raise ValueError("EXTRACTION_WILOR_WORKERS must be a positive integer") from error
    if count < 1:
        raise ValueError("EXTRACTION_WILOR_WORKERS must be a positive integer")
    return count


def run_wilor_inference_batch(names, force=False):
    """Run pending WiLoR jobs in persistent workers (EXTRACTION_WILOR_WORKERS, default 1).

    A job counts as done only when its worker acknowledges it and its output is complete.
    With several workers, unfinished jobs are retried once in one fresh worker. Returns the
    names of the finished jobs; the caller runs the rest one segment at a time.
    """
    jobs = [job for name in names if (job := _wilor_inference_job(name, force))]
    if not jobs:
        return set()
    if not WILOR_VENV_PY.exists():
        raise SystemExit(f"WiLoR venv missing: {WILOR_VENV_PY} (run third-party/wilor setup)")

    env = dict(os.environ)
    env.pop("LD_PRELOAD", None)
    with tempfile.TemporaryDirectory(prefix="gtf_wilor_") as temporary:
        worker_count = min(_wilor_worker_count(), len(jobs))
        partitions = _partition_wilor_jobs(jobs, worker_count)
        loads = [sum(job.get("frame_count_hint", 1) for job in part) for part in partitions]
        print(f"  [wilor] persistent inference for {len(jobs)} segments with {worker_count} worker(s), estimated frame loads={loads}")
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_run_wilor_worker, partition, temporary, f"primary_{index}", env) for index, partition in enumerate(partitions)]
            primary_results = [future.result() for future in futures]

        succeeded = set().union(*(result[0] for result in primary_results))
        for index, (_accepted, returncode, status_error) in enumerate(primary_results):
            if returncode != 0 or status_error is not None:
                print(f"  [wilor] worker {index} incomplete: returncode={returncode}, status={status_error or 'partial failure rows'}")

        unresolved = [job for job in jobs if job["name"] not in succeeded]
        if unresolved and worker_count > 1:
            print(f"  [wilor] recovering {len(unresolved)} unresolved segment(s) with one fresh persistent worker")
            recovered, returncode, status_error = _run_wilor_worker(unresolved, temporary, "recovery", env)
            succeeded.update(recovered)
            if returncode != 0 or status_error is not None:
                print(f"  [wilor] recovery incomplete: returncode={returncode}, status={status_error or 'partial failure rows'}")

        unresolved_count = len(jobs) - len(succeeded)
        if unresolved_count:
            print(f"  [wilor] {len(succeeded)}/{len(jobs)} outputs acknowledged; {unresolved_count} unresolved job(s) use per-segment fallback")
        return succeeded


def _longest_run(frames, max_gap):
    """longest near-contiguous run length over sorted unique frame indices."""
    if len(frames) == 0:
        return 0
    f = np.sort(np.unique(frames))
    best = run = 1
    for i in range(1, len(f)):
        if f[i] - f[i - 1] <= max_gap + 1:
            run += 1
        else:
            run = 1
        best = max(best, run)
    return best


def filter_detections(npz, min_conf=MIN_CONF, min_frames=MIN_FRAMES, min_run=MIN_RUN, max_gap=MAX_GAP):
    """Confidence + temporal-persistence FP filter. Returns ({side: (frames, verts)}, K, report)."""
    d = np.load(npz, allow_pickle=True)
    K = d["K"]
    n_frames = int(d["n_frames"]) if "n_frames" in d.files else 0
    out, report = {}, {"n_frames": n_frames}
    for side in ("left", "right"):
        frames, verts, conf = d[f"{side}_frames"], d[f"{side}_verts"], d[f"{side}_conf"]
        keep = conf >= min_conf
        frames, verts, conf = frames[keep], verts[keep], conf[keep]
        run = _longest_run(frames, max_gap)
        ok = len(frames) >= min_frames and run >= min_run
        report[side] = {"n": int(len(frames)), "longest_run": int(run), "med_conf": float(np.median(conf)) if len(conf) else 0.0, "kept": bool(ok)}
        if ok:
            out[side] = (frames.astype(np.int64), verts.astype(np.float64))
    return out, K, report


def ingest_native_matrices(
    side,
    global_orient,
    hand_pose,
    betas,
    verts,
    device="cuda",
):
    """Convert one side's WiLoR MANO rotation matrices into our hand parameters.

    global_orient (N,1,3,3), hand_pose (N,15,3,3) rotation matrices, betas (N,10) and
    the WiLoR verts (N,778,3). Returns ({hand_pose, global_orient, betas, cam_t, cHands,
    cJoints}, residual in mm). Raises on a shape mismatch, NaN/Inf or non-proper rotations.
    """
    import torch
    from scipy.spatial.transform import Rotation

    from gtf.extraction.optimizer.optimize_mano import MANOForward

    if side not in ("left", "right"):
        raise ValueError(f"unknown hand side: {side!r}")
    go_matrix = np.asarray(global_orient)
    hp_matrix = np.asarray(hand_pose)
    be = np.asarray(betas)
    target_verts = np.asarray(verts)
    n = len(target_verts)
    expected = {
        "global_orient": (n, 1, 3, 3),
        "hand_pose": (n, 15, 3, 3),
        "betas": (n, 10),
        "verts": (n, 778, 3),
    }
    actual = {
        "global_orient": go_matrix.shape,
        "hand_pose": hp_matrix.shape,
        "betas": be.shape,
        "verts": target_verts.shape,
    }
    bad = {key: (actual[key], shape) for key, shape in expected.items() if actual[key] != shape}
    if bad or n == 0:
        raise ValueError(f"native WiLoR matrix batch has a shape mismatch: {bad}")
    if not all(np.isfinite(value).all() for value in (go_matrix, hp_matrix, be, target_verts)):
        raise ValueError("native WiLoR matrix batch contains NaN/Inf")
    rotations = np.concatenate([go_matrix, hp_matrix], axis=1).astype(np.float64)
    eye = np.eye(3, dtype=np.float64)
    if np.max(np.abs(rotations @ np.swapaxes(rotations, -1, -2) - eye)) > 5e-3 or np.max(np.abs(np.linalg.det(rotations) - 1.0)) > 5e-3:
        raise ValueError("native WiLoR matrix batch contains non-proper rotations")

    go = Rotation.from_matrix(go_matrix[:, 0]).as_rotvec().astype(np.float32)
    hp = Rotation.from_matrix(hp_matrix.reshape(-1, 3, 3)).as_rotvec().reshape(n, 45).astype(np.float32)
    be = be.astype(np.float32)
    layer = MANOForward(side=side, device=device)
    go_t = torch.tensor(go, device=device)
    hp_t = torch.tensor(hp, device=device)
    be_t = torch.tensor(be, device=device)
    with torch.no_grad():
        v_nc, _ = layer.mano(torch.cat([go_t, hp_t], -1), th_betas=be_t)  # mm, right frame, no cam_t
    tgt = np.ascontiguousarray(target_verts, dtype=np.float32)
    if side == "left":
        tgt = tgt.copy()
        tgt[..., 0] = -tgt[..., 0]  # un-mirror to right frame
    cam_t = (tgt.mean(1) - (v_nc.cpu().numpy() / 1000.0).mean(1)).astype(np.float32)
    with torch.no_grad():
        vfk, jfk = layer.forward(go_t, hp_t, be_t, torch.tensor(cam_t, device=device))
    resid = float(((vfk.cpu().numpy() - np.asarray(verts, np.float32)) ** 2).sum(-1).mean() ** 0.5 * 1000)
    return {
        "hand_pose": hp,
        "global_orient": go,
        "betas": be,
        "cam_t": cam_t,
        "cHands": vfk.cpu().numpy().astype(np.float32),
        "cJoints": jfk.cpu().numpy().astype(np.float32),
    }, resid


def ingest_native(npz_path, side, frames, verts, device="cuda"):
    """Ingest WiLoR's native MANO params for the given frames (no fit).

    WiLoR uses our MANO convention (flat_hand_mean=True), so converting its rotation
    matrices to axis-angle and running our MANOForward reproduces its verts to ~0 mm with
    WiLoR's own smooth pose (a per-frame fit to the verts would be ill-conditioned and make
    the finger pose jitter). cam_t is recovered exactly from the verts (a per-frame
    constant offset)."""
    with np.load(npz_path, allow_pickle=True) as d:
        allf = d[f"{side}_frames"].astype(int)
        f2i = {int(f): i for i, f in enumerate(allf)}
        try:
            rows = [f2i[int(f)] for f in frames]
        except KeyError as error:
            raise ValueError(f"{side}: requested native frame has no row in wilor.npz: {error.args[0]}") from error
        global_orient = np.asarray(d[f"{side}_global_orient"][rows]).copy()
        hand_pose = np.asarray(d[f"{side}_hand_pose"][rows]).copy()
        betas = np.asarray(d[f"{side}_betas"][rows]).copy()
    return ingest_native_matrices(
        side,
        global_orient,
        hand_pose,
        betas,
        verts,
        device=device,
    )


def write_hand_outputs(seg_dir, side, frames, params, K, wh):
    """Write hands/{side}.pt + .npz, the packed hand format.

    The .pt holds per-frame intr (N,3,3), img_size (N,2), camera-space cHands (N,778,3)
    and cJoints (N,21,3) in meters, MANO hand_pose (N,45), global_orient (N,3), betas
    (N,10), cam_t (N,3) and the side flag `right`; the .npz holds frame_names (frame
    indices as strings).
    """
    import torch

    packed = Path(seg_dir) / "hands"
    packed.mkdir(parents=True, exist_ok=True)
    N = len(frames)
    pt = {
        "right": side == "right",
        "intr": torch.tensor(np.tile(np.asarray(K, dtype=np.float32), (N, 1, 1))),
        "img_size": torch.tensor(np.tile(np.asarray(wh, dtype=np.int64), (N, 1))),
        "cHands": torch.tensor(params["cHands"]),
        "cJoints": torch.tensor(params["cJoints"]),
        "hand_pose": torch.tensor(params["hand_pose"]),
        "global_orient": torch.tensor(params["global_orient"]),
        "betas": torch.tensor(params["betas"]),
        "cam_t": torch.tensor(params["cam_t"]),
    }
    torch.save(pt, packed / f"{side}.pt")
    np.savez(packed / f"{side}.npz", frame_names=np.array([f"{int(f)}" for f in frames]))


def process_segment(
    name,
    device="cuda",
    force=False,
    force_inference=None,
):
    """Full WiLoR hand stage for one segment. Returns True if hands/ was written.

    force_inference (default: force) re-runs WiLoR inference even when wilor/wilor.npz is complete."""
    seg = RESULTS_DIR / name
    done = seg / "hands" / ".done"
    if not force and hand_outputs_complete(seg):
        return True
    done.unlink(missing_ok=True)

    if force_inference is None:
        force_inference = force
    npz = run_wilor_inference(seg, force=force_inference)
    if npz is None:
        return False
    sides, K, report = filter_detections(npz)
    wh = (int(round(K[0, 2] * 2)), int(round(K[1, 2] * 2)))

    if not sides:
        # Persist the negative result: write the report (kept=false both sides) + the done
        # marker with NO packed payloads. Without it, (a) every resume re-runs WiLoR inference
        # on the same handless clip, and (b) the orchestrator's no-hands early exit
        # (reject_handless_segments) has no evidence to reject on, so the segment runs the
        # sam3d / object_pose / optimize stages before failing at optimize.
        (seg / "hands").mkdir(parents=True, exist_ok=True)
        _atomic_json(
            seg / "hands" / "wilor_report.json",
            {"tracker": "wilor", "report": report, "written": []},
        )
        _atomic_text(done, "wilor:no_hands\n")
        print(f"  [wilor] {name}: no side survived FP filter ({report})")
        return False

    det_entries, written = [], []
    for side, (frames, verts) in sides.items():
        params, resid = ingest_native(npz, side, frames, verts, device=device)
        write_hand_outputs(seg, side, frames, params, K, wh)
        det_entries.append({"seq": f"{name}_{side}", "valid": np.ones(len(frames), bool), "frame_idx": np.asarray(frames, int)})
        written.append((side, len(frames), resid))
    with open(seg / "hands" / "det.pkl", "wb") as f:
        pickle.dump(det_entries, f)
    # provenance and false-positive filter report
    _atomic_json(
        seg / "hands" / "wilor_report.json",
        {"tracker": "wilor", "report": report, "written": [{"side": s, "n": n, "fit_resid_mm": round(r, 2), "ingest": "native"} for s, n, r in written]},
    )
    _atomic_text(done, "wilor\n")
    print(f"  [wilor] {name}: " + ", ".join(f"{s}:{n}f native resid={r:.2f}mm" for s, n, r in written))
    return True


def run_wilor_hands_step(segment_names, args):
    """Orchestrator step: WiLoR hand tracking on validated segments."""
    from gtf.utils import is_action_valid

    valid_names = [n for n in segment_names if is_action_valid(n, args.object)]
    if not valid_names:
        print("  No validated segments for WiLoR hands")
        return
    todo = valid_names if args.force else [n for n in valid_names if not hand_outputs_complete(RESULTS_DIR / n)]
    if not todo:
        print(f"  [wilor] All {len(valid_names)} segments already done")
        return
    print(f"  Running WiLoR hands on {len(todo)} segments...")
    batch_succeeded = run_wilor_inference_batch(todo, force=args.force)
    done = failed = 0
    for vi, name in enumerate(todo):
        print(f"  [{vi + 1}/{len(todo)}] WiLoR hands: {name}")
        try:
            force_inference = args.force and name not in batch_succeeded
            if process_segment(
                name,
                device=args.device,
                force=args.force,
                force_inference=force_inference,
            ):
                done += 1
            else:
                failed += 1
        except Exception as e:
            traceback.print_exc()
            print(f"    ERROR: {e}")
            failed += 1
    print(f"  [wilor] {done} done, {failed} failed")


def main():
    ap = argparse.ArgumentParser(prog="python -m gtf.extraction.hands.run_wilor_hands")
    ap.add_argument("--segment", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ok = process_segment(args.segment, device=args.device, force=args.force)
    print("OK" if ok else "FAILED")


if __name__ == "__main__":
    main()
