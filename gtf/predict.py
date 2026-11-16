"""Run a trained model on your own segments: sample hand and object trajectories for one task.

A segment is a folder in the format of docs/data.md: `canonical/canonical.npz` (two MANO hands and one object at
keyframes), `canonical/metadata.json` and the clip as `video.mp4` (the scene input). Segments from the extraction
pipeline already have all three. Before sampling, the command builds what the model reads, once, next to the
segments: a manifest (`DIR/manifest.json`), a scene cache per segment (`canonical/scene_cache_v2.h5`: DINOv3
features on a point cloud of the scene, from MoGe-2 depth when the source has no depth of its own) and the Uni3D
embedding of each object mesh (`DIR/uni3d_embeddings.pt`). Building them needs the `corpus` setup
(`bash third-party/setup.sh corpus`).

Each window is `eval.window_seconds` (5 s) long. The task decides what the model sees: `future_prediction` observes
the first `eval.future_obs_ratio` of the window (0.1: half a second) and predicts the rest; the other tasks are
`hand_to_object`, `object_to_hand`, `temporal_infilling`, `goal_infilling`, `goal_hand_only`, `goal_object_only`.

Output: one file per window, `OUT/<segment>/<task>_<first video frame>.npz`, with N sampled trajectories in the
fields of canonical.npz (`hand_{left,right}_{orient,pose,cam_t}` (N, T, ...), `obj_rotation`, `obj_translation`
(N, T, 3)), in the camera frame of the segment (for moving-camera segments with world fields: the camera of the
window's last observed frame), plus `frame_indices`, `frame_times_s` (from the window start), `valid` (T,), `observed` and `present`
(T, 3: left hand, right hand, object), and the shape inputs `hand_{left,right}_betas`, `obj_scale`,
`obj_canonical_verts`, `obj_faces`. Observed frames hold the input.

Usage:
  python -m gtf.predict --segments DIR --out predictions [--task future_prediction] [--segment NAME ...] [--start SECONDS ...]
      [--checkpoint checkpoints/gtf_main.ckpt] [eval.n_samples=20 ...]

The checkpoint (by default the released model) carries its normalization statistics and the PCA basis of the scene
features (gtf.training.export_checkpoint); for a training checkpoint pass them with --stats and --pca-basis.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from gtf.paths import MANO_DIR, REPO_ROOT

TASKS = ("future_prediction", "hand_to_object", "object_to_hand", "temporal_infilling", "goal_infilling", "goal_hand_only", "goal_object_only")
UNI3D_WEIGHTS = REPO_ROOT / "third-party/uni3d/weights/modelzoo/uni3d-g/model.pt"


# ---------------------------------------------------------------------------
# Inputs: manifest, scene caches, object embeddings


def _python_module(*args: str) -> None:
    subprocess.run([sys.executable, "-m", *args], check=True, cwd=REPO_ROOT)


def _manifest(segments_dir: Path) -> Path:
    path = segments_dir / "manifest.json"
    _python_module("gtf.datasets.build_manifest", "--results-root", str(segments_dir), "--include-gate-rejects", "--output", str(path))
    return path


def prepare_inputs(segments_dir: Path, pca_basis: str, names: list[str] | None) -> tuple[Path, list[dict]]:
    """Manifest entries of the segments (all of them, or `names`) with their scene caches built."""
    manifest = _manifest(segments_dir)
    entries = [e for e in json.loads(manifest.read_text())["entries"] if names is None or e["segment_id"] in names]
    if names is not None and len(entries) != len(set(names)):
        missing = sorted(set(names) - {e["segment_id"] for e in entries})
        raise SystemExit(f"not valid segments of {segments_dir}: {missing} (see the build_manifest output above)")
    if any(not e.get("has_scene_features") for e in entries):
        _python_module("gtf.datasets.scene.preprocess", "--manifest", str(manifest), "--pca-basis", pca_basis)
        manifest = _manifest(segments_dir)
        entries = [e for e in json.loads(manifest.read_text())["entries"] if names is None or e["segment_id"] in names]
    no_scene = [e["segment_id"] for e in entries if not e.get("has_scene_features")]
    if no_scene:
        print(f"no scene cache for {len(no_scene)} segments (they are skipped): {no_scene[:5]}", flush=True)
    return manifest, [e for e in entries if e.get("has_scene_features")]


def object_embeddings(entries: list[dict], cache_path: Path, device: torch.device) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Uni3D embeddings (bf16-autocast, fp32) of every segment's object mesh, computed once and kept in cache_path."""
    cache = torch.load(cache_path, map_location="cpu") if cache_path.exists() else {}
    todo = [e for e in entries if e["segment_id"] not in cache]
    if todo:
        from gtf.data.uni3d_cache import generate_padded_embeddings
        from gtf.datasets.build_uni3d_cache import load_points
        from gtf.model.arch.conditioning import _load_uni3d

        uni3d = _load_uni3d(str(UNI3D_WEIGHTS)).to(device).eval()

        def encode_bf16(batch):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return uni3d.encode_pc(batch)

        def encode_fp32(batch):
            with torch.autocast(device_type="cuda", enabled=False):
                return uni3d.encode_pc(batch.float())

        by_id = {e["segment_id"]: e for e in todo}
        ids = list(by_id)
        with torch.no_grad():
            bf16, fp32, _, _ = generate_padded_embeddings(ids, lambda sid: load_points(by_id[sid]), encode_bf16, encode_fp32, device=device)
        cache.update({sid: (bf16[i].cpu(), fp32[i].cpu()) for i, sid in enumerate(ids)})
        torch.save(cache, cache_path)
        del uni3d
        torch.cuda.empty_cache()
    return {e["segment_id"]: cache[e["segment_id"]] for e in entries}


# ---------------------------------------------------------------------------
# Windows


def _select_windows(dataset, manifest: Path, starts: list[float] | None) -> None:
    """Restrict the dataset to windows that begin at the requested times (seconds from the segment start)."""
    if not starts:
        return
    from gtf.data.dataset import _load_frame_times_cache

    times = _load_frame_times_cache(str(manifest), dataset.segments)
    windows = []
    for seg_idx in range(len(dataset.segments)):
        t = np.asarray(times[seg_idx], dtype=np.float64) - float(times[seg_idx][0])
        for s in starts:
            first = int(np.searchsorted(t, s - 1e-6))
            if first >= len(t):
                continue
            end = int(np.searchsorted(t, t[first] + dataset.window_seconds - 1e-6))
            windows.append((seg_idx, first, min(end, first + dataset.window_size)))
    dataset.windows = windows


# ---------------------------------------------------------------------------
# Back to the camera frame


def _model_frame_rotation(sample: dict, gravity_direction: np.ndarray | None, present_frame: int) -> torch.Tensor:
    """Rotation from the camera frame of the predictions to the model's gravity-aligned frame."""
    from gtf.model.repr.canonicalize import compute_gravity_alignment

    if bool(sample.get("has_present_anchor", False)):
        gravity_world = sample["gravity_world"].double()
        if gravity_world.norm() < 1e-6:
            return torch.eye(3, dtype=torch.float64)
        g_cam = sample["camera_extrinsics"][present_frame, :3, :3].double() @ gravity_world
        return compute_gravity_alignment(g_cam / g_cam.norm())
    if gravity_direction is None:
        return torch.eye(3, dtype=torch.float64)
    return compute_gravity_alignment(torch.from_numpy(gravity_direction).double())


def to_camera_frame(decoded: dict[str, torch.Tensor], R_align: torch.Tensor, betas: dict[str, np.ndarray], present: torch.Tensor) -> dict[str, np.ndarray]:
    """Undo the gravity alignment of decoded samples (N, T, ...): objects rotate about the origin, hands about their
    shaped root joint with the left-hand mirror convention (gtf.model.repr.canonicalize._rotate_hand_params)."""
    from gtf.model.repr.canonicalize import _rotate_axis_angle, _rotate_hand_params

    R = R_align.T.float()
    out = {}
    N, T = decoded["obj_translation"].shape[:2]
    keep_obj = present[:, 2].view(1, T, 1)
    rot = _rotate_axis_angle(R, decoded["obj_rotation"].reshape(-1, 3).float()).view(N, T, 3)
    out["obj_rotation"] = torch.where(keep_obj, rot, 0.0)
    out["obj_translation"] = torch.where(keep_obj, decoded["obj_translation"].float() @ R.T, 0.0)
    for slot, side in enumerate(("left", "right")):
        valid = present[:, slot].view(1, T).expand(N, T).reshape(-1)
        b = torch.from_numpy(betas[side]).float()
        orient, cam_t = _rotate_hand_params(
            R, decoded[f"hand_{side}_orient"].reshape(-1, 3).float(), decoded[f"hand_{side}_cam_t"].reshape(-1, 3).float(), valid, side, betas=None if b.abs().sum() < 1e-8 else b
        )
        out[f"hand_{side}_orient"] = orient.view(N, T, 3)
        out[f"hand_{side}_cam_t"] = cam_t.view(N, T, 3)
        out[f"hand_{side}_pose"] = torch.where(present[:, slot].view(1, T, 1), decoded[f"hand_{side}_pose"].float(), 0.0)
    return {k: v.cpu().numpy() for k, v in out.items()}


# ---------------------------------------------------------------------------
# Sampling


def predict(args, overrides) -> None:
    from omegaconf import OmegaConf

    from gtf.data.dataset import GTFDataset, gtf_collate
    from gtf.eval.checkpoint import config_from_checkpoint, load_lit_module_from_checkpoint
    from gtf.eval.evaluate import task_inputs
    from gtf.model.flow.inference import generate

    device = torch.device("cuda")
    segments_dir = Path(args.segments).resolve()
    pca_basis = args.pca_basis
    if pca_basis is None:
        from gtf.training.export_checkpoint import PCA_FILE

        state = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
        if "scene_pca" not in state:
            raise SystemExit(f"{args.checkpoint} carries no PCA basis: pass --pca-basis")
        pca_basis = str(segments_dir / PCA_FILE)
        torch.save(state["scene_pca"], pca_basis)
        del state
    manifest, entries = prepare_inputs(segments_dir, pca_basis, args.segment)
    if not entries:
        raise SystemExit("no segment to run")
    embeddings = object_embeddings(entries, segments_dir / "uni3d_embeddings.pt", device)

    cfg = config_from_checkpoint(args.checkpoint)
    cfg = OmegaConf.merge(cfg, OmegaConf.load(REPO_ROOT / "configs" / "benchmark.yaml"), OmegaConf.from_dotlist(overrides))
    cfg.eval.checkpoint_path = str(args.checkpoint)
    if args.stats:
        cfg.data.stats_path = args.stats
    lit = load_lit_module_from_checkpoint(cfg, device, mano_root=MANO_DIR)

    dataset = GTFDataset(
        manifest_path=str(manifest), window_size=int(cfg.eval.window_size), split="val", mano_root=MANO_DIR,
        segment_ids=[e["segment_id"] for e in entries], window_seconds=float(cfg.eval.window_seconds), stride_seconds=float(cfg.eval.stride_seconds),
    )
    _select_windows(dataset, manifest, args.start)
    print(f"{len(dataset.windows)} windows of {len(dataset.segments)} segments, task {args.task}, {cfg.eval.n_samples} samples each", flush=True)

    entry_by_id = {e["segment_id"]: e for e in entries}
    canonicals = {}
    out_root = Path(args.out)
    for i in range(len(dataset)):
        sample = dataset[i]
        sid = str(sample["segment_id"])
        sample["obj_embedding_bf16"], sample["obj_embedding_fp32"] = embeddings[sid]
        if sid not in canonicals:
            entry = entry_by_id[sid]
            with np.load(entry["canonical_path"], allow_pickle=False) as npz:
                canonicals[sid] = {k: npz[k] for k in ("hand_left_betas", "hand_right_betas", "obj_scale", "obj_canonical_verts", "obj_faces")}
                canonicals[sid]["gravity_direction"] = npz["gravity_direction"] if entry.get("has_gravity") and "gravity_direction" in npz else None
        shape = canonicals[sid]

        tokens, obs_mask, presence, valid, conditioning, phys_dt, present_frames = task_inputs(gtf_collate([sample]), args.task, cfg.eval, device)
        samples = []
        for si in range(int(cfg.eval.n_samples)):
            torch.manual_seed(int(cfg.eval.seed) + si)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(cfg.eval.amp)):
                samples.append(generate(lit.model, tokens, obs_mask, presence, conditioning, lit.stats, n_steps=int(cfg.eval.n_steps), phys_dt=phys_dt))
        decoded = {k: torch.cat([s[k] for s in samples]).cpu() for k in samples[0]}

        T = int(valid[0].sum())
        present = presence[0, :, :3].cpu()
        R_align = _model_frame_rotation(sample, shape["gravity_direction"], int(present_frames[0]))
        record = to_camera_frame(decoded, R_align, {"left": shape["hand_left_betas"], "right": shape["hand_right_betas"]}, present)
        record = {k: v[:, :T] for k, v in record.items()}
        first_frame = int(sample["frame_indices"][0])
        record.update(
            frame_indices=sample["frame_indices"][:T].numpy(),
            frame_times_s=sample["frame_times_s"][:T].numpy(),
            valid=valid[0, :T].cpu().numpy(),
            observed=obs_mask[0, :T, :3].cpu().numpy(),
            present=present[:T].numpy(),
            hand_left_betas=shape["hand_left_betas"],
            hand_right_betas=shape["hand_right_betas"],
            obj_scale=shape["obj_scale"],
            obj_canonical_verts=shape["obj_canonical_verts"],
            obj_faces=shape["obj_faces"],
        )
        path = out_root / sid / f"{args.task}_{first_frame:06d}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **record)
        print(f"  {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m gtf.predict", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segments", required=True, help="Folder of segment folders (<segment>/canonical/canonical.npz, metadata.json, video.mp4)")
    parser.add_argument("--segment", action="append", default=None, help="Only this segment (repeatable; default: all)")
    parser.add_argument("--checkpoint", default=str(REPO_ROOT / "checkpoints" / "gtf_main.ckpt"))
    parser.add_argument("--stats", default=None, help="Normalization statistics of a training checkpoint (an exported one carries them)")
    parser.add_argument("--pca-basis", default=None, help="DINOv3 PCA basis of a training checkpoint (an exported one carries it)")
    parser.add_argument("--task", choices=TASKS, default="future_prediction")
    parser.add_argument("--start", type=float, action="append", default=None, help="Window start in seconds from the segment start (repeatable; default: every eval.stride_seconds)")
    parser.add_argument("--out", required=True)
    args, overrides = parser.parse_known_args()
    bad = [o for o in overrides if "=" not in o]
    if bad:
        parser.error(f"unrecognized arguments: {bad}")
    predict(args, overrides)


if __name__ == "__main__":
    main()
