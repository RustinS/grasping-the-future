"""Normalization statistics, dataset splits and cadence helpers.

Provides:
  - load_stats / validate_stats_provenance: load a stats file and check its conventions
  - get_split: dataset split assignment (DexYCB subject-based, recording-grouped hashes)
  - median_frame_step: effective cadence of a keyframe-subsampled canonical
  - COMPONENT_GROUPS: the velocity-statistics component groups as (token_type, slice) pairs
  - __main__: compute window-sampled stats for a training config

The statistics are accumulated over the training exposure itself: every train
window of the training dataset, weighted by its expected sampler weight over the
whole run, with the observation mask drawn per window, the present-camera
re-anchor and the centering applied exactly as in the training step.
Velocity/acceleration stds use each window's frame times and are converted to
normalized-position units for direct use by the Δ-codec.
The global SE(3) augmentation is left out (the same stats normalize evaluation inputs);
stride augmentation fires as in training.

Usage:
    python -m gtf.data.compute_stats --config configs/main.yaml [--config ...] [--output PATH] [key=value ...]
"""

import hashlib
import re
from pathlib import Path

import torch
from torch import Tensor

from gtf.paths import MANO_DIR
from gtf.model.repr.tokenizer import HAND_ORIENT_SLICE, HAND_CAM_T_SLICE, HAND_POSE_SLICE, OBJ_ROTATION_SLICE, OBJ_TRANSLATION_SLICE

# Token types, defined locally to avoid a circular import through gtf.model.flow.
TOKEN_TYPES = ["left_state", "right_state", "obj_state", "left_rel", "right_rel"]
TOKEN_SLOTS = {"left_state": 0, "right_state": 1, "obj_state": 2, "left_rel": 3, "right_rel": 4}

# Component groups of the velocity/acceleration statistics: group name → (token_type, slice)
# streams that share one σ (pooled across streams in normalized space).
COMPONENT_GROUPS: dict[str, list[tuple[str, slice]]] = {
    "hand_root_translation": [("left_state", HAND_CAM_T_SLICE), ("right_state", HAND_CAM_T_SLICE)],
    "hand_root_rotation": [("left_state", HAND_ORIENT_SLICE), ("right_state", HAND_ORIENT_SLICE)],
    "hand_articulation": [("left_state", HAND_POSE_SLICE), ("right_state", HAND_POSE_SLICE)],
    "obj_translation": [("obj_state", OBJ_TRANSLATION_SLICE)],
    "obj_rotation": [("obj_state", OBJ_ROTATION_SLICE)],
    "relation": [("left_rel", slice(None)), ("right_rel", slice(None))],
}


def load_stats(path: str | Path) -> dict:
    """Load normalization stats from a .pt file.

    Args:
        path: Path to the stats file (e.g., datasets/normalization_stats.pt).

    Returns:
        Stats dict with 'mean', 'std', 'velocity_std', 'acceleration_std' sub-dicts.
    """
    return torch.load(path, weights_only=True, map_location="cpu")


# Conventions a stats file must have been accumulated under (the token geometry
# the model normalizes): physical relation frame, point-to-triangle fingertip
# distances, per-frame timestamps, shaped-beta wrist pivot, side-aware centering,
# window-sampled exposure.
_STATS_CONVENTIONS = {
    "stats_mode": "window_sampled",
    "relation_frame": "physical",
    "contact_convention": "triangle",
    "timing_mode": "frame_times",
    "reanchor_beta_pivot": True,
    "side_aware_centering": True,
}


def validate_stats_provenance(stats: dict, strict: bool = False) -> list[str]:
    """Check that a stats file was computed under the model's token conventions.

    Normalizing with stats accumulated under another convention is a silent
    distribution mismatch (relation stds are cm-scale), so a mismatch is reported.

    Args:
        stats: loaded stats dict.
        strict: raise on mismatch instead of returning the messages.

    Returns:
        List of human-readable mismatch strings (empty = consistent).
    """
    problems: list[str] = []
    for key, want in _STATS_CONVENTIONS.items():
        if key not in stats:
            problems.append(f"stats file lacks {key!r} (expected {want!r})")
            continue
        have = stats[key]
        have = bool(have) if isinstance(want, bool) else str(have)
        if have != want:
            problems.append(f"stats.{key}={have!r}, expected {want!r}")
    if problems and strict:
        raise ValueError("Stats provenance mismatch — recompute with python -m gtf.data.compute_stats: " + "; ".join(problems))
    return problems


# Dataset split assignment.
# Recording-level split-grouping keys per source (manifest entries carry the
# segment metadata). Missing/empty keys fall back to the segment_id hash.
_SOURCE_GROUP_KEYS: dict[str, tuple[str, ...]] = {
    "taco": ("taco_triplet", "taco_sequence"),
    "oakink2": ("oakink2_sequence",),
    "show3d": ("show3d_subject_id", "show3d_scene_id"),
    "hocap": ("hocap_subject", "hocap_sequence"),
    "ho3d": ("sequence",),
    "h2o": ("h2o_sequence",),
}

_DERIVED_SEGMENT_SUFFIX_RE = re.compile(r"__(?:iw|shot)\d+$")
_ACTION100M_TIME_RANGE_RE = re.compile(r"_[-+]?\d+(?:\.\d+)?-[-+]?\d+(?:\.\d+)?s$")


def strip_derived_segment_suffixes(segment_id: str) -> str:
    """Strip any terminal chain of child suffixes (``__shotN``, ``__iwN``).

    A child can be nested (for example ``__shot1__iw2``); stripping the terminal
    component repeatedly keeps sibling windows in one split.
    """
    key = str(segment_id)
    while True:
        parent = _DERIVED_SEGMENT_SUFFIX_RE.sub("", key)
        if parent == key:
            return key
        key = parent


def fallback_split_group_key(source: str, segment_id: str) -> str:
    """Same-recording key for sources without explicit manifest group fields."""
    key = strip_derived_segment_suffixes(segment_id)
    if source == "action100m":
        # Action100M IDs are {video_uid}_{start}-{end}s. Cadence/interaction/
        # shot children and separate time clips from one video must all remain
        # in one split.
        key = _ACTION100M_TIME_RANGE_RE.sub("", key)
    return key


def get_split(segment_entry: dict) -> str:
    """Determine train/val/test split for a manifest segment entry.

    Split protocol:
      - DexYCB: subject-based using 'dexycb_subject' metadata field.
        s01-s08 = train, s09 = val, s10 = test.
      - HOT3D: clip-level hash on 'hot3d_clip_id' (not segment_id).
        One clip can produce multiple segments — hashing segment_id would leak.
      - taco/oakink2/show3d/hocap/ho3d/h2o: recording-level hash on the source's
        sequence keys (_SOURCE_GROUP_KEYS) — one recording yields many segments.
      - Others (Action100M, etc.): hash on the source video (derived-child
        suffixes and Action100M time ranges stripped).

    Args:
        segment_entry: Dict from canonical_manifest.json with at least 'source_dataset'
            and 'segment_id'. DexYCB entries need 'dexycb_subject'. HOT3D entries
            need 'hot3d_clip_id'.

    Returns:
        'train', 'val', or 'test'.
    """
    source = segment_entry.get("source_dataset", "")
    segment_id = segment_entry["segment_id"]

    if source == "dexycb":
        subject = segment_entry["dexycb_subject"]  # e.g., "s01"
        subject_num = int(subject[1:])  # "s01" → 1
        if subject_num <= 8:
            return "train"
        elif subject_num == 9:
            return "val"
        else:
            return "test"

    if source == "hot3d":
        # Hash on clip_id to keep segments from same clip in same split
        clip_id = segment_entry["hot3d_clip_id"]
        h = int(hashlib.sha256(clip_id.encode()).hexdigest(), 16) % 100
    elif source in _SOURCE_GROUP_KEYS and all(segment_entry.get(k) for k in _SOURCE_GROUP_KEYS[source]):
        # Recording-level grouping: one recording yields many segments (taco
        # tool/target pairs cover identical frames; oakink2 has several
        # per-object segments per sequence; show3d / hocap / ho3d have
        # multi-object or multi-run sequences).
        group = source + "|" + "|".join(str(segment_entry[k]) for k in _SOURCE_GROUP_KEYS[source])
        h = int(hashlib.sha256(group.encode()).hexdigest(), 16) % 100
    else:
        # Same-video grouping: `{seg}__iwN` children are windows of their parent's
        # extraction, and Action100M segments `{video_uid}_{start}-{end}s` of one
        # video share the scene, so all of them hash on the video.
        key = fallback_split_group_key(source, segment_id)
        h = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 100

    if h < 90:
        return "train"
    elif h < 95:
        return "val"
    else:
        return "test"


def median_frame_step(frame_indices) -> float:
    """Median source-video frame step of a canonical's keyframes.

    The stored ``fps`` field is the video rate, but extracted (Action100M) and
    OakInk2 canonicals are keyframe-subsampled (frame_indices stride 3-6), so
    the trajectory's effective cadence is ``fps / median_frame_step``.

    Accepts Tensor / ndarray / list / None. Returns 1.0 (stride-1, i.e. the
    stored fps is already effective) when there are <2 frames or no positive
    steps — never raises.
    """
    if frame_indices is None:
        return 1.0
    fi = torch.as_tensor(frame_indices).reshape(-1).double()
    if fi.numel() < 2:
        return 1.0
    diffs = fi[1:] - fi[:-1]
    pos = diffs[diffs > 0]
    if pos.numel() == 0:
        return 1.0
    return float(pos.median().item())


class _WelfordAccumulator:
    """Online mean/variance using Chan's parallel update.

    ``update(..., weight=w)`` accumulates each row with frequency-style mass w.
    With weight 1.0 everywhere the result uses the Bessel (n-1) variance; once
    any weight != 1.0 is used, ``finalize`` switches to the population form M2/W
    (Bessel correction is ill-defined for arbitrary masses)."""

    def __init__(self, dim: int):
        self.n = 0
        self.mean = torch.zeros(dim, dtype=torch.float64)
        self.M2 = torch.zeros(dim, dtype=torch.float64)
        self.weighted = False

    def update(self, values: Tensor, weight: float = 1.0) -> None:
        """Add a batch of values. values: (N, dim) or (dim,); weight = per-row mass.

        Vectorized Chan parallel batch update (exact and order-independent up to
        float64 rounding)."""
        values = values.reshape(-1, self.mean.shape[0]).to(dtype=torch.float64, device="cpu")
        n_b = values.shape[0]
        if n_b == 0 or weight <= 0.0:
            return
        if weight != 1.0:
            self.weighted = True
        w_b = n_b * float(weight)
        batch_mean = values.mean(dim=0)
        batch_M2 = ((values - batch_mean) ** 2).sum(dim=0) * float(weight)
        if self.n == 0:
            self.mean = batch_mean
            self.M2 = batch_M2
            self.n = w_b
            return
        delta = batch_mean - self.mean
        tot = self.n + w_b
        self.mean = self.mean + delta * (w_b / tot)
        self.M2 = self.M2 + batch_M2 + (delta * delta) * (self.n * w_b / tot)
        self.n = tot

    def finalize(self) -> tuple[Tensor, Tensor]:
        """Return (mean, std) as float32 tensors. Clamps std to min 1e-6."""
        if (self.weighted and self.n <= 0.0) or (not self.weighted and self.n < 2):
            return self.mean.float(), torch.ones_like(self.mean).float()
        variance = self.M2 / self.n if self.weighted else self.M2 / (self.n - 1)
        std = variance.sqrt().clamp(min=1e-6)
        return self.mean.float(), std.float()


def _combine_streams_normalized(
    accumulators: dict[tuple[str, str], _WelfordAccumulator],
    streams: list[tuple[str, slice]],
    group_name: str,
    std_dict: dict[str, Tensor],
    dim: int,
) -> Tensor:
    """Combine per-stream Welford accumulators in normalized space using the parallel formula.

    Converts each stream's (mean, M2) to normalized space, then combines exactly:
      M2_ab = M2_a + M2_b + delta² * n_a * n_b / (n_a + n_b)
    where delta = mean_b_norm - mean_a_norm.

    Returns the combined std in normalized space as a float32 tensor.
    """
    # Collect streams that have enough data
    converted = []  # list of (n, mean_norm, M2_norm) in float64
    any_weighted = False
    for token_type, slc in streams:
        key = (group_name, token_type)
        acc = accumulators[key]
        if (acc.weighted and acc.n <= 0.0) or (not acc.weighted and acc.n < 2):
            continue
        any_weighted = any_weighted or acc.weighted
        pos_std = _get_component_std(token_type, slc, std_dict).to(torch.float64).clamp(min=1e-6)
        mean_norm = acc.mean / pos_std
        M2_norm = acc.M2 / pos_std.pow(2)
        converted.append((acc.n, mean_norm, M2_norm))

    if not converted:
        return torch.ones(dim)

    # Combine using parallel Welford: fold streams left-to-right
    n_ab, mean_ab, M2_ab = converted[0]
    for n_b, mean_b, M2_b in converted[1:]:
        n_total = n_ab + n_b
        delta = mean_b - mean_ab
        mean_ab = (n_ab * mean_ab + n_b * mean_b) / n_total
        M2_ab = M2_ab + M2_b + delta.pow(2) * n_ab * n_b / n_total
        n_ab = n_total

    # Population form under sampler weighting (n_ab is a mass, not a count).
    variance = M2_ab / n_ab if any_weighted else M2_ab / (n_ab - 1)
    return variance.sqrt().clamp(min=1e-6).float()


def _get_component_std(token_type: str, slc: slice, std_dict: dict[str, Tensor]) -> Tensor:
    """Extract the per-component std for a specific (token_type, slice) from the position std dict."""
    full_std = std_dict[token_type]
    # Handle slice(None) for relation tokens
    if slc == slice(None):
        return full_std
    return full_std[slc]


# Window-sampled statistics.
_STATE_DIMS = {"left_state": 99, "right_state": 99, "obj_state": 9, "left_rel": 14, "right_rel": 14}
_GROUP_DIMS = {
    "hand_root_translation": 3,
    "hand_root_rotation": 6,
    "hand_articulation": 90,
    "obj_translation": 3,
    "obj_rotation": 6,
    "relation": 14,
}


def compute_window_sampled_stats(cfg, *, seed: int = 1234, mano_root: str = MANO_DIR) -> dict:
    """Accumulate normalization stats over the training exposure of ``cfg``.

    For every train window of the training dataset:
      - weight = its expected sampler weight over the whole run (the curriculum
        sampler's ``weights``, one replica), rescaled to mean 1;
      - the sample comes from the dataset's ``__getitem__`` (stride augmentation
        fires with its training probability);
      - the observation mask is drawn per window as in the training step, then
        the present-camera re-anchor and centering are applied, so positions are
        accumulated in the distribution the model normalizes;
      - velocities/accelerations use the window's per-transition dt.

    Every draw is seeded per window index, so reruns are identical. Any window
    read failure is fatal (stats must cover the exact population).

    Args:
        cfg: full GTFConfig (OmegaConf).
        seed: base of the per-window seed.
        mano_root: MANO model directory.

    Returns:
        The stats dict (``mean``, ``std``, ``velocity_std``, ``acceleration_std``
        plus convention / exposure provenance).
    """
    import time

    from omegaconf import OmegaConf

    from gtf.data.dataset import gtf_collate
    from gtf.model.flow.masking import sample_mask_batched
    from gtf.model.repr.present_anchor import reanchor_tokens_to_present_camera
    from gtf.model.repr.tokenizer import center_tokens_at_present
    from gtf.training.trainer import GTFDataModule

    position_accumulators = {name: _WelfordAccumulator(dim) for name, dim in _STATE_DIMS.items()}
    # Velocity/acceleration accumulators per stream (not pooled across left/right):
    # the conversion to normalized space must use each stream's own position std.
    vel_accumulators = {}
    acc_accumulators = {}
    for group_name, streams in COMPONENT_GROUPS.items():
        for token_type, _ in streams:
            vel_accumulators[(group_name, token_type)] = _WelfordAccumulator(_GROUP_DIMS[group_name])
            acc_accumulators[(group_name, token_type)] = _WelfordAccumulator(_GROUP_DIMS[group_name])

    dm = GTFDataModule(OmegaConf.to_container(cfg, resolve=True), mano_root=mano_root)
    dm.setup("fit")
    ds = dm._train_dataset
    sampler = dm._build_weighted_sampler()  # no Lightning trainer attached -> one replica, full weight vector
    w = torch.as_tensor(sampler.weights, dtype=torch.float64)
    if len(w) != len(ds.windows):
        raise SystemExit(f"window-sampled stats: weight vector ({len(w)}) != window count ({len(ds.windows)})")

    # Training-step regime probabilities (field order of masking.regimes).
    regime_probs = {str(k): float(v) for k, v in cfg.masking.regimes.items()}

    seg_id_of_window = [str(ds.segments[si].get("segment_id")) for si, _s, _e in ds.windows]
    used_idx = [i for i in range(len(ds.windows)) if float(w[i]) > 0.0]
    n_zero_w = len(ds.windows) - len(used_idx)
    if not used_idx:
        raise SystemExit("window-sampled stats: zero windows with positive sampler weight")
    _wscale = len(used_idx) / float(w[used_idx].sum())  # mean per-window weight = 1 (scale-invariant)

    failures: list[tuple[str, str]] = []
    n_done = 0
    n_frames = 0
    by_src: dict[str, float] = {}

    for k, i in enumerate(used_idx):
        sid = seg_id_of_window[i]
        sample = None
        err = None
        for _attempt in range(2):
            _sub0 = ds.load_substitutions
            try:
                # One seed per window covers both getitem-time draws (stride aug) and
                # the observation mask below — deterministic and order-independent.
                torch.manual_seed((seed * 1000003 + i) % (2**62))
                sample = ds[i]
                if ds.load_substitutions != _sub0:
                    sample = None
                    err = "neighbor substitution during window load (canonical unreadable)"
                    break  # a retry would substitute again
                err = None
                break
            except Exception as e:  # noqa: BLE001
                err = str(e)
                time.sleep(0.5)
        if sample is None:
            failures.append((f"window[{i}] seg={sid}", str(err)))
            continue

        batch = gtf_collate([sample])
        obs = sample_mask_batched(batch["presence_mask"], batch["valid_frames"], regime_probs)
        tokens_dict = {name: batch[name] for name in TOKEN_TYPES}
        tokens_dict, present_frames = reanchor_tokens_to_present_camera(tokens_dict, batch, obs, batch["presence_mask"], batch["valid_frames"])
        centered, _origin = center_tokens_at_present(tokens_dict, obs, batch["presence_mask"], present_frames)

        pm = batch["presence_mask"][0]  # (T, 5)
        vf = batch["valid_frames"][0]  # (T,)
        wt = float(w[i]) * _wscale
        for name in TOKEN_TYPES:
            valid = pm[:, TOKEN_SLOTS[name]] & vf
            if valid.any():
                position_accumulators[name].update(centered[name][0][valid], weight=wt)

        edge_dt = batch["edge_dt_s"][0].to(dtype=torch.float64)
        for group_name, streams in COMPONENT_GROUPS.items():
            for token_type, slc in streams:
                valid = pm[:, TOKEN_SLOTS[token_type]] & vf
                if valid.sum() < 3:
                    continue
                x_full = centered[token_type][0]
                x = x_full[:, slc] if slc != slice(None) else x_full
                stream_key = (group_name, token_type)
                valid_pair = valid[:-1] & valid[1:]
                pair_vel = (x[1:] - x[:-1]) / edge_dt.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
                if bool(valid_pair.any()):
                    vel_accumulators[stream_key].update(pair_vel[valid_pair], weight=wt)
                valid_triple = valid[:-2] & valid[1:-1] & valid[2:]
                if bool(valid_triple.any()):
                    # Interval velocities live at edge midpoints. The separation
                    # between adjacent midpoints is 0.5*(dt_i + dt_{i+1}).
                    acc_dt = 0.5 * (edge_dt[:-1] + edge_dt[1:])
                    acc_val = (pair_vel[1:] - pair_vel[:-1]) / acc_dt.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
                    acc_accumulators[stream_key].update(acc_val[valid_triple], weight=wt)

        n_done += 1
        n_frames += int((pm.any(dim=-1) & vf).sum())
        _src = str(batch["source_dataset"][0])
        by_src[_src] = by_src.get(_src, 0.0) + wt * int(vf.sum())
        if (k + 1) % 2000 == 0:
            print(f"  window-sampled: {k + 1}/{len(used_idx)} windows ({n_frames} frames)")

    if failures:
        print(f"\nStatistics failed: {len(failures)} of {len(used_idx)} windows could not be read (after retries):")
        for _sid, _err in failures[:20]:
            print(f"   - {_sid}: {_err}")
        raise SystemExit("Not writing statistics from an incomplete population. Fix the unreadable canonicals and re-run.")

    _tot = sum(by_src.values()) or 1.0
    print(f"  window-sampled exposure: {n_done} windows, {n_zero_w} zero-weight skipped")
    print("  per-source exposure share: " + ", ".join(f"{k}={v / _tot:.3f}" for k, v in sorted(by_src.items(), key=lambda kv: -kv[1])))

    mean_dict = {}
    std_dict = {}
    for name in TOKEN_TYPES:
        mean_dict[name], std_dict[name] = position_accumulators[name].finalize()

    # Velocity/acceleration std: per-stream conversion to normalized space, then
    # exact combination via the parallel Welford formula.
    vel_std_dict = {}
    acc_std_dict = {}
    for group_name, streams in COMPONENT_GROUPS.items():
        vel_std_dict[group_name] = _combine_streams_normalized(vel_accumulators, streams, group_name, std_dict, _GROUP_DIMS[group_name])
        acc_std_dict[group_name] = _combine_streams_normalized(acc_accumulators, streams, group_name, std_dict, _GROUP_DIMS[group_name])

    stats = {
        "mean": mean_dict,
        "std": std_dict,
        "velocity_std": vel_std_dict,
        "acceleration_std": acc_std_dict,
        "n_segments": len(ds.segments),
        "n_frames": n_frames,
        **_STATS_CONVENTIONS,
        "relations_precentered": True,  # relations are computed from uncentered geometry
        # velocity_std is in normalized-position units per second.
        "velocity_std_units": "normalized_position_per_second",
        "manifest_path": str(cfg.data.manifest_path),
        "ws_seed": int(seed),
        "ws_windows": n_done,
        "ws_windows_zero_weight": n_zero_w,
        "ws_augs_excluded": ["apply_global_se3_augmentation"],
    }
    return stats


if __name__ == "__main__":
    import argparse

    from omegaconf import OmegaConf

    from gtf.config import GTFConfig

    parser = argparse.ArgumentParser(prog="python -m gtf.data.compute_stats", description="Compute window-sampled normalization statistics for GTF training")
    parser.add_argument("--config", action="append", default=None, help="YAML config file; repeat to merge left-to-right (the training config chain)")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: data.stats_path of the config)")
    parser.add_argument("--seed", type=int, default=1234, help="Base of the per-window seed (identical reruns)")
    parser.add_argument("--mano-root", type=str, default=MANO_DIR)
    parser.add_argument("overrides", nargs="*", help="OmegaConf key=value overrides")
    args = parser.parse_args()

    cfg = OmegaConf.structured(GTFConfig)
    for path in args.config or []:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(args.overrides)))
    output = Path(args.output or str(cfg.data.stats_path))

    stats = compute_window_sampled_stats(cfg, seed=args.seed, mano_root=args.mano_root)

    # Write atomically: a killed save never leaves a truncated stats file.
    output.parent.mkdir(parents=True, exist_ok=True)
    _tmp = Path(str(output) + ".tmp")
    torch.save(stats, str(_tmp))
    _tmp.replace(output)
    print(f"Saved stats to {output}")

    for name in TOKEN_TYPES:
        m, s = stats["mean"][name], stats["std"][name]
        print(f"  {name}: mean range [{m.min():.4f}, {m.max():.4f}], std range [{s.min():.4f}, {s.max():.4f}]")
    for group_name, v in stats["velocity_std"].items():
        print(f"  σ_vel({group_name}): [{v.min():.4f}, {v.max():.4f}]")
