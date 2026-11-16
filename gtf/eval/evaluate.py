"""GTF evaluation on the seven completion tasks.

Tasks (one observation mask each): future_prediction, hand_to_object, object_to_hand,
temporal_infilling, goal_infilling, goal_hand_only, goal_object_only.

For every window the model generates N samples. Per window the evaluation keeps:
  - translation ADE/FDE (mm) of the object and both wrists under three protocols:
    "single" (sample 0), "best_of_N" (per-metric minimum over the samples) and
    "joint_best_of_N" (the one sample with the lowest joint masked-frame ADE);
  - an action row: per entity, the GT displacement and the max GT excursion over the
    predicted span, with the FDE/ADE of each protocol (columns {obj,rh,lh}_{fde,ade}_{s,k,j});
  - gap records: per masked run of each entity, the FDE at the run frame furthest from its
    observed anchor and at the run's middle frame, and the ADE over the run.
Aggregates: protocol means, action-filtered means (max excursion >= threshold), per-source
tables of both, and the gap-local tables.
"""

import contextlib
import math

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from gtf.eval.metrics import compute_all_metrics
from gtf.data.confidence import neutralize_confidence_leak
from gtf.data.dataset import GTFDataset, load_observed_scene_features, gtf_collate
from gtf.data.uni3d_cache import runtime_uni3d_dataset_kwargs
from gtf.model.flow.inference import generate
from gtf.model.flow.matching import TOKEN_TYPES
from gtf.model.repr.present_anchor import reanchor_tokens_to_present_camera
from gtf.model.repr.tokenizer import StateTokenizer

# Observation masks.


def _finalize_obs_mask(obs_mask: Tensor, presence_mask: Tensor, valid_frames: Tensor) -> Tensor:
    """Apply the relation dependency rule and the presence/valid gates.

    Slot 3 (left_rel) is observed iff slots 0 and 2 are; slot 4 (right_rel) iff slots 1 and 2 are.
    """
    obs_mask[:, :, 3] = obs_mask[:, :, 0] & obs_mask[:, :, 2]
    obs_mask[:, :, 4] = obs_mask[:, :, 1] & obs_mask[:, :, 2]
    obs_mask &= presence_mask
    obs_mask &= valid_frames.unsqueeze(-1)
    return obs_mask


def build_obs_mask_future_prediction(presence_mask: Tensor, valid_frames: Tensor, obs_ratio: float = 0.5) -> Tensor:
    """Future prediction: observe the first obs_ratio of the valid frames, mask the rest."""
    B, T, _ = presence_mask.shape
    obs_mask = torch.zeros_like(presence_mask)

    for b in range(B):
        T_real = valid_frames[b].sum().item()
        K = max(1, int(T_real * obs_ratio))
        obs_mask[b, :K, :3] = presence_mask[b, :K, :3]

    return _finalize_obs_mask(obs_mask, presence_mask, valid_frames)


def build_obs_mask_hand_to_object(presence_mask: Tensor, valid_frames: Tensor) -> Tensor:
    """Hand to object: observe both hands at every frame and the object at frame 0."""
    obs_mask = torch.zeros_like(presence_mask)
    obs_mask[:, :, 0] = presence_mask[:, :, 0]
    obs_mask[:, :, 1] = presence_mask[:, :, 1]
    obs_mask[:, 0, 2] = presence_mask[:, 0, 2]
    return _finalize_obs_mask(obs_mask, presence_mask, valid_frames)


def build_obs_mask_object_to_hand(presence_mask: Tensor, valid_frames: Tensor) -> Tensor:
    """Object to hand: observe the object at every frame and both hands at frame 0."""
    obs_mask = torch.zeros_like(presence_mask)
    obs_mask[:, :, 2] = presence_mask[:, :, 2]
    obs_mask[:, 0, 0] = presence_mask[:, 0, 0]
    obs_mask[:, 0, 1] = presence_mask[:, 0, 1]
    return _finalize_obs_mask(obs_mask, presence_mask, valid_frames)


def build_obs_mask_temporal_infilling(presence_mask: Tensor, valid_frames: Tensor, gap_ratio: float = 0.33) -> Tensor:
    """Temporal infilling: observe prefix and suffix, mask a centered gap of gap_ratio of the valid frames."""
    B, T, _ = presence_mask.shape
    obs_mask = presence_mask.clone()

    for b in range(B):
        T_real = valid_frames[b].sum().item()
        gap_len = max(1, int(T_real * gap_ratio))
        gap_start = (T_real - gap_len) // 2
        obs_mask[b, gap_start : gap_start + gap_len, :3] = False

    return _finalize_obs_mask(obs_mask, presence_mask, valid_frames)


def _build_goal_infilling_core(presence_mask: Tensor, valid_frames: Tensor, n_prefix: int, n_suffix: int, entity_slots: list[int]) -> Tensor:
    """Observe the given entity slots at the first n_prefix and last n_suffix valid frames.

    Frame 0 is always fully observed (all three state slots). Windows with fewer than
    3 valid frames observe frame 0 only, as in training.
    """
    B, T, _ = presence_mask.shape
    obs_mask = torch.zeros_like(presence_mask)

    for b in range(B):
        T_real = valid_frames[b].sum().item()
        if T_real < 3:
            if T_real >= 1:
                obs_mask[b, 0, :3] = presence_mask[b, 0, :3]
            continue
        pfx = min(n_prefix, T_real - 2)  # leave at least one gap frame and one suffix frame
        sfx = min(n_suffix, T_real - pfx - 1)  # leave at least one gap frame
        if pfx + sfx >= T_real:
            sfx = max(1, T_real - pfx - 1)
        for s in entity_slots:
            obs_mask[b, :pfx, s] = presence_mask[b, :pfx, s]
            obs_mask[b, T_real - sfx : T_real, s] = presence_mask[b, T_real - sfx : T_real, s]
        obs_mask[b, 0, :3] = presence_mask[b, 0, :3]

    return _finalize_obs_mask(obs_mask, presence_mask, valid_frames)


def build_obs_mask_goal_infilling(presence_mask: Tensor, valid_frames: Tensor, n_prefix: int = 1, n_suffix: int = 1) -> Tensor:
    """Goal infilling: observe all entities at the prefix and suffix, mask the middle."""
    return _build_goal_infilling_core(presence_mask, valid_frames, n_prefix, n_suffix, entity_slots=[0, 1, 2])


def build_obs_mask_goal_hand_only(presence_mask: Tensor, valid_frames: Tensor, n_prefix: int = 1, n_suffix: int = 1) -> Tensor:
    """Goal infilling (hands only): observe the hands at the prefix and suffix; the object is masked after frame 0."""
    return _build_goal_infilling_core(presence_mask, valid_frames, n_prefix, n_suffix, entity_slots=[0, 1])


def build_obs_mask_goal_object_only(presence_mask: Tensor, valid_frames: Tensor, n_prefix: int = 1, n_suffix: int = 1) -> Tensor:
    """Goal infilling (object only): observe the object at the prefix and suffix; the hands are masked after frame 0."""
    return _build_goal_infilling_core(presence_mask, valid_frames, n_prefix, n_suffix, entity_slots=[2])


_OBS_MASK_BUILDERS = {
    "future_prediction": build_obs_mask_future_prediction,
    "hand_to_object": build_obs_mask_hand_to_object,
    "object_to_hand": build_obs_mask_object_to_hand,
    "temporal_infilling": build_obs_mask_temporal_infilling,
    "goal_infilling": build_obs_mask_goal_infilling,
    "goal_hand_only": build_obs_mask_goal_hand_only,
    "goal_object_only": build_obs_mask_goal_object_only,
}


def _builder_kwargs(task: str, eval_cfg) -> dict:
    """Task-specific mask parameters from the eval config."""
    if task == "future_prediction":
        return {"obs_ratio": float(eval_cfg.future_obs_ratio)}
    if task == "temporal_infilling":
        return {"gap_ratio": float(eval_cfg.infill_gap_ratio)}
    if task in ("goal_infilling", "goal_hand_only", "goal_object_only"):
        if int(eval_cfg.goal_n_prefix) < 1 or int(eval_cfg.goal_n_suffix) < 1:
            raise ValueError(f"goal task {task!r} needs goal_n_prefix >= 1 and goal_n_suffix >= 1 (got {eval_cfg.goal_n_prefix}, {eval_cfg.goal_n_suffix})")
        return {"n_prefix": int(eval_cfg.goal_n_prefix), "n_suffix": int(eval_cfg.goal_n_suffix)}
    return {}


# Per-window records.

# (entity, decoded translation key, token slot, action-row column tag)
_ENTITIES = (
    ("obj", "obj_translation", 2, "obj"),
    ("hand_right", "hand_right_cam_t", 1, "rh"),
    ("hand_left", "hand_left_cam_t", 0, "lh"),
)
# (protocol, action-row column suffix)
_PROTOCOLS = (("single", "s"), ("best_of_N", "k"), ("joint_best_of_N", "j"))


def _finite_min(values: list[float]) -> float:
    """Minimum over the finite values (NaN when none is finite); independent of the order of NaNs."""
    finite = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return min(finite) if finite else float("nan")


def aggregate_multi_sample(per_sample_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Reduce the N per-sample metric dicts of one window.

    Returns {"single": sample 0, "best_of_N": per-key minimum over the samples}. Every
    per-sample metric is an error (lower is better); NaN samples are skipped.
    """
    if not per_sample_metrics:
        return {"single": {}, "best_of_N": {}}
    single = per_sample_metrics[0]
    if len(per_sample_metrics) == 1:
        return {"single": single, "best_of_N": single}
    keys = set()
    for m in per_sample_metrics:
        keys.update(m.keys())
    best = {}
    for key in keys:
        values = [m[key] for m in per_sample_metrics if key in m and not math.isnan(m[key])]
        best[key] = min(values) if values else float("nan")
    return {"single": single, "best_of_N": best}


def compute_motion_magnitude_record(
    gt_trans: Tensor,
    pred_trans_samples: list[Tensor],
    obs_slot: Tensor,
    presence_slot: Tensor,
    valid_frames: Tensor,
) -> dict[str, float] | None:
    """GT motion of one entity over its predicted span (the action-filter key).

    The span runs from the entity's last observed frame (frame 0 if none) to the last valid
    frame. "gt" is the endpoint displacement; "gt_excursion" is the max displacement from the
    reference over the present frames of the span (motion anywhere in the span, so
    out-and-return actions count).

    Args:
        gt_trans: (T, 3) GT translation.
        pred_trans_samples: list of (T, 3) predicted translations, one per sample.
        obs_slot: (T,) bool, observed mask of the entity's slot.
        presence_slot: (T,) bool, presence mask of the entity's slot.
        valid_frames: (T,) bool.

    Returns:
        {"gt", "gt_excursion"} in mm, or None when the window is shorter than 3 frames, the
        span is shorter than 2 frames, or the entity is absent at either end.
    """
    if not pred_trans_samples:
        return None
    T_real = int(valid_frames.sum().item())
    if T_real < 3:
        return None
    end = T_real - 1
    obs_idx = obs_slot[:T_real].nonzero(as_tuple=False)
    ref = int(obs_idx[-1].item()) if len(obs_idx) > 0 else 0
    if end - ref < 2:
        return None
    if not (bool(presence_slot[ref]) and bool(presence_slot[end])):
        return None

    gt_disp = (gt_trans[end] - gt_trans[ref]).norm().item() * 1000.0
    span = (gt_trans[ref : end + 1] - gt_trans[ref]).norm(dim=-1) * 1000.0
    span = span[presence_slot[ref : end + 1].bool()]
    gt_excursion = float(span.max().item()) if span.numel() else gt_disp
    return {"gt": gt_disp, "gt_excursion": gt_excursion}


def compute_gap_local_record(
    gt_trans: Tensor,
    pred_trans_samples: list[Tensor],
    obs_slot: Tensor,
    presence_slot: Tensor,
    valid_frames: Tensor,
    frame_times_s: Tensor,
    joint_idx: int | None = None,
    source: str | None = None,
) -> list[dict] | None:
    """Records for every masked run of one entity, scored against the run's own anchor.

    A run is a maximal stretch of masked and present frames. Its anchor is the last observed
    and present frame before the run; a run without one is anchored on the observed frame
    right after it ("entry_open"); a run with neither is skipped. Per run:
      far     = the run frame furthest from the anchor (run end, or run start when entry_open)
      gt_disp = |gt[far] - gt[anchor]|, gt_excursion = max over the run of |gt[f] - gt[anchor]|
      fde_*   = |pred[far] - gt[far]|, mid_* = the same at the run's middle frame
      ade_*   = mean error over the run frames (per sample, then min for best_of_N)
    Flags: anchored_entry (presence contiguous from the anchor into the run), anchored_exit
    (the frame after the run is observed and present), entry_open. Bounded gaps are runs with
    anchored_entry and anchored_exit; under anchored decode their endpoint converges to the
    observed exit by construction, so the middle-frame error is the skill measure.

    Returns a list of per-run records or None.
    """
    if not pred_trans_samples:
        return None
    T_real = int(valid_frames.sum().item())
    if T_real < 3:
        return None
    obs = obs_slot[:T_real].detach().bool().cpu()
    pres = presence_slot[:T_real].detach().bool().cpu()
    gt = gt_trans[:T_real].detach().float().cpu()
    preds = [p[:T_real].detach().float().cpu() for p in pred_trans_samples]
    times = frame_times_s[:T_real].detach().to(dtype=torch.float64).cpu()
    j = int(joint_idx) if (joint_idx is not None and 0 <= int(joint_idx) < len(preds)) else None

    def _errs(f: int) -> list[float]:
        return [(p[f] - gt[f]).norm().item() * 1000.0 for p in preds]

    records: list[dict] = []
    t = 0
    while t < T_real:
        if bool(obs[t]) or not bool(pres[t]):
            t += 1
            continue
        s = t
        while t < T_real and not bool(obs[t]) and bool(pres[t]):
            t += 1
        e = t - 1  # inclusive run end; absent frames split runs
        ref = -1
        for r in range(s - 1, -1, -1):
            if bool(obs[r]) and bool(pres[r]):
                ref = r
                break
        exit_ref = e + 1 if (e + 1 < T_real and bool(pres[e + 1]) and bool(obs[e + 1])) else -1
        if ref < 0 and exit_ref < 0:
            continue
        entry_open = ref < 0
        anchor = ref if ref >= 0 else exit_ref
        scored = list(range(s, e + 1))
        anchored_entry = False if entry_open else bool(pres[ref : s + 1].all())
        anchored_exit = exit_ref >= 0
        far = s if entry_open else e
        mid = scored[len(scored) // 2]
        fde = _errs(far)
        mid_fde = _errs(mid)
        ade = [float(torch.stack([(p[f] - gt[f]).norm() for f in scored]).mean().item() * 1000.0) for p in preds]
        gt_excursion = float(torch.stack([(gt[f] - gt[anchor]).norm() for f in scored]).max().item()) * 1000.0
        rec: dict = {
            "gt_disp": (gt[far] - gt[anchor]).norm().item() * 1000.0,
            "gt_excursion": gt_excursion,
            "run_len": len(scored),
            "run_s": abs(float(times[far] - times[anchor])),
            "fde_single": fde[0],
            "fde_best_of_N": _finite_min(fde),
            "mid_gt_disp": (gt[mid] - gt[anchor]).norm().item() * 1000.0,
            "mid_fde_single": mid_fde[0],
            "mid_fde_best_of_N": _finite_min(mid_fde),
            "ade_single": ade[0],
            "ade_best_of_N": _finite_min(ade),
            "anchored_entry": anchored_entry,
            "anchored_exit": anchored_exit,
            "entry_open": entry_open,
        }
        if j is not None:
            rec["fde_joint"] = fde[j]
            rec["mid_fde_joint"] = mid_fde[j]
            rec["ade_joint"] = ade[j]
        if source:
            rec["source"] = source
        records.append(rec)
    return records or None


def _joint_best_sample_idx(per_sample_trans, gt_decoded_batch, obs_b, pres_b, valid_b, b) -> int | None:
    """Index of the sample with the lowest joint masked-frame translation ADE summed over the
    predicted entities: the coherent sample behind joint_best_of_N. A NaN (failed) sample is
    never selected."""
    n_samples = len(per_sample_trans["obj"][b])
    if n_samples == 0:
        return None
    if n_samples == 1:
        return 0
    valid = valid_b.detach().cpu()
    scores = torch.zeros(n_samples)
    any_ent = False
    for ent, trans_key, slot, _ in _ENTITIES:
        preds = per_sample_trans[ent][b]
        if len(preds) != n_samples:
            continue
        m = (~obs_b[:, slot] & pres_b[:, slot]).detach().cpu() & valid
        if not bool(m.any()):
            continue
        gt = gt_decoded_batch[trans_key][b].detach().float().cpu()
        any_ent = True
        for s in range(n_samples):
            scores[s] += float((preds[s][m] - gt[m]).norm(dim=-1).mean())
    if not any_ent:
        return 0
    scores = torch.nan_to_num(scores, nan=float("inf"))
    return int(scores.argmin())


# Aggregation.


def _nanmean_dicts(dicts: list[dict[str, float]]) -> dict[str, float]:
    """Per-key mean over the dicts, skipping NaN values."""
    if not dicts:
        return {}
    all_keys = set()
    for d in dicts:
        all_keys.update(d.keys())
    result = {}
    for key in sorted(all_keys):
        values = [d[key] for d in dicts if key in d and not math.isnan(d[key])]
        result[key] = sum(values) / len(values) if values else float("nan")
    return result


def _aggregate_across_windows(all_window_metrics: list[dict]) -> dict:
    """Per-protocol means of the per-window metrics."""
    out = {}
    for protocol, _ in _PROTOCOLS:
        rows = [w[protocol] for w in all_window_metrics if w.get(protocol)]
        if rows:
            out[protocol] = _nanmean_dicts(rows)
    return out


def _aggregate_action_filtered(act_rows: list[dict], thresholds: list[float]) -> dict:
    """Per-entity FDE/ADE over the action windows: max GT excursion >= threshold (mm).

    Each entity is filtered by its own excursion. Per threshold: n (action windows),
    gt_disp_mm (mean endpoint displacement), gt_excursion_mm, the six error means and, for
    each, the number of finite values behind it ({name}_n).
    """
    if not act_rows or not thresholds:
        return {}
    import numpy as np

    out: dict[str, dict] = {}
    for ent, _, _, tag in _ENTITIES:
        gt = np.array([r.get(f"gt_{tag}", np.nan) for r in act_rows], dtype=float)
        exc = np.array([r.get(f"ex_{tag}", np.nan) for r in act_rows], dtype=float)
        cols = {
            f"{metric}_{name}": np.array([r.get(f"{tag}_{metric}_{suffix}", np.nan) for r in act_rows], dtype=float)
            for metric in ("fde", "ade")
            for name, suffix in (("single", "s"), ("best_of_N", "k"), ("joint", "j"))
        }
        if not np.isfinite(gt).any() or not any(np.isfinite(c).any() for c in cols.values()):
            continue  # entity not predicted in this task
        ent_out: dict[str, dict] = {}
        for thr in thresholds:
            mask = np.isfinite(exc) & (exc >= float(thr))
            n = int(mask.sum())
            if n == 0:
                continue
            gtm = gt[mask]
            d = {
                "n": n,
                "gt_disp_mm": float(np.nanmean(gtm)) if np.isfinite(gtm).any() else float("nan"),
                "gt_excursion_mm": float(np.nanmean(exc[mask])),
            }
            for name, arr in cols.items():
                m = mask & np.isfinite(arr)
                d[name] = float(np.mean(arr[m])) if m.any() else float("nan")
                d[f"{name}_n"] = int(m.sum())
            ent_out[f"thr_{int(thr)}mm"] = d
        if ent_out:
            out[ent] = ent_out
    return out


def _aggregate_action_filtered_by_source(act_rows: list[dict], thresholds: list[float]) -> dict:
    """_aggregate_action_filtered per source dataset (for per-source macro averages)."""
    srcs = sorted({r.get("source") for r in act_rows if r.get("source")})
    out = {}
    for s in srcs:
        agg = _aggregate_action_filtered([r for r in act_rows if r.get("source") == s], thresholds)
        if agg:
            out[s] = agg
    return out


_GAP_STAT_KEYS = (
    ("gt_disp", "gt_disp_mm"),
    ("fde_single", "fde_single"),
    ("fde_best_of_N", "fde_best_of_N"),
    ("fde_joint", "fde_joint"),
    ("mid_gt_disp", "mid_gt_disp_mm"),
    ("mid_fde_single", "mid_fde_single"),
    ("mid_fde_best_of_N", "mid_fde_best_of_N"),
    ("mid_fde_joint", "mid_fde_joint"),
    ("ade_single", "ade_single"),
    ("ade_best_of_N", "ade_best_of_N"),
    ("ade_joint", "ade_joint"),
    ("run_s", "run_s"),
)


def aggregate_gap_local(gap_records: dict[str, list[dict]], thresholds: list[float]) -> dict:
    """Aggregate gap-local run records per entity.

    The top level averages the bounded runs (anchored entry and exit). thr_<T>mm restricts
    them to runs whose max GT excursion is >= T mm, with a per-source table; by_source is the
    unfiltered per-source table. Other runs form their own strata: unanchored (presence broken
    between anchor and run), open_run (no observed exit: future prediction) and entry_open
    (anchored on the exit only). Every mean is over finite values; {key}_n counts them.
    """

    def _mean(rows: list[dict], key: str):
        vals = [r[key] for r in rows if key in r and isinstance(r[key], (int, float)) and math.isfinite(r[key])]
        return (sum(vals) / len(vals)) if vals else None

    def _stats(rows: list[dict]) -> dict:
        d: dict = {"n": len(rows)}
        for key, out_key in _GAP_STAT_KEYS:
            v = _mean(rows, key)
            if v is not None:
                d[out_key] = v
                d[f"{out_key}_n"] = sum(1 for r in rows if key in r and isinstance(r[key], (int, float)) and math.isfinite(r[key]))
        return d

    def _with_sources(rows: list[dict]) -> dict:
        d = _stats(rows)
        srcs = sorted({r.get("source") for r in rows if r.get("source")})
        if srcs:
            d["by_source"] = {s: _stats([r for r in rows if r.get("source") == s]) for s in srcs}
        return d

    out: dict[str, dict] = {}
    for ent, all_recs in gap_records.items():
        if not all_recs:
            continue
        bounded = [r for r in all_recs if r["anchored_entry"] and r["anchored_exit"]]
        unanchored = [r for r in all_recs if not r["anchored_entry"] and not r["entry_open"]]
        open_runs = [r for r in all_recs if r["anchored_entry"] and not r["anchored_exit"]]
        entry_open = [r for r in all_recs if r["entry_open"]]
        d = _with_sources(bounded)
        if unanchored:
            d["unanchored"] = _with_sources(unanchored)
            d["n_unanchored_runs"] = len(unanchored)
        d["n_open_runs"] = len(open_runs)
        if open_runs:
            d["open_run"] = _with_sources(open_runs)
        d["n_entry_open_runs"] = len(entry_open)
        if entry_open:
            d["entry_open"] = _with_sources(entry_open)
        for thr in thresholds:
            sub = [r for r in bounded if r["gt_excursion"] >= float(thr)]
            if sub:
                d[f"thr_{int(thr)}mm"] = _with_sources(sub)
        out[ent] = d
    return out


def aggregate_task(raw: dict, thresholds: list[float]) -> dict:
    """Task result from the per-window accumulators (also used to merge sharded runs)."""
    act_rows = raw["act_rows"]
    res = _aggregate_across_windows(raw["all_window_metrics"])
    action_filtered = _aggregate_action_filtered(act_rows, thresholds)
    if action_filtered:
        res["action_filtered"] = action_filtered
    by_src = _aggregate_action_filtered_by_source(act_rows, [0.0] + list(thresholds))
    if by_src:
        res["action_filtered_by_source"] = by_src  # thr_0mm = all windows
        res["source_counts"] = {s: sum(1 for r in act_rows if r.get("source") == s) for s in by_src}
    gap = aggregate_gap_local(raw["gap_records"], thresholds)
    if gap:
        res["action_filtered_gap"] = gap
    res["n_expected_windows"] = raw["n_expected_windows"]
    res["n_windows_evaluated"] = len(raw["all_window_metrics"])
    return res


# Sample generation and task evaluation.


def _generate_all_samples(
    model,
    gt_tokens,
    obs_mask,
    presence_mask,
    conditioning,
    stats,
    eval_cfg,
    phys_dt,
):
    """Generate eval_cfg.n_samples samples for the batch.

    Sample i is drawn after torch.manual_seed(eval_cfg.seed + i). eval_cfg.amp runs generation
    under bf16 autocast (the scene encoder and the codec ops stay fp32 internally). Returns a
    list of (decoded, pred_tokens) pairs at batch size B.
    """
    amp = bool(eval_cfg.amp)
    samples: list = []
    for si in range(int(eval_cfg.n_samples)):
        torch.manual_seed(int(eval_cfg.seed) + si)
        with torch.autocast("cuda", dtype=torch.bfloat16) if amp else contextlib.nullcontext():
            samples.append(
                generate(
                    model,
                    gt_tokens,
                    obs_mask,
                    presence_mask,
                    conditioning,
                    stats,
                    n_steps=int(eval_cfg.n_steps),
                    return_tokens=True,
                    phys_dt=phys_dt,
                )
            )
    return samples


def _to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, Tensor):
            out[k] = v.to(device)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], Tensor):
            out[k] = [t.to(device) for t in v]
        else:
            out[k] = v
    return out


def task_inputs(batch: dict, task: str, eval_cfg, device: torch.device):
    """Model inputs of a collated batch for one task, conditioned like in training: states re-anchored to the
    present camera, scene rows from the last fully observed frame, confidence features neutralized around masked
    frames, exact per-frame time steps.

    Returns (tokens, obs_mask, presence_mask, valid_frames, conditioning, phys_dt, present_frames) on `device`.
    """
    batch_dev = _to_device(batch, device)
    presence = batch_dev["presence_mask"]
    valid = batch_dev["valid_frames"]
    obs_mask = _OBS_MASK_BUILDERS[task](presence, valid, **_builder_kwargs(task, eval_cfg))
    tokens = {name: batch_dev[name] for name in TOKEN_TYPES}
    tokens, present_frames = reanchor_tokens_to_present_camera(tokens, batch_dev, obs_mask, presence, valid)
    conditioning = {
        "obj_embedding_bf16": batch_dev["obj_embedding_bf16"],
        "obj_embedding_fp32": batch_dev["obj_embedding_fp32"],
        "betas_left": batch_dev["betas_left"],
        "betas_right": batch_dev["betas_right"],
        "obj_scale": batch_dev["obj_scale"],
        "confidence_features": neutralize_confidence_leak(
            batch_dev["confidence_features"],
            obs_mask,
            presence_mask=presence,
            valid_frames=valid,
            frame_times_s=batch_dev["frame_times_s"],
        ),
        # scene_features is a list of per-sample dicts; the loader moves its tensors itself.
        "scene_features": load_observed_scene_features(batch_dev, present_frames, obs_mask, presence, valid, strict=True),
    }
    phys_dt = batch_dev["edge_dt_s"].to(presence.device, dtype=torch.float32)
    return tokens, obs_mask, presence, valid, conditioning, phys_dt, present_frames


@torch.no_grad()
def evaluate_task(
    model: torch.nn.Module,
    dataloader: DataLoader,
    stats: dict,
    task: str,
    eval_cfg,
    device: torch.device,
    raw_out: dict | None = None,
    samples_out: list | None = None,
    forecasts_out: list | None = None,
) -> dict:
    """Evaluate one task over the dataloader's windows.

    Each window is conditioned like in training (task_inputs). Returns the aggregated task result
    (aggregate_task). raw_out, when given, receives the per-window accumulators
    (all_window_metrics, act_rows, gap_records, n_expected_windows). samples_out, when given,
    receives one dict per window with every sample's translations (gtf/eval/benchmark.py scores them):
    identity, obs/presence/valid masks (T, 5), the joint-best sample index, and per entity the
    predicted translations (N, T, 3), the reference (T, 3) and the reference excursion in mm.
    forecasts_out, when given, receives one dict per window with the first sample and the reference
    as full decoded states, the masks, the frame times and the MANO shapes (robotics/README.md).

    Sharding (eval.num_shards > 1) assigns whole batches round-robin to shards; the union of
    the shards is the single-process window set.
    """
    tokenizer = StateTokenizer()
    num_shards = max(1, int(eval_cfg.num_shards))
    shard_idx = int(eval_cfg.shard_idx)
    max_samples = int(eval_cfg.max_samples) if int(eval_cfg.max_samples) > 0 else float("inf")
    n_expected_windows = len(dataloader.dataset)
    if math.isfinite(max_samples):
        n_expected_windows = min(n_expected_windows, int(max_samples))

    all_window_metrics: list[dict] = []
    act_rows: list[dict] = []
    gap_records: dict[str, list] = {ent: [] for ent, _, _, _ in _ENTITIES}

    n_processed = 0
    for batch_idx, batch in enumerate(dataloader):
        if n_processed >= max_samples:
            break
        if num_shards > 1 and batch_idx % num_shards != shard_idx:
            n_processed += batch["presence_mask"].shape[0]
            continue

        batch_dev = _to_device(batch, device)
        gt_tokens, obs_mask, presence, valid, conditioning, phys_dt, _ = task_inputs(batch_dev, task, eval_cfg, device)
        B = presence.shape[0]
        n_keep = B if not math.isfinite(max_samples) else min(B, int(max_samples) - n_processed)
        gt_decoded_batch = tokenizer.decode(gt_tokens)

        samples = _generate_all_samples(model, gt_tokens, obs_mask, presence, conditioning, stats, eval_cfg, phys_dt)

        per_sample_metrics: list[list[dict]] = [[] for _ in range(B)]
        per_sample_trans = {ent: [[] for _ in range(B)] for ent, _, _, _ in _ENTITIES}
        for decoded, _ in samples:
            for b in range(n_keep):
                pred_dec_b = {k: v[b] for k, v in decoded.items()}
                gt_dec_b = {k: v[b] for k, v in gt_decoded_batch.items()}
                per_sample_metrics[b].append(compute_all_metrics(pred_dec_b, gt_dec_b, obs_mask[b], presence[b], valid[b]))
                for ent, trans_key, _, _ in _ENTITIES:
                    per_sample_trans[ent][b].append(pred_dec_b[trans_key].cpu())

        for b in range(n_keep):
            agg = aggregate_multi_sample(per_sample_metrics[b])
            j_idx = _joint_best_sample_idx(per_sample_trans, gt_decoded_batch, obs_mask[b], presence[b], valid[b], b)
            if j_idx is not None:
                agg["joint_best_of_N"] = per_sample_metrics[b][j_idx]

            source = batch["source_dataset"][b]
            row = {
                "source": source,
                "segment_id": str(batch["segment_id"][b]),
                "window_start": int(batch["frame_indices"][b][0].item()),
            }
            for ent, trans_key, slot, tag in _ENTITIES:
                rec = compute_motion_magnitude_record(
                    gt_trans=gt_decoded_batch[trans_key][b],
                    pred_trans_samples=per_sample_trans[ent][b],
                    obs_slot=obs_mask[b, :, slot],
                    presence_slot=presence[b, :, slot],
                    valid_frames=valid[b],
                )
                row[f"gt_{tag}"] = rec["gt"] if rec is not None else float("nan")
                row[f"ex_{tag}"] = rec["gt_excursion"] if rec is not None else float("nan")
                gap_recs = compute_gap_local_record(
                    gt_trans=gt_decoded_batch[trans_key][b],
                    pred_trans_samples=per_sample_trans[ent][b],
                    obs_slot=obs_mask[b, :, slot],
                    presence_slot=presence[b, :, slot],
                    valid_frames=valid[b],
                    frame_times_s=batch_dev["frame_times_s"][b],
                    joint_idx=j_idx,
                    source=source,
                )
                if gap_recs:
                    gap_records[ent].extend(gap_recs)
            for proto, suffix in _PROTOCOLS:
                pm = agg.get(proto, {})
                for ent, _, _, tag in _ENTITIES:
                    for metric in ("ade", "fde"):
                        v = pm.get(f"{ent}_{metric}_mm")
                        row[f"{tag}_{metric}_{suffix}"] = float(v) if (v is not None and not math.isnan(v)) else float("nan")
            act_rows.append(row)
            all_window_metrics.append(agg)
            if samples_out is not None:
                rec = {
                    "source": source,
                    "segment_id": row["segment_id"],
                    "window_start": row["window_start"],
                    "obs": obs_mask[b].cpu().numpy(),
                    "presence": presence[b].cpu().numpy(),
                    "valid": valid[b].cpu().numpy(),
                    "joint_idx": j_idx,
                }
                for ent, trans_key, _, tag in _ENTITIES:
                    if per_sample_trans[ent][b]:
                        rec[ent] = {
                            "pred": torch.stack(per_sample_trans[ent][b]).float().numpy(),
                            "gt": gt_decoded_batch[trans_key][b].detach().float().cpu().numpy(),
                            "excursion_mm": row[f"ex_{tag}"],
                        }
                samples_out.append(rec)
            if forecasts_out is not None:
                forecasts_out.append(
                    {
                        "identity": [source, row["segment_id"], row["window_start"]],
                        "frame_times_s": batch["frame_times_s"][b].double().cpu(),
                        "valid_frames": valid[b].cpu(),
                        "presence_mask": presence[b].cpu(),
                        "obs_mask": obs_mask[b].cpu(),
                        "decoded": {k: v[b].detach().float().cpu() for k, v in samples[0][0].items()},
                        "gt_decoded": {k: v[b].detach().float().cpu() for k, v in gt_decoded_batch.items()},
                        "betas_left": batch["betas_left"][b].float().cpu(),
                        "betas_right": batch["betas_right"][b].float().cpu(),
                    }
                )

        n_processed += B

    raw = {
        "all_window_metrics": all_window_metrics,
        "act_rows": act_rows,
        "gap_records": gap_records,
        "n_expected_windows": int(n_expected_windows),
    }
    if raw_out is not None:
        raw_out.update(raw)
    return aggregate_task(raw, [float(t) for t in eval_cfg.action_thresholds_mm])


def shuffle_windows(dataset, eval_cfg):
    """Seeded permutation of the window order (eval.window_shuffle_seed).

    The window list is grouped by source; the permutation is applied before the max_samples
    cap and before sharding, so a capped run still samples every source.
    """
    g = torch.Generator().manual_seed(int(eval_cfg.window_shuffle_seed))
    perm = torch.randperm(len(dataset), generator=g).tolist()
    return Subset(dataset, perm)


def run_evaluation(
    model: torch.nn.Module,
    stats: dict,
    eval_cfg,
    data_cfg,
    mano_root: str,
    device: torch.device,
    raw_by_task: dict | None = None,
) -> dict:
    """Evaluate every task of eval_cfg.tasks on the eval split. Returns {task: result}.

    raw_by_task, when given, receives {task: per-window accumulators} (for sharded runs).
    """
    unknown = [t for t in eval_cfg.tasks if t not in _OBS_MASK_BUILDERS]
    if unknown:
        raise ValueError(f"unknown eval tasks {unknown}; known: {sorted(_OBS_MASK_BUILDERS)}")
    model.eval()

    dataset = GTFDataset(
        manifest_path=str(data_cfg.manifest_path),
        window_size=int(eval_cfg.window_size),
        split=str(eval_cfg.split),
        mano_root=mano_root,
        window_seconds=float(eval_cfg.window_seconds),
        stride_seconds=float(eval_cfg.stride_seconds),
        **runtime_uni3d_dataset_kwargs(data_cfg),
    )
    dataloader = DataLoader(
        shuffle_windows(dataset, eval_cfg),
        batch_size=int(eval_cfg.batch_size),
        shuffle=False,
        num_workers=int(eval_cfg.num_workers),
        collate_fn=gtf_collate,
        pin_memory=True,
        drop_last=False,
    )

    results = {}
    for task in eval_cfg.tasks:
        print(f"  Evaluating task: {task}")
        task_raw: dict | None = {} if raw_by_task is not None else None
        results[task] = evaluate_task(model, dataloader, stats, task, eval_cfg, device, raw_out=task_raw)
        if raw_by_task is not None:
            raw_by_task[task] = task_raw
    return results
