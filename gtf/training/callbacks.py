"""Lightning callbacks for GTF training.

  - StepPrinter: compact step-based console logger.
  - LiveEvalCallback: every N global steps, future_prediction on a fixed val-split subset under the
    evaluation protocol, logged to every logger.
"""

import math
import time

import lightning.pytorch as pl
import torch
from rich.console import Console
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from gtf.paths import MANO_DIR

console = Console()


def _format_duration(seconds: float | None) -> str:
    """Format seconds as H:MM:SS or MM:SS."""
    if seconds is None or seconds < 0:
        return "--:--"
    total_seconds = int(seconds)
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _extract_loss_value(outputs: Tensor | dict | None) -> float | None:
    """Extract a scalar loss value from Lightning's training_step output."""
    if outputs is None:
        return None
    if isinstance(outputs, Tensor):
        return float(outputs.detach().item())
    if isinstance(outputs, dict):
        loss = outputs.get("loss")
        if isinstance(loss, Tensor):
            return float(loss.detach().item())
        if isinstance(loss, (float, int)):
            return float(loss)
    return None


class StepPrinter(pl.Callback):
    """Compact step-based training logger for terminal output.

    Prints: step N/M  XX.XX%  loss X.XXXX  X.XX it/s  elapsed MM:SS  eta MM:SS
    Also prints val/loss after each validation epoch.
    """

    def __init__(self, log_every: int = 1000) -> None:
        self.log_every = log_every
        self._start_time: float | None = None
        self._last_logged_step = 0

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._start_time = time.perf_counter()
        self._last_logged_step = 0
        if trainer.is_global_zero:
            console.print("  [bold]Training[/bold]")

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int) -> None:
        if not trainer.is_global_zero:
            return
        step = trainer.global_step
        max_steps = trainer.max_steps
        if step <= 0 or step == self._last_logged_step:
            return

        should_log = step == 1 or step == max_steps or step % self.log_every == 0
        if not should_log:
            return

        elapsed = None if self._start_time is None else time.perf_counter() - self._start_time
        rate = None if not elapsed or elapsed <= 0 else step / elapsed
        eta = None if rate in (None, 0) else max(max_steps - step, 0) / rate
        pct = (100.0 * step / max_steps) if max_steps > 0 else 0.0
        loss = _extract_loss_value(outputs)
        loss_str = f"{loss:8.4f}" if loss is not None else "   n/a  "
        rate_str = f"{rate:5.2f} it/s" if rate is not None else "  n/a it/s"

        console.print(f"    step {step:>5}/{max_steps:<5}  {pct:6.2f}%  loss {loss_str}  {rate_str}  elapsed {_format_duration(elapsed)}  eta {_format_duration(eta)}")
        self._last_logged_step = step

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        val_loss = trainer.callback_metrics.get("val/loss")
        if isinstance(val_loss, Tensor):
            console.print(f"    val   step {trainer.global_step:>5}/{trainer.max_steps:<5}  loss {val_loss.item():8.4f}")


class LiveEvalCallback(pl.Callback):
    """Logs evaluation metrics every N global steps during training.

    Builds a val-split GTFDataset with the evaluation windowing (cfg.eval) and runs future_prediction on a
    fixed seeded subset of it with the evaluation seed, so the numbers are comparable across checkpoints. Logs the
    mean translation ADE/FDE per entity and the action-filtered FDE (windows whose GT displacement over the
    predicted span is at least 50 mm). Rank 0 does the work while the other ranks wait at a barrier.
    """

    def __init__(self, eval_cfg, every: int, max_samples: int = 64, mano_root: str = MANO_DIR) -> None:
        self.eval_cfg = eval_cfg
        self.every = int(every)
        self.max_samples = int(max_samples)
        self.mano_root = mano_root
        self._loader: DataLoader | None = None
        # Last `global_step // every` bucket that fired. Validation need not land on exact multiples of `every`
        # (gradient accumulation, epoch-end flushes), so the eval fires at the first validation past each boundary.
        self._last_fired_bucket = 0

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Build the val-split eval dataloader once per fit."""
        if stage != "fit" or self._loader is not None:
            return
        from gtf.data.dataset import GTFDataset, gtf_collate
        from gtf.data.uni3d_cache import runtime_uni3d_dataset_kwargs

        cfg = pl_module.cfg
        dataset = GTFDataset(
            manifest_path=str(cfg.data.manifest_path),
            window_size=int(self.eval_cfg.window_size),
            split="val",
            mano_root=self.mano_root,
            segment_cache_max_bytes=int(cfg.data.segment_cache_max_bytes),
            window_seconds=float(self.eval_cfg.window_seconds),
            stride_seconds=float(self.eval_cfg.stride_seconds),
            **runtime_uni3d_dataset_kwargs(cfg.data),
        )
        n = min(len(dataset), self.max_samples)
        # The window list is manifest-ordered (grouped by source), so the subset is a seeded permutation.
        seed = int(self.eval_cfg.window_shuffle_seed)
        if seed >= 0:
            g = torch.Generator().manual_seed(seed)
            idx = torch.randperm(len(dataset), generator=g)[:n].tolist()
        else:
            idx = list(range(n))
        self._loader = DataLoader(
            Subset(dataset, idx),
            batch_size=int(self.eval_cfg.batch_size),
            shuffle=False,
            num_workers=0,
            collate_fn=gtf_collate,
            drop_last=False,
        )

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking or trainer.global_step == 0:
            return
        if self.every <= 0:
            return
        bucket = trainer.global_step // self.every
        if bucket <= self._last_fired_bucket:
            return
        self._last_fired_bucket = bucket

        metrics: dict[str, float] = {}
        try:
            if trainer.is_global_zero and self._loader is not None:
                # Metrics only: a failure here is reported and skipped, never raised (raising on rank 0 would strand
                # the other ranks at the next collective).
                try:
                    metrics = self._run_live_eval(pl_module)
                except Exception:
                    import traceback

                    console.print(f"[red]Live evaluation failed at step {trainer.global_step}; skipped (training unaffected):[/red]")
                    console.print(traceback.format_exc())
                    metrics = {}
        finally:
            # Keep DDP ranks in lock-step while rank 0 runs the eval loop.
            try:
                trainer.strategy.barrier("live_eval")
            except Exception:
                pass

        if trainer.is_global_zero and metrics:
            payload = {f"live_eval/{k}": v for k, v in metrics.items()}
            for logger in trainer.loggers:
                try:
                    logger.log_metrics(payload, step=int(trainer.global_step))
                except Exception:
                    # A misbehaving logger never stops training.
                    continue
            watch = ("hand_left_ade_mm", "hand_right_ade_mm", "obj_fde_mm", "obj_action_fde50_mm")
            bits = [f"{k[:20]}={metrics[k]:.2f}" for k in watch if metrics.get(k) is not None]
            if bits:
                console.print(f"    live  step {trainer.global_step:>5}  " + "  ".join(bits))

    def _run_live_eval(self, pl_module: pl.LightningModule) -> dict[str, float]:
        from gtf.eval.evaluate import build_obs_mask_future_prediction
        from gtf.eval.metrics import compute_all_metrics
        from gtf.data.confidence import neutralize_confidence_leak
        from gtf.data.dataset import load_present_scene_features
        from gtf.model.flow.inference import generate
        from gtf.model.flow.matching import TOKEN_TYPES
        from gtf.model.repr.present_anchor import reanchor_tokens_to_present_camera
        from gtf.model.repr.tokenizer import StateTokenizer

        device = pl_module.device
        tokenizer = StateTokenizer()
        pl_module._move_stats_to_device()
        stats = pl_module.stats
        model = pl_module.model  # the inner GTFModel
        per_sample: list[dict[str, float]] = []

        for batch in self._loader:
            batch_dev: dict = {}
            for k, v in batch.items():
                if isinstance(v, Tensor):
                    batch_dev[k] = v.to(device)
                elif isinstance(v, list) and v and isinstance(v[0], Tensor):
                    batch_dev[k] = [t.to(device) for t in v]
                else:
                    batch_dev[k] = v

            B = batch_dev["presence_mask"].shape[0]
            obs_mask = build_obs_mask_future_prediction(
                batch_dev["presence_mask"],
                batch_dev["valid_frames"],
                obs_ratio=float(self.eval_cfg.future_obs_ratio),
            )
            gt_tokens = {name: batch_dev[name] for name in TOKEN_TYPES}
            gt_tokens, present_frames = reanchor_tokens_to_present_camera(
                gt_tokens,
                batch_dev,
                obs_mask,
                batch_dev["presence_mask"],
                batch_dev["valid_frames"],
            )
            gt_decoded_batch = tokenizer.decode(gt_tokens)
            conditioning = {
                "obj_embedding_bf16": batch_dev.get("obj_embedding_bf16"),
                "obj_embedding_fp32": batch_dev.get("obj_embedding_fp32"),
                "betas_left": batch_dev["betas_left"],
                "betas_right": batch_dev["betas_right"],
                "obj_scale": batch_dev["obj_scale"],
                "confidence_features": batch_dev.get("confidence_features"),
                # Scene rows at the present frame (fully observed in future_prediction).
                "scene_features": load_present_scene_features(batch_dev, present_frames),
            }
            # Same confidence neutralization as training.
            if conditioning["confidence_features"] is not None:
                conditioning["confidence_features"] = neutralize_confidence_leak(
                    conditioning["confidence_features"],
                    obs_mask,
                    presence_mask=batch_dev.get("presence_mask"),
                    valid_frames=batch_dev.get("valid_frames"),
                    frame_times_s=batch_dev.get("frame_times_s"),
                )
            phys_dt = batch_dev["edge_dt_s"].to(batch_dev["presence_mask"].device, dtype=torch.float32)

            # Seeded like the offline evaluation, inside fork_rng so the training RNG stream is untouched.
            devices = [batch_dev["presence_mask"].device] if batch_dev["presence_mask"].is_cuda else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(self.eval_cfg.seed))
                if devices:
                    torch.cuda.manual_seed_all(int(self.eval_cfg.seed))
                decoded = generate(
                    model,
                    gt_tokens,
                    obs_mask,
                    batch_dev["presence_mask"],
                    conditioning,
                    stats,
                    n_steps=int(self.eval_cfg.n_steps),
                    phys_dt=phys_dt,
                )

            for b in range(B):
                m = compute_all_metrics(
                    pred_decoded={k: v[b] for k, v in decoded.items()},
                    gt_decoded={k: v[b] for k, v in gt_decoded_batch.items()},
                    obs_mask=obs_mask[b],
                    presence_mask=batch_dev["presence_mask"][b],
                    valid_frames=batch_dev["valid_frames"][b],
                )
                # GT displacement between the last observed and the last predicted frame, per entity, for the
                # action-filtered FDE below.
                vf = batch_dev["valid_frames"][b]
                T_real = int(vf.sum().item())
                for ent, tk, slot in (("obj", "obj_translation", 2), ("hand_right", "hand_right_cam_t", 1), ("hand_left", "hand_left_cam_t", 0)):
                    try:
                        obs_i = obs_mask[b, :T_real, slot].nonzero(as_tuple=False)
                        pres = batch_dev["presence_mask"][b, :T_real, slot]
                        pred_i = ((~obs_mask[b, :T_real, slot]) & pres).nonzero(as_tuple=False)
                        if len(obs_i) == 0 or len(pred_i) == 0:
                            continue
                        ref, end = int(obs_i[-1].item()), int(pred_i[-1].item())
                        gt_tr = gt_decoded_batch[tk][b]
                        m[f"_actdisp_{ent}"] = float((gt_tr[end] - gt_tr[ref]).norm().item() * 1000.0)
                    except Exception:
                        pass
                per_sample.append(m)

        if not per_sample:
            return {}

        # Mean per key over windows, skipping None/NaN.
        keys: set[str] = set()
        for m in per_sample:
            keys.update(m.keys())
        agg: dict[str, float] = {}
        for k in keys:
            if k.startswith("_actdisp_"):
                continue
            vals = [float(m[k]) for m in per_sample if m.get(k) is not None and not (isinstance(m[k], float) and math.isnan(m[k]))]
            if vals:
                agg[k] = sum(vals) / len(vals)
        # Single-sample action-filtered FDE (GT displacement >= 50 mm).
        for ent, fde_key in (("obj", "obj_fde_mm"), ("hand_right", "hand_right_fde_mm"), ("hand_left", "hand_left_fde_mm")):
            rows = [
                float(m[fde_key])
                for m in per_sample
                if m.get(f"_actdisp_{ent}", 0.0) >= 50.0 and m.get(fde_key) is not None and not (isinstance(m.get(fde_key), float) and math.isnan(m[fde_key]))
            ]
            if rows:
                agg[f"{ent}_action_fde50_mm"] = sum(rows) / len(rows)
                agg[f"{ent}_action_n"] = float(len(rows))
        return agg
