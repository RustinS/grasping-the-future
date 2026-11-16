"""GTF training entry point.

The structured defaults (gtf/config.py) are the main model's recipe; YAML files and key=value overrides merge
on top. Runs single-GPU, multi-GPU DDP or multi-node under torchrun, in bf16-mixed precision, and resumes from
the run's checkpoint directory.

Usage:
    # Main model recipe (2 nodes x 8 GPUs; launch one process per GPU with torchrun)
    python -m gtf.training.train --config configs/main.yaml

    # Override hyperparameters on the command line
    python -m gtf.training.train training.lr=3e-4 trainer.devices=1 trainer.num_nodes=1

    # Resume
    python -m gtf.training.train trainer.resume=auto          # highest complete step in the run directory (default)
    python -m gtf.training.train trainer.resume=last
    python -m gtf.training.train trainer.resume=checkpoints/my_run/last.ckpt

    # Several config files merge left to right
    python -m gtf.training.train --config a.yaml --config b.yaml

    # Run name (otherwise built from the main hyperparameters)
    python -m gtf.training.train trainer.run_name=my_experiment
"""

import argparse
import logging
import os
import sys
import warnings

import torch
from omegaconf import DictConfig, OmegaConf

import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities.exceptions import SIGTERMException

from gtf.paths import MANO_DIR
from gtf.config import GTFConfig
from gtf.training.callbacks import LiveEvalCallback, StepPrinter
from gtf.training.checkpoint_resume import resolve_resume_path
from gtf.training.performance import TrainingSpeedMonitor
from gtf.training.safe_exception_checkpoint import SafeOnExceptionCheckpoint
from gtf.training.trainer import GTFDataModule, GTFLitModule


def _format_count(n: int) -> str:
    """Format a count: 3500 -> '3.5K', 350 -> '350'."""
    if n >= 1000:
        k = n / 1000
        return f"{k:.1f}K" if k != int(k) else f"{int(k)}K"
    return str(n)


def _format_lr(lr: float) -> str:
    """Format a learning rate: 0.0001 -> '1e-04'."""
    return f"{lr:.0e}"


def _total_gpus(cfg: DictConfig) -> int:
    """Total GPU count, trainer.devices x trainer.num_nodes."""
    return int(cfg.trainer.devices) * int(cfg.trainer.num_nodes)


def generate_run_name(cfg: DictConfig) -> str:
    """Run name encoding the main hyperparameters.

    Example: gtf_d512L12H8_lr1e-04_bs32_w5.0s_40Ksteps_1Kwu_16gpu_bf16
    """
    mc, tc, trc = cfg.model, cfg.training, cfg.trainer
    prec = "bf16" if "bf16" in str(trc.precision) else "fp32" if "32" in str(trc.precision) else "fp16"
    parts = [
        "gtf",
        f"d{mc.d_model}L{mc.n_layers}H{mc.n_heads}",
        f"lr{_format_lr(float(tc.lr))}_bs{tc.batch_size}_w{float(tc.window_seconds)}s",
        f"{_format_count(int(tc.max_steps))}steps_{_format_count(int(tc.warmup_steps))}wu",
        f"{_total_gpus(cfg)}gpu_{prec}",
    ]
    if int(trc.accumulate_grad_batches) > 1:
        parts.append(f"accum{trc.accumulate_grad_batches}")
    return "_".join(parts)


def build_strategy(cfg: DictConfig):
    """DDPStrategy for multi-GPU / multi-node, 'auto' for a single device."""
    trc = cfg.trainer
    if int(trc.devices) <= 1 and int(trc.num_nodes) <= 1:
        return "auto"
    # Every trainable parameter receives a gradient each step (the contact/phase heads included), so unused-
    # parameter detection stays off. The graph is not static: the energy-score rollouts run on a random subset
    # of steps.
    return DDPStrategy(
        static_graph=False,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )


def build_callbacks(cfg: DictConfig, checkpoint_dir: str) -> list:
    """Checkpoints, console/throughput logging, LR monitor and live eval."""
    trc = cfg.trainer
    callbacks = []

    # 1. Best-K checkpoint by the monitor. last.ckpt belongs to the periodic callback when there is one, so it
    # always mirrors the most recent step save.
    periodic_enabled = int(trc.save_every_n_steps) > 0
    monitor = str(trc.checkpoint_monitor)
    callbacks.append(
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best-step{step:06d}-val{" + monitor + ":.4f}",
            monitor=monitor,
            mode="min",
            save_top_k=int(trc.save_top_k),
            save_last=bool(trc.save_last) and not periodic_enabled,
            auto_insert_metric_name=False,
        )
    )

    # Exception snapshot, written only after restore and new optimizer progress.
    callbacks.append(SafeOnExceptionCheckpoint(dirpath=checkpoint_dir, filename="on_exception"))

    # 2. Periodic checkpoint every N steps.
    if periodic_enabled:
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="step{step:06d}",
                every_n_train_steps=int(trc.save_every_n_steps),
                save_top_k=-1,
                save_last=bool(trc.save_last),
                auto_insert_metric_name=False,
            )
        )

    # 3. Console step printer and throughput.
    callbacks.append(StepPrinter(log_every=int(trc.step_print_every)))
    callbacks.append(TrainingSpeedMonitor(log_every=max(1, int(trc.log_every_n_steps))))

    # 4. LR monitor.
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    # 5. Live evaluation.
    live_every = int(trc.live_eval_every)
    if live_every > 0:
        callbacks.append(
            LiveEvalCallback(
                eval_cfg=cfg.eval,
                every=live_every,
                max_samples=int(trc.live_eval_max_samples),
                mano_root=MANO_DIR,
            )
        )

    return callbacks


def build_loggers(cfg: DictConfig, run_name: str) -> list:
    """TensorBoard always, W&B when trainer.use_wandb."""
    trc = cfg.trainer
    loggers = [TensorBoardLogger(save_dir=str(trc.log_dir), name=run_name)]

    if bool(trc.use_wandb):
        try:
            import wandb
            from lightning.pytorch.loggers import WandbLogger

            loggers.append(
                WandbLogger(
                    project=str(trc.wandb_project),
                    entity=None if trc.wandb_entity is None else str(trc.wandb_entity),
                    name=run_name,
                    id=run_name,
                    save_dir=str(trc.log_dir),
                    config=OmegaConf.to_container(cfg, resolve=True),
                    resume="allow",
                    settings=wandb.Settings(init_timeout=600),
                )
            )
        except ImportError:
            print("WARNING: wandb not installed, skipping W&B logging")

    return loggers


def main():
    parser = argparse.ArgumentParser(
        prog="python -m gtf.training.train",
        description="GTF training",
        epilog="All remaining arguments are OmegaConf overrides (e.g., training.lr=3e-4)",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help="YAML config file merged over the structured defaults. Repeat to merge several files left to right.",
    )
    parser.add_argument("overrides", nargs="*", help="OmegaConf key=value overrides")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings("ignore", message=".*does not have many workers.*")
    warnings.filterwarnings("ignore", message=".*LeafSpec.*")
    warnings.filterwarnings("ignore", message=".*module.*in eval mode.*")
    # DDP records AccumulateGrad nodes on the setup stream and the forward runs on Lightning's stream; this silences
    # the warning PyTorch emits about it once per parameter on the first backward.
    if hasattr(torch.autograd.graph, "set_warn_on_accumulate_grad_stream_mismatch"):
        torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
    logging.getLogger("lightning.fabric").setLevel(logging.ERROR)

    # Config: structured defaults -> YAML file(s) -> CLI overrides.
    schema = OmegaConf.structured(GTFConfig)
    file_cfgs = [OmegaConf.load(p) for p in (args.config or [])]
    cli_cfg = OmegaConf.from_dotlist(args.overrides) if args.overrides else OmegaConf.create()
    cfg = OmegaConf.merge(schema, *file_cfgs, cli_cfg)

    tc = cfg.training
    trc = cfg.trainer

    # Seeds model init, masks, flow noise and DataLoader workers; the LightningModule re-keys the global RNG per
    # rank and per resumed step.
    pl.seed_everything(int(tc.seed), workers=True)

    run_name = str(trc.run_name) if trc.run_name is not None else generate_run_name(cfg)

    checkpoint_dir = os.path.join(str(trc.checkpoint_dir), run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(str(trc.log_dir), exist_ok=True)

    # The checkpoint to resume from (complete training state only), or None for a fresh start.
    ckpt_path = resolve_resume_path(str(trc.resume) if trc.resume is not None else None, checkpoint_dir)

    print(f"Run:            {run_name}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    if ckpt_path:
        print(f"Resuming from:  {ckpt_path}")
    print(f"Precision:      {trc.precision}")
    print(f"GPUs:           {_total_gpus(cfg)}")
    print(f"Config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    strategy = build_strategy(cfg)
    callbacks = build_callbacks(cfg, checkpoint_dir)
    loggers = build_loggers(cfg, run_name)

    has_cuda = torch.cuda.is_available()
    accelerator = "gpu" if has_cuda else "cpu"
    precision = str(trc.precision) if has_cuda else "32-true"

    # The train loader brings its own distributed weighted sampler and the val loader shards itself, so
    # Lightning must not wrap them in a DistributedSampler.
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=int(trc.devices),
        num_nodes=int(trc.num_nodes),
        strategy=strategy,
        precision=precision,
        max_steps=int(tc.max_steps),
        gradient_clip_val=float(tc.gradient_clip_norm),
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=int(trc.accumulate_grad_batches),
        log_every_n_steps=int(trc.log_every_n_steps),
        val_check_interval=int(trc.val_check_interval),
        check_val_every_n_epoch=None,
        num_sanity_val_steps=int(trc.num_sanity_val_steps),
        callbacks=callbacks,
        logger=loggers,
        enable_model_summary=False,
        enable_progress_bar=bool(trc.enable_progress_bar),
        benchmark=bool(trc.benchmark),
        use_distributed_sampler=False,
    )

    # stats_path=None: the LitModule reads cfg.data.stats_path (so CLI overrides apply).
    lit = GTFLitModule(
        cfg=cfg,
        mano_root=MANO_DIR,
        stats_path=None,
    )

    dm = GTFDataModule(
        cfg=cfg,
        mano_root=MANO_DIR,
        num_workers=int(trc.num_workers),
        pin_memory=bool(trc.pin_memory),
        persistent_workers=bool(trc.persistent_workers),
    )

    # On SIGTERM (preemption), Lightning raises SIGTERMException at the next iteration boundary after running the
    # on_exception hooks (which write the exception checkpoint). Tear down the NCCL process group here, before
    # interpreter shutdown, so every rank finalizes NCCL within the launcher's grace period instead of being
    # killed mid-collective; then exit with the conventional SIGTERM code so the job can be retried and resume.
    try:
        trainer.fit(lit, dm, ckpt_path=ckpt_path)
    except SIGTERMException:
        rank = int(os.environ.get("RANK", 0))
        print(f"[rank{rank}] SIGTERM received: destroying the NCCL process group and exiting", flush=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except Exception as e:
                print(f"[rank{rank}] destroy_process_group raised during SIGTERM teardown: {e}", flush=True)
        sys.exit(143)

    for cb in trainer.callbacks:
        if isinstance(cb, ModelCheckpoint) and cb.monitor == str(trc.checkpoint_monitor):
            if cb.best_model_path:
                best_score = cb.best_model_score
                score_str = f" ({cb.monitor}={best_score.item():.4f})" if best_score is not None else ""
                print(f"Best checkpoint: {cb.best_model_path}{score_str}")
            break


if __name__ == "__main__":
    main()
