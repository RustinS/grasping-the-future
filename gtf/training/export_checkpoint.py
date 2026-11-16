r"""Pack a training checkpoint into one file for inference, or unpack the files a packed checkpoint carries.

`export` keeps the model weights (fp32), the training config with every machine-specific field (data paths, checkpoint
and log folders, resume target, run name, topology, logging) reset to its default, the training step, and the two
files the model needs besides its weights: the normalization statistics it was trained with (checked against the
digest the training checkpoint recorded) and the PCA basis of the DINOv3 scene features. Optimizer and scheduler
states are dropped.

`unpack` writes those two files back (normalization_stats.pt, scene_dinov3_pca.pt), e.g. into datasets/ before
building a corpus to fine-tune the model on.

Usage:
    python -m gtf.training.export_checkpoint export checkpoints/my_run/last.ckpt checkpoints/my_model.ckpt \
        [--stats datasets/normalization_stats.pt] [--pca-basis datasets/scene_dinov3_pca.pt]
    python -m gtf.training.export_checkpoint unpack checkpoints/gtf_main.ckpt --out datasets
"""

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from gtf.config import GTFConfig
from gtf.eval.checkpoint import config_from_checkpoint, file_digest

# Reset to their defaults: they describe the training machine and run, not the model.
MACHINE_FIELDS = (
    "data.manifest_path",
    "data.stats_path",
    "data.uni3d_cache_path",
    "trainer.devices",
    "trainer.num_nodes",
    "trainer.checkpoint_dir",
    "trainer.log_dir",
    "trainer.resume",
    "trainer.run_name",
    "trainer.use_wandb",
    "trainer.wandb_project",
    "trainer.wandb_entity",
    "eval.checkpoint_path",
)
STATS_FILE, PCA_FILE = "normalization_stats.pt", "scene_dinov3_pca.pt"


def export_checkpoint(src: str, dst: str, stats_path: str | None = None, pca_path: str = "datasets/" + PCA_FILE) -> dict:
    state = torch.load(src, map_location="cpu", mmap=True, weights_only=False)
    cfg = config_from_checkpoint(state)
    stats_path = stats_path or str(cfg.data.stats_path)
    recorded = state.get("stats_content_digest")
    if recorded and file_digest(stats_path) != recorded:
        raise SystemExit(f"{stats_path} is not the statistics file the checkpoint was trained with (sha256 {recorded})")
    defaults = OmegaConf.structured(GTFConfig)
    for key in MACHINE_FIELDS:
        OmegaConf.update(cfg, key, OmegaConf.select(defaults, key), merge=False)
    out = {
        "state_dict": {k: (v.float() if v.is_floating_point() else v).clone() for k, v in state["state_dict"].items()},
        "hyper_parameters": {"cfg": OmegaConf.to_container(cfg, resolve=True)},
        "normalization_stats": torch.load(stats_path, map_location="cpu", weights_only=True),
        "scene_pca": torch.load(pca_path, map_location="cpu", weights_only=False),  # NumPy arrays
        "global_step": state.get("global_step"),
    }
    torch.save(out, dst)
    return out


def unpack_checkpoint(src: str, out_dir: str) -> list[Path]:
    state = torch.load(src, map_location="cpu", mmap=True, weights_only=False)
    if "normalization_stats" not in state or "scene_pca" not in state:
        raise SystemExit(f"{src} is not an exported checkpoint (no packed statistics or PCA basis)")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(state["normalization_stats"], out / STATS_FILE)
    torch.save(state["scene_pca"], out / PCA_FILE)
    return [out / STATS_FILE, out / PCA_FILE]


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m gtf.training.export_checkpoint", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("export", help="pack a training checkpoint for inference")
    e.add_argument("src", help="training checkpoint (.ckpt)")
    e.add_argument("dst", help="output path")
    e.add_argument("--stats", default=None, help="normalization statistics (default: data.stats_path of the checkpoint's config)")
    e.add_argument("--pca-basis", default="datasets/" + PCA_FILE, help="PCA basis of the DINOv3 scene features")
    u = sub.add_parser("unpack", help="write the statistics and the PCA basis of an exported checkpoint")
    u.add_argument("src", help="exported checkpoint")
    u.add_argument("--out", required=True, help="output folder")
    args = parser.parse_args()
    if args.command == "export":
        out = export_checkpoint(args.src, args.dst, args.stats, args.pca_basis)
        n = sum(v.numel() for v in out["state_dict"].values())
        print(f"wrote {args.dst}: {len(out['state_dict'])} tensors, {n / 1e6:.1f}M parameters, step {out['global_step']}")
    else:
        for path in unpack_checkpoint(args.src, args.out):
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
