"""Checkpoint loading for evaluation.

A checkpoint stores its training config under hyper_parameters["cfg"]. config_from_checkpoint
turns it into a GTFConfig; load_lit_module_from_checkpoint loads the weights with strict key and
shape matching.
"""

import hashlib
import os

import torch


def _plain(cfg):
    """OmegaConf config -> plain dict; a dict or None is returned unchanged."""
    if cfg is None or isinstance(cfg, dict):
        return cfg
    from omegaconf import OmegaConf

    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)
    raise TypeError(f"unsupported config container {type(cfg).__name__}")


def _load(checkpoint) -> dict:
    if isinstance(checkpoint, dict):
        return checkpoint
    path = str(checkpoint)
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", mmap=True, weights_only=False)


def _saved_cfg(state: dict) -> dict:
    saved = _plain((state.get("hyper_parameters") or {}).get("cfg"))
    if not isinstance(saved, dict):
        raise RuntimeError("checkpoint has no saved config (hyper_parameters['cfg'])")
    return saved


def _model_field_mismatches(saved: dict, cfg) -> list[str]:
    """Model fields present in both configs whose values differ. Dropout rates, the attention
    kernel choice and file paths do not change a loaded model's outputs and are not compared."""
    caller = _plain(cfg).get("model", {})
    out = []
    for key, value in (saved.get("model") or {}).items():
        if key not in caller or "dropout" in key or key == "use_flash_attention" or key.endswith("_path"):
            continue
        if value != caller[key]:
            out.append(f"model.{key}: checkpoint {value!r}, config {caller[key]!r}")
    return out


def config_from_checkpoint(checkpoint):
    """GTFConfig holding a checkpoint's saved values.

    checkpoint: a path or a loaded checkpoint dict. The saved config is merged over the
    GTFConfig defaults. The saved data paths are the training machine's: override
    data.manifest_path, data.stats_path and data.uni3d_cache_path as needed.
    """
    from omegaconf import OmegaConf

    from gtf.config import GTFConfig

    saved = _saved_cfg(_load(checkpoint))
    return OmegaConf.merge(OmegaConf.structured(GTFConfig), saved)


def file_digest(path: str) -> str:
    """First 16 hex digits of the file's sha256 (the normalization statistics' identity in a checkpoint)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_lit_module_from_checkpoint(
    cfg,
    device: torch.device,
    *,
    mano_root: str,
):
    """Load a GTFLitModule for evaluation from cfg.eval.checkpoint_path.

    The model is built from cfg. An exported checkpoint (gtf.training.export_checkpoint) carries its
    normalization statistics; a training checkpoint reads them from cfg.data.stats_path, which must be
    the file it was trained with. Loading fails when a model field of the checkpoint's saved config
    disagrees with cfg, on a different statistics file, or when the state-dict keys or shapes differ.
    """
    from gtf.training.trainer import GTFLitModule

    ckpt_path = str(cfg.eval.checkpoint_path)
    if not ckpt_path:
        raise ValueError("eval.checkpoint_path is required")
    state = _load(ckpt_path)
    saved = _saved_cfg(state)
    diffs = _model_field_mismatches(saved, cfg)
    if diffs:
        raise RuntimeError(f"model config differs from the checkpoint {ckpt_path}:\n  " + "\n  ".join(diffs))
    packed_stats = state.get("normalization_stats")
    stats_digest = state.get("stats_content_digest")
    if packed_stats is None and stats_digest:
        current = file_digest(str(cfg.data.stats_path))
        if current != stats_digest:
            raise RuntimeError(f"normalization stats {cfg.data.stats_path} (sha256 {current}) are not the checkpoint's (sha256 {stats_digest})")

    # The checkpoint holds every weight, the frozen scene backbone included: skip its pretrained file.
    from omegaconf import OmegaConf

    build_cfg = OmegaConf.merge(cfg, {"model": {"litept_weights_path": ""}})
    lit = GTFLitModule(
        cfg=build_cfg,
        mano_root=mano_root,
        stats_path=str(cfg.data.stats_path),
        stats=packed_stats,
    )
    sd = state["state_dict"]
    model_sd = lit.state_dict()
    missing = sorted(set(model_sd) - set(sd))
    unexpected = sorted(set(sd) - set(model_sd))
    shapes = [f"{k}: checkpoint {tuple(v.shape)}, model {tuple(model_sd[k].shape)}" for k, v in sd.items() if k in model_sd and tuple(v.shape) != tuple(model_sd[k].shape)]
    if missing or unexpected or shapes:
        lines = [f"missing: {missing[:12]}"] if missing else []
        lines += [f"unexpected: {unexpected[:12]}"] if unexpected else []
        lines += shapes[:12]
        raise RuntimeError(f"checkpoint {ckpt_path} does not match the model:\n  " + "\n  ".join(lines))
    lit.load_state_dict(sd, strict=True)

    lit.eval()
    lit = lit.to(device)
    lit._move_stats_to_device()
    print(f"Model loaded: {sum(p.numel() for p in lit.model.parameters()):,} params")
    return lit
