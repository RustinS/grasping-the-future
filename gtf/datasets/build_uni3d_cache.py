#!/usr/bin/env python3
"""Build the frozen Uni3D object-embedding cache of a manifest.

Every scene-complete segment (all splits) gets one deterministic area-weighted surface
cloud of its object mesh (seeded by crc32(segment_id)), encoded by the frozen Uni3D
encoder twice: bf16 autocast at batch 32 (the training forward) and fp32 at batch 16 (the
evaluation forward). Partial batches are padded by repeating the last cloud, because the
encoder's numerics depend on the batch shape. The cache is written to
``data.uni3d_cache_path`` of the config.

Usage:
    python -m gtf.datasets.build_uni3d_cache --config CONFIG.yaml [--config ...] [--weights PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from gtf.paths import REPO_ROOT
from gtf.model.arch.conditioning import UNI3D_ARGS, _load_uni3d, compute_vertex_normals
from gtf.config import GTFConfig
from gtf.data.compute_stats import get_split
from gtf.data.dataset import _deterministic_obj_points, _entry_has_safe_scene_features
from gtf.data.uni3d_cache import (
    UNI3D_CACHE_FORMAT_VERSION,
    UNI3D_EMBEDDING_DIM,
    UNI3D_EVAL_BATCH_SIZE,
    UNI3D_GENERATION_BATCH_SIZE,
    generate_padded_embeddings,
    ordered_segment_ids_digest,
    population_digest,
    tensor_digest,
    uni3d_cache_config_digest,
    uni3d_cache_data_contract,
    write_uni3d_cache_atomic,
)
from gtf.model.repr.mesh import sanitize_mesh_faces

ROOT = REPO_ROOT

DEFAULT_WEIGHTS = "third-party/uni3d/weights/modelzoo/uni3d-g/model.pt"


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def cache_population(entries: list[dict]) -> tuple[list[dict], dict[str, list[str]]]:
    """Validated, scene-complete segments of every split, in manifest order."""
    selected: list[dict] = []
    by_split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    seen: set[str] = set()
    for entry in entries:
        if not entry.get("validated", True):
            continue
        split = str(get_split(entry))
        if split not in by_split or not _entry_has_safe_scene_features(entry):
            continue
        sid = str(entry["segment_id"])
        if sid in seen:
            raise ValueError(f"duplicate segment_id {sid!r} in the manifest")
        seen.add(sid)
        selected.append(entry)
        by_split[split].append(sid)
    return selected, by_split


def load_points(entry: dict) -> torch.Tensor:
    """The segment's Uni3D input cloud (N, 6): area-weighted surface points + normals."""
    with np.load(_resolve(str(entry["canonical_path"])), allow_pickle=False) as npz:
        verts_raw = torch.from_numpy(npz["obj_canonical_verts"].copy())
        faces_raw = torch.from_numpy(npz["obj_faces"].copy())
    faces = sanitize_mesh_faces(verts_raw, faces_raw, context=f"{entry['segment_id']} Uni3D object mesh")
    verts = verts_raw.float()
    return _deterministic_obj_points(
        {
            "segment_id": str(entry["segment_id"]),
            "obj_canonical_verts": verts,
            "obj_faces": faces,
            "obj_vertex_normals": compute_vertex_normals(verts, faces),
        }
    )


def cache_metadata(cfg, ids, by_split, embeddings_bf16, embeddings_fp32, points_digest, padding_rows, *, device_name: str) -> dict:
    """The cache's metadata, in the form gtf.data.uni3d_cache validates on write and on load."""
    contract = uni3d_cache_data_contract(cfg, uni3d_args=UNI3D_ARGS)
    return {
        "format_version": UNI3D_CACHE_FORMAT_VERSION,
        "population_scope": "runtime_filtered_train+val+test",
        "segment_count": len(ids),
        "split_counts": {key: len(value) for key, value in by_split.items()},
        "segment_ids_digest": ordered_segment_ids_digest(ids),
        "population_digest": population_digest(ids),
        "config_digest": uni3d_cache_config_digest(contract),
        "data_contract": contract,
        "generation": {
            "matmul_precision": "high",
            "encoder_mode": "eval",
            "variants": {
                variant: {
                    "batch_size": batch_size,
                    "autocast": autocast,
                    "output_dtype": dtype,
                    "padding": "repeat_last_to_full_batch_then_slice",
                    "padding_rows": padding_rows[variant],
                }
                for variant, batch_size, autocast, dtype in (
                    ("bf16", UNI3D_GENERATION_BATCH_SIZE, "torch.bfloat16", "torch.bfloat16"),
                    ("fp32", UNI3D_EVAL_BATCH_SIZE, "disabled", "torch.float32"),
                )
            },
            "device": device_name,
            # A plain str: TorchVersion would not load under weights_only=True.
            "torch_version": str(torch.__version__),
        },
        "obj_points_digest": points_digest,
        "embedding_bf16_digest": tensor_digest(embeddings_bf16),
        "embedding_fp32_digest": tensor_digest(embeddings_fp32),
        "embedding_dim": UNI3D_EMBEDDING_DIM,
        "embedding_bf16_dtype": str(embeddings_bf16.dtype),
        "embedding_fp32_dtype": str(embeddings_fp32.dtype),
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m gtf.datasets.build_uni3d_cache", description=__doc__.splitlines()[0])
    parser.add_argument("--config", action="append", default=None, help="YAML config file; repeat to merge left-to-right")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Uni3D-g checkpoint")
    parser.add_argument("--output", default=None, help="Output path (default: data.uni3d_cache_path of the config)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    cfg = OmegaConf.structured(GTFConfig)
    for path in args.config or []:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
    manifest = _resolve(str(cfg.data.manifest_path))
    output = _resolve(args.output or str(cfg.data.uni3d_cache_path))
    raw = json.loads(manifest.read_text())
    population, by_split = cache_population(raw["entries"] if isinstance(raw, dict) else raw)
    if not population:
        raise SystemExit(f"no scene-complete segments in {manifest}")
    ids = [str(entry["segment_id"]) for entry in population]
    entry_by_id = {str(entry["segment_id"]): entry for entry in population}
    print(f"Uni3D cache population: {len(ids)} segments (train={len(by_split['train'])}, val={len(by_split['val'])}, test={len(by_split['test'])})")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    uni3d = _load_uni3d(str(_resolve(args.weights))).to(device)
    uni3d.eval()

    def _encode_bf16(batch: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return uni3d.encode_pc(batch)

    def _encode_fp32(batch: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", enabled=False):
            return uni3d.encode_pc(batch.float())

    def _progress(done: int, total: int) -> None:
        if done == total or done % 2048 < UNI3D_GENERATION_BATCH_SIZE:
            print(f"  encoded {done}/{total}", flush=True)

    embeddings_bf16, embeddings_fp32, points_digest, padding_rows = generate_padded_embeddings(
        ids,
        lambda sid: load_points(entry_by_id[sid]),
        _encode_bf16,
        _encode_fp32,
        device=device,
        batch_size=UNI3D_GENERATION_BATCH_SIZE,
        progress=_progress,
    )

    metadata = cache_metadata(cfg, ids, by_split, embeddings_bf16, embeddings_fp32, points_digest, padding_rows, device_name=torch.cuda.get_device_name(device))
    file_digest = write_uni3d_cache_atomic(
        output,
        segment_ids=ids,
        embeddings_bf16=embeddings_bf16,
        embeddings_fp32=embeddings_fp32,
        metadata=metadata,
    )
    print(f"Uni3D cache written: {output} rows={len(ids)} file_sha256={file_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
