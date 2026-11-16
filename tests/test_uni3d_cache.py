"""The Uni3D cache builder writes metadata that the cache validator accepts, on write and on load."""

import torch
from omegaconf import OmegaConf

from gtf.config import GTFConfig
from gtf.data.uni3d_cache import UNI3D_EMBEDDING_DIM, generate_padded_embeddings, load_uni3d_cache, write_uni3d_cache_atomic
from gtf.datasets.build_uni3d_cache import cache_metadata


def test_builder_metadata_round_trip(tmp_path):
    by_split = {"train": [f"tr{i}" for i in range(35)], "val": ["va0", "va1"], "test": ["te0"]}
    ids = [sid for split in ("train", "val", "test") for sid in by_split[split]]
    gen = torch.Generator().manual_seed(0)
    clouds = {sid: torch.randn(64, 6, generator=gen) for sid in ids}
    proj = torch.randn(6, UNI3D_EMBEDDING_DIM, generator=gen)

    def encode(batch):
        return batch.float().mean(1) @ proj.to(batch.device)

    bf16, fp32, points_digest, padding_rows = generate_padded_embeddings(ids, clouds.__getitem__, lambda b: encode(b).to(torch.bfloat16), encode, device="cpu")
    metadata = cache_metadata(OmegaConf.structured(GTFConfig), ids, by_split, bf16, fp32, points_digest, padding_rows, device_name="cpu")
    path = tmp_path / "uni3d_embeddings.pt"
    write_uni3d_cache_atomic(path, segment_ids=ids, embeddings_bf16=bf16, embeddings_fp32=fp32, metadata=metadata)
    _, loaded, _ = load_uni3d_cache(path, required_segment_ids=ids)
    assert loaded["segment_count"] == len(ids) and loaded["generation"]["variants"]["bf16"]["padding_rows"] == padding_rows["bf16"]
