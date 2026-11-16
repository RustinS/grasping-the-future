"""An exported checkpoint carries its statistics and PCA basis, and unpack writes them back unchanged."""

import numpy as np
import pytest
import torch

from gtf.eval.checkpoint import file_digest
from gtf.training.export_checkpoint import PCA_FILE, STATS_FILE, export_checkpoint, unpack_checkpoint


def test_export_and_unpack_round_trip(tmp_path):
    stats = {"mean": {"obj_state": torch.arange(9.0)}, "std": {"obj_state": torch.ones(9)}, "stats_mode": "window_sampled"}
    pca = {"components": np.eye(4, 8, dtype=np.float32), "mean": np.zeros(8, dtype=np.float32)}
    torch.save(stats, tmp_path / "stats.pt")
    torch.save(pca, tmp_path / "pca.pt")
    state = {
        "state_dict": {"w": torch.randn(3, 3, dtype=torch.bfloat16)},
        "hyper_parameters": {"cfg": {}},
        "stats_content_digest": file_digest(str(tmp_path / "stats.pt")),
        "global_step": 7,
        "optimizer_states": [{"dropped": True}],
    }
    torch.save(state, tmp_path / "train.ckpt")

    out = export_checkpoint(str(tmp_path / "train.ckpt"), str(tmp_path / "model.ckpt"), str(tmp_path / "stats.pt"), str(tmp_path / "pca.pt"))
    assert set(out) == {"state_dict", "hyper_parameters", "normalization_stats", "scene_pca", "global_step"}
    assert out["state_dict"]["w"].dtype == torch.float32

    unpack_checkpoint(str(tmp_path / "model.ckpt"), str(tmp_path / "unpacked"))
    stats2 = torch.load(tmp_path / "unpacked" / STATS_FILE, weights_only=True)
    pca2 = torch.load(tmp_path / "unpacked" / PCA_FILE, weights_only=False)
    assert torch.equal(stats2["mean"]["obj_state"], stats["mean"]["obj_state"]) and stats2["stats_mode"] == "window_sampled"
    assert np.array_equal(pca2["components"], pca["components"])

    torch.save({"mean": {}}, tmp_path / "other_stats.pt")
    with pytest.raises(SystemExit):
        export_checkpoint(str(tmp_path / "train.ckpt"), str(tmp_path / "bad.ckpt"), str(tmp_path / "other_stats.pt"), str(tmp_path / "pca.pt"))
