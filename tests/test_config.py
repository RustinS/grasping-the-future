"""The configuration defaults are the paper's main model, and the shipped configs merge onto them."""

from pathlib import Path

from omegaconf import OmegaConf

from gtf.config import GTFConfig

ROOT = Path(__file__).resolve().parents[1]


def test_defaults_are_the_main_model():
    cfg = OmegaConf.structured(GTFConfig)
    assert (cfg.model.d_model, cfg.model.n_layers, cfg.model.n_heads) == (512, 12, 8)
    regimes = dict(cfg.masking.regimes)
    assert regimes == {"future_prediction": 0.19, "hand_to_object": 0.19, "object_to_hand": 0.19, "random_block": 0.05,
                       "goal_infilling": 0.19, "temporal_infilling": 0.19}
    assert abs(sum(regimes.values()) - 1.0) < 1e-9
    assert cfg.training.window_seconds == 5.0 and cfg.training.max_steps == 40000
    assert cfg.loss.lambda_energy_score == 1.0 and cfg.loss.es_dense_weight == 0.5


def test_shipped_configs_merge():
    cfg = OmegaConf.merge(OmegaConf.structured(GTFConfig), OmegaConf.load(ROOT / "configs/main.yaml"), OmegaConf.load(ROOT / "configs/benchmark.yaml"))
    assert cfg.trainer.devices * cfg.trainer.num_nodes == 16
    assert (cfg.eval.n_samples, cfg.eval.n_steps, cfg.eval.amp) == (20, 10, True)
