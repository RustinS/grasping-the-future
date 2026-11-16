"""Energy score, inside rate and aggregation of the benchmark scorer on synthetic samples."""

import pickle

import numpy as np

from gtf.eval import benchmark


def test_cloud_energy_score_and_inside():
    y = np.zeros(3)
    p = np.array([[1.0, 0, 0], [-1.0, 0, 0]])  # two samples 1 m from the reference, 2 m apart
    err, pair, inside = benchmark._cloud(p, y)
    assert np.allclose(err, [1, 1]) and pair == 2.0
    assert err.mean() - 0.5 * pair == 0.0  # energy score
    assert inside == 1.0
    _, _, outside = benchmark._cloud(p, np.array([0.0, 5.0, 0.0]))
    assert outside == 0.0


def test_gap_runs_bounded():
    obs = np.array([1, 1, 0, 0, 0, 1, 0, 0], bool)
    pres = np.ones(8, bool)
    runs = benchmark._gap_runs(obs, pres)
    assert runs == [(3, True), (7, False)]  # middle of frames 2-4 (bounded), open run at the end


def _record(source, offset, n=4, T=10):
    pred = np.zeros((n, T, 3), np.float32)
    pred[:, :, 0] = offset
    gt = np.zeros((T, 3), np.float32)
    obs = np.zeros((T, 5), bool)
    obs[:2] = True
    rec = {"source": source, "segment_id": source, "window_start": 0, "obs": obs, "presence": np.ones((T, 5), bool),
           "valid": np.ones(T, bool), "joint_idx": 0}
    for ent in ("obj", "hand_left", "hand_right"):
        rec[ent] = {"pred": pred, "gt": gt, "excursion_mm": 100.0}
    return rec


def test_table2_aggregation(tmp_path):
    # Motion capture: two sources (0.1 m and 0.3 m error, source macro 0.2 m); internet video: 0.4 m. Combined 0.3 m.
    for pop, recs in {"mocap": [_record("a", 0.1), _record("b", 0.3), _record("b", 0.3)], "action100m": [_record("action100m", 0.4)]}.items():
        (tmp_path / pop).mkdir()
        pickle.dump(recs, open(tmp_path / pop / "future_prediction.pkl", "wb"))
    t2 = benchmark.table2(tmp_path)
    assert abs(t2["obj/FDE/single"]["combined"] - 300.0) < 1e-3
    assert abs(t2["obj/ADE/es"]["combined"] - 300.0) < 1e-3  # identical samples: energy score equals the error
