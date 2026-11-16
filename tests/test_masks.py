"""Observation masks of the seven tasks."""

import torch
from omegaconf import OmegaConf

from gtf.config import GTFConfig
from gtf.eval.evaluate import _OBS_MASK_BUILDERS, _builder_kwargs

LEFT, RIGHT, OBJ = 0, 1, 2


def _masks():
    eval_cfg = OmegaConf.structured(GTFConfig).eval
    T = 50
    presence = torch.ones(1, T, 5, dtype=torch.bool)
    valid = torch.ones(1, T, dtype=torch.bool)
    return {task: build(presence, valid, **_builder_kwargs(task, eval_cfg))[0] for task, build in _OBS_MASK_BUILDERS.items()}


def test_seven_tasks():
    assert set(_OBS_MASK_BUILDERS) == {"future_prediction", "hand_to_object", "object_to_hand", "temporal_infilling",
                                       "goal_infilling", "goal_hand_only", "goal_object_only"}


def test_mask_semantics():
    m = _masks()
    fp = m["future_prediction"]
    assert fp[0].all() and not fp[-1].any()  # observed prefix, everything predicted after it
    # every task observes the first (present) frame fully; the given entity is observed throughout
    assert all(mask[0].all() for mask in m.values())
    assert m["hand_to_object"][:, LEFT].all() and m["hand_to_object"][:, RIGHT].all() and not m["hand_to_object"][1:, OBJ].any()
    assert m["object_to_hand"][:, OBJ].all() and not m["object_to_hand"][1:, LEFT].any()
    ti = m["temporal_infilling"]
    assert ti[0].all() and ti[-1].all() and not ti[len(ti) // 2].any()  # both ends observed, a gap in the middle
    assert m["goal_infilling"][-1, :3].all()
    assert m["goal_hand_only"][-1, LEFT] and not m["goal_hand_only"][-1, OBJ]
    assert m["goal_object_only"][-1, OBJ] and not m["goal_object_only"][-1, LEFT]
