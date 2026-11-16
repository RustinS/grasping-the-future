"""The two extraction chains."""

from gtf.extraction import chains


def test_chains():
    assert set(chains.CHAINS) == {"production", "upgraded"}
    prod, up = chains.get("production"), chains.get("upgraded")
    assert prod.refine_bias and not prod.smooth and not prod.hand_masks and dict(prod.optimizer) == {}
    assert up.smooth and up.hand_masks and not up.refine_bias
    assert up.optimizer["hand_depth_source"] == "size_scene"
