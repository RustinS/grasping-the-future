"""A rigid transform of the hand parameters moves the MANO hand rigidly, for both hands (the left-hand mirror rule)."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gtf.utils.canonical import mano_fk_verts
from gtf.utils.hand_rigid import transform_hand_rigid


@pytest.mark.parametrize("side", ["right", "left"])
def test_rigid_transform_matches_vertices(side, mano_dir):
    rng = np.random.default_rng(0)
    T = 3
    orient = rng.normal(scale=0.5, size=(T, 3)).astype(np.float32)
    pose = rng.normal(scale=0.2, size=(T, 45)).astype(np.float32)
    betas = rng.normal(scale=1.0, size=10).astype(np.float32)
    cam_t = rng.normal(scale=0.2, size=(T, 3)).astype(np.float32) + np.array([0, 0, 0.6], np.float32)
    R = Rotation.from_rotvec([0.3, -0.5, 0.2]).as_matrix().astype(np.float32)
    t = np.array([0.05, -0.02, 0.1], np.float32)
    verts = np.asarray(mano_fk_verts(orient, pose, betas, cam_t, side, mano_dir)).reshape(T, -1, 3)
    o2, t2 = transform_hand_rigid(orient, cam_t, np.ones(T, bool), side, R=R, t=t, betas=betas)
    moved = np.asarray(mano_fk_verts(o2, pose, betas, t2, side, mano_dir)).reshape(T, -1, 3)
    assert np.abs(moved - (verts @ R.T + t)).max() < 1e-4
