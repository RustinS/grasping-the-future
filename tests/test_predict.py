"""gtf.predict returns samples in the segment's camera frame: its conversion inverts the model's gravity alignment."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from gtf.model.repr.canonicalize import compute_gravity_alignment, gravity_align
from gtf.predict import to_camera_frame


def test_to_camera_frame_inverts_gravity_align(mano_dir):
    rng = np.random.default_rng(0)
    N, T = 2, 4
    betas = {side: rng.normal(scale=1.0, size=10).astype(np.float32) for side in ("left", "right")}
    camera = {
        "obj_rotation": torch.tensor(rng.normal(scale=0.8, size=(N, T, 3)), dtype=torch.float32),
        "obj_translation": torch.tensor(rng.normal(scale=0.2, size=(N, T, 3)) + [0, 0, 0.6], dtype=torch.float32),
    }
    for side in ("left", "right"):
        camera[f"hand_{side}_orient"] = torch.tensor(rng.normal(scale=0.8, size=(N, T, 3)), dtype=torch.float32)
        camera[f"hand_{side}_pose"] = torch.tensor(rng.normal(scale=0.2, size=(N, T, 45)), dtype=torch.float32)
        camera[f"hand_{side}_cam_t"] = torch.tensor(rng.normal(scale=0.2, size=(N, T, 3)) + [0, 0, 0.6], dtype=torch.float32)
    gravity = torch.tensor([0.2, 0.9, -0.3], dtype=torch.float64)
    gravity = gravity / gravity.norm()

    # The model's frame, as the dataset builds it (one sample at a time).
    model = {k: [] for k in camera}
    for n in range(N):
        sample = {k: v[n] for k, v in camera.items()}
        for side in ("left", "right"):
            sample[f"hand_{side}_betas"] = torch.from_numpy(betas[side])
            sample[f"hand_{side}_valid"] = torch.ones(T, dtype=torch.bool)
        aligned = gravity_align(sample, gravity.float())
        for k in camera:
            model[k].append(aligned[k])
    model = {k: torch.stack(v) for k, v in model.items()}

    back = to_camera_frame(model, compute_gravity_alignment(gravity), betas, torch.ones(T, 3, dtype=torch.bool))
    for k, v in camera.items():
        if k.endswith("orient") or k == "obj_rotation":  # compare rotations, not axis-angle vectors
            a = Rotation.from_rotvec(v.reshape(-1, 3).numpy()).as_matrix()
            b = Rotation.from_rotvec(back[k].reshape(-1, 3)).as_matrix()
            assert np.abs(a - b).max() < 1e-4, k
        else:
            assert np.abs(v.numpy() - back[k]).max() < 1e-5, k
