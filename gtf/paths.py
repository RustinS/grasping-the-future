"""Repository locations. Kept free of heavy imports (no torch): the model code imports it.

REPO_ROOT is the folder that holds the ``gtf`` package (the repository root of a source checkout or an editable install).
Data paths given in configs and on the command line ("datasets/...", "checkpoints/...") are relative to the current
directory, which is expected to be REPO_ROOT; the defaults below are absolute.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY_DIR = REPO_ROOT / "third-party"

# The MANO model files: MANO_RIGHT.pkl (required) and MANO_LEFT.pkl (optional). Set MANO_DIR to use another folder.
MANO_DIR = os.path.realpath(os.environ.get("MANO_DIR") or THIRD_PARTY_DIR / "mano")


def mano_model_path(side="right"):
    """Path of the MANO model file for one side (MANO_RIGHT.pkl or MANO_LEFT.pkl) in MANO_DIR."""
    return os.path.join(MANO_DIR, f"MANO_{side.upper()}.pkl")
