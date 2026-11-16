import os

import pytest

from gtf.paths import MANO_DIR


@pytest.fixture
def mano_dir():
    if not os.path.isfile(os.path.join(MANO_DIR, "MANO_RIGHT.pkl")):
        pytest.skip("MANO model files not installed (third-party/mano/MANO_RIGHT.pkl)")
    return MANO_DIR
