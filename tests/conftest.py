import numpy as np
import pytest


@pytest.fixture
def sample_distances():
    # 3 samples, 2 neighbors each
    return np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]], dtype=np.float64)


@pytest.fixture
def sample_indices():
    # 3 samples, 2 neighbors each
    return np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)
