import numpy as np

from src.utils import pad_array, get_positions


def test_pad_array():
    assert len(pad_array([], 1000)) == 0
    assert np.all(pad_array([1, 2], 1) == np.array([1, 2]))
    assert np.all(pad_array([1, 2], 3) == np.array([1, 2, 2]))
    assert np.all(pad_array(np.array([1, 2]), 3) == np.array([1, 2, 2]))


def test_get_positions():
    assert np.all(get_positions([1, 1, 2, 2, 2, 3]) == np.array([1, 2, 1, 2, 3, 1]))
