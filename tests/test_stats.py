import numpy as np

from dsmlt import stats


def test_trimean():
    data = np.array(range(101))
    true_trimean = (25 + 50 * 2 + 75) / 4
    assert stats.trimean(data) == true_trimean

    data = np.array(range(1, 101))
    true_trimean = (25.75 + 50.5 * 2 + 75.25) / 4
    assert stats.trimean(data) == true_trimean

    data = np.array(range(1, 100))
    true_trimean = (25.5 + 50 * 2 + 74.5) / 4
    assert stats.trimean(data) == true_trimean
