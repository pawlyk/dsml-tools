from functools import reduce
from operator import mul

import pytest
import numpy as np

from dsmlt import utils


def test_random_size():
    size = utils.random_size(12)
    assert len(size) == 12

    size = utils.random_size(1)
    assert len(size) == 1

    size = utils.random_size(0)
    assert len(size) == 0

    size = utils.random_size()
    assert 0 <= len(size) < 10


def test_random_narray():
    # test dtype int
    size = utils.random_size(3, low=1, high=10)
    data = utils.random_narray(size)

    assert data.shape == size
    assert data.size == reduce(mul, size, 1)
    assert data.dtype.type == np.int64

    low = 10
    high = 100
    data = utils.random_narray(size, low=low, high=high)

    assert low <= data.min() < high
    assert 1 <= data.max() < high

    types = zip(
        [int, np.int, np.int_, np.intc, np.intp,
         np.int0, np.int8, np.int16, np.int32, np.int64],
        [np.int64, np.int64, np.int64, np.int32, np.int64,
         np.int64, np.int8, np.int16, np.int32, np.int64]
    )
    for type_, type_expected in types:
        data = utils.random_narray(size, dtype=type_)
        assert data.dtype.type == type_expected

    # test asytype int
    size = utils.random_size(3, low=1, high=10)
    data = utils.random_narray(size, dtype=np.int8, astype=np.int64)
    assert data.dtype.type == np.int64

    data = utils.random_narray(size, dtype=np.int64, astype=np.int8)
    assert data.dtype.type == np.int8

    # test dtype float
    size = utils.random_size(3, low=1, high=10)
    data = utils.random_narray(size, dtype=np.float)

    assert data.shape == size
    assert data.size == reduce(mul, size, 1)
    assert data.dtype.type == np.float64

    low = 10
    high = 100
    data = utils.random_narray(size, dtype=np.float, low=low, high=high)

    assert low <= data.min() < high
    assert 1 <= data.max() < high

    types = zip(
        [float, np.float, np.float_,
         np.float16, np.float32, np.float64, np.float128],
        [np.float64, np.float64, np.float64,
         np.float16, np.float32, np.float64, np.float128]
    )
    for type_, type_expected in types:
        data = utils.random_narray(size, dtype=type_)
        assert data.dtype.type == type_expected

    # test asytype float
    size = utils.random_size(3, low=1, high=10)
    data = utils.random_narray(size, dtype=np.float, astype=np.float16)
    assert data.dtype.type == np.float16

    data = utils.random_narray(size, dtype=np.float16, astype=np.float128)
    assert data.dtype.type == np.float128

    # test wrong type
    with pytest.raises(AttributeError) as exc:
        utils.random_narray((1, 2, 3, ), dtype=list)
    assert str(exc.value) == \
        "Passed invalid value of `astype` - <class 'list'>."

    with pytest.raises(AttributeError):
        utils.random_narray((1, 2, 3, ), dtype='wrong type')
    assert str(exc.value) == \
        "Passed invalid value of `astype` - <class 'list'>."

    # test wrong astype
    with pytest.raises(AttributeError):
        utils.random_narray((1, 2, 3, ), astype=list)
    assert str(exc.value) == \
        "Passed invalid value of `astype` - <class 'list'>."

    with pytest.raises(AttributeError):
        utils.random_narray((1, 2, 3, ), astype='wrong type')
    assert str(exc.value) == \
        "Passed invalid value of `astype` - <class 'list'>."
