from collections import OrderedDict

import numpy as np


__all__ = (
    'INTEGERS', 'FLOATS', 'NUMERICS',
    'INTEGERS_RANGES', 'FLOATS_RANGES', 'NUMERICS_RANGES',
)


# Data types
INTEGERS = {int, np.int8, np.int16, np.int32, np.int64, }
FLOATS = {float, np.float16, np.float32, np.float64, np.float128, }

NUMERICS = set().union(INTEGERS, FLOATS)


# Data types with values ranges
INTEGERS_RANGES = OrderedDict([
    (np.int64, (np.iinfo(np.int64).min, np.iinfo(np.int64).max)),
    (np.int32, (np.iinfo(np.int32).min, np.iinfo(np.int32).max)),
    (np.int16, (np.iinfo(np.int16).min, np.iinfo(np.int16).max)),
    (np.int8, (np.iinfo(np.int8).min, np.iinfo(np.int8).max)),
])

FLOATS_RANGES = OrderedDict([
    (np.float128, (np.finfo(np.float128).min, np.finfo(np.float128).max)),
    (np.float64, (np.finfo(np.float64).min, np.finfo(np.float64).max)),
    (np.float32, (np.finfo(np.float32).min, np.finfo(np.float32).max)),
    (np.float16, (np.finfo(np.float16).min, np.finfo(np.float16).max)),
])

NUMERICS_RANGES = INTEGERS_RANGES.copy()
NUMERICS_RANGES.update(FLOATS_RANGES)
