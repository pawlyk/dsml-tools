import numpy as np


__all__ = ('INTEGERS', 'FLOATS', )


INTEGERS = {int, np.int8, np.int16, np.int32, np.int64, }
FLOATS = {float, np.float16, np.float32, np.float64, np.float128, }
