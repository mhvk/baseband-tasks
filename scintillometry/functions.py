# Licensed under the GPLv3 - see LICENSE
import numpy as np

from .base import TaskBase


__all__ = ['Square']


class Square(TaskBase):
    """Converts samples to intensities by squaring.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream.
    """

    def __init__(self, ih):
        ih_dtype = np.dtype(ih.dtype)
        if ih_dtype.kind != 'c':
            square = np.square
        else:
            def square(x):
                return np.square(x.real) + np.square(x.imag)

        self.task = square
        dtype = self.task(np.zeros(1, dtype=ih_dtype)).dtype
        super().__init__(ih, dtype=dtype)
