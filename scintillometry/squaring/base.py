# Licensed under the GPLv3 - see LICENSE

import numpy as np

from ..base.base import TaskBase


__all__ = ['SquareTask']


class SquareTask(TaskBase):
    """Converts complex samples to intensities.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream.
    """

    def __init__(self, ih):
        ih_dtype = np.dtype(ih.dtype)

        if ih_dtype.kind == 'f':
            dtype = ih_dtype
            self._square = np.square
        else:
            dtype = np.dtype('f{0:d}'.format(ih_dtype.itemsize // 2))

            def abs2(x):
                return np.square(x.real) + np.square(x.imag)

            self._square = abs2

        super().__init__(ih, ih.shape, ih.sample_rate, ih.samples_per_frame,
                         dtype)

    def _read_frame(self, frame_index):
        self.ih.seek(frame_index * self._samples_per_frame)
        data = self.ih.read(self._samples_per_frame)
        return self._square(data)
