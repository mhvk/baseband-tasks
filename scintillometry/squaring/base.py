# Licensed under the GPLv3 - see LICENSE

import numpy as np

from ..base.base import TaskBase


__all__ = ['SquareTask']


class SquareTask(TaskBase):

    def __init__(self, ih):
        ih_dtype = np.dtype(ih.dtype)
        if ih_dtype.kind == 'f':
            dtype = ih_dtype
        else:
            dtype = np.dtype('f{0:d}'.format(ih_dtype.itemsize // 2))
        super().__init__(ih, ih.shape, ih.sample_rate, ih.samples_per_frame,
                         dtype)

    def _read_frame(self, frame_index):
        self.ih.seek(frame_index * self.samples_per_frame)
        data = self.ih.read(self.samples_per_frame)
        return np.real(data * np.conj(data))
