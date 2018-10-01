# Licensed under the GPLv3 - see LICENSE
import numpy as np

from .base import TaskBase


class FunctionTask(TaskBase):
    """Modulation with a user-provided profile.

    Parameters
    ----------
    ih : filehandle
        Source of data, or another task, from which samples are read.
    function : callable
        The function should take two arguments, which will be the
        Modulator task instance, with its offset at the sample for which
        the amplitude should be returned, and the frame data read from
        the underlying file or task.
    samples_per_frame : int, optional
        Number of samples which should be fed to the function in one go.
        If not given, by default the number from the underlying file.
    dtype : `~numpy.dtype`, optional
        Output dtype.  If not given, the dtype of the underlying file.
    """
    def __init__(self, ih, function, samples_per_frame=None, dtype=None):
        self._function = function
        if samples_per_frame is None:
            samples_per_frame = ih.samples_per_frame
        if dtype is None:
            dtype = ih.dtype
        super(FunctionTask, self).__init__(
            ih, shape=ih.shape, sample_rate=ih.sample_rate,
            samples_per_frame=samples_per_frame, dtype=dtype)

    def function(self, data):
        return self._function(self, data)

    def _read_frame(self, frame_index):
        self.ih.seek(frame_index * self.samples_per_frame)
        data = self.ih.read(self.samples_per_frame)
        self.seek(frame_index * self.samples_per_frame)
        return self._function(self, data)
