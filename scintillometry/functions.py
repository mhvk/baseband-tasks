# Licensed under the GPLv3 - see LICENSE
import numpy as np

from .base import TaskBase


class FunctionTask(TaskBase):
    """Apply a user-supplied function to a stream.

    Parameters
    ----------
    ih : filehandle
        Source of data, or another task, from which samples are read.
    function : callable
        The function should take two arguments, which will be the
        Function task instance, with its offset at the sample for which
        the amplitude should be returned, and the frame data read from
        the underlying file or task.
    samples_per_frame : int, optional
        Number of samples which should be fed to the function in one go.
        If not given, by default the number from the underlying file.
    dtype : `~numpy.dtype`, optional
        Output dtype.  If not given, the dtype of the underlying file.
    """
    def __init__(self, ih, function, shape=None, sample_rate=None,
                 samples_per_frame=None, dtype=None):
        self._function = function
        if shape is None:
            shape = ih.shape
        if sample_rate is None:
            sample_rate = ih.sample_rate
        if samples_per_frame is None:
            samples_per_frame = ih.samples_per_frame
        else:
            nsamples = (shape[0] // samples_per_frame) * samples_per_frame
            shape = (nsamples,) + shape[1:]
        if dtype is None:
            dtype = ih.dtype
        super().__init__(ih, shape=shape, sample_rate=sample_rate,
                         samples_per_frame=samples_per_frame, dtype=dtype)

    def _read_frame(self, frame_index):
        # Read data from underlying filehandle.
        self.ih.seek(frame_index * self.samples_per_frame)
        data = self.ih.read(self.samples_per_frame)
        # Apply function to the data.  Note that the read() function
        # in base ensures that our offset pointer is correct.
        return self._function(self, data)
