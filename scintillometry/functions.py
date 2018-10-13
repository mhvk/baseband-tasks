# Licensed under the GPLv3 - see LICENSE
import numpy as np

from .base import TaskBase


__all__ = ['FunctionTask', 'ComplexFunctionTask', 'SquareTask']


class FunctionTask(TaskBase):
    """Apply a user-supplied function to a stream.

    The function will be fed data from the underlying stream.  If
    knowledge of the state of the stream is needed, use
    `ComplexFunctionTask`.

    Parameters
    ----------
    ih : filehandle
        Source of data, or another task, from which samples are read.
    function : callable
        The function should take a single argument, which will be the
        frame data read from the underlying file or task.
    shape : tuple, optional
        Overall shape of the stream, with first entry the total number
        of complete samples, and the remainder the sample shape.  By
        default, the shape of the underlying stream, possibly adjusted
        for a difference of sample rate.
    sample_rate : `~astropy.units.Quantity`, optional
        With units of a rate.  If not given, taken from the underlying
        stream.  Should be passed in if the function reduces or expands
        the number of elements.
    samples_per_frame : int, optional
        Number of samples which should be fed to the function in one go.
        If not given, by default the number from the underlying file,
        possibly adjusted for a difference in sample rate.
    dtype : `~numpy.dtype`, optional
        Output dtype.  If not given, the dtype of the underlying file.
    """
    def __init__(self, ih, function, shape=None, sample_rate=None,
                 samples_per_frame=None, dtype=None):
        self._function = function
        super().__init__(ih, shape=shape, sample_rate=sample_rate,
                         samples_per_frame=samples_per_frame, dtype=dtype)

    def function(self, data):
        return self._function(data)

    def _read_frame(self, frame_index):
        # Read data from underlying filehandle.
        self.ih.seek(frame_index * self._raw_samples_per_frame)
        data = self.ih.read(self._raw_samples_per_frame)
        # Apply function to the data.  Note that the read() function
        # in base ensures that our offset pointer is correct.
        return self.function(data)


class ComplexFunctionTask(FunctionTask):
    """Apply a user-supplied function to a stream.

    Like `~scintillometry.functions.FunctionTask`, but for a callable
    that requires knowledge of the state of the stream.

    Parameters
    ----------
    ih : filehandle
        Source of data, or another task, from which samples are read.
    function : callable
        The function should take two arguments, which will be the
        Function task instance, with its offset at the sample for which
        the amplitude should be returned, and the frame data read from
        the underlying file or task.
    shape : tuple, optional
        Overall shape of the stream, with first entry the total number
        of complete samples, and the remainder the sample shape.  By
        default, the shape of the underlying stream, possibly adjusted
        for a difference of sample rate.
    sample_rate : `~astropy.units.Quantity`, optional
        With units of a rate.  If not given, taken from the underlying
        stream.  Should be passed in if the function reduces or expands
        the number of elements.
    samples_per_frame : int, optional
        Number of samples which should be fed to the function in one go.
        If not given, by default the number from the underlying file,
        possibly adjusted for a difference in sample rate.
    dtype : `~numpy.dtype`, optional
        Output dtype.  If not given, the dtype of the underlying file.
    """
    def function(self, data):
        return self._function(self, data)


class SquareTask(FunctionTask):
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

        dtype = square(np.zeros(1, dtype=ih_dtype)).dtype

        super().__init__(ih, function=square, dtype=dtype)
