# Licensed under the GPLv3 - see LICENSE

import operator

from .functions import FunctionTaskBase
from .fourier import get_fft_maker


__all__ = ['ChannelizeTask']


class ChannelizeTask(FunctionTaskBase):
    """Basic channelizer.

    Divides input into blocks of ``n`` time samples, Fourier transforming each
    block.  The output sample shape is ``(channel,) + ih.sample_shape``.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream, with time as the first axis.
    n : int
        Number of input samples to channelize.  For complex input, output will
        have ``n`` channels; for real input, it will have ``n // 2 + 1``.
    samples_per_frame : int, optional
        Number of complete output samples per frame (see Notes).  Default: 1.
    FFT : FFT maker or None, optional
        FFT maker.  Default: `None`, in which case the channelizer uses
        `~scintillometry.fourier.numpy.NumpyFFTMaker`.

    Notes
    -----
    Instances initialize an FFT that acts upon axis 1 of an input with shape::

        (samples_per_frame, n) + ih.sample_shape

    Setting ``samples_per_frame`` to a number larger than 1 results in the FFT
    performing channelization on multiple blocks per call.  Depending on the
    backend used, this may speed up sequential channelization, though for tests
    using `numpy.fft` the performance improvement seems to be negligible.
    """

    def __init__(self, ih, n, samples_per_frame=1, FFT=None):

        n = operator.index(n)
        samples_per_frame = operator.index(samples_per_frame)
        sample_rate = ih.sample_rate / n
        nsample = samples_per_frame * (ih.shape[0] // n // samples_per_frame)
        assert nsample > 0, "not enough samples to fill one frame of data!"

        # Initialize channelizer.
        if FFT is None:
            FFT = get_fft_maker('numpy')

        self._fft = FFT((samples_per_frame, n) + ih.sample_shape,
                        ih.dtype, axis=1)

        super().__init__(ih, shape=(nsample,) + self._fft.freq_shape[1:],
                         sample_rate=sample_rate,
                         samples_per_frame=samples_per_frame,
                         dtype=self._fft.freq_dtype)

    def function(self, data):
        return self._fft(data.reshape(self._fft.time_shape))
