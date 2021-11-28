"""Module for signal-processing of baseband signals."""

import numpy as np
from .base import TaskBase
from .fourier import fft_maker

__all__ = ['Real2Complex']


class Real2Complex(TaskBase):
    """
    Convert a real baseband signal to a complex baseband signal.

    This task computes the analytic representation of the input signal
    via a Hilbert transform, throwing away negative frequency components.
    Then, the signal is shifted in frequency domain by -B/2 where B is the
    bandwidth of the signal. Finally, the signal is decimated by a factor
    of 2, which results in the complex baseband representation of the input
    signal.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream, with time as the first axis.
    samples_per_frame : int, optional
        Number of complete output samples per frame (see Notes).
        Default: Half the number of samples per frame in the input stream.

    See Also
    --------
    baseband_tasks.fourier.fft_maker : to select the FFT package used.

    Raises
    ------
    ValueError
        If ``ih`` has complex data.

    Notes
    -----
    This function assumes the input signal is a causal signal.

    References
    ----------
    .. https://dsp.stackexchange.com/q/43278/17721
    """

    def __init__(self, ih, samples_per_frame=None):
        if ih.complex_data:
            raise ValueError("Stream should be real.")

        if samples_per_frame is None:
            assert ih.samples_per_frame % 2 == 0, \
                "need even number of input samples"
            samples_per_frame = ih.samples_per_frame // 2

        dtype = np.dtype('c{}'.format(ih.dtype.itemsize * 2))
        self._fft = fft_maker((samples_per_frame * 2, ) + ih.sample_shape,
                              dtype,
                              sample_rate=ih.sample_rate,
                              axis=0)

        self._ifft = self._fft.inverse()

        frequency = getattr(ih, 'frequency', None)
        sideband = getattr(ih, 'sideband', None)
        if frequency is not None:
            frequency = frequency + ih.sample_rate / 2 * sideband

        super().__init__(ih,
                         samples_per_frame=samples_per_frame,
                         sample_rate=ih.sample_rate / 2,
                         frequency=frequency, sideband=sideband,
                         dtype=dtype)

    def task(self, data):
        z = data.astype(self.dtype)
        N = z.shape[0]

        # Hilbert transform
        z = self._fft(z)

        h = np.zeros(N)
        if N % 2 == 0:
            h[0] = h[N // 2] = 1
            h[1:N // 2] = 2
        else:
            h[0] = 1
            h[1:(N + 1) // 2] = 2
        z = self._ifft(z * h)

        # Frequency shift signal by -B/2
        h = np.exp(-1j * np.pi / 2 * np.arange(N))
        z *= h

        z = z[::2]
        return z

    def _repr_item(self, key, default, value=None):
        if key == 'samples_per_frame' and default is None:
            default = self.ih.samples_per_frame // 2
        return super()._repr_item(key, default=default, value=value)
