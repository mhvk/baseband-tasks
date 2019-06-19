"""Module for signal-processing of baseband signals."""

import numpy as np
import os
from scintillometry.base import TaskBase
try:
    import pyfftw.interfaces.numpy_fft as fftw
    _fftargs = {
        'threads': int(os.environ.get('OMP_NUM_THREADS', 2)),
        'planner_effort': 'FFTW_ESTIMATE',
        'overwrite_input': True
    }
    FFTPACK = fftw
except ImportError:
    _fftargs = {}
    FFTPACK = np.fft

__all__ = ['real_to_complex']


def real_to_complex(z):
    """
    Convert a real baseband signal to a complex baseband signal.
    This function computes the analytic representation of the input signal
    via a Hilbert transform, throwing away negative frequency components.
    Then, the signal is shifted in frequency domain by -B/2 where B is the
    bandwidth of the signal. Finally, the signal is decimated by a factor
    of 2, which results in the complex baseband representation of the input
    signal.
    Parameters
    ----------
    z : array_like
        Input array, must be real.
    axis : int, optional
        Axis over which to convert the signal. This will be the axis that
        represents time. If not given, the last axis is used.
    Returns
    -------
    out : complex ndarray
        The complex baseband representation of the input signal, transformed
        along the axis indicated by `axis`, or the last one if `axis` is not
        specified.
    Raises
    ------
    TypeError
        If `z` is complex-valued.
    IndexError
        if `axes` is larger than the last axis of `z`.
    See Also
    --------
    complex_to_real : The inverse of `real_to_complex`.
    Notes
    -----
    This function assumes the input signal is a causal signal.
    References
    ----------
    .. https://dsp.stackexchange.com/q/43278/17721
    """

    # Pick the correct axis to work on
    N = z.shape[0]

    # Hilbert transform
    z = FFTPACK.fft(z, axis=0, **_fftargs)

    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2
    z = FFTPACK.ifft(z * h, axis=0, **_fftargs)

    # Frequency shift signal by -B/2
    h = np.exp(-1j * np.pi / 2 * np.arange(N))
    z *= h

    z = z[::2]

    return z


class Real2Complex(TaskBase):
    def __init__(self, ih, samples_per_frame=None, dtype='c8'):
        if ih.complex_data:
            raise ValueError("Stream should be real.")

        super().__init__(ih, samples_per_frame=samples_per_frame,
                         sample_rate=ih.sample_rate / 2.,
                         dtype=dtype)

    def task(self, data):
        return real_to_complex(data)
