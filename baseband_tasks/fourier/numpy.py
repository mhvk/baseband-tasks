# Licensed under the GPLv3 - see LICENSE
"""FFT maker and class using the `numpy.fft` routines."""

import numpy as np

from .base import FFTMakerBase, FFTBase


__all__ = ['NumpyFFTBase', 'NumpyFFTMaker']


class NumpyFFTBase(FFTBase):
    """Single pre-defined FFT based on `numpy.fft`.

    To use, initialize an instance, then call the instance to perform
    the transform.

    Parameters
    ----------
    direction : 'forward' or 'backward', optional
        Direction of the FFT.
    """

    def __init__(self, direction='forward'):
        super().__init__(direction=direction)
        time_complex = self._time_dtype.kind == 'c'
        if self.direction == 'forward':
            self._fft = self._cfft if time_complex else self._rfft
        else:
            self._fft = self._icfft if time_complex else self._irfft

    def _cfft(self, a):
        return np.fft.fft(a, axis=self.axis, norm=self._norm).astype(
            self._frequency_dtype, copy=False)

    def _icfft(self, a):
        return np.fft.ifft(a, axis=self.axis, norm=self._norm).astype(
            self._time_dtype, copy=False)

    def _rfft(self, a):
        return np.fft.rfft(a, axis=self.axis, norm=self._norm).astype(
            self._frequency_dtype, copy=False)

    # irfft needs explicit length for odd-numbered outputs.
    def _irfft(self, a):
        return np.fft.irfft(a, axis=self.axis, norm=self._norm,
                            n=self._time_shape[self.axis]).astype(
                                self._time_dtype, copy=False)


class NumpyFFTMaker(FFTMakerBase):
    """FFT factory class utilizing `numpy.fft` functions.

    FFTs of real-valued time-domain data use `~numpy.fft.rfft` and its inverse.
    `~numpy.fft.rfft` performs a real-input transform on one dimension of the
    input, halving that dimension's length in the output.

    `~baseband_tasks.fourier.numpy.NumpyFFTMaker.__call__` creates
    individual transforms.
    """
    # Since `numpy.fft` has no package-level options, no ``__init__`` is
    # explicitly defined.

    _FFTBase = NumpyFFTBase

    def __call__(self, shape, dtype, direction='forward', axis=0, ortho=False,
                 sample_rate=None):
        """Creates an FFT.

        Parameters
        ----------
        shape : tuple
            Shape of the time-domain data array, i.e. the input to the forward
            transform and the output of the inverse.
        dtype : str or `~numpy.dtype`
            Data type of the time-domain data array.  May pass either the
            name of the dtype or the `~numpy.dtype` object.
        direction : 'forward' or 'backward', optional
            Direction of the FFT.
        axis : int, optional
            Axis to transform.  Default: 0.
        ortho : bool, optional
            Whether to use orthogonal normalization.  Default: `False`.
        sample_rate : float, `~astropy.units.Quantity`, or None, optional
            Sample rate, used to determine the FFT sample frequencies.  If
            `None`, a unitless rate of 1 is used.

        Returns
        -------
        fft : ``NumpyFFT`` instance
            Single pre-defined FFT object.
        """
        return super().__call__(
            shape=shape, dtype=dtype, direction=direction,
            axis=axis, ortho=ortho, sample_rate=sample_rate,
            norm=('ortho' if ortho else None))
