# Licensed under the GPLv3 - see LICENSE

import numpy as np
import operator
from .base import FFTMakerBase, FFTBase


__all__ = ['NumpyFFTMaker']


class NumpyFFTBase(FFTBase):
    """Single pre-defined FFT based on `numpy.fft`.

    To use, initialize an instance, then call the instance to perform
    the transform.

    Parameters
    ----------
    direction : 'forward' or 'inverse', optional
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

    `~scintillometry.fourier.numpy.NumpyFFTMaker.__call__` creates
    individual transforms.
    """
    # Since `numpy.fft` has no package-level options, no ``__init__`` is
    # explicitly defined.

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
        direction : 'forward' or 'inverse', optional
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
        # Ensure arguments have proper types and values.
        shape = tuple(shape)
        dtype = np.dtype(dtype)
        axis = operator.index(axis)
        ortho = bool(ortho)

        # Store time and frequency-domain array shapes.
        frequency_shape, frequency_dtype = self.get_frequency_data_info(
            shape, dtype, axis=axis)

        # Declare NumpyFFT class, and populate values.
        class NumpyFFT(NumpyFFTBase):
            """Single pre-defined FFT based on `numpy.fft`.

            To use, initialize an instance, then call the instance to perform
            the transform.

            Parameters
            ----------
            direction : 'forward' or 'inverse', optional
                Direction of the FFT.
            """

            _time_shape = shape
            _time_dtype = dtype
            _frequency_shape = frequency_shape
            _frequency_dtype = frequency_dtype
            _axis = axis
            _ortho = ortho
            _norm = 'ortho' if ortho else None
            _sample_rate = sample_rate

        # Return NumpyFFT instance.
        return NumpyFFT(direction=direction)
