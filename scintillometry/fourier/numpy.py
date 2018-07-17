# Licensed under the GPLv3 - see LICENSE

import numpy as np
import operator
from .base import FFTMakerBase, FFTBase


__all__ = ['NumpyFFTMaker']


class NumpyFFTMaker(FFTMakerBase):
    """FFT factory class utilizing `numpy.fft` functions.

    FFTs of real-valued time-domain data use `~numpy.fft.rfft` and its inverse.
    `~numpy.fft.rfft` performs a real-input transform on one dimension of the
    input, halving that dimension's length in the output.
    """

    def __call__(self, time_data=None, freq_data=None, axis=0,
                 ortho=False, sample_rate=None):
        """Set up FFT.

        Parameters
        ----------
        time_data : `~numpy.ndarray`, dict, or None
            Dummy array with dimensions and dtype of time-domain data.  Can
            alternatively give a dict with 'shape' and 'dtype' entries.
            If not given, it is derived from ``freq_data``.
        freq_data : `~numpy.ndarray`, dict, or None
            Dummy array with dimensions and dtype of frequency-domain data.
            Can alternatively give a dict with 'shape' and 'dtype' entries.
            If not given, it is derived from ``time_data``.  If both are given,
            they will be checked for consistency.
        axis : int, optional
            Axis to transform.  Default: 0.
        ortho : bool, optional
            Whether to use orthogonal normalization.  Default: `False`.
        sample_rate : float, `~astropy.units.Quantity`, or None, optional
            Sample rate, used to determine the FFT sample frequencies.  If
            `None`, a unitless rate of 1 is used.
        """
        # Set direction, axis and normalization.  If axis is None, set it to 0.
        axis = operator.index(axis)
        ortho = bool(ortho)

        # Store time and frequency-domain array shapes.
        data_format = self.get_data_format(time_data=time_data,
                                           freq_data=freq_data, axis=axis)

        # Prepare either fft or rfft functions.
        if data_format['time_dtype'].kind == 'c':

            def forward_fft(self, a):
                return np.fft.fft(a, axis=self.axis, norm=self._norm).astype(
                    self.data_format['freq_dtype'], copy=False)

            def inverse_fft(self, A):
                return np.fft.ifft(A, axis=self.axis, norm=self._norm).astype(
                    self.data_format['time_dtype'], copy=False)

        else:

            def forward_fft(self, a):
                return np.fft.rfft(a, axis=self.axis, norm=self._norm).astype(
                    self.data_format['freq_dtype'], copy=False)

            # irfft needs explicit length for odd-numbered outputs.
            def inverse_fft(self, A):
                return np.fft.irfft(
                    A, n=self.data_format['time_shape'][axis], axis=self.axis,
                    norm=self._norm).astype(self.data_format['time_dtype'],
                                            copy=False)

        # Declare NumpyFFT class, and populate values.
        class NumpyFFT(FFTBase):
            """Single pre-defined FFT based on `numpy.fft`.

            To use, first initialize an instance, and then call the instance
            to perform the transform.

            Parameters
            ----------
            direction : 'forward' or 'inverse', optional
                Direction of the FFT.
            """

            _data_format = data_format
            _axis = axis
            _ortho = ortho
            _norm = 'ortho' if ortho else None
            _sample_rate = sample_rate

            _forward_fft = forward_fft
            _inverse_fft = inverse_fft

            def __init__(self, direction='forward'):
                super().__init__(direction=direction)
                if self.direction == 'forward':
                    self._fft = self._forward_fft
                else:
                    self._fft = self._inverse_fft

        return NumpyFFT
