# Licensed under the GPLv3 - see LICENSE
"""Convolution tasks."""
import numpy as np

from .base import PaddedTaskBase
from .fourier import get_fft_maker


__all__ = ['Convolve', 'FFTConvolve']


class Convolve(PaddedTaskBase):
    """Convolve a time stream with a given filter.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream, with time as the first axis.
    response : `~numpy.ndarray`
        Response to convolce the time stream with.  If one-dimensional, assumed
        to apply to the sample axis of ``ih``.
    offset : int, optional
        Where samples should be considered to be taken from.  For the default
        of 0, a given sample has the same time as the convolution of the filter
        with all preceding samples.
    samples_per_frame : int, optional
        Number of samples which should be convolved in one go. The number of
        output convolved samples per frame will be smaller to avoid wrapping.
        If not given, the minimum power of 2 needed to get at least 75%
        efficiency.

    See Also
    --------
    FFTConvolve : convolution via a Fourier transform
    """

    def __init__(self, ih, response, offset=0, samples_per_frame=None):
        if response.ndim == 1 and ih.ndim > 1:
            response = response.reshape(response.shape[:1] +
                                        (1,) * (ih.ndim - 1))

        pad = response.shape[0] - 1
        super().__init__(ih, pad_start=pad-offset, pad_end=offset,
                         samples_per_frame=samples_per_frame)
        self._response = np.broadcast_to(response, (response.shape[0],) +
                                         self.sample_shape)

    def task(self, data):
        result = np.empty((self.samples_per_frame,) + self.sample_shape,
                          dtype=self.dtype)
        for index in np.ndindex(self.sample_shape):
            index = (slice(None),) + index
            result[index] = np.convolve(data[index], self._response[index],
                                        mode='valid')
        return result


class FFTConvolve(PaddedTaskBase):
    """Convolve a time stream with a given filter.

    The convolution is done via multiplication in the Fourier domain.
    For all but very simple responses, this faster than direct convolution.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream, with time as the first axis.
    response : `~numpy.ndarray`
        Response to convolce the time stream with.  If one-dimensional, assumed
        to apply to the sample axis of ``ih``.
    offset : int, optional
        Where samples should be considered to be taken from.  For the default
        of 0, a given sample has the same time as the convolution of the filter
        with all preceding samples.
    samples_per_frame : int, optional
        Number of samples which should be convolved in one go. The number of
        output convolved samples per frame will be smaller to avoid wrapping.
        If not given, the minimum power of 2 needed to get at least 75%
        efficiency.
    FFT : FFT maker or None, optional
        FFT maker.  Default: `None`, in which case the channelizer uses the
        default from `~scintillometry.fourier.base.get_fft_maker` (pyfftw if
        available, otherwise numpy).

    See Also
    --------
    Convolve : direct convolution.
    """
    def __init__(self, ih, response, offset=0, samples_per_frame=None,
                 FFT=None):
        if response.ndim == 1 and ih.ndim > 1:
            response = response.reshape(response.shape[:1] +
                                        (1,) * (ih.ndim - 1))

        pad = response.shape[0] - 1
        super().__init__(ih, pad_start=pad-offset, pad_end=offset,
                         samples_per_frame=samples_per_frame)
        # Initialize FFTs for fine channelization and the inverse.
        if FFT is None:
            FFT = get_fft_maker()

        self._fft = FFT(shape=(self._padded_samples_per_frame,) +
                        ih.sample_shape,
                        sample_rate=ih.sample_rate, dtype=ih.dtype)
        self._ifft = self._fft.inverse()
        # FFT response, ensuring we keep the possibly simpler shape.
        long_response = np.zeros((samples_per_frame,) +
                                 response.shape[1:], ih.dtype)
        long_response[:response.shape[0]] = response
        self._ft_response = FFT(shape=long_response.shape,
                                dtype=ih.dtype)(long_response)

    def task(self, data):
        ft = self._fft(data)
        ft *= self._ft_response
        result = self._ifft(ft)
        return result[self._pad_slice]
