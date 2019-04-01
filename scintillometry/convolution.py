# Licensed under the GPLv3 - see LICENSE
"""Convolution tasks."""
import numpy as np
from astropy.utils import lazyproperty

from .base import PaddedTaskBase, check_broadcast_to
from .fourier import get_fft_maker


__all__ = ['ConvolveSamples', 'Convolve']


class ConvolveSamples(PaddedTaskBase):
    """Convolve a time stream with a response, in the time domain.

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
    Convolve : convolution in the Fourier domain (usually faster)
    """

    def __init__(self, ih, response, offset=0, samples_per_frame=None):
        if response.ndim == 1 and ih.ndim > 1:
            response = response.reshape(response.shape[:1] +
                                        (1,) * (ih.ndim - 1))
        else:
            check_broadcast_to(response, response.shape[:1] + ih.sample_shape)

        pad = response.shape[0] - 1
        super().__init__(ih, pad_start=pad-offset, pad_end=offset,
                         samples_per_frame=samples_per_frame)
        self._response = response

    def task(self, data):
        result = np.empty((self.samples_per_frame,) + self.sample_shape,
                          dtype=self.dtype)
        response = np.broadcast_to(self._response,
                                   self._response.shape[:1]+self.sample_shape)
        for index in np.ndindex(self.sample_shape):
            index = (slice(None),) + index
            result[index] = np.convolve(data[index], response[index],
                                        mode='valid')
        return result


class Convolve(ConvolveSamples):
    """Convolve a time stream with a response, in the Fourier domain.

    The convolution is done via multiplication in the Fourier domain, which
    is faster than direct convolution for all but very simple responses.

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
    ConvolveSamples : convolution in the time domain (for simple responses)
    """
    def __init__(self, ih, response, offset=0, samples_per_frame=None,
                 FFT=None):
        super().__init__(ih, response=response, offset=offset,
                         samples_per_frame=samples_per_frame)
        # Initialize FFTs for fine channelization and the inverse.
        if FFT is None:
            FFT = get_fft_maker()

        self._FFT = FFT

    @lazyproperty
    def _fft(self):
        return self._FFT(shape=(self._padded_samples_per_frame,) +
                         self.ih.sample_shape,
                         sample_rate=self.ih.sample_rate, dtype=self.ih.dtype)

    @lazyproperty
    def _ifft(self):
        return self._fft.inverse()

    @lazyproperty
    def _ft_response(self):
        long_response = np.zeros((self._padded_samples_per_frame,) +
                                 self._response.shape[1:], self.dtype)
        long_response[:self._response.shape[0]] = self._response
        fft = self._FFT(shape=long_response.shape, dtype=self.dtype)
        return fft(long_response)

    def task(self, data):
        ft = self._fft(data)
        ft *= self._ft_response
        result = self._ifft(ft)
        return result[self._pad_start + self._pad_end:]

    def close(self):
        super().close()
        # Clear the caches of the lazyproperties to release memory.
        del self._ft_response
        del self._fft
        del self._ifft
