# Licensed under the GPLv3 - see LICENSE
"""Convolution tasks."""
import warnings

import numpy as np

from .base import BaseTaskBase
from .fourier import get_fft_maker


__all__ = ['Convolve', 'FFTConvolve']


class Convolve(BaseTaskBase):
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
        if samples_per_frame is None:
            # Calculate the number of samples that ensures >75% efficiency:
            # use 4 times power of two just above pad.
            samples_per_frame = 2 ** (int((np.ceil(np.log2(pad)))) + 2)
        elif pad >= samples_per_frame:
            raise ValueError("need more than {} samples per frame to be "
                             "able to convolve without wrapping."
                             .format(pad))
        elif pad > samples_per_frame / 2.:
            warnings.warn("convolution will be inefficient since of the "
                          "{} samples per frame given, {} will be lost due "
                          "to padding.".format(samples_per_frame, pad))

        # Subtract padding since that is what we actually produce per frame,
        samples_per_frame -= pad
        shape = (ih.shape[0] - pad,) + ih.sample_shape
        super().__init__(ih, shape=shape, samples_per_frame=samples_per_frame)
        self._pad_start = pad - offset
        self._pad_end = offset
        self._start_time += self._pad_start / ih.sample_rate
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

    def _read_frame(self, frame_index):
        # Read data from underlying filehandle.
        self.ih.seek(frame_index * self.samples_per_frame)
        data = self.ih.read(self.samples_per_frame +
                            self._pad_start + self._pad_end)
        return self.task(data)


class FFTConvolve(Convolve):
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
        super().__init__(ih, response=response, offset=offset,
                         samples_per_frame=samples_per_frame)
        # Initialize FFTs for fine channelization and the inverse.
        if FFT is None:
            FFT = get_fft_maker()

        raw_frame_shape = (samples_per_frame,) + ih.sample_shape
        self._fft = FFT(shape=raw_frame_shape, sample_rate=ih.sample_rate,
                        dtype=ih.dtype)
        self._ifft = self._fft.inverse()
        long_response = np.zeros(raw_frame_shape, ih.dtype)
        long_response[:self._response.shape[0]] = self._response
        self._ft_response = self._fft(long_response)

    def task(self, data):
        ft = self._fft(data)
        ft *= self._ft_response
        return self._ifft(ft)[self._pad_start:data.shape[0]-self._pad_end]
