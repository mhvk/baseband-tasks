# Licensed under the GPLv3 - see LICENSE
import numpy as np
from astropy.utils import lazyproperty


from .base import PaddedTaskBase
from .channelize import Channelize, Dechannelize


__all__ = ['sinc_hamming', 'PolyphaseFilterBankSamples',
           'PolyphaseFilterBank', 'InversePolyphaseFilterBank']


def sinc_hamming(n_tap, n_sample, sinc_scale=1.):
    r"""Construct a sinc-hamming polyphase filter.

    The sinc-hamming filter is defined by

    .. math::  F(n_{\rm tap}, n_{\rm samle}, s) &=
               \left(\frac{\sin(\pi x)}{\pi x}\right)
               \left(0.54 - 0.46\cos\left(\frac{2\pi k}{N-1}\right)\right),\\
               {\rm with~~}
               x &= n_{\rm tap} s \left(\frac{k}{N} - 0.5\right),\\
               N &= n_{\rm tap} n_{\rm sample}, \quad 0 \leq k \leq N-1.

    Parameters
    ----------
    n_tap : int
        Number of taps of the polyphase filter.
    n_sample : int
        Number of samples to pass on to the FFT stage.
    sinc_scale : float
        Possible scaling for the sinc factor, to widen or narrow the channels.

    Examples
    --------
    Construct the CHIME and GUPPI baseband polyphase filter responses::

    >>> from baseband_tasks.pfb import sinc_hamming
    >>> chime_ppf = sinc_hamming(4, 2048)
    >>> guppi_ppf = sinc_hamming(12, 64, sinc_scale=0.95)
    """
    n = n_tap * n_sample
    x = n_tap * sinc_scale * np.linspace(-0.5, 0.5, n, endpoint=False)
    return (np.sinc(x) * np.hamming(n)).reshape(n_tap, n_sample)


class PolyphaseFilterBankSamples(Channelize):
    """Channelize using a polyphase filter bank, in the time domain.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream, with time as the first axis.
    response : `~numpy.ndarray`
        Polyphase filter.  The first dimension is taken to be the
        number of taps, and the second the number of channels.
    samples_per_frame : int, optional
        Number of complete output samples per frame.  Default: inferred from
        padding, ensuring an efficiency of at least 75%.
    frequency : `~astropy.units.Quantity`, optional
        Frequencies for each channel in ``ih`` (channelized frequencies will
        be calculated).  Default: taken from ``ih`` (if available).
    sideband : array, optional
        Whether frequencies in ``ih`` are upper (+1) or lower (-1) sideband.
        Default: taken from ``ih`` (if available).

    See Also
    --------
    PolyphaseFilterBank : apply filter in the Fourier domain (usually faster)
    """
    def __init__(self, ih, response, samples_per_frame=None,
                 frequency=None, sideband=None):
        n_tap, n = response.shape
        pad = (n_tap - 1) * n
        if samples_per_frame is not None:
            samples_per_frame = samples_per_frame * n
        assert pad % 2 == 0
        # Note: cannot easily use Convolve since this doesn't convolve
        # every sample, but rather each n_chan samples.  In principle,
        # it could be done via a ReshapeSamples followed by a Convolve.
        self.padded = PaddedTaskBase(ih, pad_start=pad//2, pad_end=pad//2,
                                     samples_per_frame=samples_per_frame)
        self.padded.task = self.ppf
        self._response = response
        super().__init__(self.padded, n, self.padded.samples_per_frame // n,
                         frequency=frequency, sideband=sideband)
        self._reshape = ((self.padded._ih_samples_per_frame // n, n)
                         + self.ih.sample_shape)

    def ppf(self, data):
        """Apply the PolyPhase Filter, in the time domain."""
        data = data.reshape(self._reshape)
        result = np.empty((self.samples_per_frame,) + data.shape[1:],
                          data.dtype)
        # TODO: use stride tricks to do this in one go.
        n_tap = len(self._response)
        for i in range(data.shape[0] + 1 - n_tap):
            result[i] = (data[i:i+n_tap] * self._response).sum(0)
        return result.reshape((-1,) + result.shape[2:])


class PolyphaseFilterBank(PolyphaseFilterBankSamples):
    """Channelize using a polyphase filter bank, in the frequency domain.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream, with time as the first axis.
    response : `~numpy.ndarray`
        Polyphase filter.  The first dimension is taken to be the
        number of taps, and the second the number of channels.
    samples_per_frame : int, optional
        Number of complete output samples per frame (see Notes).  Default: 1.
    frequency : `~astropy.units.Quantity`, optional
        Frequencies for each channel in ``ih`` (channelized frequencies will
        be calculated).  Default: taken from ``ih`` (if available).
    sideband : array, optional
        Whether frequencies in ``ih`` are upper (+1) or lower (-1) sideband.
        Default: taken from ``ih`` (if available).

    See Also
    --------
    PolyphaseFilterBankSamples : filter in the time domain (usually slower).
    InversePolyphaseFilterBank : undo the effect of a polyphase filter.
    baseband_tasks.fourier.fft_maker : to select the FFT package used.
    """
    def __init__(self, ih, response, samples_per_frame=None,
                 frequency=None, sideband=None):
        super().__init__(ih, response=response,
                         samples_per_frame=samples_per_frame,
                         frequency=frequency, sideband=sideband)
        self._ppf_fft = self._FFT(shape=self._reshape, dtype=self.ih.dtype)
        self._ppf_ifft = self._ppf_fft.inverse()

    @lazyproperty
    def _ft_response_conj(self):
        long_response = np.zeros(self._reshape[:2], self.ih.dtype)
        long_response[:self._response.shape[0]] = self._response
        long_response.shape = (long_response.shape
                               + (1,) * len(self.ih.sample_shape))
        fft = self._FFT(shape=long_response.shape, dtype=self.ih.dtype)
        return fft(long_response).conj()

    def ppf(self, data):
        """Apply the PolyPhase Filter, in the frequency domain."""
        data = data.reshape(self._reshape)
        ft = self._ppf_fft(data)
        ft *= self._ft_response_conj
        result = self._ppf_ifft(ft)
        # Remove padding, which has turned around.
        result = result[:result.shape[0]+1-self._response.shape[0]]
        # Reshape as timestream (channelizer will immediately undo this...).
        return result.reshape((-1,) + result.shape[2:])


class InversePolyphaseFilterBank(PaddedTaskBase):
    """Dechannelize a stream produced by a polyphase filter bank.

    A polyphase filterbank attempts to make channel responses squarer, by
    convolving an input timestream with sinc-like function before doing a
    Fourier transform. This class attempts to deconvolve the signal, given the
    polyphase filter response. Like in most convolutions, a polyphase filter
    generally destroys some information, especially for frequencies near the
    edges of the channels. To optimize the process, Wiener deconvolution is
    used.

    The signal-to-noise ratio required for Wiener deconvolution is a
    combination of the response-dependent quality with which any signal can be
    recovered and the quality with which the signal was sampled. For CHIME,
    where channels are digitized with 4 bits, simulations show that if 3
    levels were covering the standard deviation of the input signal, ``sn=10``
    works fairly well. For GUPPI, which has 8 bit digitization but a response
    that strongly suppresses band edges, ``sn=30`` seems a good number.

    The deconvolution necessarily works poorly near edges in time, so should
    be done in overlapping blocks. Required padding is set with ``pad_start``
    and ``pad_end`` (which are in addition to the minimum padding required by
    the response). Adequate padding depend on response; from simulations, for
    CHIME a padding of 32 on both sides seems to suffice, while for GUPPI 128
    is needed.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream, with time as the first axis, and Fourier channel
        as the second.
    response : `~numpy.ndarray`
        Polyphase filter.  The first dimension is taken to be the
        number of taps, and the second the number of channels.
    sn : float
        Effective signal-to-noise ratio; see note above.
    pad_start, pad_end : int, optional
        Extra samples at the start and end to pad the frame being deconvolved;
        see note above. Default: 128.
    samples_per_frame : int, optional
        Number of complete output samples per frame.  Default: inferred from
        padding such that a minimum efficiency of 75% is reached.
    frequency : `~astropy.units.Quantity`, optional
        Frequencies for each output channel.  Default: inferred from ``ih``
        (if available).
    sideband : array, optional
        Whether frequencies are upper (+1) or lower (-1) sideband.
        Default: taken from ``ih`` (if available).

    See Also
    --------
    PolyphaseFilterBank : apply polyphase filter.
    baseband_tasks.fourier.fft_maker : to select the FFT package used.

    """
    def __init__(self, ih, response, sn, pad_start=128, pad_end=128,
                 samples_per_frame=None, frequency=None, sideband=None,
                 dtype=None):
        n_tap, n = response.shape
        self.dechannelized = Dechannelize(
            ih, n=n, samples_per_frame=None, frequency=frequency,
            sideband=sideband, dtype=dtype)
        self._FFT = self.dechannelized._FFT
        pad_minimum = (n_tap - 1) * n
        assert pad_minimum % 2 == 0
        pad_start = pad_start * n + pad_minimum // 2
        pad_end = pad_end * n + pad_minimum // 2
        super().__init__(self.dechannelized,
                         pad_start=pad_start, pad_end=pad_end,
                         samples_per_frame=samples_per_frame)
        self._response = response
        self._reshape = ((self._ih_samples_per_frame // n, n)
                         + self.ih.sample_shape)
        self._ppf_fft = self._FFT(
            shape=self._reshape, dtype=self.dtype)
        self._ppf_ifft = self._ppf_fft.inverse()
        self._inv_sn2 = 1. / (sn * sn)

    @lazyproperty
    def _ft_inverse_response(self):
        """Wiener deconvolution filter based on the PFB response."""
        long_response = np.zeros(self._reshape[:2], self.dtype)
        long_response[:self._response.shape[0]] = self._response
        long_response.shape = (long_response.shape
                               + (1,) * len(self.ih.sample_shape))
        fft = self._FFT(shape=long_response.shape, dtype=self.ih.dtype)
        ft_response = fft(long_response).conj()
        inverse = (ft_response.conj()
                   / (ft_response.real ** 2 + ft_response.imag ** 2
                      + self._inv_sn2)) * (1 + self._inv_sn2)
        return inverse

    def task(self, data):
        """Apply the inverse polyphase filter to a frame.

        Padding is removed from the result.
        """
        data = data.reshape(self._reshape)
        ft = self._ppf_fft(data)
        ft *= self._ft_inverse_response
        result = self._ppf_ifft(ft)
        # Reshape as timestream
        result = result.reshape((-1,) + result.shape[2:])
        # Remove padding.
        return result[self._pad_start:result.shape[0]-self._pad_end]
