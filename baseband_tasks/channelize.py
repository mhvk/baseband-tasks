# Licensed under the GPLv3 - see LICENSE

import operator

from .base import TaskBase, getattr_if_none
from .fourier import fft_maker


__all__ = ['Channelize', 'Dechannelize']


class Channelize(TaskBase):
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
    frequency : `~astropy.units.Quantity`, optional
        Frequencies for each channel in ``ih`` (channelized frequencies will
        be calculated).  Default: taken from ``ih`` (if available).
    sideband : array, optional
        Whether frequencies in ``ih`` are upper (+1) or lower (-1) sideband.
        Default: taken from ``ih`` (if available).

    See Also
    --------
    baseband_tasks.fourier.fft_maker : to select the FFT package used.

    Notes
    -----
    Instances initialize an FFT that acts upon axis 1 of an input with shape::

        (samples_per_frame, n) + ih.sample_shape

    Setting ``samples_per_frame`` to a number larger than 1 results in the FFT
    performing channelization on multiple blocks per call.  Depending on the
    backend used, this may speed up sequential channelization, though for tests
    using `numpy.fft` the performance improvement seems to be negligible.
    """

    def __init__(self, ih, n, samples_per_frame=1, *,
                 frequency=None, sideband=None):

        self._n = n = operator.index(n)
        samples_per_frame = operator.index(samples_per_frame)
        # Initialize channelizer.
        self._FFT = fft_maker.get()
        self._fft = self._FFT((samples_per_frame, n) + ih.sample_shape,
                              ih.dtype, axis=1, sample_rate=ih.sample_rate)

        frequency = getattr_if_none(ih, 'frequency', frequency, required=False)
        sideband = getattr_if_none(ih, 'sideband', sideband, required=False)
        if frequency is not None:
            # Do not use in-place, since frequency may have simplified shape.
            frequency = frequency + self._fft.frequency * sideband

        sample_rate = ih.sample_rate / n
        shape = (-1,) + self._fft.frequency_shape[1:]
        super().__init__(ih, shape=shape, sample_rate=sample_rate,
                         samples_per_frame=samples_per_frame,
                         frequency=frequency, sideband=sideband,
                         dtype=self._fft.frequency_dtype)

    def task(self, data):
        return self._fft(data.reshape(self._fft.time_shape))

    def inverse(self, ih):
        """Create a Dechannelize instance that undoes this Channelization.

        Parameters
        ----------
        ih : task or `baseband` stream reader
            Input data stream to be dechannelized.
        """
        # TODO: would be nicer to somehow use _fft.inverse().
        with fft_maker.set(self._FFT):
            return Dechannelize(ih, n=self._fft.time_shape[1],
                                dtype=self._fft.time_dtype)


class Dechannelize(TaskBase):
    """Basic dechannelizer.

    Inverse Fourier transform on first sample axis (which gets removed).

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream, with time as the first axis, and Fourier channel
        as the second.
    n : int, optional
        Number of output samples to create for each spectrum.  By default,
        for complex output data, the same as the number of channels.
        For real output data, the number has to be passed in.
    samples_per_frame : int, optional
        Number of output samples to produce in one go.  Rounded to the
        nearest multiple of ``n``. Default: inferred from underlying stream,
        i.e., ``ih.samples_per_frame * n``.
    dtype : `~numpy.dtype`, optional
        Dtype of the output samples.  Default: complex (like ``ih``).
    frequency : `~astropy.units.Quantity`, optional
        Frequencies for each output channel.  Default: inferred from ``ih``
        (if available).
    sideband : array, optional
        Whether frequencies are upper (+1) or lower (-1) sideband.
        Default: taken from ``ih`` (if available).

    See Also
    --------
    baseband_tasks.fourier.fft_maker : to select the FFT package used.

    Notes
    -----
    To construct a Dechannelizer for a given Channelizer instance, use
    the ``from_channelizer`` classmethod.

    """

    def __init__(self, ih, n=None, samples_per_frame=None, *,
                 dtype=None, frequency=None, sideband=None):

        assert ih.complex_data, "Dechannelization needs complex spectra."

        if dtype is None:
            dtype = ih.dtype  # this keeps it complex by default.

        if n is None:
            if dtype.kind == 'c':
                n = ih.sample_shape[0]
            else:
                raise ValueError("need explicit 'n' for real transform.")
        else:
            n = operator.index(n)

        if samples_per_frame is None:
            ih_samples_per_frame = ih.samples_per_frame
        else:
            ih_samples_per_frame = max(int(round(samples_per_frame / n)), 1)

        # Initialize dechannelizer.
        self._FFT = fft_maker.get()
        self._ifft = self._FFT((ih_samples_per_frame, n) + ih.sample_shape[1:],
                               dtype=dtype, axis=1, direction='backward')

        sample_rate = ih.sample_rate * n
        if frequency is None and getattr(ih, 'frequency', None) is not None:
            frequency = ih.frequency[0]

        super().__init__(ih, shape=(-1,) + ih.shape[2:],
                         sample_rate=sample_rate,
                         ih_samples_per_frame=ih_samples_per_frame,
                         frequency=frequency, sideband=sideband,
                         dtype=self._ifft.time_dtype)
        self._n = n

    def task(self, data):
        return self._ifft(data).reshape((-1,) + self.sample_shape)

    def inverse(self, ih):
        """Create a Channelize instance that undoes this Dechannelization.

        Parameters
        ----------
        ih : task or `baseband` stream reader
            Input data stream to be channelized.
        """
        # TODO: would be nicer to somehow use _fft.inverse().
        with fft_maker.set(self._FFT):
            return Channelize(ih, n=self._ifft.time_shape[1])
