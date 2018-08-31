# Licensed under the GPLv3 - see LICENSE

import numpy as np
import astropy.units as u
import operator
from astropy.utils import lazyproperty

from .base import TaskBase
from .fourier import get_fft_maker
from .dm import DispersionMeasure


__all__ = ['CoherentDedispersionTask']


class CoherentDedispersionTask(TaskBase):
    """Coherent dedispersion task.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream, with time as the first axis.
    dm : float or `~scintillometry.dm.DispersionMeasure` quantity
        Dispersion measure.
    pad_samples_per_frame : int
        The number of input samples
    base_freq : frequency `~astropy.units.Quantity` or `~numpy.ndarray` thereof
        Bandwidth reference frequency or frequencies, with the same dimensions
        as the input's sample shape.  If input samples are real, ``base_freq``
        should be the edge of the band; whether they represent the top or
        bottom of the band is determined by ``freq_order``. If they are
        complex, ``base_freq`` should represent the center of the band.
    freq_order : `~numpy.ndarray` of int, optional
        Frequency order, with the same dimensions as the input's sample shape.
        If frequency increases with index, the order is 1; if it decreases, it
        is -1.  Default: an `~numpy.ndarray` of +1.
    FFT : FFT maker or None, optional
        FFT maker.  Default: `None`, in which case the channelizer uses
        `~scintillometry.fourier.numpy.NumpyFFTMaker`.
    """

    def __init__(self, ih, dm, pad_samples_per_frame, base_freq,
                 freq_order=None, FFT=None):

        # Store the number of inputs samples to read per frame.
        self._pad_samples_per_frame = operator.index(pad_samples_per_frame)

        # Set reference frequency.
        # NOTE: unless the exact bandwidth edge is returned by np.fft.rfftfreq,
        # this will give slightly different answers than Rob's code.
        self._base_freq = np.atleast_1d(np.array(base_freq)) * base_freq.unit
        # Set the frequency order (1 for increasing, -1 for decreasing).
        if freq_order is None:
            freq_order = np.ones(ih.sample_shape, dtype=int)
        self._freq_order = np.atleast_1d(np.array(freq_order))

        half_rate = ih.sample_rate.value / 2.
        if not ih.complex_data:
            bandwidth = np.array([0., half_rate]) * ih.sample_rate.unit
        else:
            bandwidth = np.array([-half_rate, half_rate]) * ih.sample_rate.unit
        extrema_freqs = self.base_freq + (self.freq_order *
                                          bandwidth[:, np.newaxis])
        self._max_freq = np.max(extrema_freqs)

        # Set dispersion measure.
        self.dm = DispersionMeasure(dm)

        # Calculate maximum time delay, and determine the size of padding
        # needed to account for dispersion.
        max_time_delay = self.dm.time_delay(np.min(extrema_freqs),
                                            self.max_freq)
        max_delay_offset = int(np.ceil((max_time_delay *
                                        ih.sample_rate).to_value(u.one)))
        samples_per_frame = self._pad_samples_per_frame - max_delay_offset
        assert samples_per_frame > 0, (
            "pad_samples_per_frame={} is smaller than the minimum number "
            "needed for dedispersion, {}".format(pad_samples_per_frame,
                                                 max_delay_offset + 1))

        # Calculate task output shape and sample rate.
        nframe, remaining_samples = divmod(ih.shape[0], samples_per_frame)
        # HACK: We don't support partial frames yet, so if nframe > 1 but
        # remaining_samples < max_delay_offset, we reduce the number of frames
        # by an appropriate amount to prevent EOF errors.
        if nframe > 1 and remaining_samples < max_delay_offset:
            if samples_per_frame > max_delay_offset:
                nframe -= 1
                remaining_samples += samples_per_frame
            else:
                subtract_nframes = max_delay_offset // samples_per_frame + 1
                nframe -= subtract_nframes
                remaining_samples += samples_per_frame * subtract_nframes
        assert nframe > 0, "not enough samples to fill one frame of data!"
        self._nsample = samples_per_frame * nframe

        # Initialize FFT.
        if FFT is None:
            FFT = get_fft_maker('numpy')
        # Dedispersion FFT and inverse.
        self._fft = FFT((self._pad_samples_per_frame,) + ih.sample_shape,
                        ih.dtype, axis=0, sample_rate=ih.sample_rate)
        self._ifft = self._fft.inverse()

        super().__init__(ih, (self._nsample,) + ih.sample_shape,
                         ih.sample_rate, samples_per_frame, ih.dtype)

    @property
    def base_freq(self):
        """Bandwidth reference frequencies."""
        return self._base_freq

    @property
    def max_freq(self):
        """Largest frequency in each band."""
        return self._max_freq

    @property
    def freq_order(self):
        """Frequency order."""
        return self._freq_order

    @lazyproperty
    def freq(self):
        """Frequencies of the Fourier-transformed frame."""
        return self.base_freq + self.freq_order * self._fft.freq

    @lazyproperty
    def phase_factor(self):
        """Phase offsets of the Fourier-transformed frame."""
        phase_factor = self.dm.phase_factor(self.freq, self.max_freq)
        phase_factor[:, self.freq_order == 1] = (
            np.conj(phase_factor[:, self.freq_order == 1]))
        return phase_factor

    def _dedisperse(self, data):
        return self._ifft(self._fft(data) *
                          self.phase_factor)[:self._samples_per_frame]

    def _read_frame(self, frame_index):
        # TODO: For speed during sequential reading, can buffer frames, then,
        # during sequential decode, copy data from self._samples_per_frame to
        # self._pad_samples_per_frame into the next buffer, rather than
        # re-decoding them.  Dangerous if self.ih gets manually shifted
        # between reads.
        self.ih.seek(frame_index * self._samples_per_frame)
        data = self.ih.read(self._pad_samples_per_frame)
        return self._dedisperse(data)
