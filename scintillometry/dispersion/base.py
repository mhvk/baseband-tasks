# Licensed under the GPLv3 - see LICENSE

import numpy as np
import astropy.units as u
import operator
import itertools
from astropy.utils import lazyproperty

from ..base.base import TaskBase
from ..fourier import get_fft_maker
from .dm import DispersionMeasure


__all__ = ['DedispersionTaskBase', 'IncoherentDedispersionTask',
           'CoherentDedispersionTask']


class DedispersionTaskBase(TaskBase):

    def __init__(self, ih, dm, base_freq, n_transform, freq_order=None,
                 pad_samples_per_frame=1):

        # Set frame and stream properties.
        pad_samples_per_frame = operator.index(pad_samples_per_frame)
        # Time domain bin for channelization.
        n_transform = operator.index(n_transform)
        sample_rate = ih.sample_rate / n_transform

        # Set reference frequency.  If not complex, base_freq is at the edge of
        # the band, and freq_order determines which edge (1 for bottom, -1 for
        # top).  If complex, base_freq is at the middle of the band.
        # NOTE: unless the exact bandwidth edge is returned by np.fft.rfftfreq,
        # this will give slightly different answers than Rob's code.
        self._base_freq = np.atleast_1d(np.array(base_freq)) * base_freq.unit
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
                                        sample_rate).to_value(u.one)))

        samples_per_frame = pad_samples_per_frame - max_delay_offset
        self._raw_padframe_len = n_transform * pad_samples_per_frame
        self._raw_frame_len = n_transform * samples_per_frame
        assert self._raw_frame_len > 0, (
            "pad_samples_per_frame={} is smaller than the minimum number "
            "needed for dedispersion, {}".format(pad_samples_per_frame,
                                                 max_delay_offset + 1))
        raw_pad_len = self._raw_padframe_len - self._raw_frame_len

        # Calculate task output shape and sample rate.
        nframe, remaining_samples = divmod(ih.shape[0], self._raw_frame_len)
        # HACK: We don't support partial frames yet, so if nframe > 1 but
        # remaining_samples < raw_pad_len, we reduce the number of frames by an
        # appropriate amount to prevent EOF errors.
        if nframe > 1 and remaining_samples < raw_pad_len:
            if self._raw_frame_len > raw_pad_len:
                nframe -= 1
                remaining_samples += self._raw_frame_len
            else:
                subtract_nframes = raw_pad_len // self._raw_frame_len + 1
                nframe -= subtract_nframes
                remaining_samples += self._raw_frame_len * subtract_nframes
        # Check if our final nframe and remaining_samples make sense.
        # NOTE: a much simpler check could just be nframe =
        # ih.shape[0] // self._raw_padframe_len, but this also checks our math.
        assert nframe > 0, "not enough samples to fill one frame of data!"
        self._nsample = samples_per_frame * nframe

        # Don't set the shape or dtype yet - subclasses use the shape to
        # initialize, which will auto-calculate both.
        super().__init__(ih, None, sample_rate, samples_per_frame,
                         None)

    @property
    def base_freq(self):
        return self._base_freq

    @property
    def max_freq(self):
        return self._max_freq

    @property
    def freq_order(self):
        return self._freq_order

    @lazyproperty
    def freq(self):
        return self.base_freq + self.freq_order * self._fft.freq

    def _read_frame(self, frame_index):
        # TODO: For speed during sequential reading, can buffer frames, then,
        # during sequential decode, copy data from self._raw_frame_len to
        # self._raw_frame_len + self._raw_pad_len into the next buffer, rather
        # than re-decoding them.  Dangerous if self.ih gets manually shifted
        # between reads.
        self.ih.seek(frame_index * self._raw_frame_len)
        data = self.ih.read(self._raw_padframe_len)
        return self.dedisperse(data)


class IncoherentDedispersionTask(DedispersionTaskBase):

    def __init__(self, ih, dm, base_freq, n_transform, freq_order=None,
                 pad_samples_per_frame=1, FFT=None):

        super().__init__(ih, dm, base_freq, n_transform, freq_order=freq_order,
                         pad_samples_per_frame=pad_samples_per_frame)

        # Initialize FFT.
        if FFT is None:
            FFT = get_fft_maker('numpy')
        self._fft = FFT(((pad_samples_per_frame, n_transform) +
                         self.ih.sample_shape),
                        self.ih.dtype, axis=1, sample_rate=self.ih.sample_rate)

        self._shape = (self._nsample,) + self._fft.freq_shape[1:]
        self._dtype = self._fft.freq_dtype

    @lazyproperty
    def dispersion_offset(self):
        dm_delay = self.dm.time_delay(self.freq, self.max_freq)
        return np.floor((dm_delay * self.sample_rate)
                        .decompose()).value.astype('int')

    @lazyproperty
    def _sample_shape_indices(self):
        return tuple(itertools.product(
            *(range(axis) for axis in self.ih.sample_shape)))

    def dedisperse(self, data):
        channelized_data = self._fft(data.reshape(self._fft.time_shape))

        for indices in self._sample_shape_indices:
            for i in range(channelized_data.shape[1]):
                channelized_data[(slice(None), i) + indices] = np.roll(
                    channelized_data[(slice(None), i) + indices],
                    -self.dispersion_offset[(i,) + indices], axis=0)

        return channelized_data[:self.samples_per_frame]


class CoherentDedispersionTask(DedispersionTaskBase):

    def __init__(self, ih, dm, base_freq, n_transform=1, freq_order=None,
                 pad_samples_per_frame=1, FFT=None):

        super().__init__(ih, dm, base_freq, n_transform, freq_order=freq_order,
                         pad_samples_per_frame=pad_samples_per_frame)

        # Initialize FFT.
        if FFT is None:
            FFT = get_fft_maker('numpy')
        # Dedispersion FFT and inverse.
        self._fft = FFT((self._raw_padframe_len,) + self.ih.sample_shape,
                        self.ih.dtype, axis=0, sample_rate=self.ih.sample_rate)
        self._ifft = self._fft.inverse()

        # Set up channelization FFT if needed.
        self._dtype = self._fft.freq_dtype
        if n_transform > 1:
            self._channelize = True
            self._chan_fft = FFT(((self.samples_per_frame, n_transform) +
                                  self.ih.sample_shape),
                                 self.ih.dtype, axis=1,
                                 sample_rate=self.ih.sample_rate)
            self._shape = (self._nsample,) + self._chan_fft.freq_shape[1:]
        else:
            self._channelize = False
            self._shape = (self._nsample,) + self._fft.freq_shape[1:]

    @lazyproperty
    def phase_factor(self):
        phase_factor = self.dm.phase_factor(self.freq, self.max_freq)
        phase_factor[:, self.freq_order == 1] = (
            np.conj(phase_factor[:, self.freq_order == 1]))
        return phase_factor

    def dedisperse(self, data):
        dedispersed_data = self._ifft(self._fft(data) *
                                      self.phase_factor)[:self._raw_frame_len]
        if self._channelize:
            return self._chan_fft(
                dedispersed_data.reshape(self._chan_fft.time_shape))
        return dedispersed_data
