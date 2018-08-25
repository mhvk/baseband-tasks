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


class IncoherentDedispersionTask(TaskBase):

    def __init__(self, ih, dm, base_freq, nchan, freq_order=None,
                 pad_samples_per_frame=1, FFT=None):

        # Set frame and stream properties.
        pad_samples_per_frame = operator.index(pad_samples_per_frame)
        # Time domain bin for channelization.
        nchan_bin = operator.index(nchan) * (2 if ih.complex_data else 1)
        sample_rate = ih.sample_rate / nchan_bin

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
        self._min_freq = np.min(extrema_freqs)
        self._max_freq = np.max(extrema_freqs)

        # Set dispersion measure.
        self.dm = DispersionMeasure(dm)

        # Calculate maximum time delay, and determine the size of padding
        # needed to account for dispersion.
        max_time_delay = self.dm.time_delay(self._min_freq, self.max_freq)
        max_delay_offset = int(np.ceil((
            max_time_delay * sample_rate).to_value(u.one)))

        samples_per_frame = pad_samples_per_frame - max_delay_offset
        self._raw_padframe_len = nchan_bin * pad_samples_per_frame
        self._raw_frame_len = nchan_bin * samples_per_frame
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
        nsample = samples_per_frame * nframe

        # Initialize FFT.
        if FFT is None:
            FFT = get_fft_maker('numpy')
        self._fft = FFT((pad_samples_per_frame, nchan_bin) + ih.sample_shape,
                        ih.dtype, axis=1)

        super().__init__(ih, (nsample,) + self._fft.freq_shape[1:],
                         sample_rate, samples_per_frame, self._fft.freq_dtype)

    @property
    def base_freq(self):
        return self._base_freq

    @property
    def max_freq(self):
        return self._max_freq

    @property
    def min_freq(self):
        return self._min_freq

    @property
    def freq_order(self):
        return self._freq_order

    @lazyproperty
    def freq(self):
        return self.base_freq + self.freq_order * self._fft.freq

    @lazyproperty
    def dispersion_offset(self):
        dm_delay = self.dm.time_delay(self.freq, self.max_freq)
        return np.floor((dm_delay * self.sample_rate)
                        .decompose()).value.astype('int')

    @lazyproperty
    def _sample_shape_indices(self):
        return tuple(itertools.product(*(range(axis)
                     for axis in self.ih.sample_shape)))

    def dedisperse(self, data):
        channelized_data = self._fft(data.reshape(self._fft.time_shape))

        for indices in self._sample_shape_indices:
            for i in range(self.nchan):
                channelized_data[(slice(None), i) + indices] = np.roll(
                    channelized_data[(slice(None), i) + indices],
                    -self.dispersion_offset[(i,) + indices], axis=0)

        return channelized_data[:self.samples_per_frame]

    def _read_frame(self, frame_index):
        # TODO: For speed during sequential reading, can buffer frames, then,
        # during sequential decode, copy data from self._raw_frame_len to
        # self._raw_frame_len + self._raw_pad_len into the next buffer, rather
        # than re-decoding them.  Dangerous if self.ih gets manually shifted
        # between reads.
        self.ih.seek(frame_index * self._raw_frame_len)
        data = self.ih.read(self._raw_padframe_len)
        return self.dedisperse(data)


class CoherentDedispersionTask(TaskBase):
    pass