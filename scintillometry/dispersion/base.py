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

    def __init__(self, ih, dm, base_freq, nchan, freq_order,
                 samples_per_frame):

        samples_per_frame = operator.index(samples_per_frame)
        self.nchan = operator.index(nchan)
        self._base_freq = np.atleast_1d(np.array(base_freq)) * base_freq.unit
        if freq_order is None:
            freq_order = np.ones(ih.sample_shape, dtype=int)
        self._freq_order = np.atleast_1d(np.array(freq_order))

        self.dm = DispersionMeasure(dm)

        # If not complex, base_freq is at the edge of the band, and
        # freq_order determines which edge (1 for bottom, -1 for top).  If
        # complex, base_freq is assumed to be in the middle of the band.
        # NOTE: unless the exact bandwidth edge is returned by np.fft.rfftfreq,
        # this will give slightly different answers than Rob's code.
        if not ih.complex_data:
            bandwidth = ih.sample_rate / 2.
            extrema_freqs = base_freq + freq_order * bandwidth
        else:
            bandwidth = ih.sample_rate
            extrema_freqs = base_freq.unit * np.concatenate([
                (base_freq + freq_order * bandwidth).value,
                (base_freq - freq_order * bandwidth).value])
        self._ref_freq = np.max(extrema_freqs)

        max_time_delay = self.dm.time_delay(np.min(extrema_freqs),
                                            self.ref_freq)
        max_delay_offset = int(np.ceil((
            max_time_delay * ih.sample_rate / self.nchan).to_value(u.one)))

        # Number of ih samples within one frame.
        self._raw_frame_len = self.nchan * samples_per_frame
        # Number of additional ih samples needed to account for dispersion.
        # _read_frame will seek using _raw_frame_len, and read
        # _raw_padframe_len number of samples.
        self._raw_padframe_len = self.nchan * (
            self._raw_frame_len + max_delay_offset)
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
        assert (nframe > 0) and (remaining_samples >= raw_pad_len), (
            "not enough samples to fill one frame of data!")
        nsample = samples_per_frame * nframe

        sample_rate = ih.sample_rate / self.nchan

        # Don't set the dtype yet - subclasses use the shape to initialize
        # fft, which will calculate the dtype.
        super().__init__(ih, (nsample, self.nchan) + ih.sample_shape,
                         sample_rate, samples_per_frame, None)

    @property
    def base_freq(self):
        return self._base_freq

    @property
    def ref_freq(self):
        return self._ref_freq

    @property
    def freq_order(self):
        return self._freq_order

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

    def __init__(self, ih, dm, base_freq, nchan,
                 freq_order=None, samples_per_frame=1, FFT=None):

        super().__init__(ih, dm, base_freq, nchan, freq_order,
                         samples_per_frame)

        # Initialize FFT.
        if FFT is None:
            FFT = get_fft_maker('numpy')
        self._fft = FFT((self._raw_padframe_len //
                         self.nchan, self.nchan) + ih.sample_shape,
                        ih.dtype, axis=1)

        self._dtype = self._fft.freq_dtype

    @lazyproperty
    def freq(self):
        return self.base_freq + self.freq_order * self._fft.freq

    @lazyproperty
    def dispersion_offset(self):
        dm_delay = self.dm.time_delay(self.freq, self.ref_freq)
        dt = self.nchan / self.ih.sample_rate * (
            2 if self.ih.complex_data else 1)
        return np.floor((dm_delay / dt).decompose()).value.astype('int')

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

        return channelized_data[:self._raw_padframe_len]


class CoherentDedispersionTask(DedispersionTaskBase):
    pass