# Licensed under the GPLv3 - see LICENSE

import operator

from ..base.base import TaskBase
from ..fourier import get_fft_maker


__all__ = ['ChannelizeTask']


class ChannelizeTask(TaskBase):

    def __init__(self, ih, nchan, samples_per_frame=1, FFT=None):

        super().__init__(ih)

        self.nchan = operator.index(nchan)
        # NOTE: should this be made private?  Should the Channelizer at least
        # have a _min_samples_per_frame value?
        self.samples_per_frame = operator.index(samples_per_frame)
        # NOTE: nsample is the number of output samples for the largest integer
        # number of frames available from the input file.
        self._nsample = self.samples_per_frame * (
            self.ih.shape[0] // self.nchan // self.samples_per_frame)
        assert self._nsample > 0, ("not enough samples to fill one frame of "
                                   "data!")
        self.sample_rate = ih.sample_rate / self.nchan

        # Initialize channelizer.
        if FFT is None:
            FFT = get_fft_maker('numpy')
        self._fft = FFT(
            (self.samples_per_frame, self.nchan) + self.ih.sample_shape,
            self.ih.dtype, axis=1)

    def _read_frame(self, frame_index):
        self.ih.seek(frame_index * self.nchan * self.samples_per_frame)
        data = self.ih.read(self.nchan * self.samples_per_frame)
        return self._fft(data.reshape(self._fft.time_shape))

    @property
    def dtype(self):
        return self._fft.freq_dtype

    @property
    def sample_shape(self):
        return self._fft.freq_shape[1:]
