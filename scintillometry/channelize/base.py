# Licensed under the GPLv3 - see LICENSE

import operator

from ..base.base import TaskBase
from ..fourier import get_fft_maker


__all__ = ['ChannelizeTask']


class ChannelizeTask(TaskBase):

    def __init__(self, ih, n, samples_per_frame=1, FFT=None):

        n = operator.index(n)
        samples_per_frame = operator.index(samples_per_frame)
        sample_rate = ih.sample_rate / n

        nsample = samples_per_frame * (
            ih.shape[0] // n // samples_per_frame)
        assert nsample > 0, "not enough samples to fill one frame of data!"
        self._raw_frame_len = n * samples_per_frame

        # Initialize channelizer.
        if FFT is None:
            FFT = get_fft_maker('numpy')
        self._fft = FFT((samples_per_frame, n) + ih.sample_shape,
                        ih.dtype, axis=1)

        super().__init__(ih, (nsample,) + self._fft.freq_shape[1:],
                         sample_rate, samples_per_frame, self._fft.freq_dtype)

    def _read_frame(self, frame_index):
        self.ih.seek(frame_index * self._raw_frame_len)
        data = self.ih.read(self._raw_frame_len)
        return self._fft(data.reshape(self._fft.time_shape))
