# Licensed under the GPLv3 - see LICENSE

import operator
from astropy.utils import lazyproperty

from ..base.base import ModuleBase
from ..fourier import get_fft_maker


__all__ = ['ChannelizeModule']


class Channelizer(object):

    # NOTE: always assumes time axis is 0.  Probably not a problem.
    def __init__(self, dtype, samples_per_block, nchan, sample_shape,
                 FFT=None):
        if FFT is None:
            FFT = get_fft_maker('numpy')
        self._fft = FFT((samples_per_block, nchan) + sample_shape,
                        dtype, axis=1)

    def __call__(self, data):
        """Split data into frequency channels."""
        return self._fft(data.reshape(self._fft.time_shape))

    @property
    def shape(self):
        return self._fft.freq_shape

    @property
    def dtype(self):
        return self._fft.freq_dtype


class ChannelizeModule(ModuleBase):

    def __init__(self, ih, nchan, samples_per_block=1, FFT=None):

        super().__init__(ih)

        self.nchan = operator.index(nchan)
        # NOTE: should this be made private?  Should the Channelizer at least
        # have a _min_samples_per_block value?
        self.samples_per_block = operator.index(samples_per_block)
        # NOTE: nsample is the number of output samples for the largest integer
        # number of blocks available from the input file.
        self._nsample = self.samples_per_block * (
            self.ih.shape[0] // self.nchan // self.samples_per_block)
        assert self._nsample > 0, ("not enough samples to fill one block of "
                                   "data!")
        self.sample_rate = ih.sample_rate / self.nchan

        # Initialize channelizer.
        self._channelizer = Channelizer(
            self.ih.dtype, self.samples_per_block, self.nchan,
            self.ih.sample_shape, FFT=FFT)

    def _read_block(self, block_index):
        self.ih.seek(block_index * self.nchan * self.samples_per_block)
        data = self.ih.read(self.nchan * self.samples_per_block)
        return self._channelizer(data)

    @property
    def dtype(self):
        return self._channelizer.dtype

    @property
    def sample_shape(self):
        return self._channelizer.shape[1:]

    @lazyproperty
    def start_time(self):
        """Start time of the file.

        See also `time` for the time of the sample pointer's current offset,
        and (if available) `stop_time` for the time at the end of the file.
        """
        return self.ih.start_time

    @lazyproperty
    def stop_time(self):
        """Time at the end of the file, just after the last sample.

        See also `start_time` for the start time of the file, and `time` for
        the time of the sample pointer's current offset.
        """
        # NOTE: can eventually replace this with seek_temporary block.
        current_pos = self.ih.tell()
        self.ih.seek(self._nsample * self.nchan)
        stop_time = self.ih.tell(unit='time')
        self.ih.seek(current_pos)
        return stop_time
