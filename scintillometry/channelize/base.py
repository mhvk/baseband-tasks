# Licensed under the GPLv3 - see LICENSE

import operator
from astropy.utils import lazyproperty

from ..base.base import ModuleBase
from ..fourier import get_fft_maker


__all__ = ['ChannelizeModule']


class ChannelizeCore(object):

    # NOTE: maybe force users to pass their own FFT.
    # NOTE: always assumes time axis is 1.  Probably not a problem.
    def __init__(self, sample_shape, dtype, nchan, samples_per_block=1,
                 FFT=None):
        self.nchan = operator.index(nchan)
        self.samples_per_block = max(1, operator.index(samples_per_block))
        FFT = get_fft_maker('numpy') if FFT is None else FFT
        self._input_shape = (self.samples_per_block, self.nchan) + sample_shape
        self.fft = FFT(self._input_shape, dtype, axis=1)

    def channelize(self, data):
        """Split data into frequency channels."""
        return self.fft(data.reshape(self._input_shape))


class ChannelizeModule(ModuleBase, ChannelizeCore):

    def __init__(self, ih, nchan, samples_per_block=1, FFT=None):
        ModuleBase.__init__(self, ih)
        ChannelizeCore.__init__(self, self.ih.sample_shape, self.ih.dtype,
                                nchan, samples_per_block=samples_per_block,
                                FFT=FFT)
        # NOTE: nsample is the number of output samples for the largest integer
        # number of blocks available from the input file.
        self._nsample = self.samples_per_block * (
            self.ih._nsample // self.nchan // self.samples_per_block)
        assert self._nsample > 0, ("not enough samples to fill one block of "
                                   "data!")
        self.sample_rate = ih.sample_rate / self.nchan

    def _read_block(self, block_index):
        self.ih.seek(block_index * self.nchan * self.samples_per_block)
        data = self.ih.read(self.nchan * self.samples_per_block)
        return self.channelize(data)

    @property
    def dtype(self):
        return self.fft.freq_dtype

    @lazyproperty
    def shape(self):
        """Shape of the (squeezed/subset) stream data."""
        return (self._nsample,) + self.sample_shape

    @property
    def sample_shape(self):
        return self.fft.freq_shape[1:]

    @property
    def size(self):
        """Total number of component samples in the (squeezed/subset) stream
        data.
        """
        prod = 1
        for dim in self.shape:
            prod *= dim
        return prod

    @property
    def ndim(self):
        """Number of dimensions of the (squeezed/subset) stream data."""
        return len(self.shape)

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
        self.ih.seek(self.shape[0])
        stop_time = self.ih.tell(unit='time')
        self.ih.seek(current_pos)
        return stop_time
