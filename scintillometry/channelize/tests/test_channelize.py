# Licensed under the GPLv3 - see LICENSE

import numpy as np
import astropy.units as u
import pytest

from .. import ChannelizeTask

from baseband import vdif
from baseband.data import SAMPLE_VDIF


class TestChannelize(object):
    """Test channelization using Baseband's sample VDIF file."""

    def setup(self):
        """Pre-calculate channelized data."""
        self.nchan = 1024

        with vdif.open(SAMPLE_VDIF) as fh:
            self.ref_start_time = fh.start_time
            self.ref_sample_rate = fh.sample_rate
            data = fh.read()

        last_sample = self.nchan * (data.shape[0] // self.nchan)
        self.ref_data = np.fft.rfft(
            data[:last_sample].reshape((-1, self.nchan) + data.shape[1:]),
            axis=1).astype('complex64')

    def test_channelizetask(self):
        """Test channelization task."""

        fh = vdif.open(SAMPLE_VDIF)
        ct = ChannelizeTask(fh, self.nchan)

        # Check sample pointer.
        assert ct.sample_rate == self.ref_sample_rate / self.nchan
        assert ct.shape == self.ref_data.shape
        assert ct.size == np.prod(self.ref_data.shape)
        assert ct.ndim == len(self.ref_data.shape)
        assert ct.tell() == 0
        assert ct.tell(unit='time') == ct.start_time == self.ref_start_time
        assert (ct.stop_time - ct.start_time -
                (self.nchan * self.ref_data.shape[0]) /
                self.ref_sample_rate) < 1*u.ns

        # Channelize everything.
        data1 = ct.read()
        assert ct.tell() == ct.shape[0]
        assert (ct.time - ct.start_time -
                ct.shape[0] / ct.sample_rate) < 1*u.ns
        assert ct.dtype is self.ref_data.dtype is data1.dtype
        assert np.allclose(self.ref_data, data1)

        # Seeking and selective decode.
        ct.seek(-3, 2)
        assert ct.tell() == ct.shape[0] - 3
        data2 = ct.read()
        assert data2.shape[0] == 3
        assert np.allclose(self.ref_data[-3:], data2)

        ct.seek(-2, 2)
        with pytest.raises(EOFError):
            ct.read(10)

        ct.close()
        assert fh.closed

    @pytest.mark.parametrize('spf', (5, 8))
    def test_channelizeblocksize(self, spf):
        """Test that different samples per frame changes length of output
        data, but not its values.
        """
        with vdif.open(SAMPLE_VDIF) as fh:
            ct = ChannelizeTask(fh, self.nchan, samples_per_frame=spf)
            ref_nsample = spf * (self.ref_data.shape[0] // spf)
            assert ct.shape == (ref_nsample,) + self.ref_data.shape[1:]
            data1 = ct.read()
            assert np.allclose(self.ref_data[:ref_nsample], data1)
            ct.seek(-2, 2)
            data2 = ct.read()
            assert data2.shape[0] == 2
            assert np.allclose(self.ref_data[ref_nsample - 2:ref_nsample],
                               data2)
            ct.seek(-5, 2)
            with pytest.raises(EOFError):
                ct.read(12)
