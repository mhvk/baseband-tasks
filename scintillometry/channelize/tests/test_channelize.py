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

        # Quick test of channel sanity check in __init__.
        with pytest.raises(AssertionError):
            ct = ChannelizeTask(fh, 400001)

        ct.close()
