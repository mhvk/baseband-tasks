# Licensed under the GPLv3 - see LICENSE

import pytest
import numpy as np
from numpy.testing import assert_array_equal
import astropy.units as u
from astropy.time import Time

from ..convolution import Convolve
from ..generators import EmptyStreamGenerator

from baseband import vdif, dada
from baseband.data import SAMPLE_VDIF, SAMPLE_DADA


class TestConvolve:
    """Test convolution with simple smoothing filter."""

    Convolve = Convolve

    def setup(self):
        self.response = np.ones(3)

    def test_convolve(self):
        # Load baseband file and get reference intensities.
        fh = dada.open(SAMPLE_DADA)
        ref_data = fh.read()
        expected = ref_data[:-2] + ref_data[1:-1] + ref_data[2:]

        # Have 16000 - 2 useful samples -> can use 842, but add 2 for response.
        ct = self.Convolve(fh, self.response, samples_per_frame=844)
        # Convolve everything.
        data1 = ct.read()
        assert ct.start_time == fh.start_time
        assert ct.tell() == ct.shape[0] == fh.shape[0] - 2
        assert np.allclose(expected, data1)

        # Seeking and selective convolution.
        ct.seek(-3, 2)
        assert ct.tell() == ct.shape[0] - 3
        data2 = ct.read()
        assert data2.shape[0] == 3
        assert np.allclose(expected[-3:], data2)

        ct.close()
