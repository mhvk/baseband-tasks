# Licensed under the GPLv3 - see LICENSE

import pytest
import numpy as np
from numpy.testing import assert_array_equal
import astropy.units as u
from astropy.time import Time

from ..convolution import Convolve, ConvolveSamples
from ..generators import EmptyStreamGenerator

from baseband import vdif, dada
from baseband.data import SAMPLE_VDIF, SAMPLE_DADA


class TestConvolve:
    """Test convolution with simple smoothing filter."""

    def setup(self):
        self.response = np.ones(3)

    @pytest.mark.parametrize('convolve_task', (ConvolveSamples, Convolve))
    def test_convolve(self, convolve_task):
        # Load baseband file and get reference intensities.
        fh = dada.open(SAMPLE_DADA)
        ref_data = fh.read()
        expected = ref_data[:-2] + ref_data[1:-1] + ref_data[2:]

        # Have 16000 - 2 useful samples -> can use 842, but add 2 for response.
        ct = convolve_task(fh, self.response, samples_per_frame=844)
        # Convolve everything.
        data1 = ct.read()
        assert ct.tell() == ct.shape[0] == fh.shape[0] - 2
        assert abs(ct.start_time - fh.start_time -
                   2 / fh.sample_rate) < 1. * u.ns
        assert abs(ct.stop_time - fh.stop_time) < 1. * u.ns
        assert np.allclose(expected, data1, atol=1.e-4)

        # Seeking and selective convolution.
        ct.seek(-3, 2)
        assert ct.tell() == ct.shape[0] - 3
        data2 = ct.read()
        assert data2.shape[0] == 3
        assert np.allclose(expected[-3:], data2, atol=1.e-4)

        ct.close()
