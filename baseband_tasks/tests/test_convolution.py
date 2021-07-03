# Licensed under the GPLv3 - see LICENSE

import pytest
import numpy as np
import astropy.units as u
from astropy.time import Time

from baseband_tasks.convolution import Convolve, ConvolveSamples
from baseband_tasks.generators import NoiseGenerator

from .common import UseDADASample


class TestConvolveDADA(UseDADASample):
    @pytest.mark.parametrize('convolve_task', (ConvolveSamples, Convolve))
    def test_convolve(self, convolve_task):
        # Load baseband file and get reference intensities.
        fh = self.fh
        ref_data = fh.read()
        expected = ref_data[:-2] + ref_data[1:-1] + ref_data[2:]

        # Have 16000 - 2 useful samples -> but samples_per_frame should
        # not influence this, so pick one unrelated.
        response = np.ones(3)
        ct = convolve_task(fh, response, samples_per_frame=1024)
        # Convolve everything.
        data1 = ct.read()
        assert ct.tell() == ct.shape[0] == fh.shape[0] - 2
        assert abs(ct.start_time - fh.start_time
                   - 2 / fh.sample_rate) < 1. * u.ns
        assert abs(ct.stop_time - fh.stop_time) < 1. * u.ns
        assert np.allclose(expected, data1, atol=1.e-4)

        # Seeking and selective convolution.
        ct.seek(-3, 2)
        assert ct.tell() == ct.shape[0] - 3
        data2 = ct.read()
        assert data2.shape[0] == 3
        assert np.allclose(expected[-3:], data2, atol=1.e-4)


class TestConvolveNoise:
    """Test convolution with simple smoothing filter."""

    def setup(self):
        self.response = np.ones(3)
        self.start_time = Time('2010-11-12T13:14:15')
        self.sample_rate = 10. * u.kHz
        self.shape = (16000, 2)
        self.nh = NoiseGenerator(shape=self.shape,
                                 start_time=self.start_time,
                                 sample_rate=self.sample_rate,
                                 samples_per_frame=200, dtype=float,
                                 seed=12345)
        self.data = self.nh.read()

    @pytest.mark.parametrize('convolve_task', (ConvolveSamples, Convolve))
    @pytest.mark.parametrize('offset', (1, 2))
    def test_offset(self, convolve_task, offset):
        ct = convolve_task(self.nh, self.response, offset=offset,
                           samples_per_frame=1024)
        assert abs(ct.start_time - self.start_time
                   - (2 - offset) / self.sample_rate) < 1. * u.ns
        expected = self.data[:-2] + self.data[1:-1] + self.data[2:]
        data1b = ct.read(10)
        assert np.allclose(data1b, expected[:10])
        ct.seek(-10, 2)
        data1e = ct.read(10)
        assert np.allclose(data1e, expected[-10:])
        ct2 = convolve_task(self.nh, np.ones((3, 2)), offset=offset,
                            samples_per_frame=1024)
        ct2.seek(5)
        data2 = ct2.read(5)
        assert np.allclose(data2, data1b[5:])

    @pytest.mark.parametrize('convolve_task', (ConvolveSamples, Convolve))
    def test_different_response(self, convolve_task):
        response = np.array([[1., 1., 1.], [1., 1., 0.]]).T
        ct = convolve_task(self.nh, response, samples_per_frame=512)
        assert abs(ct.start_time - self.start_time
                   - 2 / self.sample_rate) < 1. * u.ns
        expected = (self.data[:-2] * np.array([1, 0]) + self.data[1:-1]
                    + self.data[2:])
        data1 = ct.read()
        assert np.allclose(data1, expected)

    def test_repr(self):
        ct = ConvolveSamples(self.nh, self.response)
        r = repr(ct)
        assert r.startswith('ConvolveSamples(ih')
        assert 'response=' in r
        assert 'offset=' not in r
        assert 'samples_per_frame' in r  # since different from input.

    @pytest.mark.parametrize('convolve_task', (ConvolveSamples, Convolve))
    def test_wrong_response(self, convolve_task):
        with pytest.raises(ValueError):
            convolve_task(self.nh, np.ones((3, 3)))
