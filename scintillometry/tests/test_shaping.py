# Licensed under the GPLv3 - see LICENSE

import pytest
import numpy as np
from numpy.testing import assert_array_equal
import astropy.units as u

from ..shaping import Reshape, Transpose, ReshapeAndTranspose, SampleShapeChange

from baseband import vdif, dada
from baseband.data import SAMPLE_VDIF, SAMPLE_DADA


class TestReshape:
    @pytest.mark.parametrize('sample_shape', ((4, 2), (2, 4)))
    def test_reshape(self, sample_shape):
        fh = vdif.open(SAMPLE_VDIF)
        ref_data = fh.read()

        rt = Reshape(fh, sample_shape=sample_shape)
        assert fh.sample_shape == (8,)
        assert rt.sample_shape == sample_shape
        assert rt.start_time == fh.start_time
        assert rt.sample_rate == fh.sample_rate

        data = rt.read()
        assert_array_equal(data, ref_data.reshape((-1,) + sample_shape))

    def test_frequency_sideband_polarization_propagation1(self):
        fh = vdif.open(SAMPLE_VDIF)
        # Add frequency, sideband, and polarization information by hand.
        fh.frequency = 311.25 * u.MHz + (np.arange(8.) // 2) * 16. * u.MHz
        fh.sideband = 1
        fh.polarization = np.tile(['L', 'R'], 4)
        rt = Reshape(fh, (4, 2))
        assert rt.frequency.shape == (4, 1)
        assert np.all(rt.frequency == fh.frequency[::2].reshape(4, 1))
        assert rt.sideband.shape == ()
        assert np.all(rt.sideband == 1)
        assert rt.polarization.shape == (2,)
        assert np.all(rt.polarization == fh.polarization[:2])

    def test_frequency_sideband_polarization_propagation2(self):
        fh = vdif.open(SAMPLE_VDIF)
        # Add frequency and sideband information by hand.
        # (Note: these are all incorrect; just for testing purposes.)
        fh.frequency = 311.25 * u.MHz + (np.arange(8.) // 4) * 16. * u.MHz
        fh.sideband = np.tile([-1, 1], 4)
        fh.polarization = np.tile(['L', 'L', 'R', 'R'], 2)
        rt = Reshape(fh, (2, 2, 2))
        assert rt.frequency.shape == (2, 1, 1)
        assert np.all(rt.frequency == fh.frequency[::4].reshape(2, 1, 1))
        assert rt.sideband.shape == (2,)
        assert np.all(rt.sideband == fh.sideband[:2])
        assert rt.polarization.shape == (2, 1)
        assert np.all(rt.polarization == fh.polarization[:4:2].reshape(2, 1))


class TestTranspose:
    @staticmethod
    def get_reshape_and_transpose(fh, sample_shape=(4, 2),
                                  sample_axes=(2, 1)):
        rt = Reshape(fh, sample_shape=sample_shape)
        return Transpose(rt, sample_axes=sample_axes)

    def test_basic(self):
        fh = vdif.open(SAMPLE_VDIF)
        ref_data = fh.read().reshape((-1, 4, 2)).transpose(0, 2, 1)
        tt = self.get_reshape_and_transpose(fh, (4, 2), (2, 1))
        assert tt.start_time == fh.start_time
        assert tt.sample_rate == fh.sample_rate

        data = tt.read()
        assert_array_equal(data, ref_data)

    def test_frequency_sideband_polarization_propagation1(self):
        fh = vdif.open(SAMPLE_VDIF)
        # Add frequency, sideband, and polarization information by hand.
        fh.frequency = 311.25 * u.MHz + (np.arange(8.) // 2) * 16. * u.MHz
        fh.sideband = 1
        fh.polarization = np.tile(['L', 'R'], 4)
        tt = self.get_reshape_and_transpose(fh, (4, 2), (2, 1))
        assert tt.frequency.shape == (4,)
        assert np.all(tt.frequency == fh.frequency[::2])
        assert tt.sideband.shape == ()
        assert np.all(tt.sideband == 1)
        assert tt.polarization.shape == (2, 1)
        assert np.all(tt.polarization == fh.polarization[:2].reshape(2, 1))

    def test_frequency_sideband_polarization_propagation2(self):
        fh = vdif.open(SAMPLE_VDIF)
        # Add frequency and sideband information by hand.
        # (Note: these are all incorrect; just for testing purposes.)
        fh.frequency = 311.25 * u.MHz + (np.arange(8.) // 4) * 16. * u.MHz
        fh.sideband = np.tile([-1, 1], 4)
        fh.polarization = np.tile(['L', 'L', 'R', 'R'], 2)
        tt = self.get_reshape_and_transpose(fh, (2, 2, 2), (-1, -3, -2))
        assert tt.frequency.shape == (2, 1)
        assert np.all(tt.frequency == fh.frequency[::4].reshape(2, 1))
        assert tt.sideband.shape == (2, 1, 1)
        assert np.all(tt.sideband == fh.sideband[:2].reshape(2, 1, 1))
        assert tt.polarization.shape == (2,)
        assert np.all(tt.polarization == fh.polarization[:4:2])


class TestReshapeAndTranspose(TestTranspose):
    get_reshape_and_transpose = ReshapeAndTranspose


class TestSampleShapeChange(TestTranspose):
    @staticmethod
    def get_reshape_and_transpose(fh, sample_shape=(4, 2),
                                  sample_axes=(2, 1)):
        new_shape = (-1,) + sample_shape
        new_axes = (0,) + sample_axes

        def task(data):
            return data.reshape(new_shape).transpose(new_axes)

        return SampleShapeChange(fh, task)

    def test_swap_axes(self):
        fh = vdif.open(SAMPLE_VDIF)
        # Add frequency, sideband, and polarization information by hand.
        fh.frequency = 311.25 * u.MHz + (np.arange(8.) // 2) * 16. * u.MHz
        fh.sideband = 1
        fh.polarization = np.tile(['L', 'R'], 4)
        st = SampleShapeChange(fh, lambda data: (data.reshape(-1, 4, 2)
                                                 .swapaxes(1, 2)))
        assert st.frequency.shape == (4,)
        assert np.all(st.frequency == fh.frequency[::2])
        assert st.sideband.shape == ()
        assert np.all(st.sideband == 1)
        assert st.polarization.shape == (2, 1)
        assert np.all(st.polarization == fh.polarization[:2].reshape(2, 1))
