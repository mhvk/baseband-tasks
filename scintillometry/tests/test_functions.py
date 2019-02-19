# Licensed under the GPLv3 - see LICENSE

import pytest
import numpy as np
from numpy.testing import assert_array_equal
import astropy.units as u
from astropy.time import Time

from ..functions import Square, Power
from ..generators import EmptyStreamGenerator

from baseband import vdif, dada
from baseband.data import SAMPLE_VDIF, SAMPLE_DADA


class TestSquare:
    """Test getting simple intensities using Baseband's sample DADA file."""

    def test_square(self):
        # Load baseband file and get reference intensities.
        fh = dada.open(SAMPLE_DADA)
        ref_data = fh.read()
        ref_data = np.real(ref_data * np.conj(ref_data))

        st = Square(fh)

        # Square everything.
        data1 = st.read()
        assert st.tell() == st.shape[0]
        assert (st.time - st.start_time -
                st.shape[0] / st.sample_rate) < 1*u.ns
        assert st.dtype is ref_data.dtype is data1.dtype
        assert np.allclose(ref_data, data1)

        # Seeking and selective squaring.
        st.seek(-3, 2)
        assert st.tell() == st.shape[0] - 3
        data2 = st.read()
        assert data2.shape[0] == 3
        assert np.allclose(ref_data[-3:], data2)

        st.close()

    def test_frequency_sideband_propagation(self):
        fh = vdif.open(SAMPLE_VDIF)
        # Add frequency and sideband information by hand.
        # (Note: sideband is incorrect; just for testing purposes)
        fh.frequency = 311.25 * u.MHz + (np.arange(8.) // 2) * 16. * u.MHz
        fh.sideband = np.tile([-1, +1], 4)
        fh.polarization = np.tile(['L', 'R'], 4)
        st = Square(fh)
        assert np.all(st.frequency == fh.frequency)
        assert np.all(st.sideband == fh.sideband)
        assert np.all(st.polarization == np.tile(['LL', 'RR'], 4))

    def test_missing_frequency_sideband_polarization(self):
        fh = vdif.open(SAMPLE_VDIF)
        st = Square(fh)
        with pytest.raises(AttributeError):
            st.frequency
        with pytest.raises(AttributeError):
            st.sideband
        with pytest.raises(AttributeError):
            st.polarization


class TestPower:
    """Test getting polarized intensities using Baseband's sample DADA file."""

    def test_power(self):
        # Load baseband file and get reference intensities.
        fh = dada.open(SAMPLE_DADA)
        ref_data = fh.read()
        r0, i0, r1, i1 = ref_data.view('f4').T
        ref_data = np.stack((r0 * r0 + i0 * i0,
                             r1 * r1 + i1 * i1,
                             r0 * r1 + i0 * i1,
                             i0 * r1 - r0 * i1), axis=1)

        pt = Power(fh, polarization=['L', 'R'])
        assert np.all(pt.polarization == np.array(['LL', 'RR', 'LR', 'RL']))

        # Square everything.
        data1 = pt.read()
        assert (pt.time - fh.start_time -
                fh.shape[0] / fh.sample_rate) < 1*u.ns
        assert pt.dtype is ref_data.dtype is data1.dtype
        assert np.allclose(ref_data, data1)

    def test_polarization_propagation(self):
        fh = dada.open(SAMPLE_DADA)
        # Add polarization information by hand.
        fh.polarization = np.array(['L', 'R'])
        pt = Power(fh)
        assert np.all(pt.polarization == np.array(['LL', 'RR', 'LR', 'RL']))
        # Swap order.
        fh.polarization = np.array(['R', 'L'])
        pt = Power(fh)
        assert np.all(pt.polarization == np.array(['RR', 'LL', 'RL', 'LR']))
        # Check it also works in other axes, or with an overly detailed array.
        # Use a fake stream a bit like the VDIF one, but with complex data.
        eh = EmptyStreamGenerator((10000, 2, 4), sample_rate=1.*u.Hz,
                                  start_time=Time('2018-01-01'))
        pt = Power(eh, polarization=[['L'], ['R']])
        expected = np.array([['LL'], ['RR'], ['LR'], ['RL']])
        assert np.all(pt.polarization == expected)
        pt = Power(eh, polarization=np.array([['L'] * 4, ['R'] * 4]))
        assert np.all(pt.polarization == expected)

    def test_frequency_sideband_propagation(self):
        # Regression test for gh-60
        frequency = np.array([[320.25], [320.25], [336.25], [336.25]]) * u.MHz
        sideband = np.array([[-1], [1], [-1], [1]])
        polarization = ['R', 'L']
        # Create a fake stream a bit like the VDIF one, but with complex data.
        eh = EmptyStreamGenerator((10000, 4, 2), sample_rate=1.*u.Hz,
                                  start_time=Time('2018-01-01'),
                                  frequency=frequency, sideband=sideband)
        pt = Power(eh, polarization=polarization)
        assert_array_equal(pt.polarization, np.array(['RR', 'LL', 'RL', 'LR']))
        assert_array_equal(pt.frequency, eh.frequency)
        assert_array_equal(pt.sideband, eh.sideband)

    def test_missing_polarization(self):
        fh = dada.open(SAMPLE_DADA)
        with pytest.raises(AttributeError):
            Power(fh)

    def test_wrong_polarization(self):
        fh = dada.open(SAMPLE_DADA)
        with pytest.raises(ValueError):
            Power(fh, polarization=['L'])
        with pytest.raises(ValueError):
            Power(fh, polarization=[['L'], ['R']])
        with pytest.raises(ValueError):
            Power(fh, polarization=[['L'], ['L']])

        fh = vdif.open(SAMPLE_VDIF)
        with pytest.raises(ValueError):
            Power(fh)

    def test_frequency_sideband_mismatch(self):
        frequency = np.array([[320.25], [320.25], [336.25], [336.25]]) * u.MHz
        sideband = np.array([[-1], [1], [-1], [1]])
        polarization = ['R', 'L']
        # Create a fake stream a bit like the VDIF one, but with complex data.
        bad_freq = np.array([[320, 320], [320, 320],
                             [336, 336], [336, 337]]) * u.MHz
        eh = EmptyStreamGenerator((10000, 4, 2), sample_rate=1.*u.Hz,
                                  start_time=Time('2018-01-01'),
                                  frequency=bad_freq, sideband=sideband)
        with pytest.raises(ValueError):
            Power(eh, polarization=polarization)
        bad_side = np.array([[-1, -1], [1, -1], [-1, -1], [1, 1]])
        eh = EmptyStreamGenerator((10000, 4, 2), sample_rate=1.*u.Hz,
                                  start_time=Time('2018-01-01'),
                                  frequency=frequency, sideband=bad_side)
        with pytest.raises(ValueError):
            Power(eh, polarization=polarization)
