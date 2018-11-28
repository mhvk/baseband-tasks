# Licensed under the GPLv3 - see LICENSE

import pytest
import numpy as np
import astropy.units as u

from ..functions import Square, Power

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
        # Add frequency and sideband information by hand.
        # (Note: sideband is incorrect; just for testing purposes)
        fh.polarization = np.array(['L', 'R'])
        pt = Power(fh)
        assert np.all(pt.polarization == np.array(['LL', 'RR', 'LR', 'RL']))

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
