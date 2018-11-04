# Licensed under the GPLv3 - see LICENSE

import pytest
import numpy as np
import astropy.units as u

from ..functions import Square

from baseband import vdif, dada
from baseband.data import SAMPLE_VDIF, SAMPLE_DADA


class TestSquare:
    """Test getting simple intensities using Baseband's sample DADA file."""

    def test_squaretask(self):
        """Test squarer."""

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
        st = Square(fh)
        assert np.all(st.frequency == fh.frequency)
        assert np.all(st.sideband == st.sideband)

    def test_missing_frequency_sideband(self):
        fh = vdif.open(SAMPLE_VDIF)
        st = Square(fh)
        with pytest.raises(AttributeError):
            st.frequency
        with pytest.raises(AttributeError):
            st.sideband
