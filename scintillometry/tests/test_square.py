# Licensed under the GPLv3 - see LICENSE

import numpy as np
import astropy.units as u

from ..square import SquareTask

from baseband import dada
from baseband.data import SAMPLE_DADA


class TestSquare(object):
    """Test getting simple intensities using Baseband's sample DADA file."""

    def test_squaretask(self):
        """Test squarer."""

        # Load baseband file and get reference intensities.
        fh = dada.open(SAMPLE_DADA)
        ref_data = fh.read()
        ref_data = np.real(ref_data * np.conj(ref_data))

        st = SquareTask(fh)

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
