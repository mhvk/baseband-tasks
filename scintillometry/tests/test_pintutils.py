# Licensed under the GPLv3 - see LICENSE
"""Full-package tests of pint_utils sources."""
import pytest
import numpy as np
import os
import astropy.units as u
from astropy.time import Time
pint_utils = pytest.importorskip('scintillometry.utils.pint_utils')


test_data = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')


class TestPintUtils:
    """Test the utilities of PINT"""

    def setup(self):
        self.start_time = Time('2018-05-06T23:00:00', format='isot',
                               scale='utc')
        self.times = self.start_time + np.arange(1000) * 1.50 * u.min
        self.obs = 'AO'
        self.freq = 1400

    def test_make_toalist(self):
        toalist = pint_utils.make_toa_list(self.times, self.obs, self.freq)
        assert len(toalist) == 1000

    def test_time_input(self):
        pt = pint_utils.PintToas()
        toalist = pint_utils.make_toa_list(self.times, self.obs, self.freq)
        toas = pt.make_toas(toa_list=toalist)
        assert toas.ntoas == 1000, \
            "TOAs did not input properly from timestamps"
        assert set(toas.get_obss()) == {'arecibo'}, \
            "Observatory did not input correctly."
        assert set(toas.get_freqs()) == {1400 * u.MHz}, \
            "Observing frequency did not input correctly."

    def test_control_param(self):
        # Test from default control param values
        pt = pint_utils.PintToas()
        toalist = pint_utils.make_toa_list(self.times, self.obs, self.freq)
        toas = pt.make_toas(toa_list=toalist)
        assert toas.ephem == 'de436', \
            "Failed to initialize with default control parameters."
        assert toas.clock_corr_info['bipm_version'] == 'BIPM2015', \
            "Failed to initialize with default control parameters."

        # Test input control parameters from initializing
        pt2 = pint_utils.PintToas(ephem='de421', include_gps=False)
        toalist2 = pint_utils.make_toa_list(self.times, self.obs, self.freq)
        toas2 = pt2.make_toas(toa_list=toalist2)
        assert toas2.ephem == 'de421', \
            "Fail to initialize with input control parameters."
        assert not toas2.clock_corr_info['include_gps'], \
            "Fail to initialize with input control parameters."
