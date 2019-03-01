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
        self.times = self.start_time + np.arange(30) * 1.50 * u.min
        self.obs = 'AO'
        self.freq = 1.4 * u.GHz

    def test_time_input(self):
        pt = pint_utils.PintToas(self.obs, self.freq)
        toas = pt(self.times)
        assert toas.ntoas == len(self.times), \
            "TOAs did not input properly from timestamps"
        assert set(toas.get_obss()) == {'arecibo'}, \
            "Observatory did not input correctly."
        assert set(toas.get_freqs()) == {1400 * u.MHz}, \
            "Observing frequency did not input correctly."
        # Check control parameters passed on correctly.
        assert toas.ephem == 'jpl', \
            "Failed to initialize with default control parameters."
        assert toas.clock_corr_info['bipm_version'] == 'BIPM2015', \
            "Failed to initialize with default control parameters."
        assert toas.clock_corr_info['include_gps'], \
            "Failed to initialize with default control parameters."

    def test_different_control_param(self):
        # Test input control parameters from initializing;
        # Here, we pick the astropy/erfa built-in ephemeris, to avoid
        # downloading a(nother) big ephemeris file.  In principle,
        # this should not download anything, but PINT downloads IERS_B.
        pt = pint_utils.PintToas(self.obs, self.freq, ephem='builtin',
                                 include_bipm=False, include_gps=False)
        toas = pt(self.times)
        assert toas.ephem == 'builtin', \
            "Failed to initialize with input control parameters."
        assert not toas.clock_corr_info['include_bipm'], \
            "Failed to initialize with input control parameters."
        assert not toas.clock_corr_info['include_gps'], \
            "Failed to initialize with input control parameters."
