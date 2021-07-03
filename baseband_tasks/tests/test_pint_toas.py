# Licensed under the GPLv3 - see LICENSE
"""Full-package tests of pint_toas sources."""
import pytest
import numpy as np
import os
import astropy.units as u
from astropy.time import Time

from baseband_tasks.phases import pint_toas


def needs_pint(func=None):
    try:
        import pint  # noqa
    except ImportError:
        skip = True
    else:
        skip = False

    skipif = pytest.mark.skipif(skip, reason='pint not available')
    return skipif(func) if func else skipif


pytestmark = needs_pint()

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
        pt = pint_toas.PintToas(self.obs, self.freq)
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
        # downloading a(nother) big ephemeris file.
        pt = pint_toas.PintToas(self.obs, self.freq, ephem='builtin',
                                include_bipm=False, include_gps=False)
        toas = pt(self.times)
        assert toas.ephem == 'builtin', \
            "Failed to initialize with input control parameters."
        assert not toas.clock_corr_info['include_bipm'], \
            "Failed to initialize with input control parameters."
        assert not toas.clock_corr_info['include_gps'], \
            "Failed to initialize with input control parameters."

    def test_multiple_frequencies(self):
        # Regression test for gh-95
        freq = np.array([[1.4], [1.5]]) * u.GHz
        pt = pint_toas.PintToas(self.obs, frequency=freq, ephem='builtin',
                                include_bipm=False, include_gps=False)
        toas = pt(self.times)
        assert toas.shape == (2, 30)
        freq = freq.squeeze()
        pt2 = pint_toas.PintToas(self.obs, frequency=freq, ephem='builtin',
                                 include_bipm=False, include_gps=False)
        toas2 = pt2(self.times[:, np.newaxis])
        assert toas2.shape == (30, 2)
