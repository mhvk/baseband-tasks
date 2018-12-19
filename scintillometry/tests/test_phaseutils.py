# Licensed under the GPLv3 - see LICENSE
"""Full-package tests of phaseutils sources."""

import pytest
import numpy as np
import os
import astropy.units as u
from astropy.time import Time

from ..utils.phase_utils import PolycoPhase, PintPhase

pytest.importorskip('scintillometry.utils.pint_utils')

test_data = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')


class TestPhaseBase:
    def setup(self):
        self.start_time = Time('2018-05-06T23:00:00', format='isot',
                               scale='utc')
        self.times = self.start_time + np.arange(1000) * 1.50 * u.min
        self.polyco_file = os.path.join(test_data, 'B1937_polyco.dat')
        self.par_file = os.path.join(test_data,
                                     'B1937+21_NANOGrav_11yv1.gls.par')
        self.obs = 'ao'
        self.obs_freq = 1440.0
        self.pint_pu = PintPhase(self.par_file, obs=self.obs,
                                 frequency=self.obs_freq)
        self.polyco_pu = PolycoPhase(self.polyco_file)


class TestPhase(TestPhaseBase):
    def test_one_phase(self):
        # Compute PINT phase and f0 for one time
        pint_phase_one = self.pint_pu(self.times[0])
        pint_f0_one = self.pint_pu.apparent_spin_freq(self.times[0])
        assert isinstance(pint_phase_one[0], u.Quantity), \
            "PINT phase did not return the expected list."
        assert isinstance(pint_phase_one[1], u.Quantity), \
            "PINT phase did not return the expected list."
        assert pint_phase_one[0].shape == (1,), \
            "PINT phase did not return the expected list length."
        assert pint_phase_one[1].shape == (1,), \
            "PINT phase did not return the expected list length."
        assert isinstance(pint_f0_one, u.Quantity), \
            "PINT f0 did not return the expected list."
        assert len(pint_f0_one) == 1, \
            "PINT f0 did not return the expected list length."
        # Compute Polyco phase and f0 for one time
        polyco_phase_one = self.polyco_pu(self.times[0])
        polyco_f0_one = self.polyco_pu.apparent_spin_freq(self.times[0])
        assert isinstance(polyco_phase_one[0], u.Quantity), \
            "Polyco phase did not return the expected Quantity."
        assert isinstance(polyco_phase_one[1], u.Quantity), \
            "Polyco phase did not return the expected Quantity."
        assert isinstance(polyco_f0_one, u.Quantity), \
            "Polyco f0 did not return the expected Quantity."

    def test_phase(self):
        # Compute PINT phase and f0
        pint_phase = self.pint_pu(self.times)
        pint_f0 = self.pint_pu.apparent_spin_freq(self.times)
        # Compute Polyco phase and f0
        polyco_phase = self.polyco_pu(self.times)
        polyco_f0 = self.polyco_pu.apparent_spin_freq(self.times)
        # compare phases
        diff_int = pint_phase[0] - polyco_phase[0]
        diff_frac = pint_phase[1] - polyco_phase[1]
        diff_total = diff_int + diff_frac
        diff_total = diff_total - diff_total.mean()
        assert np.all(diff_total < 1e-4 * u.cy), \
            "The phase difference between PINT and polyco is too big."

        diff_f0 = pint_f0 - polyco_f0
        assert np.all(diff_f0 < 2e-7 * u.Hz),  \
            "The apparent spin frequencyies do now match."
