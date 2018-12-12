# Licensed under the GPLv3 - see LICENSE
"""Full-package tests of pint_utils sources."""

import pytest
import numpy as np
import os
import astropy.units as u
from astropy.time import Time

from ..phase_utils import PintPhase, PolycoPhase


test_data = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         'test_data')

class TestPhaseBase:
    def setup(self):
        self.start_time = Time('2018-05-05T12:00:00', format='isot',
                                scale='utc')
        self.times = self.start_time + np.arange(200) * 20.2048 * u.min
        self.polyco_file = os.path.join(test_data, 'B1937+21_58245.dat')
        self.par_file = os.path.join(test_data,
                                     'B1937+21_NANOGrav_11yv1.gls.par')
        self.obs = 'ao'
        self.obs_freq = 375.438
        self.pint_pu = PintPhase(self.par_file, obs=self.obs,
                                 obs_freq=self.obs_freq)
        self.polyco_pu = PolycoPhase(self.polyco_file)

class TestPhase(TestPhaseBase):

    def test_phase(self):
        # Compute PINT phase and f0
        pint_phase = self.pint_pu(self.times)
        pint_f0 = self.pint_pu.apparent_spin_freq(self.times)
        assert False
        # Compute PINT phase and f0 for one time
        pint_phase_one = self.pint_pu(self.times[0])
        pint_f0_one = self.pint_pu.apparent_spin_freq(self.times[0])
        assert False
        # Compute Polyco phase and f0
        polyco_phase = self.polyco_pu(self.times)
        polyco_f0 = self.polyco_phase.apparent_spin_freq(self.times)
        assert False
        # Compute Polyco phase and f0 for one time
        polyco_phase_one = self.polyco_pu(self.times[0])
        polyco_f0_one = self.polyco_phase.apparent_spin_freq(self.times[0])
        # test compare
        diff_int =  pint_phase[0] - polyco_phase[0]
        diff_frac = pint_phase[1] - polyco_phase[1]
        assert False
