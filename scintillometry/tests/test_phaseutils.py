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


class Base:
    def setup(self):
        self.start_time = Time('2018-05-06T23:00:00', format='isot',
                               scale='utc')
        self.times = self.start_time + np.arange(30) * 1.50 * u.min


class PolycoBase(Base):
    def setup(self):
        super().setup()
        self.polyco_file = os.path.join(test_data, 'B1937_polyco.dat')
        self.polyco_pu = PolycoPhase(self.polyco_file)
        self.pu = self.polyco_pu


class PintBase(Base):
    def setup(self):
        super().setup()
        self.obs = 'ao'
        self.obs_freq = 1440.0
        self.par_file = os.path.join(test_data,
                                     'B1937+21_NANOGrav_11yv1.gls.par')
        self.pint_pu = PintPhase(self.par_file, observatory=self.obs,
                                 frequency=self.obs_freq)
        self.pu = self.pint_pu


class PhaseTest:
    @pytest.mark.parametrize('index', (0, slice(None)))
    def test_basics(self, index):
        # Compute phase and f0 for a single time
        time = self.times[index]
        phase = self.pu(time)
        f0 = self.pu.apparent_spin_freq(time)
        assert isinstance(phase[0], u.Quantity)
        assert isinstance(phase[1], u.Quantity)
        assert phase[0].shape == time.shape
        assert phase[1].shape == time.shape
        assert isinstance(f0, u.Quantity)
        assert f0.shape == time.shape


class TestPintPhase(PhaseTest, PintBase):
    pass


class TestPolycoPhase(PhaseTest, PolycoBase):
    pass


class TestPhaseComparison(PintBase, PolycoBase):
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
