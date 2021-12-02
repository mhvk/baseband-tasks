# Licensed under the GPLv3 - see LICENSE
"""Full-package tests of phaseutils sources."""

import pytest
import numpy as np
import os
import astropy.units as u
from astropy.time import Time
from astropy.tests.helper import assert_quantity_allclose

from baseband_tasks.phases import PolycoPhase, PintPhase, Phase
from .test_pint_toas import needs_pint


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
    @pytest.mark.parametrize(
        'index', [0, slice(None), np.array([[1, 2], [3, 4]])])
    def test_basics(self, index):
        # Compute phase and f0 for a single time
        time = self.times[index]
        phase = self.pu(time)
        f0 = self.pu.apparent_spin_freq(time)
        assert isinstance(phase, Phase)
        assert phase.shape == time.shape
        assert isinstance(f0, u.Quantity)
        assert f0.shape == time.shape


@needs_pint
class TestPintPhase(PhaseTest, PintBase):
    pass


class TestPolycoPhase(PhaseTest, PolycoBase):
    pass


@needs_pint
class TestPhaseComparison(PintBase, PolycoBase):
    def test_phase(self):
        # Compute PINT phase and f0
        pint_phase = self.pint_pu(self.times)
        pint_f0 = self.pint_pu.apparent_spin_freq(self.times)
        # Compute Polyco phase and f0
        polyco_phase = self.polyco_pu(self.times)
        polyco_f0 = self.polyco_pu.apparent_spin_freq(self.times)
        # compare phases
        diff_phase = pint_phase - polyco_phase
        # The test polyco file is made by tempo2 which has a constant phase
        # offset vs PINT.
        diff_phase -= diff_phase[0]
        assert_quantity_allclose(diff_phase, 0*u.cy, atol=1e-4*u.cy, rtol=0)
        # Frequency is for B1937+21, so this atol corresponds to rtol~5e-8.
        assert_quantity_allclose(pint_f0, polyco_f0, atol=30*u.uHz, rtol=0)


@needs_pint
class TestPintFrequencyBroadcasting(Base):
    def setup(self):
        super().setup()
        self.obs = 'ao'
        self.obs_freq = 1440.0
        self.par_file = os.path.join(test_data,
                                     'B1937+21_NANOGrav_11yv1.gls.par')

    def test_multiple_frequencies(self):
        # Regression test for gh-95
        freq = np.array([[1.4], [1.5]]) * u.GHz
        times = self.times
        pu = PintPhase(self.par_file, observatory=self.obs, frequency=freq)
        phases = pu(times)
        assert phases.shape == (2, 30)
        freq2 = freq.squeeze()
        times2 = self.times[:, np.newaxis]
        pu2 = PintPhase(self.par_file, observatory=self.obs, frequency=freq2)
        phases2 = pu2(times2)
        assert phases2.shape == (30, 2)
        assert np.all(phases2.T == phases)
