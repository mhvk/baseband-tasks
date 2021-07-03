# Licensed under the GPLv3 - see LICENSE
import pytest
import numpy as np
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose

from baseband_tasks.dm import DispersionMeasure


class TestDM:

    def setup(self):
        self.dm_val = 29.1168    # DM of B1957+20.

    def test_quantity(self):
        """Test `~astropy.unit.Quantity` creation and unit conversion."""
        dm = DispersionMeasure(self.dm_val)
        assert dm == self.dm_val * u.pc / u.cm**3
        dm_explicitunit = DispersionMeasure(self.dm_val, unit=dm.unit)
        assert dm == dm_explicitunit
        newunit = u.lyr / u.m**3
        dm_differentunit = DispersionMeasure(dm.to_value(newunit),
                                             unit=newunit)
        assert dm_differentunit == dm
        with pytest.raises(u.UnitsError) as excinfo:
            DispersionMeasure(self.dm_val * u.s)
        assert "require units equivalent to 'pc / cm3'" in str(excinfo.value)
        # Check that we can copy (and the new object is the same class).
        dm_copy = dm.copy()
        assert dm_copy == dm.copy()
        assert dm_copy.__class__ == dm.__class__

    def test_calculation(self):
        """Test time and phase offset calculation."""
        # Simple test with dm = 1, freq = 1 * u.MHz.
        dm = DispersionMeasure(1.)
        assert dm.time_delay(1. * u.MHz) == 1. / 2.41e-4 * u.s
        phase_delay = (1. / 2.41e-4) * (u.cycle * u.MHz * u.s)
        assert dm.phase_delay(1. * u.MHz) == phase_delay
        assert np.isclose(dm.phase_factor(1. * u.MHz),
                          np.exp(1j * phase_delay.to_value(u.rad)), rtol=1e-6)

        # Array of random frequencies within 64 MHz of the reference frequency.
        freqs = np.array([369.66462, 373.56482, 319.541562,
                          297.2516, 321.053234]) * u.MHz
        ref_freq = 321.582761 * u.MHz
        dm = DispersionMeasure(self.dm_val)

        time_delay = (dm.dispersion_delay_constant * dm
                      * (1. / freqs**2 - 1. / ref_freq**2))
        assert_quantity_allclose(dm.time_delay(freqs, ref_freq),
                                 time_delay, rtol=1e-13)
        time_delay_infref = dm.dispersion_delay_constant * dm / freqs**2
        assert_quantity_allclose(dm.time_delay(freqs),
                                 time_delay_infref, rtol=1e-13)

        phase_delay = (2. * np.pi * u.rad * dm.dispersion_delay_constant * dm
                       * freqs * (1. / ref_freq - 1. / freqs)**2)
        assert_quantity_allclose(dm.phase_delay(freqs, ref_freq),
                                 phase_delay, rtol=1e-13)
        phase_delay_infref = (2. * np.pi * u.rad
                              * dm.dispersion_delay_constant * dm * 1. / freqs)
        assert_quantity_allclose(dm.phase_delay(freqs),
                                 phase_delay_infref, rtol=1e-13)

        phase_factor = np.exp(1j * phase_delay.to_value(u.rad))
        # Not great rtol, since we used np.exp() and phase factors are large
        # numbers.
        assert_quantity_allclose(dm.phase_factor(freqs, ref_freq),
                                 phase_factor, rtol=1e-6)
        phase_factor_infref = np.exp(1j * phase_delay_infref.to_value(u.rad))
        assert_quantity_allclose(dm.phase_factor(freqs),
                                 phase_factor_infref, rtol=1e-6)
