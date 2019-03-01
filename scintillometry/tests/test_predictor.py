# Licensed under the GPLv3 - see LICENSE
"""Full-package tests of predictor.py sources."""
import pytest
import numpy as np
import os
import astropy.units as u
from astropy.time import Time
from ..utils.predictor import Polyco


test_data = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')


class TestPredictor:
    def setup(self):
        self.start_time = Time('2018-05-06T23:00:00', format='isot',
                               scale='utc')
        self.polyco_file = os.path.join(test_data, 'B1937_polyco.dat')
        self.polyco = Polyco(self.polyco_file)

    def test_polyco_reading(self):
        assert len(self.polyco) == 48
        for l in self.polyco:
            assert len(l['coeff']) == l['ncoeff']

    def test_polyco_interpolation(self):
        # Test astropy time object input
        time = self.start_time + np.linspace(0, 100) * 10.0 * u.min
        # Test scalar input
        p = self.polyco(time[0])
        assert p.shape == ()
        # Test array input
        pa = self.polyco(time)
        assert len(pa) == len(time)
        assert pa.unit == u.cycle
        f0 = self.polyco(time, deriv=1)
        assert len(f0) == len(time)
        assert f0.unit == u.cycle / u.s
        f1 = self.polyco(time, deriv=2, time_unit=u.min)
        assert len(f1) == len(time)
        assert f1.unit == u.cycle/(u.min**2)
        pr0 = self.polyco(self.polyco[0]['mjd_mid'], rphase=0)
        assert pr0 == 0.0 * u.cycle, "`reference phase` is not setup right."
        pf = self.polyco(self.polyco['mjd_mid'], rphase='fraction')
        assert np.all(pf.value - self.polyco['rphase'] % 1.0  < 2e-5), \
            "`reference phase` is not setup right."

        # Test mjd input
        time2 = (self.start_time.mjd +
                 (np.linspace(0, 100) * 10.0 * u.min).to(u.day).value)
        p = self.polyco(time2[0])
        assert p.shape == ()
        # Test array input
        pa = self.polyco(time2)
        assert len(pa) == len(time2)

    def test_over_range(self):
        time = self.start_time + 10 * u.day
        with pytest.raises(ValueError):
            self.polyco(time)
