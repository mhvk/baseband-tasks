# Licensed under the GPLv3 - see LICENSE
"""Full-package tests of predictor.py sources."""
import pytest
import numpy as np
import os
import astropy.units as u
from astropy.time import Time

from baseband_tasks.phases import Polyco


test_data = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')


class TestPredictor:
    def setup(self):
        self.start_time = Time('2018-05-06T23:00:00', format='isot',
                               scale='utc')
        self.polyco_file = os.path.join(test_data, 'B1937_polyco.dat')
        self.polyco = Polyco(self.polyco_file)

    def test_polyco_reading(self):
        assert len(self.polyco) == 4
        for entry in self.polyco:
            assert len(entry['coeff']) == entry['ncoeff']

    def test_polyco_writing_roundtrip_tempo1(self, tmpdir):
        name = str(tmpdir.join('polyco.dat'))
        self.polyco.to_polyco(name, style='tempo1')
        with open(name, 'r') as fh:
            text = fh.readlines()
        with open(self.polyco_file) as fh:
            ref = fh.readlines()
        assert text == ref

    def test_polyco_writing_roundtrip_tempo2(self, tmpdir):
        name = str(tmpdir.join('polyco.dat'))
        polyco_file2 = os.path.join(test_data, 'B1957_polyco.dat')
        polyco2 = Polyco(polyco_file2)
        polyco2.to_polyco(name, style='tempo2')
        with open(name, 'r') as fh:
            text = fh.readlines()
        with open(polyco_file2) as fh:
            ref = fh.readlines()
        assert text == ref

    def test_polyco_interpolation(self):
        # Test astropy time object input
        time = self.start_time + np.linspace(0, 30) * 5. * u.min
        # Test scalar input
        p = self.polyco(time[0])
        assert p.shape == ()
        assert p > self.polyco['rphase'][0]
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
        fraction = self.polyco['rphase'] % (1 * u.cy)
        assert np.all(np.abs(pf - fraction) < 2e-5 * u.cy), \
            "`reference phase` is not setup right."

        pr = self.polyco(time, rphase='fraction')
        assert pr[0] < 1 * u.cy

        # Test mjd input
        p2 = self.polyco(time[0].mjd)
        assert p2.shape == ()
        assert u.allclose(p2, p)

        # Test array input
        pa2 = self.polyco(time.mjd)
        assert len(pa2) == len(time)
        assert u.allclose(pa2, pa)

    def test_polyco_interpolation_index_helper(self):
        # Test that for finely spaced times, a single item can get index.
        time = self.start_time + np.linspace(0, 30) * 5. * u.ms
        # Test scalar input
        p1 = self.polyco(time)
        p2 = self.polyco(time, index=time[15])
        assert np.all(p1 == p2)

    def test_over_range(self):
        time = self.start_time + 10 * u.day
        with pytest.raises(ValueError, match='outside of polyco range'):
            self.polyco(time)
