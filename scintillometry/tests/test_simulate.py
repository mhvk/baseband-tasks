# Licensed under the GPLv3 - see LICENSE
"""Full-package tests of simulating sources."""

import pytest
import numpy as np
import astropy.units as u
from astropy.time import Time

from ..generators import NoiseGenerator
from ..functions import ComplexFunctionTask, SquareTask


class TestModulation:
    """Basic tests of using a FunctionTask to create a modulator."""

    def block_profile(self, fh, data):
        """Multiply with 1 between 0.45--0.55 s, 0.125 otherwise."""
        t = (fh.tell() + fh.samples_per_frame / 2) / fh.sample_rate
        data *= (1 if abs(t - 0.5 * u.s) < 0.05 * u.s else 0.125)
        return data

    def setup(self):
        self.seed = 1234567
        self.start_time = Time('2010-11-12T13:14:15')
        self.sample_rate = 1. * u.kHz
        self.shape = (1000,)
        self.nh = NoiseGenerator(seed=self.seed,
                                 shape=self.shape, start_time=self.start_time,
                                 sample_rate=self.sample_rate,
                                 samples_per_frame=200, dtype=np.complex64)

    def test_modulator_unbinned(self):
        nh = self.nh
        mt = ComplexFunctionTask(nh, self.block_profile, samples_per_frame=1)
        nh.seek(0)
        n1 = nh.read(10)
        m1 = mt.read(10)
        assert np.all(m1 == n1 * 0.125)
        nh.seek(0.5 * u.s)
        mt.seek(0.5 * u.s)
        n2 = nh.read(10)
        m2 = mt.read(10)
        assert np.all(m2 == n2)
        nh.seek(0)
        mt.seek(0)
        n = nh.read()
        m = mt.read()
        assert n.shape == m.shape == self.shape
        assert np.all(m[:450] == 0.125 * n[:450])
        assert np.all(m[450:550] == n[450:550])
        assert np.all(m[550:] == 0.125 * n[550:])

    def test_modulator_binned(self):
        nh = self.nh
        mt = ComplexFunctionTask(nh, self.block_profile, samples_per_frame=50)
        nh.seek(0)
        n = nh.read()
        m = mt.read()
        assert np.all(m[:450] == 0.125 * n[:450])
        assert np.all(m[450:550] == n[450:550])
        assert np.all(m[550:] == 0.125 * n[550:])
        # Just to show one has to be careful: not giving
        # samples_per_frame takes it from nh, which is 200.
        mt = ComplexFunctionTask(nh, self.block_profile)
        m = mt.read()
        assert np.all(m[:400] == 0.125 * n[:400])
        assert np.all(m[400:600] == n[400:600])
        assert np.all(m[600:] == 0.125 * n[600:])
        # And more cases to show one has to be careful...
        mt = ComplexFunctionTask(nh, self.block_profile, samples_per_frame=500)
        m = mt.read()
        assert np.all(m == 0.125 * n)
        mt = ComplexFunctionTask(nh, self.block_profile, samples_per_frame=1000)
        m = mt.read()
        assert np.all(m == n)
