# Licensed under the GPLv3 - see LICENSE
import pytest
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.tests.helper import assert_quantity_allclose

from ..dispersion import Disperse, Dedisperse, DispersionMeasure
from ..generators import StreamGenerator


class TestDispersion:

    def setup(self):
        self.start_time = Time('2010-11-12T13:14:15')
        self.sample_rate = 128. * u.kHz
        self.shape = (164000, 2)
        self.gp_sample = 64000
        self.gp = StreamGenerator(self.make_giant_pulse,
                                  shape=self.shape, start_time=self.start_time,
                                  sample_rate=self.sample_rate,
                                  samples_per_frame=1000, dtype=np.complex64,
                                  frequency=300*u.MHz,
                                  sideband=np.array((1, -1)))
        # Time delay of 0.05 s over 128 kHz band.
        self.dm = DispersionMeasure(1000.*0.05/0.039342251)

    def make_giant_pulse(self, sh):
        data = np.empty((sh.samples_per_frame,) + sh.shape[1:], sh.dtype)
        do_gp = (sh.tell() + np.arange(sh.samples_per_frame) ==
                 self.gp_sample)
        data[...] = do_gp[:, np.newaxis]
        return data

    def test_time_delay(self):
        time_delay = self.dm.time_delay(
            self.gp.frequency - self.sample_rate / 2.,
            self.gp.frequency + self.sample_rate / 2.)
        assert abs(time_delay - 0.05 * u.s) < 1. * u.ns

    def test_giant_pulse(self):
        data = self.gp.read()
        assert np.allclose(data, np.where(
            np.arange(data.shape[0])[:, np.newaxis] == self.gp_sample, 1., 0.))

    def test_disperse(self):
        disperse = Disperse(self.gp, self.dm)
        pad = int(np.ceil((0.05 * u.s * self.sample_rate).to_value(u.one)))
        assert disperse.samples_per_frame + pad == 32768
        disperse.seek(self.start_time + 0.5 * u.s)
        disperse.seek(-6400 * 5, 1)
        around_gp = disperse.read(6400 * 10)
        p = (np.abs(around_gp) ** 2).reshape(-1, 10, 640, 2).sum(2)
        # Note: FT leakage means that not everything outside of the dispersed
        # pulse is zero.  But the total power there is small.
        assert np.all(p[:5].sum(1) < 0.01) and np.all(p[6:].sum(1) < 0.01)
        assert np.all(p[5].sum(0) > 0.99)
        assert np.all(p[5] > 0.098)

    def test_disperse_negative_dm(self):
        disperse = Disperse(self.gp, -self.dm)
        pad = int(np.ceil((0.05 * u.s * self.sample_rate).to_value(u.one)))
        assert disperse.samples_per_frame + pad == 32768
        disperse.seek(self.start_time + 0.5 * u.s)
        disperse.seek(-6400 * 5, 1)
        around_gp = disperse.read(6400 * 10)
        p = (np.abs(around_gp) ** 2).reshape(-1, 10, 640, 2).sum(2)
        # Note: FT leakage means that not everything outside of the dispersed
        # pulse is zero.  But the total power there is small.
        assert np.all(p[:4].sum(1) < 0.01) and np.all(p[5:].sum(1) < 0.01)
        assert np.all(p[4].sum(0) > 0.99)
        assert np.all(p[4] > 0.098)

    def test_dedisperse(self):
        disperse = Disperse(self.gp, self.dm)
        dedisperse = Dedisperse(disperse, self.dm)
        dedisperse.seek(self.start_time + 0.5 * u.s)
        dedisperse.seek(-6400 * 5, 1)
        data = dedisperse.read(6400 * 10)
        self.gp.seek(self.start_time + 0.5 * u.s)
        self.gp.seek(-6400 * 5, 1)
        expected = self.gp.read(6400 * 10)
        assert np.all(np.abs(data - expected) < 1e-3)
