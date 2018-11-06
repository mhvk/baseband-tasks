# Licensed under the GPLv3 - see LICENSE
import pytest
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.tests.helper import assert_quantity_allclose

from ..dispersion import Disperse, Dedisperse, DispersionMeasure
from ..generators import StreamGenerator


REFERENCE_FREQUENCIES = (None,
                         300 * u.MHz,
                         300.064 * u.MHz,
                         299.936 * u.MHz,
                         300.128 * u.MHz,
                         299.872 * u.MHz)


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

    @pytest.mark.parametrize('reference_frequency', REFERENCE_FREQUENCIES)
    def test_disperse_reference_frequency(self, reference_frequency):
        disperse = Disperse(self.gp, self.dm,
                            reference_frequency=reference_frequency)
        assert (disperse.samples_per_frame == 32768 - 6400 or
                disperse.samples_per_frame == 32768 - 6401)
        offset = disperse.start_time - self.start_time
        # Start time kept if ref freq equal to lowest frequency.
        expected = self.dm.time_delay(299.936 * u.MHz,
                                      disperse.reference_frequency)
        assert abs(offset - expected) < 1./self.sample_rate

    @pytest.mark.parametrize('reference_frequency', REFERENCE_FREQUENCIES)
    def test_dedisperse_reference_frequency(self, reference_frequency):
        dedisperse = Dedisperse(self.gp, self.dm,
                                reference_frequency=reference_frequency)
        assert (dedisperse.samples_per_frame == 32768 - 6400 or
                dedisperse.samples_per_frame == 32768 - 6401)
        offset = dedisperse.start_time - self.start_time
        # Start time kept if ref freq equal to highest frequency.
        expected = -self.dm.time_delay(300.064 * u.MHz,
                                       dedisperse.reference_frequency)
        assert abs(offset - expected) < 1./self.sample_rate

    @pytest.mark.parametrize('reference_frequency', REFERENCE_FREQUENCIES)
    def test_disperse(self, reference_frequency):
        disperse = Disperse(self.gp, self.dm,
                            reference_frequency=reference_frequency)
        # Seek input time of the giant pulse, and read around it.
        disperse.seek(self.start_time + 0.5 * u.s)
        disperse.seek(-6400 * 5, 1)
        around_gp = disperse.read(6400 * 10)
        # Power in 20 bins of 0.025 s around the giant pulse.
        p = (np.abs(around_gp) ** 2).reshape(-1, 10, 320, 2).sum(2)
        # Note: FT leakage means that not everything outside of the dispersed
        # pulse is zero.  But the total power there is small.
        bin_off = int(np.round((
            self.dm.time_delay(300. * u.MHz, disperse.reference_frequency) *
            self.sample_rate / 3200).to(u.one)))
        power_bins = np.array([9, 10]) + bin_off
        assert np.all(p[:power_bins[0]].sum(1) < 0.005)
        assert np.all(p[power_bins[1]+1:].sum(1) < 0.005)
        assert np.all(p[power_bins[0]:power_bins[1]+1].sum() > 0.99)
        assert np.all(p[power_bins[0]:power_bins[1]+1] > 0.048)

    def test_disperse_negative_dm(self):
        disperse = Disperse(self.gp, -self.dm)
        disperse.seek(self.start_time + 0.5 * u.s)
        disperse.seek(-6400 * 5, 1)
        around_gp = disperse.read(6400 * 10)
        p = (np.abs(around_gp) ** 2).reshape(-1, 10, 320, 2).sum(2)
        # Note: FT leakage means that not everything outside of the dispersed
        # pulse is zero.  But the total power there is small.
        assert np.all(p[:9].sum(1) < 0.005)
        assert np.all(p[11:].sum(1) < 0.01)
        assert np.all(p[9:11].sum() > 0.99)
        assert np.all(p[9:11] > 0.048)

    @pytest.mark.parametrize('reference_frequency', REFERENCE_FREQUENCIES)
    def test_dedisperse(self, reference_frequency):
        disperse = Disperse(self.gp, self.dm,
                            reference_frequency=reference_frequency)
        dedisperse = Dedisperse(disperse, self.dm,
                                reference_frequency=reference_frequency)
        dedisperse.seek(self.start_time + 0.5 * u.s)
        dedisperse.seek(-6400 * 5, 1)
        data = dedisperse.read(6400 * 10)
        self.gp.seek(self.start_time + 0.5 * u.s)
        self.gp.seek(-6400 * 5, 1)
        expected = self.gp.read(6400 * 10)
        assert np.all(np.abs(data - expected) < 1e-3)
