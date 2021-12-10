# Licensed under the GPLv3 - see LICENSE
import pytest
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.tests.helper import assert_quantity_allclose

from baseband_tasks.fourier import fft_maker
from baseband_tasks.dispersion import (Disperse, Dedisperse, DispersionMeasure,
                                       DisperseSamples, DedisperseSamples)
from baseband_tasks.generators import StreamGenerator


REFERENCE_FREQUENCIES = (
    None,  # Default, will use mean
    300 * u.MHz,  # Centre frequency
    300.0123456789 * u.MHz,  # More random.
    300.064 * u.MHz,  # Upper edge
    299.936 * u.MHz,  # Lower edge
    300.128 * u.MHz,  # Above upper edge
    300.123456789 * u.MHz,  # More random, above upper edge
    299.872 * u.MHz)  # Below lower edge


class GiantPulseSetup:
    def setup_class(self):
        self.start_time = Time('2010-11-12T13:14:15')
        self.sample_rate = 128. * u.kHz
        self.shape = (164000, 2)
        self.gp_sample = 64000
        # Complex timestream
        self.gp = StreamGenerator(self.make_giant_pulse,
                                  shape=self.shape, start_time=self.start_time,
                                  sample_rate=self.sample_rate,
                                  samples_per_frame=1000, dtype=np.complex64,
                                  frequency=300*u.MHz,
                                  sideband=np.array((1, -1)))
        # Time delay of 0.05 s over 128 kHz band.
        self.dm = DispersionMeasure(1000.*0.05/0.039342251)

    @classmethod
    def make_giant_pulse(self, sh):
        data = np.empty((sh.samples_per_frame,) + sh.shape[1:], sh.dtype)
        do_gp = sh.tell() + np.arange(sh.samples_per_frame) == self.gp_sample
        data[...] = do_gp[:, np.newaxis]
        return data


class TestDispersion(GiantPulseSetup):

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
    def test_disperse_samples_per_frame(self, reference_frequency):
        disperse = Disperse(self.gp, self.dm,
                            reference_frequency=reference_frequency)
        assert (disperse.samples_per_frame == 32768 - 6400
                or disperse.samples_per_frame == 32768 - 6401)

    @pytest.mark.parametrize('reference_frequency', REFERENCE_FREQUENCIES)
    def test_disperse_time_offset(self, reference_frequency):
        disperse = Disperse(self.gp, self.dm,
                            reference_frequency=reference_frequency)
        offset = disperse.start_time - self.start_time
        # Start time kept if ref freq equal to lowest frequency.
        expected = self.dm.time_delay(299.936 * u.MHz,
                                      disperse.reference_frequency)
        assert abs(offset - expected) < 1. / self.sample_rate

    @pytest.mark.parametrize('reference_frequency', REFERENCE_FREQUENCIES)
    def test_disperse(self, reference_frequency):
        disperse = Disperse(self.gp, self.dm,
                            reference_frequency=reference_frequency)
        # Seek input time of the giant pulse, corrected to the reference
        # frequency, and read around it.
        t_gp = (self.start_time + self.gp_sample / self.sample_rate
                + self.dm.time_delay(300. * u.MHz,
                                     disperse.reference_frequency))
        disperse.seek(t_gp)
        disperse.seek(-self.gp_sample // 2, 1)
        around_gp = disperse.read(self.gp_sample)
        # Power in 20 bins of 0.025 s around the giant pulse.
        p = (np.abs(around_gp) ** 2).reshape(
            -1, 10, self.gp_sample // 20 // 10, 2).sum(2)
        # Note: FT leakage means that not everything outside of the dispersed
        # pulse is zero.  But the total power there is small.
        assert np.all(p[:9].sum(1) < 0.005)
        assert np.all(p[11:].sum(1) < 0.005)
        assert np.all(p[9:11].sum() > 0.99)
        assert np.all(p[9:11] > 0.047)

    @pytest.mark.parametrize('reference_frequency', REFERENCE_FREQUENCIES)
    def test_disperse_roundtrip1(self, reference_frequency):
        self.gp.seek(self.start_time + 0.5 * u.s)
        self.gp.seek(-1024, 1)
        gp = self.gp.read(2048)
        # Set up dispersion as above, and check that one can invert
        disperse = Disperse(self.gp, self.dm,
                            reference_frequency=reference_frequency)
        dedisperse = Dedisperse(disperse, self.dm,
                                reference_frequency=reference_frequency)
        dedisperse.seek(self.start_time + self.gp_sample / self.sample_rate)
        dedisperse.seek(-1024, 1)
        gp_dd = dedisperse.read(2048)
        # Note: rounding errors mean this doesn't work perfectly.
        assert np.all(np.abs(gp_dd - gp) < 1.e-4)

    @pytest.mark.parametrize('reference_frequency', REFERENCE_FREQUENCIES)
    def test_disperse_roundtrip2(self, reference_frequency):
        # Now check dedispersing using mean frequency, which means that
        # the giant pulse should still be at the dispersed t_gp, i.e., there
        # should be a net time shift as well as a phase shift.
        disperse = Disperse(self.gp, self.dm,
                            reference_frequency=reference_frequency)
        time_delay = self.dm.time_delay(300. * u.MHz,
                                        disperse.reference_frequency)
        # The difference in phase delay between dispersion and dedispersion is:
        # phase_delay(freq, 300MHz) - phase_delay(freq, reference_frequency)
        # This yields a time shift as well as a phase shift given by:
        d = self.dm.dispersion_delay_constant * self.dm * u.cycle
        phase_delay = -2. * d * (1./(300. * u.MHz)
                                 - 1./disperse.reference_frequency)
        # Sanity check of analytical derivation.
        assert_quantity_allclose(
            - phase_delay - time_delay * 300 * u.MHz * u.cycle,
            self.dm.phase_delay(300 * u.MHz, disperse.reference_frequency),
            atol=0.001 * u.cycle)

        # Seek input time of the giant pulse, corrected to the reference
        # frequency, and read around it.
        t_gp = (self.start_time + self.gp_sample / self.sample_rate
                + time_delay)
        # Dedisperse to mean frequency = 300 MHz, and read dedispersed pulse.
        dedisperse = Dedisperse(disperse, self.dm)
        dedisperse.seek(t_gp)
        dedisperse.seek(-1024, 1)
        dd_gp = dedisperse.read(2048)
        # First check power is concentrated where it should be.
        p = np.abs(dd_gp) ** 2
        # TODO: why is real data not just 2?
        half_size = 1 if self.gp.complex_data else 3
        assert np.all(p[1024-half_size:1024+half_size+1].sum(0) > 0.9)
        # Now check that, effectively, we just shifted the giant pulse.
        # Read the original giant pulse
        self.gp.seek(0)
        gp = self.gp.read()
        # Shift in time using a phase gradient in the Fourier domain
        # (plus the phase offset between new and old reference frequency).
        fft = fft_maker(shape=gp.shape, dtype=gp.dtype,
                        sample_rate=self.sample_rate)
        ifft = fft.inverse()
        ft = fft(gp)
        freqs = self.gp.frequency + fft.frequency * self.gp.sideband
        phases = time_delay * freqs * u.cycle + phase_delay
        phases *= self.gp.sideband
        ft *= np.exp(-1j * phases.to_value(u.rad))
        gp_exp = ifft(ft)
        offset = self.gp_sample + int(
            (time_delay * self.sample_rate).to(u.one).round())
        assert np.all(np.abs(gp_exp[offset-1024:offset+1024] - dd_gp) < 1e-3)

    def test_disperse_negative_dm(self):
        disperse = Disperse(self.gp, -self.dm)
        disperse.seek(self.start_time + self.gp_sample / self.sample_rate)
        disperse.seek(-self.gp_sample // 2, 1)
        around_gp = disperse.read(self.gp_sample)
        p = (np.abs(around_gp) ** 2).reshape(
            -1, 10, self.gp_sample // 10 // 20, 2).sum(2)
        # Note: FT leakage means that not everything outside of the dispersed
        # pulse is zero.  But the total power there is small.
        assert np.all(p[:9].sum(1) < 0.01)
        assert np.all(p[11:].sum(1) < 0.01)
        assert np.all(p[9:11].sum() > 0.99)
        assert np.all(p[9:11] > 0.047)

    def test_disperse_closing(self):
        # This tests implementation, so can be removed if the implementation
        # changes. It is meant to ensure memory is released upon closing.
        disperse = Disperse(self.gp, -self.dm)
        assert 'phase_factor' not in disperse.__dict__
        disperse.read(1)
        assert 'phase_factor' in disperse.__dict__
        disperse.close()
        assert 'phase_factor' not in disperse.__dict__


class GiantPulseSetupReal(GiantPulseSetup):
    def setup_class(self):
        self.start_time = Time('2010-11-12T13:14:15')
        self.sample_rate = 256. * u.kHz
        self.shape = (328000, 2)
        self.gp_sample = 128000
        # Real timestream; mean frequecies of the two bands are the same.
        self.gp = StreamGenerator(self.make_giant_pulse,
                                  shape=self.shape,
                                  start_time=self.start_time,
                                  sample_rate=self.sample_rate,
                                  samples_per_frame=1000,
                                  dtype=np.float32,
                                  frequency=[299.936, 300.064]*u.MHz,
                                  sideband=np.array((1, -1)))
        # Time delay of 0.05 s over 128 kHz band.
        self.dm = DispersionMeasure(1000.*0.05/0.039342251)


class TestDispersionReal(TestDispersion, GiantPulseSetupReal):
    # Override tests that do not simply work for the real data,
    # since the sample rate is twice as high.
    def test_time_delay(self):
        time_delay = self.dm.time_delay(
            self.gp.frequency.mean() - self.sample_rate / 4.,
            self.gp.frequency.mean() + self.sample_rate / 4.)
        assert abs(time_delay - 0.05 * u.s) < 1. * u.ns

    @pytest.mark.parametrize('reference_frequency', REFERENCE_FREQUENCIES)
    def test_disperse_samples_per_frame(self, reference_frequency):
        disperse = Disperse(self.gp, self.dm,
                            reference_frequency=reference_frequency)
        assert (disperse.samples_per_frame == 65536 - 12800
                or disperse.samples_per_frame == 65536 - 12801)


class TestDispersionRealDisjoint(TestDispersion):
    def setup(self):
        self.start_time = Time('2010-11-12T13:14:15')
        self.sample_rate = 128. * u.kHz
        self.shape = (164000, 2)
        self.gp_sample = 64000
        # Real timestream; mean frequecies of the two bands are the same.
        self.gp = StreamGenerator(self.make_giant_pulse,
                                  shape=self.shape,
                                  start_time=self.start_time,
                                  sample_rate=self.sample_rate,
                                  samples_per_frame=1000,
                                  dtype=np.float32,
                                  frequency=300.*u.MHz,
                                  sideband=np.array((1, -1)))
        # Time delay of 0.05 s over 128 kHz band.
        self.dm = DispersionMeasure(1000.*0.05/0.039342251)

    def make_giant_pulse(self, sh):
        data = np.empty((sh.samples_per_frame,) + sh.shape[1:], sh.dtype)
        do_gp = sh.tell() + np.arange(sh.samples_per_frame) == self.gp_sample
        data[...] = do_gp[:, np.newaxis]
        return data

    # Override two tests that are different for contiguous bands.
    def test_disperse(self):
        gp = StreamGenerator(self.make_giant_pulse,
                             shape=self.shape,
                             start_time=self.start_time,
                             sample_rate=self.sample_rate,
                             samples_per_frame=1000,
                             dtype=np.float32,
                             frequency=300.*u.MHz,
                             sideband=np.array((1, -1)))
        disperse = Disperse(gp, self.dm)
        assert_quantity_allclose(disperse.reference_frequency,
                                 300. * u.MHz)
        disperse.seek(self.start_time + self.gp_sample / self.sample_rate)
        disperse.seek(-self.gp_sample // 2, 1)
        around_gp = disperse.read(self.gp_sample)
        assert around_gp.dtype == np.float32
        p = (around_gp ** 2).reshape(-1, self.gp_sample // 20, 2).sum(1)
        # Note: FT leakage means that not everything outside of the dispersed
        # pulse is zero.  But the total power there is small.
        assert np.all(p[:9] < 0.006)
        assert np.all(p[11:] < 0.006)
        # Lower sideband [1] has lower frequencies and thus is dispersed
        # to later.
        assert p[9, 0] > 0.99 and p[10, 0] < 0.006
        assert p[10, 1] > 0.99 and p[9, 1] < 0.006

    def test_disperse_negative_dm(self):
        disperse = Disperse(self.gp, -self.dm)
        disperse.seek(self.start_time + self.gp_sample / self.sample_rate)
        disperse.seek(-self.gp_sample // 2, 1)
        around_gp = disperse.read(self.gp_sample)
        p = (around_gp ** 2).reshape(-1, self.gp_sample // 20, 2).sum(1)
        # Note: FT leakage means that not everything outside of the dispersed
        # pulse is zero.  But the total power there is small.
        assert np.all(p[:9].sum(1) < 0.006)
        assert np.all(p[11:].sum(1) < 0.006)
        # Lower sideband [1] is dedispersed to earlier.
        assert p[10, 0] > 0.99 and p[9, 0] < 0.006
        assert p[9, 1] > 0.99 and p[10, 1] < 0.006


class TestDispersSample(GiantPulseSetupReal):

    @pytest.mark.parametrize('reference_frequency', REFERENCE_FREQUENCIES)
    @pytest.mark.parametrize('frequency, sideband', [
        (None, None),
        ([199.936, 200.064]*u.MHz, np.array((1, -1))),  # Far off from normal.
        ([200.064, 199.936]*u.MHz, np.array((-1, -1)))])
    def test_disperse_sample(self, reference_frequency, frequency, sideband):
        disperse = DisperseSamples(self.gp, self.dm, frequency=frequency,
                                   reference_frequency=reference_frequency,
                                   sideband=sideband)

        # Seek input time of the giant pulse, corrected to the reference
        # frequency, and read around it.
        center_frequency = (disperse.frequency
                            + disperse.sideband * disperse.sample_rate / 2.)
        time_delay = self.dm.time_delay(center_frequency,
                                        disperse.reference_frequency)
        t_gp = (self.start_time + self.gp_sample / self.sample_rate
                + time_delay)

        disperse.seek(t_gp.min())
        around_gp = disperse.read(self.gp_sample)

        sample_shift_diff = ((time_delay.max() - time_delay.min())
                             * self.sample_rate)
        sample_shift_diff = np.round(sample_shift_diff.to(u.one)).astype(int)
        expected = np.zeros_like(around_gp)
        expected[0, t_gp.argmin()] = 1.0
        expected[sample_shift_diff, t_gp.argmax()] = 1.0
        assert np.all(around_gp == expected)

    @pytest.mark.parametrize('reference_frequency', [200 * u.MHz, 300 * u.MHz])
    def test_disperse_roundtrip1(self, reference_frequency):
        self.gp.seek(self.start_time + self.gp_sample / self.sample_rate)
        self.gp.seek(-1024, 1)
        gp = self.gp.read(2048)
        # Set up dispersion as above, and check that one can invert
        disperse = DisperseSamples(self.gp, self.dm,
                                   reference_frequency=reference_frequency)
        dedisperse = DedisperseSamples(disperse, self.dm,
                                       reference_frequency=reference_frequency)
        assert dedisperse.dm == self.dm
        assert dedisperse._dm == -self.dm
        dedisperse.seek(self.start_time + self.gp_sample / self.sample_rate)
        dedisperse.seek(-1024, 1)
        gp_dd = dedisperse.read(2048)
        # Note: rounding errors mean this doesn't work perfectly.
        assert np.any(gp_dd == 1.0)
        assert np.all(gp_dd == gp)
