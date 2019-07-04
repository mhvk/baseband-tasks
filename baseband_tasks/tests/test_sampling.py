#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test resampling."""

import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.time import Time

from baseband_tasks.base import Task, SetAttribute
from baseband_tasks.channelize import Channelize
from baseband_tasks.combining import Stack
from baseband_tasks.sampling import (
    Resample, seek_float, TimeDelay, DelayAndResample)
from baseband_tasks.generators import (
    EmptyStreamGenerator, StreamGenerator, Noise)


class PureTone:
    def __init__(self, frequency, start_time, phi0=0.):
        self.frequency = frequency
        self.start_time = start_time
        self.phi0 = phi0

    @staticmethod
    def pure_tone(phi, dtype):
        if dtype.kind == 'f':
            cosine = np.cos(phi)
        else:
            cosine = np.exp(1j * phi)
        return cosine.astype(dtype, copy=False)

    def __call__(self, ih, dtype=None):
        dt = ((ih.time - self.start_time).to(u.s)
              + np.arange(ih.samples_per_frame) / ih.sample_rate)
        dt = dt.reshape((-1,) + (1,) * len(ih.sample_shape))
        phi = self.phi0 + self.frequency * dt * u.cycle
        return self.pure_tone(phi.to_value(u.rad), dtype or ih.dtype)


class TestFloatOffset:
    def setup(self):
        self.ih = EmptyStreamGenerator(shape=(2048, 3, 2),
                                       sample_rate=1.*u.kHz,
                                       start_time=Time('2010-11-12T13:14:15'))

    @pytest.mark.parametrize('offset',
                             (0., 1., 10.5,
                              10.*u.ms, 0.015*u.s,
                              Time('2010-11-12T13:14:15.013'),
                              [1.75, 10.5],
                              np.linspace(1, 10, 6).reshape(3, 2) * u.ms,
                              Time(['2010-11-12T13:14:15.013',
                                    '2010-11-12T13:14:15.0135'])))
    def test_seek_float(self, offset):
        floats = seek_float(self.ih, offset)

        if isinstance(offset, Time):
            offset = (offset - self.ih.start_time).to(u.s)
        if isinstance(offset, u.Quantity):
            offset = (offset * self.ih.sample_rate).to_value(1)

        assert_allclose(floats, offset)

    def test_invalid_seek_float(self):
        with pytest.raises(TypeError):
            seek_float(self.ih, object())
        with pytest.raises(u.UnitsError):
            seek_float(self.ih, 1.*u.m)
        with pytest.raises(ValueError):
            seek_float(self.ih, [1, 2, 3]*u.s)


class TestResampleReal:

    dtype = np.dtype('f4')
    atol = 1e-4
    sample_rate = 1 * u.kHz
    samples_per_frame = 1024
    start_time = Time('2010-11-12T13:14:15')
    frequency = 400. * u.kHz
    sideband = np.array([-1, 1])
    shape = (2048,) + sideband.shape

    def setup(self):
        f_signal = self.sample_rate / 32 * np.ones(self.shape[1:])

        cosine = PureTone(f_signal, self.start_time)

        self.full_fh = StreamGenerator(
            cosine, shape=self.shape,
            sample_rate=self.sample_rate,
            samples_per_frame=self.samples_per_frame,
            frequency=self.frequency, sideband=self.sideband,
            start_time=self.start_time, dtype=self.dtype)

        self.part_fh = StreamGenerator(
            cosine, shape=(self.shape[0] // 4,) + self.shape[1:],
            sample_rate=self.sample_rate / 4,
            samples_per_frame=self.samples_per_frame // 4,
            frequency=self.frequency, sideband=self.sideband,
            start_time=self.start_time, dtype=self.dtype)

    def test_setup(self):
        full = self.full_fh.read()
        part = self.part_fh.read()
        assert np.all(part == full[::4])

    @pytest.mark.parametrize('offset',
                             (0., 0.25, 0.5, 1., 1.75, 10.5,
                              10.*u.ms, 0.015*u.s,
                              Time('2010-11-12T13:14:15.013')))
    def test_resample(self, offset):
        ih = Resample(self.part_fh, offset, samples_per_frame=511)
        # Always lose 1 sample per frame.
        assert ih.shape == ((self.part_fh.shape[0] - 1,)
                            + self.part_fh.sample_shape)
        # Check we are at the given offset.
        if isinstance(offset, Time):
            expected_time = offset
        elif isinstance(offset, u.Quantity):
            expected_time = self.part_fh.start_time + offset
        else:
            expected_time = (self.part_fh.start_time
                             + offset / self.part_fh.sample_rate)
        assert abs(ih.time - expected_time) < 1. * u.ns

        ioffset, fraction = divmod(seek_float(self.part_fh, offset), 1)
        assert ih.offset == ioffset
        expected_start_time = (self.part_fh.start_time
                               + fraction / self.part_fh.sample_rate)
        assert abs(ih.start_time - expected_start_time) < 1. * u.ns
        ih.seek(0)
        data = ih.read()
        expected = self.full_fh.read()[int(fraction*4):-(4-int(fraction*4)):4]
        assert_allclose(data, expected, atol=self.atol, rtol=0)

    def test_repr(self):
        ih = Resample(self.part_fh, 0.5, samples_per_frame=511)
        r = repr(ih)
        assert r.startswith('Resample(ih')
        assert 'offset=0.5' in r

    @pytest.mark.parametrize('offset',
                             ([0., 0.25], [1.75, 10.5],
                              [10., 12.5]*u.ms,
                              Time(['2010-11-12T13:14:15.013',
                                    '2010-11-12T13:14:15.0135'])))
    def test_resample_different_offset(self, offset):
        # ih = Resample(self.part_fh, offset, samples_per_frame=512)
        pass


class TestResampleComplex(TestResampleReal):

    dtype = np.dtype('c16')
    atol = 1e-8


class StreamArray(StreamGenerator):
    def __init__(self, data, *args, **kwargs):
        def from_data(handle):
            return data[handle.offset:
                        handle.offset+handle.samples_per_frame]
        super().__init__(from_data, *args, **kwargs)


class TestResampleNoise(TestResampleComplex):

    dtype = np.dtype('c8')
    atol = 1e-4

    def setup(self):
        # Make noise with only frequencies covered by part.
        part_ft_noise = (np.random.normal(size=512*2*2)
                         .view('c16').reshape(-1, 2))
        # Make corresponding FT for full frame.
        full_ft_noise = np.concatenate((part_ft_noise[:256],
                                        np.zeros((512*3, 2), 'c16'),
                                        part_ft_noise[-256:]), axis=0)
        part_data = np.fft.ifft(part_ft_noise, axis=0)
        # Factor 2048/512 to ensure data have same power.
        full_data = np.fft.ifft(full_ft_noise * 2048 / 512, axis=0)

        self.full_fh = StreamArray(
            full_data, shape=self.shape,
            sample_rate=self.sample_rate,
            samples_per_frame=self.samples_per_frame,
            frequency=self.frequency, sideband=self.sideband,
            start_time=self.start_time, dtype=self.dtype)

        self.part_fh = StreamArray(
            part_data, shape=(self.shape[0] // 4,) + self.shape[1:],
            sample_rate=self.sample_rate / 4,
            samples_per_frame=self.samples_per_frame // 4,
            frequency=self.frequency, sideband=self.sideband,
            start_time=self.start_time, dtype=self.dtype)


class BaseDelayAndResampleTestsReal:
    """Base class for DelayAndResample tests.

    Sub-classes need to have a ``setup()`` that defines ``self.raw``,
    which is a simulated voltage baseband stream that can will mixed,
    low-pass filtered and downsampled

    The idea behind all tests is to similate a voltage stream and
    the whole receiver chain, i.e., mix it with an IF, low-pass filter
    it, and then detect.  The mixing can be with real or complex (quadrature).
    """
    dtype = np.dtype('f4')  # type of mixing and output data.
    atol = atol_channelized = 1.e-4  # tolerance per sample
    full_sample_rate = 204.8 * u.kHz  # For the real-valued input signal
    samples_per_frame = 1024
    start_time = Time('2010-11-12T13:14:15')
    lo = full_sample_rate * 7 / 16  # IF frequency.
    sideband = np.array([-1, 1])    # IF sideband
    n_frames = 16
    phi0_mixer = -12.3456789 * u.degree

    def setup(self):
        self.full_shape = ((self.samples_per_frame * self.n_frames,)
                           + self.sideband.shape)
        self.downsample = (16 if self.dtype.kind == 'c' else 8)
        self.sample_rate = self.full_sample_rate / self.downsample
        # Create the IF (which can produce a complex tone for quadrature).
        self.mixer = PureTone(self.lo, self.start_time,
                              self.phi0_mixer)
        # Create a real-valued stream with a test-specific signal.
        self.raw = StreamGenerator(
            self.signal, shape=self.full_shape, start_time=self.start_time,
            sample_rate=self.full_sample_rate, dtype=np.dtype('f4'),
            samples_per_frame=self.samples_per_frame)

    def mix_downsample(self, ih, data):
        """Mix, low-pass filter, and downsample."""
        if ih.complex_data:
            # Get quadrature mix: data * cos, data * +/-sin
            # (+/- for lower/upper sideband; note data are always real).
            mixed = data * self.mixer(ih.ih, dtype=ih.dtype)
            np.conjugate(mixed, out=mixed, where=(ih.sideband > 0))
            # Convert to real to simulate mixing more properly.
            mixed = mixed.view(data.dtype).reshape(data.shape + (2,))
        else:
            # For real data, need to filter out the wrong sideband first.
            ft = np.fft.rfft(data, axis=0)
            i_lo = int(round((ft.shape[0] * self.lo
                              / (ih.ih.sample_rate / 2)).to_value(1)))
            # for lower/upper sideband, keep only lower/upper part
            ft[:i_lo] *= (ih.sideband < 0)
            ft[i_lo:] *= (ih.sideband > 0)
            data = np.fft.irfft(ft, axis=0)
            # And then mix.
            mixed = data * self.mixer(ih.ih)

        # The mixer produces signals at (f+fmix) and (f-fmix).
        # Apply a low-pass filter to remove (f+fmix), leaving (f-fmix).
        ft = np.fft.rfft(mixed, axis=0)
        ft[ft.shape[0]//self.downsample:] = 0
        # Account for half of signal lost with (f+fmix) removal.
        ft *= 2.
        # Turn back into data stream.
        filtered = np.fft.irfft(ft, axis=0).astype(data.dtype)
        # Downsample.
        filtered = filtered[::self.downsample]
        # Turn into complex quadrature if needed.
        if ih.complex_data:
            return filtered[..., 0] + 1j * filtered[..., 1]
        else:
            return filtered

    def get_tel(self, delay=None, n=None):
        """Get signal as observed at a telescope with the given delay."""
        if delay is None:
            fh = self.raw
        else:
            delay_time = delay / self.raw.sample_rate
            fh = SetAttribute(self.raw, start_time=self.start_time-delay_time)
        # Observe the raw, possibly delayed samples, using mix_downsample.
        obs = Task(fh, self.mix_downsample, dtype=self.dtype,
                   sample_rate=self.sample_rate,
                   frequency=self.lo, sideband=self.sideband)
        if n is None:
            return obs
        else:
            return Channelize(obs, n)

    def assert_tel_same(self, tel1, tel2, atol=None):
        if atol is None:
            atol = self.atol
        # Read both telescopes together.
        both_tel = Stack((tel1, tel2), axis=1)
        # Sanity check that we are actually comparing anything
        assert both_tel.size > 500
        # Compare the data.
        data = both_tel.read()
        assert_allclose(data[:, 0], data[:, 1], atol=atol, rtol=0)

    @pytest.mark.parametrize('delay', (-18.25, -np.pi, -8, 0.1, 65.4321))
    @pytest.mark.parametrize('n', (None, 32))
    def test_resample_delayed(self, delay, n):
        """Create delayed and non-delayed versions; check we can undo delay."""
        # delay is in units of raw samples.  We possibly channelize.
        tel1 = self.get_tel(delay=None, n=n)
        tel2 = self.get_tel(delay=delay, n=n)
        # Undo the delay and ensure we resample such that we're on the same
        # time grid as the undelayed telescope.
        if n is None:
            tel2_rs = DelayAndResample(tel2, delay / self.full_sample_rate,
                                       tel1.start_time, lo=self.lo)
            self.assert_tel_same(tel1, tel2_rs)
        else:
            # For channelized data, we have to ensure we pass in an explicit
            # local oscillator frequency.
            tel2_rs = DelayAndResample(tel2, delay / self.full_sample_rate,
                                       tel1.start_time, lo=self.lo,
                                       samples_per_frame=1)
            self.assert_tel_same(tel1, tel2_rs, atol=self.atol_channelized)


class BaseDelayAndResampleTestsComplex(BaseDelayAndResampleTestsReal):
    """Base tests for complex signals.

    These are like for the real ones, except that we can also test the
    TimeDelay class (which only works on complex data).
    """
    dtype = np.dtype('c8')

    @pytest.mark.parametrize('delay', (-8, 16))
    def test_time_delay(self, delay):
        # Pure time delays must be in units of telescope samples,
        # otherwise the start times no longer line up. This also means
        # it is useless to separately check channelized: phases line up
        # anyway.
        tel1 = self.get_tel(delay=None)
        tel2 = self.get_tel(delay=delay)
        time_delay = TimeDelay(tel2, delay / self.full_sample_rate,
                               lo=self.lo)
        self.assert_tel_same(tel1, time_delay)

    @pytest.mark.parametrize('delay', (-1, 15.4321))
    @pytest.mark.parametrize('n', (None, 32))
    def test_time_delay_align(self, delay, n):
        tel1 = self.get_tel(delay=None, n=n)
        tel2 = self.get_tel(delay=delay, n=n)
        time_delay = TimeDelay(tel2, delay / self.full_sample_rate,
                               lo=self.lo)
        # Check aligned data now the same.
        if n is None:
            aligned = Resample(time_delay, tel1.start_time)
            self.assert_tel_same(tel1, aligned)
        else:
            aligned = Resample(time_delay, tel1.start_time,
                               samples_per_frame=1)
            self.assert_tel_same(tel1, aligned, atol=self.atol_channelized)


class TestDelayAndResampleToneReal(BaseDelayAndResampleTestsReal):
    """Test DelayAndResample using a signal with just a single frequency.

    With this simple signal, we know exactly what is expected, so we
    add some explicit tests to those in the base, which only check
    the delaying itself, not whether the simulation is correct.
    """
    atol_channelized = 4e-4  # Channelization makes tone Resampling worse.
    signal_offset = 1/128    # Offset from lo in units of full_sample_rate.

    def setup(self):
        self.f_signal = self.lo + (self.signal_offset * self.sideband
                                   * self.full_sample_rate)
        self.phi0_signal = 98.7654321 * u.degree
        self.signal = PureTone(self.f_signal, self.start_time,
                               self.phi0_signal)
        super().setup()

    @pytest.mark.parametrize('n', (None, 32))
    def test_setup_no_delay(self, n):
        tel = self.get_tel(delay=None, n=n)
        assert tel.start_time == self.start_time
        data = tel.read()
        # Calculate expected phase using time at telescope, relative
        # to start of the simulated signal.
        i = np.arange(data.shape[0]).reshape(
            (-1,)+(1,)*len(self.raw.sample_shape))
        dt = i / tel.sample_rate
        # Phase of the signal is that of the sine wave.
        phi = self.phi0_signal + dt * self.f_signal * u.cycle
        # Subtract the mixer phase.
        # Note: CHIME has zero phi0_mixer and lo
        phi -= self.phi0_mixer + dt * self.lo * u.cycle
        phi *= self.sideband
        expected = PureTone.pure_tone(phi.to_value(u.radian), data.dtype)
        if n is None:
            assert_allclose(data, expected, atol=self.atol, rtol=0)
        else:
            data_ok = data[:, tel.frequency == np.abs(self.f_signal)]
            if tel.ih.complex_data:
                factor = n
            else:
                factor = n // 2
            assert_allclose(data_ok.reshape(expected.shape),
                            expected*factor,
                            atol=self.atol_channelized*factor, rtol=0)

    @pytest.mark.parametrize('delay', (-13, -2, 1, 111))
    @pytest.mark.parametrize('n', (None, 32))
    def test_setup_delay(self, delay, n):
        tel = self.get_tel(delay=delay, n=n)
        delay_time = delay / self.raw.sample_rate
        assert abs(tel.start_time - self.start_time + delay_time) < 1. * u.ns

        data = tel.read()
        # Calculate expected phase using time at telescope, working relative
        # to the start time of the simulation.
        i = np.arange(data.shape[0]).reshape((-1,)+(1,)*(data.ndim-1))
        dt = (tel.start_time - self.start_time) + i / tel.sample_rate
        # Calculate the signal phase, taking into account it was delayed.
        phi = self.phi0_signal + (dt + delay_time) * self.f_signal * u.cycle
        # Subtract the mixer phase, which was not delayed.
        # Note: CHIME has zero phi0_mixer and lo
        phi -= self.phi0_mixer + dt * self.lo * u.cycle
        phi *= self.sideband
        expected = PureTone.pure_tone(phi.to_value(u.radian), data.dtype)
        if n is None:
            assert_allclose(data, expected, atol=self.atol, rtol=0)
        else:
            data_ok = data[:, tel.frequency == np.abs(self.f_signal)]
            if tel.ih.complex_data:
                factor = n
            else:
                factor = n // 2
            assert_allclose(data_ok.reshape(expected.shape),
                            expected*factor,
                            atol=self.atol_channelized*factor, rtol=0)


class TestDelayAndResampleToneComplex(TestDelayAndResampleToneReal,
                                      BaseDelayAndResampleTestsComplex):
    pass


class TestDelayAndResampleNoiseReal(BaseDelayAndResampleTestsReal):
    atol_channelized = 1e-4

    def setup(self):
        self.noise = Noise(seed=12345)
        super().setup()

    def signal(self, ih):
        # For real data, we should make sure that our noise stream
        # does not have signal above the Nyquist frequency given our
        # full sample rate.
        # Here, we create a full-noise stream, but remove the upper
        # half of the band in frequency space.  This of course is
        # wasteful, but easy...  And we do need to use Noise to ensure
        # that rereading the filehandle gives identical results
        # (with caching, it doesn't seem the tests need it, but better
        # safe than sorry).
        data = self.noise(ih)
        ft = np.fft.rfft(data, axis=0)
        ft[ft.shape[0]//2:] = 0
        return np.fft.irfft(ft, axis=0).astype(data.dtype)


class TestDelayAndResampleNoiseComplex(TestDelayAndResampleNoiseReal,
                                       BaseDelayAndResampleTestsComplex):
    # In principle, a full band of noise is fine with complex sampling,
    # but easier to just keep it the same.
    pass


class CHIMELike:
    dtype = np.dtype('c8')
    atol_channelized = 1e-4
    lo = 0 * u.kHz  # no mixing at all.
    phi0_mixer = 0 * u.cycle  # Ensure we don't get any mixer phases.
    sideband = np.array(-1)
    ns_chan = 32

    def get_tel(self, delay=None, n=None):
        """Get signal from CHIME-like telescope."""
        if delay is None:
            fh = self.raw
        else:
            delay_time = delay / self.raw.sample_rate
            fh = SetAttribute(self.raw, start_time=self.start_time-delay_time)
        if n is None:
            n = self.ns_chan
        # Observe the raw, possibly delayed samples, using channelizer
        return Channelize(fh, n, frequency=self.full_sample_rate,
                          sideband=self.sideband)

    # Redefined to use delays that are multiple of self.ns_chan.
    @pytest.mark.parametrize('delay', (-32, 64))
    def test_time_delay(self, delay):
        # Pure time delays must be in units of telescope samples.
        super().test_time_delay(delay)

    # Redefined to remove the parametrization in n.
    @pytest.mark.parametrize('delay', (-1, 15.4321))
    def test_time_delay_align(self, delay):
        super().test_time_delay_align(delay, n=self.ns_chan)

    # Redefined to remove the parametrization in n.
    @pytest.mark.parametrize('delay', (-18.25, -np.pi, -8, 0.1, 65.4321))
    def test_resample_delayed(self, delay):
        """Create delayed and non-delayed versions; check we can undo delay."""
        super().test_resample_delayed(delay, n=self.ns_chan)


class TestDelayAndResampleToneCHIMELike(CHIMELike,
                                        TestDelayAndResampleToneComplex):
    signal_offset = -7/8  # w/ sideband, tone at full_sample_rate * 7/8

    # Redefined to remove the parametrization in n.
    def test_setup_no_delay(self):
        super().test_setup_no_delay(self.ns_chan)

    # Redefined to remove the parametrization in n.
    @pytest.mark.parametrize('delay', (-13, -2, 1, 111))
    def test_setup_delay(self, delay):
        super().test_setup_delay(delay, n=self.ns_chan)


class TestDelayAndResampleNoiseCHIMELike(CHIMELike,
                                         TestDelayAndResampleNoiseComplex):
    def setup(self):
        self.noise = Noise(seed=12345)
        super().setup()

    def signal(self, ih):
        # For CHIME data, lower part is filtered out.
        data = self.noise(ih)
        ft = np.fft.rfft(data, axis=0)
        ft[:ft.shape[0]//2] = 0
        return np.fft.irfft(ft, axis=0).astype(data.dtype)
