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
    seek_float, ShiftAndResample, Resample, TimeDelay, ShiftSamples)
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
    @classmethod
    def setup_class(self):
        # Only needed for the shape, sample_rate and start_time attributes.
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
    """Tests for resampling a signal.

    Here used for a real signal, but subclassed below for complex.
    """

    dtype = np.dtype('f4')
    full_sample_rate = 1 * u.kHz
    samples_per_full_frame = 4096  # Per full frame.
    start_time = Time('2010-11-12T13:14:15')
    frequency = 400. * u.kHz
    sideband = np.array([-1, 1])
    n_frames = 3

    pad = 32  # Size of response = 2*pad + 1.
    atol = 7e-4  # Tolerance within which we expect to reproduce signal.

    @classmethod
    def setup_class(self):
        # Chose signals that are not commensurate with quarter-sample offsets,
        # or with the frames.
        f_signal = (self.full_sample_rate * 2 / self.samples_per_full_frame
                    * np.array([31.092, 65.1234]))
        cosine = PureTone(f_signal, self.start_time, np.pi*u.deg)
        # Create a stream that just contains the two tones, and one sampled
        # at 4 times lower rate.
        self.full_fh = StreamGenerator(
            cosine,
            shape=((self.samples_per_full_frame * self.n_frames,)
                   + self.sideband.shape),
            sample_rate=self.full_sample_rate,
            samples_per_frame=self.samples_per_full_frame,
            frequency=self.frequency, sideband=self.sideband,
            start_time=self.start_time, dtype=self.dtype)
        self.part_fh = StreamGenerator(
            cosine,
            shape=((self.samples_per_full_frame // 4 * self.n_frames,)
                   + self.sideband.shape),
            sample_rate=self.full_sample_rate / 4,
            samples_per_frame=self.samples_per_full_frame // 4,
            frequency=self.frequency, sideband=self.sideband,
            start_time=self.start_time, dtype=self.dtype)

    def test_setup(self):
        self.full_fh.seek(0)
        self.part_fh.seek(0)
        full = self.full_fh.read()
        part = self.part_fh.read()
        assert_allclose(part, full[::4], atol=1e-8, rtol=0)

    @pytest.mark.parametrize('offset',
                             (34, 34.5, 35.75,
                              50.*u.ms, 0.065*u.s,
                              Time('2010-11-12T13:14:15.073')))
    def test_resample(self, offset):
        # Offsets equal to quarter samples to allow check with full_fh.
        ih = Resample(self.part_fh, offset, pad=self.pad)
        # Always lose 2 * pad per frame.
        assert ih.shape[0] == self.part_fh.shape[0] - 2 * self.pad
        assert ih.sample_shape == self.part_fh.sample_shape
        # Check we are at the given offset.
        if isinstance(offset, Time):
            expected_time = offset
        elif isinstance(offset, u.Quantity):
            expected_time = self.part_fh.start_time + offset
        else:
            expected_time = (self.part_fh.start_time
                             + offset / self.part_fh.sample_rate)
        assert abs(ih.time - expected_time) < 1. * u.ns

        part_fh_offset = seek_float(self.part_fh, offset)
        ioffset = round(part_fh_offset)
        fraction = part_fh_offset - ioffset
        assert ih.offset + self.pad == ioffset
        expected_start_time = (self.part_fh.start_time
                               + ((self.pad + fraction)
                                  / self.part_fh.sample_rate))
        assert abs(ih.start_time - expected_start_time) < 1. * u.ns
        # Check that data is correctly resampled.
        ih.seek(0)
        data = ih.read()
        # Find corresponding fully sampled data.
        self.full_fh.seek(ih.start_time)
        # Check time: should be exact, since our offsets are quarter samples.
        assert np.abs(self.full_fh.time - ih.start_time) < 1. * u.ns
        expected = self.full_fh.read(data.shape[0]*4)[::4]
        assert_allclose(data, expected, atol=self.atol, rtol=0)

    def test_repr(self):
        ih = Resample(self.part_fh, 0.5, samples_per_frame=511)
        r = repr(ih)
        assert r.startswith('Resample(ih')
        assert 'offset=0.5' in r

    @pytest.mark.parametrize('shift',
                             (0., 0.25, -5.25, [1.75, 10.25],
                              [-1., 13]*u.ms))
    @pytest.mark.parametrize('offset', (None, 0, 0.25))
    def test_shift_and_resample(self, shift, offset):
        # Shifts and offsets at quarter samples to allow check with full_fh.
        ih = ShiftAndResample(self.part_fh, shift, offset=offset, pad=self.pad)
        # start_time should be at expected offset from old grid.
        expected_offset = seek_float(
            self.part_fh, offset if offset is not None else np.mean(shift))
        d_off = ((ih.start_time - self.start_time) * ih.sample_rate
                 - expected_offset).to_value(u.one)
        assert abs(d_off - np.around(d_off)) < u.ns * ih.sample_rate
        expected_length = (self.part_fh.shape[0]
                           - 2 * self.pad
                           - seek_float(self.part_fh, np.ptp(shift)))
        assert abs(ih.shape[0] - expected_length) <= 0.5

        # Data should be shifted by the expected amounts.
        shift = np.atleast_1d(shift)
        for i, s in enumerate(shift):
            ih.seek(0)
            time_shift = seek_float(self.part_fh, s) / ih.sample_rate
            fh_pos = self.full_fh.seek(ih.time - time_shift)
            assert fh_pos >= 0
            assert abs(ih.time - time_shift - self.full_fh.time) < 1.*u.ns

            data = ih.read()
            expected = self.full_fh.read(len(data)*4)[::4]
            sel = i if shift.size > 1 else Ellipsis
            assert_allclose(data[:, sel], expected[:, sel],
                            atol=self.atol, rtol=0)


class TestResampleComplex(TestResampleReal):

    dtype = np.dtype('c16')

    def test_wrong_shape(self):
        # Test on this class just because it only needs to be done once.
        with pytest.raises(ValueError, match='broadcast to sample shape'):
            ShiftAndResample(self.part_fh, np.array([[1], [2]]))


class StreamArray(StreamGenerator):
    def __init__(self, data, **kwargs):
        def from_data(handle):
            result = data[handle.offset:
                          handle.offset+handle.samples_per_frame]
            assert result.shape[0] == handle.samples_per_frame
            return result
        super().__init__(from_data, shape=data.shape, **kwargs)


class TestResampleNoise(TestResampleComplex):

    dtype = np.dtype('c8')
    pad = 64  # need more padding for noise
    high_ft_cut = 2*pad  # and removal of high frequencies.

    @classmethod
    def setup_class(self):
        # Make noise with only frequencies covered by part_fh.
        n = self.samples_per_full_frame // 4 * self.n_frames
        np.random.seed(123456)
        part_data = np.random.normal(size=(n, 2*2)).view('c16')
        part_ft = np.fft.fft(part_data, axis=0)
        # Set high frequencies to zero; resampling doesn't work well with
        # noise there, as the FT mixes it from positive to negative and
        # vice versa, thus messing up the phases.
        part_ft[n//2-self.high_ft_cut:n//2+self.high_ft_cut] = 0
        # Make corresponding FT for full frame.
        full_ft = np.concatenate((part_ft[:n//2],
                                  np.zeros((n*3, 2), 'c16'),
                                  part_ft[-n//2:]), axis=0)
        # Factor 4 to ensure data have same power.
        self.full_data = np.fft.ifft(full_ft * 4, axis=0)
        self.part_data = np.fft.ifft(part_ft, axis=0)
        self.full_fh = StreamArray(
            self.full_data,
            sample_rate=self.full_sample_rate,
            samples_per_frame=self.samples_per_full_frame,
            frequency=self.frequency, sideband=self.sideband,
            start_time=self.start_time, dtype=self.dtype)

        self.part_fh = StreamArray(
            self.part_data,
            sample_rate=self.full_sample_rate / 4,
            samples_per_frame=self.samples_per_full_frame // 4,
            frequency=self.frequency, sideband=self.sideband,
            start_time=self.start_time, dtype=self.dtype)


class BaseDelayAndResampleTestsReal:
    """Base class for ShiftAndResample tests with time delay phase shifts.

    Sub-classes need to define ``self.signal``, which is used for the
    simulated voltage baseband stream that will be mixed, low-pass filtered
    and downsampled.

    The idea behind all tests is to similate a voltage stream and
    the whole receiver chain, i.e., mix it with an IF, low-pass filter
    it, and then detect.  The mixing can be with real or complex (quadrature).

    """
    dtype = np.dtype('f4')  # type of mixing and output data.
    atol = atol_channelized = 1.e-2  # tolerance per sample
    full_sample_rate = 204.8 * u.kHz  # For the real-valued input signal
    samples_per_full_frame = 1024
    start_time = Time('2010-11-12T13:14:15')
    sideband = np.array([-1, 1])    # IF sideband
    # Place lo on right side of signal
    lo = full_sample_rate * (7 / 16 - sideband / 128)  # IF frequency.
    n_frames = 32
    phi0_mixer = -12.3456789 * u.degree  # "random" angle.

    @classmethod
    def setup_class(self):
        self.full_shape = ((self.samples_per_full_frame * self.n_frames,)
                           + self.sideband.shape)
        self.downsample = (16 if self.dtype.kind == 'c' else 8)
        self.sample_rate = self.full_sample_rate / self.downsample
        # Create the IF (which can produce a complex tone for quadrature).
        self.mixer = PureTone(self.lo, self.start_time, self.phi0_mixer)
        # Create a real-valued stream with a test-specific signal.
        self.raw = StreamGenerator(
            self.signal, shape=self.full_shape, start_time=self.start_time,
            sample_rate=self.full_sample_rate, dtype=np.dtype('f4'),
            samples_per_frame=self.samples_per_full_frame)

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
            f = np.fft.rfftfreq(data.shape[0], 1/ih.ih.sample_rate).reshape(
                (-1,)+(1,)*(ft.ndim-1))
            wrong_side = (f < self.lo) ^ (ih.sideband < 0)
            ft[wrong_side] = 0
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
        """Get signal as observed at a telescope with the given delay
        and number of channels."""
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
            # For unchannelized data, the frequency should equal the lo.
            assert np.all(tel1.frequency == self.lo)
            tel2_rs = ShiftAndResample(tel2, delay / self.full_sample_rate,
                                       tel1.start_time, lo=self.lo)
            self.assert_tel_same(tel1, tel2_rs)
        else:
            # For channelized data, we have to ensure we pass in an explicit
            # local oscillator frequency.  We reduce padding and
            # samples_per_frame since we have rather little data.
            tel2_rs = ShiftAndResample(tel2, delay / self.full_sample_rate,
                                       tel1.start_time, lo=self.lo,
                                       samples_per_frame=32, pad=6)
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
        # For more random time delays, we need to Resample.  Of course, in
        # this case we might just as well have used ShiftAndResample.
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
                               samples_per_frame=32, pad=6)
            self.assert_tel_same(tel1, aligned, atol=self.atol_channelized)


class TestDelayAndResampleToneReal(BaseDelayAndResampleTestsReal):
    """Test DelayAndResample using a signal with just a single frequency.

    With this simple signal, we know exactly what is expected, so we
    add some explicit tests to those in the base, which only check
    the delaying itself, not whether the simulation is correct.
    """
    atol_channelized = 4e-4  # Channelization makes tone Resampling worse.
    signal_offset = 7 / 16   # Signal frequency in units of full_sample_rate.

    @classmethod
    def setup_class(self):
        self.f_signal = self.signal_offset * self.full_sample_rate
        self.phi0_signal = 98.7654321 * u.degree
        self.signal = PureTone(self.f_signal, self.start_time,
                               self.phi0_signal)
        super().setup_class()

    @pytest.mark.parametrize('n', (None, 32))
    def test_setup_no_delay(self, n):
        tel = self.get_tel(delay=None, n=n)
        assert tel.start_time == self.start_time
        data = tel.read()
        # Calculate expected phase using time at telescope, relative
        # to start of the simulated signal.
        i = np.arange(data.shape[0]).reshape(
            (-1,) + (1,)*len(self.raw.sample_shape))
        dt = i / tel.sample_rate
        # Phase of the signal is that of the sine wave.
        phi = self.phi0_signal + dt * self.f_signal * u.cycle
        # Subtract the mixer phase.
        # Note: CHIME has zero phi0_mixer and lo
        phi = phi - (self.phi0_mixer + dt * self.lo * u.cycle)
        phi *= self.sideband
        expected = PureTone.pure_tone(phi.to_value(u.radian), data.dtype)
        if n is None:
            assert_allclose(data, expected, atol=self.atol, rtol=0)
        else:
            # Pick out relevant channel.
            data_ok = data[:, np.isclose(tel.frequency, np.abs(self.f_signal))]
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
        phi = phi - (self.phi0_mixer + dt * self.lo * u.cycle)
        phi *= self.sideband
        expected = PureTone.pure_tone(phi.to_value(u.radian), data.dtype)
        if n is None:
            assert_allclose(data, expected, atol=self.atol, rtol=0)
        else:
            data_ok = data[:, np.isclose(tel.frequency, np.abs(self.f_signal))]
            if tel.ih.complex_data:
                factor = n
            else:
                factor = n // 2
            assert_allclose(data_ok.reshape(expected.shape),
                            expected*factor,
                            atol=self.atol_channelized*factor, rtol=0)


class TestDelayAndResampleToneComplex(TestDelayAndResampleToneReal,
                                      BaseDelayAndResampleTestsComplex):
    # Base tests plus one that checks our understanding of TimeDelay.

    @pytest.mark.parametrize('delay', (-8., 12.3456789))
    def test_time_delay_understanding(self, delay):
        tel = self.get_tel(delay=delay)
        time_delay1 = TimeDelay(tel, delay / self.full_sample_rate, lo=self.lo)
        time_delay2 = ShiftAndResample(tel, delay / self.full_sample_rate,
                                       offset=None, lo=self.lo, pad=0)
        assert time_delay1.shape == time_delay2.shape
        assert time_delay1.start_time == time_delay2.start_time
        self.assert_tel_same(time_delay1, time_delay2)


class TestDelayAndResampleNoiseReal(BaseDelayAndResampleTestsReal):
    atol_channelized = 1e-4

    @classmethod
    def setup_class(self):
        self.noise = Noise(seed=12345)
        super().setup_class()

    @classmethod
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
    @classmethod
    def signal(self, ih):
        # For complex data, a full band of noise is fine.
        return self.noise(ih)


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
    signal_offset = 7/8  # Tone at full_sample_rate * 7/8

    # Redefined to remove the parametrization in n.
    def test_setup_no_delay(self):
        super().test_setup_no_delay(self.ns_chan)

    # Redefined to remove the parametrization in n.
    @pytest.mark.parametrize('delay', (-13, -2, 1, 111))
    def test_setup_delay(self, delay):
        super().test_setup_delay(delay, n=self.ns_chan)


class TestDelayAndResampleNoiseCHIMELike(CHIMELike,
                                         TestDelayAndResampleNoiseComplex):
    @classmethod
    def signal(self, ih):
        # For CHIME data, lower part is filtered out.
        data = self.noise(ih)
        ft = np.fft.rfft(data, axis=0)
        ft[:ft.shape[0]//2] = 0
        return np.fft.irfft(ft, axis=0).astype(data.dtype)


class TestShiftSamples:
    @classmethod
    def make_arange_data(self, ih):
        test_data = (np.arange(ih.offset, ih.offset + ih.samples_per_frame)
                     .reshape((ih.samples_per_frame,)
                              + (1,) * len(ih.shape[1:])))
        new_shape = (ih.samples_per_frame,) + ih.shape[1:]
        return np.broadcast_to(test_data, new_shape)

    @classmethod
    def make_non_uniform_arange_data(self, ih):
        axis = ih.non_uniform_axis
        data = self.make_arange_data(ih)
        multiplier = np.arange(1, data.shape[axis] + 1) * ih.intensity
        return data * multiplier.reshape((data.shape[axis],)
                                         + (1,) * (data.ndim-1-axis))

    @classmethod
    def setup_class(self):
        self.shape = (1000, 5, 3)
        self.ih = StreamGenerator(self.make_arange_data,
                                  self.shape, Time('2010-11-12'), 1.*u.Hz,
                                  samples_per_frame=100, dtype=float)
        self.ih.intensity = 2
        self.ih.non_uniform_axis = 1

    @pytest.mark.parametrize('start, n', [(0, 5), (90, 20)])
    def test_shift_back(self, start, n):
        shift_axis = 1
        shift = np.arange(-self.shape[shift_axis] + 1, 1)
        assert shift.max() == 0
        shifter = ShiftSamples(self.ih, shift.reshape(-1, 1),
                               samples_per_frame=100)
        assert shifter.start_time == self.ih.start_time
        shifter.seek(start)
        shifted = shifter.read(n)
        self.ih.seek(start)
        raw_data = self.ih.read(100)
        for i, sf in enumerate(-shift):
            assert np.all(shifted[:, i] == raw_data[sf:sf + n, i])

    @pytest.mark.parametrize('start, n', [(0, 5), (100, 20)])
    def test_shift_both(self, start, n):
        shift = np.array([-2, 0, 3])
        shifter = ShiftSamples(self.ih, shift, samples_per_frame=100)
        assert abs(shifter.start_time - 3 / self.ih.sample_rate
                   - self.ih.start_time) < 1.*u.ns
        shifter.seek(start)
        shifted = shifter.read(n)
        self.ih.seek(start)
        raw_data = self.ih.read(100)
        for i, sf in enumerate(3-shift):
            assert np.all(shifted[:, :, i] == raw_data[sf:sf + n, :, i])

    def test_compare_with_shift_and_resample(self):
        shift = np.array([-2, 1, 4])
        shifter = ShiftSamples(self.ih, shift, samples_per_frame=100)
        resampler = ShiftAndResample(self.ih, shift, offset=0,
                                     pad=32, samples_per_frame=200)
        # Note: resampler has larger padding, so start time is later.
        shifter.seek(90)
        resampler.seek(shifter.time)
        # integer shifts, so time should be same.
        assert abs(resampler.time - shifter.time) < 1. * u.ns
        shifted = shifter.read(20)
        resampled = resampler.read(20)
        assert_allclose(shifted, resampled)

    @pytest.mark.parametrize('fshift, ishift', [
        (np.array([1., 2., 3.25]), [1, 2, 3]),
        (np.array([[-1.9], [-5.], [5.25], [3.49], [-1.2]]),
         np.reshape([-2, -5, 5, 3, -1], (-1, 1)))])
    def test_non_integer_shift(self, fshift, ishift):
        shifter1 = ShiftSamples(self.ih, fshift)
        comparison = ShiftSamples(self.ih, ishift)
        assert np.all(shifter1._shift == comparison._shift)
        data1 = shifter1.read()
        expected = comparison.read()
        assert np.all(data1 == expected)
        shifter2 = ShiftSamples(self.ih, fshift / self.ih.sample_rate)
        assert np.all(shifter2._shift == comparison._shift)
        data2 = shifter2.read()
        assert np.all(data2 == expected)

    def test_wrong_shape(self):
        with pytest.raises(ValueError, match='broadcast to sample shape'):
            ShiftSamples(self.ih, np.array([[1], [2]]))
