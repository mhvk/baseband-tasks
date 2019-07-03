#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test resampling."""

import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.time import Time

from baseband_tasks.base import Task, SetAttribute
from baseband_tasks.combining import Stack
from baseband_tasks.sampling import Resample, float_offset, TimeShift, ShiftAndResample
from baseband_tasks.generators import StreamGenerator


class PureTone:
    def __init__(self, frequency, start_time):
        self.frequency = frequency
        self.start_time = start_time

    @staticmethod
    def pure_tone(phi, dtype):
        if dtype.kind == 'f':
            cosine = np.cos(phi)
        else:
            cosine = np.exp(1j * phi)
        return cosine.astype(dtype, copy=False)

    def __call__(self, ih):
        dt = ((ih.time - self.start_time).to(u.s)
              + np.arange(ih.samples_per_frame) / ih.sample_rate)
        dt = dt.reshape((-1,) + (1,) * len(ih.sample_shape))
        phi = (self.frequency * dt * u.cycle).to_value(u.rad)
        return self.pure_tone(phi, ih.dtype)


class TestResampleReal:

    dtype = 'f4'
    atol = 1e-4
    sample_rate = 1 * u.kHz
    samples_per_frame = 1024
    start_time = Time('2010-11-12T13:14:15')
    frequency = 400. * u.kHz
    shape = (2048, 2)
    sideband = np.array([-1, 1])

    def setup(self):
        f_sine = self.sample_rate / 32 * np.ones(self.shape[1:])

        cosine = PureTone(f_sine, self.start_time)

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

        ioffset, fraction = divmod(float_offset(self.part_fh, offset), 1)
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


class TestResampleComplex(TestResampleReal):

    dtype = 'c16'
    atol = 1e-8


class StreamArray(StreamGenerator):
    def __init__(self, data, *args, **kwargs):
        def from_data(handle):
            return data[handle.offset:
                        handle.offset+handle.samples_per_frame]
        super().__init__(from_data, *args, **kwargs)


class TestResampleNoise(TestResampleComplex):

    dtype = 'c8'
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


class TestTimeShift:
    dtype = np.dtype('c8')
    full_sample_rate = 102.4 * u.kHz
    sample_rate = full_sample_rate / 8
    samples_per_frame = 1024
    start_time = Time('2010-11-12T13:14:15')
    frequency = full_sample_rate * 7 / 8
    sideband = np.array(1)
    shape = (102400,) + sideband.shape
    f_sine = frequency + sample_rate / 16 * sideband

    def setup(self):
        self.signal = PureTone(self.f_sine, self.start_time)
        self.mixer = PureTone(self.frequency * self.sideband, self.start_time)

        self.full_fh = StreamGenerator(
            self.signal, shape=self.shape,
            sample_rate=self.full_sample_rate,
            samples_per_frame=self.samples_per_frame,
            start_time=self.start_time, dtype=self.dtype)

    def get_tel(self, delay=None):
        if delay is None:
            fh = self.full_fh
        else:
            delay_time = delay / self.full_fh.sample_rate
            fh = SetAttribute(self.full_fh,
                              start_time=self.start_time-delay_time)

        def mix_downsample8(ih, data):
            # Mix.
            data = data * self.mixer(ih.ih).conj()
            # Apply a low-pass filter:
            ft = np.fft.fft(data, axis=0)
            s, r = divmod(ft.shape[0], 16)
            assert r == 0
            ft[s:-s] = 0
            data = np.fft.ifft(ft, axis=0).astype('c8')
            # And down-sample.
            return data[::8]

        return Task(fh, mix_downsample8,
                    sample_rate=self.sample_rate,
                    shape=(fh.shape[0] // 8,)+fh.sample_shape,
                    frequency=self.frequency, sideband=self.sideband)

    def test_setup_no_delay(self):
        tel = self.get_tel(delay=None)
        assert tel.start_time == self.start_time
        data = tel.read()
        # Calculate expected phase using time at telescope, relative
        # to start of the simulated signal.
        dt = np.arange(tel.shape[0]) / tel.sample_rate
        dt.shape = (-1,) + (1,) * len(tel.sample_shape)
        # Phase of the signal is that of the sine wave, minus mixer phase.
        phi = dt * (self.f_sine - self.frequency) * u.cycle
        expected = PureTone.pure_tone(phi.to_value(u.radian), data.dtype)
        assert_allclose(data, expected, atol=1e-4, rtol=0)

    @pytest.mark.parametrize('delay', (-2, -1, 1, 13, 111))
    def test_setup_delay(self, delay):
        tel = self.get_tel(delay=delay)
        delay_time = delay / self.full_fh.sample_rate
        assert abs(tel.start_time - self.start_time + delay_time) < 1. * u.ns
        assert tel.shape == (self.shape[0] // 8,) + self.shape[1:]
        data = tel.read()
        # Calculate expected phase using time at telescope, working relative
        # to the start time of the simulation.
        dt = ((tel.start_time - self.start_time)
              + np.arange(tel.shape[0]) / tel.sample_rate)
        dt.shape = (-1,) + (1,) * len(tel.sample_shape)
        # Calculate the signal phase, taking into account it was delayed.
        phi = (dt + delay_time) * self.f_sine * u.cycle
        # And subtract the mixer phase, which was not delayed.
        phi -= dt * self.frequency * self.sideband * u.cycle
        expected = PureTone.pure_tone(phi.to_value(u.radian), data.dtype)
        assert_allclose(data, expected, atol=1e-4, rtol=0)

    @pytest.mark.parametrize('delay', (-8, 8, 64))
    def test_time_shift(self, delay):
        tel1 = self.get_tel(delay=None)
        tel2 = self.get_tel(delay=delay)
        time_shift = TimeShift(tel2, delay / self.full_sample_rate)
        # read both telescopes together.
        both_tel = Stack((tel1, time_shift), axis=1)
        data = both_tel.read()
        assert_allclose(data[:, 0], data[:, 1], atol=1e-4)

    @pytest.mark.parametrize('delay', (-18.25, -np.pi, -1, 0.1, 11, 65.4321))
    def test_resample_shifted(self, delay):
        tel1 = self.get_tel(delay=None)
        tel2 = self.get_tel(delay=delay)
        rs = ShiftAndResample(tel2, delay / self.full_sample_rate,
                              tel1.start_time)
        # read both telescopes together.
        both_tel = Stack((tel1, rs), axis=1)
        data = both_tel.read()
        assert_allclose(data[:, 0], data[:, 1], atol=1e-4)
