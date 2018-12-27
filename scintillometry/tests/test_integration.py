# Licensed under the GPLv3 - see LICENSE
"""Tests of integration and pulse folding."""

import pytest
import numpy as np
import astropy.units as u
from astropy.time import Time

from ..base import Task
from ..generators import EmptyStreamGenerator
from ..integration import (Integrate, IntegrateSamples, IntegrateTime,
                           Fold, Stack)
from ..functions import Square


class TestFakePulsarBase:
    def setup(self):
        self.start_time = Time('2010-11-12T13:14:15')
        self.sample_rate = 10. * u.kHz
        self.shape = (16000, 2)
        self.eh = EmptyStreamGenerator(shape=self.shape,
                                       start_time=self.start_time,
                                       sample_rate=self.sample_rate,
                                       samples_per_frame=200, dtype=np.float)
        self.sh = Task(self.eh, self.pulse_simulate)
        self.period_bin = 125
        self.F0 = 1.0 / (self.period_bin / self.sh.sample_rate)
        self.n_phase = 50
        self.raw_data = self.sh.read()
        self.raw_power = self.raw_data ** 2
        self.sh.seek(0)

    def phase(self, t):
        return u.cycle * self.F0 * (t - self.start_time)

    def pulse_simulate(self, fh, data):
        idx = fh.tell() + np.arange(data.shape[0])
        result = np.where(idx % self.period_bin == 0, 10., 0.125)
        result.shape = (-1,) + (1,) * (data.ndim - 1)
        data[:] = result
        return data


class TestIntegrate(TestFakePulsarBase):
    """Test integrating intensities using Baseband's sample DADA file."""

    @pytest.mark.parametrize('integrate_cls', (Integrate, IntegrateSamples,
                                               IntegrateTime))
    def test_integrate_all(self, integrate_cls):
        # Load baseband file and get reference intensities.
        ref_data = self.raw_power.mean(0)

        st = Square(self.sh)
        ip = integrate_cls(st)
        assert ip.start_time == self.sh.start_time
        assert abs(ip.stop_time - self.sh.stop_time) < 1. * u.ns
        assert abs(ip.stop_time - self.sh.start_time - 1./ip.sample_rate) < 1. * u.ns

        # Square and integrate everything.
        data = ip.read()
        assert ip.tell() == ip.shape[0]
        assert st.dtype is ref_data.dtype is data.dtype
        assert data.shape == (1, 2)
        assert np.allclose(data, ref_data)

    def test_integrate_all_no_average(self):
        # Load baseband file and get reference intensities.
        ref_data = self.raw_power.sum(0)

        st = Square(self.sh)
        ip = Integrate(st, average=False)
        assert ip.start_time == self.sh.start_time
        assert abs(ip.stop_time - self.sh.stop_time) < 1. * u.ns
        assert abs(ip.stop_time - self.sh.start_time - 1./ip.sample_rate) < 1. * u.ns

        # Square and integrate everything.
        integrated = ip.read()
        data = integrated['data']
        count = integrated['count']
        assert ip.tell() == ip.shape[0]
        assert st.dtype is ref_data.dtype is data.dtype
        assert data.shape == (1, 2)
        assert np.allclose(data, ref_data)
        assert np.all(count == self.sh.shape[0])

    @pytest.mark.parametrize('samples_per_frame', (1, 4, 10))
    @pytest.mark.parametrize('n', (1, 3))
    def test_integrate_n(self, n, samples_per_frame):
        ref_data = self.raw_power[121 * n:131 * n].reshape(-1, n, 2).sum(1)

        st = Square(self.sh)
        ip = Integrate(st, n, average=False,
                       samples_per_frame=samples_per_frame)
        assert ip.start_time == self.sh.start_time
        assert ip.sample_rate == self.sh.sample_rate / n

        ip.seek(121)
        integrated = ip.read(10)
        data = integrated['data']
        count = integrated['count']
        assert ip.tell() == 131
        assert st.dtype is ref_data.dtype is data.dtype
        assert data.shape == ref_data.shape
        assert np.allclose(data, ref_data)
        assert np.all(count == n)

    @pytest.mark.parametrize('samples_per_frame', (1, 4, 10))
    @pytest.mark.parametrize('n', (1, 3))
    def test_integrate_n_via_time(self, n, samples_per_frame):
        ref_data = self.raw_power[151*n:161*n].reshape(-1, n, 2).sum(1)

        st = Square(self.sh)
        ip = Integrate(st, n/self.sh.sample_rate, average=False,
                       samples_per_frame=samples_per_frame)
        assert ip.start_time == self.sh.start_time
        assert ip.sample_rate == self.sh.sample_rate / n

        ip.seek(151)
        integrated = ip.read(10)
        data = integrated['data']
        count = integrated['count']
        assert ip.tell() == 161
        assert st.dtype is ref_data.dtype is data.dtype
        assert data.shape == ref_data.shape
        assert np.allclose(data, ref_data)
        assert np.all(count == n)

    @pytest.mark.parametrize('samples_per_frame', (1, 4, 10))
    def test_integrate_time_non_integer_ratio(self, samples_per_frame):
        # Getting non-integer number of samples.  With 2.26,
        # 1st sample will be 0.00 - 2.26 -> 2 samples;
        # 2nd ..             2.26 - 4.52 -> 3 samples;
        # 3rd ..             4.52 - 6.78 -> 2 samples; etc.
        expected_count = [2, 3, 2, 2, 2, 3, 2, 2]
        step = 2.26
        raw = self.raw_power[:18]
        ref_data = np.add.reduceat(raw, np.add.accumulate([0] + expected_count[:-1]))

        st = Square(self.sh)
        ip = Integrate(st, step / self.sh.sample_rate, average=False,
                       samples_per_frame=samples_per_frame)
        assert ip.start_time == self.sh.start_time
        assert ip.sample_rate == self.sh.sample_rate / step

        # Square and integrate everything.
        integrated = ip.read(8)
        data = integrated['data']
        count = integrated['count']
        assert ip.tell() == 8
        assert st.dtype is ref_data.dtype is data.dtype
        assert data.shape == ref_data.shape
        assert np.allclose(data, ref_data)
        assert np.all(count.T == expected_count)

    def test_time_too_large(self):
        with pytest.raises(AssertionError):
            Integrate(self.sh, step=1.*u.hr)


class TestFold(TestFakePulsarBase):
    def test_input_data(self):
        indata = self.raw_data[:1000]
        pulses = np.where(indata[:, 0] == 10)[0]
        # Check if the input data is set up right.
        assert len(pulses) == 8, "Pulses are not simulated right."

    def test_step_shorter_than_period(self):
        step = 11 * u.ms
        fh = Fold(self.sh, self.n_phase, self.phase, step,
                  samples_per_frame=1, average=False)
        fh.seek(0)
        fr = fh.read(3)
        # Note: API for accessing counts may change.
        fr_count = fr['count']
        max_count = int(np.ceil(
            (self.sample_rate / self.F0 / self.n_phase).to_value(1)))
        assert np.all(fr_count <= max_count)

        # For each of the samples, check that the correct ones have not
        # gotten any data.
        eff_n_phase = (step * self.F0 * self.n_phase).to_value(1)
        for ii, count in enumerate(fr_count):
            n_count_0 = (count == 0).sum()
            assert np.isclose(self.n_phase - n_count_0, eff_n_phase, 1), \
                ("Sample {} has the wrong number of zero-count phase bins"
                 .format(ii))

    def test_step_longer_than_period(self):
        step = 26 * u.ms
        fh = Fold(self.sh, self.n_phase, self.phase, step,
                  samples_per_frame=1, average=False)
        fh.seek(0)
        fr = fh.read(10)
        fr_count = fr['count']
        fr_data = fr['data']
        # Compare the total counts of all the samples.
        tot_counts = np.sum(fr_count, axis=1)
        assert np.all(np.abs(tot_counts - tot_counts.mean()) <= 1), \
            "Folding counts vary more than expected."
        # Test the output on and off gates.
        ph0_bins = [0, 1, -1, -2]
        pulse_power = np.sum(fr_data[:, ph0_bins, 0], axis=1)
        assert 30 < pulse_power[0] < 33, \
            "Folded power of on-gate is incorrect."

        assert np.all((pulse_power[1:] > 20) & (pulse_power[1:] < 23)), \
            "Folding power of off-gates is incorrect."

    def test_folding_with_averaging(self):
        # Test averaging
        fh = Fold(self.sh, self.n_phase, self.phase, step=26 * u.ms,
                  samples_per_frame=20, average=True)
        fh.seek(0)
        fr = fh.read(10)
        assert np.all(fr[:, 2:-1] == 0.125), \
            "Average off-gate power is incorrect."

    def test_non_integer_sample_rate_ratio(self):
        step = (1./3.) * u.s
        fh = Fold(self.sh, self.n_phase, self.phase, step, average=True)
        fr = fh.read(1)
        assert np.all(fr[:, 2:-1] == 0.125), \
            "Average off-gate power is incorrect."

    def test_read_whole_file(self):
        ref_data = self.raw_data[:, 0]
        phase = self.phase(self.start_time +
                           np.arange(self.sh.shape[0]) / self.sh.sample_rate)
        i_phase = ((phase.to_value(u.cycle) * self.n_phase) % self.n_phase).astype(int)
        expected = np.bincount(i_phase, ref_data) / np.bincount(i_phase)

        fh = Fold(self.sh, self.n_phase, self.phase, average=True)
        assert abs(fh.stop_time - self.sh.stop_time) < 1. * u.ns
        fr = fh.read(1)
        assert np.all(fr[:, 2:-1] == 0.125), \
            "Average off-gate power is incorrect."
        assert np.all(fr[0, :, 0] == expected)

    def test_time_too_large(self):
        with pytest.raises(AssertionError):
            Fold(self.sh, 8, self.phase, step=1.*u.hr)
        with pytest.raises(AssertionError):
            Fold(self.sh, 8, self.phase, samples_per_frame=2)


class TestIntegratePhase(TestFakePulsarBase):
    # More detailed tests done in TestStack.
    @pytest.mark.parametrize('samples_per_frame', (1, 160))
    def test_basics(self, samples_per_frame):
        ref_data = self.raw_data.reshape(-1, 5, 2).mean(1)

        fh = Integrate(self.sh, u.cycle/25, self.phase,
                       samples_per_frame=samples_per_frame)
        assert fh.start_time == self.sh.start_time
        assert fh.stop_time == self.sh.stop_time
        assert fh.sample_rate == 25 / u.cycle
        assert fh.samples_per_frame == samples_per_frame

        data = fh.read(20)
        assert np.all(data == ref_data[:20])
        fh.seek(250)
        data = fh.read(75)
        assert np.all(data == ref_data[250:325])
        if samples_per_frame > 1:  # very slow otherwise.
            data = fh.read()
            assert np.all(data == ref_data[325:])


class TestStack(TestFakePulsarBase):
    @pytest.mark.parametrize('samples_per_frame', (1, 16))
    def test_basics(self, samples_per_frame):
        ref_data = self.raw_data.reshape(-1, 25, 5, 2).mean(2)

        fh = Stack(self.sh, 25, self.phase,
                   samples_per_frame=samples_per_frame)
        assert fh.start_time == self.sh.start_time
        assert fh.stop_time == self.sh.stop_time
        assert fh.sample_rate == 1. / u.cycle
        assert fh.samples_per_frame == samples_per_frame

        data = fh.read(2)
        assert np.all(data == ref_data[:2])
        fh.seek(10)
        data = fh.read(3)
        assert np.all(data == ref_data[10:13])
        data = fh.read()
        assert np.all(data == ref_data[13:])
