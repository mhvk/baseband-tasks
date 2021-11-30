# Licensed under the GPLv3 - see LICENSE
"""Tests of integration and pulse folding."""

import pytest
import numpy as np
from numpy.testing import assert_array_equal
import astropy.units as u
from astropy.time import Time

from baseband_tasks.base import Task
from baseband_tasks.generators import EmptyStreamGenerator
from baseband_tasks.integration import Integrate, Fold, Stack
from baseband_tasks.functions import Square
from baseband_tasks.phases import Phase


class FakePulsarBase:
    def setup_class(self):
        self.start_time = Time('2010-11-12T13:14:15')
        self.sample_rate = 10. * u.kHz
        self.shape = (16000, 2)
        self.eh = EmptyStreamGenerator(shape=self.shape,
                                       start_time=self.start_time,
                                       sample_rate=self.sample_rate,
                                       samples_per_frame=200, dtype=float)
        self.sh = Task(self.eh, self.pulse_simulate)
        self.period_bin = 125
        self.F0 = 1.0 / (self.period_bin / self.sh.sample_rate)
        self.n_phase = 50
        self.raw_data = self.sh.read()
        self.raw_power = self.raw_data ** 2

    @classmethod
    def pulse_simulate(self, fh, data):
        idx = fh.tell() + np.arange(data.shape[0])
        result = np.where(idx % self.period_bin == 0, 10., 0.125)
        result.shape = (-1,) + (1,) * (data.ndim - 1)
        data[:] = result
        return data

    def phase(self, t):
        return u.cycle * self.F0 * (t - self.start_time)


class UsePhaseClass(FakePulsarBase):
    def phase(self, t):
        return Phase(super().phase(t))


class TestIntegrate(FakePulsarBase):
    """Test integrating intensities using Baseband's sample DADA file."""

    def test_integrate_all(self):
        # Load baseband file and get reference intensities.
        ref_data = self.raw_power.mean(0)

        st = Square(self.sh)
        ip = Integrate(st)
        assert ip.start_time == self.sh.start_time
        assert abs(ip.stop_time - self.sh.stop_time) < 1. * u.ns
        assert abs(ip.stop_time
                   - self.sh.start_time - 1./ip.sample_rate) < 1. * u.ns

        # Square and integrate everything.
        data = ip.read()
        assert ip.tell() == ip.shape[0]
        assert st.dtype is ref_data.dtype is data.dtype
        assert data.shape == (1, 2)
        assert np.allclose(data, ref_data)

    def test_integrate_part(self):
        # Use start to integrate only part.
        ref_data = self.raw_power[151:].mean(0)

        st = Square(self.sh)
        ip = Integrate(st, start=151)
        assert abs(ip.start_time - (self.sh.start_time
                                    + 151 / self.sh.sample_rate)) < 1. * u.ns
        assert abs(ip.stop_time - self.sh.stop_time) < 1. * u.ns

        # Square and integrate everything.
        data = ip.read()
        assert np.allclose(data, ref_data)

        r = repr(ip)
        assert r.startswith('Integrate(ih')
        assert 'start=151' in r
        assert 'step' not in r

    def test_integrate_all_no_average(self):
        # Load baseband file and get reference intensities.
        ref_data = self.raw_power.sum(0)

        st = Square(self.sh)
        ip = Integrate(st, average=False)
        assert ip.start_time == self.sh.start_time
        assert abs(ip.stop_time - self.sh.stop_time) < 1. * u.ns
        assert abs(ip.stop_time
                   - self.sh.start_time - 1./ip.sample_rate) < 1. * u.ns

        # Square and integrate everything.
        integrated = ip.read()
        data = integrated['data']
        count = integrated['count']
        assert ip.tell() == ip.shape[0]
        assert st.dtype is ref_data.dtype is data.dtype
        assert data.shape == (1, 2)
        assert np.allclose(data, ref_data)
        assert np.all(count == self.sh.shape[0])

    @pytest.mark.parametrize('seek', [121, -10])
    @pytest.mark.parametrize('samples_per_frame', (1, 4, 10))
    @pytest.mark.parametrize('n', (1, 3))
    def test_integrate_n(self, n, samples_per_frame, seek):
        n_sample = self.raw_power.shape[0] // n
        seek = seek if seek > 0 else n_sample+seek
        ref_data = (self.raw_power[seek * n:(seek+10) * n]
                        .reshape(-1, n, 2).sum(1))

        st = Square(self.sh)
        ip = Integrate(st, n, average=False,
                       samples_per_frame=samples_per_frame)
        assert ip.shape[0] == n_sample
        assert ip.start_time == self.sh.start_time
        assert ip.sample_rate == self.sh.sample_rate / n

        ip.seek(seek)
        assert abs(ip.time - (self.sh.start_time
                              + seek / ip.sample_rate)) < 1. * u.ns
        integrated = ip.read(10)
        data = integrated['data']
        count = integrated['count']
        assert ip.tell() == seek+10

        assert st.dtype is ref_data.dtype is data.dtype
        assert data.shape == ref_data.shape
        assert np.allclose(data, ref_data)
        assert np.all(count == n)

        r = repr(ip)
        assert f"step={n}" in r
        assert 'average=False' in r

    @pytest.mark.parametrize('samples_per_frame', (1, 4, 10))
    @pytest.mark.parametrize('n', (1, 3))
    def test_integrate_n_part(self, n, samples_per_frame):
        # Also test same as above by passing on start.
        ref_data = self.raw_power[121 * n:131 * n].reshape(-1, n, 2).sum(1)

        st = Square(self.sh)
        ip = Integrate(st, n/self.sh.sample_rate, start=121*n, average=False,
                       samples_per_frame=samples_per_frame)
        assert abs(ip.start_time - (self.sh.start_time
                                    + 121 * n / self.sh.sample_rate)) < 1.*u.ns
        assert ip.sample_rate == self.sh.sample_rate / n

        integrated = ip.read(10)
        data = integrated['data']
        count = integrated['count']
        assert ip.tell() == 10
        assert abs(ip.time - (self.sh.start_time
                              + 131 * n / self.sh.sample_rate)) < 1. * u.ns

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
        assert abs(ip.time - (self.sh.start_time
                              + 161 * n / self.sh.sample_rate)) < 1. * u.ns
        assert st.dtype is ref_data.dtype is data.dtype
        assert data.shape == ref_data.shape
        assert np.allclose(data, ref_data)
        assert np.all(count == n)

    @pytest.mark.parametrize('samples_per_frame', (1, 4, 10))
    @pytest.mark.parametrize('n', (1, 3))
    def test_integrate_n_via_time_part(self, n, samples_per_frame):
        # Also test same as above by passing on start time.
        ref_data = self.raw_power[151*n:161*n].reshape(-1, n, 2).sum(1)
        st = Square(self.sh)
        st.seek(151 * n)
        start = st.time
        ip = Integrate(st, n/self.sh.sample_rate, start=start, average=False,
                       samples_per_frame=samples_per_frame)
        integrated = ip.read(10)
        data = integrated['data']
        count = integrated['count']
        assert ip.tell() == 10
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
        step = 2.26 / self.sample_rate
        raw = self.raw_power[:18]
        ref_data = np.add.reduceat(
            raw, np.add.accumulate([0] + expected_count[:-1]))

        st = Square(self.sh)
        ip = Integrate(st, step, average=False,
                       samples_per_frame=samples_per_frame)
        assert ip.start_time == self.sh.start_time
        assert ip.sample_rate == 1. / step

        # Square and integrate everything.
        integrated = ip.read(8)
        data = integrated['data']
        count = integrated['count']
        assert ip.tell() == 8
        assert abs(ip.time - (ip.start_time + 8 / ip.sample_rate)) < 1.*u.ns
        assert st.dtype is ref_data.dtype is data.dtype
        assert data.shape == ref_data.shape
        assert np.allclose(data, ref_data)
        assert np.all(count.T == expected_count)

        # Check we can get there with an offset as well
        start_time = self.start_time + step
        ip = Integrate(st, step, start=start_time, average=False,
                       samples_per_frame=samples_per_frame)
        assert abs(ip.start_time - start_time) < 1*u.ns
        integrated2 = ip.read(7)
        assert np.all(integrated2 == integrated[1:])

        start_time = self.start_time + 3 * step
        ip = Integrate(st, step, start=start_time, average=False,
                       samples_per_frame=samples_per_frame)
        assert abs(ip.start_time - start_time) < 1.*u.ns
        integrated2 = ip.read(5)
        assert np.all(integrated2 == integrated[3:])

    def test_times_wrong(self):
        with pytest.raises(ValueError):
            Integrate(self.sh, start=self.start_time-1.*u.s)
        with pytest.raises(ValueError):
            Integrate(self.sh, start=self.start_time+3.*u.s)
        with pytest.raises(AssertionError):
            Integrate(self.sh, step=1.*u.hr)


class TestFold(FakePulsarBase):
    def test_input_data(self):
        # Spot check on pulsar simulation.
        assert np.all(self.raw_data[::125] == 10.)
        assert np.all(self.raw_data[1:125] == 0.125)
        assert np.all(self.raw_data[1::125] == 0.125)
        assert np.all(self.raw_data[50::125] == 0.125)
        assert np.all(self.raw_data[124::125] == 0.125)

    def test_step_shorter_than_period(self):
        step = 10 * u.ms
        fh = Fold(self.sh, self.n_phase, self.phase, step,
                  samples_per_frame=1, average=False)
        fh.seek(0)
        fr = fh.read(3)
        # Note: API for accessing counts may change.
        fr_count = fr['count']
        fr_data = fr['data']

        # With 10 ms and a sample rate of 10 kHz, we should always get
        # 100 samples
        assert np.all(fr_count.sum(1) == 100)

        # period = 12.5 ms; hence of first sample only the first 40 bins
        # should have gotten anything.  (Rounding may spill it in one
        # further bin.)
        assert np.all((fr_count[0, :40] == 3) | (fr_count[0, :40] == 2))
        assert np.all(fr_count[0, 41:] == 0)
        # further samples, shifted by 10 each:
        assert np.all(fr_count[1, :30] != 0)
        assert np.all(fr_count[1, 40:] != 0)
        assert np.all(fr_count[1, 31:39] == 0)
        assert np.all(fr_count[2, :20] != 0)
        assert np.all(fr_count[2, 30:] != 0)
        assert np.all(fr_count[2, 21:29] == 0)

        # Phases near 0 should have the pulse and the rest background.
        assert np.all(fr_data[:, (0, 1, -1)].sum(1) > 10)
        assert np.all(fr_data[:, 2:49] <= 0.125 * 3)

    def test_step_longer_than_period(self):
        step = 30 * u.ms
        fh = Fold(self.sh, self.n_phase, self.phase, step,
                  samples_per_frame=1, average=False)
        fh.seek(0)
        fr = fh.read(10)
        fr_count = fr['count']
        fr_data = fr['data']
        # Compare the total counts of all the samples.
        assert np.all(fr_count.sum(1) == 300)

        # Test the output on and off gates.  3 gates -> 7.5 bins.
        pulse_power = (fr_data[:, (0, 1, -1)].sum(1)
                       / fr_count[:, (0, 1, -1)].sum(1))
        assert np.all(np.abs(pulse_power - 10. / 7.5 - 0.125) < 0.5), \
            "On-gate power is incorrect."

        assert np.allclose(fr_data[:, 2:-1] / fr_count[:, 2:-1], 0.125), \
            "Off-gate power is incorrect."

        # Try with offset time
        fh2 = Fold(self.sh, self.n_phase, self.phase, step,
                   start=self.sh.start_time + step,
                   samples_per_frame=1, average=False)
        fr2 = fh2.read(9)
        assert np.all(fr2 == fr[1:])

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
        fr = fh.read()
        assert np.all(fr[:, 2:-1] == 0.125), \
            "Average off-gate power is incorrect."
        fh1 = Fold(self.sh, self.n_phase, self.phase, step,
                   start=self.start_time + step, average=True)
        fr1 = fh1.read()
        assert np.all(fr1 == fr[1:]), \
            "Fold applied start offset incorrectly."
        fh2 = Fold(self.sh, self.n_phase, self.phase, step,
                   start=self.start_time+2*step, average=True)
        fr2 = fh2.read()
        assert np.all(fr2 == fr[2:]), \
            "Fold applied start offset incorrectly."

    def test_read_whole_file(self):
        ref_data = self.raw_data[:, 0]
        phase = self.phase(self.start_time
                           + np.arange(self.shape[0]) / self.sample_rate)
        i_phase = ((phase.to_value(u.cycle) * self.n_phase) %
                   self.n_phase).astype(int)
        expected = np.bincount(i_phase, ref_data) / np.bincount(i_phase)

        fh = Fold(self.sh, self.n_phase, self.phase, average=True)
        assert abs(fh.stop_time - self.sh.stop_time) < 1. * u.ns
        fr = fh.read(1)
        assert np.all(fr[:, 2:-1] == 0.125), \
            "Average off-gate power is incorrect."
        assert np.all(fr[0, :, 0] == expected)

    def test_read_part(self):
        ref_data = self.raw_data[10000:, 0]
        start = self.start_time + 10000 / self.sample_rate
        phase = self.phase(start + (np.arange(self.shape[0] - 10000)
                                    / self.sample_rate))
        i_phase = ((phase.to_value(u.cycle) * self.n_phase)
                   % self.n_phase).astype(int)
        expected = np.bincount(i_phase, ref_data) / np.bincount(i_phase)

        fh = Fold(self.sh, self.n_phase, self.phase, average=False,
                  start=start)
        assert abs(fh.start_time - start) < 1. * u.ns
        assert abs(fh.stop_time - self.sh.stop_time) < 1. * u.ns
        fr = fh.read(1)
        assert np.all(fr['count'].sum((0, 1)) == 6000)
        average = fr['data'][0] / fr['count'][0]
        assert np.all(average[2:-1] == 0.125), \
            "Average off-gate power is incorrect."
        assert np.all(average[:, 0] == expected), \
            "On-gate power is incorrect."

    def test_times_wrong(self):
        with pytest.raises(ValueError):
            Fold(self.sh, 8, self.phase, start=self.start_time - 1. * u.s)
        with pytest.raises(ValueError):
            Fold(self.sh, 8, self.phase, start=self.start_time + 3. * u.s)
        with pytest.raises(AssertionError):
            Fold(self.sh, 8, self.phase, step=1.*u.hr)


class TestFoldwithPhase(TestFold, UsePhaseClass):
    pass


class TestIntegratePhase(FakePulsarBase):
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


class TestIntegratePhasewithPhase(TestIntegratePhase, UsePhaseClass):
    pass


class TestStack(FakePulsarBase):
    @pytest.mark.parametrize('samples_per_frame', (1, 16))
    def test_basics(self, samples_per_frame):
        ref_data = self.raw_data.reshape(-1, 25, 5, 2).mean(2)

        fh = Stack(self.sh, 25, self.phase,
                   samples_per_frame=samples_per_frame)
        assert fh.start_time == self.sh.start_time
        assert fh.stop_time == self.sh.stop_time
        assert fh.sample_rate == 1. / u.cycle
        assert fh.samples_per_frame == samples_per_frame
        fh.seek(5)
        assert abs(fh.time - (self.sh.start_time + 5 / self.F0)) < 1.*u.ns

        data = fh.read(2)
        assert np.all(data == ref_data[:2])
        fh.seek(10)
        data = fh.read(3)
        assert np.all(data == ref_data[10:13])
        data = fh.read()
        assert np.all(data == ref_data[13:])

    @pytest.mark.parametrize('samples_per_frame', (1, 16))
    def test_sliced_input(self, samples_per_frame):
        ref_data = self.raw_data[-360:-110].reshape(-1, 25, 5, 2).mean(2)

        fh = Stack(self.sh[-360:-10], 25, self.phase,
                   samples_per_frame=samples_per_frame)
        assert fh.shape == ref_data.shape
        data = fh.read()
        assert np.all(data == ref_data)

    def test_offset(self):
        ref_data = self.raw_data[124:-1].reshape(-1, 25, 5, 2).mean(2)

        fh = Stack(self.sh, 25, self.phase, start=124)
        assert abs(fh.start_time
                   - self.start_time - 124 / self.sample_rate) < 1. * u.ns
        assert abs(fh.stop_time
                   - (self.sh.stop_time - 1 / self.sample_rate)) < 1. * u.ns
        assert fh.sample_rate == 1. / u.cycle

        data = fh.read(2)
        assert np.all(data == ref_data[:2])
        fh.seek(10)
        assert abs(fh.time - self.start_time - 124 / self.sample_rate
                   - 10 / self.F0) < 1. * u.ns
        data = fh.read()
        assert np.all(data == ref_data[10:])

    @pytest.mark.parametrize('item', [
        slice(10, 100), slice(-10, None), slice(None, 10), slice(None),
        (slice(10, 100), 0)])
    def test_slice(self, item):
        fh = Stack(self.sh, 25, self.phase, start=124)
        sliced = fh[item]
        if not isinstance(item, tuple):
            item = (item,)
        start, stop, _ = item[0].indices(fh.shape[0])
        expected = self.start_time + 124 / self.sample_rate + start / self.F0
        assert abs(sliced.start_time - expected) < 1.*u.ns
        assert abs(sliced.time - expected) < 1.*u.ns
        expected = self.start_time + 124 / self.sample_rate + stop / self.F0
        assert abs(sliced.stop_time - expected) < 1.*u.ns
        sliced.seek(5)
        expected = sliced.start_time + 5 / self.F0
        assert abs(sliced.time - expected) < 1.*u.ns
        sliced.seek(0)
        ref_data = self.raw_data[124:-1].reshape(-1, 25, 5, 2).mean(2)[item]
        assert sliced.shape == ref_data.shape
        data = sliced.read()
        assert_array_equal(data, ref_data)

    def test_integrate_stack(self):
        fh = Stack(self.sh, 25, self.phase)
        data = fh.read(3)
        ih = Integrate(fh, 3)
        data2 = ih.read(1)
        assert np.all(data2 == data.mean(0))
        assert ih.tell() == 1
        assert abs(ih.time - (self.sh.start_time + 3 / self.F0)) < 1.*u.ns
