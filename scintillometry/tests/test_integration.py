# Licensed under the GPLv3 - see LICENSE
"""Full-package tests of pulse folding."""

import pytest
import numpy as np
import astropy.units as u
from astropy.time import Time

from ..base import Task
from ..generators import EmptyStreamGenerator
from ..integration import Fold
from ..functions import Square


class TestFoldBase:
    def setup(self):
        self.start_time = Time('2010-11-12T13:14:15')
        self.sample_rate = 10. * u.kHz
        self.shape = (6000, 2, 4)
        self.eh = EmptyStreamGenerator(shape=self.shape,
                                       start_time=self.start_time,
                                       sample_rate=self.sample_rate,
                                       samples_per_frame=200, dtype=np.float)
        self.sh = Task(self.eh, self.pulse_simulate)
        self.period_bin = 125
        self.F0 = 1.0 / (self.period_bin / self.sh.sample_rate)
        self.n_phase = 50

    def phase(self, t):
        return self.F0 * (t - self.start_time)

    def pulse_simulate(self, fh, data):
        idx = fh.tell() + np.arange(data.shape[0])
        result = np.where(idx % self.period_bin == 0, 10., 0.125)
        result.shape = (-1,) + (1,) * (data.ndim - 1)
        data[:] = result
        return data


class TestFold(TestFoldBase):
    def test_input_data(self):
        indata = self.sh.read(1000)
        pulses = np.where(indata[:, 0, 0] == 10)[0]
        # Check if the input data is set up right.
        assert len(pulses) == 8, "Pulses are not simulated right."

    def test_fold_time_shorter_than_period(self):
        fold_time = 11 * u.ms
        fh = Fold(self.sh, self.n_phase, self.phase, fold_time,
                  samples_per_frame=1, average=False)
        fh.seek(0)
        fr = fh.read(3)
        max_count = int(np.ceil(
            (self.sample_rate / self.F0 / self.n_phase).to_value(1)))
        assert np.all(fr.count <= max_count)

        # For each of the samples, check that the correct ones have not
        # gotten any data.
        eff_n_phase = (fold_time * self.F0 * self.n_phase).to_value(1)
        for ii, count in enumerate(fr.count):
            n_count_0 = (count == 0).sum()
            assert np.isclose(self.n_phase - n_count_0, eff_n_phase, 1), \
                ("Sample {} has the wrong number of zero-count phase bins"
                 .format(ii))

    def test_fold_time_longer_than_period(self):
        fold_time = 26 * u.ms
        fh = Fold(self.sh, self.n_phase, self.phase, fold_time,
                  samples_per_frame=1, average=False)
        fh.seek(0)
        fr = fh.read(10)
        # Compare the total counts of all the samples.
        tot_counts = np.sum(fr.count, axis=1)
        assert np.all(np.abs(tot_counts - tot_counts.mean()) <= 1), \
            "Folding counts vary more than expected."
        # Test the output on and off gates.
        ph0_bins = [0, 1, -1, -2]
        pulse_power = np.sum(fr[:, ph0_bins, 0, 0], axis=1)
        assert 30 < pulse_power[0] < 33, \
            "Folded power of on-gate is incorrect."

        assert np.all((pulse_power[1:] > 20) & (pulse_power[1:] < 23)), \
            "Folding power of off-gates is incorrect."

    def test_folding_with_averaging(self):
        # Test averaging
        fh = Fold(self.sh, self.n_phase, self.phase, fold_time=26 * u.ms,
                  samples_per_frame=20, average=True)
        fh.seek(0)
        fr = fh.read(10)
        assert np.all(fr[:, 2:-1] == 0.125), \
            "Average off-gate power is incorrect."

    def test_non_integer_sample_rate_ratio(self):
        fold_time = (1./3.) * u.s
        fh = Fold(self.sh, self.n_phase, self.phase, fold_time, average=True)
        fr = fh.read(1)
        assert np.all(fr[:, 2:-1] == 0.125), \
            "Average off-gate power is incorrect."

    def test_read_whole_file(self):
        ref_data = self.sh.read()[:, 0, 0]
        phase = self.phase(self.start_time +
                           np.arange(self.sh.shape[0]) / self.sh.sample_rate)
        i_phase = ((phase * self.n_phase) % self.n_phase).astype(int)
        expected = np.bincount(i_phase, ref_data) / np.bincount(i_phase)

        fh = Fold(self.sh, self.n_phase, self.phase, average=True)
        assert abs(fh.stop_time - self.sh.stop_time) < 1. * u.ns
        fr = fh.read(1)
        assert np.all(fr[:, 2:-1] == 0.125), \
            "Average off-gate power is incorrect."
        assert np.all(fr[0, :, 0, 0] == expected)
