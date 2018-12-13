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
        self.n_phase = 50

    def phase(self, t):
        dt = (t - self.start_time).to(u.s)
        F0 = 1.0 / (self.period_bin / self.sh.sample_rate)
        return F0 * dt

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

    def test_under_period(self):
        # Test when folding time smaller than the pulse period.
        fold_time = 11 * u.ms
        eff_n_phase = (11 * u.ms / (self.period_bin / self.sh.sample_rate) *
                       self.n_phase).to(u.Unit(1))
        self.fh = Fold(self.sh, self.n_phase, self.phase, fold_time,
                       samples_per_frame=1)
        self.fh.seek(0)
        fr = self.fh.read(3)

        for ii, count in enumerate(fr.count):
            u_eff_phb = np.where(count == 0)
            assert np.isclose(self.n_phase - len(u_eff_phb[0]) / 8,
                              eff_n_phase, 1), \
                ("Sample {}'s does not have correct effective phase bin "
                 "number.".format(ii))

    def test_over_period(self):
        # Test when folding time is bigger than one or multiple pulse period
        fold_time = 26 * u.ms
        self.fh = Fold(self.sh, self.n_phase, self.phase, fold_time,
                       samples_per_frame=1)
        self.fh.seek(0)
        fr = self.fh.read(10)
        # Compare the total counts of all the samples.
        tot_counts = np.sum(fr.count, axis=1)
        abs_diff = np.abs(tot_counts - tot_counts.mean())
        assert abs_diff.max() <= 1, ("Folding counts are not correct for over "
                                     "period folding.")
        # Test the output result
        ph0_bins = [0, 1, -1, -2]
        pulse_power = np.sum(fr[:, ph0_bins, 0, 0], axis=1)
        assert np.logical_and(pulse_power[0] > 30, pulse_power[0] < 33), \
            "Folding power is not correct for over period folding."

        assert np.logical_and(np.all(pulse_power[1:] > 20),
                              np.all(pulse_power[1:] < 23)), \
            "Folding power is not correct for over period folding."
        # Test average
        self.fh2 = Fold(self.sh, self.n_phase, self.phase, fold_time,
                        samples_per_frame=20, average=True)
        self.fh2.seek(0)
        fr2 = self.fh2.read(10)
        assert np.all(fr2[:, 2:-1] == 0.125), "Averaged result is not correct."
