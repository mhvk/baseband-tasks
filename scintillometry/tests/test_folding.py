# Licensed under the GPLv3 - see LICENSE
"""Full-package tests of pulse folding."""

import pytest
import numpy as np
import astropy.units as u
from astropy.time import Time

from ..base import Task
from ..generators import EmptyStreamGenerator
from ..pulse_folding import TimeFold
from ..functions import Square


class TestFoldingBase:
    def setup(self):
        self.start_time = Time('2010-11-12T13:14:15')
        self.sample_rate = 10. * u.kHz
        self.shape = (6000, 2, 4)
        self.nh = EmptyStreamGenerator(shape=self.shape,
                                       start_time=self.start_time,
                                       sample_rate=self.sample_rate,
                                       samples_per_frame=200, dtype=np.float)
        self.sh = Task(self.nh, self.pulse_simulate, samples_per_frame=1)
        self.period_bin = 125
        self.phase_bin = 50

    def phase(self, t):
        dt = (t - t[0]).to(u.s)
        F0 = 1.0 / (self.period_bin / self.sh.sample_rate)
        return F0 * dt

    def pulse_simulate(self, fh, data):
        idx = fh.tell()
        data [:]= 0
        data += (10 if idx % self.period_bin ==0 else 0.125)
        return data


class TestTimeFolding(TestFoldingBase):
    def test_input_data(self):
        indata = self.sh.read(1000)
        pulses = np.where(indata[:, 0, 0] == 10)[0]
        # Check if the input data is set up right.
        assert (len(pulses) == 8), "Pulses are not simulated right."

    def test_under_period(self):
        # Test when folding time smaller than the pulse period.
        self.fold_time = 11 * u.ms
        eff_phase_bin = (11* u.ms / (self.period_bin / self.sh.sample_rate) * \
                         self.phase_bin).to(u.Unit(1))
        self.fh = TimeFold(self.sh, self.phase_bin, self.fold_time, \
                           self.phase, samples_per_frame=20)
        self.fh.seek(0)
        fr = self.fh.read(3)

        for ii in range(self.fh.counts.shape[0]):
            u_eff_phb =  np.where( self.fh.counts[ii] == 0)
            assert (self.phase_bin - len(u_eff_phb[0]) == eff_phase_bin), \
                    "Sample {}'s does not have correct effective phase bin \
                    number.".format(ii)

    def test_over_period(self):
        # Test when folding time is bigger than one or multiple pulse period
        self.fold_time = 26 * u.ms
        self.fh = TimeFold(self.sh, self.phase_bin, self.fold_time, \
                           self.phase, samples_per_frame=20)
        self.fh.seek(0)
        fr = self.fh.read(10)
        # Compare the total counts of all the samples.
        tot_counts = np.sum(self.fh.counts, axis=1)
        abs_diff = np.abs(tot_counts - tot_counts.mean())
        assert abs_diff.max() <= 1, ("Folding counts are not correct for over " \
                                   "period folding.")
        # Test the output result
        ph0_bins = [0,1,-1,-2]
        pulse_power = np.sum(fr[:,ph0_bins, 0, 0], axis=1)
        assert np.logical_and(pulse_power[0] > 30, pulse_power[0] < 33), \
               "Folding power is not correct for over period folding."

        assert np.logical_and(np.all(pulse_power[1:] > 20),
                              np.all(pulse_power[1:] < 23)), \
               "Folding power is not correct for over period folding."
        # Test average
        self.fh2 = TimeFold(self.sh, self.phase_bin, self.fold_time, \
                           self.phase, samples_per_frame=20, average=True)
        self.fh2.seek(0)
        fr2 = self.fh2.read(10)
        pulse_power2 = np.sum(fr2[:,ph0_bins, 0, 0], axis=1)
        assert np.all(fr2[:,2:-1] == 0.125), "Averaged result is not correct."
