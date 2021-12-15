"""Test script for pulsar gating
"""
import os
import pytest
import numpy as np
import astropy.units as u
from astropy.time import Time

from baseband_tasks.pulse_gate import GatePulse
from baseband_tasks.phases import PolycoPhase, Phase
from baseband_tasks.generators import StreamGenerator


test_data = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')


class TestPulseGating:
    def setup(self):
        self.start_time = Time('2018-05-06T23:00:00', format='isot',
                               scale='utc')
        self.polyco_file = os.path.join(test_data, 'B1937_polyco.dat')
        self.polyco = PolycoPhase(self.polyco_file)
        self.sample_rate = 128. * u.kHz
        self.shape = (164000, 2)
        self.gp_sample = 64000
        self.gp = StreamGenerator(self.make_giant_pulse,
                                  shape=self.shape, start_time=self.start_time,
                                  sample_rate=self.sample_rate,
                                  samples_per_frame=1000, dtype=np.float32,
                                  frequency=[299.936, 300.064]*u.MHz,
                                  sideband=np.array((1, -1)))

    @classmethod
    def make_giant_pulse(self, sh):
        data = np.empty((sh.samples_per_frame,) + sh.shape[1:], sh.dtype)
        do_gp = sh.tell() + np.arange(sh.samples_per_frame) == self.gp_sample
        data[...] = do_gp[:, np.newaxis]
        return data

    @pytest.mark.parametrize('gate', [[0.2, 0.5], [1.3, 1.4], [0.8, 0.2],
                                      [3.8, 3.2]])
    @pytest.mark.parametrize('tol', [None, 0.001])
    @pytest.mark.parametrize('pulse_period', [None, 1.6 * u.ms, 2*u.min])
    def test_pulse_gate_build(self, gate, tol, pulse_period):
        gate_pulse = GatePulse(self.gp, self.polyco, gate, tol, pulse_period)
        assert gate_pulse.gate[0] == gate[0] - np.modf(gate[0])[1]
        if gate[0] > gate[1]:
            assert gate_pulse.gate[1] == gete[1] + 1 - np.modf(gate[0])[1]
