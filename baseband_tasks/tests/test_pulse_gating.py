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
        #self.gp_sample = 64000
        self.P0 = 31.4 * u.ms
        self.P1 = 2.01 * u.us / u.s
        self.pulse_phase_location = 0.21 * u.cycle
        self.ps = StreamGenerator(self.make_pulses,
                                  shape=self.shape, start_time=self.start_time,
                                  sample_rate=self.sample_rate,
                                  samples_per_frame=1000, dtype=np.float32,
                                  frequency=[299.936, 300.064]*u.MHz,
                                  sideband=np.array((1, -1)))

    # @classmethod
    # def make_giant_pulse(self, sh):
    #     data = np.empty((sh.samples_per_frame,) + sh.shape[1:], sh.dtype)
    #     do_gp = sh.tell() + np.arange(sh.samples_per_frame) == self.gp_sample
    #     data[...] = do_gp[:, np.newaxis]
    #     return data

    def simple_phase(self, t):
        F0 = 1 / self.P0
        F1 = -self.P1 * F0 ** 2
        dt = (t - self.start_time).to(u.s)
        phase = F0 * dt + 0.5 * F1 * dt ** 2
        return Phase(phase.to_value(u.one))

    def make_pulses(self, sh):
        data = np.zeros((sh.samples_per_frame,) + sh.shape[1:], sh.dtype)
        times = sh.time + np.arange(sh.samples_per_frame) / sh.sample_rate
        phase = self.simple_phase(times)
        # Search for the phase where pulse happens
        offser_from_pulse = np.abs(phase.frac - self.pulse_phase_location)
        tol = 1.0 / (self.P0 * sh.sample_rate) * u.cycle
        pulse_sample = np.where(offser_from_pulse < tol)[0]
        data[pulse_sample, ...] = 1.0
        return data

    @pytest.mark.parametrize('gate', [[0.2, 0.5], [1.3, 1.4], [0.8, 0.2],
                                      [3.8, 3.2]])
    @pytest.mark.parametrize('tol', [None, 0.001 * u.cycle])
    @pytest.mark.parametrize('pulse_period', [None, 1.6 * u.ms, 2*u.min])
    @pytest.mark.filterwarnings("ignore:Tolarence is smaller")
    def test_pulse_gate_build(self, gate, tol, pulse_period):
        gate_pulse = GatePulse(self.ps, self.polyco, gate, tol, pulse_period)
        assert gate_pulse.gate[0] == (gate[0] - np.modf(gate[0])[1]) * u.cycle
        if gate[0] > gate[1]:
            assert gate_pulse.gate[1] == (gete[1] + 1 - np.modf(gate[0])[1]) * u.cycle

    @pytest.mark.parametrize('tol', [None, 0.001 * u.cycle])
    @pytest.mark.parametrize('pulse_period', [None, 1.6 * u.ms])
    @pytest.mark.filterwarnings("ignore:Tolarence is smaller")
    def test_computing_phase(self, tol, pulse_period):
        gate = [0.34, 0.48]
        gate_pulse = GatePulse(self.ps, self.polyco, gate, tol, pulse_period)
        phase = gate_pulse.next_nsample_phase(self.shape[0])[1]
        phase_diff = np.diff(phase.frac)
        #assert np.all(np.isclose(phase_diff[phase_diff  > 0],
        #                         gate_pulse.tol * u.cycle))
    @pytest.mark.filterwarnings("ignore:Tolarence is smaller")
    @pytest.mark.parametrize('gate', [[0.2, 0.5], [1.3, 1.4], [0.8, 0.2],
                                      [3.8, 3.2]])
    def test_get_offset(self, gate):
        gate_pulse = GatePulse(self.ps, self.polyco, gate)
        gate_offsets = gate_pulse.get_gate_offsets()
        # Compute the gate time
        gate_times = (gate_pulse.ih.time + gate_pulse.ih.offset + gate_offsets /
                      gate_pulse.ih.sample_rate)
        gate_phase = gate_pulse.phase(gate_times)
        # Tolerance is higher than the phase resolution
        if gate_pulse.tol_sample == 1:
            # Since the phase per each sample is averaged, the one sample of
            # accurate will not be possible. We relax our tolerance 50%.
            tol = gate_pulse.tol * 1.5
        else:
            tol = gate_pulse.tol
        # If the fraction part is negative add 1
        gate_phase.frac.value[gate_phase.frac.value < 0] += 1
        assert np.all(np.isclose(gate_phase[0].frac.value,
                                 gate_pulse.gate[0].value,
                                 atol=tol.value))
        compare_gate_end = gate_phase[1].frac.value
        if gate_pulse.gate[1].value > 1:
            compare_gate_end += 1
        assert np.all(np.isclose(compare_gate_end,
                                 gate_pulse.gate[1].value,
                                 atol=tol.value))

    def test_read_gated_pulse(self):
        gate_pulse = GatePulse(self.ps, self.simple_phase, [0.2, 0.25],
                               pulse_period=self.P0)
        pulse, gsh = gate_pulse.read()
        pulse_pos = np.where(pulse == 1.0)
        time_axis = gsh.time - np.arange(pulse.shape[0])[::-1] / gsh.sample_rate
        phase = self.simple_phase(time_axis)
        assert np.all(np.isclose(phase[0].frac, gate_pulse.gate[0],
                      atol=gate_pulse.tol))
        assert np.all(np.isclose(phase[-1].frac, gate_pulse.gate[1],
                      atol=gate_pulse.tol))
        assert np.all(np.isclose(phase[pulse_pos[0]].frac,
                      self.pulse_phase_location, atol=gate_pulse.tol))
