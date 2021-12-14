"""Test script for pulsar gating
"""
import baseband_tasks.pulse_gate as pg
from baseband_tasks.phases import PolycoPhase, Phase


test_data = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')


class TestPulseGating:
    def setup(self):
        self.start_time = Time('2018-05-06T23:00:00', format='isot',
                               scale='utc')
        self.polyco_file = os.path.join(test_data, 'B1937_polyco.dat')
        self.polyco = Polyco(self.polyco_file)
        self.sample_rate = 128. * u.kHz
        self.shape = (164000, 2)
        self.gp_sample = 64000
        self.gp = StreamGenerator(self.make_giant_pulse,
                                  shape=self.shape, start_time=self.start_time,
                                  sample_rate=self.sample_rate,
                                  samples_per_frame=1000, dtype=np.complex64,
                                  frequency=300*u.MHz,
                                  sideband=np.array((1, -1)))

    @classmethod
    def make_giant_pulse(self, sh):
        data = np.empty((sh.samples_per_frame,) + sh.shape[1:], sh.dtype)
        do_gp = sh.tell() + np.arange(sh.samples_per_frame) == self.gp_sample
        data[...] = do_gp[:, np.newaxis]
        return data
