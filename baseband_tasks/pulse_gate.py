"""Functions to produce pulse gated data(only with the pulse part)
"""
import numpy as np
import astropy.units as u
from .phases import PolycoPhase, PintPhase


class GatePulse:
    """Return the gated pulse data.

    Parameters
    ----------
    fh : `baseband` file handle or task stream reader
        Input data stream
    phase : callable
        Should return pulse phases (with or without cycle count) for given
        input time(s), passed in as an '~astropy.time.Time' object.  The output
        can be an `~astropy.units.Quantity` with angular units or a regular
        array of float (in which case units of cycles are assumed).

    gate : list of two float numbers
        The start gate pulse phase and the end gate pulse phase. The order of
        the input gate is given by `(start_gate, end_gate)`. Both gates should
        be in between [0, 1]. If the `start_gate` is bigger than the `end_gate`,
        it assumes that the gate is from (start_gate, end_gate + 1)
    tol : float, optional
        The tolarence of the pulse phaes gating. Tolarence can not be bigger
        than 10% of the gate size. Default is 5% of the given gate size.
    pulse_period: `~astropy.unit.Quantity` object, optional
        Input pulse period. If not given, it will search from the input
        callable `phase` object. If given, it has to be in the unit of time.

    """
    def __init__(self, fh, phase, gate, tol=None, pulse_period=None):
        self.ih = fh
        if pulse_period is None: # Get pulse period from phase class
            try:
                self.pulse_period = 1.0 / phase.apparent_spin_freq(fh.start_time)
            except AttributeError:
                raise ValueError("Can not find pulse period from the `phase` "
                                 "class. Please provide a valid"
                                 " `pulse_period`.")
        else:
            self.pulse_period = pulse_period.to(u.s)

        self.gate = self.verify_gate(gate)
        self.samples_per_period = self.pulse_period * self.ih.sample_rate
        self.phase_per_sample = 1.0 / self.samples_per_period
        # Check the tolerence range, tolerance has to smaller than 10% of gate size
        if tol is None:
            tol =  (self.gate[1] - self.gate[0]) / 20
        assert tol > 0
        assert tol < (self.gate[1] - self.gate[0]) / 10
        # Compute the tolarence sample numbers
        self.tol_sample = tol / self.phase_per_sample
        self.pulse_offset = 0

    def __call__(self):
        pass

    def next_nperiod_phase(self, number_of_pulse):
        """Compute the pulse phase for next N pulse period
        """
        # Compute the data time axis
        time_axis = (np.arange(0, self.samples_per_period * number_of_pulse,
                               tol_sample) / dedisperse_fh.time)
        time_axis = self.ih.time + time_axis
        pulse_phase = phase(time_axis)
        # Search the gate
        return pulse_phase

    def verify_gate(self, gate):
        assert len(gate) == 2
        if gate[0] > gate[1]: #
            gate[1] += 1
        assert gate[1] - gate[0] < 1
        # Normalize gate to 0 to 1
        return np.array(gate) - np.modf(gate[0])[1]
