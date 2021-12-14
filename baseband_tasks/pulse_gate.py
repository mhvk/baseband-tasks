"""Functions to produce pulse gated data(only with the pulse part)
"""
import numpy as np
from .phases import PolycoPhase, PintPhase
from .dispersion import DedisperseShift


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

    gate : tuple
        The start gate pulse phase and the end gate pulse phase
    tol : float, optional
        The tolarence of the pulse phaes gating. Default is 0.01
    """
    def __init__(fh, phase, gate, tol=0.01, pulse_period=None):
        self.ih = fh
        if not pulse_period: # Get pulse period from phase class
            try:
                self.pulse_period = 1.0 / phase.apparent_spin_freq(fh.start_time)
            except AttributeError:
                raise ValueError("Can not find pulse period from the `phase` "
                                 "class. Please provide a valid"
                                 " `pulse_period`.")
        else:
            self.pulse_period = pulse_period

        self.samples_per_period = self.pulse_period * dedisperse_fh.sample_rate
        self.phase_per_sample = 1.0 / self.samples_per_period
        # Compute the tolarence sample numbers
        self.tol_sample = tol / self.phase_per_sample
        self.gate = gate
        self.pulse_offset = 0

    def __call__(self, number_of_pulse):
        # Compute the data time axis
        time_axis = (np.arange(0, self.samples_per_period * number_of_pulse,
                               tol_sample) / dedisperse_fh.time)
        time_axis = self.ih.time + time_axis
        pulse_phase = phase(time_axis)
        # Search the gate
        return pulse_phase
