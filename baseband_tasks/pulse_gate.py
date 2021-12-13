"""Functions to produce pulse gated data(only with the pulse part)
"""
import numpy as np
from .phases import PolycoPhase, PintPhase
from .dispersion import DedisperseShift


def gate_data(dedisperse_fh, phase, gate, tol=0.01, pulse_period=None):
    """Return the gated pulse data.

    Parameters
    ----------
    dedisperse_fh : `DedisperseShift` task stream reader
        Input dedisperse shifted data stream
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
    # Compute the rough phase resolution
    phase_per_sample = pulse_period * dedisperse_fh.sample_rate
    # Compute the tolarence sample numbers
    tol_sample = tol / phase_per_sample
    # Compute the data time
    time_axis = (np.arange(0, dedisperse_fh.shape[0], tol_sample) /
                 dedisperse_fh.sample_rate)
    time_axis = dedisperse_fh.start_time + time_axis
    pulse_phase = phase(time_axis)
    # Search the gate
