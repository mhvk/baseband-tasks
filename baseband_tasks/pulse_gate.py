"""Functions to produce pulse gated data(only with the pulse part)
"""
import numpy as np
import warnings
import astropy.units as u
from .base import getattr_if_none, SetAttribute
from .phases import Phase
from .shaping import GetSlice
from .integration import Integrate


class GatePulse:
    """Return the gated pulse data.

    Parameters
    ----------
    ih : `baseband` file handle or task stream reader
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
    tol : `~astropy.units.Quantity`
        The tolarence of the pulse phaes gating in the units of cycle. Tolarence
        can not be bigger than 10% of the gate size. Default is 5% of the given
        gate size.
    pulse_period: `~astropy.unit.Quantity` object, optional
        Input pulse period. If not given, it will search from the input
        callable `phase` object. If given, it has to be in the unit of time.

    """
    def __init__(self, ih, phase, gate, tol=None, pulse_period=None, dtype=None):
        self.ih = ih
        if pulse_period is None: # Get pulse period from phase class
            try:
                self.pulse_period = 1.0 / phase.apparent_spin_freq(ih.start_time)
            except AttributeError:
                raise ValueError("Can not find pulse period from the `phase` "
                                 "class. Please provide a valid"
                                 " `pulse_period`.")
        else:
            self.pulse_period = pulse_period.to(u.s)

        self.dtype = getattr_if_none(ih, 'dtype', dtype)
        # Use the phase searching code in the Integrate class.
        self.phase = phase
        self.gate = self.verify_gate(gate)
        self.samples_per_period = int(self.pulse_period * self.ih.sample_rate)
        self.phase_per_sample = 1.0 * u.cycle / self.samples_per_period
        # Check the tolerence range, tolerance has to smaller than 10% of gate size
        if tol is None:
            tol =  (self.gate[1] - self.gate[0]) / 20
        assert tol > 0
        assert tol < (self.gate[1] - self.gate[0]) / 10
        self.tol = tol
        # Compute the number of samples in the tolerance phase, which means, the
        # Gated data will be accurate to `self.tol_sample` of samples.
        self.tol_sample = tol / self.phase_per_sample
        if self.tol_sample < 1:
            warnings.warn("Tolarence is smaller than one input time sample. "
                          "Use one time sample as the tolarence and the edge of"
                          "the gate will not as accurate as requested.")
            self.tol_sample = np.ceil(self.tol_sample)
            self.tol = self.tol_sample * self.phase_per_sample
        self.tol_sample = self.tol_sample.astype(int)
        self.pulse_offset = 0
        self.gate_offsets = self.get_gate_offsets()

    def read(self, gate_index=None):
        """Read the next pulse.

        Parameter
        ---------
        pulse_index: int, optional
            The index of gate to read. If not provide, it reads the next pulse
            from the current pulse_offset.
        """
        if gate_index is None:
            gate_index = self.pulse_offset

        if gate_index >= len(self.gate_offsets):
            raise ValueError("The requested gate index is beyond the total data"
                             "stream.")

        gsh = GetSlice(self.ih, slice(self.gate_offsets[0][gate_index],
                                self.gate_offsets[1][gate_index]))
        data = gsh.read()
        self.pulse_offset = gate_index + 1
        return data, gsh

    def next_nsample_phase(self, n_samples):
        """Compute the pulse phase from the current offset with a resolution of input phase tolerance

        Parameters
        ----------
        n_sample: int
            The phase for next n upstream samples.

        """
        # Compute the data time axis
        #
        time_axis = (np.arange(0, n_samples,
                               self.tol_sample) / self.ih.sample_rate)
        time_axis = self.ih.time + time_axis
        pulse_phase = self.phase(time_axis)
        # Search the gate
        return time_axis, pulse_phase

    def get_gate_offsets(self):
        """Get the offsets for the gate time.

        Phase is assumed to increase monotonously with time.
        """
        n_sample = self.ih.shape[0] - self.ih.offset
        times, phase = self.next_nsample_phase(n_sample)
        n_cycles = phase.int[-1] - phase.int[0]
        # Find gates for each period
        start_phase_int = np.modf(phase[0].value)[1]
        # The data's start phase does not cover the whole gate, ignore, and go
        # to the next phase.
        if start_phase_int * u.cycle + self.gate[0] < phase[0]:
            start_phase_int += 1
        end_phase_int = np.modf(phase[-1].value)[1]
        search_gate = np.arange(start_phase_int, end_phase_int + 1)
        search_gate = (Phase(np.broadcast_to(search_gate,
                                             (2, len(search_gate)))) +
                       Phase(self.gate).reshape(2,1))
        gate_idx = np.searchsorted(phase.value, search_gate.value)
        # Cut off the gates that is beyond the total samples which is created
        # by search gate.
        cut_off = min(np.searchsorted(gate_idx[0], len(times)),
                      np.searchsorted(gate_idx[1], len(times)))
        # if self.gate[0].value  == 0.8:
        #     raise ValueError
        gate_idx = gate_idx[:, 0:cut_off]
        gate_times = times[gate_idx]
        # Map the gate_times to the upstream samples
        gate_offsets = (((gate_times - self.ih.time) *
                          self.ih.sample_rate).to(u.one)).astype(int)
        return gate_offsets

    def verify_gate(self, gate):
        assert len(gate) == 2
        if gate[0] > gate[1]: #
            gate[1] += 1
        assert gate[1] - gate[0] < 1
        # Normalize gate to 0 to 1
        result = np.array(gate) - np.modf(gate[0])[1]
        if not hasattr(result, 'unit'):
            return result * u.cycle
        else:
            return result.to(u.cycle)

    def close(self):
        self.ih.close()
