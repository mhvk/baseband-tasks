# Licensed under the GPLv3 - see LICENSE

import operator
import warnings

import numpy as np
import astropy.units as u
import astropy.time as time
from .base import TaskBase


__all__ = ['TimeFold',]

#TODO add Fold base class for the future folding
class TimeFold(TaskBase):
    """ Fold pulse using a fixed time interval.

    NOTE
    ----
    This method does not require the folding time to be the pulse period.
    However, the time integration for each phase bin will not be uniform.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream, with time as the first axis.
    n_phase : int
        Number of bins in one result phase period.
    phase : callable
        The method to compute pulse phases at a given time.
        Note, the input time is in the format of '~astropy.time.Time' object.
        The output phase should be an array.
    fold_time : float or `~astropy.units.Quantity`,
        Time interval for folding into one pulse period(result sample).
        Default unit is second.
    average : bool
        If the output pulse profile averaged by the number of time integration
        in each phase bin.
    samples_per_frame : int, optional
        Number of request folded period. If not given, it assumes 1.
    """

    def __init__(self, ih, n_phase, phase, fold_time, average=False,
                 samples_per_frame=1):
        self.n_phase = n_phase
        self.fold_time = u.Quantity(fold_time, u.s)
        self.phase = phase
        self.average = average

        #NOTE The left over time samples will be truncated. Need warning msg
        n = int(((ih.stop_time
                  - ih.start_time) / self.fold_time).to(u.Unit(1)))

        nsample = samples_per_frame * (n // samples_per_frame)

        super().__init__(ih, samples_per_frame=samples_per_frame,
                         sample_rate=1./fold_time,
                         shape=(nsample, n_phase) + ih.shape[1:],
                         dtype=ih.dtype)

    def _read_frame(self, frame_index):
        # Move to the start position
        result = np.zeros(self.shape, dtype=self.dtype)
        counts = np.zeros(self.shape[0:2], dtype=np.int)
        self.ih.seek(frame_index * self.samples_per_frame)
        req_samples = int(self.samples_per_frame / self.sample_rate
                          * self.ih.sample_rate)
        time_offset = (np.arange(self._raw_samples_per_frame)
                       / self.ih.sample_rate)
        # Evaluate the phase
        phases = self.phase(self.ih.time + time_offset)
        data = self.ih.read(req_samples)
        # Map the phases to result indice.
        # normalize the phase
        # TODO, give a phase reference parameter, right now it is disabled
        # phases = phases - phases[0]

        # Compute the phase bin index
        phase_index = ((phases.to_value(u.one) * self.n_phase)
                        % self.n_phase).astype(int)
        sample_index = (time_offset
                        / self.fold_time).to_value(u.one).astype(int)

        # Do fold
        np.add.at(result, (sample_index, phase_index), data)
        # Consturct the fold counts
        np.add.at(counts, (sample_index, phase_index), 1)

        # NOTE below is the out put for debugging.
        self.dd = data
        self.sample_index = sample_index
        self.phase_index = phase_index
        # The counts will be added to the designed data class.
        self.counts = counts
        # This is not very efficient
        if self.average:
            counts = counts.reshape(counts.shape + (1,)
                                    * (len(result.shape) - 2))
            result /= counts
        return result
