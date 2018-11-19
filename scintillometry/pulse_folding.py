# Licensed under the GPLv3 - see LICENSE

import operator
import warnings

import numpy as np
import astropy.units as u
from astropy.utils import lazyproperty

from .base import TaskBase


__all__ = ['Fold']


class Fold(TaskBase):
    """ Fold pulses.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream, with time as the first axis.

    fold_periods : int
        Number of periods folded into one pulse phase.

    phase_bin : int
        Number of bins in one phase period.

    pulse_period : float or `astropy.units.quantity`
        The apparent pulse period at the beginning folding time.

    phase_method : function
        The method to compute pulse phases at a given time.

    samples_per_frame: int, optional
        Number of request folded period. If not given, it assumes 1


    """

    def __init__(self, ih, fold_periods, phase_bin, pulse_period,
                 phase_method, samples_per_frame=None):
        self.ih = ih
        self.fold_periods = fold_periods
        self.phase_bin = phase_bin
        self.pulse_period = pulse_period
        # Setup the number of returned periods.
        if samples_per_frame is None:
            samples_per_frame = 1
        print(type(samples_per_frame))
        # Compute the result sample rate
        sample_rate = (1.0 / (fold_periods * pulse_period)).to(u.Hz)
        self.phase_method = phase_method
        super().__init__(ih, samples_per_frame=samples_per_frame,
                         shape=(samples_per_frame, phase_bin) + ih.shape[1:],
                         sample_rate=sample_rate, dtype=ih.dtype)

    def eval_phase(self, t):
        """ Evaluate the pulse phase at a given time. """
        return self.phase_method(t)

    def _read_frame(self, frame_index):
        # Move to the start position
        result = np.zeros(self.shape, dtype=self.dtype)
        counts = np.zeros(self.shape[0:2], dtype=np.int)
        self.ih.seek(frame_index * self.samples_per_frame)
        req_samples = int(self.samples_per_frame / self.sample_rate * \
                      self.ih.sample_rate)
        start_time = self.ih.time
        data = self.ih.read(req_samples)
        time_axis = start_time + np.arange(req_samples) / \
                    self.ih.sample_rate
        # Evaluate the phase
        phases =  self.eval_phase(time_axis)
        # Map the phases to result indice.
        # normalize the phase
        # TODO, give a normalize parameter
        phases = phases - phases[0]
        # TODO Use the apparent period here
        sample_index, _ = np.divmod(phases, self.fold_periods)

        sample_index = (sample_index.astype(int)).value
        # Compute the phase bin index
        phase_index = ((np.modf(phases)[0] * self.phase_bin).astype(int)).value
        print(data.shape, result.shape)
        # Construct the sum index map
        index_pair = np.column_stack((sample_index, phase_index))
        print(type(index_pair))
        index_set = list(set(map(tuple,index_pair)))
        sum = np.zeros((len(index_set),) + self.shape[2:])
        sum_map = dict(zip(index_set, sum))
        # Add data to the same index
        for ii, d in enumerate(data[:]):
            sum_map[(sample_index[ii], phase_index[ii])] += d
            counts[(sample_index[ii], phase_index[ii])] += 1
        for idx, rd in sum_map.items():
            result[idx] = rd
        self.dd = data
        self.sample_index = sample_index
        self.phase_index = phase_index
        self.temp = sum_map
        self.counts = counts
        return result

    def close(self):
        super().close()
