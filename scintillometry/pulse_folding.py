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

    phase_bin : int
        Number of bins in one phase period.

    n_fold_phase : int or `~astropy.units.Quantity`,
        Number of phase cycles folded into one pulse phase.

    phase_method : function
        The method to compute pulse phases at a given time.

    samples_per_frame: int, optional
        Number of request folded period. If not given, it assumes 1


    """

    def __init__(self, ih, phase_bin, n_fold_phase, phase_method,
                 samples_per_frame=None):
        self.ih = ih
        self.phase_bin = phase_bin
        self.n_fold_phase= n_fold_phase
        # Setup the number of returned samples
        if samples_per_frame is None:
            samples_per_frame = 1

        self.phase_method = phase_method
        super().__init__(ih, samples_per_frame=samples_per_frame,
                         sample_rate=n_fold_phase,
                         shape=(samples_per_frame, phase_bin) + ih.shape[1:],
                         dtype=ih.dtype)

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
        sample_index, _ = np.divmod(phases, self.n_fold_phase)
        sample_index = (sample_index.astype(int)).value
        # Compute the phase bin index
        phase_index = ((np.modf(phases)[0] * self.phase_bin).astype(int)).value
        index_pair = np.column_stack((sample_index, phase_index))
        # Do fold
        np.add.at(result, (sample_index, phase_index), data)
        # Consturct the fold counts
        np.add.at(counts, (sample_index, phase_index), 1)
        # NOTE this part is for optimizing
        # print(type(index_pair))
        # index_set = list(set(map(tuple,index_pair)))
        # sum = np.zeros((len(index_set),) + self.shape[2:])
        # sum_map = dict(zip(index_set, sum))
        # # Add data to the same index
        # for ii, d in enumerate(data[:]):
        #     sum_map[(sample_index[ii], phase_index[ii])] += d
        #     counts[(sample_index[ii], phase_index[ii])] += 1
        # for idx, rd in sum_map.items():
        #     result[idx] = rd
        self.dd = data
        self.sample_index = sample_index
        self.phase_index = phase_index
        self.counts = counts
        return result
