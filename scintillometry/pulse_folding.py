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

    pulse_period : float
        The apparent pulse period at the beginning folding time.

    phase_method : function
        The method to compute pulse phases at a given time.

    samples_per_frame: int, optional
        Number of request folded period. If not given, it assumes 1


    """

    def __init__(self, ih, fold_periods, phase_bin, pulse_period,
                 phase_method, samples_per_frame=None):
        self.ih = ih
        # Setup the number of returned periods.
        if samples_per_frame is None:
            samples_per_frame = 1

        # Compute the result sample rate
        period = self.period_method()
        sample_rate = fold_periods * pulse_period

        self.phase_method = phase_method

        super().__init__(ih, samples_per_frame=samples_per_frame,
                         shape=(samples_per_frame, phase_bin) + ih.shape[1:],
                         sample_rate=sample_rate, dtype=ih.dtype)

    def eval_phase(self, t):
        """ Evaluate the pulse phase at a given time. """
        return self.phase_method(time)

    def fold(self):
        """ Fold data to one phase"""
        pass

    def _read_frame(self, frame_index):
        self.ih.seek(frame_index * self.samples_per_frame)
        time_axis = self.ih.time + np.arange(self.samples_per_frame) / \
                    self.ih.sample_rate
        


    def task(self, data):
        result = np.zeros(self.shape, self.dtype)
        time_axis = np.linsapce(self.ih.start_time, self.ih

    def close(self):
        super().close()
