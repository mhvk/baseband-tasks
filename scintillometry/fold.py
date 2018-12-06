# Licensed under the GPLv3 - see LICENSE

import operator
import warnings

import numpy as np
import astropy.units as u
import astropy.time as time
from .base import TaskBase, BaseData


__all__ = ['TimeFold', ]

# TODO add Fold base class for the future folding


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
    fold_time : float or `~astropy.units.Quantity`, optional
        Time interval for folding into one pulse period(result sample).
        Default unit is second. If not given, the default value is whole file.
    average : bool
        If the output pulse profile averaged by the number of time integration
        in each phase bin.
    samples_per_frame : int, optional
        Number of request folded period. If not given, it assumes 1.
    """

    def __init__(self, ih, n_phase, phase, fold_time=None, average=False,
                 samples_per_frame=1):
        self.n_phase = n_phase
        if fold_time is None:
            self.fold_time = ih.shape[0] / ih.sample_rate
        else:
            self.fold_time = u.Quantity(fold_time, u.s)
        self.phase = phase
        self.average = average

        # NOTE The left over time samples will be truncated. Need warning msg
        nsample = int(np.floor(ih.shape[0] / ih.sample_rate / self.fold_time //
                      samples_per_frame) * samples_per_frame)

        super().__init__(ih, samples_per_frame=samples_per_frame,
                         sample_rate=1./fold_time,
                         shape=(nsample, n_phase) + ih.shape[1:],
                         dtype=ih.dtype)

    def _read_frame(self, frame_index):
        # Move to the start position
        result = np.zeros((self.samples_per_frame,) + self.shape[1:],
                          dtype=self.dtype)
        count = np.zeros((self.samples_per_frame, self.shape[1]), dtype=np.int)
        count = count.reshape(count.shape + (1,) * (len(result.shape) - 2))
        self.ih.seek(frame_index * self._raw_samples_per_frame)
        time_offset = (np.arange(self._raw_samples_per_frame) /
                       self.ih.sample_rate)
        sample_index = (time_offset /
                        self.fold_time).to_value(u.one).astype(int)
        # Evaluate the phase
        phases = self.phase(self.ih.time + time_offset)
        data = self.ih.read(self._raw_samples_per_frame)
        self.readin_data.append(data)
        self.phases.append(phases)
        # Map the phases to result indice.
        # normalize the phase
        # TODO, give a phase reference parameter, right now it is disabled
        # phases = phases - phases[0]

        # Compute the phase bin index
        phase_index = ((phases.to_value(u.one) * self.n_phase)
                       % self.n_phase).astype(int)
        self.sample_index.append(sample_index)
        self.phase_index.append(phase_index)
        # Do fold
        np.add.at(result, (sample_index, phase_index), data)
        # Consturct the fold counts
        np.add.at(count, (sample_index, phase_index), 1)

        # This is not very efficient
        if self.average:
            result /= count
        result = result.view(BaseData)
        result.count = count
        return result

    def read(self, count=None, out=None):
        """Read a number of complete samples.

        Parameters
        ----------
        count : int or None, optional
            Number of complete samples to read.  If `None` (default) or
            negative, the entire input data is processed.  Ignored if ``out``
            is given.
        out : None or array, optional
            Array to store the output in. If given, ``count`` will be inferred
            from the first dimension; the other dimension should equal
            `sample_shape`.

        Returns
        -------
        out : `~numpy.ndarray` of float or complex
            The first dimension is sample-time, and the remainder given by
            `sample_shape`.
        """
        if self.closed:
            raise ValueError("I/O operation on closed task/generator.")

        # NOTE: this will return an EOF error when attempting to read partial
        # frames, making it identical to fh.read().

        samples_left = max(0, self.shape[0] - self.offset)
        if out is None:
            if count is None or count < 0:
                count = samples_left
            out = np.empty((count, ) + self.shape[1:], dtype=self.dtype)
            out = out.view(BaseData)
            # NOTE the shape of fold_count will be chnaged
            fold_count = np.empty((count, ) + self.shape[1:], dtype=np.int)
            out.count = fold_count
        else:
            assert out.shape[1:] == self.shape[1:], (
                "'out' should have trailing shape {}".format(self.sample_shape))
            count = out.shape[0]

        # TODO: should this just return the maximum possible?
        if count > samples_left:
            raise EOFError("cannot read from beyond end of input.")

        offset0 = self.offset
        sample = 0
        while count > 0:
            # For current position, get frame plus offset in that frame.
            frame_index, sample_offset = divmod(self.offset,
                                                self._samples_per_frame)
            if frame_index != self._frame_index:
                self._frame = self._read_frame(frame_index)
                self._frame_index = frame_index

            nsample = min(count, len(self._frame) - sample_offset)
            data = self._frame[sample_offset:sample_offset + nsample]
            # Copy to relevant part of output.
            out[sample:sample + nsample] = data
            sample += nsample
            self.offset = offset0 + sample
            count -= nsample
        return out
