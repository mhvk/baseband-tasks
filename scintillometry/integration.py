# Licensed under the GPLv3 - see LICENSE

import operator
import warnings

import numpy as np
import astropy.units as u
import astropy.time as time
from .base import Base, BaseData


__all__ = ['Fold']


class Fold(Base):
    """Fold pulse profiles in fixed time intervals.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream, with time as the first axis.
    n_phase : int
        Number of bins per pulse period.
    phase : callable
        Should return pulse phases for given input time(s), passed in as an
        '~astropy.time.Time' object.  The output should be an array of float;
        the phase can include the cycle count.
    fold_time : `~astropy.units.Quantity`, optional
        Time interval over which to fold, i.e., the sample time of the output.
        If not given, the whole file will be folded into a single profile.
    average : bool, optional
        Whether the output pulse profile should be the average of all entries
        that contributed to it, or rather the sum, in an array that also has
        a ``count`` attribute.
    samples_per_frame : int, optional
        Number of fold times to process in one go.  This can be used to
        optimize the process, though in general the default of 1 should work.
    dtype : `~numpy.dtype`, optional
        Output dtype.  Generally, the default of the dtype of the underlying
        stream is good enough, but can be used to increase precision.

    Notes
    -----
    Since the fold time is not necessarily an integer multiple of the pulse
    period, the returned profiles will generally not contain the same number
    of samples in each phase bin.  For the default of ``average=False``, the
    arrays returned by ``read`` contain the sum of the data in each bin as well
    as a ``count`` attribute; for ``average=True``, the sums have been divided
    by this ``count`` attribute, with bins with no points set to ``NaN``.
    """
    def __init__(self, ih, n_phase, phase, fold_time=None, average=False,
                 samples_per_frame=1, dtype=None):
        self.ih = ih
        self.n_phase = n_phase
        total_time = ih.stop_time - ih.start_time
        if fold_time is None:
            fold_time = total_time
        self.fold_time = fold_time
        self.phase = phase
        self.average = average

        # Note that there may be some time at the end that is never used.
        # Might want to include it if, e.g., it is more than half used.
        nsample = int(np.floor(total_time / self.fold_time //
                               samples_per_frame) * samples_per_frame)
        shape = (nsample, n_phase) + ih.shape[1:]
        # This probably should be moved to a better base class; unfortuantely,
        # we cannot use TaskBase since it does not allow non-integer sample
        # rate ratios.
        frequency = getattr(ih, 'frequency', None)
        sideband = getattr(ih, 'sideband', None)
        polarization = getattr(ih, 'polarization', None)
        if dtype is None:
            dtype = ih.dtype

        super().__init__(shape=shape, start_time=ih.start_time,
                         sample_rate=1./fold_time,
                         samples_per_frame=samples_per_frame,
                         frequency=frequency, sideband=sideband,
                         polarization=polarization, dtype=dtype)

    def _read_frame(self, frame_index):
        # Determine which raw samples to read, and read them.
        frame_rate = self.sample_rate / self.samples_per_frame
        raw_stop = self.ih.seek((frame_index + 1) / frame_rate)
        raw_start = self.ih.seek(frame_index / frame_rate)
        raw_time = self.ih.time
        n_raw = raw_stop - raw_start
        print(frame_index, raw_start, raw_stop, raw_time)
        raw = self.ih.read(n_raw)
        # Set up output arrays.
        result = np.zeros((self.samples_per_frame,) + self.shape[1:],
                          dtype=self.dtype)
        count = np.zeros(result.shape[:2] + (1,) * (result.ndim - 2),
                         dtype=np.int)
        # Get sample and phase indices.
        time_offset = np.arange(n_raw) / self.ih.sample_rate
        sample_index = (time_offset /
                        self.fold_time).to_value(u.one).astype(int)
        # TODO, give a phase reference parameter, right now it is disabled
        phases = self.phase(raw_time + time_offset)
        phase_index = ((phases.to_value(u.one) * self.n_phase)
                       % self.n_phase).astype(int)
        # Do the actual folding; note np.add.at is not very efficient.
        np.add.at(result, (sample_index, phase_index), raw)
        # Consturct the fold counts
        np.add.at(count, (sample_index, phase_index), 1)

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
