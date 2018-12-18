# Licensed under the GPLv3 - see LICENSE
"""Tasks for integration over time and pulse phase."""

import numpy as np
import astropy.units as u

from .base import Base


__all__ = ['Integrate', 'Fold']


class Integrate(Base):
    """Integrate a stream over specific time steps.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream, with time as the first axis.
    sample_time : `~astropy.units.Quantity`, optional
        With units of time.  By default, will integrate the whole input stream.
    average : bool, optional
        Whether to calculate sums (with a ``count`` attribute) or to average
        values.
    dtype : `~numpy.dtype`, optional
        Output dtype.  Generally, the default of the dtype of the underlying
        stream is good enough, but can be used to increase precision.  Note
        that if ``average=True``, it is the user's responsibilty to pass in
        a structured dtype.

    Notes
    -----
    Since the sample time is not necessarily an integer multiple of the pulse
    period, the returned profiles will generally not contain the same number
    of samples in each phase bin.  The actual number of samples is counted,
    and for ``average=True``, the sums have been divided by these counts, with
    bins with no points set to ``NaN``.  For ``average=False``, the arrays
    returned by ``read`` are structured arrays with ``data`` and ``count``
    fields.

    .. warning: The format for ``average=False`` may change in the future.
    """

    def __init__(self, ih, sample_time=None, average=True, dtype=None):
        self.ih = ih
        self.average = average
        total_time = ih.stop_time - ih.start_time
        if sample_time is None:
            sample_time = total_time
        nsample = int((total_time / sample_time).to_value(1))

        shape = (nsample,) + ih.shape[1:]
        frequency = getattr(ih, 'frequency', None)
        sideband = getattr(ih, 'sideband', None)
        polarization = getattr(ih, 'polarization', None)
        if dtype is None:
            if self.average:
                dtype = ih.dtype
            else:
                dtype = np.dtype([('data', ih.dtype), ('count', int)])

        super().__init__(start_time=ih.start_time,
                         shape=shape, sample_rate=1./sample_time,
                         samples_per_frame=1, frequency=frequency,
                         sideband=sideband, polarization=polarization,
                         dtype=dtype)

    def _read_frame(self, frame_index):
        raw_stop = self.ih.seek((frame_index + 1) / self.sample_rate)
        raw_start = self.ih.seek(frame_index / self.sample_rate)
        data = self.ih.read(raw_stop - raw_start)
        out = np.zeros((self.samples_per_frame,) + self.shape[1:],
                       dtype=self.dtype)
        if self.average:
            result = out
            count = np.zeros(result.shape[:1] + (1,) * (result.ndim - 1),
                             dtype=int)
        else:
            result = out['data']
            count = out['count']
        result[:] = data.sum(0, keepdims=True)
        count[:] = len(data)
        if self.average:
            out /= count
        return out


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
    sample_time : `~astropy.units.Quantity`, optional
        Time interval over which to fold, i.e., the sample time of the output.
        If not given, the whole file will be folded into a single profile.
    average : bool, optional
        Whether the output pulse profile should be the average of all entries
        that contributed to it, or rather the sum, in a structured array that
        holds both ``'data'`` and ``'count'`` items.
    samples_per_frame : int, optional
        Number of sample times to process in one go.  This can be used to
        optimize the process, though in general the default of 1 should work.
    dtype : `~numpy.dtype`, optional
        Output dtype.  Generally, the default of the dtype of the underlying
        stream is good enough, but can be used to increase precision.  Note
        that if ``average=True``, it is the user's responsibilty to pass in
        a structured dtype.

    Notes
    -----
    Since the sample time is not necessarily an integer multiple of the pulse
    period, the returned profiles will generally not contain the same number
    of samples in each phase bin.  The actual number of samples is counted,
    and for ``average=True``, the sums have been divided by these counts, with
    bins with no points set to ``NaN``.  For ``average=False``, the arrays
    returned by ``read`` are structured arrays with ``data`` and ``count``
    fields.

    .. warning: The format for ``average=False`` may change in the future.
    """
    def __init__(self, ih, n_phase, phase, sample_time=None, average=True,
                 samples_per_frame=1, dtype=None):
        self.ih = ih
        self.n_phase = n_phase
        total_time = ih.stop_time - ih.start_time
        if sample_time is None:
            sample_time = total_time
        self.sample_time = sample_time
        self.phase = phase
        self.average = average

        # Note that there may be some time at the end that is never used.
        # Might want to include it if, e.g., it is more than half used.
        nsample = int((total_time / self.sample_time).to_value(u.one) //
                      samples_per_frame) * samples_per_frame
        shape = (nsample, n_phase) + ih.shape[1:]
        # This probably should be moved to a better base class; unfortunately,
        # we cannot use TaskBase since it does not allow non-integer sample
        # rate ratios.
        frequency = getattr(ih, 'frequency', None)
        sideband = getattr(ih, 'sideband', None)
        polarization = getattr(ih, 'polarization', None)
        if dtype is None:
            if self.average:
                dtype = ih.dtype
            else:
                dtype = np.dtype([('data', ih.dtype), ('count', int)])

        super().__init__(shape=shape, start_time=ih.start_time,
                         sample_rate=1./sample_time,
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
        raw = self.ih.read(n_raw)
        # Set up output arrays.
        out = np.zeros((self.samples_per_frame,) + self.shape[1:],
                       dtype=self.dtype)
        if self.average:
            result = out
            count = np.zeros(result.shape[:2] + (1,) * (result.ndim - 2),
                             dtype=int)
        else:
            result = out['data']
            count = out['count']

        # Get sample and phase indices.
        time_offset = np.arange(n_raw) / self.ih.sample_rate
        sample_index = (time_offset *
                        self.sample_rate).to_value(u.one).astype(int)
        # TODO: allow having a phase reference.
        phases = self.phase(raw_time + time_offset)
        phase_index = ((phases.to_value(u.one) * self.n_phase)
                       % self.n_phase).astype(int)
        # Do the actual folding, adding the data to the sums and counts.
        # TODO: np.add.at is not very efficient; replace?
        np.add.at(result, (sample_index, phase_index), raw)
        np.add.at(count, (sample_index, phase_index), 1)

        if self.average:
            out /= count

        return out
