# Licensed under the GPLv3 - see LICENSE
"""Tasks for integration over time and pulse phase."""

import numpy as np
import astropy.units as u

from .base import Base


__all__ = ['Integrate', 'Fold']


class _IntegratorCallBack:
    def __init__(self, shape, setitem):
        self.shape = shape
        self.setitem = setitem

    def __setitem__(self, item, value):
        return self.setitem(item, value)


class IntegrateBase(Base):
    """Base class for integrations over fixed sample times."""

    def __init__(self, ih, sample_time=None, average=True,
                 samples_per_frame=1, sample_shape=None, dtype=None):
        self.ih = ih
        self.average = average

        total_time = ih.stop_time - ih.start_time
        if sample_time is None:
            sample_time = total_time

        nsample = int((total_time / sample_time).to_value(1) //
                      samples_per_frame) * samples_per_frame
        if sample_shape is None:
            sample_shape = ih.sample_shape
        shape = (nsample,) + sample_shape

        if dtype is None:
            if average:
                dtype = ih.dtype
            else:
                dtype = np.dtype([('data', ih.dtype), ('count', int)])

        frequency = getattr(ih, 'frequency', None)
        sideband = getattr(ih, 'sideband', None)
        polarization = getattr(ih, 'polarization', None)

        super().__init__(start_time=ih.start_time,
                         shape=shape, sample_rate=1./sample_time,
                         samples_per_frame=samples_per_frame,
                         frequency=frequency, sideband=sideband,
                         polarization=polarization, dtype=dtype)
        self._raw_samples_per_frame = samples_per_frame * sample_time


class Integrate(IntegrateBase):
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
        super().__init__(ih, sample_time=sample_time, average=average, dtype=dtype)

    def _read_frame(self, frame_index):
        # Determine which raw samples to read, and read them.
        # Note: self._raw_samples_per_frame can have units of time.
        raw_stop = self.ih.seek((frame_index + 1) * self._raw_samples_per_frame)
        raw_start = self.ih.seek(frame_index * self._raw_samples_per_frame)
        n_raw = raw_stop - raw_start
        out = np.zeros((self.samples_per_frame,) + self.sample_shape,
                       dtype=self.dtype)
        if self.average:
            self._result = out
            self._count = np.zeros(out.shape[:1] + (1,) * (out.ndim - 1),
                                   dtype=int)
        else:
            self._result = out['data']
            self._count = out['count']
        fake_out = _IntegratorCallBack((n_raw,) + self.ih.sample_shape,
                                       self._callback)
        fake_out = self.ih.read(out=fake_out)
        if self.average:
            out /= self._count
        return out

    def _callback(self, item, data):
        assert type(item) is slice
        self._result[:] += data.sum(0, keepdims=True)
        self._count[:] += len(data)


class Fold(IntegrateBase):
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
        self.n_phase = n_phase
        self.phase = phase

        super().__init__(ih, sample_time=sample_time, average=average,
                         sample_shape=(n_phase,) + ih.sample_shape,
                         samples_per_frame=samples_per_frame, dtype=dtype)

    def _read_frame(self, frame_index):
        # Determine which raw samples to read, and read them.
        # Note: self._raw_samples_per_frame can have units of time.
        raw_stop = self.ih.seek((frame_index + 1) * self._raw_samples_per_frame)
        raw_start = self.ih.seek(frame_index * self._raw_samples_per_frame)
        self._raw_time = self.ih.time
        n_raw = raw_stop - raw_start
        # Set up output arrays.
        out = np.zeros((self.samples_per_frame,) + self.sample_shape,
                       dtype=self.dtype)
        if self.average:
            self._result = out
            self._count = np.zeros(out.shape[:2] + (1,) * (out.ndim - 2),
                                   dtype=int)
        else:
            self._result = out['data']
            self._count = out['count']

        fake_out = _IntegratorCallBack((n_raw,) + self.ih.sample_shape,
                                       self._callback)
        fake_out = self.ih.read(out=fake_out)
        if self.average:
            out /= self._count

        return out

    def _callback(self, item, raw):
        assert type(item) is slice
        # Get sample and phase indices.
        time_offset = np.arange(item.start, item.stop) / self.ih.sample_rate
        sample_index = (time_offset *
                        self.sample_rate).to_value(u.one).astype(int)
        # TODO: allow having a phase reference.
        phases = self.phase(self._raw_time + time_offset)
        phase_index = ((phases.to_value(u.one) * self.n_phase)
                       % self.n_phase).astype(int)
        # Do the actual folding, adding the data to the sums and counts.
        # TODO: np.add.at is not very efficient; replace?
        np.add.at(self._result, (sample_index, phase_index), raw)
        np.add.at(self._count, (sample_index, phase_index), 1)
