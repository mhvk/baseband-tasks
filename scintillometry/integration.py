# Licensed under the GPLv3 - see LICENSE
"""Tasks for integration over time and pulse phase."""

import operator

import numpy as np
import astropy.units as u
from astropy.utils import ShapedLikeNDArray

from .base import Base


__all__ = ['Integrate', 'Fold']


class _FakeOutput(ShapedLikeNDArray):
    """Pretend output class that diverts setting.

    It subclasses `~astropy.utils.ShapedLikeNDArray` to mimic all the
    shape properties of `~numpy.ndarray` given a (fake) shape.
    """
    def __init__(self, shape, setitem):
            self._shape = shape
            self._setitem = setitem

    def __setitem__(self, item, value):
        # Call back on out[item] = value.
        return self._setitem(item, value)

    # The two required parts for ShapedLikeNDArray.
    @property
    def shape(self):
        return self._shape

    def _apply(self, *args, **kwargs):
        raise NotImplementedError("No _apply possible for _FakeOutput")


class IntegrateBase(Base):
    """Base class for integrations over fixed sample times."""

    def __init__(self, ih, n_sample=None, average=True,
                 samples_per_frame=1, sample_shape=None, dtype=None):
        self.ih = ih
        self.average = average

        total_time = ih.stop_time - ih.start_time
        if n_sample is None:
            n_sample = self.ih.shape[0]

        try:
            n_sample = operator.index(n_sample)
        except TypeError:
            sample_rate = 1. / n_sample
            nframes = int((total_time / n_sample).to_value(u.one))
        else:
            sample_rate = self.ih.sample_rate / n_sample
            nframes = self.ih.shape[0] // n_sample

        if sample_shape is None:
            sample_shape = ih.sample_shape

        nframes = (nframes // samples_per_frame) * samples_per_frame
        assert nframes > 0, "time per frame larger than total time in stream"
        shape = (nframes,) + sample_shape

        if dtype is None:
            if average:
                dtype = ih.dtype
            else:
                dtype = np.dtype([('data', ih.dtype), ('count', int)])

        frequency = getattr(ih, 'frequency', None)
        sideband = getattr(ih, 'sideband', None)
        polarization = getattr(ih, 'polarization', None)

        super().__init__(start_time=ih.start_time,
                         shape=shape, sample_rate=sample_rate,
                         samples_per_frame=samples_per_frame,
                         frequency=frequency, sideband=sideband,
                         polarization=polarization, dtype=dtype)
        self._n_sample = n_sample
        self._raw_samples_per_frame = samples_per_frame * n_sample

    def _read_frame(self, frame_index):
        """Determine which raw samples to read, and read them using integrating read.

        Uses the ``_n_sample`` attribute to seek in the underlying stream;
        note that this can have units of time.
        """
        base_offset = frame_index * self.samples_per_frame
        raw_offsets = [self.ih.seek((base_offset + i) * self._n_sample)
                       for i in range(self.samples_per_frame + 1)]
        # Use seek to find the stop and start positions; this will round to the
        # nearest offset if necessary.
        return self._integrating_read(*raw_offsets)

    def _integrating_read(self, *raw_offsets):
        """Set up fake output for a read from the underlying stream.

        Subclasses have to provide an ``_integrate method`` which will be
        used to override ``__setitem__`` in the fake output.
        """
        out = np.zeros((self.samples_per_frame,) + self.sample_shape,
                       dtype=self.dtype)
        if self.average:
            self._result = out
            self._count = np.zeros(out.shape[:2] + (1,) * (out.ndim - 2),
                                   dtype=int)
        else:
            self._result = out['data']
            self._count = out['count']

        self.ih.seek(raw_offsets[0])
        self._offsets = [(off - raw_offsets[0]) for off in raw_offsets]
        # Set up fake output with a shape that tells read how many samples to read
        # (and a remaining part that should pass consistency checks), plus a
        # call-back for out[...]=....
        integrating_out = _FakeOutput((self._offsets[-1],) + self.ih.sample_shape,
                                      self._integrate)
        self.ih.read(out=integrating_out)
        if self.average:
            out /= self._count

        return out


class Integrate(IntegrateBase):
    """Integrate a stream over specific time steps.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream, with time as the first axis.
    n_sample : int or `~astropy.units.Quantity`, optional
        Number of input samples or time interval over which to integrate.
        If not given, the whole file will be integrated over.
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

    If ``n_sample`` is a time interval and not an integer multiple of the
    sample time of the underlying file, the returned integrated samples may
    not all contain the same number of samples.  The actual number of samples
    is counted, and for ``average=True``, the sums have been divided by these
    counts, with bins with no points set to ``NaN``.  For ``average=False``,
    the arrays returned by ``read`` are structured arrays with ``data`` and
    ``count`` fields.

    .. warning: The format for ``average=False`` may change in the future.

    """

    def __init__(self, ih, n_sample=None, average=True, samples_per_frame=1,
                 dtype=None):
        super().__init__(ih, n_sample=n_sample, average=average,
                         samples_per_frame=samples_per_frame, dtype=dtype)

    def _integrate(self, item, data):
        assert type(item) is slice
        indices = [0] + [(i - item.start) for i in self._offsets
                         if item.start < i < item.stop]
        self._result[:] += np.add.reduceat(data, indices)
        self._count[:] += (np.diff(indices + [item.stop])
                           .reshape((-1,) + (1,) * (self._count.ndim - 1)))


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
    n_sample : int or `~astropy.units.Quantity`, optional
        Number of input samples or time interval over which to fold.
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
    def __init__(self, ih, n_phase, phase, n_sample=None, average=True,
                 samples_per_frame=1, dtype=None):
        self.n_phase = n_phase
        self.phase = phase

        super().__init__(ih, n_sample=n_sample, average=average,
                         sample_shape=(n_phase,) + ih.sample_shape,
                         samples_per_frame=samples_per_frame, dtype=dtype)

    def _read_frame(self, frame_index):
        # Override base implementation since we need to know the start time
        # in the underlying frame in order to calculate phases in _integrate.
        raw_stop = self.ih.seek((frame_index + 1) * self._raw_samples_per_frame)
        raw_start = self.ih.seek(frame_index * self._raw_samples_per_frame)
        self._raw_time = self.ih.time
        return self._integrating_read(raw_start, raw_stop)

    def _integrate(self, item, raw):
        assert type(item) is slice
        # Get sample and phase indices.
        time_offset = np.arange(item.start, item.stop) / self.ih.sample_rate
        if self.samples_per_frame == 1:
            sample_index = 0
        else:
            # This is not quite correct: should take into account possible
            # time offset between self and underlying stream.
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
