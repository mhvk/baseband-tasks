# Licensed under the GPLv3 - see LICENSE
"""Tasks for integration over time and pulse phase."""

import operator

import numpy as np
import astropy.units as u
from astropy.utils import ShapedLikeNDArray, lazyproperty

from .base import BaseTaskBase


__all__ = ['Integrate', 'Fold', 'Stack']


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


def is_index(n):
    """Helper that checks whether n is suitable for indexing."""
    try:
        operator.index(n)
    except TypeError:
        return False
    else:
        return True


class Integrate(BaseTaskBase):
    """Integrate a stream stepwise.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream, with time as the first axis.
    step : int or `~astropy.units.Quantity`, optional
        Interval over which to integrate.  For time invervals, should have
        units of time.  For phase intervals, units should be consistent with
        what the ``phase`` callable returns.  If no ``phase`` callable is
        passed in, then if step is integer, it is taken to be a number of
        samples in the underlying stream that should be integrated over,
        and if step is omitted, integration is over all samples.
    phase : callable
        Should return full pulse phase (i.e., including cycle count) for given
        input times (passed in as '~astropy.time.Time').  The output should be
        compatible with ``step``, i.e., generally an `~astropy.units.Quantity`
        with angular units.
    average : bool, optional
        Whether to calculate sums (with a ``count`` attribute) or to average
        values in each bin.
    samples_per_frame : int, optional
        Number of samples to process in one go.  This can be used to optimize
        the process.  With many samples per bin, the default of 1 should be OK.
    dtype : `~numpy.dtype`, optional
        Output dtype.  Generally, the default of the dtype of the underlying
        stream is good enough, but can be used to increase precision.  Note
        that if ``average=True``, it is the user's responsibilty to pass in
        a structured dtype.

    Notes
    -----

    Since time or phase bins are typically not an integer multiple of the
    underlying bin spacing, the integrated samples will generally not contain
    the same number of samples.  The actual number of samples is counted, and
    for ``average=True``, the sums have been divided by these counts, with bins
    with no points set to ``NaN``.  For ``average=False``, the arrays returned
    by ``read`` are structured arrays with ``data`` and ``count`` fields.

    .. warning: The format for ``average=False`` may change in the future.

    """
    def __init__(self, ih, step=None, phase=None, *,
                 average=True, samples_per_frame=1, dtype=None):
        if phase is None and (step is None or is_index(step)):
            # For integer step, avoid doing any calculations of Time,
            # since if a Stack was run before, start_time and stop_time
            # are no longer reliable.
            if step is None:
                step = ih.shape[0]

            sample_rate = ih.sample_rate / step
            shape = (ih.shape[0] // step,) + ih.sample_shape
            # Initialize values for _get_offsets.
            self._mean_offset_size = 1. / step
        else:
            start = ih.start_time
            stop = ih.stop_time
            if phase is not None:
                start = phase(start)
                stop = phase(stop)

            sample_rate = 1. / step
            n_sample = ((stop - start) / step).to_value(u.one)
            shape = (int(n_sample),) + ih.sample_shape
            # Initialize values for _get_offsets.
            self._mean_offset_size = n_sample / ih.shape[0]
            self._start = start
            self._last_phase = start
            self._last_offset = 0

        if dtype is None:
            if average:
                dtype = ih.dtype
            else:
                dtype = np.dtype([('data', ih.dtype), ('count', int)])

        super().__init__(ih, shape=shape, sample_rate=sample_rate,
                         samples_per_frame=samples_per_frame, dtype=dtype)
        self.average = average
        self._phase = phase

    @lazyproperty
    def stop_time(self):
        """Time at the end of the output, just after the last sample.

        See also `start_time` and `time`.
        """
        raw_stop = self._get_offsets(self.shape[0])
        raw_offset = self.ih.tell()
        try:
            self.ih.seek(raw_stop)
            return self.ih.time
        finally:
            self.ih.seek(raw_offset)

    def _get_offsets(self, samples):
        """Get offsets in the underlying stream nearest to samples."""
        if self._phase is None:
            return np.round((samples / self._mean_offset_size)).astype(int)

        # We assume that the frequency changes are sufficiently small that
        # the bin size does not change significantly.
        offset_phase_size = u.Unit(self._mean_offset_size / self.sample_rate)
        phase = self._start + np.atleast_1d(samples) / self.sample_rate
        offsets = self._last_offset
        check = np.round((phase - self._last_phase)
                         .to_value(offset_phase_size)).astype(int)
        mask = check != 0
        while np.any(mask):
            offsets += check
            # Use mask to avoid calculating more phases than necessary.
            ih_time = self.ih.start_time + offsets[mask] / self.ih.sample_rate
            ih_phase = self._phase(ih_time)
            check[mask] = np.round((phase[mask] - ih_phase)
                                   .to_value(offset_phase_size))
            mask = check != 0

        self._last_offset = offsets[-1]
        self._last_phase = ih_phase[-1]
        return offsets[0] if getattr(samples, 'shape', ()) is () else offsets

    def _read_frame(self, frame_index):
        """Determine which samples to read, and integrate over them.

        Uses the a ``_get_offsets`` method to determine where in the underlying
        stream the samples should be gotten from.

        Integration is done by setting up a fake output array whose setter
        calls back to the ``_integrate`` method that does the actual summing.
        """
        # Get offsets in the underlying stream for the current samples (and
        # the next one to get the upper edge). For integration over time
        # intervals, these offsets are not necessarily evenly spaced.
        samples = (frame_index * self.samples_per_frame +
                   np.arange(self.samples_per_frame + 1))
        offsets = self._get_offsets(samples)
        self.ih.seek(offsets[0])
        offsets -= offsets[0]
        # Set up fake output with a shape that tells the reader of the
        # underlying stream how many samples should be read (and a remaining
        # part that should pass consistency checks), and which has a callback
        # for the actual setting of output in the reader.
        integrating_out = _FakeOutput((offsets[-1],) + self.ih.sample_shape,
                                      setitem=self._integrate)
        # Set up real output and store information used in self._integrate
        out = np.zeros((self.samples_per_frame,) + self.sample_shape,
                       dtype=self.dtype)
        if self.average:
            self._result = out
            self._count = np.zeros(out.shape[:2] + (1,) * (out.ndim - 2),
                                   dtype=int)
        else:
            self._result = out['data']
            self._count = out['count']
        self._offsets = offsets

        # Do the actual reading.
        self.ih.read(out=integrating_out)
        if self.average:
            out /= self._count

        return out

    def _integrate(self, item, data):
        """Sum data in the correct samples.

        Here, item will be a slice with start and stop being indices in the
        underlying stream relative to the start of the current output frame,
        and data the corresponding slice of underlying stream.
        """
        # Note that this is not entirely trivial even for integrating over an
        # integer number of samples, since underlying data frames do not
        # necessarily contain integer multiples of this number of samples.
        #
        # First find all samples that have any overlap with the slice, i.e.,
        # for which start < offset_right and stop > offset_left.  Here, we use
        # the offsets in the underlying stream for each sample in the current
        # frame, plus the one just above, i.e., f[0]=0, f[1], f[2], ..., f[n].
        # (E.g., bin 0 should be included only when start < f[1]; bin n-1
        # only when stop > f[n-1].)
        start = np.searchsorted(self._offsets[1:], item.start, side='right')
        stop = np.searchsorted(self._offsets[:-1], item.stop, side='left')
        # Calculate corresponding indices in ``data`` by extracting the offsets
        # that have any overlap (we take one more -- guaranteed to exist --
        # so that we can count the number of items more easily), subtracting
        # the start offset, and clipping to the right range.
        indices = self._offsets[start:stop + 1] - item.start
        indices[0] = 0
        indices[-1] = item.stop - item.start
        # Finally, sum within slices constructed from consecutive indices
        # (reduceat always adds the end point itself).
        self._result[start:stop] += np.add.reduceat(data, indices[:-1])
        self._count[start:stop] += (np.diff(indices)
                                    .reshape((-1,) + (1,) * (self.ndim - 1)))


class Fold(Integrate):
    """Fold pulse profiles in fixed time intervals.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream, with time as the first axis.
    n_phase : int
        Number of bins per pulse period.
    phase : callable
        Should return pulse phases (with or without cycle count) for given
        input time(s), passed in as an '~astropy.time.Time' object.  The output
        can be an `~astropy.units.Quantity` with angular units or a regular
        array of float (in which case units of cycles are assumed).
    step : int or `~astropy.units.Quantity`, optional
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

    See Also
    --------
    Stack : to integrate over pulse phase and create pulse stacks

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
    def __init__(self, ih, n_phase, phase, step=None, average=True,
                 samples_per_frame=1, dtype=None):
        super().__init__(ih, step=step, average=average,
                         samples_per_frame=samples_per_frame)
        self.n_phase = n_phase
        self.phase = phase
        self._shape = (self._shape[0], n_phase) + self.sample_shape

    def _read_frame(self, frame_index):
        # Before calling the underlying implementation, get the start time in
        # the underlying frame, to be used to calculate phases in _integrate.
        offset0 = self._get_offsets(frame_index * self.samples_per_frame)
        self.ih.seek(offset0)
        self._raw_time = self.ih.time
        return super()._read_frame(frame_index)

    def _integrate(self, item, raw):
        # Get sample and phase indices.
        raw_items = np.arange(item.start, item.stop)
        if self.samples_per_frame == 1:
            sample_index = 0
        else:
            sample_index = np.searchsorted(self._offsets[1:], raw_items)

        # TODO: allow having a phase reference.
        phases = self.phase(self._raw_time + raw_items / self.ih.sample_rate)
        if type(phases) is not np.ndarray:
            phases = u.Quantity(phases, u.cycle, copy=False).value
        phase_index = ((phases * self.n_phase) % self.n_phase).astype(int)
        # Do the actual folding, adding the data to the sums and counts.
        # TODO: np.add.at is not very efficient; replace?
        np.add.at(self._result, (sample_index, phase_index), raw)
        np.add.at(self._count, (sample_index, phase_index), 1)


class Stack(BaseTaskBase):
    """Create a stream of pulse profiles.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream, with time as the first axis.
    n_phase : int
        Number of bins per pulse period.
    phase : callable
        Should return pulse phases for given input time(s), passed in as an
        '~astropy.time.Time' object.  The output should be an array of float,
        and has to include the cycle count.
    average : bool, optional
        Whether to calculate sums (with a ``count`` attribute) or to average
        values in each phase bin.
    samples_per_frame : int, optional
        Number of sample times to process in one go.  This can be used to
        optimize the process, though with many samples per pulse perdiod the
        default of 1 should be fine.
    dtype : `~numpy.dtype`, optional
        Output dtype.  Generally, the default of the dtype of the underlying
        stream is good enough, but can be used to increase precision.  Note
        that if ``average=True``, it is the user's responsibilty to pass in
        a structured dtype.

    See Also
    --------
    Fold : to calculate pulse profiles integrated over a given amount of time.

    Notes
    -----
    Since phase bins are typically not an integer multiple of the underlying
    bin spacing, the integrated samples will generally not contain the same
    number of samples.  The actual number of samples is counted, and for
    ``average=True``, the sums have been divided by these counts, with bins
    with no points set to ``NaN``.  For ``average=False``, the arrays returned
    by ``read`` are structured arrays with ``data`` and ``count`` fields.

    .. warning: The format for ``average=False`` may change in the future.

    """
    def __init__(self, ih, n_phase, phase, average=True,
                 samples_per_frame=1, dtype=None):
        # Set up the integration in phase bins.
        phased = Integrate(ih, u.cycle/n_phase, phase, average=average,
                           samples_per_frame=samples_per_frame*n_phase,
                           dtype=dtype)
        # And ensure we reshape it to cycles.
        shape = (phased.shape[0] // n_phase, n_phase) + phased.shape[1:]
        super().__init__(phased, shape=shape,
                         sample_rate=phased.sample_rate / n_phase,
                         samples_per_frame=samples_per_frame,
                         dtype=dtype)
        self._n_phase = n_phase

    def _read_frame(self, frame_index):
        # Read frame in phased directly, bypassing its ``read`` method.
        out = self.ih._read_frame(frame_index)
        return out.reshape((self.samples_per_frame,) + self.sample_shape)

    @property
    def stop_time(self):
        return self.ih.stop_time
