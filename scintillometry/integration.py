# Licensed under the GPLv3 - see LICENSE
"""Tasks for integration over time and pulse phase."""

import operator

import numpy as np
import astropy.units as u
from astropy.utils import ShapedLikeNDArray

from .base import BaseTaskBase


__all__ = ['Integrate', 'Fold', 'IntegrateByPhase']


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


class Integrate(BaseTaskBase):
    """Integrate a stream over specific time steps.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream, with time as the first axis.
    step : int or `~astropy.units.Quantity`, optional
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

    If ``step`` is a time interval and not an integer multiple of the
    sample time of the underlying file, the returned integrated samples may
    not all contain the same number of samples.  The actual number of samples
    is counted, and for ``average=True``, the sums have been divided by these
    counts, with bins with no points set to ``NaN``.  For ``average=False``,
    the arrays returned by ``read`` are structured arrays with ``data`` and
    ``count`` fields.

    .. warning: The format for ``average=False`` may change in the future.

    """
    def __init__(self, ih, step=None, average=True, samples_per_frame=1,
                 dtype=None):
        self.ih = ih
        self.average = average

        total_time = ih.stop_time - ih.start_time
        if step is None:
            step = self.ih.shape[0]

        try:
            step = operator.index(step)
        except TypeError:
            sample_rate = 1. / step
            nframes = int((total_time / step).to_value(u.one))
        else:
            sample_rate = self.ih.sample_rate / step
            nframes = self.ih.shape[0] // step

        nframes = (nframes // samples_per_frame) * samples_per_frame
        assert nframes > 0, "time per frame larger than total time in stream"
        shape = (nframes,) + ih.sample_shape

        if dtype is None:
            if average:
                dtype = ih.dtype
            else:
                dtype = np.dtype([('data', ih.dtype), ('count', int)])

        super().__init__(ih, shape=shape, sample_rate=sample_rate,
                         samples_per_frame=samples_per_frame, dtype=dtype)
        self._step = step

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

    def _get_offsets(self, samples):
        """Get offsets in the underlying stream nearest to samples."""
        if type(self._step) is int:
            return samples * self._step
        else:
            return np.round((samples * self._step * self.ih.sample_rate)
                            .to_value(u.one)).astype(int)

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
        Should return pulse phases for given input time(s), passed in as an
        '~astropy.time.Time' object.  The output should be an array of float;
        the phase can include the cycle count.
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
        self.n_phase = n_phase
        self.phase = phase
        # First set up as for integration.
        super().__init__(ih, step=step, average=average,
                         samples_per_frame=samples_per_frame, dtype=dtype)
        # But then adjust the shape to take into account that we're folding.
        self._shape = (self._shape[0], n_phase) + ih.sample_shape

    def _read_frame(self, frame_index):
        # Before calling the base implementation, get the start time in the
        # underlying frame, which we need to calculate phases in _integrate.
        self.ih.seek(frame_index * self._step * self.samples_per_frame)
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
        phase_index = ((phases.to_value(u.one) * self.n_phase)
                       % self.n_phase).astype(int)
        # Do the actual folding, adding the data to the sums and counts.
        # TODO: np.add.at is not very efficient; replace?
        np.add.at(self._result, (sample_index, phase_index), raw)
        np.add.at(self._count, (sample_index, phase_index), 1)


class IntegrateByPhase(Integrate):
    """Integrate a stream over fixed phase intervals.

    This class is not that useful directly, but is used to help produce pulse
    stacks in `~scintillometry.integration.Stack`.

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
        values in each bin.
    samples_per_frame : int, optional
        Number of sample times to process in one go.  This can be used to
        optimize the process, though with many samples per bin the default
        of 1 should work.
    dtype : `~numpy.dtype`, optional
        Output dtype.  Generally, the default of the dtype of the underlying
        stream is good enough, but can be used to increase precision.  Note
        that if ``average=True``, it is the user's responsibilty to pass in
        a structured dtype.

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
        self.phase = phase
        start_phase = phase(ih.start_time)
        stop_phase = phase(ih.stop_time)
        step_phase = 1. / n_phase

        mean_time_step = ((ih.stop_time - ih.start_time) /
                          ((stop_phase - start_phase) / step_phase))
        super().__init__(ih, step=mean_time_step,
                         average=average, samples_per_frame=samples_per_frame,
                         dtype=dtype)
        self._start_phase = start_phase
        self._step_phase = step_phase
        self._ih_mean_step_phase = (mean_time_step / step_phase *
                                    self.ih.sample_rate)  # bin/cycle
        self._last_offset = 0
        self._last_phase = start_phase
        self._sample_rate = n_phase / u.cycle
        self._stop_time = self.start_time + self.shape[0] * mean_time_step

    @property
    def stop_time(self):
        return self._stop_time

    def _get_offsets(self, samples):
        """Get offsets in the underlying stream nearest to samples."""
        phase = self._start_phase + samples * self._step_phase
        offsets = self._last_offset
        check = ((phase - self._last_phase) *
                 self._ih_mean_step_phase).to_value(u.one).round().astype(int)
        mask = check != 0
        while np.any(mask):
            offsets += check
            # Use mask to avoid calculating more phases than necessary.
            ih_time = self.ih.start_time + offsets[mask] / self.ih.sample_rate
            ih_phase = self.phase(ih_time)
            check[mask] = ((phase[mask] - ih_phase) *
                           self._ih_mean_step_phase).to_value(u.one).round()
            mask = check != 0

        self._last_offset = offsets[-1] if phase.shape else offsets
        self._last_phase = ih_phase[-1] if phase.shape else ih_phase
        return offsets


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
        phased = IntegrateByPhase(ih, n_phase, phase, average=average,
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
