# Licensed under the GPLv3 - see LICENSE
import numpy as np
from astropy import units as u

from .base import TaskBase, Task


__all__ = ['CombineStreamsBase', 'CombineStreams', 'Concatenate', 'Stack']


class CombineStreamsBase(TaskBase):
    """Base class for combining streams.

    Similar to `~baseband_tasks.base.TaskBase`, where a subclass can define
    a ``task`` method to operate on data, but specifically for methods that
    combine data from multiple streams that share a time axis.  This base
    class ensures the operation is possible and that the ``frequency``,
    ``sideband``, and ``polarization`` attributes are adjusted similarly.

    Parameters
    ----------
    ihs : tuple of task or `baseband` stream readers
        Input data streams.
    atol : `~astropy.units.Quantity`
        Tolerance in units of time within which streams should be considered
        aligned.  By default, the lesser of 1 ns or 0.01 sample.
    samples_per_frame : int, optional
        Number of samples which should be fed to the function in one go.
        If not given, by default the number from the first underlying file.
        Useful mostly in case the stream readers have time offsets, since
        the output stream will be shortened by an integer number of frames.
    """

    def __init__(self, ihs, atol=None, samples_per_frame=None):
        try:
            ih0 = ihs[0]
        except (TypeError, IndexError) as exc:
            exc.args += ("Need an iterable containing at least one stream.",)
            raise

        # Check consistency of the streams, and determine common time.
        self.ihs = ihs
        start_time = ih0.start_time
        stop_time = ih0.stop_time
        for ih in ihs[1:]:
            assert ih.sample_rate == ih0.sample_rate
            assert ih.dtype == ih0.dtype
            start_time = max(start_time, ih.start_time)
            stop_time = min(stop_time, ih.stop_time)

        n_sample = ih0.seek(stop_time) - ih0.seek(start_time)

        # Calculate offsets for each file and check they are aligned well.
        # TODO: use future Resample class to lift this restriction?
        self._start_offsets = []
        if atol is None:
            atol = min(1. * u.ns, 0.01 / ih0.sample_rate)
        for ih in ihs:
            offset = ih.seek(start_time)
            if abs(ih.time - start_time) > atol:
                raise ValueError("streams only aligned to {}, not within {}"
                                 .format((ih.time-start_time).to(u.ns), atol))
            self._start_offsets.append(offset)

        # Check that the stream samples can be combined.
        fakes = [np.empty((7,) + ih.sample_shape, ih.dtype) for ih in ihs]
        try:
            a = self.task(fakes)
        except Exception as exc:
            exc.args += ("streams with sample shapes {} cannot be combined "
                         "as required".format([f.shape[1:] for f in fakes]),)
            raise
        if a.shape[0] != 7:
            raise ValueError("combination affected the sample axis (0).")

        # Calculate the combined shape and attributes.
        shape = (n_sample,) + a.shape[1:]
        attrs = {attr: self._combine_attr(attr)
                 for attr in ('frequency', 'sideband', 'polarization')}
        super().__init__(ih0, shape=shape, samples_per_frame=samples_per_frame,
                         **attrs)
        # Could just pass in start_time above if we based ourselves on
        # BaseTaskBase, which is more logical, but then cannot mixin Task
        # below.  TODO: remove TaskBase intermediary?
        self._start_time = start_time

    def _combine_attr(self, attr):
        """Combine the given attribute from all streams.

        Parameters
        ----------
        attr : str
            Attribute to look up and combine

        Returns
        -------
        combined : None or combined array
             `None` if all attributes were `None`.
        """
        values = [getattr(ih, attr, None) for ih in self.ihs]

        if all(value is None for value in values):
            return None

        # Don't count on our task to pass on subclasses
        # (needs astropy >=4.0 and numpy >=1.17).
        value0 = values[0]
        unit = getattr(value0, 'unit', None)
        if unit is not None:
            values = [value.to_value(unit) for value in values]

        values = [np.broadcast_to(value, (1,) + ih.sample_shape)
                  for value, ih in zip(values, self.ihs)]

        try:
            result = self.task(values)
        except Exception as exc:
            exc.args += ("the {} attribute of the streams cannot be combined "
                         "as required".format(attr),)
            raise

        if unit is None:
            return result[0]
        else:
            return value0.__class__(result[0], unit, copy=False)

    def close(self):
        super().close()
        for ih in self.ihs[1:]:
            ih.close()

    def _read_frame(self, frame_index):
        """Read and combine data from the underlying filehandles."""
        data = []
        for ih, start_offset in zip(self.ihs, self._start_offsets):
            ih.seek(start_offset + frame_index * self._samples_per_frame)
            data.append(ih.read(self._samples_per_frame))
        return self.task(data)


class CombineStreams(Task, CombineStreamsBase):
    """Combining streams using a callable.

    Parameters
    ----------
    ihs : tuple of task or `baseband` stream readers
        Input data streams.
    task : callable
        The function or method-like callable. The task must work with
        any number of data samples and combine the samples only.
        It will also be applied to the ``frequency``, ``sideband``, and
        ``polarization`` attributes of the underlying stream (if present).
    method : bool, optional
        Whether ``task`` is a method (two arguments) or a function
        (one argument).  Default: inferred by inspection.
    atol : `~astropy.units.Quantity`
        Tolerance in units of time within which streams should be considered
        aligned.  By default, the lesser of 1 ns or 0.01 sample.
    samples_per_frame : int, optional
        Number of samples which should be fed to the function in one go.
        If not given, by default the number from the first underlying file.
        Useful mostly in case the stream readers have time offsets, since
        the output stream will be shortened by an integer number of frames.

    See Also
    --------
    Concatenate : to concatenate streams along an existing axis
    Stack : to stack streams together along a new axis
    """
    # Override __init__ only to get rid of kwargs of Task, since these cannot
    # be passed on to ChangeSampleShapeBase anyway.

    def __init__(self, ihs, task, method=None, atol=None,
                 samples_per_frame=None):
        super().__init__(ihs, task, method=method, atol=atol,
                         samples_per_frame=samples_per_frame)


class Concatenate(CombineStreamsBase):
    """Concatenate streams along an existing axis.

    Parameters
    ----------
    ihs : tuple of task or `baseband` stream readers
        Input data streams.
    axis : int
        Axis along which to combine the samples. Should be a sample
        axis and thus cannot be 0.
    atol : `~astropy.units.Quantity`
        Tolerance in units of time within which streams should be considered
        aligned.  By default, the lesser of 1 ns or 0.01 sample.
    samples_per_frame : int, optional
        Number of samples which should be fed to the function in one go.
        If not given, by default the number from the first underlying file.
        Useful mostly in case the stream readers have time offsets, since
        the output stream will be shortened by an integer number of frames.

    See Also
    --------
    Stack : to stack streams along a new axis
    CombineStreams : to combine streams with a user-supplied function
    """

    def __init__(self, ihs, axis=1, atol=None, samples_per_frame=None):
        self.axis = axis
        super().__init__(ihs, atol=atol, samples_per_frame=samples_per_frame)

    def task(self, data):
        """Concatenate the pieces of data together."""
        # Reuse frame for in-place output if possible.
        if getattr(self._frame, 'shape', [-1])[0] == data[0].shape[0]:
            out = self._frame
        else:
            out = None
        return np.concatenate(data, axis=self.axis, out=out)


class Stack(CombineStreamsBase):
    """Stack streams along a new axis.

    Parameters
    ----------
    ihs : tuple of task or `baseband` stream readers
        Input data streams.
    axis : int
        New axis along which to stack the samples. Should be a sample
        axis and thus cannot be 0.
    atol : `~astropy.units.Quantity`
        Tolerance in units of time within which streams should be considered
        aligned.  By default, the lesser of 1 ns or 0.01 sample.
    samples_per_frame : int, optional
        Number of samples which should be fed to the function in one go.
        If not given, by default the number from the first underlying file.
        Useful mostly in case the stream readers have time offsets, since
        the output stream will be shortened by an integer number of frames.

    See Also
    --------
    Concatenate : to concatenate streams along an existing axis
    CombineStreams : to combine streams with a user-supplied function
    """

    def __init__(self, ihs, axis=1, atol=None, samples_per_frame=None):
        self.axis = axis
        super().__init__(ihs, atol=atol, samples_per_frame=samples_per_frame)

    def task(self, data):
        """Stack the pieces of data."""
        # Reuse frame for in-place output if possible.
        if getattr(self._frame, 'shape', [-1])[0] == data[0].shape[0]:
            out = self._frame
        else:
            out = None
        return np.stack(data, axis=self.axis, out=out)
