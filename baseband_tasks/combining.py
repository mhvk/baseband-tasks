# Licensed under the GPLv3 - see LICENSE
import numpy as np
from astropy import units as u

from .base import TaskBase, Task, META_ATTRIBUTES


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
        Useful mostly to change a possibly very large number.
    **kwargs
        Additional arguments to be passed on to the base class.
    """

    # Implementation detail: kwargs allows easier combining with Task below.

    def __init__(self, ihs, *, atol=None, samples_per_frame=None, **kwargs):
        try:
            ih0 = ihs[0]
        except (TypeError, IndexError) as exc:
            exc.args += ("Need an iterable containing at least one stream.",)
            raise

        # Check consistency of the streams, and determine common time.
        start_time = ih0.start_time
        stop_time = ih0.stop_time
        for ih in ihs[1:]:
            assert ih.sample_rate == ih0.sample_rate
            assert ih.dtype == ih0.dtype
            start_time = max(start_time, ih.start_time)
            stop_time = min(stop_time, ih.stop_time)

        # Extract relevant parts of each file, checking they are aligned well.
        # TODO: use future Resample class to lift alignment restriction?
        ihs = [ih[ih.seek(start_time):ih.seek(stop_time)] for ih in ihs]
        max_offset = max(abs(ih.start_time - start_time) for ih in ihs)
        if atol is None:
            atol = min(1. * u.ns, 0.01 / ih0.sample_rate)
        if max_offset > atol:
            raise ValueError(f"streams only aligned to {max_offset}, "
                             f"not within {atol}.")

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

        self.ihs = ihs
        shape = ihs[0].shape[:1] + a.shape[1:]
        for attr in META_ATTRIBUTES:
            if attr not in kwargs:
                kwargs[attr] = self._combine_attr(attr)

        super().__init__(ihs[0], start_time=start_time, shape=shape,
                         samples_per_frame=samples_per_frame, **kwargs)

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

        values = [np.broadcast_to(value, (1,) + ih.sample_shape, subok=True)
                  for value, ih in zip(values, self.ihs)]

        try:
            result = self.task(values)
        except Exception as exc:
            exc.args += ("the {} attribute of the streams cannot be combined "
                         "as required".format(attr),)
            raise

        return result[0]

    def close(self):
        super().close()
        for ih in self.ihs[1:]:
            ih.close()

    def _seek_frame(self, frame_index):
        for ih in self.ihs:
            ih.seek(frame_index * self._ih_samples_per_frame)
        return ih.tell()

    def _read_frame(self, frame_index):
        """Read and combine data from the underlying filehandles."""
        start = self._seek_frame(frame_index)
        stop = min(start + self._ih_samples_per_frame, self._ih_stop)
        data = [ih.read(stop-start) for ih in self.ihs]
        return self.task(data)

    def _repr_item(self, key, default, value=None):
        if key == 'ihs':
            return 'ihs'
        else:
            return super()._repr_item(key, default=default, value=value)

    def __repr__(self):
        extra = f"\nihs: {len(self.ihs)} streams of which the first is:\n    "
        return super().__repr__().replace('\nih: ', extra)


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
        Useful mostly to change a possibly very large number.

    See Also
    --------
    Concatenate : to concatenate streams along an existing axis
    Stack : to stack streams together along a new axis
    """
    # Override init just to change name of ih to ihs.
    def __init__(self, ihs, task, method=None, *,
                 atol=None, samples_per_frame=None):
        super().__init__(ihs, task, method=method,
                         atol=atol, samples_per_frame=samples_per_frame)


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
        Useful mostly to change a possibly very large number.

    See Also
    --------
    Stack : to stack streams along a new axis
    CombineStreams : to combine streams with a user-supplied function
    """

    def __init__(self, ihs, axis=1, *, atol=None, samples_per_frame=None):
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
        Useful mostly to change a possibly very large number.

    See Also
    --------
    Concatenate : to concatenate streams along an existing axis
    CombineStreams : to combine streams with a user-supplied function
    """

    def __init__(self, ihs, axis=1, *, atol=None, samples_per_frame=None):
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
