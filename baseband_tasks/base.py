# Licensed under the GPLv3 - see LICENSE

import inspect
import operator
import types
import warnings

import numpy as np
from astropy import units as u


__all__ = ['Base', 'BaseTaskBase', 'TaskBase', 'PaddedTaskBase',
           'SetAttribute', 'Task']


Empty = np.empty


def check_broadcast_to(value, sample_shape):
    """Broadcast values to the given shape.

    Like `~numpy.broadcast_to`, but with an addition to any error message.
    """
    try:
        broadcast = np.broadcast_to(value, sample_shape, subok=True)
    except ValueError as exc:
        exc.args += ("value cannot be broadcast to sample shape",)
        raise
    return broadcast


def simplify_shape(value):
    """Replace axes that contain only duplicates with broadcast values.

    For each axis, get first element of the sample, and keep only it if
    all other elements are the same (numpy broadcasting rules will ensure
    any operations using the result will work correctly).
    """
    for axis in range(value.ndim):
        value_0 = value[(slice(None),) * axis + (slice(0, 1),)]
        if value.strides[axis] == 0 or np.all(value == value_0):
            value = value_0
    # Remove leading ones, which are not needed in numpy broadcasting.
    first_not_unity = next((i for (i, s) in enumerate(value.shape)
                            if s > 1), value.ndim)
    # Make copy to ensure we're not a view of the input value (so
    # things cannot get changed under our feet).
    return value.reshape(value.shape[first_not_unity:]).copy()


class Base:
    """Base class of all tasks and generators.

    Following the design of `baseband` stream readers, features properties
    describing the size, shape, data type, sample rate and start/stop times of
    the task's output.  Also defines methods to move a sample pointer across
    the output data in units of either complete samples or time.

    Subclasses should define

      ``_read_frame``: method to read (or generate) a single block of data.

    Parameters
    ----------
    shape : tuple, optional
        Overall shape of the stream, with first entry the total number
        of complete samples, and the remainder the sample shape.
    start_time : `~astropy.time.Time`
        Start time of the stream.
    sample_rate : `~astropy.units.Quantity`
        Rate at which complete samples are produced.
    samples_per_frame : int, optional
        Number of samples dealt with in one go.  The number of complete
        samples (``shape[0]``) should be an integer multiple of this.
    frequency : `~astropy.units.Quantity`, optional
        Frequencies for each channel.  Should be broadcastable to the
        sample shape.  Default: unknown.
    sideband : array, optional
        Whether frequencies are upper (+1) or lower (-1) sideband.
        Should be broadcastable to the sample shape.  Default: unknown.
    polarization : array or (nested) list of char, optional
        Polarization labels.  Should broadcast to the sample shape,
        i.e., the labels are in the correct axis.  For instance,
        ``['X', 'Y']``, or ``[['L'], ['R']]``.  Default: unknown.
    dtype : `~numpy.dtype`, optional
        Dtype of the samples.
    """

    # Initial values for sample and frame pointers, etc.
    offset = 0
    _frame_index = None
    _frame = None
    closed = False

    def __init__(self, shape, start_time, sample_rate, *,
                 samples_per_frame=1,
                 frequency=None, sideband=None, polarization=None,
                 dtype=np.complex64):
        self._shape = shape
        self._start_time = start_time
        self._samples_per_frame = operator.index(samples_per_frame)
        self._sample_rate = sample_rate
        self._dtype = np.dtype(dtype, copy=False)
        if frequency is not None or sideband is not None:
            if frequency is None or sideband is None:
                raise ValueError('frequency and sideband should both '
                                 'be passed in.')
            frequency = self._check_shape(frequency)
            sideband = self._check_shape(np.where(sideband > 0,
                                                  np.int8(1), np.int8(-1)))

        if polarization is not None:
            polarization = self._check_shape(polarization)

        self._frequency = frequency
        self._sideband = sideband
        self._polarization = polarization

    def _repr_item(self, key, default, value=None):
        """Representation of one argument.

        Subclasses can override this, either to return something else than
        the base key=value or to set a different default for specific keys.

        """

        if value is None:
            value = getattr(self, key, None)
            if value is None:
                value = getattr(self, '_' + key, None)
                if value is None:
                    return None

        if default is not inspect._empty:
            try:
                if np.all(value == default):
                    return None
            except Exception:
                pass

        return f"{key}={value}".replace('\n', ',')

    def __str__(self):
        name = self.__class__.__name__
        pars = inspect.signature(self.__class__).parameters
        overrides = [self._repr_item(key, par.default)
                     for key, par in pars.items()]

        overrides = ', '.join([override for override in overrides if override])
        return f"{name}({overrides})"

    def __repr__(self):
        """Representation which lists non-default arguments.

        Finds possible arguments by inspection of the whole class hierarchy
        (as long as kwargs are passed along) and creates a list of all whose
        values on the instance are different from the default. Subclasses
        can override the assumed default and what to return in _repr_item.

        """
        name = self.__class__.__name__
        pars = {}
        for cls in self.__class__.__mro__:
            for key, par in inspect.signature(cls).parameters.items():
                pars.setdefault(key, par)
            if 'kwargs' not in pars:
                break

        overrides = [self._repr_item(key, par.default)
                     for key, par in pars.items()]

        overrides = (',\n '+' '*len(name)).join(
            [override for override in overrides if override])
        return f"{name}({overrides})"

    def _check_shape(self, value):
        """Check that value can be broadcast to the sample shape."""
        broadcast = check_broadcast_to(value, self.sample_shape)
        return simplify_shape(broadcast)

    @property
    def shape(self):
        """Shape of the output."""
        return self._shape

    @property
    def sample_shape(self):
        """Shape of a complete sample."""
        return self.shape[1:]

    @property
    def samples_per_frame(self):
        """Number of samples per frame of data.

        For compatibility with file readers, to help indicate what
        a nominal chunk of data is.
        """
        return self._samples_per_frame

    @property
    def size(self):
        """Number of component samples in the output."""
        prod = 1
        for dim in self.shape:
            prod *= dim
        return prod

    @property
    def ndim(self):
        """Number of dimensions of the output."""
        return len(self.shape)

    @property
    def dtype(self):
        """Data type of the output."""
        return self._dtype

    @property
    def complex_data(self):
        return self._dtype.kind == 'c'

    @property
    def sample_rate(self):
        """Number of complete samples per second."""
        return self._sample_rate

    @property
    def start_time(self):
        """Start time of the output.

        See also `time` and `stop_time`.
        """
        # We don't just return self._start_time so classes like Integrate
        # can get correct results by just overriding _tell_time.
        return self._tell_time(0)

    @property
    def time(self):
        """Time of the sample pointer's current offset in the output.

        See also `start_time` and `stop_time`.
        """
        return self._tell_time(self.offset)

    @property
    def stop_time(self):
        """Time at the end of the output, just after the last sample.

        See also `start_time` and `time`.
        """
        return self._tell_time(self.shape[0])

    @property
    def frequency(self):
        if self._frequency is None:
            raise AttributeError("frequencies not set.")
        return self._frequency

    @property
    def sideband(self):
        if self._sideband is None:
            raise AttributeError("sidebands not set.")
        return self._sideband

    @property
    def polarization(self):
        if self._polarization is None:
            raise AttributeError("polarizations not set.")
        return self._polarization

    def seek(self, offset, whence=0):
        """Change the sample pointer position.

        This works like a normal filehandle seek, but the offset is in samples
        (or a relative or absolute time).

        Parameters
        ----------
        offset : int, `~astropy.units.Quantity`, or `~astropy.time.Time`
            Offset to move to.  Can be an (integer) number of samples,
            an offset in time units, or an absolute time.  For the latter
            two, the pointer will be moved to the nearest integer sample.
        whence : {0, 1, 2, 'start', 'current', or 'end'}, optional
            Like regular seek, the offset is taken to be from the start if
            ``whence=0`` (default), from the current position if 1,
            and from the end if 2.  One can alternativey use 'start',
            'current', or 'end' for 0, 1, or 2, respectively.  Ignored if
            ``offset`` is a time.
        """
        try:
            offset = operator.index(offset)
        except Exception:
            try:
                offset = offset - self.start_time
            except Exception:
                pass
            else:
                whence = 0

            offset = int((offset * self.sample_rate).to(u.one).round())

        if whence == 0 or whence == 'start':
            self.offset = offset
        elif whence == 1 or whence == 'current':
            self.offset += offset
        elif whence == 2 or whence == 'end':
            self.offset = self.shape[0] + offset
        else:
            raise ValueError("invalid 'whence'; should be 0 or 'start', 1 or "
                             "'current', or 2 or 'end'.")

        return self.offset

    def tell(self, unit=None):
        """Current offset in the file.

        Parameters
        ----------
        unit : `~astropy.units.Unit` or str, optional
            Time unit the offset should be returned in.  By default, no unit
            is used, i.e., an integer enumerating samples is returned. For the
            special string 'time', the absolute time is calculated.

        Returns
        -------
        offset : int, `~astropy.units.Quantity`, or `~astropy.time.Time`
             Offset in current file (or time at current position).
        """
        if unit is None:
            return self.offset

        # "isinstance" avoids costly comparisons of an actual unit with 'time'.
        if not isinstance(unit, u.UnitBase) and unit == 'time':
            return self._tell_time(self.offset)

        return (self.offset / self.sample_rate).to(unit)

    def _tell_time(self, offset):
        """Calculate time for given offset.

        Used for ``start_time``, ``time``, ``stop_time`` and
        ``tell(unit='time')``.  Simple implementation is present mostly so
        subclasses like Integration and Stack can override as appropriate.

        """
        return self._start_time + offset / self.sample_rate

    def read(self, count=None, out=None):
        """Read a number of complete samples.

        Parameters
        ----------
        count : int or None, optional
            Number of complete samples to read. If `None` (default) or
            negative, the number of samples left. Ignored if ``out`` is given.
        out : None or array, optional
            Array to store the samples in. If given, ``count`` will be inferred
            from the first dimension; the remaining dimensions should equal
            `sample_shape`.

        Returns
        -------
        out : `~numpy.ndarray` of float or complex
            The first dimension is sample-time, and the remaining ones are
            as given by `sample_shape`.
        """
        if self.closed:
            raise ValueError("I/O operation on closed stream.")

        samples_left = self.shape[0] - self.offset
        if out is None:
            if count is None or count < 0:
                count = max(0, samples_left)

            out = Empty((count,) + self.sample_shape, dtype=self.dtype)
        else:
            assert out.shape[1:] == self.sample_shape, (
                "'out' must have trailing shape {}".format(self.sample_shape))
            count = out.shape[0]

        if count > samples_left:
            raise EOFError("cannot read from beyond end of input.")

        offset0 = self.offset
        sample = 0
        while sample < count:
            # For current position, get frame plus offset in that frame.
            frame, sample_offset = self._get_frame(self.offset)
            nsample = min(count - sample, len(frame) - sample_offset)
            data = frame[sample_offset:sample_offset + nsample]
            # Copy to relevant part of output.
            out[sample:sample + nsample] = data
            sample += nsample
            # Explicitly set offset (leaving get_frame free to adjust it).
            self.offset = offset0 + sample

        return out if isinstance(out, np.ndarray) else getattr(out, 'array', out)

    def _get_frame(self, offset):
        """Get a frame that includes given offset.

        Finds the index corresponding to the needed frame, assuming frames
        are all the same length.  If not already cached, it retrieves a
        frame by calling ``self._read_frame(index)``.

        Parameters
        ----------
        offset : int
            Offset in the stream for which a frame should be found.

        Returns
        -------
        frame : `~baseband.base.frame.FrameBase`
            Frame holding the sample at ``offset``.
        sample_offset : int
            Offset within the frame corresponding to ``offset``.
        """
        frame_index, sample_offset = divmod(offset, self.samples_per_frame)
        if frame_index != self._frame_index:
            # Read the frame required. Set offset to start so that _read_frame
            # can count on tell() being correct.
            self.offset = frame_index * self.samples_per_frame
            self._frame = self._read_frame(frame_index)
            self._frame_index = frame_index

        return self._frame, sample_offset

    def __getitem__(self, item):
        from .shaping import GetSlice

        return GetSlice(self, item)

    def __array__(self, dtype=None):
        old_offset = self.tell()
        try:
            self.seek(0)
            return np.array(self.read(), dtype=dtype, copy=False)
        finally:
            self.seek(old_offset)

    def __array_ufunc__(self, *args, **kwargs):
        return NotImplemented

    def __array_function__(self, *args, **kwargs):
        return NotImplemented

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.closed = True
        self._frame = None  # clear possibly cached frame


class BaseTaskBase(Base):
    """Base for all classes that operate on underlying streams.

    Following the design of `baseband` stream readers, features properties
    describing the size, shape, data type, sample rate and start/stop times of
    the task's output.  Also defines methods to move a sample pointer across
    the output data in units of either complete samples or time.

    Subclasses should define

      ``_read_frame``: method to read and process a single block of data

    By default, all parameters are taken from the underlying stream.

    Note that no consistency checks are done between the parameters passed in
    and those of the underlying stream; the appropriate base class for most
    tasks is `~baseband_tasks.base.TaskBase`.

    Parameters
    ----------
    ih : stream handle
        Handle of a stream reader or another task.
    ih_samples_per_frame : int, optional
        Number of input samples which should be dealt with in one go.
    shape : tuple, optional
        Overall shape of the stream, with first entry the total number
        of complete samples, and the remainder the sample shape.
    start_time : `~astropy.time.Time`, optional
        Start time of the stream.
    sample_rate : `~astropy.units.Quantity`, optional
        With units of a rate.
    samples_per_frame : int, optional
        Number of output samples produced per frame.  By default, equal
        to the number of input samples.
    frequency : `~astropy.units.Quantity`, optional
        Frequencies for each channel.  Should be broadcastable to the
        sample shape.
    sideband : array, optional
        Whether frequencies are upper (+1) or lower (-1) sideband.
        Should be broadcastable to the sample shape.
    polarization : array or (nested) list of char, optional
        Polarization labels.  Should broadcast to the sample shape,
        i.e., the labels are in the correct axis.  For instance,
        ``['X', 'Y']``, or ``[['L'], ['R']]``.
    dtype : `~numpy.dtype`, optional
        Output dtype.

    """

    def __init__(self, ih, *, ih_samples_per_frame=None,
                 start_time=None, shape=None, sample_rate=None,
                 samples_per_frame=None, frequency=None, sideband=None,
                 polarization=None, dtype=None):
        self.ih = ih
        if ih_samples_per_frame is None:
            ih_samples_per_frame = ih.samples_per_frame
        self._ih_samples_per_frame = ih_samples_per_frame

        if shape is None:
            shape = ih.shape
        if start_time is None:
            start_time = ih.start_time
        if sample_rate is None:
            sample_rate = ih.sample_rate
        if samples_per_frame is None:
            samples_per_frame = self._ih_samples_per_frame
        if dtype is None:
            dtype = ih.dtype
        if frequency is None:
            frequency = getattr(ih, 'frequency', None)
        if sideband is None:
            sideband = getattr(ih, 'sideband', None)
        if polarization is None:
            polarization = getattr(ih, 'polarization', None)

        super().__init__(shape=shape, start_time=start_time,
                         sample_rate=sample_rate,
                         samples_per_frame=samples_per_frame,
                         frequency=frequency, sideband=sideband,
                         polarization=polarization, dtype=dtype)

    def _repr_item(self, key, default, value=None):
        if key == 'ih':
            return 'ih'
        if default is None:
            if key == 'samples_per_frame':
                default = self._ih_samples_per_frame
            elif key == 'ih_samples_per_frame':
                default = self.ih.samples_per_frame
            else:
                default = getattr(self.ih, key, None)

        return super()._repr_item(key, default=default, value=value)

    def __repr__(self):
        base = super().__repr__()
        if base.count('\n') == 1:
            base = ' '.join(b.strip() for b in base.split('\n'))
        return (base
                + "\nih: "
                + "\n    ".join(repr(self.ih).split('\n')))

    def close(self):
        """Close task.

        Note that this does not explicitly close the underlying source;
        instead, it just deletes the reference to it.
        """
        super().close()
        # Delete the reference to the underlying filehandle, so that it
        # can be freed if used nowhere else.
        del self.ih


class TaskBase(BaseTaskBase):
    """Base class of all tasks.

    Following the design of `baseband` stream readers, features properties
    describing the size, shape, data type, sample rate and start/stop times of
    the task's output.  Also defines methods to move a sample pointer across
    the output data in units of either complete samples or time.

    This class provides a base ``_read_frame`` method that will read
    a frame worth of data from the underlying stream and pass it on to
    a task method.  Hence, subclasses should define:

      ``task(self, data)`` : return processed data from one frame.

    Parameters
    ----------
    ih : stream handle
        Handle of a stream reader or another task.
    ih_samples_per_frame : int, optional
        Number of input samples which should be dealt with in one go.
        If not given, inferred from ``samples_per_frame``; if that
        is also not given, taken from the underlying stream.
    shape : tuple, optional
        Overall shape of the stream, with first entry the total number
        of complete samples, and the remainder the sample shape.  If
        not given, the sample shape is that of the underlying stream,
        and the number of complete samples is inferred from the number
        of frames implied by ``ih_samples_per_frame``, combined with
        ``samples_per_frame``. The latter inference also happens if
        the first entry is ``-1``. If a shape inconsistent with an
        integer number of frames is given, the task should be able to
        deal with a partial frame (which, when the sample rate is reduced,
        will always contain an integer multiple of the reduction factor).
    sample_rate : `~astropy.units.Quantity`, optional
        With units of a rate.  If not given, taken from the underlying
        stream.
    samples_per_frame : int, optional
        Number of samples the task produces per frame.  If not given,
        inferred from the input number of samples using the ratio of
        the ``sample_rate`` passed in and that of the underlying stream.
    **kwargs
        Possible further arguments; see `~baseband_tasks.base.BaseTaskBase`.
    """

    def __init__(self, ih, *, ih_samples_per_frame=None,
                 shape=None, sample_rate=None, samples_per_frame=None,
                 **kwargs):

        if sample_rate is None:
            sample_rate = ih.sample_rate
            sample_rate_ratio = 1.
        else:
            sample_rate_ratio = (ih.sample_rate / sample_rate).to(1).value
        if samples_per_frame is None:
            if ih_samples_per_frame is None:
                ih_samples_per_frame = ih.samples_per_frame
            samples_per_frame = ih_samples_per_frame / sample_rate_ratio
            assert samples_per_frame % 1 == 0, (
                "inferred samples per frame must be integer")
            samples_per_frame = int(samples_per_frame)

        elif ih_samples_per_frame is None:
            ih_samples_per_frame = samples_per_frame * sample_rate_ratio
            assert ih_samples_per_frame % 1 == 0, (
                "inferred input samples per frame must be integer")
            ih_samples_per_frame = int(ih_samples_per_frame)

        assert ih_samples_per_frame <= ih.shape[0], (
            "time per frame larger than total time in stream")

        if shape is None or shape[0] == -1:
            ns = ((ih.shape[0] // ih_samples_per_frame)
                  * samples_per_frame)
            shape = (ns,) + (ih.shape[1:] if shape is None else shape[1:])

        super().__init__(ih=ih, ih_samples_per_frame=ih_samples_per_frame,
                         shape=shape, sample_rate=sample_rate,
                         samples_per_frame=samples_per_frame,
                         **kwargs)
        alignment = max(1, int(sample_rate_ratio))
        self._ih_stop = (self.ih.shape[0] // alignment) * alignment

    def _seek_frame(self, frame_index):
        return self.ih.seek(frame_index * self._ih_samples_per_frame)

    def _read_frame(self, frame_index):
        # Read data from underlying filehandle.
        start = self._seek_frame(frame_index)
        stop = min(start + self._ih_samples_per_frame, self._ih_stop)
        # Temporary workaround
        if (isinstance(self.ih, SetAttribute)
                and self.ih.read.__func__ is self.ih.simple_read.__func__
                and not isinstance(self.ih.ih, Base)
                and Empty is not np.empty):
            from dask import array as da
            data = da.from_array(self.ih[start:stop],
                                 chunks=(-1,)+(1,)*(self.ndim-1),
                                 lock=True, asarray=True)
        else:
            data = self.ih.read(stop-start)
        # Apply function to the data.  Note that the _get_frame() function
        # in base ensures that our offset pointer is correct.
        return self.task(data)


class PaddedTaskBase(TaskBase):
    """Base for tasks which need more points than they produce.

    Like `~baseband_tasks.base.TaskBase`, subclasses should define:

      ``task(self, data)`` : return processed data from one frame.

    Where ``data`` will contain extra padding.  The ``task`` method has to
    ensure the right selection is returned, and can use the ``_pad_start``
    and ``_pad_end`` attributes for this purpose.

    Parameters
    ----------
    ih : stream handle
        Handle of a stream reader or another task.
    pad_start, pad_end : int
        Padding to apply at the start and end.  Default: 0.
    samples_per_frame : int, optional
        Number of output samples which should be produced in each frame.
        The number of input samples will be larger by the padding.
        If not given, the minimum power of 2 needed to get at least 75%
        efficiency.
    **kwargs
        Possible further arguments; see `~baseband_tasks.base.BaseTaskBase`.

    """

    def __init__(self, ih, pad_start=0, pad_end=0, *,
                 samples_per_frame=None, **kwargs):
        self._pad_start = operator.index(pad_start)
        self._pad_end = operator.index(pad_end)
        if self._pad_start < 0 or self._pad_end < 0:
            raise ValueError("padding values must be 0 or positive.")

        pad = self._pad_start + self._pad_end
        if samples_per_frame is None:
            # Calculate the number of samples that ensures >75% efficiency:
            # use 4 times power of two just above pad.
            ih_samples_per_frame = ih.samples_per_frame
            if pad > 0:
                ih_samples_per_frame = max(ih_samples_per_frame, 2 ** (
                    int((np.ceil(np.log2(pad)))) + 2))
            samples_per_frame = ih_samples_per_frame - pad
        else:
            ih_samples_per_frame = samples_per_frame + pad

        if pad > samples_per_frame:
            warnings.warn("task will be inefficient; for {} samples "
                          "per frame, more ({}) will be added for padding."
                          .format(samples_per_frame, pad))

        n_sample = (((ih.shape[0] - pad) // samples_per_frame)
                    * samples_per_frame)
        shape = (n_sample,) + ih.sample_shape
        start_time = ih.start_time + self._pad_start / ih.sample_rate
        super().__init__(ih, ih_samples_per_frame=ih_samples_per_frame,
                         shape=shape, samples_per_frame=samples_per_frame,
                         start_time=start_time, **kwargs)

    def _seek_frame(self, frame_index):
        return self.ih.seek(frame_index * self.samples_per_frame)


class Task(TaskBase):
    """Apply a user-supplied callable to a stream.

    The task can either behave like a function or a method.  If a function, it
    will be passed just the frame data read from the underlying file or task;
    if a method, it will be passed the Task instance (with its offset at the
    correct sample) as well as the frame data read.

    Note that for common functions it is recommended to instead define a
    new subclass of `~baseband_tasks.base.TaskBase` in which a ``task``
    (static)method is defined.

    Parameters
    ----------
    ih : filehandle
        Source of data, or another task, from which samples are read.
    task : callable
        The function or method-like callable.
    method : bool, optional
        Whether ``task`` is a method (two arguments) or a function
        (one argument).  Default: inferred by inspection.
    **kwargs
        Additional arguments to be passed on to the base class

    --- Possible arguments : (see `~baseband_tasks.base.TaskBase`)

    shape : tuple, optional
        Overall shape of the stream, with first entry the total number
        of complete samples, and the remainder the sample shape.  By
        default, the shape of the underlying stream, possibly adjusted
        for a difference of sample rate.
    sample_rate : `~astropy.units.Quantity`, optional
        With units of a rate.  If not given, taken from the underlying
        stream.  Should be passed in if the function reduces or expands
        the number of elements.
    samples_per_frame : int, optional
        Number of samples which should be fed to the function in one go.
        If not given, by default the number from the underlying file,
        possibly adjusted for a difference in sample rate.
    frequency : `~astropy.units.Quantity`, optional
        Frequencies for each channel.  Should be broadcastable to the
        sample shape.  Default: taken from the underlying stream, if available.
    sideband : array, optional
        Whether frequencies are upper (+1) or lower (-1) sideband.
        Should be broadcastable to the sample shape.
        Default: taken from the underlying stream, if available.
    polarization : array or (nested) list of char, optional
        Polarization labels.  Should broadcast to the sample shape,
        i.e., the labels are in the correct axis.  For instance,
        ``['X', 'Y']``, or ``[['L'], ['R']]``.  Default: taken from the
        underlying stream, if available.
    dtype : `~numpy.dtype`, optional
        Output dtype.  If not given, the dtype of the underlying file.

    Raises
    ------
    TypeError
        If inspection of the task does not work.
    AssertionError
        If the task has zero or more than 2 arguments.
    """

    def __init__(self, ih, task, method=None, **kwargs):
        if method is None:
            try:
                argspec = inspect.getfullargspec(task)
                narg = len(argspec.args)
                if argspec.defaults:
                    narg -= len(argspec.defaults)
                if inspect.ismethod(task):
                    narg -= 1
                assert 1 <= narg <= 2
                method = narg == 2
            except Exception as exc:
                exc.args += ("cannot determine whether ``task`` is a "
                             "function or method. Pass in ``method``.",)
                raise

        if method:
            self.task = types.MethodType(task, self)
        else:
            self.task = task

        super().__init__(ih, **kwargs)

    def _repr_item(self, key, default, value=None):
        if key == 'task' and isinstance(self.task, types.MethodType):
            value = self.task.__func__
        return super()._repr_item(key, default=default, value=value)


class SetAttribute(TaskBase):
    """Wrapper for streams that allows one to set or change attributes.

    Can be used to add ``frequency``, ``sideband`` and ``polarization``
    attributes, which most baseband readers do not provide, checking that
    the values broadcast properly to the sample shape.

    Can also be used to apply a clock correction by changing ``start_time``.

    The ``sample_rate`` can also be set, but no check is done to ensure it
    remains consistent with the ``frequency``.

    One can also change the ``shape``, ``samples_per_frames``, or ``dtype``
    but that will be slightly slower, as the class can then not read directly
    from the underlying stream.

    By default, all parameters are taken from the underlying stream.

    Parameters
    ----------
    ih : stream handle
        Handle of a stream reader or another task.
    start_time : `~astropy.time.Time`, optional
        Start time of the stream.
    sample_rate : `~astropy.units.Quantity`, optional
        With units of a rate.
    frequency : `~astropy.units.Quantity`, optional
        Frequencies for each channel.  Should be broadcastable to the
        sample shape.
    sideband : array, optional
        Whether frequencies are upper (+1) or lower (-1) sideband.
        Should be broadcastable to the sample shape.
    polarization : array or (nested) list of char, optional
        Polarization labels.  Should broadcast to the sample shape,
        i.e., the labels are in the correct axis.  For instance,
        ``['X', 'Y']``, or ``[['L'], ['R']]``.
    **kwargs
        Any other parameters.  See `~baseband_tasks.base.TaskBase`.
    """

    def __init__(self, ih, *, start_time=None, sample_rate=None,
                 frequency=None, sideband=None, polarization=None,
                 **kwargs):
        super().__init__(ih, start_time=start_time, sample_rate=sample_rate,
                         frequency=frequency, sideband=sideband,
                         polarization=polarization, **kwargs)
        if not kwargs:
            # No overrides of anything related to data, so can use read of
            # underlying file directly.
            self.read = self.simple_read

    def simple_read(self, *args, **kwargs):
        """Read data from the underlying stream at the current offset."""
        # Used as ``read`` if the data aspects are not changed. See above.
        self.ih.seek(self.offset)
        return self.ih.read(*args, **kwargs)

    def task(self, data):
        return data
