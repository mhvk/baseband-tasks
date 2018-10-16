# Licensed under the GPLv3 - see LICENSE

import operator
import numpy as np
import astropy.units as u


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
    dtype : `~numpy.dtype`, optional
        Dtype of the samples.
    """

    def __init__(self, shape, start_time, sample_rate, samples_per_frame=1,
                 frequency=None, sideband=None, dtype=np.complex64):
        self._shape = shape
        self._start_time = start_time
        self._samples_per_frame = samples_per_frame
        assert shape[0] % samples_per_frame == 0
        self._sample_rate = sample_rate
        self._dtype = np.dtype(dtype, copy=False)
        if frequency is not None:
            frequency = self._check_shape(frequency)
        if sideband is not None:
            sideband = self._check_shape(np.where(sideband > 0,
                                                  np.int8(1), np.int8(-1)))
        self._frequency = frequency
        self._sideband = sideband

        # Sample and frame pointers.
        self.offset = 0
        self._frame_index = None
        self.closed = False

    def _check_shape(self, value):
        value = np.array(value, subok=True, copy=False)
        try:
            np.broadcast_to(value, self.sample_shape)
        except ValueError as exc:
            exc.args += ("value cannot be broadcast to sample shape",)
            raise
        return value

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
        return self._start_time

    @property
    def time(self):
        """Time of the sample pointer's current offset in the output.

        See also `start_time` and `stop_time`.
        """
        return self.tell(unit='time')

    @property
    def stop_time(self):
        """Time at the end of the output, just after the last sample.

        See also `start_time` and `time`.
        """
        return self.start_time + self.shape[0] / self.sample_rate

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

    def seek(self, offset, whence=0):
        """Change the sample pointer position."""
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

        if unit == 'time':
            return self.start_time + self.tell(unit=u.s)

        return (self.offset / self.sample_rate).to(unit)

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
            out = np.empty((count,) + self.shape[1:], dtype=self.dtype)
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.closed = True


class TaskBase(Base):
    """Base class of all tasks.

    Following the design of `baseband` stream readers, features properties
    describing the size, shape, data type, sample rate and start/stop times of
    the task's output.  Also defines methods to move a sample pointer across
    the output data in units of either complete samples or time.

    This class provides a base ``_read_frame`` method that will read
    a frame worth of data from the underlying stream and pass it on to
    a function method.  Hence, subclasses should define:

      ``function(self, data)`` : return processed data from one frame.

    Parameters
    ----------
    ih : stream handle
        Handle of a stream reader or another task.
    shape : tuple, optional
        Overall shape of the stream, with first entry the total number
        of complete samples, and the remainder the sample shape.  By
        default, identical to the shape of the underlying stream.
    sample_rate : `~astropy.units.Quantity`, optional
        With units of a rate.  If not given, taken from the underlying
        stream.
    samples_per_frame : int, optional
        Number of samples the task should handle in one go.  If given,
        ``shape`` will be adjusted to make the total number of samples
        an integer multiple of ``samples_per_frame``.  If not given,
        the number from the underlying stream.
    dtype : `~numpy.dtype`, optional
        Output dtype.  If not given, the dtype of the underlying file.
    """

    def __init__(self, ih, shape=None, sample_rate=None,
                 frequency=None, sideband=None, samples_per_frame=None,
                 dtype=None):
        self.ih = ih
        if sample_rate is None:
            sample_rate = ih.sample_rate
            sample_rate_ratio = 1.
        else:
            sample_rate_ratio = (ih.sample_rate / sample_rate).to(1).value
        if samples_per_frame is None:
            (samples_per_frame, r) = divmod(ih.samples_per_frame /
                                            sample_rate_ratio, 1.)
            assert r == 0, "inferred samples per frame must be integer"
            samples_per_frame = int(samples_per_frame)
            self._raw_samples_per_frame = ih.samples_per_frame
        else:
            (raw_samples_per_frame, r) = divmod(samples_per_frame *
                                                sample_rate_ratio, 1.)
            assert r == 0, "inferred raw samples per frame must be integer"
            self._raw_samples_per_frame = int(raw_samples_per_frame)

        nraw_frames = ih.shape[0] // self._raw_samples_per_frame
        if shape is None:
            shape = (nraw_frames * samples_per_frame,) + ih.shape[1:]
        else:
            assert shape[0] <= nraw_frames * samples_per_frame, \
                "passed in shape[0] too large"

        if dtype is None:
            dtype = ih.dtype

        if frequency is None:
            frequency = getattr(ih, 'frequency', None)

        if sideband is None:
            sideband = getattr(ih, 'sideband', None)

        super().__init__(shape=shape, start_time=ih.start_time,
                         sample_rate=sample_rate,
                         samples_per_frame=samples_per_frame,
                         frequency=frequency, sideband=sideband, dtype=dtype)

    def _read_frame(self, frame_index):
        # Read data from underlying filehandle.
        self.ih.seek(frame_index * self._raw_samples_per_frame)
        data = self.ih.read(self._raw_samples_per_frame)
        # Apply function to the data.  Note that the read() function
        # in base ensures that our offset pointer is correct.
        return self.function(data)

    def close(self):
        """Close task, in particular closing its input source."""
        super().close()
        self.ih.close()
