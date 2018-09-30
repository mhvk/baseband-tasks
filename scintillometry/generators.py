# Licensed under the GPLv3 - see LICENSE
"""Collection of source generator classes.

All these look like stream readers and thus are useful to test pipelines
with artificial data.
"""
import numpy as np

from .base import Base


__all__ = ['StreamGenerator', 'EmptyStreamGenerator', 'Noise', 'NoiseGenerator']


class StreamGeneratorBase(Base):
    """Base for generators.

    Defines a ``_read_frame`` method that calls ``self.function``,
    as well as a ``close`` method.
    """
    def __init__(self, shape, start_time, sample_rate, samples_per_frame,
                 dtype=np.complex64):
        super().__init__(shape=shape, start_time=start_time,
                         sample_rate=sample_rate,
                         samples_per_frame=samples_per_frame, dtype=dtype)
        self.closed = False

    def _read_frame(self, frame_index):
        if self.closed:
            raise ValueError("I/O operation on closed stream generator.")

        self.seek(frame_index * self.samples_per_frame)
        if self.tell() + self.samples_per_frame > self.shape[0]:
            raise EOFError("cannot generate samples beyond end of generator.")
        return self.function()

    def close(self):
        self.closed = True


class StreamGenerator(StreamGeneratorBase):
    """Generator of data produced by a user-provided function.

    The function needs to be aware of stream structure.  As an alternative, generate
    an empty stream with `~scintillometry.generator.EmptyStreamGenerator` and add a
    `~scintillometry.functions.FunctionTask` that fills data arrays.

    Parameters
    ----------
    function : callable
        Function that takes one argument, the Source instance, and returns
        data with the correct shape, i.e., ``samples_per_frame`` samples
        of sample shape ``shape[1:]``.  The function can count on the instance
        being at the start of the frame (i.e., ``instance.tell()`` is correct).
    shape : tuple
        First element is the total number of samples of the fake file,
        the others are the sample shape.
    start_time : `~astropy.time.Time`
        Start time of the fake file.
    sample_rate : `~astropy.units.Quantity`
        Sample rate, in units of frequency.
    samples_per_frame : int
        Blocking factor.  This can be used for efficiency to reduce the overhead
        of calling the source function.
    dtype : `~numpy.dtype` or anything that initializes one, optional
        Type of data produced.  Default: ``complex64``.

    Examples
    --------
    Produce alternating ones and zeros.

    >>> from scintillometry.generators import StreamGenerator
    >>> import numpy as np
    >>> from astropy import time as t, units as u
    >>> def alternate(sh):
    ...     return np.full((1,) + sh.shape[1:], sh.tell() % 2 == 1, sh.dtype)
    ...
    >>> sh = StreamGenerator(alternate, (10, 6), t.Time('2010-11-12'),
    ...                      10.*u.Hz, samples_per_frame=1)
    >>> sh.seek(5)
    5
    >>> sh.read()  # doctest: +FLOAT_CMP
    array([[1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j]], dtype=complex64)
    """
    def __init__(self, function, shape, start_time, sample_rate,
                 samples_per_frame, dtype=np.complex64):
        super().__init__(shape=shape, start_time=start_time,
                         sample_rate=sample_rate,
                         samples_per_frame=samples_per_frame, dtype=dtype)
        self._function = function
        self.closed = False

    def function(self):
        return self._function(self)


class EmptyStreamGenerator(StreamGeneratorBase):
    """Generator of an empty data stream.

    The stream is meant to be filled with a `~scintillometry.functions.FunctionTask`.

    Parameters
    ----------
    shape : tuple
        First element is the total number of samples of the fake file,
        the others are the sample shape.
    start_time : `~astropy.time.Time`
        Start time of the fake file.
    sample_rate : `~astropy.units.Quantity`
        Sample rate, in units of frequency.
    samples_per_frame : int
        Blocking factor.  This is mostly useful to make the function task
        that uses the stream more efficient.
    dtype : `~numpy.dtype` or anything that initializes one, optional
        Type of data produced.  Default: ``complex64``.

    Examples
    --------
    Produce alternating +/-1 in single-channel data with decent-sized blocks.

    >>> from scintillometry.generators import EmptyStreamGenerator
    >>> from scintillometry.functions import FunctionTask
    >>> import numpy as np
    >>> from astropy import time as t, units as u
    >>> def alternate(data):
    ...     value = 2 * (np.arange(data.shape[0]) % 2) - 1
    ...     data[...] = value
    ...     return data
    ...
    >>> eh = EmptyStreamGenerator((1000,), t.Time('2010-11-12'),
    ...                           1.*u.kHz, samples_per_frame=100,
    ...                           dtype='f4')
    >>> sh = FunctionTask(eh, alternate)
    >>> sh.seek(995)
    995
    >>> sh.read()  # doctest: +FLOAT_CMP
    array([ 1., -1.,  1., -1.,  1.], dtype=float32)
    """
    def function(self):
        return np.empty((self.samples_per_frame,) + self.shape[1:],
                        self.dtype)


class Noise(np.random.RandomState):
    """Helper class providing source callables for NoiseSource.

    When called, will provide a frame worth of normally distributed data,
    but also keep the state of the random number generator, so that if the
    same frame is read again, this state can be reused to ensure the same
    data are regenerated.

    Parameters
    ----------
    seed : int
       Initial seed for `~numpy.random.RandomState`.

    Notes
    -----
    Data is identical between invocations only if seeded identically *and*
    read in the same order.
    """
    def __init__(self, seed=None):
        super(Noise, self).__init__(seed)
        self._states = {}

    def __call__(self, sh):
        offset = sh.tell()
        if offset in self._states:
            self.set_state(self._states[offset])
        else:
            self._states[offset] = self.get_state()
        shape = (sh.samples_per_frame,) + sh.sample_shape
        if sh.complex_data:
            shape = shape[:-1] + (shape[-1] * 2,)
        numbers = self.normal(size=shape)
        if sh.complex_data:
            numbers = numbers.view(np.complex128)
        return numbers.astype(sh.dtype, copy=False)


class NoiseGenerator(StreamGenerator):
    """Genertator of a stream of normally distributed noise.

    To mimic proper streams, data is guaranteed to be identical if read
    multiple times from a given instance.  This is done by storing the state
    of the random number generator for each "data frame". Given this, it is
    important to choose ``samples_per_frame`` wisely, such that frame sizes
    are at least of order millions of samples.

    Parameters
    ----------
    shape : tuple
        First element is the total number of samples of the fake file,
        the others are the sample shape.
    start_time : `~astropy.time.Time`
        Start time of the fake file.
    sample_rate : `~astropy.units.Quantity`
        Sample rate, in units of frequency.
    samples_per_frame : int
        Blocking factor, setting the size of the fake data frames (see above).
    dtype : `~numpy.dtype` or anything that initializes one, optional
        Type of data produced.  Default: ``complex64``
    seed : int, optional
        Possible seed to initialize the random number generator.

    Notes
    -----
    Between instances, data is identical only if seeded identically *and* if
    first access of frames is done in the same order, with the same number of
    samples per frame.
    """
    def __init__(self, shape, start_time, sample_rate, samples_per_frame,
                 dtype=np.complex64, seed=None):
        generator = Noise(seed)
        super().__init__(function=generator, shape=shape,
                         start_time=start_time, sample_rate=sample_rate,
                         samples_per_frame=samples_per_frame, dtype=dtype)
