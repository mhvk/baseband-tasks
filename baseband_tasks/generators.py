# Licensed under the GPLv3 - see LICENSE
"""Collection of source generator classes.

All these look like stream readers and thus are useful to test pipelines
with artificial data.
"""
import numpy as np

from .base import Base


__all__ = ['StreamGenerator', 'EmptyStreamGenerator',
           'Noise', 'NoiseGenerator']


class StreamGenerator(Base):
    """Generator of data produced by a user-provided function.

    The function needs to be aware of stream structure.  As an alternative,
    use `~baseband_tasks.generators.EmptyStreamGenerator` to generate an empty
    stream and add a `~baseband_tasks.base.Task` that fills data arrays.

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
        Blocking factor.  This can be used for efficiency to reduce the
        overhead of calling the source function.
    dtype : `~numpy.dtype` or anything that initializes one, optional
        Type of data produced.  Default: ``complex64``.

    --- **kwargs : meta data for the stream, which usually include

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

    Examples
    --------
    Produce alternating ones and zeros.

    >>> from baseband_tasks.generators import StreamGenerator
    >>> import numpy as np
    >>> from astropy.time import Time
    >>> from astropy import units as u
    >>> def alternate(sh):
    ...     return np.full((1,) + sh.shape[1:], sh.tell() % 2 == 1, sh.dtype)
    ...
    >>> sh = StreamGenerator(alternate, (10, 6), Time('2010-11-12'), 10.*u.Hz)
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
                 samples_per_frame=1, dtype=np.complex64, **kwargs):
        super().__init__(shape=shape, start_time=start_time,
                         sample_rate=sample_rate,
                         samples_per_frame=samples_per_frame, dtype=dtype,
                         **kwargs)
        self._function = function

    def _read_frame(self, frame_index):
        # Apply function to generate data.  Note that the read() function
        # in base ensures that our offset pointer is correct.
        return self._function(self)


class EmptyStreamGenerator(Base):
    """Generator of an empty data stream.

    The stream is meant to be filled with a `~baseband_tasks.base.Task`.

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

    --- **kwargs : meta data for the stream, which usually include

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

    Examples
    --------
    Produce alternating +/-1 in single-channel data with decent-sized blocks.

    >>> from baseband_tasks.generators import EmptyStreamGenerator
    >>> from baseband_tasks.base import Task
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
    >>> sh = Task(eh, alternate)
    >>> sh.seek(995)
    995
    >>> sh.read()  # doctest: +FLOAT_CMP
    array([ 1., -1.,  1., -1.,  1.], dtype=float32)
    """

    def _read_frame(self, frame_index):
        return np.empty((self.samples_per_frame,) + self.shape[1:],
                        self.dtype)


class Noise:
    """Helper class providing source callables for NoiseSource.

    When called, will provide a frame worth of normally distributed data,
    but using the `~numpy.random.Philox` bit generator to ensure that if the
    same frame is read again, the same random data are generated.

    Parameters
    ----------
    seed : int
       Initial seed for `~numpy.random.Philox`.

    Notes
    -----
    Data is identical between invocations only if seeded identically.
    """

    def __init__(self, seed=None):
        self.seed = seed
        self.rng = np.random.Generator(np.random.Philox(self.seed))
        # We store a base state with no buffers set, etc., since we
        # can use that to quickly reset the state for a new counter.
        self.bg_state = self.rng.bit_generator.state

    def __call__(self, sh):
        # We're guaranteed to be at the start of a frame here.
        # Use the offset as the second uint64 in the counter to
        # ensure we get independent but reproducible frame data.
        self.bg_state['state']['counter'][1] = sh.tell()
        self.rng.bit_generator.state = self.bg_state
        shape = (sh.samples_per_frame,) + sh.sample_shape
        if sh.complex_data:
            shape = shape[:-1] + (shape[-1] * 2,)
        numbers = self.rng.normal(size=shape)
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
    samples_per_frame : int, optional
        Blocking factor, setting the size of the fake data frames.
        No default, since should typically be large (see above).
    dtype : `~numpy.dtype` or anything that initializes one, optional
        Type of data produced.  Default: ``complex64``
    seed : int, optional
        Possible seed to initialize the random number generator.

    --- **kwargs : meta data for the stream, which usually include

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

    Notes
    -----
    Between instances, data is identical only if seeded identically *and* if
    first access of frames is done in the same order, with the same number of
    samples per frame.
    """

    def __init__(self, shape, start_time, sample_rate, samples_per_frame,
                 dtype=np.complex64, seed=None, **kwargs):
        generator = Noise(seed)
        super().__init__(function=generator, shape=shape,
                         start_time=start_time, sample_rate=sample_rate,
                         samples_per_frame=samples_per_frame,
                         dtype=dtype, **kwargs)
