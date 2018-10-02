# Licensed under the GPLv3 - see LICENSE
"""Collection of source generator classes.

All these look like stream readers and thus are useful to test pipelines
with artificial data.
"""
import numpy as np

from ..base import Base


__all__ = ['Source', 'ConstantSource', 'NoiseSource']


class Source(Base):
    """Data source that looks like filehandle, produced by a callable.

    Parameters
    ----------
    source : callable
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

    >>> from scintillometry.simulate import ConstantSource
    >>> import numpy as np
    >>> from astropy import time as t, units as u
    >>> def alternate(sh):
    ...     return np.full((1,) + sh.shape[1:], sh.tell() % 2 == 1, sh.dtype)
    ...
    >>> sh = Source(alternate, (10, 6), t.Time('2010-11-12'), 10.*u.Hz, samples_per_frame=1)
    >>> sh.seek(5)
    5
    >>> sh.read()  # doctest: +FLOAT_CMP
    array([[1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j]], dtype=complex64)
    """
    def __init__(self, source, shape, start_time, sample_rate,
                 samples_per_frame, dtype=np.complex64):
        super(Source, self).__init__(shape=shape, start_time=start_time,
                                     sample_rate=sample_rate,
                                     samples_per_frame=samples_per_frame,
                                     dtype=dtype)
        self._source = source

    def _read_frame(self, frame_index):
        self.seek(frame_index * self.samples_per_frame)
        if self.tell() + self.samples_per_frame > self.shape[0]:
            raise EOFError("cannot generate beyond end of source.")
        return self._source(self)

    def close(self):
        self._source = None


class Constant(object):
    """Helper class providing source callables for ConstantSource.

    When called with a source file handle, will create a frame of
    data filled with the constant.

    Parameters
    ----------
    constant : array-like
        Values to be produced.
    """
    def __init__(self, constant):
        self.constant = constant

    def __call__(self, sh):
        data = np.empty((sh.samples_per_frame,) + sh.sample_shape,
                        sh.dtype)
        data[...] = self.constant
        return data


class ConstantSource(Source):
    """Source of constant data that looks like filehandle.

    Parameters
    ----------
    constant : array-like
        Constant signal that is to be produced repeatedly.
    shape : tuple
        First element is the total number of samples of the fake file,
        the others are the sample shape.
    start_time : `~astropy.time.Time`
        Start time of the fake file.
    sample_rate : `~astropy.units.Quantity`
        Sample rate, in units of frequency.
    samples_per_frame : int, optional
        Blocking factor.  By default, the number of samples in ``constant``.
        If passed in, an empty frame of this shape will be filled with
        ``constant``, which can help efficiency.
    dtype : `~numpy.dtype` or anything that initializes one, optional
        Type of data produced.  Default: ``complex64``

    Examples
    --------

    Produce a constant "tone"::

    >>> from scintillometry.simulate import ConstantSource
    >>> import numpy as np
    >>> from astropy import time as t, units as u
    >>> tone = np.zeros(6, dtype=np.complex64)
    >>> tone[3] = 1.
    >>> sh = ConstantSource(tone, (10, 6), t.Time('2010-11-12'), 10.*u.Hz)
    >>> sh.seek(5)
    5
    >>> sh.tell(unit='time')
    <Time object: scale='utc' format='iso' value=2010-11-12 00:00:00.500>
    >>> sh.read(2)  # doctest: +FLOAT_CMP
    array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j]], dtype=complex64)
    """
    def __init__(self, constant, shape, start_time, sample_rate,
                 samples_per_frame=None, dtype=np.complex64):
        constant = np.array(constant, subok=True, copy=False)
        if samples_per_frame is None:
            if len(constant.shape) == len(shape):
                samples_per_frame = constant.shape[0]
            else:
                samples_per_frame = 1

        source = Constant(constant)
        super(ConstantSource, self).__init__(
            source, shape=shape, start_time=start_time, sample_rate=sample_rate,
            samples_per_frame=samples_per_frame, dtype=dtype)


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


class NoiseSource(Source):
    """Source of normally distributed noise that looks like filehandle.

    To mimic proper file handles, data is guaranteed to be identical if read
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
    first access of frames is done in the same order.
    """
    def __init__(self, shape, start_time, sample_rate, samples_per_frame,
                 dtype=np.complex64, seed=None):
        source = Noise(seed)
        super(NoiseSource, self).__init__(source=source, shape=shape,
                                          start_time=start_time,
                                          sample_rate=sample_rate,
                                          samples_per_frame=samples_per_frame,
                                          dtype=dtype)
