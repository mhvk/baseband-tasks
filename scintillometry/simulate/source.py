# Licensed under the GPLv3 - see LICENSE
import numpy as np

from ..base import Base


class Source(Base):
    """Data source that looks like filehandle, produced by a callable.

    Parameters
    ----------
    source : callable
        Function that takes one argument, the Source instance, and returns
        data with the correct shape, i.e., ``samples_per_frame`` samples
        of sample shape ``shape[1:]``.
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
        Type of data produced.  Default: ``complex64``
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
        return self._source(self)

    def close(self):
        self._source = None


class Constant(object):
    def __init__(self, constant):
        self.constant = constant

    def __call__(self, sh):
        return np.broadcast_to(self.constant.astype(sh.dtype, copy=False),
                               (sh.samples_per_frame,) + sh.sample_shape)


class ConstantSource(Source):
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
