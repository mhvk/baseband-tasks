# Licensed under the GPLv3 - see LICENSE
"""Resampling of baseband signals."""

import numpy as np
from astropy import units as u
from astropy.utils import lazyproperty

from .base import PaddedTaskBase, TaskBase, SetAttribute, check_broadcast_to
from .fourier import fft_maker

__all__ = ['Resample', 'float_offset', 'TimeDelay', 'DelayAndResample']

__doctest_requires__ = {'Resample*': ['pyfftw']}


def float_offset(ih, offset, whence=0):
    """Get a float sample position.

    Similar to ``ih.seek()``, but without rounding, and allowing
    offsets that are different for different sample streams.

    Parameters
    ----------
    ih : stream handle
        Handle of a stream reader or task.
    offset : int, `~astropy.units.Quantity`, or `~astropy.time.Time`
        Offset to move to.  Can be an (float) number of samples,
        an offset in time units, or an absolute time.  Should be
        broadcastable to the stream sample shape.
    whence : {0, 1, 2, 'start', 'current', or 'end'}, optional
        Like regular seek, the offset is taken to be from the start if
        ``whence=0`` (default), from the current position if 1,
        and from the end if 2.  One can alternativey use 'start',
        'current', or 'end' for 0, 1, or 2, respectively.  Ignored if
        ``offset`` is a time.
    """
    try:
        offset = u.Quantity(offset)
    except Exception:
        offset = (offset - ih.start_time) * ih.sample_rate
        whence = 0
    else:
        if not offset.unit.is_equivalent(u.one):
            offset = offset * ih.sample_rate

    offset = offset.to_value(u.one)
    check_broadcast_to(offset, ih.sample_shape)

    if whence == 0 or whence == 'start':
        return offset
    elif whence == 1 or whence == 'current':
        return ih.offset + offset
    elif whence == 2 or whence == 'end':
        return ih.shape[0] + offset
    else:
        raise ValueError("invalid 'whence'; should be 0 or 'start', 1 or "
                         "'current', or 2 or 'end'.")


class ResampleBase(PaddedTaskBase):
    def __init__(self, ih, offset, *,
                 samples_per_frame=None, **kwargs):
        if samples_per_frame is None:
            samples_per_frame = max(ih.samples_per_frame-1, 1023)

        sample_offset = round(np.mean(offset))
        pad_start = int(np.ceil(-np.min(offset - sample_offset)))
        pad_end = int(np.floor(np.max(offset - sample_offset))) + 1
        super().__init__(ih, pad_start=pad_start, pad_end=pad_end,
                         samples_per_frame=samples_per_frame, **kwargs)

        self._fft = fft_maker(shape=(self._ih_samples_per_frame,)
                              + ih.sample_shape, sample_rate=ih.sample_rate,
                              dtype=ih.dtype)
        self._ifft = self._fft.inverse()

        self._fraction = offset - sample_offset
        self._start_time += sample_offset / ih.sample_rate
        self._pad_slice = slice(self._pad_start,
                                self._pad_start + self.samples_per_frame)

    @lazyproperty
    def phase_factor(self):
        """Phase offsets of the Fourier-transformed frame."""
        phase_delay = (self._fraction / self.sample_rate * u.cycle
                       * self._fft.frequency)
        phase_factor = np.exp(phase_delay.to_value(u.rad) * 1j)
        phase_factor = phase_factor.astype(self._fft.frequency_dtype,
                                           copy=False)
        return phase_factor

    def task(self, data):
        ft = self._fft(data)
        ft *= self.phase_factor
        result = self._ifft(ft)
        return result[self._pad_slice]

    def close(self):
        super().close()
        # Clear the caches of the lazyproperties to release memory.
        del self.phase_factor
        del self._fft
        del self._ifft

    def _repr_item(self, key, default, value=None):
        if key == 'offset':
            value = self._offset
        return super()._repr_item(key, default=default, value=value)


class Resample(ResampleBase):
    """Resample a stream such that a sample occurs at the given offset.

    The offset pointer is left at the requested time.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream, with time as the first axis.
    offset : int, `~astropy.units.Quantity`, or `~astropy.time.Time`
        Offset to ensure a sample falls on.  Can be a float number of samples,
        an offset in time units, or an absolute time.
    whence : {0, 1, 2, 'start', 'current', or 'end'}, optional
        Like regular seek, the offset is taken to be from the start if
        ``whence=0`` (default), from the current position if 1,
        and from the end if 2.  One can alternativey use 'start',
        'current', or 'end' for 0, 1, or 2, respectively.  Ignored if
        ``offset`` is a time.
    samples_per_frame : int, optional
        Number of resampled samples which should be produced in one go.
        The number of input samples used will be larger to avoid wrapping.
        If not given, works on the larger of the samples per frame from
        the underlying stream or 1024.

    Examples
    --------
    Suppose one wanted to read the 8 samples surrounding a precise time::

      >>> from baseband_tasks.sampling import Resample
      >>> from astropy.time import Time
      >>> from baseband import data, vdif
      >>> fh = vdif.open(data.SAMPLE_VDIF)
      >>> texact = Time('2014-06-16T05:56:07.000123456')
      >>> ((texact - fh.start_time) * fh.sample_rate).to(1)
      ... # doctest: +FLOAT_CMP
      <Quantity 3950.59201992>
      >>> rh = Resample(fh, texact)
      >>> rh.time.isot
      '2014-06-16T05:56:07.000123456'
      >>> rh.seek(-4, 1)
      3946
      >>> data = rh.read(8)
      >>> data[4]  # doctest: +SKIP
      array([-1.6905369 ,  0.52486056, -2.00316   ,  0.9242443 , -0.0470082 ,
              1.6006405 , -1.8970288 ,  1.1860422 ], dtype=float32)

    For comparison, if one uses the underlying filehandle directly, one gets
    the data only at the approximate time::

      >>> fh.seek(texact)
      3951
      >>> fh.time.isot
      '2014-06-16T05:56:07.000123469'
      >>> fh.seek(-4, 1)
      3947
      >>> data = fh.read(8)
      >>> data[4]  # doctest: +FLOAT_CMP
      array([-3.316505,  3.316505, -3.316505, -1.      ,  1.      ,  1.      ,
             -3.316505,  1.      ], dtype=float32)
      >>> fh.close()
    """

    def __init__(self, ih, offset, whence='start', *,
                 samples_per_frame=None):

        self._offset = offset
        self._whence = whence

        ih_offset = float_offset(ih, offset, whence)
        rounded_offset = np.around(ih_offset)
        fraction = ih_offset - rounded_offset
        super().__init__(ih, fraction, samples_per_frame=samples_per_frame)
        self._start_time += self._fraction / ih.sample_rate
        self.seek(int(rounded_offset) - self._pad_start)


class TimeDelay(TaskBase):
    def __init__(self, ih, delay, *, lo=None,
                 frequency=None, sideband=None):
        assert ih.complex_data, "Time delay only works on complex data."
        super().__init__(ih, frequency=frequency, sideband=sideband)
        if lo is None:
            lo = self.frequency
        self._start_time += delay
        lo_phase_delay = -delay * lo * self.sideband * u.cycle
        self._phase_factor = (np.exp(lo_phase_delay.to_value(u.rad) * 1j)
                              .astype(ih.dtype))

    def task(self, data):
        data *= self._phase_factor
        return data


class DelayAndResample(ResampleBase):
    def __init__(self, ih, delay, offset=None, *, lo=None,
                 samples_per_frame=None, frequency=None, sideband=None):
        if offset is None:
            offset = ih.start_time

        delayed = SetAttribute(ih, start_time=ih.start_time+delay)
        ih_offset = float_offset(delayed, offset)
        fraction = ih_offset - round(ih_offset)

        super().__init__(ih, fraction, samples_per_frame=samples_per_frame,
                         frequency=frequency, sideband=sideband)
        if lo is None:
            lo = self.frequency

        self._start_time += delay
        self._lo_phase_delay = -delay * lo * self.sideband * u.cycle

    @lazyproperty
    def phase_factor(self):
        """Phase offsets of the Fourier-transformed frame."""
        phase_delay = (self._fraction / self.sample_rate * u.cycle
                       * self._fft.frequency) + self._lo_phase_delay
        phase_factor = np.exp(phase_delay.to_value(u.rad) * 1j)
        phase_factor = phase_factor.astype(self._fft.frequency_dtype,
                                           copy=False)
        return phase_factor
