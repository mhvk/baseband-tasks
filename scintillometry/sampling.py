# Licensed under the GPLv3 - see LICENSE
"""Resampling of baseband signals."""

import numpy as np
from astropy import units as u
from astropy.utils import lazyproperty

from .base import PaddedTaskBase
from .fourier import fft_maker

__all__ = ['Resample', 'float_offset']

__doctest_requires__ = {'Resample*': ['pyfftw']}


def float_offset(ih, offset, whence=0):
    """Get a float sample position.

    Same as ``ih.seek()``, but without rounding.
    """
    try:
        offset = float(offset)
    except Exception:
        try:
            offset = offset - ih.start_time
        except Exception:
            pass
        else:
            whence = 0

        offset = (offset * ih.sample_rate).to_value(u.one)

    if whence == 0 or whence == 'start':
        return offset
    elif whence == 1 or whence == 'current':
        return ih.offset + offset
    elif whence == 2 or whence == 'end':
        return ih.shape[0] + offset
    else:
        raise ValueError("invalid 'whence'; should be 0 or 'start', 1 or "
                         "'current', or 2 or 'end'.")


class Resample(PaddedTaskBase):
    """Rsample a stream such that a sample occurs at the given offset.

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
        Number of samples which should be resampled in one go. The number of
        output samples per frame will be one less than this.  If not given,
        the larger of the sampler per frame in the underlying stream or 1024.

    Examples
    --------
    Suppose one wanted to read the 8 samples surrounding a precise time::

      >>> from scintillometry.sampling import Resample
      >>> from astropy.time import Time
      >>> from baseband import data, vdif
      >>> fh = vdif.open(data.SAMPLE_VDIF)
      >>> texact = Time('2014-06-16T05:56:07.000123456')
      >>> ((texact - fh.start_time) * fh.sample_rate).to(1)  # doctest: +FLOAT_CMP
      <Quantity 3950.59201992>
      >>> rh = Resample(fh, texact)
      >>> rh.time.isot
      '2014-06-16T05:56:07.000123456'
      >>> rh.seek(-4, 1)
      3946
      >>> data = rh.read(8)
      >>> data[4]  # doctest: +FLOAT_CMP
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

        ih_offset = float_offset(ih, offset, whence)
        rounded_offset = np.around(ih_offset)
        fraction = ih_offset - rounded_offset
        if fraction < 0:
            pad_start, pad_end = 1, 0
        else:
            pad_start, pad_end = 0, 1

        if samples_per_frame is None:
            samples_per_frame = max(ih.samples_per_frame, 1024)

        super().__init__(ih, pad_start=pad_start, pad_end=pad_end,
                         samples_per_frame=samples_per_frame)

        self._fft = fft_maker(shape=(self._padded_samples_per_frame,) +
                              ih.sample_shape, sample_rate=ih.sample_rate,
                              dtype=ih.dtype)
        self._ifft = self._fft.inverse()

        self._fraction = fraction
        self._start_time += fraction / ih.sample_rate
        self._pad_slice = slice(self._pad_start,
                                self._padded_samples_per_frame - self._pad_end)
        self.seek(int(rounded_offset) - self._pad_start)

    @lazyproperty
    def phase_factor(self):
        """Phase offsets of the Fourier-transformed frame."""
        phase_delay = (self._fraction / self.sample_rate * u.cycle *
                       self._fft.frequency)
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
