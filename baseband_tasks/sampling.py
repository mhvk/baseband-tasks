# Licensed under the GPLv3 - see LICENSE
"""Resampling of baseband signals."""

import numpy as np
from astropy import units as u
from astropy.utils import lazyproperty
from astropy.time import Time

from baseband_tasks.base import TaskBase, check_broadcast_to
from baseband_tasks.convolution import Convolve

__all__ = ['float_offset', 'seek_float',
           'ShiftAndResample', 'Resample', 'TimeDelay', 'DelayAndResample']

# The tests do not strictly require pyfftw to run, but they do require it
# to give numbers that are equal to within +FLOAT_CMP precision.
__doctest_requires__ = {'Resample*': ['pyfftw']}


def float_offset(ih, offset):
    """The offset in possible fractional samples.

    Parameters
    ----------
    ih : stream handle
        Handle of a stream reader or task.
    offset : float or `~astropy.units.Quantity`
        Requested offset.  Can be an (float) number of samples or
        an offset in time units.

    Returns
    -------
    offset : float
        Offset in units samples.
    """
    offset = u.Quantity(offset, copy=False)
    return offset.to_value(u.one, equivalencies=[
        (u.one, u.Unit(1/ih.sample_rate))])


def seek_float(ih, offset, whence=0):
    """Get a float sample position.

    Similar to ``ih.seek()``, but without rounding, and allowing
    offsets that are different for different sample streams.

    Parameters
    ----------
    ih : stream handle
        Handle of a stream reader or task.
    offset : float, `~astropy.units.Quantity`, or `~astropy.time.Time`
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
    if isinstance(offset, Time):
        offset = (offset - ih.start_time).to(1./ih.sample_rate)
        whence = 0

    offset = float_offset(ih, offset)

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


class ShiftAndResample(Convolve):
    """Shift and optionally resample a stream in time.

    The shift is added to the sample times, and the stream is optionally
    resampled to ensure a sample falls on the given offset.  The precision
    with which the shifting and resampling is done depends on ``pad``.

    Note that no account is taken of possible phase rotations, which are
    important if the time shift represents a physical delay of the original
    radio signal.  For that, see `~scintillometry.sampling.DelayAndResample`.
    This task is meant to be used for clock corrections, post-mixing
    cable delays, etc.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream, with time as the first axis.
    shift : float, float array-like, or `~astropy.units.Quantity`
        Amount by which to shift samples in time, as (float) samples, or a
        quantity with units of time.  Should broadcast to the sample shape.
    offset : float, `~astropy.units.Quantity`, or `~astropy.time.Time`
        Offset that the output stream should include. Can be an absolute
        time, or a (float) number of samples or time offset relative to the
        start of the underlying stream.  The default of `None` implies that
        the output stream is free to adjust.  Hence, if a single shift is
        given, all that will happen is a change in ``start_time``. To
        ensure the grid stays fixed, pass in ``0``.
    whence : {0, 1, 2, 'start', 'current', or 'end'}, optional
        Like regular seek, the offset is taken to be from the start if
        ``whence=0`` (default), from the current position if 1, and from the
        end if 2.  One can alternativey use 'start', 'current', or 'end' for
        0, 1, or 2, respectively.  Ignored if ``offset`` is a time.
    pad : int, optional
        Padding to apply on each side when shifting data. This sets the size
        of the sinc function which which the data is convolved (see above).
        The default of 32 ensures an accuracy of about 1/32π ~ 0.01.
    samples_per_frame : int, optional
        Number of resampled samples which should be produced in one go.
        The number of input samples used will be larger by ``2*pad+1``.
        If not given, works on the larger of the samples per frame from
        If not given, the larger of the sampler per frame in the underlying
        stream or 14 times the padding (to ensure ~87.5% efficiency).

    """
    def __init__(self, ih, shift, offset=None, whence='start', *,
                 pad=32, samples_per_frame=None):
        self._shift = float_offset(ih, shift)
        shift_mean = np.mean(self._shift)

        if offset is None:
            # Just use the average shift as a time shift, and do
            # the individual shifts relative to it.
            d_time = shift_mean
            self._offset = None
        else:
            # Ensure the time shift lands on the grid given by offset,
            # but remove as many integer cycles as possible, such that
            # individual shifts are as symmetric around 0 as possible.
            self._offset = seek_float(ih, offset, whence)
            d_time = self._offset + np.around(shift_mean - self._offset)

        # The remainder we actually need to shift.
        sample_shift = np.array(self._shift - d_time, ndmin=ih.ndim-1,
                                copy=False, subok=True)
        response = self.sinc_hanning(pad, sample_shift)

        if samples_per_frame is None:
            samples_per_frame = max(ih.samples_per_frame, pad * 14)

        super().__init__(ih, response, offset=pad - round(sample_shift.min()),
                         samples_per_frame=samples_per_frame)

        self._start_time += d_time / ih.sample_rate

    @staticmethod
    def sinc_hanning(pad, sample_shift):
        """Response for shifting samples.

        A combination of a sinc function, and a Hanning filter that ensures
        the response drops to zero at the edges.
        """
        ishift_max = round(sample_shift.max())
        ishift_min = round(sample_shift.min())
        n_result = 2*pad + 1 + ishift_max - ishift_min
        result = np.zeros((n_result,) + sample_shift.shape)
        for shift, res in zip(sample_shift.ravel(),
                              result.reshape(n_result, -1).T):
            ishift = round(shift.item())
            res[ishift - ishift_min:ishift - ishift_max + n_result] = (
                np.sinc(np.arange(-pad, pad+1) - (shift - ishift))
                * np.hanning(2*pad+1))
        return result

    def _repr_item(self, key, default, value=None):
        if key == 'offset':
            value = self._offset
        return super()._repr_item(key, default=default, value=value)


class Resample(ShiftAndResample):
    """Resample a stream such that a sample occurs at the given offset.

    The offset pointer is left at the requested time, so one can think of
    this task as a precise version of the ``seek()`` method.

    Generally, the stream start time will change, by up to one sample, and
    the stream length reduced by one frame.  The precision with which the
    resampling is done depends on ``pad``.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream, with time as the first axis.
    offset : float, `~astropy.units.Quantity`, or `~astropy.time.Time`
        Offset to ensure the output stream includes.  Can an absolute time,
        or a (float) number of samples or time offset relative to the start
        of the underlying stream.
    whence : {0, 1, 2, 'start', 'current', or 'end'}, optional
        Like regular seek, the offset is taken to be from the start if
        ``whence=0`` (default), from the current position if 1,
        and from the end if 2.  One can alternativey use 'start',
        'current', or 'end' for 0, 1, or 2, respectively.  Ignored if
        ``offset`` is a time.
    pad : int, optional
        Padding to apply on each side when shifting data. This sets the size
        of the sinc function which which the data is convolved (see above).
        The default of 32 ensures an accuracy of about 1/32π ~ 0.01.
    samples_per_frame : int, optional
        Number of resampled samples which should be produced in one go.
        The number of input samples used will be larger by ``2*pad+1``.
        If not given, works on the larger of the samples per frame from
        If not given, the larger of the sampler per frame in the underlying
        stream or 14 times the padding (to ensure ~87.5% efficiency).

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
      3914
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
                 pad=32, samples_per_frame=None):
        super().__init__(ih, shift=0., offset=offset,
                         pad=pad, samples_per_frame=samples_per_frame)
        self.seek(ih.start_time + self._offset / ih.sample_rate)


class TimeDelay(TaskBase):
    r"""Delay a stream by a given amount, taking care of phase rotations.

    The delay is added to the sample times (by adding to the ``start_time``
    of the stream), and the sample phases are rotated as needed if the
    signal was recorded after mixing with a local oscillator. For an upper
    sideband, the phases are rotated by

    .. math:: \phi = - \tau f_{lo}.

    For the lower sideband, the rotation is in the opposite direction.

    Note that the input data stream must be complex.  For real-valued
    streams, use `~scintillometry.sampling.DelayAndResample` without
    ``offset``.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream, with time as the first axis.
    delay : float, `~astropy.units.Quantity`
        Delay to apply to all times. Can be (float) samples, or a
        quantity with units of time.
    lo : `~astropy.units.Quantity`, or `None`
        Local oscillator frequency.  For raw data, this can just be
        ``if.frequency``.  But for channelized data, the actual
        frequency needs to be passed in.  If data were recorded without
        mixing (like for CHIME), pass in `None`.
    frequency : `~astropy.units.Quantity`, optional
        Frequencies for each channel.  Should be broadcastable to the
        sample shape.  By default, taken from the underlying stream.
        (Note that these frequencies are not used in the calculations here.)
    sideband : array, optional
        Whether frequencies are upper (+1) or lower (-1) sideband.
        Should be broadcastable to the sample shape.  By default, taken
        from the underlying stream.  Assumed to be correct for the lo.

    """
    def __init__(self, ih, delay, *, lo, frequency=None, sideband=None):
        assert ih.complex_data, "Time delay only works on complex data."
        self._delay = float_offset(ih, delay)
        self._lo = lo
        delay = self._delay / ih.sample_rate
        super().__init__(ih, frequency=frequency, sideband=sideband)

        self._start_time += delay
        if lo is None:
            self._phase_factor = None
        else:
            lo_phase_delay = delay * lo * self.sideband * u.cycle
            self._phase_factor = np.exp(-1j * lo_phase_delay.to_value(u.rad)
                                        ).astype(ih.dtype)

    def task(self, data):
        if self._phase_factor is not None:
            data *= self._phase_factor
        return data


class DelayAndResample(ShiftAndResample):
    r"""Delay and optionally resample a stream, taking care of phase rotations.

    The delay is added to the sample times, and the stream is optionally
    resampled to ensure a sample falls on the given offset. Furthermore,
    the sample phases are corrected for rotations needed if the signal was
    recorded after mixing with a local oscillator. For an upper sideband,
    their phases are rotated by

    .. math:: \phi = - \tau f_{lo}.

    For the lower sideband, the rotation is in the opposite direction.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream, with time as the first axis.
    delay : float, `~astropy.units.Quantity`
        Delay to apply to all times.  Can be (float) samples or a quantity
        with units of time.
    offset : float, `~astropy.units.Quantity`, or `~astropy.time.Time`
        Offset to ensure the output stream includes.  Can an absolute time,
        or a (float) number of samples or time offset relative to the start
        of the underlying stream.  The default of `None` implies that
        the output stream is free to adjust.  Hence, if a single delay is
        given, all that will happen is a change in ``start_time`` plus a
        phase rotation. To ensure the grid stays fixed, pass in ``0``.
    whence : {0, 1, 2, 'start', 'current', or 'end'}, optional
        Like regular seek, the offset is taken to be from the start if
        ``whence=0`` (default), from the current position if 1,
        and from the end if 2.  One can alternativey use 'start',
        'current', or 'end' for 0, 1, or 2, respectively.  Ignored if
        ``offset`` is a time.
    lo : `~astropy.units.Quantity`, or `None`
        Local oscillator frequency.  For raw data, this can just be
        ``if.frequency``.  But for channelized data, the actual
        frequency needs to be passed in.  If data were recorded without
        mixing (like for CHIME), pass in `None`.
    pad : int, optional
        Padding to apply on each side when shifting data. This sets the size
        of the sinc function which which the data is convolved (see above).
        The default of 32 ensures an accuracy of about 1/32π ~ 0.01.
    samples_per_frame : int, optional
        Number of resampled samples which should be produced in one go.
        The number of input samples used will be larger by ``2*pad+1``.
        If not given, works on the larger of the samples per frame from
        If not given, the larger of the sampler per frame in the underlying
        stream or 14 times the padding (to ensure ~87.5% efficiency).

    """
    def __init__(self, ih, delay, offset=None, whence='start', *,
                 lo, pad=32, samples_per_frame=None):
        super().__init__(ih, shift=delay, offset=offset, pad=pad,
                         samples_per_frame=samples_per_frame)
        self._delay = self._shift / ih.sample_rate
        if lo is None:
            self._lo_phase_delay = 0. * u.cycle
        else:
            self._lo_phase_delay = self._delay * lo * self.sideband * u.cycle

    @lazyproperty
    def _ft_response(self):
        """Phase offsets of the Fourier-transformed frame."""
        return (super()._ft_response
                * np.exp(-1j * self._lo_phase_delay.to_value(u.rad)))
