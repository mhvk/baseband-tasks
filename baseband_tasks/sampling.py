# Licensed under the GPLv3 - see LICENSE
"""Resampling of baseband signals."""

import numpy as np
from astropy import units as u
from astropy.utils import lazyproperty
from astropy.time import Time

from baseband_tasks.base import TaskBase, check_broadcast_to, PaddedTaskBase
from baseband_tasks.convolution import Convolve

__all__ = ['seek_float', 'ShiftAndResample', 'Resample', 'TimeDelay',
           'ShiftSamples']


def to_sample(ih, offset):
    """The offset in units of samples."""
    return u.Quantity(offset, copy=False).to_value(u.one, equivalencies=[
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

    offset = to_sample(ih, offset)

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
    r"""Shift and optionally resample a stream in time.

    The shift is added to the sample times, and the stream is optionally
    resampled to ensure a sample falls on the given offset.

    If the shift corresponds to a time delay, and the signal was recorded
    after mixing, the phases should be adjusted, which can be done by
    passing in the local oscillator frequency. For an upper sideband, the
    phase correction is,

    .. math:: \phi = - \tau f_{lo}.

    For the lower sideband, the rotation is in the opposite direction.

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
    lo : `~astropy.units.Quantity`, or `None`, optional
        Local oscillator frequency.  Should be passed in when the signal was
        mixed down before sampling and the shift should be interpreted as a
        time delay (rather than a clock correction). For raw data, this can
        usually just be ``if.frequency``.  But for channelized data, the
        actual frequency needs to be passed in.  If data were recorded
        without mixing (like for CHIME), no phase correction is necessary
        and one should pass in `None`.  Default: `None`.
    pad : int, optional
        Padding to apply on each side when shifting data. This sets the size
        of the sinc function which which the data is convolved (see below).
        Numerical tests suggest that with the default of 64, the accuracy
        is better than 0.1%.
    samples_per_frame : int, optional
        Number of resampled samples which should be produced in one go.
        The number of input samples used will be larger by ``2*pad+1``.
        If not given, works on the larger of the samples per frame from
        If not given, the larger of the sampler per frame in the underlying
        stream or 14 times the padding (to ensure ~87.5% efficiency).

    Notes
    -----
    The precision of the shifting and resampling is controlled by ``pad``,
    as it sets the length of the response, which consists of a sinc function
    combined with a Hann window,

    .. math:: R(x) &= {\rm sinc}(x-s) \cos^2(\pi(x-s)/L),\\
                 -{\rm pad} &\le x \le {\rm pad},\quad L = 2{\rm pad} + 2

    Here, :math:`s` is the fractional pixel offset.  Note that :math:`L`
    is chosen such that :math:`R(\pm{\rm pad})` is still non-zero.

    The convolution is done in the Fourier domain, and the Hann window,
    combined with with the padding done when reading, ensures there is no
    aliasing between the front and the back of each frame.  There is,
    however, aliasing near the Nyquist frequency. For most recorded signals,
    this will not be important, as the band-pass filter will likely have
    ensured there is very little signal near the band edges.  For
    channelized data, however, it may be more problematic and some
    pre-filtering stage may be necessary.

    See Also
    --------
    Resample : resample a stream to a new grid, without time shifts
    ShiftSamples : shift channels by integer number of samples
    TimeDelay : delay a complex stream, changing phases but no resampling
    baseband_tasks.base.SetAttribute : change start time without resampling

    """
    def __init__(self, ih, shift, offset=None, whence='start', *,
                 lo=None, pad=64, samples_per_frame=None):
        self._shift = to_sample(ih, shift)
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
        response = self._windowed_sinc(pad, sample_shift)

        if samples_per_frame is None:
            samples_per_frame = max(ih.samples_per_frame, pad * 14)

        super().__init__(ih, response,
                         offset=pad - int(round(sample_shift.min())),
                         samples_per_frame=samples_per_frame)
        self._lo = lo
        self._start_time += d_time / ih.sample_rate

    def _windowed_sinc(self, pad, sample_shift):
        """Response for shifting samples.

        A combination of a sinc function, and a Hanning filter that ensures
        the response drops to zero at the edges.
        """
        ishift_max = int(round(sample_shift.max()))
        ishift_min = int(round(sample_shift.min()))
        n_result = 2*pad + 1 + ishift_max - ishift_min
        result = np.zeros((n_result,) + sample_shift.shape)
        for shift, res in zip(sample_shift.ravel(),
                              result.reshape(n_result, -1).T):
            ishift = int(round(shift.item()))
            x = np.arange(-pad, pad+1) - (shift - ishift)
            res[ishift - ishift_min:ishift - ishift_max + n_result] = (
                np.sinc(x) * np.cos(np.pi*x/(2*pad+2))**2)
        return result

    # def _window(self, pad):
    #     # Note: tried various window functions (ignoring the shift in them),
    #     # but Hann was clearly superior for the test_sampling test cases.
    #     # return np.hanning(2*pad+3)[1:-1]  # Remove final zeros.
    #     # lanczos
    #     # return np.sinc(np.arange(-pad, pad+1)/pad)
    #     # blackman
    #     # a0, a1, a2, a3 = 0.42, 0.5, 0.08, 0.
    #     # nuttall
    #     a0, a1, a2, a3 = 0.355768, 0.487396, 0.144232, 0.012604
    #     nbyN = np.arange(2*pad+1)/(2*pad)
    #     return (a0
    #             - a1*np.cos(2*np.pi*nbyN)
    #             + a2*np.cos(4*np.pi*nbyN)
    #             - a3*np.cos(6*np.pi*nbyN))

    @lazyproperty
    def _ft_response(self):
        """Phase offsets of the Fourier-transformed frame."""
        if self._lo is None:
            return super()._ft_response
        else:
            lo_phase_delay = (self._shift / self.sample_rate * u.cycle
                              * self._lo * self.sideband)
            return (super()._ft_response
                    * np.exp(-1j * lo_phase_delay.to_value(u.rad)))

    def _repr_item(self, key, default, value=None):
        # Our 'offset' input argument should not be looked up as 'offset',
        # since that can vary.
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
        The default of 64 ensures an accuracy of better than 0.1%.
    samples_per_frame : int, optional
        Number of resampled samples which should be produced in one go.
        The number of input samples used will be larger by ``2*pad+1``.
        If not given, works on the larger of the samples per frame from
        If not given, the larger of the sampler per frame in the underlying
        stream or 14 times the padding (to ensure ~87.5% efficiency).

    See Also
    --------
    ShiftAndResample : also shift a stream, possibly including phase delay

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
      3883
      >>> data = rh.read(8)
      >>> data[:, 4]  # doctest: +SKIP
      array([ 3.8278387 ,  2.0259624 , -0.3738842 , -1.2480919 , -0.04606577,
              2.6100893 ,  3.4867156 ,  3.2312815 ], dtype=float32)

    For comparison, if one uses the underlying filehandle directly, one gets
    the data only at the approximate time::

      >>> fh.seek(texact)
      3951
      >>> fh.time.isot
      '2014-06-16T05:56:07.000123469'
      >>> fh.seek(-4, 1)
      3947
      >>> data = fh.read(8)
      >>> data[:, 4]  # doctest: +FLOAT_CMP
      array([ 3.316505,  1.      , -1.      , -1.      ,  1.      ,  3.316505,
              3.316505,  3.316505], dtype=float32)
      >>> fh.close()

    """

    def __init__(self, ih, offset, whence='start', *,
                 pad=64, samples_per_frame=None):
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
    streams, use `~baseband_tasks.sampling.ShiftAndResample` with
    ``shift`` as the delay, no ``offset``, and ``pad=0`` (this works for
    complex data as well, but is slower as it involves fourier transforms).

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

    See Also
    --------
    ShiftAndResample : also resample a stream, or delay a real-valued stream
    baseband_tasks.base.SetAttribute : change start time without phase delay

    """
    def __init__(self, ih, delay, *, lo, frequency=None, sideband=None):
        assert ih.complex_data, "Time delay only works on complex data."
        self._delay = to_sample(ih, delay)
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


class ShiftSamples(PaddedTaskBase):
    """Shift channels in a stream by integer numbers of samples.

    The shifts are interpreted as additive to the sample times, i.e.,
    positive shifts imply that a stream will appear delayed in time.

    Note that no resampling is done; for shifting by non-integer numbers of
    samples, use `~baseband_tasks.sampling.ShiftAndResample`.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream, with time as the first axis.
    shift : int or float array-like, or `~astropy.units.Quantity`
        Amount by which to shift samples along the stream.  Normally integer,
        but can also be float or quantity with units of time, in which case
        the shifts will be rounded to the nearest integer.  Should broadcast
        to the sample shape.  For instance, to shift along the one-but-last
        axis with length ``N``, the shape of shift should be ``(N, 1)``.
    samples_per_frame : int
        Number of shifted samples which should be produced in one go.
        The number of input samples used will be larger to avoid wrapping.
        If not given, as produced by the minimum power of 2 of input
        samples that yields at least 75% efficiency.

    See Also
    --------
    ShiftAndResample : shift channels by fractional samples and resample
    Resample : resample a stream to a new grid, without time shifts
    """
    def __init__(self, ih, shift, *, samples_per_frame=None):
        shift = self._shift = np.round(to_sample(ih, shift)).astype(int)
        check_broadcast_to(shift, ih.sample_shape)
        start_time = ih.start_time + shift.max() / ih.sample_rate
        super().__init__(ih, pad_start=0, pad_end=shift.ptp(),
                         samples_per_frame=samples_per_frame,
                         start_time=start_time)
        # Form the advanced index used to select the shifted samples.
        # Start with one that just takes unshifted output elements for a
        # single frame, then add shifts for the first (sample) dimension.
        indices = np.ix_(np.arange(self.samples_per_frame),
                         *[np.arange(sh) for sh in self.sample_shape])
        self._indices = (shift.max() - shift + indices[0],) + indices[1:]

    def task(self, data):
        return data[self._indices]
