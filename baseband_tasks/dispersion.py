# Licensed under the GPLv3 - see LICENSE

import numpy as np
from astropy import units as u
from astropy.utils import lazyproperty

from .base import PaddedTaskBase, getattr_if_none, SetAttribute
from .fourier import fft_maker
from .dm import DispersionMeasure
from .sampling import ShiftSamples


__all__ = ['Disperse', 'Dedisperse', 'DisperseSamples', 'DedisperseSamples']


class Disperse(PaddedTaskBase):
    """Coherently disperse a time stream.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream, with time as the first axis.
    dm : float or `~baseband_tasks.dm.DispersionMeasure` quantity
        Dispersion measure.  If negative, will dedisperse correctly, but
        clearer to use the `~baseband_tasks.dispersion.Dedisperse` class.
    reference_frequency : `~astropy.units.Quantity`, optional
        Frequency to which the data should be dispersed.  Can be an array.
        By default, the mean frequency.
    samples_per_frame : int, optional
        Number of dispersed samples which should be produced in one go.
        The number of input samples used will be larger to avoid wrapping.
        If not given, as produced by the minimum power of 2 of input
        samples that yields at least 75% efficiency.
    frequency : `~astropy.units.Quantity`, optional
        Frequencies for each channel in ``ih`` (channelized frequencies will
        be calculated).  Default: taken from ``ih`` (if available).
    sideband : array, optional
        Whether frequencies in ``ih`` are upper (+1) or lower (-1) sideband.
        Default: taken from ``ih`` (if available).

    See Also
    --------
    baseband_tasks.fourier.fft_maker : to select the FFT package used.
    baseband_tasks.dispersion.Dedisperse : for coherent dedispersion
    baseband_tasks.dispersion.DisperseSamples : for incoherent dispersion
    """

    def __init__(self, ih, dm, *, reference_frequency=None,
                 samples_per_frame=None, frequency=None, sideband=None):
        dm = DispersionMeasure(dm)
        frequency = getattr_if_none(ih, 'frequency', frequency)
        sideband = getattr_if_none(ih, 'sideband', sideband)

        # Calculate frequencies at the top and bottom of each band.
        half_rate = ih.sample_rate / 2.
        if ih.complex_data:
            freq_low = frequency - half_rate
            freq_high = frequency + half_rate
        else:
            freq_low = frequency + np.minimum(sideband, 0.) * half_rate
            freq_high = frequency + np.maximum(sideband, 0.) * half_rate

        if reference_frequency is None:
            reference_frequency = (freq_low + freq_high).mean() / 2.

        # Calculate the maximum positive and negative delays that will
        # be corrected for.
        delay_low = dm.time_delay(freq_low, reference_frequency)
        delay_high = dm.time_delay(freq_high, reference_frequency)
        delay_max = max(delay_low.max(), delay_high.max())
        delay_min = min(delay_low.min(), delay_high.min())
        # Calculate the padding needed to avoid wrapping in what we extract.
        pad_start = int(np.ceil((delay_max * ih.sample_rate).to_value(u.one)))
        pad_end = int(np.ceil((-delay_min * ih.sample_rate).to_value(u.one)))
        # Generally, the padding will be on both sides.  If either is negative,
        # that indicates that the reference frequency is outside of the band,
        # and we can do part of the work with a simple sample shift.
        if pad_start < 0:
            # Both delays less than 0; do not need start, so shift by
            # that number of samples, reducing the padding at the end.
            assert pad_end > 0
            sample_offset = pad_start
            pad_end += pad_start
            pad_start = 0
        elif pad_end < 0:
            # Both delays greater than 0; do not need end, so shift by
            # that number of samples, reducing the padding at the start.
            sample_offset = -pad_end
            pad_start += pad_end
            pad_end = 0
        else:
            # Default case: passing on both sides; not useful to offset.
            sample_offset = 0

        start_time = ih.start_time + sample_offset / ih.sample_rate
        super().__init__(ih, pad_start=pad_start, pad_end=pad_end,
                         samples_per_frame=samples_per_frame,
                         frequency=frequency, sideband=sideband,
                         start_time=start_time)

        # Initialize FFTs for fine channelization and the inverse.
        # TODO: remove duplication with Convolve.
        self._fft = fft_maker(shape=(self._ih_samples_per_frame,)
                              + self.ih.sample_shape, dtype=self.ih.dtype,
                              sample_rate=self.ih.sample_rate)
        self._ifft = self._fft.inverse()
        self._dm = dm
        self.reference_frequency = reference_frequency
        self._sample_offset = sample_offset
        self._pad_slice = slice(self._pad_start,
                                self._pad_start + self.samples_per_frame)

    @lazyproperty
    def phase_factor(self):
        """Phase offsets of the Fourier-transformed frame."""
        frequency = self.frequency + self._fft.frequency * self.sideband
        phase_delay = self._dm.phase_delay(frequency, self.reference_frequency)
        phase_delay *= self.sideband
        # Correct for any time offset applied because the reference frequency
        # was out of range.
        if self._sample_offset != 0:
            phase_delay += (self._sample_offset / self.sample_rate * u.cycle
                            * self._fft.frequency)
        phase_factor = np.exp(phase_delay.to_value(u.rad) * 1j)
        phase_factor = phase_factor.astype(self._fft.frequency_dtype,
                                           copy=False)
        return phase_factor

    @property
    def dm(self):
        return self._dm

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


class Dedisperse(Disperse):
    """Coherently dedisperse a time stream.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream, with time as the first axis.
    dm : float or `~baseband_tasks.dm.DispersionMeasure` quantity
        Dispersion measure.  If negative, will disperse correctly, but
        clearer to use the `~baseband_tasks.dispersion.Disperse` class.
    reference_frequency : `~astropy.units.Quantity`
        Frequency to which the data should be dedispersed.  Can be an array.
        By default, the mean frequency.  If one doesn't want to change the
        start time, choose the maximum frequency.
    samples_per_frame : int, optional
        Number of dedispersed samples which should be produced in one go.
        The number of input samples used will be larger to avoid wrapping.
        If not given, as produced by the minimum power of 2 of input
        samples that yields at least 75% efficiency.
    frequency : `~astropy.units.Quantity`, optional
        Frequencies for each channel in ``ih`` (channelized frequencies will
        be calculated).  Default: taken from ``ih`` (if available).
    sideband : array, optional
        Whether frequencies in ``ih`` are upper (+1) or lower (-1) sideband.
        Default: taken from ``ih`` (if available).

    See Also
    --------
    baseband_tasks.fourier.fft_maker : to select the FFT package used.
    baseband_tasks.dispersion.Disperse : for coherent dispersion
    baseband_tasks.dispersion.DedisperseSamples : for incoherent dedispersion
    """

    def __init__(self, ih, dm, *, reference_frequency=None,
                 samples_per_frame=None, frequency=None, sideband=None):
        super().__init__(ih, -dm, reference_frequency=reference_frequency,
                         samples_per_frame=samples_per_frame,
                         frequency=frequency, sideband=sideband)

    @property
    def dm(self):
        return -self._dm


class DisperseSamples(ShiftSamples):
    """Incoherently shift a time stream to give it a dispersive time delay.

    This task does not handle any in-channel dispersive smearing, but only
    shifts the samples according to the mid-channel frequency.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream, with time as the first axis.
    dm : float or `~baseband_tasks.dm.DispersionMeasure` quantity
        Dispersion measure to disperse with.  If negative, will dedisperse,
        but clearer to use `~baseband_tasks.dispersion.DedisperseSamples`.
    reference_frequency : `~astropy.units.Quantity`
        Frequency to which the data should be dispersed.  Can be an array.
        By default, the mean frequency.
    samples_per_frame : int, optional
        Number of dispersed samples which should be produced in one go.
        The number of input samples used will be larger to avoid wrapping.
        If not given, as produced by the minimum power of 2 of input
        samples that yields at least 75% efficiency.
    frequency : `~astropy.units.Quantity`, optional
        Frequencies for each channel in ``ih``.
        Default: taken from ``ih`` (if available).
    sideband : array, optional
        Whether frequencies in ``ih`` are upper (+1) or lower (-1) sideband.
        Default: taken from ``ih`` (if available). Note that while this is
        only used if the data is real (to calculate the mid-channel
        frequency), it should always be passed in together with ``frequency``,
        since otherwise other tasks cannot interpret frequency correctly.

    See Also
    --------
    baseband_tasks.dispersion.DedisperseSamples : for incoherent dedispersion
    baseband_tasks.dispersion.Disperse : for coherent dispersion
    """

    def __init__(self, ih, dm, *, reference_frequency=None,
                 samples_per_frame=None, frequency=None, sideband=None):
        # Set possible missing frequency/sideband attributes
        if frequency is not None or sideband is not None:
            ih = SetAttribute(ih, frequency=frequency, sideband=sideband)
        frequency = ih.frequency
        if not ih.complex_data:
            # Calculate mid-channel frequency for real data (for complex,
            # the frequencies are already mid-channel).
            frequency = frequency + ih.sideband * ih.sample_rate / 2.

        if reference_frequency is None:
            reference_frequency = frequency.mean()

        # Compute the time shift and use it to set up ShiftSamples.
        dm = DispersionMeasure(dm)
        time_delay = dm.time_delay(frequency, reference_frequency)
        super().__init__(ih, time_delay, samples_per_frame=samples_per_frame)

        self.reference_frequency = reference_frequency
        self._dm = dm


class DedisperseSamples(DisperseSamples):
    """Incoherently shift a time stream to correct for a dispersive time delay.

    This task does not handle any in-channel dispersive smearing, but only
    shifts the samples according to the mid-channel frequency.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream, with time as the first axis.
    dm : float or `~baseband_tasks.dm.DispersionMeasure` quantity
        Dispersion measure to correct for.  If negative, will disperse,
        but clearer to use `~baseband_tasks.dispersion.DisperseSamples`.
    reference_frequency : `~astropy.units.Quantity`
        Frequency to which the data should be dispersed.  Can be an array.
        By default, the mean frequency.
    samples_per_frame : int, optional
        Number of dedispersed samples which should be produced in one go.
        The number of input samples used will be larger to avoid wrapping.
        If not given, as produced by the minimum power of 2 of input
        samples that yields at least 75% efficiency.
    frequency : `~astropy.units.Quantity`, optional
        Frequencies for each channel in ``ih``.
        Default: taken from ``ih`` (if available).
    sideband : array, optional
        Whether frequencies in ``ih`` are upper (+1) or lower (-1) sideband.
        Default: taken from ``ih`` (if available). Note that while this is
        only used if the data is real (to calculate the mid-channel
        frequency), it should always be passed in together with ``frequency``,
        since otherwise other tasks cannot interpret frequency correctly.

    See Also
    --------
    baseband_tasks.dispersion.DisperseSamples : for incoherent dispersion
    baseband_tasks.dispersion.Dedisperse : for coherent dedispersion
    """

    def __init__(self, ih, dm, *, reference_frequency=None,
                 samples_per_frame=None, frequency=None, sideband=None):
        super().__init__(ih, -dm, reference_frequency=reference_frequency,
                         samples_per_frame=samples_per_frame,
                         frequency=frequency, sideband=sideband)

    @property
    def dm(self):
        return -self._dm
