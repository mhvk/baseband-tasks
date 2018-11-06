# Licensed under the GPLv3 - see LICENSE

import operator
import warnings

import numpy as np
import astropy.units as u
from astropy.utils import lazyproperty

from .base import TaskBase
from .fourier import get_fft_maker
from .dm import DispersionMeasure


__all__ = ['Disperse', 'Dedisperse']


class Disperse(TaskBase):
    """Coherently disperse a time stream.

    Dispersion is always to the maximum frequency in the underlying
    time stream (such that the stop time does not change).

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream, with time as the first axis.
    dm : float or `~scintillometry.dm.DispersionMeasure` quantity
        Dispersion measure.  If negative, will dedisperse correctly, but
        clearer to use the `~scintillometry.dispersion.Dedisperse` class.
    reference_frequency : `~astropy.units.Quantity`, optional
        Frequency to which the data should be dispersed.  Can be an array.
        By default, the mean frequency.
    samples_per_frame : int, optional
        Number of samples which should be dispersed in one go. The number of
        output dispersed samples per frame will be smaller to avoid wrapping.
        If not given, the minimum power of 2 needed to get at least 75%
        efficiency.
    frequency : `~astropy.units.Quantity`, optional
        Frequencies for each channel in ``ih`` (channelized frequencies will
        be calculated).  Default: taken from ``ih`` (if available).
    sideband : array, optional
        Whether frequencies in ``ih`` are upper (+1) or lower (-1) sideband.
        Default: taken from ``ih`` (if available).
    FFT : FFT maker or None, optional
        FFT maker.  Default: `None`, in which case the channelizer uses
        `~scintillometry.fourier.numpy.NumpyFFTMaker`.
    """

    def __init__(self, ih, dm, reference_frequency=None,
                 samples_per_frame=None, frequency=None, sideband=None,
                 FFT=None):
        dm = DispersionMeasure(dm)
        if frequency is None:
            frequency = ih.frequency
        if sideband is None:
            sideband = ih.sideband

        half_rate = ih.sample_rate / 2.
        if ih.complex_data:
            freq_low = frequency - half_rate
            freq_high = frequency + half_rate
        else:
            freq_low = frequency + np.minimum(sideband, 0.) * half_rate
            freq_high = frequency + np.maximum(sideband, 0.) * half_rate

        if reference_frequency is None:
            reference_frequency = (freq_low + freq_high).mean() / 2.

        # NOTE: don't do this, have to broadcast reference_frequency
        delay_low = dm.time_delay(freq_low, reference_frequency)
        delay_high = dm.time_delay(freq_high, reference_frequency)
        delay_max = max(delay_low.max(), delay_high.max())
        delay_min = min(delay_low.min(), delay_high.min())
        if delay_max < 0.:
            # both delays less than 0 -> shift in bulk.
            assert delay_min < 0.
            time_offset = delay_max
            delay_min -= delay_max
            delay_max = 0. * u.s
        elif delay_min > 0.:
            # both delays greater than 0 -> shift in bulk.
            time_offset = delay_min
            delay_max -= delay_min
            delay_min = 0. * u.s
        else:
            # default case: a bit on each side.
            time_offset = 0. * u.s

        pad_start = int(np.ceil((delay_max * ih.sample_rate).to_value(u.one)))
        pad_end = int(np.ceil((-delay_min * ih.sample_rate).to_value(u.one)))
        pad = pad_start + pad_end
        if samples_per_frame is None:
            # 4 times power of two just above pad.
            samples_per_frame = 2 ** (int((np.ceil(np.log2(pad)))) + 2)
        elif pad >= samples_per_frame:
            raise ValueError("need more than {} samples per frame to be "
                             "able to dedisperse without wrapping."
                             .format(pad))
        elif pad > samples_per_frame / 2.:
            warnings.warn("Dedispersion will be inefficient since of the "
                          "{} samples per frame given, {} will be lost due "
                          "to padding.".format(samples_per_frame, pad))

        # Initialize FFT.
        if FFT is None:
            FFT = get_fft_maker('numpy')
        # Fine channelization FFT and inverse.
        self._fft = FFT(shape=(samples_per_frame,) + ih.sample_shape,
                        dtype=ih.dtype, sample_rate=ih.sample_rate)
        self._ifft = self._fft.inverse()
        # Subtract padding since that is what we actually produce per frame.
        samples_per_frame -= pad
        n_frames = (ih.shape[0] - pad) // samples_per_frame
        super().__init__(ih, samples_per_frame=samples_per_frame,
                         shape=(n_frames * samples_per_frame,) + ih.shape[1:],
                         frequency=frequency, sideband=sideband)
        self.dm = dm
        self.reference_frequency = reference_frequency
        self._pad_start = pad_start
        self._pad_end = pad_end
        self._start_time += time_offset + pad_start / ih.sample_rate
        self._time_offset = time_offset

    @lazyproperty
    def phase_factor(self):
        """Phase offsets of the Fourier-transformed frame."""
        frequency = self.frequency + self._fft.frequency * self.sideband
        phase_factor = self.dm.phase_factor(frequency, self.reference_frequency)
        # Correct for time offsets applied because the reference
        # frequency was out of range.
        if self._time_offset != 0.:
            phase_factor *= np.exp((frequency * self._time_offset * u.cycle)
                                   .to_value(u.rad) * 1j)
        phase_factor = phase_factor.astype(self._fft.frequency_dtype,
                                           copy=False)
        np.conjugate(phase_factor, where=self.sideband < 0, out=phase_factor)
        return phase_factor

    def task(self, data):
        ft = self._fft(data)
        ft *= self.phase_factor
        return self._ifft(ft)[self._pad_start:data.shape[0]-self._pad_end]

    # Need to override _read_frame from TaskBase to include the padding.
    def _read_frame(self, frame_index):
        # Read data from underlying filehandle.
        self.ih.seek(frame_index * self.samples_per_frame)
        data = self.ih.read(self.samples_per_frame +
                            self._pad_start + self._pad_end)
        return self.task(data)

    def close(self):
        super().close()
        # Clear the cache of the lazyproperty to release memory.
        del self.phase_factor


class Dedisperse(Disperse):
    """Coherently dedisperse a time stream.

    Dedispersion is always to the maximum frequency in the underlying
    time stream (such that the start time does not change).

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream, with time as the first axis.
    dm : float or `~scintillometry.dm.DispersionMeasure` quantity
        Dispersion measure.  If negative, will disperse correctly, but
        clearer to use the `~scintillometry.dispersion.Disperse` class.
    reference_frequency : `~astropy.units.Quantity`
        Frequency to which the data should be dedispersed.  Can be an array.
        By default, the mean frequency.
    samples_per_frame : int, optional
        Number of samples which should be dedispersed in one go. The number of
        output dedispersed samples per frame will be smaller to avoid wrapping.
        If not given, the minimum power of 2 needed to get at least 75%
        efficiency.
    frequency : `~astropy.units.Quantity`, optional
        Frequencies for each channel in ``ih`` (channelized frequencies will
        be calculated).  Default: taken from ``ih`` (if available).
    sideband : array, optional
        Whether frequencies in ``ih`` are upper (+1) or lower (-1) sideband.
        Default: taken from ``ih`` (if available).
    FFT : FFT maker or None, optional
        FFT maker.  Default: `None`, in which case the channelizer uses
        `~scintillometry.fourier.numpy.NumpyFFTMaker`.
    """

    def __init__(self, ih, dm, reference_frequency=None,
                 samples_per_frame=None, frequency=None, sideband=None,
                 FFT=None):
        super().__init__(ih, -dm, reference_frequency, samples_per_frame,
                         frequency, sideband, FFT)
