# Licensed under the GPLv3 - see LICENSE
"""Fourier transform module."""

from .base import NumpyFFTMaker

FFT_MAKER_CLASSES = (NumpyFFTMaker,)

# If pyfftw is available, import PyfftwFFTMaker.
try:
    from .pyfftw import PyfftwFFTMaker
except ImportError:
    pass
else:
    FFT_MAKER_CLASSES += (PyfftwFFTMaker,)

FFT_MAKER_CLASSES = {fftclass._engine_name: fftclass
                     for fftclass in FFT_MAKER_CLASSES}

from .get_fft import get_fft
