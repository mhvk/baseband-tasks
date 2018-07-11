# Licensed under the GPLv3 - see LICENSE
"""Fourier transform module."""

from .base import NumpyFFTMaker, get_fft_maker

# If pyfftw is available, import PyfftwFFTMaker.
try:
    from .pyfftw import PyfftwFFTMaker
except ImportError:
    pass