# Licensed under the GPLv3 - see LICENSE
"""Fourier transform module."""

from .base import get_fft_maker
from .numpy import NumpyFFTMaker

# If pyfftw is available, import PyfftwFFTMaker.
try:
    from .pyfftw import PyfftwFFTMaker
except ImportError:
    pass
