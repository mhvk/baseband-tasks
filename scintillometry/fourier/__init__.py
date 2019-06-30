# Licensed under the GPLv3 - see LICENSE
"""Fourier transform module."""
from .base import fft_maker
from .numpy import NumpyFFTMaker

# If pyfftw is available, import PyfftwFFTMaker.
try:
    from .pyfftw import PyfftwFFTMaker
    from os import environ
    fft_maker.system_default = PyfftwFFTMaker(
        flags=['FFTW_ESTIMATE', 'FFTW_DESTROY_INPUT'],
        threads=int(environ.get('OMP_NUM_THREADS', 2)))
    del environ
except ImportError:
    fft_maker.system_default = NumpyFFTMaker()
