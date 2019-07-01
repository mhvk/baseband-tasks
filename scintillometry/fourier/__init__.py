# Licensed under the GPLv3 - see LICENSE
"""Fourier transform module.

This module provides a standard interface to various FFT packages.

Routine listings
----------------

`fft_maker` : primary interface for creating FFT instances.

Implementation Notes
--------------------

For each packages, there is a corresponding ``*FFTMaker`` class, which
holds default information needed for creating an FFT instance. For
instance, for `PyfftwFFTMaker`, this holds ``flags``, ``threads``, etc.

These ``*FFTMaker`` instances in turn can be used to create ``*FFT``
instances which are set up to do the FFT on data with a given shape, in
a given direction, etc.
"""
from .base import fft_maker
from .numpy import NumpyFFTMaker

# If pyfftw is available, import PyfftwFFTMaker.
try:
    from .pyfftw import PyfftwFFTMaker
    from os import environ
    fft_maker._system_default = PyfftwFFTMaker(
        flags=['FFTW_ESTIMATE', 'FFTW_DESTROY_INPUT'],
        threads=int(environ.get('OMP_NUM_THREADS', 2)))
    del environ
except ImportError:
    fft_maker._system_default = NumpyFFTMaker()
