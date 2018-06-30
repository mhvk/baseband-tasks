# Licensed under the GPLv3 - see LICENSE
"""Fourier transform linkage module."""

from .base import FFTBase, NumpyFFT

# If pyfftw is available, import PyfftwFFT.
try:
    from .pyfftw import PyfftwFFT
except ImportError:
    pass
