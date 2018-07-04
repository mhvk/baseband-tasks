# Licensed under the GPLv3 - see LICENSE

import pyfftw
from .base import FFTMakerBase, FFT


class PyfftwFFTMaker(FFTMakerBase):
    """FFT factory class utilizing the `pyfftw` package.

    Analagous to `~numpy.fft.rfft`, FFTs of real-valued time-domain data
    perform a real-input transform on one dimension of the input, halving that
    dimension's length in the output.

    Currently does not support Hermitian FFTs.

    Parameters
    ----------
    n_simd : int or None, optional
        Single Instruction Multiple Data (SIMD) alignment in bytes.  If `None`,
        uses `pyfftw.simd_alignment`, which is found by inspecting the CPU.
    **kwargs
        Optional keywords to `pyfftw.FFTW` class, including planning flags, the
        number of threads to be used, and the planning time limit.
    """

    _engine_name = 'pyfftw'

    def __init__(self, n_simd=None, **kwargs):
        # Set n-byte boundary.
        if n_simd is None:
            n_simd = pyfftw.simd_alignment
        self._n_simd = n_simd

        if 'flags' in kwargs and 'FFTW_DESTROY_INPUT' in kwargs['flags']:
            raise ValueError('Fourier module does not support destroying '
                             'input arrays.')

        self._kwargs = kwargs

        super().__init__()

    def _setup_transform(self, data_format, direction, axis, norm,
                         sample_rate):

        # Create dummy byte-aligned arrays.  These will be stored as
        # fft_engine.input_array, fft_engine.output_array, etc., but we'll be
        # replacing the input and output arrays each time we use a transform.
        a = pyfftw.empty_aligned(data_format['time_shape'],
                                 dtype=data_format['time_dtype'],
                                 n=self._n_simd)
        A = pyfftw.empty_aligned(data_format['freq_shape'],
                                 dtype=data_format['freq_dtype'],
                                 n=self._n_simd)

        # Set up normalization keywords.
        if norm == 'ortho':
            normalise_idft = False
            ortho = True
        else:
            normalise_idft = True
            ortho = False

        # Create either forward or inverse transform function.
        if direction == 'forward':
            fft_engine = pyfftw.FFTW(a, A, axes=(axis,),
                                     direction='FFTW_FORWARD', **self._kwargs)

            def fft(a):
                # Make an empty array to store transform output.
                A = pyfftw.empty_aligned(data_format['freq_shape'],
                                         dtype=data_format['freq_dtype'],
                                         n=self._n_simd)
                # A is returned by self._fft.
                return fft_engine(input_array=a, output_array=A,
                                  normalise_idft=normalise_idft,
                                  ortho=ortho)

        else:
            fft_engine = pyfftw.FFTW(A, a, axes=(axis,),
                                     direction='FFTW_BACKWARD', **self._kwargs)

            # Note that only multi-dimensional real transforms destroy their
            # input arrays.  See https://pyfftw.readthedocs.io/en/latest/source/pyfftw/pyfftw.html#scheme-table
            def fft(self, A):

                # Make an empty array to store transform output.
                a = pyfftw.empty_aligned(data_format['time_shape'],
                                         dtype=data_format['time_dtype'],
                                         n=self._n_simd)
                # a is returned by self._fft.
                return fft_engine(input_array=A, output_array=a,
                                  normalise_idft=normalise_idft,
                                  ortho=ortho)

        return FFT(fft, data_format, direction, axis, norm, sample_rate, self)
