# Licensed under the GPLv3 - see LICENSE

import operator
from .base import FFTMakerBase, FFTBase

__all__ = ['PyfftwFFTMaker']

# Only instantiate the class if FFTW exists.
try:
    import pyfftw
except ImportError:
    pass
else:

    class PyfftwFFTMaker(FFTMakerBase):
        """FFT factory class utilizing the `pyfftw` package.

        Analagous to `~numpy.fft.rfft`, FFTs of real-valued time-domain data
        perform a real-input transform on one dimension of the input, halving
        that dimension's length in the output.

        Currently does not support Hermitian FFTs.

        Parameters
        ----------
        n_simd : int or None, optional
          Single Instruction Multiple Data (SIMD) alignment in bytes.  If
          `None`, uses `pyfftw.simd_alignment`, which is found by inspecting
          the CPU.
        **kwargs
          Optional keywords to `pyfftw.FFTW` class, including planning flags,
          the number of threads to be used, and the planning time limit.
        """

        def __init__(self, n_simd=None, **kwargs):
            # Set n-byte boundary.
            if n_simd is None:
                n_simd = pyfftw.simd_alignment
            self._n_simd = n_simd

            if 'flags' in kwargs and 'FFTW_DESTROY_INPUT' in kwargs['flags']:
                raise ValueError('Fourier module does not support destroying '
                                 'input arrays.')

            self._fftw_kwargs = kwargs

            super().__init__()

        def __call__(self, time_data=None, freq_data=None, axis=0,
                     ortho=False, sample_rate=None):
            """Set up FFT.

            Parameters
            ----------
            time_data : `~numpy.ndarray` or dict
                Dummy array with dimensions and dtype of time-domain data.  Can
                alternatively give a dict with 'shape' and 'dtype' entries.
            freq_dtype : str
                dtype of frequency-domain data.
            axis : int, optional
                Axis to transform.  Default: 0.
            ortho : bool, optional
                Whether to use orthogonal normalization.  Default: `False`.
            sample_rate : float or `~astropy.units.Quantity`
                Sample rate.
            """
            # Set direction, axis and normalization.  If axis is None, set it
            # to 0.
            axis = operator.index(axis)
            ortho = bool(ortho)

            # Store time and frequency-domain array shapes.
            data_format = self.get_data_format(time_data=time_data,
                                               freq_data=freq_data, axis=axis)

            PyfftwFFT = self._setup_fft_class(data_format, axis, ortho,
                                              sample_rate)

            return PyfftwFFT

        def _setup_fft_class(self, data_format, axis, ortho, sample_rate):
            # Set up normalization keywords.
            if ortho:
                _normalise_idft = False
            else:
                _normalise_idft = True

            def _forward_fft(self, a):
                # Make an empty array to store transform output.
                A = pyfftw.empty_aligned(self.data_format['freq_shape'],
                                         dtype=self.data_format['freq_dtype'],
                                         n=self._n_simd)
                # A is returned by self._fft.
                return self._FFTW(input_array=a, output_array=A,
                                  normalise_idft=self._normalise_idft,
                                  ortho=self.ortho)

            # Note that only multi-dimensional real transforms destroy their
            # input arrays.  See
            # https://pyfftw.readthedocs.io/en/latest/source/pyfftw/pyfftw.html#scheme-table
            def _inverse_fft(self, A):

                # Make an empty array to store transform output.
                a = pyfftw.empty_aligned(self.data_format['time_shape'],
                                         dtype=self.data_format['time_dtype'],
                                         n=self._n_simd)
                # a is returned by self._fft.
                return self._FFTW(input_array=A, output_array=a,
                                  normalise_idft=self._normalise_idft,
                                  ortho=self.ortho)

            def constructor(self, direction='forward'):
                super(PyfftwFFT, self).__init__(direction=direction)
                # Create dummy byte-aligned arrays.  These will be stored
                # in the FFTW instance as input_array and output_array, but
                # we'll be replacing those each time we perform a transform.
                a = pyfftw.empty_aligned(self.data_format['time_shape'],
                                         dtype=self.data_format['time_dtype'],
                                         n=self._n_simd)
                A = pyfftw.empty_aligned(self.data_format['freq_shape'],
                                         dtype=self.data_format['freq_dtype'],
                                         n=self._n_simd)

                if self.direction == 'forward':
                    self._FFTW = pyfftw.FFTW(a, A, axes=(self.axis,),
                                             direction='FFTW_FORWARD',
                                             **self._fftw_kwargs)
                    self._fft = self._forward_fft
                else:
                    self._FFTW = pyfftw.FFTW(A, a, axes=(self.axis,),
                                             direction='FFTW_BACKWARD',
                                             **self._fftw_kwargs)
                    self._fft = self._inverse_fft

            def equality(self, other):
                # Assumes that class names are unique, which should be the case
                # unless the user improperly initializes the class factory.
                return (super(PyfftwFFT, self).__eq__(other) and
                        self._fftw_kwargs == other._fftw_kwargs)

            PyfftwFFT = type("PyfftwFFT", (FFTBase,), {
                "_data_format": data_format,
                "_axis": axis,
                "_ortho": ortho,
                "_normalise_idft": _normalise_idft,
                "_sample_rate": sample_rate,
                "_n_simd": self._n_simd,
                "_fftw_kwargs": self._fftw_kwargs,
                "__init__": constructor,
                "__eq__": equality,
                "_forward_fft": _forward_fft,
                "_inverse_fft": _inverse_fft
            })

            return PyfftwFFT
