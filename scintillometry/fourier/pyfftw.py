# Licensed under the GPLv3 - see LICENSE

import operator
import numpy as np
import pyfftw
from .base import FFTMakerBase, FFTBase


__all__ = ['PyfftwFFTMaker']


class PyfftwFFTMaker(FFTMakerBase):
    """FFT factory class utilizing the `pyfftw` package.

    Analagous to `~numpy.fft.rfft`, FFTs of real-valued time-domain data
    perform a real-input transform on one dimension of the input, halving that
    dimension's length in the output.

    ``__init__`` is used to set package-level options, such as ``n_simd``,
    while `~scintillometry.fourier.pyfftw.PyfftwFFTMaker.__call__` creates
    individual transforms.

    Parameters
    ----------
    n_simd : int or None, optional
      Single Instruction Multiple Data (SIMD) alignment in bytes.  If `None`,
      uses `pyfftw.simd_alignment`, which is found by inspecting the CPU.
    **kwargs
      Optional keywords to `pyfftw.FFTW` class, including planning flags, the
      number of threads to be used, and the planning time limit.
    """

    def __init__(self, n_simd=None, **kwargs):
        self._n_simd = pyfftw.simd_alignment if n_simd is None else n_simd

        if 'flags' in kwargs and 'FFTW_DESTROY_INPUT' in kwargs['flags']:
            raise ValueError('Fourier module does not support destroying '
                             'input arrays.')
        self._fftw_kwargs = kwargs

        super().__init__()

    def __call__(self, shape, dtype, direction='forward', axis=0, ortho=False,
                 sample_rate=None):
        """Creates an FFT.

        Parameters
        ----------
        shape : tuple
            Shape of the time-domain data array, i.e. the input to the forward
            transform and the output of the inverse.
        dtype : str or `~numpy.dtype`
            Data type of the time-domain data array.  May pass either the
            name of the dtype or the `~numpy.dtype` object.
        direction : 'forward' or 'inverse', optional
            Direction of the FFT.
        axis : int, optional
            Axis to transform.  Default: 0.
        ortho : bool, optional
            Whether to use orthogonal normalization.  Default: `False`.
        sample_rate : float, `~astropy.units.Quantity`, or None, optional
            Sample rate, used to determine the FFT sample frequencies.  If
            `None`, a unitless rate of 1 is used.

        Returns
        -------
        fft : ``PyfftwFFT`` instance
            Single pre-defined FFT object.
        """
        # Ensure arguments have proper types and values.
        shape = tuple(shape)
        dtype = np.dtype(dtype)
        axis = operator.index(axis)
        ortho = bool(ortho)

        # Store time and frequency-domain array shapes.
        frequency_shape, frequency_dtype = self.get_frequency_data_info(
            shape, dtype, axis=axis)

        # Declare PyfftwFFT class, and populate values.
        class PyfftwFFT(FFTBase):
            """Single pre-defined FFT based on `pyfftw.FFTW`."""

            _time_shape = shape
            _time_dtype = dtype
            _frequency_shape = frequency_shape
            _frequency_dtype = frequency_dtype
            _axis = axis
            _ortho = ortho
            _normalise_idft = False if ortho else True
            _sample_rate = sample_rate
            _n_simd = self._n_simd
            _fftw_kwargs = self._fftw_kwargs

            def __init__(self, direction='forward'):
                super().__init__(direction=direction)
                # Create dummy byte-aligned arrays.  These will be stored in
                # the FFTW instance as input_array and output_array, but we'll
                # be replacing those each time we transform.
                a = pyfftw.empty_aligned(self._time_shape, self._time_dtype,
                                         n=self._n_simd)
                A = pyfftw.empty_aligned(self._frequency_shape,
                                         self._frequency_dtype,
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

            def _forward_fft(self, a):
                # Make an empty array to store transform output.
                A = pyfftw.empty_aligned(self._frequency_shape,
                                         self._frequency_dtype,
                                         n=self._n_simd)
                return self._FFTW(input_array=a, output_array=A,
                                  normalise_idft=self._normalise_idft,
                                  ortho=self.ortho)

            # Note that only multi-dimensional real transforms destroy
            # their input arrays.  See
            # https://pyfftw.readthedocs.io/en/latest/source/pyfftw/pyfftw.html#scheme-table
            def _inverse_fft(self, A):
                # Make an empty array to store transform output.
                a = pyfftw.empty_aligned(self._time_shape, self._time_dtype,
                                         n=self._n_simd)
                return self._FFTW(input_array=A, output_array=a,
                                  normalise_idft=self._normalise_idft,
                                  ortho=self.ortho)

            def __eq__(self, other):
                base_eq = super().__eq__(other)
                if hasattr(other, '_fftw_kwargs'):
                    return base_eq and (self._fftw_kwargs ==
                                        other._fftw_kwargs)
                else:
                    return base_eq

        # Return PyfftwFFT instance.
        return PyfftwFFT(direction=direction)
