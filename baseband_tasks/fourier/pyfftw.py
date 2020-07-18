# Licensed under the GPLv3 - see LICENSE
"""FFT maker and class using `pyfftw` routines.

Implementation Notes
--------------------

The code for `PyfftwFFTBase` is relatively complex to ensure that the
input and output arrays are re-used, even between the forward and
backward transforms (if created using :meth:`PyfftwFFTBase.inverse`)

"""

import pyfftw

from .base import FFTMakerBase, FFTBase


__all__ = ['PyfftwFFTBase', 'PyfftwFFTMaker']


class PyfftwFFTBase(FFTBase):
    """Single pre-defined FFT based on `pyfftw.FFTW`.

    To use, initialize an instance, then call the instance to perform
    the transform.

    Parameters
    ----------
    direction : 'forward' or 'backward', optional
        Direction of the FFT.
    """

    _fftw = None
    _inverse = None

    def _fft(self, a):
        if self._fftw is None:
            a = pyfftw.byte_align(a, n=self._n_simd)
            self._setup_fftw(a)

        # Save a bit of useless checking in FFTW if possible.
        if a is self._fftw.input_array:
            a = None
        if self._inverse is None:
            b = None
        else:
            b = self._inverse._fftw.input_array
            if b is self._fftw.output_array:
                b = None
        return self._fftw(a, b)

    def inverse(self):
        inverse = super().inverse()
        inverse._inverse = self  # Note: _fftw doesn't necessarily exist yet.
        return inverse

    def _setup_fftw(self, a, b=None):
        # Setup FFTW, creating its byte-aligned input_array and output_array.
        # We do this on the first call so that we can use the strides of an
        # actual input array.  For any further calls, the inputs will then
        # simply replace input_array instead of being copied (at least, if
        # the byte alignment is correct).
        if self._inverse is not None and self._inverse._fftw is not None:
            a = self._inverse._fftw.output_array
            b = self._inverse._fftw.input_array

        else:
            if self.direction == 'forward':
                assert a.shape == self._time_shape
                assert a.dtype == self._time_dtype
                if b is None:
                    b = pyfftw.empty_aligned(self._frequency_shape,
                                             self._frequency_dtype,
                                             n=self._n_simd)

            else:
                assert a.shape == self._frequency_shape
                assert a.dtype == self._frequency_dtype
                if b is None:
                    b = pyfftw.empty_aligned(self._time_shape,
                                             self._time_dtype,
                                             n=self._n_simd)

        direction = 'FFTW_{}'.format(self.direction.upper())
        self._fftw = pyfftw.FFTW(a, b, axes=(self.axis,),
                                 direction=direction,
                                 normalise_idft=self._normalise_idft,
                                 ortho=self._ortho,
                                 **self._fftw_kwargs)
        # Set up original with same arrays if it wasn't set up before us,
        # so that self._inverse._fftw is guaranteed to exist in _fft.
        if self._inverse is not None and self._inverse._fftw is None:
            self._inverse._setup_fftw(b, a)


class PyfftwFFTMaker(FFTMakerBase):
    """FFT factory class utilizing the `pyfftw` package.

    Analagous to `~numpy.fft.rfft`, FFTs of real-valued time-domain data
    perform a real-input transform on one dimension of the input, halving that
    dimension's length in the output.

    ``__init__`` is used to set package-level options, such as ``n_simd``,
    while `~baseband_tasks.fourier.pyfftw.PyfftwFFTMaker.__call__` creates
    individual transforms.

    Parameters
    ----------
    n_simd : int or None, optional
      Single Instruction Multiple Data (SIMD) alignment in bytes.  If `None`,
      uses ``pyfftw.simd_alignment``, which is found by inspecting the CPU.
    **kwargs
      Optional keywords to `pyfftw.FFTW` class, including planning flags, the
      number of threads to be used, and the planning time limit.
    """
    _FFTBase = PyfftwFFTBase

    def __init__(self, n_simd=None, **kwargs):
        self._n_simd = pyfftw.simd_alignment if n_simd is None else n_simd
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
        direction : 'forward' or 'backward', optional
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
        return super().__call__(
            shape=shape, dtype=dtype, direction=direction,
            axis=axis, ortho=ortho, sample_rate=sample_rate,
            normalise_idft=(False if ortho else True),
            n_simd=self._n_simd, fftw_kwargs=self._fftw_kwargs)

    def __repr__(self):
        self._repr_kwargs = dict(n_simd=self._n_simd)
        self._repr_kwargs.update(self._fftw_kwargs)
        return super().__repr__()
