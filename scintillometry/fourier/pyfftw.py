# Licensed under the GPLv3 - see LICENSE

import pyfftw
from .base import FFTBase, _check_fft_exists


class PyfftwFFT(FFTBase):
    """FFT class that wraps `pyfftw` classes.

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

    def setup(self, time_data, freq_dtype, axes=None, norm=None,
              verify=True):
        """Set up FFT (including FFTW planning).

        Parameters
        ----------
        time_data : `~numpy.ndarray` or dict
            Dummy array with dimensions and dtype of time-domain data.  Can
            alternatively give a dict with 'shape' and 'dtype' entries.
        freq_dtype : str
            dtype of frequency-domain data.
        axes : int, tuple, or None, optional
            Axis or axes to transform.  If an int is passed, it is turned into
            a tuple.  If `None` (default), all axes are used.  For real-valued
            time-domain data, the real-input transform is performed on
            ``axes[-1]``.
        norm : 'ortho' or None, optional
            If `None` (default), uses an unscaled forward transform and 1 / n
            scaled inverse transform.  If 'ortho', uses a 1 / sqrt(n) scaling
            for both.
        verify : bool, optional
            Verify setup is successful and self-consistent.
        """
        # Store info about arrays, axes, normalization.
        super().setup(time_data, freq_dtype, axes=axes, norm=norm,
                      verify=verify)

        # Set up normalization keywords.
        if self.norm == 'ortho':
            self._normalise_idft = False
            self._ortho = True
        else:
            self._normalise_idft = True
            self._ortho = False

        # Create dummy byte-aligned arrays.  These will be stored as
        # self._fft.input_array, self._fft.output_array, etc., but we'll be
        # replacing the input and output arrays each time we use a transform.
        a = pyfftw.empty_aligned(self.data_format['time_shape'],
                                 dtype=self.data_format['time_dtype'],
                                 n=self._n_simd)
        A = pyfftw.empty_aligned(self.data_format['freq_shape'],
                                 dtype=self.data_format['freq_dtype'],
                                 n=self._n_simd)

        # Create forward and backward transforms.  Since we're replacing the
        # input and output, not an issue that a and A are stored in both.
        self._fft = pyfftw.FFTW(a, A, axes=self.axes, direction='FFTW_FORWARD',
                                **self._kwargs)
        self._ifft = pyfftw.FFTW(A, a, axes=self.axes,
                                 direction='FFTW_BACKWARD', **self._kwargs)

    @_check_fft_exists
    def fft(self, a):
        """FFT, using the `pyfftw.FFTW` class.

        Parameters
        ----------
        a : array_like
            Time-domain data.

        Returns
        -------
        out : `~numpy.ndarray`
            Transformed frequency-domain data.
        """
        # Make an empty array to store transform output.
        A = pyfftw.empty_aligned(self.data_format['freq_shape'],
                                 dtype=self.data_format['freq_dtype'],
                                 n=self._n_simd)
        # A is returned by self._fft.
        return self._fft(input_array=a, output_array=A,
                         normalise_idft=self._normalise_idft,
                         ortho=self._ortho)

    @_check_fft_exists
    def ifft(self, A):
        """Inverse FFT, using the `pyfftw.FFTW` class.

        Parameters
        ----------
        a : array_like
            Frequency-domain data.

        Returns
        -------
        out : `~numpy.ndarray`
            Inverse transformed time-domain data.
        """
        # Multi-dimensional real transforms destroy their input arrays, so
        # make a (if necessary, byte-aligned) copy.
        # See https://pyfftw.readthedocs.io/en/latest/source/pyfftw/pyfftw.html#scheme-table
        if 'complex' in self.data_format['time_dtype'] and len(self.axes) > 1:
            A_copy = pyfftw.byte_align(A)
            # If A is already byte-aligned, this doesn't copy anything, so:
            if A_copy is A:
                A_copy = A.copy()
        else:
            A_copy = A
        # Make an empty array to store transform output.
        a = pyfftw.empty_aligned(self.data_format['time_shape'],
                                 dtype=self.data_format['time_dtype'],
                                 n=self._n_simd)
        # a is returned by self._fft.
        return self._ifft(input_array=A_copy, output_array=a,
                          normalise_idft=self._normalise_idft,
                          ortho=self._ortho)
