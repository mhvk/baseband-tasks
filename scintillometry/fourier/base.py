# Licensed under the GPLv3 - see LICENSE

import numpy as np
import operator
from functools import wraps


def _check_fft_exists(func):
    # Decorator for checking that _fft and _ifft exist.
    @wraps(func)
    def check_fft(self, *args, **kwargs):
        if '_fft' in self.__dict__ and '_ifft' in self.__dict__:
            return func(self, *args, **kwargs)
        raise AttributeError('Fourier transform functions have not been '
                             'linked; run self.setup first.')
    return check_fft


class FFTBase(object):
    """Base class for all fast Fourier transforms.

    Provides a `setup` method to store relevant transform information in the
    `data_format`, `norm` and `axes` properties, which respectively contain the
    shapes and dtypes of the FFT arrays, the axes over which to perform the
    FFT, and the normalization convention.  Also provides an `fftfreqs` method
    to return the sample frequencies of the FFT along one axis.

    Actual FFT classes must define a forward and backward transform as `fft`
    and `ifft`, respectively (see placeholders in this class for the format).

    Currently does not support Hermitian FFTs (where frequency-domain data is
    real-valued).
    """

    _data_format = {'time_shape': None,
                    'time_dtype': None,
                    'freq_shape': None,
                    'freq_dtype': None}
    _axes = None
    _norm = None

    def setup(self, time_data, freq_dtype, axes=None, norm=None,
              verify=True):
        """Store information about arrays, transform axes, and normalization.

        Parameters
        ----------
        time_data : `~numpy.ndarray` or dict
            Dummy array with dimensions and dtype of time-domain data.  Can
            alternatively give a dict with 'shape' and 'dtype' entries.
        freq_dtype : str
            dtype of frequency-domain data.
        axes : int, tuple, or None, optional
            Axis or axes to transform.  If an int is passed, it is turned into
            a tuple.  If `None` (default), all axes are used.
        norm : 'ortho' or None, optional
            If `None` (default), uses an unscaled forward transform and 1 / n
            scaled inverse transform.  If 'ortho', uses a 1 / sqrt(n) scaling
            for both.
        verify : bool, optional
            Verify setup is successful and self-consistent.
        """
        # Extract information if user passed in a dummy array.
        if isinstance(time_data, np.ndarray):
            time_data = {'shape': time_data.shape,
                         'dtype': time_data.dtype.name}

        # Set axes and normalization.  If axes is None, cycle through all axes
        # (like with numpy.fft).
        if axes is None:
            axes = tuple(range(len(time_data['shape'])))
        else:
            # If axes is an integer, convert to a 1-element tuple.
            try:
                axes = operator.index(axes)
            # If not, typecast to a tuple.
            except TypeError:
                axes = tuple(axes)
            else:
                axes = (axes,)
        self._axes = axes
        self._norm = norm if norm == 'ortho' else None

        # Determine frequency-domain shape.
        freq_shape = list(time_data['shape'])
        # If time-domain data is real, halve the relevant axis in frequency.
        if 'float' in time_data['dtype']:
            freq_shape[self.axes[-1]] = (
                freq_shape[self.axes[-1]] // 2 + 1)

        # Store time and frequency-domain array shapes.
        self._data_format = {'time_shape': time_data['shape'],
                             'time_dtype': time_data['dtype'],
                             'freq_shape': tuple(freq_shape),
                             'freq_dtype': freq_dtype}

        if verify:
            self.verify()

    def verify(self):
        """Verify setup is successful and self-consistent."""
        assert 'complex' in self.data_format['freq_dtype'], (
            "frequency-domain data must be complex.")
        assert len(self.axes) > 0, "must transform over one or more axes!"

    @property
    def axes(self):
        """Axes over which to perform the FFT."""
        return self._axes

    @property
    def norm(self):
        """Normalization convention.

        As in `numpy.fft`, `None` is an unscaled forward transform and 1 / n
        scaled inverse one, and 'ortho' is a 1 / sqrt(n) scaling for both.
        """
        return self._norm

    @property
    def data_format(self):
        """Shapes and dtypes of the FFT arrays.

        'time_' and 'freq_' entries are for time and frequency-domain arrays,
        respectively.
        """
        return self._data_format

    def fft(self, a):
        """Placeholder for forward FFT."""
        raise NotImplementedError()

    def ifft(self, a):
        """Placeholder for inverse FFT."""
        raise NotImplementedError()

    def fftfreq(self, a_length, sample_rate=None, positive_freqs_only=False):
        """Obtains FFT sample frequencies.

        Uses `numpy.fft.fftfreq` or `numpy.fft.rfftfreq`.  As with those, given
        a window of length ``a_length`` and sample rate ``sample_rate``,

            freqs  = [0, 1, ...,   a_length/2-1, -a_length/2,
                      ..., -1] * sample_rate

        if a_length is even, and

            freqs  = [0, 1, ..., (a_length-1)/2, -(a_length-1)/2,
                      ..., -1] * sample_rate

        if a_length is odd.

        Parameters
        ----------
        a_length : int
            Length of the array being transformed.
        sample_rate : `~astropy.units.Quantity`, optional
            Sample rate.  If `None` (default), output is unitless.
        positive_freqs_only : bool, optional
            Whether to return only the positive frequencies.  Default: `False`.

        Returns
        -------
        freqs : `~numpy.ndarray`
            Sample frequencies.
        """
        if sample_rate is None:
            sample_rate = 1.
        if positive_freqs_only:
            return np.fft.rfftfreq(operator.index(a_length)) * sample_rate
        return np.fft.fftfreq(operator.index(a_length)) * sample_rate

    def __repr__(self):
        return ("<{s.__class__.__name__} time_shape={fmt[time_shape]},"
                " time_dtype={fmt[time_dtype]}\n"
                "    freq_shape={fmt[freq_shape]},"
                " freq_dtype={fmt[freq_dtype]}\n"
                "    axes={s.axes}, norm={s.norm}>"
                .format(s=self, fmt=self.data_format))


class NumpyFFT(FFTBase):
    """FFT class that wraps `numpy.fft` functions.

    FFTs of real-valued time-domain data use `~numpy.fft.rfft`,
    `~numpy.fft.rfftn`, and their inverses.  These perform a real-input
    transform on one dimension of the input, halving that dimension's length in
    the output.

    Currently does not support Hermitian FFTs (`~numpy.fft.hfft`, etc.).
    """

    def setup(self, time_data, freq_dtype, axes=None, norm=None,
              verify=True):
        """Set up FFT.

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

        complex_data = 'complex' in self.data_format['time_dtype']

        # Select the forward and backward FFT functions to use.
        if len(self.axes) > 1:
            if complex_data:
                def fft(a):
                    return np.fft.fftn(a, axes=self.axes, norm=self.norm)

                def ifft(A):
                    return np.fft.ifftn(A, axes=self.axes, norm=self.norm)

            else:
                def fft(a):
                    return np.fft.rfftn(a, axes=self.axes, norm=self.norm)

                # irfftn needs explicit shape for odd-numbered output shapes.
                def ifft(A):
                    return np.fft.irfftn(A, s=self.data_format['time_shape'],
                                         axes=self.axes, norm=self.norm)

        else:
            if complex_data:
                def fft(a):
                    return np.fft.fft(a, axis=self.axes[0], norm=self.norm)

                def ifft(A):
                    return np.fft.ifft(A, axis=self.axes[0], norm=self.norm)

            else:
                def fft(a):
                    return np.fft.rfft(a, axis=self.axes[0], norm=self.norm)

                # irfft needs explicit length for odd-numbered outputs.
                def ifft(A):
                    return np.fft.irfft(A, n=self.data_format['time_shape'][0],
                                        axis=self.axes[0], norm=self.norm)

        self._fft = fft
        self._ifft = ifft

    @_check_fft_exists
    def fft(self, a):
        """FFT, using the `numpy.fft` functions.

        Parameters
        ----------
        a : array_like
            Time-domain data.

        Returns
        -------
        out : `~numpy.ndarray`
            Transformed frequency-domain data.
        """
        return self._fft(a)

    @_check_fft_exists
    def ifft(self, a):
        """Inverse FFT, using the `numpy.fft` functions.

        Parameters
        ----------
        a : array_like
            Frequency-domain data.

        Returns
        -------
        out : `~numpy.ndarray`
            Inverse transformed time-domain data.
        """
        return self._ifft(a)
