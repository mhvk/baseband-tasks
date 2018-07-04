# Licensed under the GPLv3 - see LICENSE

import numpy as np
import operator
from astropy.utils import lazyproperty


class FFTMakerBase(object):
    """Base class for all FFT factory classes."""

    _engine_name = None

    def __call__(self, input_data, direction='forward', axis=0,
                 real_transform=False, norm=None, sample_rate=None):
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

        # Extract information if user passed in a dummy array.
        if isinstance(input_data, np.ndarray):
            input_data = {'shape': input_data.shape,
                          'dtype': input_data.dtype}
        elif isinstance(input_data['dtype'], str):
            input_data['dtype'] = np.dtype(input_data['dtype'])

        # Set direction, axis and normalization.  If axis is None, set it to 0.
        direction = direction if direction == 'inverse' else 'forward'
        axis = operator.index(axis)
        norm = norm if norm == 'ortho' else None

        # Determine frequency-domain shape.
        output_shape = list(input_data['shape'])
        # The inverse of a real-valued transform should return real values.
        if direction == 'inverse':
            assert 'complex' in input_data['dtype'].name, (
                "frequency-domain array must be complex.")
            if real_transform:
                output_shape[axis] = output_shape[axis] * 2
                output_dtype = np.dtype('float{0:d}'.format(
                    input_data['dtype'].itemsize * 8 // 2))
            else:
                output_dtype = input_data['dtype']

            freq_data = input_data
            time_data = {'shape': tuple(output_shape),
                         'dtype': output_dtype}
        else:
            if 'float' in input_data['dtype'].name:
                # assert not (input_data['shape'][axis] % 2), (
                #     "time-domain array must have an even number of elements "
                #     "along the axis being transformed.")
                output_shape[axis] = output_shape[axis] // 2 + 1
                output_dtype = np.dtype('complex{0:d}'.format(
                    input_data['dtype'].itemsize * 8 * 2))
            else:
                output_dtype = input_data['dtype']

            time_data = input_data
            freq_data = {'shape': tuple(output_shape),
                         'dtype': output_dtype}

        # Store time and frequency-domain array shapes.
        data_format = {'time_shape': time_data['shape'],
                       'time_dtype': time_data['dtype'],
                       'freq_shape': freq_data['shape'],
                       'freq_dtype': freq_data['dtype']}

        return self._setup_transform(data_format, direction, axis, norm,
                                     sample_rate)

    def _setup_transform(data_format, direction, axis, norm, sample_rate):
        raise NotImplementedError()

    def inverse(self, data_format, direction, axis, norm, sample_rate):
        inverse_direction = 'forward' if direction == 'inverse' else 'inverse'
        return self._setup_transform(data_format, inverse_direction,
                                     axis, norm, sample_rate)


class FFT(object):
    """Single pre-defined FFT and its associated metadata."""

    def __init__(self, fft, data_format, direction, axis, norm,
                 sample_rate, parent):
        self._fft = fft
        self._data_format = data_format
        self._direction = direction
        self._axis = axis
        self._norm = norm
        self._sample_rate = sample_rate
        self._parent = parent

    @property
    def data_format(self):
        """Shapes and dtypes of the FFT arrays.

        'time_' and 'freq_' entries are for time and frequency-domain arrays,
        respectively.
        """
        return self._data_format

    @property
    def direction(self):
        """Axis over which to perform the FFT."""
        return self._direction

    @property
    def axis(self):
        """Axis over which to perform the FFT."""
        return self._axis

    @property
    def norm(self):
        """Normalization convention.

        As in `numpy.fft`, `None` is an unscaled forward transform and 1 / n
        scaled inverse one, and 'ortho' is a 1 / sqrt(n) scaling for both.
        """
        return self._norm

    @property
    def sample_rate(self):
        """Rate of samples in the time domain."""
        return self._sample_rate

    @lazyproperty
    def freq(self):
        """FFT sample frequencies.

        Uses `numpy.fft.fftfreq` for complex time-domain data, which returns,
        for an array of length n and a time-domain ``sample_rate``,

            f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] * sample_rate

        if n is even, and

            f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] * sample_rate

        if n is odd.

        For real time-domain data, `numpy.fft.rfftfreq` is used, which returns

            f = [0, 1, ...,     n/2-1,     n/2] * sample_rate

        if n is even, and

            f = [0, 1, ..., (n-1)/2-1, (n-1)/2] * sample_rate

        if n is odd.

        If ``self.sample_rate`` is `None`, output is unitless.

        Returns
        -------
        freqs : `~numpy.ndarray`
            Sample frequencies.
        """
        sample_rate = self.sample_rate
        if sample_rate is None:
            sample_rate = 1.
        a_length = self.data_format['time_shape'][self.axis]
        if 'float' in self.data_format['time_dtype'].name:
            return np.fft.rfftfreq(a_length) * sample_rate
        return np.fft.fftfreq(a_length) * sample_rate

    def inverse(self):
        return self._parent.inverse(self.data_format, self.direction,
                                    self.axis, self.norm, self.sample_rate)

    def __call__(self, a):
        """Fast Fourier transform.

        For the direction of the transform and shapes and dtypes of the arrays,
        use repr().

        Parameters
        ----------
        a : array_like
            Input data.

        Returns
        -------
        out : `~numpy.ndarray`
            Transformed data.
        """
        return self._fft(a)

    def __repr__(self):
        return ("<{s.__class__.__name__} "
                " engine={s._parent._engine_name},"
                " direction={s.direction},\n"
                "    axis={s.axis}, norm={s.norm},"
                " sample_rate={s.sample_rate}\n"
                "    Time domain: shape={fmt[time_shape]},"
                " dtype={fmt[time_dtype]}\n"
                "    Frequency domain: shape={fmt[freq_shape]},"
                " dtype={fmt[freq_dtype]}>"
                .format(s=self, fmt=self.data_format))


class NumpyFFTMaker(FFTMakerBase):
    """FFT factory class utilizing `numpy.fft` functions.

    FFTs of real-valued time-domain data use `~numpy.fft.rfft` and its inverse.
    `~numpy.fft.rfft` performs a real-input transform on one dimension of the
    input, halving that dimension's length in the output.

    Currently does not support Hermitian FFTs (`~numpy.fft.hfft`, etc.).
    """

    _engine_name = 'numpy'

    def _setup_transform(self, data_format, direction, axis, norm,
                         sample_rate):
        """Set up `numpy.fft` based FFT using metadata."""

        complex_data = 'complex' in data_format['time_dtype']

        if direction == 'forward':
            if complex_data:

                def fft(a):
                    return np.fft.fft(a, axis=axis, norm=norm).astype(
                        data_format['freq_dtype'])

            else:

                def fft(a):
                    return np.fft.rfft(a, axis=axis, norm=norm).astype(
                        data_format['freq_dtype'])

        else:
            if complex_data:

                def fft(A):
                    return np.fft.ifft(A, axis=axis, norm=norm).astype(
                        data_format['time_dtype'])

            else:

                # irfft needs explicit length for odd-numbered outputs.
                def fft(A):
                    return np.fft.irfft(A, n=data_format['time_shape'][axis],
                                        axis=axis, norm=norm).astype(
                                            data_format['time_dtype'])

        return FFT(fft, data_format, direction, axis, norm, sample_rate, self)


# from . import FFT_MAKER_CLASSES


# def get_fft(engine_name, **kwargs):
#     """FFT factory selector."""
#     return FFT_MAKER_CLASSES[engine_name](**kwargs)
