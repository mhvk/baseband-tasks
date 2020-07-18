# Licensed under the GPLv3 - see LICENSE
"""Base classes and tools for the fourier module.

Implementation Notes
--------------------

The base classes provide common code for adding new FFT engines.

In particular, the `FFTMakerBase` class is subclassed to create
``*FFTMaker`` classes (where ``*`` stands for a package such as `pyfftw`
or `numpy`).  These classes are initialized with default information
needed for creating an FFT instance. For instance, for
``PyfftwFFTMaker``, this holds ``flags``, ``threads``, etc. Via the
`FFTMakerMeta` meta class, all such maker classes are registered in the
`FFT_MAKER_CLASSES` dict, keyed by a lower-case version of the name
(with ``fftmaker`` removed).

These ``*FFTMaker`` instances in turn can be used, via their
``__call__`` method, to create ``*FFT`` subclass instances, setting
relevant attributes such as ``time_shape``, ``time_dtype``, ``axis``,
etc., and instantiating them for a given direction.

The ``*FFT`` classes themselves are most easily based on `FFTBase`,
which defines properties for accessing the various attributes, a default
``__init__`` method that allows on to create an instance for a given
direction of the FFT, and a :meth:`FFTBase.__call__` method for actually
doing the FFT on data.  The `FFTBase` class also defines convenience
properties, such as `FFTBase.frequency` to get the sample frequencies,
and :meth:`FFTBase.inverse` for getting the FFT in the inverse
direction.

Selection of a default FFT package is done via `fft_maker`, which stores
a default ``*FFTMaker`` instance.  It is based on
`astropy.utils.state.ScienceState`, but adds a `fft_maker.system_default`
factory that one can access by setting the state to `None`, as well as
``__new__`` method which allows one to use the default ``*FFTMaker``
instance to create an ``*FFT`` instance.

"""

import operator

import numpy as np
from astropy.utils.decorators import classproperty
from astropy.utils.state import ScienceState


__all__ = ['FFTMakerBase', 'FFTBase', 'fft_maker',
           'FFTMakerMeta', 'FFT_MAKER_CLASSES']


__doctest_requires__ = {'fft_maker*': ['pyfftw']}


FFT_MAKER_CLASSES = {}
"""Dict for storing FFT maker classes, indexed by their name or prefix."""


class FFTBase:
    """Framework for single pre-defined FFT and its associated metadata."""

    def __init__(self, direction):
        self._direction = direction if direction == 'backward' else 'forward'

    @property
    def direction(self):
        """Direction of the FFT ('forward' or 'backward')."""
        return self._direction

    @property
    def time_shape(self):
        """Shape of the time-domain data."""
        return self._time_shape

    @property
    def time_dtype(self):
        """Data type of the time-domain data."""
        return self._time_dtype

    @property
    def frequency_shape(self):
        """Shape of the frequency-domain data."""
        return self._frequency_shape

    @property
    def frequency_dtype(self):
        """Data type of the frequency-domain data."""
        return self._frequency_dtype

    @property
    def axis(self):
        """Axis over which to perform the FFT."""
        return self._axis

    @property
    def ortho(self):
        """Use orthogonal normalization.

        If `True`, both forward and backward transforms are scaled by
        1 / sqrt(n), where n is the size of time-domain array's transform
        axis.  If `False`, forward transforms are unscaled and inverse ones
        scaled by 1 / n.
        """
        return self._ortho

    @property
    def sample_rate(self):
        """Rate of samples in the time domain."""
        return self._sample_rate

    # While calculating the frequencies is fairly involved, we do not cache
    # the result using, e.g., a lazyproperty, since the array could be quite
    # large, and internally at least we access this information only once.
    @property
    def frequency(self):
        """FFT sample frequencies.

        Uses `numpy.fft.fftfreq` for complex time-domain data, which returns,
        for an array of length n and a time-domain ``sample_rate``,

            f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] * sample_rate / n

        if n is even, and

            f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] * sample_rate / n

        if n is odd.

        For real time-domain data, `numpy.fft.rfftfreq` is used, which returns

            f = [0, 1, ...,     n/2-1,     n/2] * sample_rate / n

        if n is even, and

            f = [0, 1, ..., (n-1)/2-1, (n-1)/2] * sample_rate / n

        if n is odd.

        If ``sample_rate`` is `None`, a unitless rate of 1 is used.

        Returns
        -------
        frequency : `~numpy.ndarray` or array of `~astropy.units.Quantity`
            Sample frequencies, with shape (len(f), 1, ..., 1).  The trailing
            dimensions of length unity are to facilitate broadcasting when
            operating on ``frequency``.
        """
        sample_rate = 1. if self.sample_rate is None else self.sample_rate
        a_length = self._time_shape[self.axis]
        if self._time_dtype.kind == 'f':
            frequency = np.fft.rfftfreq(a_length, d=(1. / sample_rate))
        else:
            frequency = np.fft.fftfreq(a_length, d=(1. / sample_rate))
        # Reshape frequencys to add trailing dimensions.
        frequency.shape = (frequency.shape
                           + (len(self._time_shape) - self.axis - 1) * (1,))
        return frequency

    def __call__(self, a):
        """Perform FFT.

        To display the direction of the transform and shapes and dtypes of the
        arrays, use `print` or `repr`.

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

    def inverse(self):
        """Return inverse transform.

        If the type of fourier transform allows it (e.g., PyFFTW), the inverse
        transform will have its input array matched to the output array and
        vice versa, i.e., a sequence transforming some input data, do in-place
        calculations on the output fourier spectrum, and then transform back,
        will overwrite the input data.  If this is not wanted, instantiate a
        new class directly (e.g., ``forward.__class__(direction='backward')``.

        Returns
        -------
        inverse_transform : `~baseband_tasks.fourier.base.FFTBase` subclass
            Returns a new instance of the calling class with reversed transform
            direction.
        """
        return self.__class__(
            direction=('forward' if self.direction == 'backward'
                       else 'backward'))

    def __copy__(self):
        return self.__class__(direction=self.direction)

    def __eq__(self, other):
        return (self.direction == other.direction
                and self.time_shape == other.time_shape
                and self.time_dtype == other.time_dtype
                and self.frequency_shape == other.frequency_shape
                and self.frequency_dtype == other.frequency_dtype
                and self.axis == other.axis
                and self.ortho == other.ortho
                and self.sample_rate == other.sample_rate)

    def __repr__(self):
        return ("<{s.__class__.__name__}"
                " direction={s.direction},\n"
                "    axis={s.axis}, ortho={s.ortho},"
                " sample_rate={s.sample_rate}\n"
                "    Time domain: shape={s.time_shape},"
                " dtype={s.time_dtype}\n"
                "    Frequency domain: shape={s.frequency_shape},"
                " dtype={s.frequency_dtype}>".format(s=self))


class FFTMakerMeta(type):
    """Registry of FFT maker classes.

    Registers classes using the `FFT_MAKER_CLASSES` dict, using a key
    generated by lowercasing the class's name and removing any trailing
    'fftmaker' (eg. the key for 'NumpyFFTMaker' is 'numpy').  The class
    automatically registers any subclass of `FFTMakerBase`, checking for key
    conflicts before registering.  Used by `fft_maker` to select classes.

    Users that wish to register their own FFT maker class should either
    subclass `FFTMakerBase` or use `FFTMakerMeta` as the metaclass.
    """
    _registry = FFT_MAKER_CLASSES

    def __init__(cls, name, bases, dct):

        # Ignore FFTMakerBase.
        if name != 'FFTMakerBase':

            # Extract name from class.
            key = name.lower()

            if key.endswith('fftmaker') and len(key) > 8:
                key = key[:-8]

            # Check if class is already registered.
            if key in FFTMakerMeta._registry:
                raise ValueError("key {0} already registered in "
                                 "FFT_MAKER_CLASSES.".format(key))

            FFTMakerMeta._registry[key] = cls

        super().__init__(name, bases, dct)


class FFTMakerBase(metaclass=FFTMakerMeta):
    """Base class for all FFT factories."""

    _FFTBase = FFTBase
    _repr_kwargs = {}

    def __call__(self, shape, dtype, direction='forward', axis=0, ortho=False,
                 sample_rate=None, **kwargs):
        """Create an FFT instance.

        This method should generally by overridden by subclasses.

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
        fft : ``cls._FFTBase`` instance
            Single pre-defined FFT object.
        """
        # Ensure arguments have proper types and values.
        time_shape = tuple(shape)
        time_dtype = np.dtype(dtype)
        axis = operator.index(axis)
        # Store time and frequency-domain array shapes.
        frequency_shape, frequency_dtype = self.get_frequency_data_info(
            time_shape, time_dtype, axis=axis)
        attributes = dict(
            _time_shape=time_shape,
            _time_dtype=time_dtype,
            _frequency_shape=frequency_shape,
            _frequency_dtype=frequency_dtype,
            _axis=axis,
            _ortho=bool(ortho),
            _sample_rate=sample_rate)
        for key, value in kwargs.items():
            attributes['_' + key] = value

        cls = type(self._FFTBase.__name__.replace('Base', ''),
                   (self._FFTBase,), attributes)
        return cls(direction)

    def get_frequency_data_info(self, shape, dtype, axis=0):
        """Determine frequency-domain array shape and dtype.

        Parameters
        ----------
        shape : tuple
            Shape of the time-domain data array, i.e. the input to the forward
            transform and the output of the inverse.
        dtype : str or `~numpy.dtype`
            Data type of the time-domain data array.  May pass either the
            name of the dtype or the `~numpy.dtype` object.
        axis : int, optional
            Axis of transform.  Default: 0.

        Returns
        -------
        frequency_shape : tuple
            Shape of the frequency-domain data array.
        frequency_dtype : `~numpy.dtype`
            Data type of the frequency-domain data array.
        """
        if dtype.kind == 'f':
            frequency_shape = list(shape)
            frequency_shape[axis] = shape[axis] // 2 + 1
            frequency_dtype = np.dtype('c{0:d}'.format(2 * dtype.itemsize))
            return tuple(frequency_shape), frequency_dtype
        # No need to make a copy, since we're not altering shape.
        return shape, dtype

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__,
                               ', '.join(['{}={}'.format(k, v) for k, v
                                          in self._repr_kwargs.items()]))


class fft_maker(ScienceState):
    """Create an FFT, with a settable default engine.

    Parameters
    ----------
    shape : tuple
        Shape of the time-domain data array, i.e. the input to the forward
        transform and the output of the inverse.
    dtype : `~numpy.dtype`
        Data type of the time-domain data array.  May pass either a
        `~numpy.dtype` instance, or anything that initializes one.
    direction : 'forward' or 'backward', optional
        Direction of the FFT.  Default: 'forward'
    axis : int, optional
        Axis to transform.  Default: 0.
    ortho : bool, optional
        Whether to use orthogonal normalization.  Default: `False`.
    sample_rate : float, `~astropy.units.Quantity`, or None, optional
        Sample rate, used to determine the FFT sample frequencies.  If
        `None`, a unitless rate of 1 is used.

    Notes
    -----
    The `fft_maker.set` method can be used to set the default engine for
    all tasks that require fourier transforms.


    Examples
    --------
    To set up to use numpy's fft, and then create an fft function acting
    on arrays of 1000 elements::

      >>> from baseband_tasks.fourier import fft_maker
      >>> from astropy import units as u
      >>> with fft_maker.set('numpy'):
      ...     fft = fft_maker((1000,), 'complex64', sample_rate=1.*u.kHz)
      >>> fft
      <NumpyFFT direction=forward,
          axis=0, ortho=False, sample_rate=1.0 kHz
          Time domain: shape=(1000,), dtype=complex64
          Frequency domain: shape=(1000,), dtype=complex64>

    """

    # This is set in __init__.
    _system_default = None

    _value = None

    def __new__(cls, shape, dtype, *,
                direction='forward', axis=0, ortho=False, sample_rate=None):
        fft_engine = cls.get()
        return fft_engine(shape, dtype, direction=direction, axis=axis,
                          ortho=ortho, sample_rate=sample_rate)

    @classproperty
    def system_default(cls):
        """System default FFT factory."""
        return cls._system_default

    @classmethod
    def validate(cls, value):
        if value is None:
            value = cls.system_default
        if not isinstance(value, FFTMakerBase):
            raise TypeError("Can only set the default to an instance of "
                            "a FFT maker such as NumpyFFTMaker().")
        return value

    @classmethod
    def set(cls, fft_engine, **kwargs):
        """Set the FFT factory to be used in new tasks.

        This method can be used to set the factory only temporarily, by
        using it as a context in a ``with`` statement.

        Parameters
        ----------
        fft_engine : {'numpy', 'pyfftw'}, FFTMaker instance, or `None`
            Keyword identifying the FFT maker class to create, or an FFT
            maker instance.  If `None`, the engine stored in the
            ``system_default`` attribute is used.
        **kwargs
            Additional keyword arguments for initializing the maker class
            (eg. ``n_simd`` for 'pyfftw').  Only allowed when ``fft_engine``
            is a name of a maker class.

        Examples
        --------
        To use PyFFTW on a machine with 4 threads::

          >>> from baseband_tasks.fourier import fft_maker
          >>> fft_maker.set('pyfftw', threads=4)
          <ScienceState fft_maker: PyfftwFFTMaker(...)>

        This factory will be used by default when defining new tasks that
        need fourier transforms (channelization, dedispersion, etc.)

        To reset to the system default, pass in `None`::

          >>> fft_maker.set(None)  # doctest: +IGNORE_OUTPUT
          >>> assert fft_maker.get() is fft_maker.system_default

        To use the settings only for a few specific tasks::

          >>> with fft_maker.set('numpy'):  # doctest: +SKIP
          ...     ch = Channelize(fh, 1024)
        """
        if fft_engine is None:
            fft_engine = cls.system_default

        elif not isinstance(fft_engine, FFTMakerBase):
            fft_engine = FFT_MAKER_CLASSES[fft_engine](**kwargs)

        elif kwargs:
            raise TypeError("cannot pass keyword arguments except if "
                            "fft_engine is the name of an FFT maker.")

        return super().set(fft_engine)
