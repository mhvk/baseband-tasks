.. _fourier:

*********************************************
Fourier Transforms (`scintillometry.fourier`)
*********************************************

Introduction
============

The Fourier transform module contains classes that wrap various fast Fourier
transform (FFT) packages, in particular `numpy.fft` and `pyfftw.FFTW`.  The
purpose of the module is to give the packages a common interface, and to allow
individual transforms to be defined once, then re-used multiple times.  This is
especially useful for FFTW, which achieves its fast transforms through prior
planning.

Currently does not support Hermitian FFTs.

.. _fourier_usage:

Using the Fourier Module
========================

To get an FFT maker, we may use the `~scintillometry.fourier.get_fft_maker`
function::

    >>> from scintillometry import fourier
    >>> FFTMaker = fourier.get_fft_maker('numpy')

To create a transform, we pass information about the input or output data
arrays, transform axis (if the input data is multi-dimensional), normalization
convention and sample rate to ``FFTMaker``::

    >>> import numpy as np
    >>> import astropy.units as u
    >>> FFT = FFTMaker(time_data=np.ones(1000), ortho=True,
    ...                sample_rate=1.*u.kHz)

Here, we passed a dummy array to ``time_data``, from which ``FFTMaker``
extracts the data type and shape (the dummy array is not used for the
transform). We can also pass this data directly as a dictionary (see below). 
We have chosen orthogonal normalization, which normalizes both the
frequency and time-domain outputs by :math:`1 / n`, where :math:`n` is the
length of the time-domain array.

``FFT`` is a transform class, and to perform a transform, we must create an
instance of it, and then pass that instance some input data::

    >>> fft = FFT(direction='forward')
    >>> y = np.sin(2. * np.pi * np.arange(1000))
    >>> Y = fft(y)

``Y`` is the Fourier transform of ``y``.  To obtain the inverse, we use the
``inverse`` method in ``fft``::

    >>> ifft = fft.inverse()
    >>> yn = ifft(Y)
    >>> np.allclose(y, yn)
    True

To show information about the transform, we can simply print instances::

    >>> fft
    <NumpyFFT direction=forward,
        axis=0, ortho=True, sample_rate=1.0 kHz
        Time domain: shape=(1000,), dtype=float64
        Frequency domain: shape=(501,), dtype=complex128>

Since we usually only need ``fft`` and not its class definition, ``FFTMaker``
has convenience methods `~scintillometry.fourier.base.FFTMakerBase.fft` and
`~scintillometry.fourier.base.FFTMakerBase.ifft` to produce forward and inverse
transform instances, respectively::

    >>> fft2 = FFTMaker.fft(time_data={'shape': (1000,), 'dtype': 'float64'},
    ...                     ortho=True, sample_rate=1.*u.kHz)
    >>> fft == fft2
    True

When creating an inverse transform, if only frequency-domain array information
is provided the time-domain array is assumed to have a complex data type.  To
create an inverse transform that outputs arrays with floating point dtypes,
explicitly pass in time-domain array info::

    >>> ifft2 = FFTMaker.ifft(time_data={'shape': (1000,),
    ...                                  'dtype': 'float64'},
    ...                       ortho=True, sample_rate=1.*u.kHz)
    >>> ifft == ifft2
    True

Note that it is unnecessary to pass the frequency-domain data.  Indeed, because
Hermitian FFTs are currently not supported, providing the time-domain array
information is sufficient to create either the forward or inverse transform. 
The option of passing in frequency-domain information is for the convenience of
the user.

.. _fourier_api:

Reference/API
=============

.. automodapi:: scintillometry.fourier.base
.. automodapi:: scintillometry.fourier.numpy
.. automodapi:: scintillometry.fourier.pyfftw