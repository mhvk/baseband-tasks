# Licensed under the GPLv3 - see LICENSE

import numpy as np
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose
import pytest

from ...fourier import FFTBase, FFT, NumpyFFT
from ... import FFT_MAKER_CLASSES, get_fft

# class TestFFTBase(object):
#     """Test FFTBase's initialization and frequency generator."""

#     def test_setup(self):
#         """Check that we can set up properties, and they can't be reset."""
#         fft = FFTBase()
#         a = np.empty((100, 10), dtype='float')
#         fft.setup(a, 'complex128', axes=(0, 1), norm='ortho')
#         assert fft.axes == (0, 1)
#         assert fft.norm == 'ortho'
#         assert fft.data_format['time_shape'] == (100, 10)
#         assert fft.data_format['time_dtype'] == 'float64'
#         assert fft.data_format['freq_shape'] == (100, 6)
#         assert fft.data_format['freq_dtype'] == 'complex128'
#         with pytest.raises(AttributeError):
#             fft.norm = None
#         # Check alternate initializations.
#         fft.setup({'shape': (100, 10), 'dtype': 'float64'},
#                   'complex128', axes=-1, norm='not ortho')
#         assert fft.axes == (-1,)
#         assert fft.norm is None
#         assert fft.data_format['time_shape'] == (100, 10)
#         assert fft.data_format['time_dtype'] == 'float64'
#         assert fft.data_format['freq_shape'] == (100, 6)
#         assert fft.data_format['freq_dtype'] == 'complex128'
#         assert repr(fft).startswith('<FFTBase')

#         # Error checking test.
#         with pytest.raises(AssertionError) as excinfo:
#             fft.setup({'shape': (100, 10), 'dtype': 'float64'},
#                       'float64', verify=True)
#         assert 'must be complex' in str(excinfo)
#         # Check verification can be turned off.
#         fft.setup({'shape': (100, 10), 'dtype': 'float64'},
#                   'float64', verify=False)

#     @pytest.mark.parametrize(
#         ('nsamp', 'sample_rate'), [(1337, 100. * u.Hz),
#                                    (9400, 89. * u.MHz),
#                                    (12, 5.2 * u.GHz),
#                                    (10000, 23.11)])
#     def test_fftfreq(self, nsamp, sample_rate):
#         """Test FFT sample frequency generation - with and without units."""
#         fft = FFTBase()
#         fftbasefreqs = fft.fftfreq(nsamp, sample_rate=sample_rate)
#         # Override default behaviour and return real sample frequencies.
#         fftbaserealfreqs = fft.fftfreq(nsamp, sample_rate=sample_rate,
#                                        positive_freqs_only=True)
#         if isinstance(sample_rate, u.Quantity):
#             unit = sample_rate.unit
#             sample_rate = sample_rate.value
#         else:
#             unit = None
#         npfreqs = np.fft.fftfreq(nsamp, d=(1. / sample_rate))
#         # Only test up to nsamp // 2 in the real case, since the largest
#         # absolute frequency is positive for rfftfreq, but negative for
#         # fftfreq.
#         if unit is None:
#             assert np.allclose(fftbasefreqs, npfreqs, rtol=1e-14, atol=0.)
#             assert np.allclose(fftbaserealfreqs[:-1], npfreqs[:nsamp // 2],
#                                rtol=1e-14, atol=0.)
#         else:
#             assert_quantity_allclose(fftbasefreqs, npfreqs * unit, rtol=1e-14)
#             assert_quantity_allclose(
#                 fftbaserealfreqs[:-1], npfreqs[:nsamp // 2] * unit, rtol=1e-14)


class TestFFTClasses(object):
    """Test FFTs against NumPy's implementation."""

    def setup(self):
        """Pre-calculate NumPy FFTs."""
        self.x = np.linspace(0., 10., 10000)
        # Simple 1D complex sinusoid.
        self.y_exp = np.exp(1.j * 2. * np.pi * self.x)
        # Simple 1D real sinusoid.
        self.y_rnsine = np.sin(2. * np.pi * self.x)
        # More complex 2D transform.
        self.y_r2D = np.random.uniform(low=-13., high=29., size=(100, 25))
        # More complex 3D transform.
        self.y_3D = (np.random.uniform(low=-13., high=29.,
                                       size=(100, 10, 30)) +
                     1.j * np.random.uniform(low=-13., high=29.,
                                             size=(100, 10, 30)))

        # Transforms of the above.
        self.Y_exp = np.fft.fft(self.y_exp)
        self.Y_rnsine = np.fft.rfft(self.y_rnsine, norm='ortho')
        self.Y_r2D = np.fft.rfft(self.y_r2D, axis=0)
        self.Y_3D = np.fft.fft(self.y_3D, axis=1, norm='ortho')

    @pytest.mark.parametrize('FFTClass', tuple(FFT_MAKER_CLASSES.values()))
    def test_fft(self, FFTClass):
        """Test various FFT implementations."""
        fft = FFTClass()
        assert FFTClass.__name__ in repr(fft)

        # Set common array comparison tolerances.
        tolerances = {'atol': 1e-13, 'rtol': 1e-6}

        # 1D complex sinusoid.
        fft.setup(self.y_exp, 'complex128')
        Y = fft.fft(self.y_exp)
        assert np.allclose(Y, self.Y_exp, **tolerances)
        assert np.allclose(fft.ifft(Y), self.y_exp, **tolerances)

        # Check frequency.
        sample_rate = len(self.x) / max(self.x)
        freqs = fft.fftfreq(len(self.x), sample_rate=sample_rate)
        assert np.argmax(np.abs(Y)) == 10
        assert freqs[10] == 1.0

        # 1D real sinusoid, orthogonal normalization.
        fft.setup(self.y_rnsine, 'complex128', norm='ortho')
        Y = fft.fft(self.y_rnsine)
        assert np.allclose(Y, self.Y_rnsine, **tolerances)
        assert np.allclose(fft.ifft(Y), self.y_rnsine, **tolerances)

        # Check Parseval's Theorem (factor of 2 from using a real transform).
        assert np.isclose(np.sum(self.y_rnsine**2),
                          2 * np.sum(np.abs(Y)**2), **tolerances)

        # 2D real.
        fft.setup(self.y_r2D, 'complex128', axis=0)
        # Go from ifft to fft this time, just to see if that matters.
        y = fft.ifft(self.Y_r2D)
        assert np.allclose(y, self.y_r2D, **tolerances)
        assert np.allclose(fft.fft(y), self.Y_r2D, **tolerances)

        # 3D complex.
        fft.setup(self.y_3D, 'complex128', axis=1, norm='ortho')
        Y = fft.fft(self.y_3D)
        assert np.allclose(Y, self.Y_3D, **tolerances)
        assert np.allclose(fft.ifft(Y), self.y_3D, **tolerances)
