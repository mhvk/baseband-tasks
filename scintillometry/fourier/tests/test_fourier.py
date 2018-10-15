# Licensed under the GPLv3 - see LICENSE

import numpy as np
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose
import pytest

from ..base import FFT_MAKER_CLASSES
from .. import get_fft_maker


class TestFFTClasses(object):
    """Test FFTs against NumPy's implementation."""

    def setup(self):
        """Pre-calculate NumPy FFTs."""
        x = np.linspace(0., 10., 7919)    # 7919 is the 1000th prime number!
        self.sample_rate = x.size / (x.max() - x.min()) * u.kHz
        # 1D complex sinusoid.
        self.y_exp = np.exp(1.j * 2. * np.pi * x)
        # 1D real sinusoid.
        self.y_rnsine = np.sin(2. * np.pi * x)
        # 2D real random.
        self.y_r2D = np.random.uniform(low=-13., high=29., size=(100, 25))
        # 3D complex random.
        self.y_3D = (np.random.uniform(low=-19., high=16.,
                                       size=(100, 10, 30)) +
                     1.j * np.random.uniform(low=-13., high=23.,
                                             size=(100, 10, 30)))

        # Transforms of the above.
        self.Y_exp = np.fft.fft(self.y_exp)
        self.frequency_Y_exp = np.fft.fftfreq(len(self.y_exp),
                                              d=(1. / self.sample_rate))
        self.Y_rnsine = np.fft.rfft(self.y_rnsine, norm='ortho')
        self.frequency_Y_rnsine = np.fft.rfftfreq(len(self.y_rnsine),
                                                  d=(1. / self.sample_rate))
        self.Y_r2D = np.fft.rfft(self.y_r2D, axis=0)
        self.Y_3D = np.fft.fft(self.y_3D, axis=1, norm='ortho')
        self.frequency_Y_3D = np.fft.fftfreq(len(self.y_3D[1]))

        # Set common array comparison tolerances.
        self.tolerances = {'atol': 1e-13, 'rtol': 1e-6}

    @pytest.mark.parametrize('key', tuple(FFT_MAKER_CLASSES.keys()))
    def test_fft(self, key):
        """Test various FFT implementations."""
        # Load class using get_fft_maker, check that we have the right one.
        kwargs = {}
        if key == 'pyfftw':
            kwargs['flags'] = ['FFTW_ESTIMATE']
        FFTMaker = get_fft_maker(key, **kwargs)

        # 1D complex sinusoid.
        fft = FFTMaker(self.y_exp.shape, self.y_exp.dtype,
                       sample_rate=self.sample_rate)
        Y = fft(self.y_exp)
        assert Y.dtype is self.y_exp.dtype
        assert np.allclose(Y, self.Y_exp, **self.tolerances)
        ifft = fft.inverse()
        assert np.allclose(ifft(Y), self.y_exp, **self.tolerances)

        # Check frequency.
        assert_quantity_allclose(fft.frequency, self.frequency_Y_exp)
        # We expect the peak amplitude at 1 kHz.
        assert np.argmax(np.abs(Y)) == 10
        assert fft.frequency[10] == 1.0 * u.kHz

        # Check repr.
        assert repr(fft).startswith('<' + fft.__class__.__name__)

        # 1D real sinusoid, orthogonal normalization, start with inverse
        # transform.
        ifft = FFTMaker((7919,), 'float64', direction='inverse', ortho=True,
                        sample_rate=self.sample_rate)
        y = ifft(self.Y_rnsine)
        # Check frequency.
        assert_quantity_allclose(ifft.frequency, self.frequency_Y_rnsine)
        fft = ifft.inverse()
        Y = fft(y)
        assert np.allclose(y, self.y_rnsine, **self.tolerances)
        assert np.allclose(Y, self.Y_rnsine, **self.tolerances)

        # Check that we can explicitly make the forward transform.
        fftc = FFTMaker((7919,), 'float64', direction='forward', ortho=True,
                        sample_rate=self.sample_rate)
        assert fftc == fft
        # Check that we can copy an FFT.
        assert fft.copy() == fft

        # Check Parseval's Theorem (factor of 2 from using a real transform).
        assert np.isclose(np.sum(self.y_rnsine**2),
                          2 * np.sum(np.abs(Y)**2), **self.tolerances)

        # 2D real.
        fft = FFTMaker(self.y_r2D.shape, self.y_r2D.dtype)
        Y = fft(self.y_r2D)
        ifft = fft.inverse()
        assert np.allclose(Y, self.Y_r2D, **self.tolerances)
        assert np.allclose(ifft(Y), self.y_r2D, **self.tolerances)

        # 3D complex, orthogonal normalization, start with inverse transform.
        ifft = FFTMaker(self.Y_3D.shape, self.Y_3D.dtype, direction='inverse',
                        axis=1, sample_rate=None, ortho=True)
        y = ifft(self.Y_3D)
        fft = ifft.inverse()
        assert np.allclose(y, self.y_3D, **self.tolerances)
        assert np.allclose(fft(y), self.Y_3D, **self.tolerances)

        # Check frequency.
        assert_quantity_allclose(fft.frequency,
                                 self.frequency_Y_3D[:, np.newaxis])
