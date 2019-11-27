# Licensed under the GPLv3 - see LICENSE
import copy

import numpy as np
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose
import pytest

from ..base import FFT_MAKER_CLASSES, FFTMakerBase
from .. import fft_maker
from ... import fourier


class TestFFTMaker:
    def setup(self):
        # Ensure we start with a clean slate
        fft_maker.set(None)
        self.default_maker = fft_maker.get()

    def test_system_default(self):
        assert self.default_maker is fft_maker.system_default
        if 'pyfftw' in FFT_MAKER_CLASSES:
            assert isinstance(self.default_maker, fourier.PyfftwFFTMaker)
        else:
            assert isinstance(self.default_maker, fourier.NumpyFFTMaker)

    def test_set_default(self):
        my_maker = fourier.base.FFTMakerBase()
        with fft_maker.set(my_maker):
            assert fft_maker.get() is my_maker

        assert fft_maker.get() is self.default_maker

    def test_invalid_input(self):
        with pytest.raises(TypeError):
            fft_maker('nonsense')
        with pytest.raises(TypeError):
            fft_maker(None, a='nonsense')
        with pytest.raises(KeyError):
            fft_maker.set('nonsense')


class TestFFTClasses:
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
                                       size=(100, 10, 30))
                     + 1.j * np.random.uniform(low=-13., high=23.,
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

    def test_basic(self):
        # 1D complex sinusoid.
        fft = fft_maker(self.y_exp.shape, self.y_exp.dtype,
                        sample_rate=self.sample_rate)
        y = self.y_exp.copy()  # ensure we don't write back to it!
        Y = fft(y)
        assert Y.dtype is self.y_exp.dtype
        assert np.allclose(Y, self.Y_exp, **self.tolerances)
        ifft = fft.inverse()
        y_back = ifft(Y)
        assert np.allclose(y_back, self.y_exp, **self.tolerances)

    @pytest.mark.parametrize('key', tuple(FFT_MAKER_CLASSES.keys()))
    def test_fft(self, key):
        """Test various FFT implementations."""
        # Load class using fft_maker, check that we have the right one.
        kwargs = {}
        if key == 'pyfftw':
            kwargs['flags'] = ['FFTW_ESTIMATE']

        with fft_maker.set(key, **kwargs):
            FFTMaker = fft_maker.get()

        # 1D complex sinusoid.
        fft = FFTMaker(self.y_exp.shape, self.y_exp.dtype,
                       sample_rate=self.sample_rate)
        y = self.y_exp.copy()  # ensure we don't write back to it!
        Y = fft(y)
        assert Y.dtype is self.y_exp.dtype
        assert np.allclose(Y, self.Y_exp, **self.tolerances)
        ifft = fft.inverse()
        y_back = ifft(Y)
        assert np.allclose(y_back, self.y_exp, **self.tolerances)

        # Check frequency.
        assert_quantity_allclose(fft.frequency, self.frequency_Y_exp)
        # We expect the peak amplitude at 1 kHz.
        assert np.argmax(np.abs(Y)) == 10
        assert fft.frequency[10] == 1.0 * u.kHz

        # Check repr.
        assert repr(fft).startswith('<' + fft.__class__.__name__)

        # 1D real sinusoid, orthogonal normalization, start with inverse
        # transform.
        ifft = FFTMaker((7919,), 'float64', direction='backward', ortho=True,
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
        fft_copy = copy.copy(fft)
        assert fft_copy is not fft
        assert fft_copy == fft

        # Check Parseval's Theorem (factor of 2 from using a real transform).
        assert np.isclose(np.sum(self.y_rnsine**2),
                          2 * np.sum(np.abs(Y)**2), **self.tolerances)

        # 2D real.
        fft = FFTMaker(self.y_r2D.shape, self.y_r2D.dtype)
        y = self.y_r2D.copy()
        Y = fft(y)
        ifft = fft.inverse()
        y_back = ifft(Y)
        assert np.allclose(Y, self.Y_r2D, **self.tolerances)
        assert np.allclose(y_back, self.y_r2D, **self.tolerances)

        # 3D complex, orthogonal normalization, start with inverse transform.
        ifft = FFTMaker(self.Y_3D.shape, self.Y_3D.dtype, direction='backward',
                        axis=1, sample_rate=None, ortho=True)
        Y = self.Y_3D.copy()
        y = ifft(Y)
        fft = ifft.inverse()
        Y_back = fft(y)
        assert np.allclose(y, self.y_3D, **self.tolerances)
        assert np.allclose(Y_back, self.Y_3D, **self.tolerances)

        # Check frequency.
        assert_quantity_allclose(fft.frequency,
                                 self.frequency_Y_3D[:, np.newaxis])


def test_against_duplication():
    with pytest.raises(ValueError):
        class NumpyFFTMaker(FFTMakerBase):
            pass


@pytest.mark.skipif('pyfftw' not in FFT_MAKER_CLASSES,
                    reason="Test is PyFFTW specific")
class TestPyfftwFFT:
    def setup(self):
        self.maker = FFT_MAKER_CLASSES['pyfftw']

    def test_inverse_overrides_input(self):
        import pyfftw

        x = np.linspace(0., 10., 8192)
        y = pyfftw.empty_aligned(x.shape, dtype=complex)
        y[:] = np.exp(1.j * 2. * np.pi * x)
        fft = self.maker()(y.shape, y.dtype)
        Y = fft(y)
        ifft = fft.inverse()
        y_back = ifft(Y)
        assert y_back is y
        Y_back = fft(y_back)
        assert Y_back is Y
        assert fft._fftw.input_array is ifft._fftw.output_array
        assert ifft._fftw.input_array is fft._fftw.output_array

    def test_inverse_overrides_input2(self):
        # As above, but defining inverse before calling fft
        import pyfftw

        x = np.linspace(0., 10., 8192)
        y = pyfftw.empty_aligned(x.shape, dtype=complex)
        y[:] = np.exp(1.j * 2. * np.pi * x)
        fft = self.maker()(y.shape, y.dtype)
        ifft = fft.inverse()
        Y = fft(y)
        y_back = ifft(Y)
        assert y_back is y
        Y_back = fft(y_back)
        assert Y_back is Y
        assert fft._fftw.input_array is ifft._fftw.output_array
        assert ifft._fftw.input_array is fft._fftw.output_array

    def test_inverse_overrides_input_reverse(self):
        # As above but calling inverse first.
        import pyfftw

        x = np.linspace(0., 10., 8192)
        Y = pyfftw.empty_aligned(x.shape, dtype=complex)
        Y[:] = np.exp(1.j * 2. * np.pi * x)
        fft = self.maker()(Y.shape, Y.dtype)
        ifft = fft.inverse()
        y = ifft(Y)
        Y_back = fft(y)
        assert Y_back is Y
        y_back = ifft(Y_back)
        assert y_back is y
        assert fft._fftw.input_array is ifft._fftw.output_array
        assert ifft._fftw.input_array is fft._fftw.output_array

    def test_normalization(self):
        x = np.linspace(0., 1., 16)
        fft1 = self.maker()(x.shape, x.dtype, ortho=True)
        fft2 = self.maker()(x.shape, x.dtype, ortho=False)
        y1 = fft1(x.copy())
        y2 = fft2(x.copy())
        assert np.allclose(y1, y2 / np.sqrt(16))
