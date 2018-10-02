# Licensed under the GPLv3 - see LICENSE
import pytest
import numpy as np
import astropy.units as u
from astropy.time import Time

from ..simulate import Source, ConstantSource, NoiseSource
from ..square import SquareTask


class SourceBase(object):
    def setup(self):
        self.start_time = Time('2010-11-12T13:14:15')
        self.sample_rate = 1. * u.kHz
        self.shape = (1000, 4, 2)


class TestSource(SourceBase):
    """Test sources that produce things looking like a stream reader."""

    def my_source(self, sh):
        data = np.empty((sh.samples_per_frame,) + sh.sample_shape,
                        sh.dtype)
        data[...] = sh.tell()
        return data

    def test_basics(self):
        with Source(self.my_source,
                    shape=self.shape, start_time=self.start_time,
                    sample_rate=self.sample_rate, samples_per_frame=1) as sh:
            assert sh.size == np.prod(self.shape)
            assert sh.shape == self.shape
            assert sh.dtype == np.complex64
            assert sh.samples_per_frame == 1
            assert abs(sh.stop_time - sh.start_time - 1. * u.s) < 1. * u.ns
            sh.seek(980)
            data1 = sh.read(1)
            assert data1.dtype == sh.dtype
            assert np.all(data1 == 980.)
            data2 = sh.read()
            assert data2.shape == (1000 - 981, 4, 2)
            assert np.all(data2 == np.arange(981, 1000).reshape(19, 1, 1))

            with pytest.raises(EOFError):
                sh.seek(-10, 2)
                sh.read(20)


class TestConstant(SourceBase):
    """Test sources that produce constant signals."""
    def test_zeros(self):
        with ConstantSource(0., shape=self.shape, start_time=self.start_time,
                            sample_rate=self.sample_rate,
                            samples_per_frame=10) as sh:
            assert sh.size == np.prod(self.shape)
            assert sh.shape == self.shape
            assert sh.samples_per_frame == 10
            assert abs(sh.stop_time - sh.start_time - 1. * u.s) < 1. * u.ns
            sh.seek(10)
            data1 = sh.read(2)
            assert data1.dtype == sh.dtype == np.complex64
            assert np.all(data1 == 0.)
            data2 = sh.read()
            assert data2.shape == (1000 - 12, 4, 2)
            assert np.all(data2 == 0.)

    def test_1p1j(self):
        with ConstantSource(1 + 1j, shape=self.shape,
                            start_time=self.start_time,
                            sample_rate=self.sample_rate,
                            samples_per_frame=20) as sh:
            assert sh.size == np.prod(self.shape)
            assert sh.shape == self.shape
            assert sh.samples_per_frame == 20
            assert abs(sh.stop_time - sh.start_time - 1. * u.s) < 1. * u.ns
            sh.seek(900)
            data1 = sh.read(20)
            assert data1.shape == (20, 4, 2)
            assert np.all(data1 == 1+1j)

    def test_tone(self):
        tone = np.zeros((1000,), dtype=np.complex64)
        tone[200] = 1.
        with ConstantSource(tone, shape=(10, 1000),
                            start_time=self.start_time,
                            sample_rate=10. * u.Hz, samples_per_frame=2) as sh:
            assert abs(sh.stop_time - sh.start_time - 1. * u.s) < 1. * u.ns
            sh.seek(5)
            data1 = sh.read(1)
            assert np.all(data1 == tone)
            data2 = sh.read()
            assert data2.shape == (10 - 6, 1000)
            assert np.all(data2 == tone)

    def test_wavering_tone(self):
        tone = np.zeros((2, 1000,), dtype=np.complex64)
        tone[0, 200] = 1.
        tone[1, 201] = 1.
        with ConstantSource(tone, shape=(10, 1000), start_time=self.start_time,
                            sample_rate=10. * u.Hz) as sh:
            assert sh.samples_per_frame == 2
            assert abs(sh.stop_time - sh.start_time - 1. * u.s) < 1. * u.ns
            sh.seek(4)
            data1 = sh.read(2)
            assert np.all(data1 == tone)
            data2 = sh.read()
            assert data2.shape == (10 - 6, 1000)
            assert np.all(data2.reshape(-1, 2, 1000) == tone)

    def test_use_as_source(self):
        """Test that it looks like a file also to the squarer."""
        tone = np.zeros((1000,), dtype=np.complex64)
        tone[200] = 1.
        sh = ConstantSource(tone, shape=(10, 1000),
                            start_time=self.start_time,
                            sample_rate=10. * u.Hz, samples_per_frame=2)
        st = SquareTask(sh)
        data1 = st.read()
        assert st.tell() == st.shape[0]
        assert abs(st.time - st.start_time - 1. * u.s) < 1*u.ns
        assert np.all(data1 == np.abs(tone)**2)
        # Seeking and selective squaring.
        st.seek(-3, 2)
        assert st.tell() == st.shape[0] - 3
        data2 = st.read()
        assert data2.shape[0] == 3
        assert np.all(data2 == np.abs(tone)**2)


class TestNoise(object):
    """Test that we can produce normally distribute noise.

    And that the noise generator looks like a streamreader.
    """
    def setup(self):
        self.seed = 1234567
        self.start_time = Time('2010-11-12T13:14:15')
        self.sample_rate = 10. * u.kHz
        self.shape = (10000, 4, 2)

    def test_basics(self):
        with NoiseSource(seed=self.seed,
                         shape=self.shape, start_time=self.start_time,
                         sample_rate=self.sample_rate,
                         samples_per_frame=10, dtype=np.complex128) as nh:
            assert nh.size == np.prod(self.shape)
            assert abs(nh.stop_time - nh.start_time - 1. * u.s) < 1. * u.ns
            nh.seek(10)
            data1 = nh.read(2)
            assert data1.shape == (2,) + nh.sample_shape
            assert data1.dtype == np.complex128
            nh.seek(0)
            data = nh.read()
            assert data.shape == nh.shape
            # Check repeatability.
            assert np.all(data1 == data[10:12])
            nh.seek(10)
            data2 = nh.read(2)
            assert np.all(data2 == data[10:12])
            nh.seek(9000)
            data3 = nh.read()
            assert np.all(data3 == data[9000:])

        assert abs(data.mean()) < 10. / data.size ** 0.5
        assert abs(data.std() - np.sqrt(2.)) < 14. / data.size ** 0.5

    def test_use_as_source(self):
        """Test that noise routine with squarer gives expected levels."""
        nh = NoiseSource(seed=self.seed,
                         shape=self.shape, start_time=self.start_time,
                         sample_rate=self.sample_rate,
                         samples_per_frame=10, dtype=np.complex128)
        st = SquareTask(nh)
        assert st.dtype == np.float64
        data1 = st.read()
        assert st.tell() == st.shape[0]
        assert abs(st.time - st.start_time - 1. * u.s) < 1*u.ns
        assert abs(data1.mean() - 2.) < 10. / st.size ** 0.5
        # Seeking and selective squaring.
        st.seek(-3, 2)
        assert st.tell() == st.shape[0] - 3
        data2 = st.read()
        assert data2.shape[0] == 3
        assert np.all(data2 == data1[-3:])
        nh.seek(-3, 2)
        noise2 = nh.read()
        assert np.all(data2 == noise2.real**2 + noise2.imag**2)
