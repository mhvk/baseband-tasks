# Licensed under the GPLv3 - see LICENSE
import pytest
import numpy as np
import astropy.units as u
from astropy.time import Time

from baseband_tasks.generators import (
    StreamGenerator, EmptyStreamGenerator, NoiseGenerator)
from baseband_tasks.functions import Square
from baseband_tasks.base import Task


class StreamBase:
    def setup(self):
        self.start_time = Time('2010-11-12T13:14:15')
        self.sample_rate = 1. * u.kHz
        self.shape = (1000, 4, 2)


class TestGenerator(StreamBase):
    """Test sources that produce things looking like a stream reader."""

    def my_source(self, sh):
        data = np.empty((sh.samples_per_frame,) + sh.sample_shape,
                        sh.dtype)
        data[...] = sh.tell()
        return data

    def test_basics(self):
        with StreamGenerator(self.my_source,
                             shape=self.shape, start_time=self.start_time,
                             sample_rate=self.sample_rate,
                             samples_per_frame=1) as sh:
            assert sh.size == np.prod(self.shape)
            assert sh.shape == self.shape
            assert isinstance(sh.dtype, np.dtype)
            assert sh.dtype == np.dtype('c8')
            assert sh.samples_per_frame == 1
            assert abs(sh.stop_time - sh.start_time - 1. * u.s) < 1. * u.ns
            sh.seek(980)
            data1 = sh.read(1)
            assert data1.dtype == sh.dtype
            assert np.all(data1 == 980.)
            data2 = sh.read()
            assert data2.shape == (1000 - 981, 4, 2)
            assert np.all(data2 == np.arange(981, 1000).reshape(19, 1, 1))

    def test_frequency_sideband_setting(self):
        frequency = np.array([320., 350., 380., 410.])[:, np.newaxis] * u.MHz
        sideband = np.array([-1, 1])
        with StreamGenerator(self.my_source,
                             frequency=frequency, sideband=sideband,
                             shape=self.shape, start_time=self.start_time,
                             sample_rate=self.sample_rate,
                             samples_per_frame=1) as sh:
            assert np.all(sh.frequency == frequency)
            assert np.all(sh.sideband == sideband)
            with pytest.raises(AttributeError):
                sh.polarization

    def test_repr(self):
        frequency = np.array([320., 350., 380., 410.])[:, np.newaxis] * u.MHz
        sideband = np.array([-1, 1])
        with StreamGenerator(self.my_source,
                             frequency=frequency, sideband=sideband,
                             shape=self.shape, start_time=self.start_time,
                             sample_rate=self.sample_rate,
                             samples_per_frame=1) as sh:
            r = repr(sh)

        assert r.startswith('StreamGenerator(')
        assert 'start_time=' in r
        assert 'samples_per_frame' not in r  # has default
        assert 'frequency=' in r
        assert 'polarization' not in r

    def test_frequency_sideband_polarization_setting(self):
        frequency = np.array([320., 320., 350., 350.])[:, np.newaxis] * u.MHz
        sideband = np.array([-1, 1, -1, 1])[:, np.newaxis]
        polarization = np.array(['X', 'Y'])
        with StreamGenerator(self.my_source, polarization=polarization,
                             frequency=frequency, sideband=sideband,
                             shape=self.shape, start_time=self.start_time,
                             sample_rate=self.sample_rate,
                             samples_per_frame=1) as sh:
            assert np.all(sh.frequency == frequency)
            assert np.all(sh.sideband == sideband)
            assert np.all(sh.polarization == polarization)

    def test_sample_slice(self):
        sh = StreamGenerator(self.my_source,
                             shape=self.shape, start_time=self.start_time,
                             sample_rate=self.sample_rate,
                             samples_per_frame=1)
        sliced = sh[-400:]
        assert sliced.shape == (400,)+sh.sample_shape
        assert abs(sliced.stop_time - sh.stop_time) < 1.*u.ns
        assert abs(sliced.start_time
                   - (sh.stop_time - 400/sh.sample_rate)) < 1.*u.ns
        sh.seek(-400, 2)
        expected = sh.read()
        data = sliced.read()
        assert np.all(data == expected)

        r = repr(sliced)
        assert r.startswith('GetSlice(ih, item=')
        assert '\nih: StreamGenerator(' in r

    def test_exceptions(self):
        with StreamGenerator(self.my_source,
                             shape=self.shape, start_time=self.start_time,
                             sample_rate=self.sample_rate,
                             samples_per_frame=1) as sh:
            with pytest.raises(EOFError):
                sh.seek(-10, 2)
                sh.read(20)
            with pytest.raises(AttributeError):
                sh.frequency
            with pytest.raises(AttributeError):
                sh.sideband
            with pytest.raises(AttributeError):
                sh.polarization

        with pytest.raises(ValueError):
            StreamGenerator(self.my_source,
                            shape=self.shape, start_time=self.start_time,
                            sample_rate=self.sample_rate,
                            sideband=np.ones((2, 2), dtype='i1'))


class TestConstant(StreamBase):
    """Test sources that produce constant signals."""

    def test_zeros_generation(self):
        def generate_zero(sh):
            return np.zeros((sh.samples_per_frame,) + sh.shape[1:],
                            sh.dtype)

        with StreamGenerator(generate_zero, shape=self.shape,
                             start_time=self.start_time,
                             sample_rate=self.sample_rate,
                             samples_per_frame=10) as sh:
            assert sh.size == np.prod(self.shape)
            assert sh.shape == self.shape
            assert sh.samples_per_frame == 10
            assert abs(sh.stop_time - sh.start_time - 1. * u.s) < 1. * u.ns
            sh.seek(10)
            data1 = sh.read(2)
            assert data1.dtype == sh.dtype == np.dtype('c8')
            assert np.all(data1 == 0.)
            data2 = sh.read()
            assert data2.shape == (1000 - 12, 4, 2)
            assert np.all(data2 == 0.)

    def test_1p1j_setting(self):
        def set_constant(data):
            data[...] = 1 + 1j
            return data

        with EmptyStreamGenerator(
                shape=self.shape, start_time=self.start_time,
                sample_rate=self.sample_rate, samples_per_frame=20) as eh, \
                Task(eh, set_constant) as sh:
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

        def set_tone(data):
            data[...] = tone
            return data

        with EmptyStreamGenerator(
                shape=(10, 1000), start_time=self.start_time,
                sample_rate=10. * u.Hz, samples_per_frame=2) as eh, \
                Task(eh, set_tone) as sh:
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

        def set_tone(data):
            data[...] = tone
            return data

        with EmptyStreamGenerator(
                shape=(10, 1000), start_time=self.start_time,
                sample_rate=10. * u.Hz, samples_per_frame=2) as eh, \
                Task(eh, set_tone) as sh:
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

        def set_tone(data):
            data[...] = tone
            return data

        eh = EmptyStreamGenerator(shape=(10, 1000), start_time=self.start_time,
                                  sample_rate=10. * u.Hz, samples_per_frame=2)
        sh = Task(eh, set_tone)
        st = Square(sh)
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


class TestNoise:
    """Test that we can produce normally distribute noise.

    And that the noise generator looks like a streamreader.
    """

    def setup(self):
        self.seed = 1234567
        self.start_time = Time('2010-11-12T13:14:15')
        self.sample_rate = 10. * u.kHz
        self.shape = (10000, 4, 2)

    def test_basics(self):
        with NoiseGenerator(seed=self.seed,
                            shape=self.shape, start_time=self.start_time,
                            sample_rate=self.sample_rate,
                            samples_per_frame=10, dtype=np.complex128) as nh:
            assert nh.size == np.prod(self.shape)
            assert abs(nh.stop_time - nh.start_time - 1. * u.s) < 1. * u.ns
            nh.seek(10)
            data1 = nh.read(2)
            assert data1.shape == (2,) + nh.sample_shape
            assert data1.dtype == np.dtype('c16')
            nh.seek(0)
            data = nh.read()
            assert data.shape == nh.shape
            # Check repeatability.
            assert np.all(data1 == data[10:12])
            # On purpose read over a frame boundary; gh-52.
            nh.seek(9)
            data2 = nh.read(3)
            assert np.all(data2 == data[9:12])
            nh.seek(9000)
            data3 = nh.read()
            assert np.all(data3 == data[9000:])

        assert abs(data.mean()) < 10. / data.size ** 0.5
        assert abs(data.std() - np.sqrt(2.)) < 14. / data.size ** 0.5

    def test_no_repitition(self):
        with NoiseGenerator(seed=self.seed,
                            shape=self.shape, start_time=self.start_time,
                            sample_rate=self.sample_rate,
                            samples_per_frame=1, dtype=np.complex128) as nh:
            d0 = nh.read(1)
            nh.seek(3)
            d3 = nh.read(1)
            nh.seek(2)
            d2 = nh.read(1)
            d3_2 = nh.read(1)
            d4 = nh.read(1)
            assert not np.any(d0 == d3)
            assert not np.any(d3 == d2)
            assert not np.any(d3 == d4)
            # This used to fail, as the state was reset.  Regression test.
            assert not np.any(d2 == d4)
            assert np.all(d3 == d3_2)

    def test_reproducible(self):
        # Should be independent of order data is read in.
        kwargs = dict(seed=self.seed,
                      shape=self.shape, start_time=self.start_time,
                      sample_rate=self.sample_rate,
                      samples_per_frame=4, dtype=np.complex128)
        with NoiseGenerator(**kwargs) as nh1:
            reference = nh1.read(40)

        with NoiseGenerator(**kwargs) as nh1:
            # Read in reverse order, in 10 samples at time time, i.e.,
            # on purpose not respecting frame boundaries.
            pieces = []
            for i in range(4):
                nh1.seek(30-i*10)
                pieces.append(nh1.read(10))
            data = np.concatenate(pieces[::-1])
        assert np.all(data == reference)

    @pytest.mark.parametrize('item', [slice(-400, None), slice(5, 15)])
    def test_sample_slice(self, item):
        sh = NoiseGenerator(seed=self.seed,
                            shape=self.shape, start_time=self.start_time,
                            sample_rate=self.sample_rate,
                            samples_per_frame=10, dtype=np.complex128)
        expected = sh.read()[item]
        sliced = sh[item]
        data = sliced.read()
        assert np.all(data == expected)

    def test_use_as_source(self):
        """Test that noise routine with squarer gives expected levels."""
        nh = NoiseGenerator(seed=self.seed,
                            shape=self.shape, start_time=self.start_time,
                            sample_rate=self.sample_rate,
                            samples_per_frame=10, dtype=np.complex128)
        st = Square(nh)
        assert st.dtype == np.dtype('f8')
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
