# Licensed under the GPLv3 - see LICENSE

import numpy as np
import astropy.units as u
import pytest

from baseband_tasks.base import SetAttribute
from baseband_tasks.channelize import Channelize, Dechannelize
from baseband_tasks.fourier import fft_maker

from .common import UseVDIFSample, UseDADASample


class TestChannelizeReal(UseVDIFSample):
    """Test channelization using Baseband's sample VDIF file."""

    def setup(self):
        """Pre-calculate channelized data."""
        super().setup()
        self.n = 1024

        self.ref_start_time = self.fh.start_time
        self.ref_sample_rate = self.fh.sample_rate
        data = self.fh.read()

        self.raw_data = data
        last_sample = self.n * (data.shape[0] // self.n)
        part = data[:last_sample].reshape((-1, self.n) + data.shape[1:])
        rfft = fft_maker(shape=part.shape, dtype=part.dtype, axis=1,
                         sample_rate=self.ref_sample_rate)
        self.ref_data = rfft(part)
        # Note: sideband is actually incorrect for this VDIF file;
        # this is for testing only.
        self.ref_sideband = np.tile([-1, 1], 4)
        self.ref_frequency = ((311.25 + 16 * (np.arange(8) // 2)) * u.MHz
                              + self.ref_sideband * rfft.frequency)
        self.fh_freq = SetAttribute(
            self.fh,
            frequency=311.25*u.MHz+(np.arange(8.)//2)*16.*u.MHz,
            sideband=np.tile([-1, +1], 4))

    def test_channelizetask(self):
        """Test channelization task."""
        ct = Channelize(self.fh, self.n)

        # Channelize everything.
        data1 = ct.read()
        assert ct.tell() == ct.shape[0]
        assert (ct.time - ct.start_time
                - ct.shape[0] / ct.sample_rate) < 1*u.ns
        assert ct.dtype is self.ref_data.dtype is data1.dtype
        assert np.all(self.ref_data == data1)

        # Seeking and selective decode.
        ct.seek(-3, 2)
        assert ct.tell() == ct.shape[0] - 3
        data2 = ct.read()
        assert data2.shape[0] == 3
        assert np.all(self.ref_data[-3:] == data2)

        ct.seek(-2, 2)
        with pytest.raises(EOFError):
            ct.read(10)

        ct.close()
        assert ct.closed
        with pytest.raises(ValueError):
            ct.read(1)
        with pytest.raises(AttributeError):
            ct.ih

    @pytest.mark.parametrize('samples_per_frame', [1, 16, 33])
    def test_channelize_samples_per_frame(self, samples_per_frame):
        """Test channelization task."""
        ct = Channelize(self.fh, self.n, samples_per_frame=samples_per_frame)

        # Channelize everything.
        data1 = ct.read()
        assert len(data1) % samples_per_frame == 0
        assert (len(data1) // samples_per_frame
                == len(self.ref_data) // samples_per_frame)
        ref_data = self.ref_data[:len(data1)]
        assert np.all(data1 == ref_data)

        # Seeking and selective decode.
        ct.seek(-3, 2)
        assert ct.tell() == ct.shape[0] - 3
        data2 = ct.read()
        assert data2.shape[0] == 3
        assert np.all(data2 == ref_data[-3:])

    def test_channelize_frequency_real(self):
        """Test frequency calculation."""
        ct = Channelize(self.fh_freq, self.n)
        assert np.all(ct.sideband == self.ref_sideband)
        assert np.all(ct.frequency == self.ref_frequency)

    def test_dechannelizetask_real(self):
        """Test dechannelization round-tripping."""
        ct = Channelize(self.fh_freq, self.n)
        dt = Dechannelize(ct, self.n, dtype=self.fh.dtype)
        nrec = (self.fh.shape[0] // self.n) * self.n
        assert dt.shape == (nrec,) + self.fh.shape[1:]
        data = dt.read()
        # Note: round-trip is not perfect due to rounding errors.
        assert np.allclose(data, self.raw_data[:nrec], atol=1.e-5)
        assert np.all(dt.frequency == self.fh_freq.frequency)
        assert np.all(dt.sideband == self.fh_freq.sideband)
        # Check class method
        dt2 = ct.inverse(ct)
        data2 = dt2.read()
        assert np.all(data2 == data)

    def test_size_check(self):
        # Quick test of channel sanity check in __init__.
        with pytest.raises(AssertionError):
            Channelize(self.fh, 400001)

    def test_repr(self):
        """Test channelization task."""
        ct = Channelize(self.fh, self.n)
        cr = repr(ct)
        assert cr.startswith('Channelize(ih')
        assert f"n={self.n}" in cr
        dt = Dechannelize(ct, self.n)
        dr = repr(dt)
        assert dr.startswith('Dechannelize(ih')
        assert f"n={self.n}" in dr

    def test_missing_frequency_sideband(self):
        with Channelize(self.fh, self.n) as ct:
            with pytest.raises(AttributeError):
                ct.frequency
            with pytest.raises(AttributeError):
                ct.sideband

    def test_dechannelize_real_needs_n(self):
        # For real data, need to pass in `n`
        ct = Channelize(self.fh_freq, self.n)
        with pytest.raises(ValueError):
            Dechannelize(ct, dtype=self.fh_freq.dtype)


class TestChannelizeComplex(UseDADASample):
    def setup(self):
        super().setup()
        self.n = 1024
        # Add frequency information by hand for now.
        self.fh_freq = SetAttribute(
            self.fh,
            frequency=self.fh.header0['FREQ']*u.MHz,
            sideband=np.where(self.fh.header0.sideband, 1, -1))

    def test_channelize_frequency_complex(self):
        """Test frequency calculation."""
        fh = self.fh_freq
        ct = Channelize(fh, self.n)
        ref_frequency = (320. * u.MHz
                         + np.fft.fftfreq(self.n, 1. / fh.sample_rate))
        assert np.all(ct.sideband == fh.sideband)
        assert np.all(ct.frequency == ref_frequency[:, np.newaxis])

        fh = SetAttribute(self.fh, frequency=self.fh_freq.frequency,
                          sideband=-self.fh_freq.sideband)
        ct = Channelize(fh, self.n)
        ref_frequency = (320. * u.MHz
                         - np.fft.fftfreq(self.n, 1. / fh.sample_rate))
        assert np.all(ct.sideband == fh.sideband)
        assert np.all(ct.frequency == ref_frequency[:, np.newaxis])

    def test_dechannelizetask_complex(self):
        """Test dechannelization round-tripping."""
        fh = self.fh_freq
        raw_data = fh.read()
        ct = Channelize(fh, self.n)
        dt = Dechannelize(ct)
        nrec = (fh.shape[0] // self.n) * self.n
        assert dt.shape == (nrec,) + fh.shape[1:]
        data = dt.read()
        # Note: round-trip is not perfect due to rounding errors.
        assert np.allclose(data, raw_data[:nrec], atol=1.e-5)
        assert np.all(dt.frequency == fh.frequency)
        assert np.all(dt.sideband == fh.sideband)
        # Check class method
        dt2 = ct.inverse(ct)
        data2 = dt2.read()
        assert np.all(data2 == data)
        # Check inverse inverse as well.
        ct2 = dt2.inverse(fh)
        ct.seek(0)
        ft = ct.read()
        ft2 = ct2.read()
        assert np.all(ft == ft2)
        dt2.close()
