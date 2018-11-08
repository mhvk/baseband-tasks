# Licensed under the GPLv3 - see LICENSE

import numpy as np
import astropy.units as u
import pytest

from ..channelize import Channelize, Dechannelize
from ..fourier import get_fft_maker

from baseband import vdif, dada
from baseband.data import SAMPLE_VDIF, SAMPLE_DADA


class TestChannelize:
    """Test channelization using Baseband's sample VDIF file."""

    def setup(self):
        """Pre-calculate channelized data."""
        self.n = 1024

        with vdif.open(SAMPLE_VDIF) as fh:
            self.ref_start_time = fh.start_time
            self.ref_sample_rate = fh.sample_rate
            data = fh.read()

        self.raw_data = data
        last_sample = self.n * (data.shape[0] // self.n)
        part = data[:last_sample].reshape((-1, self.n) + data.shape[1:])
        FFT = get_fft_maker()
        rfft = FFT(shape=part.shape, dtype=part.dtype, axis=1,
                   sample_rate=self.ref_sample_rate)
        self.ref_data = rfft(part)
        self.ref_sideband = np.tile([-1, 1], 4)
        self.ref_frequency = ((311.25 + 16 * (np.arange(8) // 2)) * u.MHz +
                              self.ref_sideband * rfft.frequency)

    def test_channelizetask(self):
        """Test channelization task."""

        fh = vdif.open(SAMPLE_VDIF)
        ct = Channelize(fh, self.n)

        # Channelize everything.
        data1 = ct.read()
        assert ct.tell() == ct.shape[0]
        assert (ct.time - ct.start_time -
                ct.shape[0] / ct.sample_rate) < 1*u.ns
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

        # Quick test of channel sanity check in __init__.
        with pytest.raises(AssertionError):
            ct = Channelize(fh, 400001)

        ct.close()

    def test_channelize_frequency_real(self):
        """Test frequency calculation."""

        fh = vdif.open(SAMPLE_VDIF)
        # Add frequency information by hand for now.
        fh.frequency = 311.25 * u.MHz + (np.arange(8.) // 2) * 16. * u.MHz
        # Note: sideband is actually incorrect for this VDIF file;
        # this is for testing only.
        fh.sideband = np.tile([-1, +1], 4)

        ct = Channelize(fh, self.n)

        assert np.all(ct.sideband == self.ref_sideband)
        assert np.all(ct.frequency == self.ref_frequency)

    def test_channelize_frequency_complex(self):
        """Test frequency calculation."""

        fh = dada.open(SAMPLE_DADA)
        # Add frequency information by hand for now.
        fh.frequency = fh.header0['FREQ'] * u.MHz
        fh.sideband = np.where(fh.header0.sideband, 1, -1)

        ct = Channelize(fh, self.n)

        ref_frequency = (320. * u.MHz +
                         np.fft.fftfreq(self.n, 1. / fh.sample_rate))
        assert np.all(ct.sideband == fh.sideband)
        assert np.all(ct.frequency == ref_frequency[:, np.newaxis])

        fh.sideband = -fh.sideband
        ct = Channelize(fh, self.n)
        ref_frequency = (320. * u.MHz -
                         np.fft.fftfreq(self.n, 1. / fh.sample_rate))
        assert np.all(ct.sideband == fh.sideband)
        assert np.all(ct.frequency == ref_frequency[:, np.newaxis])

    def test_dechannelizetask_real(self):
        """Test dechannelization round-tripping."""
        fh = vdif.open(SAMPLE_VDIF)
        fh.frequency = 311.25 * u.MHz + (np.arange(8.) // 2) * 16. * u.MHz
        fh.sideband = np.tile([-1, +1], 4)
        ct = Channelize(fh, self.n)
        dt = Dechannelize(ct, self.n, dtype=fh.dtype)
        nrec = (fh.shape[0] // self.n) * self.n
        assert dt.shape == (nrec,) + fh.shape[1:]
        data = dt.read()
        # Note: round-trip is not perfect due to rounding errors.
        assert np.allclose(data, self.raw_data[:nrec], atol=1.e-5)
        assert np.all(dt.frequency == fh.frequency)
        assert np.all(dt.sideband == fh.sideband)
        # Check class method
        dt2 = ct.inverse(ct)
        data2 = dt2.read()
        assert np.all(data2 == data)

    def test_dechannelizetask_complex(self):
        """Test dechannelization round-tripping."""
        fh = dada.open(SAMPLE_DADA)
        # Add frequency information by hand for now.
        fh.frequency = fh.header0['FREQ'] * u.MHz
        fh.sideband = np.where(fh.header0.sideband, 1, -1)
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

    def test_missing_frequency_sideband(self):
        fh = vdif.open(SAMPLE_VDIF)
        ct = Channelize(fh, self.n)
        with pytest.raises(AttributeError):
            ct.frequency
        with pytest.raises(AttributeError):
            ct.sideband
