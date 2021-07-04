# Licensed under the GPLv3 - see LICENSE
"""Tests of the polyphase filterbank module."""
import os

import pytest
import numpy as np
from numpy.testing import assert_allclose

import astropy.units as u
from astropy.time import Time

from baseband_tasks.base import Task
from baseband_tasks.generators import NoiseGenerator
from baseband_tasks.pfb import (
    sinc_hamming, PolyphaseFilterBankSamples, PolyphaseFilterBank,
    InversePolyphaseFilterBank)


test_data = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')


def digitize(ft, level):
    return np.round(ft.view(float) / level).view(complex) * level


class TestSincHamming:
    def setup(self):
        a = np.loadtxt(os.path.join(
            test_data, 'bGDSP_U1_0032_T12_W095_get_pfb_coeffs.txt'))
        self.guppi_ppf = a.reshape(8, -1).T.reshape(12, 64)

    def test_guppi(self):
        model = sinc_hamming(12, 64, sinc_scale=0.95)
        assert_allclose(model, self.guppi_ppf)


class TestBasics:
    def setup(self):
        self.nh = NoiseGenerator(shape=(2500 * 2048,),
                                 start_time=Time('2010-01-01'),
                                 sample_rate=1.*u.kHz, seed=12345,
                                 samples_per_frame=128, dtype='f8')
        self.nc = NoiseGenerator(shape=(2500 * 2048,),
                                 start_time=Time('2010-01-01'),
                                 sample_rate=1.*u.kHz, seed=12345,
                                 samples_per_frame=128, dtype='c16')
        self.n_tap = 4
        self.n_chan = 2048
        self.chime_pfb = sinc_hamming(4, 2048)  # CHIME PFB
        self.guppi_pfb = sinc_hamming(12, 64, sinc_scale=0.95)

    @pytest.mark.parametrize('offset', (0, 1000))
    def test_understanding(self, offset):
        """Stepping or frequency selection should give same answer.

        Often think of PFB as multiplying with an array that is, e.g.,
        4 times larger, then FFTing and taking every 4th frequency.

        But this is equivalent to, after multiplication, summing 4
        subsequent arrays and taking the FT of that.
        """
        self.nh.seek(offset * 2048)
        # First check for real data.
        d = self.nh.read(5 * 2048).reshape(-1, 2048)
        hd = self.chime_pfb * d[:4]
        ft1_hd = np.fft.rfft(hd.ravel())[::4]
        ft2_hd = np.fft.rfft(hd.sum(0))
        assert_allclose(ft1_hd, ft2_hd)

        # Check actual implementations.
        pfb = PolyphaseFilterBankSamples(self.nh, self.chime_pfb)
        pfb.seek(offset)
        ft_pfb = pfb.read(2)
        assert_allclose(ft_pfb[0], ft2_hd)
        assert_allclose(ft_pfb[1],
                        np.fft.rfft((self.chime_pfb * d[1:]).sum(0)))
        pfb2 = PolyphaseFilterBank(self.nh, self.chime_pfb)
        pfb2.seek(offset)
        ft_pfb2 = pfb2.read(2)
        assert_allclose(ft_pfb2, ft_pfb)

    @pytest.mark.parametrize('offset', (0, 1000))
    def test_understanding_complex(self, offset):
        # Check above holds for complex too.
        self.nc.seek(offset * 2048)
        c = self.nc.read(4 * 2048).reshape(-1, 2048)
        hc = self.chime_pfb * c
        ft1_hc = np.fft.fft(hc.ravel())[::4]
        ft2_hc = np.fft.fft(hc.sum(0))
        assert_allclose(ft1_hc, ft2_hc)

        # And check actual implementation.
        pfb = PolyphaseFilterBankSamples(self.nc, self.chime_pfb)
        pfb.seek(offset)
        ft_pfb = pfb.read(1)[0]
        assert_allclose(ft_pfb, ft2_hc)
        pfb2 = PolyphaseFilterBank(self.nc, self.chime_pfb)
        pfb2.seek(offset)
        ft_pfb2 = pfb2.read(1)[0]
        assert_allclose(ft_pfb2, ft2_hc)

    def test_inversion_understanding(self):
        # From simulations, thresh = 0.05 is about right for no rounding
        # with n_sample=128 (it is 0.03 for n_sample=1024).
        n_sample = 128
        self.nh.seek(3 * 2048)
        d_in = self.nh.read(n_sample * 2048).reshape(-1, 2048)
        pfb = PolyphaseFilterBank(self.nh, self.chime_pfb,
                                  samples_per_frame=n_sample)
        ft_pfb = pfb.read(n_sample+3)
        # Dechannelize.
        d_pfb = np.fft.irfft(ft_pfb, axis=1)
        # Deconvolve.
        ft_fine = np.fft.rfft(d_pfb, axis=0)
        ft_resp = pfb._ft_response_conj
        ft_dec = ft_fine / ft_resp
        d_out = np.fft.irfft(ft_dec, axis=0, n=d_pfb.shape[0])
        d_out = d_out[3:]
        # We cannot hope to deconvolve near the edges and the PFB is such
        # that we loose the middle samples.
        assert_allclose(d_in[32:-32, :900], d_out[32:-32, :900], atol=0.001)
        assert_allclose(d_in[32:-32, 1150:], d_out[32:-32, 1150:], atol=0.001)
        # We can do better by Wiener filtering.  For no digitization noise,
        # threshold=(d_in-d_out)[32:-32].var()~0.01 seems roughly right.
        threshold = 0.05
        inverse = (ft_resp.conj() / (threshold**2+np.abs(ft_resp)**2)
                   * (1 + threshold**2))
        ft_dec2 = ft_fine * inverse
        d_out2 = np.fft.irfft(ft_dec2, axis=0, n=d_pfb.shape[0])
        d_out2 = d_out2[3:]
        # Cannot help near the edges, but middle elements are now better.
        assert_allclose(d_in[32:-32], d_out2[32:-32], atol=0.3)

    def test_inversion_understanding_digitization(self):
        # From simulations, thresh = 0.1 is about right for rounding with
        # levels at S/3 -> S/N ~ 10.
        n_sample = 128
        self.nh.seek(3 * 2048)
        d_in = self.nh.read(n_sample * 2048).reshape(-1, 2048)

        # Check effect of digitization.
        ft_check = np.fft.rfft(d_in, axis=1)
        ft_check_dig = digitize(ft_check, ft_check.real.std() / 3.)
        d_check = np.fft.irfft(ft_check_dig, axis=1, n=2048)
        assert np.isclose((d_check-d_in).std(), 0.1, atol=0.005)

        pfb = PolyphaseFilterBank(self.nh, self.chime_pfb,
                                  samples_per_frame=n_sample)
        ft_pfb = pfb.read(n_sample+3)
        ft_pfb_level = ft_pfb.real.std() / 3.
        ft_pfb_dig = (np.round(ft_pfb.view(float) / ft_pfb_level).view(complex)
                      * ft_pfb_level)
        d_pfb = np.fft.irfft(ft_pfb_dig, axis=1)
        # Deconvolve.
        ft_fine = np.fft.rfft(d_pfb, axis=0)
        ft_resp = pfb._ft_response_conj
        threshold = 0.1
        inverse = (ft_resp.conj() / (threshold**2+np.abs(ft_resp)**2)
                   * (1 + threshold**2))
        ft_dec = ft_fine * inverse
        d_out = np.fft.irfft(ft_dec, axis=0, n=d_pfb.shape[0])
        d_out = d_out[3:]
        assert np.isclose((d_out-d_in)[32:-32].std(), 0.125, atol=0.01)
        # Still get noisier data near middle, of course, but recover
        # to within 1 sigma of input noise signal.
        assert_allclose(d_in[32:-32], d_out[32:-32], atol=1)

    def test_inversion_chime_pfb(self):
        # Now test the same, but with the actual inversion class.
        # Here, we do not give samples_per_frame for the PFB, since we do
        # not need its FT (and it is exact for any value).
        n_sample = 128
        pad = 48
        self.nh.seek(pad * 2048 + 3 * 2048 // 2)
        d_in = self.nh.read(n_sample * 2048).reshape(-1, 2048)
        pfb = PolyphaseFilterBank(self.nh, self.chime_pfb)
        ipfb = InversePolyphaseFilterBank(
            pfb, self.chime_pfb, sn=100, pad_start=pad, pad_end=pad,
            samples_per_frame=n_sample*2048, dtype=self.nh.dtype)
        d_out = ipfb.read(n_sample * 2048).reshape(-1, 2048)
        assert_allclose(d_in[:, 50:-50], d_out[:, 50:-50], atol=0.01)

    def test_inversion_chime_pfb_digitized(self):
        # Now test the same, but with the actual inversion class.
        # Here, we do not give samples_per_frame for the PFB, since we do
        # not need its FT (and it is exact for any value).
        n_sample = 128
        pad = 32
        self.nh.seek(pad * 2048 + 3 * 2048 // 2)
        d_in = self.nh.read(n_sample * 2048).reshape(-1, 2048)
        pfb = PolyphaseFilterBank(self.nh, self.chime_pfb)
        dig_level = pfb.read(n_sample).real.std() / 3.
        pfb_dig = Task(pfb, task=lambda ft: digitize(ft, dig_level),
                       samples_per_frame=n_sample)
        ipfb = InversePolyphaseFilterBank(
            pfb_dig, self.chime_pfb, sn=10, pad_start=pad, pad_end=pad,
            samples_per_frame=n_sample*2048, dtype=self.nh.dtype)
        d_out = ipfb.read(n_sample * 2048).reshape(-1, 2048)
        assert np.isclose((d_out-d_in).std(), 0.125, atol=0.01)
        assert_allclose(d_in, d_out, atol=1.1)

    def test_inversion_guppi_pfb(self):
        n_sample = 512
        pad = 128
        self.nh.seek(pad * 64 + 11 * 64 // 2)
        d_in = self.nh.read(n_sample * 64).reshape(-1, 64)
        pfb = PolyphaseFilterBank(self.nh, self.guppi_pfb)
        ipfb = InversePolyphaseFilterBank(
            pfb, self.guppi_pfb, sn=30, pad_start=pad, pad_end=pad,
            samples_per_frame=n_sample*64, dtype=self.nh.dtype)
        d_out = ipfb.read(n_sample * 64).reshape(-1, 64)
        # Note that the PFB cuts off the channel edges so badly that
        # it is not possible to reproduce the original well.
        assert_allclose(d_in, d_out, atol=0.15)
        # It is almost exclusively the edge samples that are bad.
        ipfb2 = InversePolyphaseFilterBank(
            pfb, self.guppi_pfb, sn=1e9, pad_start=pad, pad_end=pad,
            samples_per_frame=n_sample*64, dtype=self.nh.dtype)
        d_out2 = ipfb2.read(n_sample * 64).reshape(-1, 64)
        assert_allclose(d_in[:, 2:-2], d_out2[:, 2:-2], atol=0.005)

    def test_inversion_guppi_pfb_digitized(self):
        n_sample = 512
        pad = 128
        self.nh.seek(pad * 64 + 11 * 64 // 2)
        d_in = self.nh.read(n_sample * 64).reshape(-1, 64)
        pfb = PolyphaseFilterBank(self.nh, self.guppi_pfb)
        dig_level = pfb.read(n_sample).real.std() / 30.
        pfb_dig = Task(pfb, task=lambda ft: digitize(ft, dig_level),
                       samples_per_frame=n_sample)
        ipfb = InversePolyphaseFilterBank(
            pfb_dig, self.guppi_pfb, sn=30, pad_start=pad, pad_end=pad,
            samples_per_frame=n_sample*64, dtype=self.nh.dtype)
        d_out = ipfb.read(n_sample * 64).reshape(-1, 64)
        # Not much effect of digitization since it introduces little noise.
        assert_allclose(d_in, d_out, atol=0.15)
