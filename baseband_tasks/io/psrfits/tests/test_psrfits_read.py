# Licensed under the GPLv3 - see LICENSE
"""Full-package tests of psrfits reading routines."""

import os

import pytest
import numpy as np
import astropy.units as u
from astropy.coordinates import Longitude, Latitude, EarthLocation
from astropy.time import Time

from ... import psrfits

test_data = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')


class TestRead:
    def setup(self):
        self.fold_data = os.path.join(test_data,
                                      "B1855+09.430.PUPPI.11y.x.sum.sm")
        self.psrchive_fold = os.path.join(test_data, "B1855_nano.npz")


class TestFoldRead(TestRead):
    def setup(self):
        super().setup()
        self.reader = psrfits.open(self.fold_data, weighted=False)
        self.psrchive_res = np.load(self.psrchive_fold)

    def teardown(self):
        self.reader.close()

    def test_hdu_wrapper_properties(self):
        hdu = self.reader.ih
        assert hdu.mode == 'PSR'
        assert hdu.sample_shape == (2048, 1, 1)
        assert hdu.shape == (1, 2048, 1, 1)
        assert u.allclose(hdu.frequency, 433.12399292 * u.MHz)
        assert hdu.sideband == 1

        with pytest.raises(EOFError):
            hdu.read_data_row(hdu.nrow)

    def test_primary_hdu_properties(self):
        # Oddly, frequency in primary header is inconsistent with SUBINT;
        # here just testing that we return what the FITS file has.
        primary = self.reader.ih.primary_hdu
        assert primary.header['OBSNCHAN'] == 64
        assert np.allclose(primary.header['OBSFREQ'], 372.34375)
        # Channel 32 should have mid-frequency; given that channel 0
        # is assumed omitted, this is index 31.
        assert u.allclose(primary.frequency[31], 372.34375 * u.MHz)
        assert primary.sideband == 1
        ra = primary.ra
        assert isinstance(ra, Longitude)
        assert u.allclose(ra, Longitude(18.96010972, u.hourangle))
        dec = primary.dec
        assert isinstance(dec, Latitude)
        assert u.allclose(dec, Latitude(9.72148333, u.deg))
        assert primary.telescope == 'Arecibo'

    def test_reader_properties(self):
        hdu = self.reader.ih
        assert self.reader.sample_shape == hdu.sample_shape
        assert self.reader.shape == hdu.shape
        assert self.reader.frequency == hdu.frequency

    def test_start_time(self):
        # Header start time
        start_time0 = self.reader.ih.primary_hdu.start_time
        mjdi = int(start_time0.mjd)
        mjdf = start_time0 - Time(mjdi, 0.0, format='mjd')
        sec_frac, sec_int = np.modf(mjdf.to(u.s).value)
        assert self.reader.ih.primary_hdu.header['STT_IMJD'] == mjdi, \
            "HDU start time's integer MJD is not reading correctly."
        assert self.reader.ih.primary_hdu.header['STT_SMJD'] == sec_int, \
            "HDU start time's integer second is not reading correctly."
        assert np.isclose(self.reader.ih.primary_hdu.header['STT_OFFS'],
                          sec_frac), \
            "HDU start time's fractional second is not reading correctly."
        # Subint start time
        start_time1 = self.reader.start_time
        psrchive_time = self.psrchive_res['t'][0]
        assert np.isclose(start_time1.mjd, psrchive_time), \
            "Subint start time did not read correctly."

    def test_location(self):
        header = self.reader.ih.primary_hdu.header
        expected = EarthLocation(header['ANT_X'],
                                 header['ANT_Y'],
                                 header['ANT_Z'], u.m)
        assert self.reader.start_time.location == expected

    def test_read(self):
        # Test subint shape
        shape1 = self.reader.shape
        assert shape1 == (1, 2048, 1, 1), \
            "Fold data shape did not read correctly."
        # Test read data shape
        fold_data = self.reader.read(1)
        assert fold_data.shape == (1,) + self.reader.sample_shape, \
            "The result data shape does not match the header reported shape."

        # Test against psrchive result
        psrdata = self.psrchive_res['data']
        assert psrdata.shape == ((self.reader.shape[0],)
                                 + self.reader.sample_shape[::-1]), \
            "Data shape is not the same with PSRCHIVE result"
        psrdata = psrdata.reshape(shape1)
        assert np.all(np.isclose(psrdata, fold_data)), \
            "Fold mode data does not match the PSRCHIVE result data."

        with pytest.raises(EOFError):
            self.reader.read(1)

    def test_weighted_read(self):
        self.reader.seek(0)
        unweighted = self.reader.read(1)
        with psrfits.open(self.fold_data, weighted=True) as reader:
            weighted = reader.read(1)
            weights = self.reader.ih.hdu.data['DAT_WTS']
        assert np.all(weighted == unweighted * weights.reshape(-1, 1))
