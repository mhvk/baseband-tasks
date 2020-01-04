# Licensed under the GPLv3 - see LICENSE
"""Full-package tests of psrfits writer routines."""

import os

import pytest
from astropy import units as u

from ... import psrfits
from ..hdu import PSRFITSPrimaryHDU, PSRSubintHDU

test_data = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')


class TestWriter:
    def setup(self):
        self.update_file = os.path.join(test_data, "write_update.fits")
        self.fold_data = os.path.join(test_data,
                                      "B1855+09.430.PUPPI.11y.x.sum.sm")

    def test_empty_open(self):
        empty_primary = PSRFITSPrimaryHDU()
        empty_primary.obs_mode = 'PSR'
        writer = psrfits.open("test_empty.fits", "w",
                              primary_hdu=empty_primary)
        assert isinstance(writer.hdu, PSRSubintHDU)
        with pytest.raises(AttributeError):
            writer.hdu.data

    def test_open_kwarg(self):
        empty_primary = PSRFITSPrimaryHDU()
        empty_primary.obs_mode = 'PSR'
        writer = psrfits.open("test_empty.fits", "w",
                              primary_hdu=empty_primary,
                              shape=(50, 20, 32, 2),
                              sample_rate=1.0 / u.s)
        assert isinstance(writer.hdu, PSRSubintHDU)
        assert writer.hdu.shape == (50, 20, 32, 2)
        assert writer.hdu.sample_rate == 1.0 / u.s
        assert writer.hdu.data.shape == (50,)

    def test_open_raise(self):
        with pytest.raises(ValueError):
            psrfits.open("test_raise.fits", "w")

        with pytest.raises(KeyError):
            no_mode_primary = PSRFITSPrimaryHDU()
            psrfits.open("test_raise2.fits", "w",
                         primary_hdu=no_mode_primary)

        with pytest.raises(AssertionError):
            wrong_mode_primary = PSRFITSPrimaryHDU()
            wrong_mode_primary.obs_mode = 'WRONG'
            psrfits.open("test_raise3.fits", "w",
                         primary_hdu=no_mode_primary)
