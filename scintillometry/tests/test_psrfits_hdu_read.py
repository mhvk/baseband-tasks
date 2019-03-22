# Licensed under the GPLv3 - see LICENSE
"""Full-package tests of psrfits reading routines."""

import pytest
import numpy as np
import os
import astropy.units as u
from astropy.time import Time

from ..io import open

test_data = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')


class TestRead:
    def setup(self):
        self.fold_data = os.path.join(test_data, "B1855+09.430.PUPPI.11y.x.sum."
                                                 "sm")
        self.psrchive_fold = os.path.join(test_data, "B1855_nano.npz")
        self.search_data = None


class TestFoldRead(TestRead):
    def setup(self):
        super().setup()
        self.reader = open(self.fold_data, weighted=False)
        self.psrchive_res = np.load(self.psrchive_fold)

    def test_start_time(self):
        # Header start time
        start_time0 = self.reader.fh_raw.primary_hdu.start_time
        mjdi = int(start_time0.mjd)
        mjdf = start_time0 - Time(mjdi, 0.0, format='mjd')
        sec_frac, sec_int = np.modf(mjdf.to(u.s).value)
        assert np.isclose(self.reader.fh_raw.primary_hdu.header['STT_IMJD'], mjdi), \
            "The header HDU start time's integer MJD is not reading correctly."
        assert np.isclose(self.reader.fh_raw.primary_hdu.header['STT_SMJD'], sec_int), \
            ("The header HDU start time's integer second is not reading"
             "correctly.")
        assert np.isclose(self.reader.fh_raw.primary_hdu.header['STT_OFFS'], sec_frac), \
            ("The header HDU start time's fractional second is not reading"
             "correctly.")
        # Subint start time
        start_time1 = self.reader.start_time
        psrchive_time = self.psrchive_res['t'][0]
        assert np.isclose(start_time1.mjd, psrchive_time), \
            "Subint start time did not read correctly."

    def test_read(self):
        # Test subint shape
        shape1 = self.reader.shape
        assert shape1 == (1, 2048, 1, 1), \
            "Fold data shape did not read correctly."
        # Test read data shape
        fold_data = self.reader.read(1)
        assert fold_data.shape == (1,) + self.reader.sample_shape, \
            "The result data shape does not match the header reported shape."

        # Test aganist psrchive result
        psrdata = self.psrchive_res['data']
        assert psrdata.shape == self.reader.fh_raw.data_shape, \
            "Data shape is not the same with PSRCHIVE result"
        psrdata = psrdata.reshape(shape1)
        assert np.all(np.isclose(psrdata, fold_data)), \
            "Fold mode data does not match the PSRCHIVE result data."
