# Licensed under the GPLv3 - see LICENSE
"""Tests of the various property setters of PSRFITS HDUs."""

import os

import pytest
import numpy as np
from astropy.coordinates import Latitude, Longitude
from astropy import units as u

from ... import psrfits
from ..hdu import PSRFITSPrimaryHDU, SubintHDU, PSRSubintHDU

test_data = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')


class TestPrimaryHDUSetter:
    def setup(self):
        self.fold_data = os.path.join(test_data,
                                      "B1855+09.430.PUPPI.11y.x.sum.sm")
        self.reader = psrfits.open(self.fold_data, 'r', weighted=False)
        self.input_p_hdu = self.reader.ih.primary_hdu
        # init Primary
        self.p_hdu = PSRFITSPrimaryHDU()

    def teardown(self):
        self.reader.close()

    def test_set_location(self):
        self.p_hdu.location = self.input_p_hdu.location
        assert self.p_hdu.header['ANT_X'] == self.input_p_hdu.header['ANT_X']
        assert self.p_hdu.header['ANT_Y'] == self.input_p_hdu.header['ANT_Y']
        assert self.p_hdu.header['ANT_Z'] == self.input_p_hdu.header['ANT_Z']

    def test_set_telescope(self):
        self.p_hdu.telescope = self.input_p_hdu.telescope
        assert (self.p_hdu.header['TELESCOP']
                == self.input_p_hdu.header['TELESCOP'])

    def test_set_start_time(self):
        self.p_hdu.start_time = self.input_p_hdu.start_time
        assert np.isclose(int(self.p_hdu.header['STT_IMJD']),
                          self.input_p_hdu.header['STT_IMJD'])
        assert np.isclose(int(self.p_hdu.header['STT_SMJD']),
                          self.input_p_hdu.header['STT_SMJD'])
        assert np.isclose(float(self.p_hdu.header['STT_OFFS']),
                          self.input_p_hdu.header['STT_OFFS'])
        assert self.p_hdu.header['DATE-OBS'].startswith(
            self.input_p_hdu.header['DATE-OBS'])

    def test_set_freq(self):
        self.p_hdu.frequency = self.input_p_hdu.frequency
        assert (self.p_hdu.header['OBSNCHAN']
                == self.input_p_hdu.header['OBSNCHAN'])
        assert (self.p_hdu.header['OBSFREQ']
                == self.input_p_hdu.header['OBSFREQ'])
        assert (self.p_hdu.header['OBSBW']
                == self.input_p_hdu.header['OBSBW'])

    def test_set_sideband(self):
        self.p_hdu.header['OBSBW'] = self.input_p_hdu.header['OBSBW']
        self.p_hdu.sideband = -self.input_p_hdu.sideband
        assert (self.p_hdu.header['OBSBW'] ==
                -self.input_p_hdu.header['OBSBW'])
        self.p_hdu.sideband = self.input_p_hdu.sideband
        assert (self.p_hdu.header['OBSBW']
                == self.input_p_hdu.header['OBSBW'])

    def test_set_mode(self):
        self.p_hdu.obs_mode = 'PSR'
        assert self.p_hdu.header['OBS_MODE'] == 'PSR'
        with pytest.raises(AssertionError):
            self.p_hdu.obs_mode = 'BASEBAND'

    def test_set_skycoord(self):
        self.p_hdu.ra = self.input_p_hdu.ra
        self.p_hdu.dec = self.input_p_hdu.dec
        assert (Longitude(self.p_hdu.header['RA'], unit=u.hourangle)
                == Longitude(self.input_p_hdu.header['RA'], unit=u.hourangle))
        assert (Latitude(self.p_hdu.header['DEC'], unit=u.deg)
                == Latitude(self.input_p_hdu.header['DEC'], unit=u.deg))


class TestPSRHDUSetter(TestPrimaryHDUSetter):
    def setup(self):
        super().setup()
        # Create SUBINT using primary header.
        self.psr_hdu_no_shape = SubintHDU(primary_hdu=self.input_p_hdu)
        self.psr_hdu = SubintHDU(primary_hdu=self.input_p_hdu)
        self.psr_hdu.nrow = 1
        self.psr_hdu.sample_shape = self.reader.sample_shape
        # Since this test only have one channel, we will not test this setting
        # here.
        self.psr_hdu.header['CHAN_BW'] = self.reader.ih.header['CHAN_BW']

    def test_mode(self):
        assert (self.psr_hdu.mode == 'PSR')
        assert isinstance(self.psr_hdu, PSRSubintHDU)

    def test_init_data(self):
        # The data should be initialied in the setup.
        assert self.psr_hdu.data['DATA'].shape == (
            (self.reader.shape[0],) + self.reader.shape[1:][::-1])

    def test_no_sample_init(self):
        # Without sample shape set, we cannot get shape or data information.
        with pytest.raises(AttributeError):
            self.psr_hdu_no_shape.nbin
        with pytest.raises(AttributeError):
            self.psr_hdu_no_shape.shape
        with pytest.raises(AttributeError):
            self.psr_hdu_no_shape.data

    def test_set_nrow(self):
        assert self.psr_hdu.nbin == self.reader.ih.nbin

    def test_set_nchan(self):
        assert self.psr_hdu.nchan == self.reader.ih.nchan

    def test_set_npol(self):
        assert self.psr_hdu.npol == self.reader.ih.npol

    def test_set_nbin(self):
        assert self.psr_hdu.nbin == self.reader.ih.nbin

    def test_shape(self):
        assert self.psr_hdu.shape == self.reader.ih.shape
        # Since this was not explicitly set above, check we can change it.
        self.psr_hdu.shape = (10, 11, 12, 13)
        assert self.psr_hdu.shape == (10, 11, 12, 13)

    def test_set_start_time(self):
        self.psr_hdu.start_time = self.reader.start_time
        assert np.abs(self.psr_hdu.start_time
                      - self.reader.start_time) < 1 * u.ns

    def test_set_frequency(self):
        self.psr_hdu.frequency = self.reader.frequency
        assert self.psr_hdu.data['DAT_FREQ'] == self.reader.ih.data['DAT_FREQ']

    def test_data_column_writing(self, tmpdir):
        # Only test setting array here. no real scaling and offseting
        # calculation.
        test_fits = str(tmpdir.join('test_column_writing.fits'))
        test_data = self.reader.read(1)
        # PSRFITS rounds, not truncates data.
        in_data = np.around(((test_data - self.reader.ih.data['DAT_OFFS'])
                             / self.reader.ih.data['DAT_SCL'])).reshape(
            self.psr_hdu.data['DATA'].shape)
        self.psr_hdu.data['DATA'] = in_data
        self.psr_hdu.data['DAT_SCL'] = self.reader.ih.data['DAT_SCL']
        self.psr_hdu.data['DAT_OFFS'] = self.reader.ih.data['DAT_OFFS']
        # Implicitly set 'TSUBINT' and 'OFFS_SUB'.
        self.psr_hdu.sample_rate = self.reader.sample_rate
        self.psr_hdu.start_time = self.reader.start_time
        # Write to FITS file.
        hdul = self.psr_hdu.get_hdu_list()
        hdul.writeto(test_fits)
        # Re-open FITS file and check contents are the same.
        with psrfits.open(test_fits, 'r', weighted=False) as column_reader:
            assert np.array_equal(self.reader.ih.data['DATA'],
                                  column_reader.ih.data['DATA'])
            assert np.abs(column_reader.start_time
                          - self.reader.start_time) < 1 * u.ns
            assert u.isclose(column_reader.sample_rate,
                             self.reader.sample_rate)
            # And read from it, checking the output is the same as well.
            new_data = column_reader.read(1)
        assert np.array_equal(test_data, new_data)
