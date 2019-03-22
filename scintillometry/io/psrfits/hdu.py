"""hdu.py defines the object for PSRFTIS header-data-units(HDUs).
"""


from collections import namedtuple
import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle, Latitude, Longitude
from astropy.io import fits
from astropy.utils import lazyproperty
import numpy as np


__all__ = ["HDU_map", "PsrfitsPrimaryHDU", "SubintHDUBase", "PSRSubint"]


class PsrfitsPrimaryHDU(fits.PrimaryHDU):
    """Helper class to translate between FITS primary HDU and baseband-style
    file reader.

    Parameters
    ----------
    primary_hdu : `primary_hdu`
        PSRFITS primary HDU object.

    Notes
    -----
    the frequency marker is on the center of the channels
    """
    _properties = ('start_time', 'observatory', 'frequency', 'ra', 'dec',
                   'shape', 'sample_rate')

    def __init__(self, primary_hdu):
        super().__init__(header=primary_hdu.header, data=primary_hdu.data)
        self.verify()

    def verify(self):
        assert self.header['SIMPLE'], "The input HDU is not a fits headers HDU."
        assert self.header['FITSTYPE'] == "PSRFITS", \
            "The input fits header is not a PSRFITS type."

    @property
    def start_time(self):
        # TODO add location
        return (Time(self.header['STT_IMJD'], format='mjd', precision=9) +
                TimeDelta(self.header['STT_SMJD'], self.header['STT_OFFS'],
                          format='sec', scale='tai'))

    @property
    def observatory(self):
        return self.header['TELESCOP']

    @property
    def frequency(self):
        try:
            n_chan = float(self.header['OBSNCHAN'])
            c_chan = float(self.header['OBSFREQ'])
            bw = float(self.header['OBSBW'])
        except Exception:
            return None
        chan_bw = bw / n_chan
        freq = np.arange(-n_chan / 2, n_chan / 2) * chan_bw + c_chan
        return u.Quantity(freq, u.MHz, copy=False)

    @property
    def ra(self):
        return Longitude(self.header['RA'], unit=u.hourangle)

    @property
    def dec(self):
        return Latitude(self.header['DEC'], unit=u.deg)

    @property
    def obs_mode(self):
        return self.header['OBS_MODE']


class SubintHDUBase(fits.BinTableHDU):
    """SubintHDU class provides the translator functions between baseband-style
    file object and the PSRFITS SUBINT HDU.

    Parameters
    ----------
    primary_hdu : PsrfitsPrimaryHDU
        The psrfits main header object
    subint_hdu : HDU object
        The psrfits data HDU.
    verify: bool, optional
        Does the hdu need to be verified? Default is True.

    Notes
    -----
    Right now we are assuming the data rows are continuous in time and the
    frequency are the same.
    """

    _properties = ('start_time', 'sample_rate', 'shape', 'samples_per_frame',
                   'polarization', 'frequency')

    def __new__(cls, primary_hdu, subint_hdu=None, verify=True):
        # Map Subint subclasses
        mode = primary_hdu.obs_mode
        try:
            cls = subint_map[mode]
        except KeyError:
            raise ValueError("'{}' is not a valid mode.".format(mode))
        return super(SubintHDUBase, cls).__new__(cls)

    def __init__(self, primary_hdu, subint_hdu=None, verify=True):
        self.primary_hdu = primary_hdu
        super().__init__(header=subint_hdu.header, data=subint_hdu.data)
        if verify:
            self.verify()
        self.offset = 0

    def verify(self):
        assert self.header['EXTNAME'].strip() == "SUBINT", \
            "Input HDU is not a SUBINT type."

    @property
    def mode(self):
        return self.primary_hdu.obs_mode

    @property
    def start_time(self):
        # NOTE should we get the start time for each raw, in case the time gaps
        # in between the rows
        file_start = self.primary_hdu.start_time
        return file_start

    @property
    def nrow(self):
        return self.header['NAXIS2']

    @property
    def nchan(self):
        return self.header['NCHAN']

    @property
    def npol(self):
        return self.header['NPOL']

    @property
    def nbin(self):
        return self.header['NBIN']

    @property
    def shape(self):
        raw_shape = self.raw_shape
        new_shape = namedtuple('shape', ['nsample', 'nbin', 'nchan', 'npol'])
        result = new_shape(raw_shape.nrow * raw_shape.samples_per_frame,
                           raw_shape.nbin, raw_shape.nchan, raw_shape.npol)
        return result

    @property
    def raw_shape(self):
        r_shape = namedtuple('shape', ['nrow', 'samples_per_frame', 'nbin',
                                       'nchan', 'npol'])
        result = r_shape(self.nrow, self.samples_per_frame, self.nbin,
                         self.nchan, self.npol)
        return result

    @property
    def polarization(self):
        pol_len = int(len(self.header['POL_TYPE']) / self.npol)
        return map(''.join, zip(*[iter(self.header['POL_TYPE'])] * pol_len))

    @property
    def frequency(self):
        if 'DAT_FREQ' in self.columns.names:
            freqs = u.Quantity(self.data['DAT_FREQ'],
                               u.MHz, copy=False)[0]
        else:
            freqs = getattr(self.primary_hdu, 'frequency', None)
        if freqs is not None:
            freqs_shape = self.shape._replace(npol=1, nbin=1)[1:]
            freqs = freqs.reshape(freqs_shape)
        return freqs

    @property
    def dtype(self):
        """
        Notes
        -----
        PSRFITS subint defines the DAT_SCL using float, thus we choose the
        highest precision.
        """

        return self.data['DAT_SCL'].dtype

    def read_data_row(self, row_index, weighted=False):
        if row_index >= self.shape[0]:
            raise EOFError("cannot read from beyond end of input SUBINT HDU.")

        row = self.data[row_index]
        # Reversed the header shape to match the data
        new_shape = self.raw_shape._replace(samples_per_frame=1,
                                            nbin=1)[-1:1:-1]
        data_scale = row['DAT_SCL'].reshape(new_shape)
        data_off_set = row['DAT_OFFS'].reshape(new_shape)
        try:
            zero_off = float(self.header['ZERO_OFF'])
        except Exception:
            zero_off = 0.0
        result = (row['DATA'] - zero_off) * data_scale + data_off_set
        if 'DAT_WTS' in self.columns.names and weighted:
            data_wts = row['DAT_WTS']
            wts_shape = self.raw_shape._replace(samples_per_frame=1, nbin=1,
                                                npol=1)[-1:1:-1]
            result *= data_wts.reshape(wts_shape)
        return result


class PSRSubint(SubintHDUBase):
    """PSRSubint class is designed for handling the pulsar folding mode PSRFITS
    Subint HDU.

    Parameters
    ----------
    primary_hdu : PsrfitsHearderHDU
        The psrfits main header object
    psr_subint : HDU object
        The psrfits subint HDU.
    """
    def __init__(self, primary_hdu, psr_subint=None, verify=True):
        super().__init__(primary_hdu, psr_subint, verify=verify)

    def verify(self):
        super().verify()
        assert self.primary_hdu.obs_mode.upper() == 'PSR', \
            "Header HDU is not in the folding mode."

        assert int(self.header['NBIN']) > 1, \
            ("Folding mode requires a valid 'NBIN' field ('NBIN' > 1) in the"
             " header.")

        # Check frequency
        if 'DAT_FREQ' in self.columns.names:
            freqs = u.Quantity(self.data['DAT_FREQ'],
                               u.MHz, copy=False)
            assert np.array_equiv(freqs[0], freqs), \
                "Frequencies are different within one subint rows."

        tsubint = self.data['TSUBINT']
        assert all(np.isclose(tsubint[0], tsubint, atol=1e-1)), \
            "Subints' durations has big difference between each other."

        d_shape_raw = self.data['DATA'].shape
        d_shape_header = (self.nbin, self.nchan, self.npol)
        assert d_shape_raw == (self.nrow, ) + d_shape_header[::-1], \
            "Data shape does not match with the header information."

    @property
    def start_time(self):
        """Start_time returns start time of the first sub-integration.

        Notes
        -----
        The start time is accurate to one pulse period. This calculation below
        is consistent with PSRCHIVE's definition
        (defined in psrchive/Base/Classes/Integration.C)
        """
        file_start = self.primary_hdu.start_time
        if "OFFS_SUB" in self.columns.names:
            subint_times = u.Quantity(self.data['OFFS_SUB'], u.s, copy=False)
            start_time = (file_start + subint_times[0] -
                          self.samples_per_frame / self.sample_rate / 2)
        else:
            start_time = file_start
        return start_time

    @lazyproperty
    def samples_per_frame(self):
        return int(np.prod(self.data_shape)/(self.nrow * self.nchan *
                                             self.npol * self.nbin))

    @property
    def sample_rate(self):
        # NOTE we are assuming TSUBINT is uniform.
        sample_time = u.Quantity(self.data[0]['TSUBINT'] /
                                 self.samples_per_frame, u.s)
        return 1.0 / sample_time

    @property
    def data_shape(self):
        # Data are save in the FORTRAN order. Reversed from the header label.
        d_shape = namedtuple('d_shape', ['nrow', 'npol', 'nchan', 'nbin'])
        result = d_shape(self.nrow, self.npol, self.nchan, self.nbin)
        return result


HDU_map = {'PRIMARY': PsrfitsPrimaryHDU,
           'SUBINT': SubintHDUBase}

subint_map = {'PSR': PSRSubint, }
