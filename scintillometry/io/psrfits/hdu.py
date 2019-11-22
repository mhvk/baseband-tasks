# Licensed under the GPLv3 - see LICENSE
"""Wrappers for PSRFTIS Header Data Units (HDUs)."""
from collections import namedtuple

import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.coordinates import Angle, Latitude, Longitude
from astropy.io import fits
from astropy.io.fits.column import FITS2NUMPY
from astropy.io.fits.fitsrec import FITS_rec
from astropy.utils import lazyproperty
import numpy as np
import operator
from .psrfits_htm_parser import hdu_templates, column_map


__all__ = ["HDU_map", "HDUWrapper", "PSRFITSPrimaryHDU",
           "SubintHDU", "PSRSubintHDU"]


class HDUWrapper:
    def __init__(self, hdu=None, verify=True, hdu_type=None):
        if hdu is None:
            hdu, self.hdu_column = self.init_hdu(hdu_type)
            verify = False
        self.hdu = hdu
        if verify:
            self.verify()
        # This is for selecting the dimension for different mode (i.e., folding
        # or searching)
        self._dim_opt = 0

    def verify(self):
        assert isinstance(self.header, fits.Header)

    def init_hdu(self, hdu_name):
        target_hdu, hdu_parts = hdu_templates[hdu_name]
        hdu = target_hdu()
        hdu, hdu_parts = self._pre_process_hdu(hdu, hdu_parts)
        print(hdu_parts)
        # Init header
        # HDU part should always have cards.
        for card in hdu_parts['card']:
            hdu.header.set(card['name'], value=card['value'],
                           comment=card['comment'])
        # add comment cards
        # TODO comment need a little bit tuning
        if hdu_parts['comment'] != []:
            for cmt in hdu_parts['comment']:
                hdu.header.set(cmt['name'], value=' '.join(cmt['value'].split()),
                               after=cmt['after'])
        return hdu, hdu_parts['column']

    def _pre_process_hdu(self, hdu, hdu_parts):
        return hdu, hdu_parts

    def init_columns(self):
        cols = []
        for col in self.hdu_column:
            # init array
            # Translate dimensions.
            init_dim = tuple()
            for dim in col['col_dim'][self._dim_opt]:
                try:
                    init_dim += (int(dim),)
                except ValueError:
                    # TODO this will broken when handling something like
                    # (NCHAN,NPOL,NSBLK*NBITS/8)
                    init_dim += (getattr(self, dim.lower()),)

            np_dtype = FITS2NUMPY[col['dtype']]
            new_format = str(np.prod(init_dim)) + col['dtype']
            init_array = np.zeros(init_dim, dtype=np_dtype).reshape((1,) +
                                                                    init_dim)
            # TODO maybe we can use **kwarg to pass the parameters
            if col['dim'] is None:
                input_dim = None
            else:
                # Do we need reverse the order here?
                input_dim = str(init_dim)
            fits_col = fits.Column(name=col['name'][0], format=new_format,
                                   unit=col['unit'][0], dim=input_dim,
                                   array=init_array)
            cols.append(fits_col)
        self.hdu.data = FITS_rec.from_columns(cols)
        # add description on header
        for ii, col in enumerate(self.hdu_column):
            ttype_key = "TTYPE{}".format(ii + 1)
            # Check if the column cards are in the same order as the input.
            assert self.header[ttype_key] == col['name'][0], \
                "Card TTYPE{} is not for column {}.".format(ii + 1,
                                                            col['name'][0])
            for key_pattern, col_key in column_map.items():
                key = key_pattern + str(ii + 1)
                value = self.header.get(key, None)
                if value is not None and len(col[col_key]) >= 2:
                    self.header[key] = (value, col[col_key][1])

    @property
    def header(self):
        return self.hdu.header

    @property
    def data(self):
        return self.hdu.data

    def close(self):
        del self.hdu.data
        del self.hdu


class PSRFITSPrimaryHDU(HDUWrapper):
    """Wrapper for PSRFITS primary HDU, providing baseband-style properties.

    Parameters
    ----------
    hdu : `~astropy.io.fits.PrimaryHDU`
        PSRFITS primary HDU instance.

    Notes
    -----
    Frequencies are defined to be in the center of the channels.
    """

    _properties = ('location', 'start_time', 'observatory', 'frequency',
                   'ra', 'dec', 'shape', 'sample_rate')

    def __init__(self, hdu=None, verify=True):
        # When input hdu is None, the Primary header will be initialize as empty
        super().__init__(hdu, hdu_type="PRIMARY", verify=verify)

    def verify(self):
        assert self.header['SIMPLE'], "The HDU is not a FITS primary HDU."
        assert self.header['FITSTYPE'] == "PSRFITS", \
            "The header is not from a PSRFITS file."

    def _pre_process_hdu(self, hdu, hdu_parts):
        # Preprocess the header parts
        for card_name in ['SIMPLE', 'EXTEND']:
            # Those two fields need value to be bool
            for card in hdu_parts['card']:
                if card['name'] == card_name:
                    # value can only be True or False
                    if 'T' in card['value']:
                        card['value'] = True
                    else:
                        card['value'] = False
        return hdu, hdu_parts

    @property
    def location(self):
        try:
            return EarthLocation(self.header['ANT_X'],
                                 self.header['ANT_Y'],
                                 self.header['ANT_Z'], u.m)
        except (KeyError, TypeError):   # Sometimes PSRFITS uses '*' as no data.
            return None

    @location.setter
    def location(self, loc):
        """ Location setter. input should be an Astropy EarthLocation"""
        # TODO, for the space base observatory, Earth Location may not apply.
        self.hdu.header['ANT_X'] = loc.x.to_value(u.m)
        self.hdu.header['ANT_Y'] = loc.y.to_value(u.m)
        self.hdu.header['ANT_Z'] = loc.z.to_value(u.m)

    @property
    def start_time(self):
        return (Time(float(self.header['STT_IMJD']), format='mjd', precision=9,
                     location=self.location) +
                TimeDelta(float(self.header['STT_SMJD']),
                          float(self.header['STT_OFFS']),
                          format='sec', scale='tai'))

    @start_time.setter
    def start_time(self, time):
        """ Set the start_time, the input value should be an Time object"""
        # Should we allow time set location
        if time.location is not None:
            self.location = time.location
        mjd_int = int(time.mjd)
        mjd_frac = (time - Time(mjd_int, scale=time.scale, format='mjd'))
        frac_sec, int_sec = np.modf(mjd_frac.to(u.s).value)
        self.hdu.header['STT_IMJD'] = '{0:05d}'.format(mjd_int)
        self.hdu.header['STT_SMJD'] = '{}'.format(int(int_sec))
        self.hdu.header['STT_OFFS'] = '{0:17.15f}'.format(frac_sec)
        self.hdu.header['DATE-OBS'] = time.fits

    @property
    def telescope(self):
        return self.header['TELESCOP']

    @telescope.setter
    def telescope(self, value):
        self.hdu.header['TELESCOP'] = value

    @property
    def frequency(self):
        try:
            n_chan = int(self.header['OBSNCHAN'])
            c_freq = float(self.header['OBSFREQ'])
            bw = float(self.header['OBSBW'])
        except (KeyError, ValueError):
            return None

        chan_bw = bw / n_chan
        # According to the PSRFITS definition document, channels are
        # numbered 1 to n_chan, with the zeroth channel assumed removed
        # and c_freq is the frequency of channel n_nchan / 2.  We use
        # (n_chan + 1) // 2 to ensure this makes sense for n_chan = 1
        # and is consistent with the document at least for even n_chan.

        freq = c_freq + (np.arange(1, n_chan + 1) -
                         ((n_chan + 1) // 2)) * chan_bw
        return u.Quantity(freq, u.MHz, copy=False)

    @frequency.setter
    def frequency(self, freq):
        """Frequency setter. The input should be the frequency array."""
        freq = freq.to_value(u.MHz)
        n_chan = len(freq)
        # add the channel 0
        # we assume the frequency resolution is the same across the band.
        freq_pad = np.insert(freq, 0, 2 * freq[0] - freq[1])
        c_chan = freq_pad[((n_chan + 1) // 2)]
        bw = freq_pad.ptp()
        self.hdu.header['OBSNCHAN'] = n_chan
        self.hdu.header['OBSFREQ'] = c_chan
        self.hdu.header['OBSBW'] = bw

    @property
    def sideband(self):
        return np.where(self.hdu.header['OBSBW'] > 0, np.int8(1), np.int8(-1))

    @sideband.setter
    def sideband(self, sideband):
        assert np.all(np.abs(sideband) == 1), "sideband should be +/- 1"
        self.hdu.header['OBSBW'] = sideband * abs(self.hdu.header['OBSBW'])

    @property
    def ra(self):
        return Longitude(self.header['RA'], unit=u.hourangle)

    @ra.setter
    def ra(self, value):
        self.hdu.header['RA'] = value.to_string(sep=':', pad=True)

    @property
    def dec(self):
        return Latitude(self.header['DEC'], unit=u.deg)

    @dec.setter
    def dec(self, value):
        self.hdu.header['DEC'] = value.to_string(sep=':', alwayssign=True,
                                                 pad=True)

    @property
    def obs_mode(self):
        return self.header['OBS_MODE']

    @obs_mode.setter
    def obs_mode(self, value):
        assert value in {'PSR', 'CAL', 'SEARCH'}, \
            "obs_mode can only be 'PSR', 'CAL', or 'SEARCH'."
        self.hdu.header['OBS_MODE'] = value


class SubintHDU(HDUWrapper):
    """Base for PSRFITS SUBINT HDU wrappers.

    Parameters
    ----------
    hdu : `~astropy.io.fits.BinTableHDU` instance
        The PSRFITS table HDU of SUBINT type.
    primary : `~scintillometry.io.psrfits.PSRFITSPrimaryHDU`
        The wrapped PSRFITS main header.
    verify: bool, optional
        Whether to do basic verification.  Default is `True`.

    Notes
    -----
    Right now we are assuming the data rows are continuous in time and the
    frequencies do not vary.
    """

    _properties = ('start_time', 'sample_rate', 'sample_shape',
                   'shape', 'samples_per_frame',
                   'polarization', 'frequency')

    _sample_shape_maker = namedtuple('SampleShape', 'nbin, nchan, npol')
    _shape_maker = namedtuple('Shape', 'nsample, nbin, nchan, npol')

    def __new__(cls, hdu=None, primary_hdu=None, verify=True):
        # Map Subint subclasses;
        # TODO: switch to__init_subclass__ when we only support python>=3.6.
        try:
            mode = primary_hdu.obs_mode
            cls = subint_map[mode]
        except AttributeError:
            raise ValueError("need a primary HDU to determine the mode.")
        except KeyError:
            raise ValueError("'{}' is not a valid mode.".format(mode))

        return super().__new__(cls)

    def __init__(self, hdu=None, primary_hdu=None, verify=True):
        self.primary_hdu = primary_hdu
        self.offset = 0
        super().__init__(hdu, verify=verify, hdu_type='SUBINT')
        self.sample_label = ('nbin', 'nchan', 'npol')

    def verify(self):
        assert self.header['EXTNAME'].strip() == "SUBINT", \
            "Input HDU is not a SUBINT type."
        assert isinstance(self.primary_hdu, PSRFITSPrimaryHDU), \
            "Primary HDU needs to be a PSRFITSPrimaryHDU instance."

    def _pre_process_hdu(self, hdu, hdu_parts):
        #S UBINT need preprocess of the hdu cards value
        # HDU part should always have cards.
        for card in hdu_parts['card']:
            try:
                card['value'] = int(card['value'])
            except:
                pass
        return hdu, hdu_parts

    @property
    def mode(self):
        return self.primary_hdu.obs_mode

    @property
    def start_time(self):
        # Note: subclasses can use or override this.
        return self.primary_hdu.start_time

    @property
    def nrow(self):
        return self.header['NAXIS2']

    @nrow.setter
    def nrow(self, value):
        self.hdu.header['NAXIS2'] = operator.index(value)

    @property
    def nchan(self):
        return self.header['NCHAN']

    @nchan.setter
    def nchan(self, value):
        self.hdu.header['NCHAN'] = operator.index(value)

    @property
    def npol(self):
        return self.header['NPOL']

    @npol.setter
    def npol(self, value):
        self.hdu.header['NPOL'] = operator.index(value)

    @property
    def nbin(self):
        return self.header['NBIN']

    @nbin.setter
    def nbin(self, value):
        self.hdu.header['NBIN'] = operator.index(value)

    @property
    def sample_shape(self):
        return self._sample_shape_maker(self.nbin, self.nchan, self.npol)

    @sample_shape.setter
    def sample_shape(self, shape):
        self.nbin = shape[0]
        self.nchan = shape[1]
        self.npol = shape[2]

    @property
    def shape(self):
        return self._shape_maker(self.nrow * self.samples_per_frame,
                                 self.nbin, self.nchan, self.npol)

    @property
    def polarization(self):
        pol_type = self.header['POL_TYPE']
        # split into equal parts using zip;
        # see https://docs.python.org/3.5/library/functions.html#zip
        return np.array([map(''.join, zip(*[iter(self.header['POL_TYPE'])] *
                                          (len(pol_type) // self.npol)))])

    @polarization.setter
    def polarization(self, value):
        """ Setter for polarization labels.
        Parameter
        ---------
        value : array-like
            The names of polarization
        """
        # check if the input value length matches the npol
        assert len(value) == self.npol, \
            ("The input polarization name does not match the number of"
             " polarizations.")
        self.hdu.header['POL_TYPE'] = ''.join(value)

    @property
    def frequency(self):
        if 'DAT_FREQ' in self.data.names:
            freqs = u.Quantity(self.data['DAT_FREQ'],
                               u.MHz, copy=False)[0]
        else:
            freqs = super().frequency

        if freqs is not None:
            freqs = freqs.reshape(-1, 1)

        return freqs

    @frequency.setter
    def frequency(self, freqs):
        freqs_value = freqs.to_value(u.MHz)
        self.hdu.data['DAT_FREQ'] = np.broadcast_to(freqs_value, (self.nrow,
                                                                  self.nchan))
        # TODO I am not sure if this is the best way to get the channel bandwidth
        self.hdu.header['CHAN_BW'] = freqs_value[1] - freqs_value[0]
        if self.primary_hdu.frequency is None:
            self.primary_hdu.frequency = freqs

    @property
    def sideband(self):
        return np.where(self.header['CHAN_BW'] > 0, np.int8(1), np.int8(-1))

    @sideband.setter
    def sideband(self, sideband):
        assert np.all(np.abs(sideband) == 1), "sideband should be +/- 1"
        self.header['CHAN_BW'] = sideband * abs(self.header['CHAN_BW'])

    @lazyproperty
    def dtype(self):
        """Data type of the data.  Inferred from ``read_data_row(0)``."""
        return self.read_data_row(0).dtype

    def read_data_row(self, index, weighted=False):
        if index >= self.nrow:
            raise EOFError("cannot read from beyond end of input SUBINT HDU.")

        row = self.data[index]
        # Reversed the header shape to match the data
        data_scale = row['DAT_SCL'].reshape(-1, 1)
        data_off_set = row['DAT_OFFS'].reshape(-1, 1)
        try:
            zero_off = self.header['ZERO_OFF']
            # Sometimes zero_off equals * or some such
            float(zero_off)
        except Exception:
            zero_off = 0
        result = (row['DATA'] - zero_off) * data_scale + data_off_set
        if weighted and 'DAT_WTS' in self.data.names:
            result *= row['DAT_WTS'].reshape(-1, 1)
        return result


class PSRSubintHDU(SubintHDU):
    """Wrapper for PSRFITS SUBINT HDUs, providing baseband-style properties.

    Parameters
    ----------
    hdu : `~astropy.io.fits.BinTableHDU` instance
        The PSRFITS table HDU of SUBINT type.
    primary : `~scintillometry.io.psrfits.PSRFITSPrimaryHDU`
        The wrapped PSRFITS main header.
    verify: bool, optional
        Whether to do basic verification.  Default is `True`.

    Notes
    -----
    Right now we are assuming the data rows are continuous in time and the
    frequency are the same.
    """
    def verify(self):
        super().verify()
        assert self.mode.upper() == 'PSR', \
            "Header HDU is not in the folding mode."

        assert int(self.header['NBIN']) > 1, \
            "Invalid 'NBIN' field in the header."

        # Check frequency
        if 'DAT_FREQ' in self.data.names:
            freqs = u.Quantity(self.data['DAT_FREQ'],
                               u.MHz, copy=False)
            assert np.array_equiv(freqs[0], freqs), \
                "Frequencies are not all the same for different rows."

        # NOTE some files has large amount of TSUBING differ. comment this part
        # for right now.
        # tsubint = self.data['TSUBINT']
        # assert all(np.isclose(tsubint[0], tsubint, atol=1e-1)), \
        #     "TSUBINT differ by large amounts in different rows."

        d_shape_raw = self.data['DATA'].shape
        d_shape_header = (self.nbin, self.nchan, self.npol)
        # The shape has to be inversed since FITS is in the Fortran order.
        assert d_shape_raw == (self.nrow,) + d_shape_header[::-1], \
            "Data shape does not match with the header information."

    def _pre_process_hdu(self, hdu, hdu_parts):
        #S UBINT need preprocess of the hdu cards value
        # HDU part should always have cards.
        # TODO, should this part lives here
        search_mode_cards = {'NBITS': 1, 'SIGNINT': 0, 'NSBLK':, 1, 'NSTOT': 1}
        for k, v in search_mode_cards.items():
            for card in hdu_parts['card']:
                if card['name'] == k:
                    card['value'] = v
        hdu, hdu_parts = super()._preprocess_hdu(hdu, hdu_parts)
        return hdu, hdu_parts

    @property
    def start_time(self):
        """Start time of the first sub-integration.

        Notes
        -----
        The start time is accurate to one pulse period. This calculation below
        is consistent with PSRCHIVE's definition
        (defined in psrchive/Base/Classes/Integration.C)
        """
        start_time = super().start_time
        if "OFFS_SUB" in self.data.names:
            offset0 = (u.Quantity(self.data['OFFS_SUB'][0], u.s, copy=False) -
                       self.samples_per_frame / 2 / self.sample_rate)
            start_time += offset0

        return start_time

    @start_time.setter
    def start_time(self, time):
        """
        Note
        ----
        this sets the start time of the HDU, not the file start time.
        """
        try:
            _ = self.primary_hdu.start_time
        except ValueError:
            self.primary_hdu.start_time = time
        dt = (time - self.primary_hdu.start_time)

        center_off0 = dt + 1.0 / self.sample_rate / 2
        self.hdu.data['OFFS_SUB'][0] = (center_off0.to(u.s)).value

    @property
    def samples_per_frame(self):
        return 1

    @property
    def sample_rate(self):
        # NOTE we are assuming TSUBINT is uniform; tested in verify,
        # but as individual numbers seem to vary, take the mean.
        # TODO: check whether there really isn't a better way!.
        sample_time = u.Quantity(self.data['TSUBINT'], u.s).mean()
        return 1.0 / sample_time

    @sample_rate.setter
    def sample_rate(self, value):
        sample_time = 1.0 / value
        self.hdu.data['TSUBINT'] = sample_time.to_value(u.s)

    def close(self):
        super().close()
        self.primary_hdu.close()


HDU_map = {'PRIMARY': PSRFITSPrimaryHDU,
           'SUBINT': SubintHDU}

subint_map = {'PSR': PSRSubintHDU}

hdu_list_template = {'PSR': {'primary': PSRFITSPrimaryHDU,
                             'data': PSRSubintHDU},
                     'SEARCH': [PSRFITSPrimaryHDU,] # Need to add search HDU
                    }
