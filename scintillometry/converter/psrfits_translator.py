"""psrfits_translator.py defines the translator for psrfits HDUs
"""

from .translator import Translator
import astropy.units as u
from astropy.time import Time


class HeaderTranslator(Translator):
    """This class defines the PSRFIT header HDU translate functions.

    Parameter
    ---------
    name : str
        The name of the translator instance.
    header_hdu : object
        The psrfits main header object.
    functions : dict, optional
        The input translate functions, the key is the taret key name and the
        value is the translate function.
    """
    def __init__(self, name, header_hdu, functions={}):
        self.header_hdu = header_hdu
        check_input(self.header_hdu)
        super(SubintTranslator, self).__init__(name, 'MAIN',
                                               'StreamGenerator')
    def check_input(self, input_hdu):
        try:
            self.header =  input_hdu.read_header()
            assert self.header['SIMPLE']
        except:
            raise ValueError("Input HDU is not a main header HDU.")

    def get_start_time(self):
        MJD_d_int = self.header['STT_IMJD'] * u.day
        MJD_s_int = self.header['STT_SMJD'] * u.s
        MJD_s_frac = self.header['STT_OFFS'] * u.s
        #NOTE I am assuming this is UTC
        return Time(MJD_d_int, MJD_s_int + MJD_s_frac, format='mjd',
                    scale='utc', precision=9)

    def get_filename(self):
        return self.header_hdu.get_filename()


class SubintTranslator(Translator):
    """This class defines the PSRFITS subint HDU translator.

    Parameter
    ---------
    name : str
        The name of the translator instance.
    header_hdu : object
        The psrfits main header object
    data_hdu : HDU oject
        The psrfits data HDU.
    functions : dict, optional
        The input translate functions, the key is the taret key name and the
        value is the translate function.
    """
    # NOTE should we have one Translator class for all subint or one subint one
    # translator class.
    def __init__(self, name, header_hdu, data_hdu, functions={}):
        self.header_hdu = HeaderTranslator("file_header", header_hdu)
        self.data_hdu = data_hdu
        self.data_header = self.data_hdu.read_header()
        super(SubintTranslator, self).__init__(name, 'SUBINT',
                                               'StreamGenerator')
        check_input(self.data_hdu)

    def check_input(self, input_hdu):
        if input_hdu.get_filename() != self.header_hdu.get_filename():
            raise ValueError("Main header HDU and input HDU are not from the "
                             "same file.")
        else:
            if input_hdu.get_extname() != self.source:
                raise ValueError("Input HDU is not a SUBINT type.")
            else:
                return

    def setup(self):
        # This is just an example here. 
        self.update({'shape': self.get_shape,
                     'sample_rate': self.get_sample_rate,
                     'polarization': self.get_pok,
                     'start_time': self.get_start_times})


    def get_sample_rate(self):
        return 1.0 / (self.data_header['TBIN'] * u.s)

    def get_shape(self):
        nrows = self.data_hdu.get_nrows()
        samples_per_row = self.data_header['NSBLK']
        nchan = self.data_header['NCHAN']
        npol = self.data_header['NPOL']
        nbin = self.data_header['NBIN']
        return (nrows * samples_per_row, nbin, npol, nchan)

    def get_dim_label(self):
        return ('time', 'phase', 'pol', 'freq')

    def get_pol(self):
        return self.data_header['POL_TYPE']

    def get_start_times(self):
        # NOTE should we get the start time for each raw, in case the time gaps
        # in between the rows
        file_start = self.header_hdu.get_start_time()
        subint_times = u.Quantity(self.header_hdu.read_column(col='OFFS_SUB'),
                                  u.s, copy=False)
        samples_per_row = self.data_header['NSBLK']
        sample_time = self.data_header['TBIN']
        start_time = (file_start + subint_times -
                      samples_per_row / 2 * sample_time)
        return start_time

    def get_freqs(self):
        # Those are the frequency for all the rows.
        freqs = u.Quantity(self.header_hdu.read_column(col='DAT_FREQ'),
                           u.MHz, copy=False)
        return freqs

    def get_sideband(self):
        # It is not clear for now.
        return 1

    def get_data(self, time_samples):
        # The seek is not working
        samples_per_row = self.data_header['NSBLK']
        num_rows = int(time_samples / samples_per_row)
        rest_samples = time_samples - num_rows * samples_per_row
        data = self.data_hdu.read(row=np.arange(num_rows))
        result = data['DATA'].reshape((num_rows * samples_per_row,
                                       self.get_shape[1::]))
        return result[0: time_samples]
