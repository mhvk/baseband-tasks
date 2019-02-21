""" template.py defines a set of light templeta class for preparing the methods
of converting the header information and data.
"""


import astropy.units as u

__all__ = ['TemplateBase', 'PsrfitsSubint' ]


class TemplateBase:
    """The base translator class which defines set of functions to
    translator the data in a certain formate of file handler

    Parameter
    ---------
    handler : instance
        The file handler. Note this is a gerneric file handler, not limited to
        the baseband handler.
    """
    def __init__(self, handler):
        self.handler = handler
        self.info_funcs = {}

    def setup(self):
        pass

    def get_info(self, info_name, **kwargs):
        """This is a high-lever wrapper function that interprates the
           information in the file.

        Parameter
        ---------
        info_name : str
            The name of required information

        Return
        ------
        The required infomation
        """
        return self.info_funcs[info_name](**kwargs)

    def get_data(self):
        raise NotImplementedError

class PsrfitsHDU(TemplateBase):
    """ PsrfitHDU is the base template for PSRFIT Header Data Unit.

    Parameter
    ---------
        HDU_handler: `~fitsio.fitslib.HDU`
    """
    def __init__(self, HDU_handler):
        super().__init__(handler)
        self.setup()

    def setup(self):
        self.header = self.handler.read_header()
        self.num_fields = self.header.get('TFIELDS')
        self.num_row = self.header.get('NAXIS2')
        self._map_fields()

    def _map_fields(self):
        result = {}
        for fi in range(1, len(num_fields + 1):
            name = self.header.get('TTYPE{}'.format(fi))
            unit = u.Unit(self.header.get('TUNIT{}'.format(fi)))
            result[name] = (fi, unit)
        return result

    def read(self, rows=None, columns=None):
        return self.handler.read(rows=rows, columns=columns)

class PsrfitsMainHeader(PsrfitsHDU):
    """PsrfitsSubint template is designed to interprate the PSTFITS main header
     HDU.

    Parameter
    ---------
        handler : `~pdat.psrfits` main header HDU handler.
    """
    def __init__(self, handler):
        super().__init__(handler)
        self.header = None
        self.setup()

    def setup(self):
        pass



class PsrfitsSubint(PsrfitsHDU):
    """PsrfitsSubint template is designed to interprate the PSTFITS subint HDU.

    Parameter
    ---------
        handler : `~pdat.psrfits` SUBINT HDU handler.
    """
    def __init__(self, handler):
        super().__init__(handler)
        self.header = None
        self.setup()

    def setup(self):
        """ Setup the PSRFITS SUBINT template.
        """
        # Check if hander is a PSRFITS subint handler.
        try:
            extname = self.handle.get_extname()
            assert exname == 'SUBINT'
        except:
            raise RuntimeError("Handler is not a 'SUBINT' type of handler.")
        super().setup()
        # Set up the info_funcs
        # Read header

        # Read data

    def get_time_axis(self):
        pass

    def get_frequency(self):
        pass

    def get_data_dim(self):
        pass

    def get_time_sample(self):
        pass

    def get_nchan(self):
        pass

    def get_chan_bw(self):
        pass

    def get_n_time_sample(self):
        pass
