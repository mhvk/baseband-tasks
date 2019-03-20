"""Converting between file handler and psrfits format
"""

from .base import FormatReader, FormatWritter
import numpy as np


class HDUReader(FormatReader):
    """PsrfitsReader class is a base class for reading the PSRFIT files. It
    reads PSRFITS's HDUs into a StreamGenerator style of object.

    Parameter
    ---------
    hdu: psrfits HDU objects.
        The psrfits file HDU objects.
    translator: dict or dict-like
        PSRFIT HDU translator.
    kwargs : dict
        Additional input arguments.
    """
    def __init__(self, hdu, translator, **kwargs):
        super(PsrfitsReader, self).__init__(hdu, translator, **kwargs)
        
    def read_format_data(self):
        """This function defines the function to read data from the psrfits
        format object.
        """
        return self.translator['data'](self.format_object)
