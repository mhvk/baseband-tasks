"""Converting between file handler and psrfits format
"""

from .base import FormatReader, FormatWritter
import numpy as np


class HDUReader(FormatReader):
    """PsrfitsReader class is a base class for reading the PSRFIT files. It
    reads PSRFITS's HDUs into a StreamGenerator style of object.

    Parameter
    ---------
    translator: dict or dict-like
        PSRFIT HDU translator.
    kwargs : dict
        Additional input arguments.

    Example
    -------
    >>> import pdat
    >>> from .psrfits_translator import SubintTranslator
    >>> ft = pdat.psrfits("psrfits.fits")
    >>> translator = SubintTranslator("subint", ft[0], ft[1])
    >>> reader = HDUReader(translator)
    >>> reader.seek(1000)
    >>> data = reader.read(2000)
    """
    def __init__(self, translator, **kwargs):
        super(PsrfitsReader, self).__init__(hdu, translator, **kwargs)

    def read_format_data(self, time_samples):
        """This function defines the function to read data from the psrfits
        format object.
        """
        return self.translator['data'](time_samples)
