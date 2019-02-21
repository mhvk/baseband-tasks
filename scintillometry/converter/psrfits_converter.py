"""Converting between file handler and psrfits format
"""

import pdat
from ..generators import StreamGenerator
import numpy as np


def psrfits2fh(psrfits_file, template=None):
    """The coverter from psrfits file to a file handler

    Parameter
    ---------
    psrfits_file: str
        The psrfits file name.

    Return
    ------
    File handler with header information from the PSRFITS file
    """
    psrft = pdat.psrfits(psrfits_file)
    num_HDU = len(psrft)
    ft_header = psrft[0].read_header()








def ih2psrfits(ih):
    pass
