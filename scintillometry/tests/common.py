# Licensed under the GPLv3 - see LICENSE
"""Common parts to the tests."""
import numpy as np
from astropy import units as u

from baseband import vdif, dada
from baseband.data import SAMPLE_VDIF, SAMPLE_DADA

from ..base import SetAttribute


class UseVDIFSample:
    def setup(self):
        self.fh = vdif.open(SAMPLE_VDIF)

    def teardown(self):
        self.fh.close()


class UseDADASample:
    def setup(self):
        self.fh = dada.open(SAMPLE_DADA)

    def teardown(self):
        self.fh.close()


class UseVDIFSampleWithAttrs:
    def setup(self):
        self._fh = vdif.open(SAMPLE_VDIF)
        self.fh = SetAttribute(
            self._fh,
            frequency=311.25*u.MHz+(np.arange(8.)//2)*16.*u.MHz,
            sideband=np.array(1),
            polarization=np.tile(['L', 'R'], 4))

    def teardown(self):
        self.fh.close()
        self._fh.close()
