# Licensed under the GPLv3 - see LICENSE

import warnings

import numpy as np
import astropy.units as u
import astropy.time as time
try:
    from .pint_utils import PintUtils
except:
    pass

__all__ = ['PhaseBase']


class PintPhase(object):
    """A utility class designated for computing phase using PINT.
       Parameter
       ---------
       par_file : str
           TEMPO/TEMPO2 style parameter fileself.
       obs : str
           Observatory name or observatory codeself.
       obs_freq : float or `~astropy.units.Quantity`.
           Observing frequency default units is 'MHz'
    """
    def __init__(self, par_file, obs, obs_freq, solar_ephem='de436',
                 bipm_version='BIPM2017' frac_phase=False):
        self.par_file = par_file
        self.pu = PintUtils(self.par_file)
        self.obs = obs
        self.obs_freq = obs_freq
        self.solar_ephem = solar_ephem
        self.bipm_version = bipm_version

   def __call__(self, t):
       self.pu.get_toas(times=t, obs=self.obs, obs_freq=self.obs_freq,
                        solar_ephem=self.solar_ephem,
                        bipm_version=self.bipm_version)
       ph = self.pu.compute_phase()
       return (ph.frac, ph.int)
