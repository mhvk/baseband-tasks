# Licensed under the GPLv3 - see LICENSE

import warnings

import numpy as np
import astropy.units as u
import astropy.time as time

try:
    from .pint_utils import PintUtils
except:
    pass
from .predictor import Polyco

__all__ = ['PintPhase', 'PolycoPhase']


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
       solar_ephem : str
           The solar system ephemeris version.
       bipm_version : str
           The BIPM clock correction version.

       Note
       ----
       This method provides high precision phase calculation. The fractional
       phase is between -0.5 to 0.5.
    """
    def __init__(self, par_file, obs, obs_freq, solar_ephem='de436',
                 bipm_version='BIPM2015'):
        self.par_file = par_file
        self.pu = PintUtils(self.par_file)
        self.obs = obs
        self.obs_freq = obs_freq
        self.solar_ephem = solar_ephem
        self.bipm_version = bipm_version

    def __call__(self, t):
        """Compute the apparent phase at one or more time stemp

           Parameter
           ---------
           t : `astropy.time.Time` object
               The input time stemps.

           Return
           ------
           The apparent pulse phase at time t. The phases are return as
           'integer phase, fractional phase'
        """
        self.pu.get_toas(time=t, obs=self.obs, obs_freq=self.obs_freq,
                         solar_ephem=self.solar_ephem,
                         bipm_version=self.bipm_version)
        ph = self.pu.compute_phase()
        return u.Quantity((ph.int, ph.frac), u.cycle)

    def apparent_spin_freq(self, t):
        """Compute the apparent spin frequency at one or more time stemps.

            Parameter
            ---------
            t : `astropy.time.Time` object
                The input time stemps.

            Return
            ------
            The apparent spin frequency at time t. The phases are return as a
            tuple with '(integer phase, fractional phase)'.
        """
        self.pu.get_toas(time=t, obs=self.obs, obs_freq=self.obs_freq,
                         solar_ephem=self.solar_ephem,
                         bipm_version=self.bipm_version)
        apprnt_f0 = self.pu.model.d_phase_d_toa(self.pu.toas)
        return apprnt_f0


class PolycoPhase(object):
    """A utility class for a Tempo style polyco phase calculation.

       Parameter
       ---------
       polyco_file : str
           Tempo style polyco file.
    """
    def __init__(self, polyco_file):
        self.polyco = Polyco(polyco_file)

    def __call__(self, t):
        """Compute the apparent phase at one or more time stemp

           Parameter
           ---------
           t : `astropy.time.Time` object
               The input time stemps.

           Return
           ------
           The apparent pulse phase at time t. The phases are return as
           'integer phase, fractional phase'.
        """
        t = t._apply(np.atleast_1d)
        ph = self.polyco(t)
        return u.Quantity(np.divmod(ph, 1), u.cycle)

    def apparent_spin_freq(self, t):
        """Compute the apparent spin frequency at one or more time stemps.

            Parameter
            ---------
            t : `astropy.time.Time` object
                The input time stemps.

            Return
            ------
            The apparent spin frequency at time t. The phases are return as a
            tuple with '(integer phase, fractional phase)'.
        """
        t = t._apply(np.atleast_1d)
        apprnt_f0 = self.polyco(t, deriv=1)
        return apprnt_f0.to(u.Hz, equivalencies=[(u.cy/u.s, u.Hz)])
