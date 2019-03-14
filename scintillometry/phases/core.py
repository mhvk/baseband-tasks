# Licensed under the GPLv3 - see LICENSE
"""Phase_utils.py defines the phase calculation utility class. Currently, the
pulse phase at a given time can be computed by PINT or polycos.
"""
import warnings

import numpy as np
import astropy.units as u
from .predictor import Polyco


__all__ = ['PintPhase', 'PolycoPhase']


class PintPhase(object):
    """A utility class designated for computing phase using PINT.
       Parameter
       ---------
       par_file : str
           TEMPO/TEMPO2 style parameter file.
       observatory : str
           Observatory name or observatory code.
       frequency : float or `~astropy.units.Quantity`.
           Observing frequency default units is 'MHz'
       **kwargs
           Additional key words arguments for making TOAs. Please see
           documentation of `PintToas` class.

       Return
       ------
       Two-part phases. The phase are return as `~astropy.units.Quantity` in
       the format of integer part and fractional part.

       Note
       ----
       This method provides high precision phase calculation(~10 Nanosecond
       timing precision). The fractional phase is between -0.5 to 0.5.
    """
    def __init__(self, par_file, observatory, frequency, **kwargs):
        from .pint_toas import PintToas
        from pint.models import get_model
        self.par_file = par_file
        self.model = get_model(self.par_file)
        self.toa_maker = PintToas(observatory, frequency, **kwargs)

    def __call__(self, t):
        """Compute the apparent phase at one or more time stemp

           Parameter
           ---------
           t : `astropy.time.Time` instance
               The input time stamps.

           Return
           ------
           The apparent pulse phase at time t. The phases are return as
           'integer phase, fractional phase'
        """
        toas = self.toa_maker(t)
        ph = self.model.phase(toas)
        shape = getattr(t, 'shape', ())
        return (u.Quantity(ph.int.reshape(shape), u.cycle, copy=False),
                u.Quantity(ph.frac.reshape(shape), u.cycle, copy=False))

    def apparent_spin_freq(self, t):
        """Compute the apparent spin frequency at one or more time stamps.

            Parameter
            ---------
            t : `astropy.time.Time` instance
                The input time stamps.

            Return
            ------
            The apparent spin frequency at time t.
        """
        toas = self.toa_maker(t)
        return self.model.d_phase_d_toa(toas).reshape(getattr(t, 'shape', ()))


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
           t : `astropy.time.Time` instance
               The input time stamps.

           Return
           ------
           The apparent pulse phase at time t. The phases are return as
           'integer phase, fractional phase'.
        """
        ph = self.polyco(t)
        with u.set_enabled_equivalencies([(u.cycle, None)]):
            ph_int, ph_frac = divmod(ph, 1)
            return (u.Quantity(ph_int, u.cycle, copy=False),
                    u.Quantity(ph_frac, u.cycle, copy=False))

    def apparent_spin_freq(self, t):
        """Compute the apparent spin frequency at one or more time stamps.

            Parameter
            ---------
            t : `astropy.time.Time` instance
                The input time stamps.

            Return
            ------
            The apparent spin frequency at time t.
        """
        f0 = self.polyco(t, deriv=1)
        return f0.to(u.Hz, equivalencies=[(u.cy / u.s, u.Hz)])
