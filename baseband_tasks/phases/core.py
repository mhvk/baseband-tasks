# Licensed under the GPLv3 - see LICENSE
"""Phase_utils.py defines the phase calculation utility class. Currently, the
pulse phase at a given time can be computed by PINT or polycos.
"""

from astropy import units as u

from .phase import Phase
from .predictor import Polyco
from .pint_toas import PintToas


__all__ = ['PintPhase', 'PolycoPhase']


class PintPhase:
    """Helper class for computing pulsar phases using PINT.

    Parameters
    ----------
    par_file : str
        TEMPO/TEMPO2 style parameter file.
    observatory : str
        Observatory name or observatory code.
    frequency : `~astropy.units.Quantity`.
        Observing frequency.  If not a scalar, one has to ensure it can be
        broadcast properly against time arrays for which phases and spin
        frequencies are calculated.
    **kwargs
        Additional key words arguments for making TOAs.  Please see the
        documentation of `~baseband_tasks.phases.pint_toas.PintToas`.

    Notes
    -----
    This method provides high precision phase calculation(~10 Nanosecond
    timing precision).
    """

    def __init__(self, par_file, observatory, frequency, **kwargs):
        from pint.models import get_model

        self.par_file = par_file
        self.model = get_model(self.par_file)
        self.toa_maker = PintToas(observatory, frequency, **kwargs)

    def __call__(self, t):
        """Compute the apparent phase at one or more times.

        Parameters
        ----------
        t : `~astropy.time.Time`
            The input time stamps.

        Returns
        -------
        phase : `~baseband_tasks.phases.Phase`
            The apparent pulse phase at time ``t``, using a 2-part double of
            the integer cycle and the fractional phase.  The latter is
            between -0.5 and 0.5.
        """
        toas = self.toa_maker(t)
        ph = self.model.phase(toas)
        shape = getattr(toas, 'shape', ())
        # TODO: Once PINT uses the Phase class, we can return the
        # result directly.
        with u.add_enabled_equivalencies([(u.dimensionless_unscaled, u.cy)]):
            return Phase(ph.int, ph.frac).reshape(shape)

    def apparent_spin_freq(self, t):
        """Compute the apparent spin frequency at one or more times.

        Parameters
        ----------
        t : `~astropy.time.Time`
            The input time stamps.

        Returns
        -------
        f0 : `~astropy.units.Quantity`
            The apparent spin frequency at time ``t``.
        """
        toas = self.toa_maker(t)
        return self.model.d_phase_d_toa(toas).reshape(getattr(t, 'shape', ()))


class PolycoPhase:
    """Helper class for computing pulsar phases using polycos.

    Parameters
    ----------
    polyco_file : str
        Tempo style polyco file.
    """

    def __init__(self, polyco_file):
        self.polyco = Polyco(polyco_file)

    def __call__(self, t):
        """Compute the apparent phase at one or more times.

        Parameters
        ----------
        t : `~astropy.time.Time`
            The input time stamps.

        Returns
        -------
        phase : `~baseband_tasks.phases.Phase`
            The apparent pulse phase at time ``t``, using a 2-part double of
            the integer cycle and the fractional phase.  The latter is
            between -0.5 and 0.5.
        """
        return self.polyco(t)

    def apparent_spin_freq(self, t):
        """Compute the apparent spin frequency at one or more times.

        Parameters
        ----------
        t : `~astropy.time.Time`
            The input time stamps.

        Returns
        -------
        f0 : `~astropy.units.Quantity`
            The apparent spin frequency at time ``t``.
        """
        f0 = self.polyco(t, deriv=1)
        return f0.to(u.Hz, equivalencies=[(u.cy / u.s, u.Hz)])
