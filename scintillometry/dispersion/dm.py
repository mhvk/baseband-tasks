# Licensed under the GPLv3 - see LICENSE
import numpy as np
import astropy.units as u


class DispersionMeasure(u.Quantity):
    """Dispersion measure quantity, with methods to calculate time and phase
    delays, and phase factor.

    Parameters
    ----------
    dm : `~astropy.units.Quantity` or float
        Dispersion measure.  If a `~astropy.units.Quantity` is passed, it must
        have units equivalent to pc/cm^3.  If passed a `float`, units may be
        passed to ``unit``, or will otherwise be assumed to be pc/cm^3.
    unit : `~astropy.units.UnitBase` or None
        Units of ``dm``.  If `None` (default), will be set either to the units
        of ``dm`` if ``dm`` is an `~astropy.units.Quantity`, or pc/cm^3
        otherwise.  If ``dm`` is a `~astropy.units.Quantity` and ``unit`` is
        also passed, will try to convert ``dm`` to ``unit``.
    """
    # Eqn. 4.6 of Lorimer & Kramer.
    dispersion_delay_constant = 4148.808 * u.s * u.MHz**2 * u.cm**3 / u.pc
    _default_unit = u.pc / u.cm**3

    def __new__(cls, dm, unit=None, **kwargs):
        if unit is None:
            unit = getattr(dm, 'unit', cls._default_unit)
        self = super(DispersionMeasure, cls).__new__(cls, dm, unit, **kwargs)
        if not self.unit.is_equivalent(cls._default_unit):
            raise u.UnitsError("dispersion measures should have units "
                               "equivalent to pc/cm^3")
        return self

    def __quantity_subclass__(self, unit):
        if unit.is_equivalent(self._default_unit):
            return DispersionMeasure, True
        else:
            return super(DispersionMeasure,
                         self).__quantity_subclass__(unit)[0], False

    def time_delay(self, freq, ref_freq=None):
        r"""Time delay due to dispersion.

        Parameters
        ----------
        freq : `~astropy.units.Quantity`
            Frequency at which to evaluate the dispersion delay.
        ref_freq : `~astropy.units.Quantity`, optional
            Reference frequency relative to which the dispersion delay is
            evaluated.  If not given, infinite frequency is assumed.

        Notes
        -----
        Given the dispersion measure :math:`\mathrm{DM}`, frequency
        :math:`f` and reference frequency :math:`f_\mathrm{ref}`, calculates
        the time delay (Eqn. 4.7, Lorimer & Kramer's Handbook of Pulsar
        Astronomy):

        .. math::

            \Delta t = \frac{e^2}{2\pi m_ec} \mathrm{DM}\left(\frac{1}
                         {f_\mathrm{ref}^2} - \frac{1}{f^2}\right)

        where the dispersion constant is (Eqn. 4.6):

        .. math::

            \frac{e^2}{2\pi m_ec} = (4.148808 \pm 0.000003) \times 10^3\,
                                    \mathrm{MHz}^2\,\mathrm{pc}^{-1}
                                    \,\mathrm{cm}^3\,\mathrm{s}.
        """
        d = self.dispersion_delay_constant * self
        ref_freq_inv2 = 0. if ref_freq is None else 1. / ref_freq**2
        return d * (1. / freq**2 - ref_freq_inv2)

    def phase_delay(self, freq, ref_freq=None):
        r"""Phase delay due to dispersion.

        Parameters
        ----------
        freq : `~astropy.units.Quantity`
            Frequency at which to evaluate the dispersion delay.
        ref_freq : `~astropy.units.Quantity`, optional
            Reference frequency relative to which the dispersion delay is
            evaluated.  If not given, infinite frequency is assumed.

        Notes
        -----
        Given the dispersion measure :math:`\mathrm{DM}`, frequency
        :math:`f` and reference frequency :math:`f_\mathrm{ref}`, calculates
        the phase amplitude of the transfer function (Eqn. 5.21, Lorimer &
        Kramer's Handbook of Pulsar Astronomy, rewritten to use absolute
        frequency):

        .. math::

            \Delta \phi = \frac{e^2\mathrm{DM}}{m_ec} f \left(\frac{1}
                          {f_\mathrm{ref}} - \frac{1}{f}\right)^2
        """
        # Eqn. 5.13 of Lorimer & Kramer integrated along line of sight.
        d = self.dispersion_delay_constant * u.cycle * self
        ref_freq_inv = 0. if ref_freq is None else 1. / ref_freq
        return d * freq * (ref_freq_inv - 1. / freq)**2

    def phase_factor(self, freq, ref_freq=None):
        """Complex exponential factor due to dispersion.

        This is just ``exp(1j * phase_delay)``.

        Parameters
        ----------
        freq : `~astropy.units.Quantity`
            Frequency at which to evaluate the dispersion delay.
        ref_freq : `~astropy.units.Quantity`, optional
            Reference frequency relative to which the dispersion delay is
            evaluated.  If not given, infinite frequency is assumed.
        """
        return np.exp(self.phase_delay(freq, ref_freq).to_value(u.rad) * 1j)
