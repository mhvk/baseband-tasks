# Licensed under the GPLv3 - see LICENSE

import numpy as np
import astropy.units as u


class DispersionMeasure(u.SpecificTypeQuantity):
    """Dispersion measure quantity.

    Quantity for electron column density, normally with units of pc / cm**3,
    with additional methods to help correct for dispersion delays:
    `time_delay`, `phase_delay`, and `phase_factor`.

    Parameters
    ----------
    dm : `~astropy.units.Quantity` or float
        Dispersion measure value.  If a `~astropy.units.Quantity` is passed, it
        must have units equivalent to pc/cm**3.  If a float is passed, units
        may be passed to ``unit``, or will otherwise be assumed to be pc/cm**3.
    unit : `~astropy.units.UnitBase` or None
        Units of ``dm``.  If `None` (default), will be set either to the units
        of ``dm`` if ``dm`` is an `~astropy.units.Quantity`, or pc/cm**3
        otherwise.  If ``dm`` is a `~astropy.units.Quantity` and ``unit`` is
        also passed, will try to convert ``dm`` to ``unit``.
    *args, **kwargs
        As for `~astropy.units.Quantity`.

    Notes
    -----
    The constant relating dispersion measure to delay is hardcoded to match
    that of Tempo.  See `Taylor, Manchester, & Lyne (1993)
    <http://adsabs.harvard.edu/abs/1993ApJS...88..529T>`_.  It is accessible as
    the `dispersion_delay_constant` attribute.
    """

    # Constant hardcoded to match assumption made by tempo.
    dispersion_delay_constant = u.s / 2.41e-4 * u.MHz**2 * u.cm**3 / u.pc
    """Dispersion delay constant, hardcoded to match that for Tempo."""

    _equivalent_unit = _default_unit = u.pc / u.cm**3

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

        where the dispersion delay constant is taken to be exactly (inverse of
        Eqn. 6 of Taylor, Manchester, & Lyne 1993):

        .. math::

            \frac{e^2}{2\pi m_ec} = \frac{1}{2.410} \times 10^4\,
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
