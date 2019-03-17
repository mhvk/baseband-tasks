# -*- coding: utf-8 -*-
"""Provide a Phase class with integer and fractional part."""
import operator

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle, Longitude
from astropy.time import Time
from astropy.utils import ShapedLikeNDArray
from astropy.time.utils import day_frac


__all__ = ['Phase', 'FractionalPhase']


FRACTION_UFUNCS = {np.cos, np.sin, np.tan}


class FractionalPhase(Longitude):
    _default_wrap_angle = Angle(0.5, u.cycle)


class Phase(Angle):
    """Represent two-part phase.

    The phase is absolute and hence has more limited operations available to it
    than a relative phase (e.g., it cannot be multiplied).  This is analogous
    to the difference between an absolute time and a time difference.

    Parameters
    ----------
    phase1, phase2 : array or `~astropy.units.Quantity`
        Two-part phase.  If arrays, the assumed units are cycles.
    copy : bool, optional
        Make a copy of the input values

    """
    _set_unit = _unit = u.cycle
    _phase_dtype = np.dtype({'names': ['int', 'frac'],
                             'formats': [np.float64]*2})

    # Make sure that reverse arithmetic (e.g., Phase.__rmul__)
    # gets called over the __mul__ of Numpy arrays.
    __array_priority__ = 20000

    def __new__(cls, phase1, phase2=0):
        phase1 = u.Quantity(phase1, u.cycle, copy=False)
        phase2 = u.Quantity(phase2, u.cycle, copy=False)
        return cls.from_angles(phase1, phase2)

    @classmethod
    def from_angles(cls, phase1, phase2, factor=None, divisor=None):
        count, fraction = day_frac(phase1.to_value(u.cycle), phase2.to_value(u.cycle),
                                   factor=factor, divisor=divisor)
        value = np.empty(count.shape, cls._phase_dtype)
        value['int'] = count
        value['frac'] = fraction
        return value.view(cls)

    def __getitem__(self, item):
        result = super().__getitem__(item)
        if result.dtype is self.dtype:
            return result.view(self.__class__)
        elif item == 'frac':
            return result.view(FractionalPhase)
        else:
            return result.view(Angle)

    def __quantity_subclass__(self, unit):
        if unit != self._set_unit:
            return type(self), True
        else:
            return super().__quantity_subclass__(unit)[0], False

    def _set_unit(self, unit):
        if unit is None or not unit != self._set_unit:
            raise u.UnitTypeError(
                "{0} instances require units of '{1}'"
                .format(type(self).__name__, self._equivalent_unit) +
                (", but no unit was given." if unit is None else
                 ", so cannot set it to '{0}'.".format(unit)))

        super()._set_unit(unit)

    def __repr__(self):
        return "{0}({1}, {2})".format(self.__class__.__name__,
                                      self['int'], self['frac'])

    def __str__(self):
        return str(self.value)

    @property
    def int(self):
        return self['int']

    @property
    def frac(self):
        return self['frac']

    @property
    def cycle(self):
        """Full cycle, including phase."""
        return self['int'] + self['frac']

    def to_value(self, unit=None, equivalencies=[]):
        """The numerical value, possibly in a different unit."""
        return self.cycle.to_value(unit, equivalencies)

    value = property(to_value,
                     doc="""The numerical value of this instance.

    See also
    --------
    to_value : Get the numerical value in a given unit.
    """)

    def to(self, *args, **kwargs):
        return self.cycle.to(*args, **kwargs)

    def _advanced_index(self, indices, axis=None, keepdims=False):
        ai = Time._advanced_index(self, indices, axis=axis, keepdims=keepdims)
        return tuple(ai)

    def argmin(self, axis=None, out=None):
        """Return indices of the minimum values along the given axis."""
        phase = self['int'] + self['frac']
        approx = np.min(phase, axis, keepdims=True)
        dt = (self['int'] - approx) + self['frac']
        return dt.argmin(axis, out)

    def argmax(self, axis=None, out=None):
        """Return indices of the maximum values along the given axis."""
        phase = self['int'] + self['frac']
        approx = np.max(phase, axis, keepdims=True)
        dt = (self['int'] - approx) + self['frac']
        return dt.argmax(axis, out)

    def argsort(self, axis=-1):
        """Returns the indices that would sort the phase array."""
        phase_approx = self.value
        phase_remainder = (self - self.__class__(phase_approx)).value
        if axis is None:
            return np.lexsort((phase_remainder.ravel(), phase_approx.ravel()))
        else:
            return np.lexsort(keys=(phase_remainder, phase_approx), axis=axis)

    min = Time.min
    max = Time.max
    ptp = Time.ptp
    sort = Time.sort

    def __add__(self, other):
        if not isinstance(other, Phase):
            try:
                other = Phase(other)
            except Exception:
                return NotImplemented

        return self.from_angles(self['int'] + other['int'],
                                self['frac'] + other['frac'])

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if not isinstance(other, Phase):
            try:
                other = Phase(other)
            except Exception:
                return NotImplemented

        return self.from_angles(self['int'] - other['int'],
                                self['frac'] - other['frac'])

    def __rsub__(self, other):
        out = self.__sub__(other)
        return -out if out is not NotImplemented else out

    def __neg__(self):
        return self.from_angles(-self['int'], -self['frac'])

    def __abs__(self):
        return self._apply(np.copysign, self.value)

    def __mul__(self, other):
        # Check needed since otherwise the self.jd1 * other multiplication
        # would enter here again (via __rmul__)
        if (not isinstance(other, Phase) and
            ((isinstance(other, u.UnitBase) and
              other == u.dimensionless_unscaled) or
             (isinstance(other, str) and other == ''))):
            return self.copy()

        # If other is something consistent with a dimensionless quantity
        # (could just be a float or an array), then we can just multiple in.
        try:
            other = u.Quantity(other, u.dimensionless_unscaled, copy=False)
        except Exception:
            # If not consistent with a dimensionless quantity, try downgrading
            # self to a quantity and see if things work.
            try:
                return self.to(u.cycle) * other
            except Exception:
                # The various ways we could multiply all failed;
                # returning NotImplemented to give other a final chance.
                return NotImplemented

        return self.from_angles(self['int'], self['frac'],
                                factor=other.value)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        # Cannot do __mul__(1./other) as that looses precision
        if ((isinstance(other, u.UnitBase) and
             other == u.dimensionless_unscaled) or
                (isinstance(other, str) and other == '')):
            return self.copy()

        # If other is something consistent with a dimensionless quantity
        # (could just be a float or an array), then we can just divide in.
        try:
            other = u.Quantity(other, u.dimensionless_unscaled, copy=False)
        except Exception:
            # If not consistent with a dimensionless quantity, try downgrading
            # self to a quantity and see if things work.
            try:
                return self.to(u.cycle) / other
            except Exception:
                # The various ways we could divide all failed;
                # returning NotImplemented to give other a final chance.
                return NotImplemented

        return self.from_angles(self['int'], self['frac'],
                                divisor=other.value)

    def __rtruediv__(self, other):
        # Here, we do not have to worry about returning NotImplemented,
        # since other has already had a chance to look at us.
        return other / self.to(u.cycle)

    def _phase_comparison(self, other, op):
        if not isinstance(other, self.__class__):
            try:
                other = self.__class__(other)
            except Exception:
                return NotImplemented

        return op((self['int'] - other['int']) +
                  (self['frac'] - other['frac']), 0.)

    def __lt__(self, other):
        return self._phase_comparison(other, operator.lt)

    def __le__(self, other):
        return self._phase_comparison(other, operator.le)

    def __eq__(self, other):
        return self._phase_comparison(other, operator.eq)

    def __ne__(self, other):
        return self._phase_comparison(other, operator.ne)

    def __gt__(self, other):
        return self._phase_comparison(other, operator.gt)

    def __ge__(self, other):
        return self._phase_comparison(other, operator.ge)

    def __array_ufunc__(self, function, method, *inputs, **kwargs):
        """Wrap numpy ufuncs, taking care of units.

        Parameters
        ----------
        function : callable
            ufunc to wrap.
        method : str
            Ufunc method: ``__call__``, ``at``, ``reduce``, etc.
        inputs : tuple
            Input arrays.
        kwargs : keyword arguments
            As passed on, with ``out`` containing possible quantity output.

        Returns
        -------
        result : `~astropy.units.Quantity`
            Results of the ufunc, with the unit set properly.
        """
        if function in FRACTION_UFUNCS:
            # Only trig functions supported, so just one input.
            quantity = self.frac
            inputs = (quantity,)
        else:
            quantity = self.cycle
            inputs = tuple((input_ if input_ is not self else quantity)
                           for input_ in inputs)

        return quantity.__array_ufunc__(function, method, *inputs,
                                        **kwargs)
