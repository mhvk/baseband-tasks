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


COMPARISON_UFUNCS = {np.equal, np.not_equal,
                     np.less, np.less_equal,
                     np.greater, np.greater_equal}


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
    _fixed_unit = _unit = u.cycle
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
        if unit != self._fixed_unit:
            return type(self), True
        else:
            return super().__quantity_subclass__(unit)[0], False

    def _set_unit(self, unit):
        if unit is None or unit != self._fixed_unit:
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
        return str(self.cycle)

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

    def __neg__(self):
        return self.from_angles(-self['int'], -self['frac'])

    def __abs__(self):
        return self._apply(np.copysign, self.value)

    def __eq__(self, other):
        try:
            return np.equal(self, other)
        except Exception:
            return NotImplemented

    def __ne__(self, other):
        try:
            return np.not_equal(self, other)
        except Exception:
            return NotImplemented

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
            assert self is inputs[0]
            return self.frac.__array_ufunc__(function, method,
                                             self.frac, **kwargs)

        elif function in {np.add, np.subtract} | COMPARISON_UFUNCS:
            inputs = list(inputs)
            i_other = 1 if inputs[0] is self else 0
            if not isinstance(inputs[i_other], Phase):
                try:
                    inputs[i_other] = Phase(inputs[i_other])
                except Exception:
                    return NotImplemented

            if function is np.add or function is np.subtract:
                return self.from_angles(
                    function(inputs[0]['int'], inputs[1]['int']),
                    function(inputs[0]['frac'], inputs[1]['frac']))
            else:
                return self.int.__array_ufunc__(
                    function, method,
                    (inputs[0]['int'] - inputs[1]['int']) +
                    (inputs[0]['frac'] - inputs[1]['frac']), 0, **kwargs)

        elif (function is np.multiply or
              function is np.divide and inputs[0] is self):
            inputs = list(inputs)
            i_other = 1 if inputs[0] is self else 0
            other = inputs[i_other]
            if not isinstance(other, Phase):
                try:
                    other = u.Quantity(other, u.dimensionless_unscaled,
                                       copy=False)
                except Exception:
                    # If not consistent with a dimensionless quantity,
                    # we follow the standard route of downgrading ourself
                    # to a quantity and see if things work.
                    pass
                else:
                    if function is np.multiply:
                        return self.from_angles(self['int'], self['frac'],
                                                factor=other.value)
                    else:
                        return self.from_angles(self['int'], self['frac'],
                                                divisor=other.value)

        quantity = self.cycle
        inputs = tuple((input_ if input_ is not self else quantity)
                       for input_ in inputs)

        return quantity.__array_ufunc__(function, method, *inputs,
                                        **kwargs)

    def _new_view(self, obj=None, unit=None):
        obj_dtype = getattr(obj, 'dtype', None)
        if unit is None or unit == self._fixed_unit:
            if obj is not None and obj_dtype != self.dtype:
                return self.__class__(obj)

            return super()._new_view(obj, unit)
        else:
            if obj_dtype == self.dtype:
                obj = obj.cycle
            return self.cycle._new_view(obj, unit)
