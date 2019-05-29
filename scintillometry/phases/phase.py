# -*- coding: utf-8 -*-
# Licensed under the GPLv3 - see LICENSE
"""Provide a Phase class with integer and fractional part."""
import operator

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle, Longitude
from astropy.time import Time
from astropy.time.utils import two_sum, two_product
from astropy.utils import minversion


__all__ = ['Phase', 'FractionalPhase']


NUMPY_LT_1_16 = not minversion('numpy', '1.16')

FRACTION_UFUNCS = {np.cos, np.sin, np.tan, np.spacing}

COMPARISON_UFUNCS = {
    np.equal, np.not_equal,
    np.less, np.less_equal, np.greater, np.greater_equal}


def day_frac(val1, val2, factor=None, divisor=None):
    """
    Return the sum of ``val1`` and ``val2`` as two float64s, an integer part
    and the fractional remainder.  If ``factor`` is given, then multiply the
    sum by it.  If ``divisor`` is given, then divide the sum by it.

    The arithmetic is all done with exact floating point operations so no
    precision is lost to rounding error.  This routine assumes the sum is less
    than about 1e16, otherwise the ``frac`` part will be greater than 1.0.

    Returns
    -------
    day, frac : float64
        Integer and fractional part of val1 + val2.
    """
    # Note that this is basically as astropy has it, but with an extra round.
    # See https://github.com/astropy/astropy/pull/8763
    #
    # Add val1 and val2 exactly, returning the result as two float64s.
    # The first is the approximate sum (with some floating point error)
    # and the second is the error of the float64 sum.
    sum12, err12 = two_sum(val1, val2)

    if factor is not None:
        sum12, carry = two_product(sum12, factor)
        carry += err12 * factor
        sum12, err12 = two_sum(sum12, carry)

    if divisor is not None:
        q1 = sum12 / divisor
        p1, p2 = two_product(q1, divisor)
        d1, d2 = two_sum(sum12, -p1)
        d2 += err12
        d2 -= p2
        q2 = (d1 + d2) / divisor  # 3-part float fine here; nothing can be lost
        sum12, err12 = two_sum(q1, q2)

    # get integer fraction
    day = np.round(sum12)
    extra, frac = two_sum(sum12, -day)
    frac += extra + err12
    # This part was missed in astropy...
    excess = np.round(frac)
    day += excess
    extra, frac = two_sum(sum12, -day)
    frac += extra + err12
    return day, frac


class FractionalPhase(Longitude):
    """Phase without the cycle count, i.e., with a range of 1 cycle.

    The input parser is flexible and supports all of the input formats
    supported by :class:`~astropy.coordinates.Angle`: array, list,
    scalar, tuple, string, :class:`~astropy.units.Quantity`
    or another :class:`~astropy.coordinates.Angle`.

    Parameters
    ----------
    angle : array, list, scalar, `~astropy.units.Quantity`,
        :class:`~astropy.coordinates.Angle` The angle value(s). If a tuple,
        will be interpreted as ``(h, m s)`` or ``(d, m, s)`` depending
        on ``unit``. If a string, it will be interpreted following the
        rules described for :class:`~astropy.coordinates.Angle`.
    unit : :class:`~astropy.units.UnitBase`, str, optional
        The unit of the value specified for the angle.  This may be any
        string that `~astropy.units.Unit` understands.  Must be an angular
        unit.
    wrap_angle : :class:`~astropy.coordinates.Angle` or equivalent, optional
        Angle at which to wrap back to ``wrap_angle - 1 cycle``.
        If ``None`` (default), it will be taken to be 0.5 cycle unless ``angle``
        has a ``wrap_angle`` attribute.

    Raises
    ------
    `~astropy.units.UnitsError`
        If a unit is not provided or it is not an angular unit.
    `TypeError`
        If the angle parameter is an instance of :class:`~astropy.coordinates.Latitude`.
    """
    _default_wrap_angle = Angle(0.5, u.cycle)
    _equivalent_unit = _default_unit = u.cycle

    def __new__(cls, angle, unit=None, wrap_angle=None, **kwargs):
        # TODO: ideally, the Longitude/Angle/Quantity initializer by
        # default tries to convert to float also for structured arrays,
        # maybe via astype.
        if isinstance(angle, Phase):
            angle = angle['frac']
        return super().__new__(cls, angle, unit=unit, wrap_angle=wrap_angle,
                               **kwargs)


class Phase(Angle):
    """Represent two-part phase.

    With one part the integer cycle count and the other the fractional phase.

    Parameters
    ----------
    phase1, phase2 : array or `~astropy.units.Quantity`
        Two-part phase.  If arrays, the assumed units are cycles.
    copy : bool, optional
        Ensure a copy is made.  Only relevant if ``phase1`` is a `Phase`
        and ``phase2`` is not given.
    subok : bool, optional
        If `False` (default), the returned array will be forced to be a
        `Phase`.  Otherwise, `Phase` subclasses will be passed through.
        Only relevant if ``phase1`` or ``phase2`` is a `Phase` subclass.
    """
    _equivalent_unit = _unit = _default_unit = u.cycle
    _phase_dtype = np.dtype({'names': ['int', 'frac'],
                             'formats': [np.float64]*2})

    def __new__(cls, phase1, phase2=None, copy=True, subok=False):
        if isinstance(phase1, Phase):
            if phase2 is not None:
                phase1 = phase1 + phase2
                copy = False
            if not subok and type(phase1) is not cls:
                phase1 = phase1.view(cls)
            return phase1.copy() if copy else phase1

        phase1 = Angle(phase1, cls._unit, copy=False)

        if phase2 is not None:
            if isinstance(phase2, Phase):
                phase2 = phase2 + phase1
                return phase2 if subok or type(phase2) is cls else phase2.view(cls)
            phase2 = Angle(phase2, cls._unit, copy=False)

        return cls.from_angles(phase1, phase2)

    @classmethod
    def from_angles(cls, phase1, phase2=None, factor=None, divisor=None, out=None):
        # TODO: would be nice if day_frac had an out parameter.
        phase1_value = phase1.to_value(cls._unit)
        if phase2 is None:
            phase2_value = 0.
        else:
            phase2_value = phase2.to_value(cls._unit)
        count, fraction = day_frac(phase1_value, phase2_value,
                                   factor=factor, divisor=divisor)
        if out is None:
            value = np.empty(count.shape, cls._phase_dtype)
            out = value.view(cls)
        else:
            value = out.view(np.ndarray)
        value['int'] = count
        value['frac'] = fraction
        return out

    def __getitem__(self, item):
        result = super().__getitem__(item)
        if result.dtype is self.dtype:
            return result

        if item == 'frac':
            return result.view(FractionalPhase)
        else:
            assert item == 'int'
            return result.view(Angle)

    def _set_unit(self, unit):
        if unit is None or unit != self._unit:
            raise u.UnitTypeError(
                "{0} instances require units of '{1}'"
                .format(type(self).__name__, self._unit) +
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
        """Rounded cycle count."""
        return self['int']

    @property
    def frac(self):
        """Fractional phase, between -0.5 and 0.5 cycles."""
        return self['frac']

    @property
    def cycle(self):
        """Full cycle, including phase."""
        return self['int'] + self['frac']

    def to_value(self, unit=None, equivalencies=[]):
        """The numerical value, possibly in a different unit."""
        return self.cycle.to_value(unit, equivalencies)

    value = property(to_value,
                     doc="""The numerical value, using standard doubles.

    See also
    --------
    to_value : Get the numerical value in a given unit.
    """)

    def to(self, *args, **kwargs):
        """The phase in a different unit, using standard doubles."""
        return self.cycle.to(*args, **kwargs)

    def _take_along_axis(self, indices, axis=None, keepdims=False):
        if axis is None:
            return self[np.unravel_index(indices, self.shape)]

        if indices.ndim == self.ndim - 1:
            indices = np.expand_dims(indices, axis)

        if NUMPY_LT_1_16:
            ndim = self.ndim
            if axis < 0:
                axis = axis + ndim

            ai = tuple([
                (indices if i == axis else
                 np.arange(s).reshape((1,)*i + (s,) + (1,)*(ndim-i-1)))
                for i, s in enumerate(self.shape)])
            result = self[ai]

        else:
            result = np.take_along_axis(self, indices, axis)

        return result if keepdims else result.squeeze(axis)

    def argmin(self, axis=None, out=None):
        """Return indices of the minimum values along the given axis."""
        approx = np.min(self.cycle, axis, keepdims=True)
        dt = (self['int'] - approx) + self['frac']
        return dt.argmin(axis, out)

    def argmax(self, axis=None, out=None):
        """Return indices of the maximum values along the given axis."""
        approx = np.max(self.cycle, axis, keepdims=True)
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

    # Below are basically straight copies from Time
    def min(self, axis=None, out=None, keepdims=False):
        """Minimum along a given axis.

        This is similar to :meth:`~numpy.ndarray.min`, but adapted to ensure
        that the full precision is used.
        """
        if out is not None:
            raise ValueError("An `out` argument is not yet supported.")
        return self._take_along_axis(self.argmin(axis), axis, keepdims)

    def max(self, axis=None, out=None, keepdims=False):
        """Maximum along a given axis.

        This is similar to :meth:`~numpy.ndarray.max`, but adapted to ensure
        that the full precision is used.
        """
        if out is not None:
            raise ValueError("An `out` argument is not yet supported.")
        return self._take_along_axis(self.argmax(axis), axis, keepdims)

    def ptp(self, axis=None, out=None, keepdims=False):
        """Peak to peak (maximum - minimum) along a given axis.

        This is similar to :meth:`~numpy.ndarray.ptp`, but adapted to ensure
        that the full precision is used.
        """
        if out is not None:
            raise ValueError("An `out` argument is not yet supported.")
        return (self.max(axis, keepdims=keepdims) -
                self.min(axis, keepdims=keepdims))

    def sort(self, axis=-1):
        """Return a copy sorted along the specified axis.

        This is similar to :meth:`~numpy.ndarray.sort`, but internally uses
        indexing with :func:`~numpy.lexsort` to ensure that the full precision
        given by the two doubles is kept.

        Parameters
        ----------
        axis : int or None
            Axis to be sorted.  If ``None``, the flattened array is sorted.
            By default, sort over the last axis.
        """
        return self._take_along_axis(self.argsort(axis), axis, keepdims=True)

    # Quantity lets ndarray.__eq__, __ne__ deal with structured arrays (like us).
    # Override this so we can deal with it in __array_ufunc__.
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
        result : `~scintillometry.phases.Phase`, `~astropy.units.Quantity`, or `~numpy.ndarray`
            Results of the ufunc, with the unit set properly,
            `~scintillometry.phases.phase` if possible (i.e., units of cycles,
            others `~astropyy.units.Quantity` or `~numpy.ndarray` as
            appropriate.
        """
        # Do *not* use inputs.index(self) since that will use __eq__
        for i_self, input_ in enumerate(inputs):
            if input_ is self:
                break
        else:
            i_self += 1

        out = kwargs.get('out', None)
        if out is not None and len(out) == 1 and isinstance(out[0], Phase):
            phase_out = kwargs.pop('out')[0]
            out = None
        else:
            phase_out = None

        if function in FRACTION_UFUNCS and i_self == 0:
            if phase_out is not None:  # TODO: spacing is in principle OK.
                return NotImplemented

            return self.frac.__array_ufunc__(function, method,
                                             self.frac, **kwargs)

        elif function in {np.add, np.subtract} and out is None:
            try:
                phases = [Phase(input_, copy=False, subok=True)
                          for input_ in inputs]
            except Exception:
                return NotImplemented

            return self.from_angles(
                function(phases[0]['int'], phases[1]['int']),
                function(phases[0]['frac'], phases[1]['frac']),
                out=phase_out)

        elif function in COMPARISON_UFUNCS and i_self <= 1:
            if phase_out is not None:
                return NotImplemented

            phases = list(inputs)
            try:
                phases[1-i_self] = Phase(inputs[1-i_self], copy=False,
                                         subok=True)
            except Exception:
                return NotImplemented

            diff = ((phases[0]['int'] - phases[1]['int']) +
                    (phases[0]['frac'] - phases[1]['frac']))
            return diff.__array_ufunc__(function, method, diff, 0, **kwargs)

        elif ((function is np.multiply and i_self < 2 or
               function is np.divide and i_self == 0) and out is None and
              not isinstance(inputs[1-i_self], Phase)):
            try:
                other = u.Quantity(inputs[1-i_self], u.dimensionless_unscaled,
                                   copy=False)
            except Exception:
                # If not consistent with a dimensionless quantity,
                # we follow the standard route of downgrading ourself
                # to a quantity and see if things work.
                pass
            else:
                if function is np.multiply:
                    return self.from_angles(self['int'], self['frac'],
                                            factor=other.value, out=phase_out)
                else:
                    return self.from_angles(self['int'], self['frac'],
                                            divisor=other.value, out=phase_out)

        elif (function in {np.floor_divide, np.remainder, np.divmod} and
              i_self == 0):
            fd = np.floor_divide(self.cycle, inputs[1])
            corr = Phase.from_angles(inputs[1], factor=fd)
            remainder = np.subtract(self, corr)
            fdx = np.floor_divide(remainder.cycle, inputs[1])
            # This can likely be optimized...
            # Note: one cannot just loop, because rounding of exact 0.5.
            # TODO: check this method is really correct.
            if np.count_nonzero(fdx):
                fd += fdx
                corr = Phase.from_angles(inputs[1], factor=fd)
                remainder = np.subtract(self, corr, out=remainder)

            if function is np.floor_divide:
                return fd
            elif function is np.remainder:
                return remainder
            else:
                return fd, remainder

        elif function is np.positive and i_self == 0 and out is None:
            return self.from_angles(self['int'], self['frac'],
                                    out=phase_out)

        elif function is np.negative and i_self == 0 and out is None:
            return self.from_angles(-self['int'], -self['frac'],
                                    out=phase_out)

        elif function in {np.absolute, np.fabs} and out is None:
            return self.from_angles(self['int'], self['frac'],
                                    factor=np.sign(self.value),
                                    out=phase_out)

        # Fall-back: treat Phase as a simple Quantity.
        if i_self < function.nin:
            inputs = tuple((input_.cycle if isinstance(input_, Phase)
                            else input_) for input_ in inputs)
            quantity = inputs[i_self]
        else:
            quantity = self.cycle

        if phase_out is None:
            return quantity.__array_ufunc__(function, method, *inputs,
                                            **kwargs)
        else:
            # We won't be able to store in a phase directly, but might
            # as well use one of its elements to store the angle.
            result = quantity.__array_ufunc__(function, method, *inputs,
                                              out=(phase_out['int'],), **kwargs)
            return phase_out.from_angles(result, out=phase_out)

    def _new_view(self, obj=None, unit=None):
        # If the unit is not right, we should ensure we change our two-float
        # dtype to a single float.
        if unit is not None and unit != self._unit:
            if obj is None:
                obj = self.cycle
            elif isinstance(obj, Phase):
                obj = obj.cycle

        return super()._new_view(obj, unit)
