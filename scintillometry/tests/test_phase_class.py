# Licensed under the GPLv3 - see LICENSE
"""Full-package tests of the Phase class."""

import operator

import pytest
import numpy as np
import astropy.units as u
from astropy.coordinates import Angle
from astropy.time import Time

from ..phases import Phase, FractionalPhase


def assert_equal(phase, other):
    """Check matching type, matching phase1,2 and that phase1 is integer."""
    assert type(phase) is type(other)
    assert np.all(phase == other)
    if isinstance(phase, Phase):
        assert np.all(phase.int % (1. * u.cycle) == 0)


class TestPhase:
    def setup(self):
        self.phase1 = Angle(np.array([1000., 1001., 999., 1005, 1006.]),
                            u.cycle)[:, np.newaxis]
        self.phase2 = Angle(2.**(-53) * np.array([1, -1., 1., -1.]) +
                            np.array([-0.5, 0., 0., 0.5]), u.cycle)
        self.phase = Phase(self.phase1, self.phase2)
        self.delta = Phase(0., self.phase2)

    def test_basics(self):
        assert isinstance(self.phase, Phase)
        assert np.all(self.phase.int % (1. * u.cycle) == 0)
        cycle = self.phase1 + self.phase2
        assert_equal(self.phase.cycle, cycle)
        assert_equal(self.phase.int, Angle(self.phase1))
        assert_equal(self.phase.frac, FractionalPhase(self.phase2))

    @pytest.mark.parametrize('in1,in2', ((1.1111111, 0),
                                         (1.5, 0.111),
                                         (0.11111111, 1),
                                         (1.*u.deg, 0),
                                         (1.*u.cycle, 1.*u.deg)))
    def test_phase1_always_integer(self, in1, in2):
        phase = Phase(in1, in2)
        assert phase.int % (1. * u.cycle) == 0
        expected = u.Quantity(in1 + in2, u.cycle).value
        assert (phase.int + phase.frac).value == expected
        assert phase.value == expected

    def test_conversion(self):
        degrees = self.phase.to(u.degree)
        assert_equal(degrees, Angle(self.phase1 + self.phase2))

    def test_selection(self):
        phase2 = self.phase[0]
        assert phase2.shape == self.phase.shape[1:]
        assert_equal(phase2.cycle, self.phase.cycle[0])

    def test_equality(self):
        phase2 = self.phase[:, 1:2]
        eq = self.phase == phase2
        expected = [False, True, False, False]
        assert np.all(eq == expected)

    def test_addition(self):
        add = self.phase + self.phase
        assert_equal(add, Phase(2. * self.phase1, 2. * self.phase2))
        t = self.phase1 + self.phase
        add2 = self.phase2 + t
        assert_equal(add2, add)
        t = self.phase + self.phase1.to(u.degree)
        add3 = t + self.phase2.to(u.degree)
        assert_equal(add3, add)
        add4 = self.phase + 1. * u.cycle
        assert_equal(add4, Phase(self.phase1 + 1 * u.cycle, self.phase2))
        add5 = 360. * u.deg + self.phase
        assert_equal(add5, add4)

    def test_subtraction(self):
        double = Phase(self.phase1 * 2., self.phase2 * 2.)
        sub = double - self.phase
        assert_equal(sub, self.phase)
        t = self.phase2 * 2. - self.phase
        sub2 = self.phase1 * 2. + t
        assert_equal(sub2, sub)
        t = double - self.phase1.to(u.degree)
        sub3 = t - self.phase2
        assert_equal(sub3, sub)
        sub4 = self.phase - 1. * u.cycle
        assert_equal(sub4, Phase(self.phase1 - 1 * u.cycle, self.phase2))

    @pytest.mark.parametrize('op', (operator.eq, operator.ne,
                                    operator.le, operator.lt,
                                    operator.ge, operator.ge))
    def test_comparison(self, op):
        result = op(self.phase, self.phase[0])
        assert_equal(result, op((self.phase - self.phase[0]).cycle, 0.))
        # Also for small differences.
        result = op(self.phase, self.phase[:, 1:2])
        assert_equal(result, op((self.phase - self.phase[:, 1:2]).cycle, 0.))

    @pytest.mark.parametrize('op', (operator.eq, operator.ne,
                                    operator.le, operator.lt,
                                    operator.ge, operator.ge))
    def test_comparison_quantity(self, op):
        ref = 1005. * u.cy
        result = op(self.phase, ref.to(u.deg))
        assert_equal(result, op((self.phase - ref).cycle, 0.))

    def test_comparison_invalid_quantity(self):
        # Older astropy uses UnitConversionError
        with pytest.raises((TypeError, u.UnitConversionError)):
            self.phase > 1. * u.m

        with pytest.raises((TypeError, u.UnitConversionError)):
            self.phase <= 1. * u.m

        assert (self.phase == 1. * u.m) is False
        assert (self.phase != 1. * u.m) is True

    def test_negation(self):
        neg = -self.phase
        assert_equal(neg, Phase(-self.phase1, -self.phase2))

    def test_absolute(self):
        check = abs(-self.phase)
        assert_equal(check, self.phase)

    def test_unitless_multiplication(self):
        mul = self.phase * 2
        assert_equal(mul, Phase(self.phase1 * 2, self.phase2 * 2))
        mul2 = self.phase * (2. * u.dimensionless_unscaled)
        assert_equal(mul2, mul)
        mul3 = self.phase * 2. * u.one
        assert_equal(mul3, mul)
        mul4 = 2. * self.phase
        assert_equal(mul4, mul)
        mul5 = self.phase * np.full(self.phase.shape, 2.)
        assert_equal(mul5, mul)

    def test_unitless_division(self):
        div = self.phase / 0.5
        assert_equal(div, Phase(self.phase1 * 2, self.phase2 * 2))
        div2 = self.phase / (0.5 * u.dimensionless_unscaled)
        assert_equal(div2, div)
        div3 = self.phase / 0.5 / u.one
        assert_equal(div3, div)
        div4 = self.phase / np.full(self.phase.shape, 0.5)
        assert_equal(div4, div)

    def test_unitfull_multiplication(self):
        mul = self.phase * (2 * u.Hz)
        assert_equal(mul, u.Quantity(self.phase.cycle * 2 * u.Hz))
        mul2 = self.phase * 2. * u.Hz
        assert_equal(mul2, mul)
        mul3 = 2. * u.Hz * self.phase
        assert_equal(mul3, mul)

    def test_unitfull_division(self):
        div = self.phase / (0.5 * u.s)
        assert_equal(div, u.Quantity(self.phase.cycle * 2 / u.s))
        div2 = self.phase / 0.5 / u.s
        assert_equal(div2, div)
        div3 = 0.5 * u.s / self.phase
        assert_equal(div3, 1. / div)

    def test_floor_division_mod(self):
        fd = self.phase // (1. * u.cycle)
        fd_exp = self.phase.int.copy()
        fd_exp[self.phase.frac < 0] -= 1 * u.cycle
        fd_exp = fd_exp / u.cycle
        assert_equal(fd, fd_exp)
        mod = self.phase % (1. * u.cycle)
        mod_exp = Phase(np.where(self.phase.frac >= 0., 0., 1.),
                        self.phase.frac)
        assert_equal(mod, mod_exp)
        exp_cycle = Angle(self.phase.frac, copy=True)
        exp_cycle[exp_cycle < 0.] += 1. * u.cycle
        assert_equal(mod.cycle, exp_cycle)
        dm = divmod(self.phase, 1. * u.cycle)
        assert_equal(dm[0], fd_exp)
        assert_equal(dm[1], mod_exp)
        #
        fd2 = self.phase // (360. * u.degree)
        assert_equal(fd2, fd_exp)
        mod2 = self.phase % (360 * u.degree)
        assert_equal(mod2, mod_exp)
        dm2 = divmod(self.phase, 360 * u.degree)
        assert_equal(dm2[0], fd_exp)
        assert_equal(dm2[1], mod_exp)
        #
        fd3 = self.phase // (240. * u.hourangle)
        fd3_exp = fd_exp // 10
        assert_equal(fd3, fd3_exp)
        mod3 = self.phase % (240. * u.hourangle)
        mod_int_exp = self.phase.int % (10 * u.cy)
        mod_int_exp[0][self.phase.frac[0] < 0] += 10. * u.cy
        mod3_exp = Phase(mod_int_exp, self.phase.frac)
        assert_equal(mod3, mod3_exp)
        dm3 = divmod(self.phase, 240. * u.hourangle)
        assert_equal(dm3[0], fd3_exp)
        assert_equal(dm3[1], mod3_exp)

    @pytest.mark.parametrize('axis', (None, 0, 1))
    def test_min(self, axis):
        m = self.phase.min(axis=axis)
        index = (slice(None) if axis == 1 else self.phase1.argmin(),
                 slice(None) if axis == 0 else self.phase2.argmin())
        assert_equal(m, self.phase[index])

    @pytest.mark.parametrize('axis', (None, 0, 1))
    def test_max(self, axis):
        m = self.phase.max(axis=axis)
        index = (slice(None) if axis == 1 else self.phase1.argmax(),
                 slice(None) if axis == 0 else self.phase2.argmax())
        assert_equal(m, self.phase[index])

    @pytest.mark.parametrize('axis', (None, 0, 1))
    def test_ptp(self, axis):
        ptp = self.phase.ptp(axis)
        assert_equal(ptp, self.phase.max(axis) - self.phase.min(axis))

    @pytest.mark.parametrize('axis', (0, 1))
    def test_sort(self, axis):
        sort = self.phase.sort(axis=axis)
        if axis == 1:
            index = ()
        else:
            index = self.phase1.ravel().argsort()
        assert_equal(sort, self.phase[index])

    @pytest.mark.parametrize('ufunc', (np.sin, np.cos, np.tan))
    def test_trig(self, ufunc):
        d = np.arange(-177, 180, 10) * u.degree
        cycle = 1e10 * u.cycle
        expected = ufunc(d)
        assert not np.isclose(ufunc(cycle + d), expected,
                              atol=1e-14, rtol=1.e-14).any()
        phase = Phase(cycle, d)
        assert np.isclose(ufunc(phase), expected, rtol=1e-14,
                          atol=1e-14).all()

    def test_isnan(self):
        expected = np.zeros(self.phase.shape)
        assert_equal(np.isnan(self.phase), expected)
        # For older astropy, we set input to nan rather than Phase directly,
        # since setting of nan exposes a Quantity bug.
        phase2 = self.phase2.copy()
        phase2[1] = np.nan
        phase = Phase(self.phase1, phase2)

        expected[:, 1] = True
        assert_equal(np.isnan(phase), expected)
        trial = Phase(np.nan)
        assert np.isnan(trial)
