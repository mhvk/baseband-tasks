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
        self.phase2 = Angle(np.arange(0.125, 0.99, 0.25), u.cycle)
        self.phase = Phase(self.phase1, self.phase2)
        self.delta = Phase(0., self.phase2)

    def test_basics(self):
        assert isinstance(self.phase, Phase)
        assert np.all(self.phase.int % (1. * u.cycle) == 0)
        cycle = self.phase1 + self.phase2
        assert_equal(self.phase.cycle, cycle)
        count = cycle.round()
        assert_equal(self.phase.int, count)
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

    @pytest.mark.parametrize('op', (operator.eq, operator.ne,
                                    operator.le, operator.lt,
                                    operator.ge, operator.ge))
    def test_comparison(self, op):
        result = op(self.phase, self.phase[0])
        assert_equal(result, op(self.phase.cycle, self.phase[0].cycle))

    @pytest.mark.parametrize('op', (operator.eq, operator.ne,
                                    operator.le, operator.lt,
                                    operator.ge, operator.ge))
    def test_comparison_quantity(self, op):
        ref = 1005. * u.cy
        result = op(self.phase, ref.to(u.deg))
        assert_equal(result, op(self.phase.cycle, ref))

    def test_comparison_invalid_quantity(self):
        with pytest.raises(TypeError):
            self.phase > 1. * u.m

        with pytest.raises(TypeError):
            self.phase <= 1. * u.m

        assert (self.phase == 1. * u.m) is False
        assert (self.phase != 1. * u.m) is True

    def test_addition(self):
        add = self.phase + self.phase
        assert_equal(add, Phase(2. * self.phase1, 2. * self.phase2))
        add2 = self.phase.to(u.cycle) + self.phase
        assert_equal(add2, add)
        add3 = self.phase + self.phase.to(u.degree)
        assert_equal(add3, add)
        add4 = self.phase + 1. * u.cycle
        assert_equal(add4, Phase(self.phase1 + 1 * u.cycle, self.phase2))
        add5 = 360. * u.deg + self.phase
        assert_equal(add5, add4)

    def test_subtraction(self):
        half = Phase(self.phase1 / 2., self.phase2 / 2.)
        sub = half - self.phase
        assert_equal(sub, Phase(-self.phase1 / 2., -self.phase2 / 2.))
        sub2 = self.phase.to(u.cycle) * 0.5 - self.phase
        assert_equal(sub2, sub)
        sub3 = half - self.phase.to(u.degree)
        assert_equal(sub3, sub)
        sub4 = self.phase - 1. * u.cycle
        assert_equal(sub4, Phase(self.phase1 - 1 * u.cycle, self.phase2))

    def test_negation(self):
        neg = -self.phase
        assert_equal(neg, Phase(-self.phase1, -self.phase2))

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
        phase = self.phase[self.phase != Phase(0, 0)]
        div = phase / (0.5 * u.s)
        assert_equal(div, u.Quantity(phase.cycle * 2 / u.s))
        div2 = phase / 0.5 / u.s
        assert_equal(div2, div)
        div3 = 0.5 * u.s / phase
        assert_equal(div3, 1. / div)

    @pytest.mark.parametrize('axis', (None, 0, 1))
    def test_min(self, axis):
        m = self.phase.min(axis=axis)
        assert_equal(m, Phase(self.phase.cycle.min(axis=axis)))

    @pytest.mark.parametrize('axis', (None, 0, 1))
    def test_max(self, axis):
        m = self.phase.max(axis=axis)
        assert_equal(m, Phase(self.phase.cycle.max(axis=axis)))

    @pytest.mark.parametrize('axis', (None, 0, 1))
    def test_ptp(self, axis):
        ptp = self.phase.ptp(axis)
        assert_equal(ptp, Phase(self.phase.cycle.ptp(axis=axis)))

    @pytest.mark.parametrize('axis', (0, 1))
    def test_sort(self, axis):
        sort = self.phase.sort(axis=axis)
        comparison = self.phase.cycle.copy()
        comparison.sort(axis=axis)
        assert_equal(sort, Phase(comparison))

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
