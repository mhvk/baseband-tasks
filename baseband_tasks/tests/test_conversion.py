#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `pulsarbat` package."""

import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.time import Time

from baseband_tasks.conversion import Real2Complex
from baseband_tasks.generators import StreamGenerator, EmptyStreamGenerator


def test_real_to_complex_delta():
    """Test converting a real delta function to complex."""

    def real_delta(handle):
        real_delta = np.zeros(handle.samples_per_frame, dtype=np.float64)
        if handle.offset == 0:
            real_delta[0] = 1.0
        return real_delta

    delta_fh = StreamGenerator(real_delta,
                               samples_per_frame=1024,
                               start_time=Time('2010-11-12T13:14:15'),
                               sample_rate=1. * u.kHz,
                               frequency=400 * u.kHz,
                               sideband=1,
                               shape=(2048, ),
                               dtype='f8')
    real_data = delta_fh.read()
    assert real_data[0] == 1.
    assert np.all(real_data[1:] == 0.)

    complex_delta = np.zeros(2048 // 2, dtype=np.complex128)
    complex_delta[0] = 1.0

    real2complex = Real2Complex(delta_fh)
    complex_signal = real2complex.read()
    assert complex_signal.shape == (1024, )
    assert np.iscomplexobj(complex_signal)
    assert np.isclose(complex_signal, complex_delta).all()
    assert real2complex.frequency == 400.5 * u.kHz
    assert real2complex.sideband == 1

    r = repr(real2complex)
    assert r.startswith('Real2Complex(ih)')


def test_expected_failures():
    with pytest.raises(ValueError):
        Real2Complex(
            EmptyStreamGenerator(samples_per_frame=1024,
                                 start_time=Time('2010-11-12T13:14:15'),
                                 sample_rate=1. * u.kHz,
                                 shape=(2048, ),
                                 dtype='c8'))


@pytest.mark.parametrize('f_nyquist', (0.75, 0.5, 0.25, 0.125, 0.5 + 1 / 32))
def test_real_to_complex_sine(f_nyquist):
    """Test converting a real sine function to complex."""

    def real_sine(handle):

        real_sine = np.sin(f_nyquist * np.pi
                           * np.arange(handle.samples_per_frame))
        return real_sine

    sine_fh = StreamGenerator(real_sine,
                              samples_per_frame=1024,
                              start_time=Time('2010-11-12T13:14:15'),
                              sample_rate=1. * u.kHz,
                              frequency=400 * u.kHz,
                              sideband=-1,
                              shape=(2048, ),
                              dtype='f8')

    f_complex = f_nyquist - 0.5
    complex_dc = np.exp(2j * np.pi
                        * (-0.25 + np.arange(2048 // 2) * f_complex))

    real2complex = Real2Complex(sine_fh)
    complex_signal = real2complex.read()

    assert complex_signal.shape == (1024, )
    assert np.iscomplexobj(complex_signal)
    assert_allclose(complex_signal, complex_dc, atol=1e-8)
    assert real2complex.frequency == 399.5 * u.kHz
    assert real2complex.sideband == -1
