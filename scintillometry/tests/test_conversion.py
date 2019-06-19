#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `pulsarbat` package."""

import pytest
import numpy as np
import astropy.units as u
from astropy.time import Time

from ..conversion import Real2Complex
from ..generators import StreamGenerator


def real_delta(handle):

    real_delta = np.zeros(handle.samples_per_frame, dtype=np.float64)
    if handle.offset == 0:
        real_delta[0] = 1.0
    return real_delta


def real_sine(handle):

    real_sine = np.sin(np.pi * np.arange(handle.samples_per_frame) / 2)
    return real_sine


def test_real_to_complex_delta():
    """Test converting a real delta function to complex."""
    delta_fh = StreamGenerator(real_delta,
                               samples_per_frame=1024,
                               start_time=Time('2010-11-12T13:14:15'),
                               sample_rate=1. * u.kHz,
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


def test_real_to_complex_sine():
    """Test converting a real sine function to complex."""

    sine_fh = StreamGenerator(real_sine,
                              samples_per_frame=1024,
                              start_time=Time('2010-11-12T13:14:15'),
                              sample_rate=1. * u.kHz,
                              shape=(2048, ),
                              dtype='f8')
    complex_dc = -1j * np.ones(2048 // 2, dtype=np.complex128)

    real2complex = Real2Complex(sine_fh)
    complex_signal = real2complex.read()

    assert complex_signal.shape == (1024, )
    assert np.iscomplexobj(complex_signal)
    assert np.isclose(complex_signal, complex_dc).all()
