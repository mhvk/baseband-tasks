# Licensed under the GPLv3 - see LICENSE

import pytest
import numpy as np
import astropy.units as u

from ..functions import FunctionTask, ComplexFunctionTask, SquareTask

from baseband import vdif, dada
from baseband.data import SAMPLE_VDIF, SAMPLE_DADA


def zero_channel_4(data):
    data[:, 4] = 0.
    return data


def zero_every_8th_sample(data):
    data[::8] = 0.
    return data


def remove_every_other(data):
    return data[::2]


def double_data(data):
    return np.stack((data, data), axis=1).reshape((-1,) + data.shape[1:])


def zero_every_8th_complex(fh, data):
    every_8th = (fh.tell() + np.arange(fh.samples_per_frame)) % 8 == 0
    data[every_8th] = 0.
    return data


class TestFunctions(object):
    """Test applying functions to Baseband's sample VDIF file."""

    @pytest.mark.parametrize('function, sample_factor',
                             ((zero_channel_4, 1.),
                              (zero_every_8th_sample, 1.),
                              (remove_every_other, 0.5),
                              (double_data, 2.)))
    def test_functions(self, function, sample_factor):
        """Test setting a channel to zero."""

        # Load baseband file and get reference intensities.
        fh = vdif.open(SAMPLE_VDIF)
        ref_data = function(fh.read())

        ft = FunctionTask(fh, function,
                          sample_rate=fh.sample_rate * sample_factor)

        assert ft.shape[0] == fh.shape[0] * sample_factor
        # Apply to everything.
        data1 = ft.read()
        assert ft.tell() == ft.shape[0]
        assert (ft.time - ft.start_time -
                ft.shape[0] / ft.sample_rate) < 1*u.ns
        assert ft.dtype is ref_data.dtype is data1.dtype
        assert np.allclose(ref_data, data1)

        # Seeking and selective zeroing.
        ft.seek(-3, 2)
        assert ft.tell() == ft.shape[0] - 3
        data2 = ft.read()
        assert data2.shape[0] == 3
        assert np.allclose(ref_data[-3:], data2)

        ft.close()

    @pytest.mark.parametrize('samples_per_frame', (None, 15, 1000))
    def test_complex(self, samples_per_frame):
        fh = vdif.open(SAMPLE_VDIF)
        count = fh.shape[0]
        if samples_per_frame is not None:
            count = (count // samples_per_frame) * samples_per_frame
        ref_data = zero_every_8th_sample(fh.read(count))

        ft = ComplexFunctionTask(fh, zero_every_8th_complex,
                                 samples_per_frame=samples_per_frame)

        data1 = ft.read()

        assert np.all(data1 == ref_data)


class TestSquare(object):
    """Test getting simple intensities using Baseband's sample DADA file."""

    def test_squaretask(self):
        """Test squarer."""

        # Load baseband file and get reference intensities.
        fh = dada.open(SAMPLE_DADA)
        ref_data = fh.read()
        ref_data = np.real(ref_data * np.conj(ref_data))

        st = SquareTask(fh)

        # Square everything.
        data1 = st.read()
        assert st.tell() == st.shape[0]
        assert (st.time - st.start_time -
                st.shape[0] / st.sample_rate) < 1*u.ns
        assert st.dtype is ref_data.dtype is data1.dtype
        assert np.allclose(ref_data, data1)

        # Seeking and selective squaring.
        st.seek(-3, 2)
        assert st.tell() == st.shape[0] - 3
        data2 = st.read()
        assert data2.shape[0] == 3
        assert np.allclose(ref_data[-3:], data2)

        st.close()

    def test_freq_sideband_propagation(self):
        fh = vdif.open(SAMPLE_VDIF)
        # Add frequency and sideband information by hand.
        # (Note: sideband is incorrect; just for testing purposes)
        fh.freq = 311.25 * u.MHz + (np.arange(8.) // 2) * 16. * u.MHz
        fh.sideband = np.tile([-1, +1], 4)
        st = SquareTask(fh)
        assert np.all(st.freq == fh.freq)
        assert np.all(st.sideband == st.sideband)

    def test_missing_freq_sideband(self):
        fh = vdif.open(SAMPLE_VDIF)
        st = SquareTask(fh)
        with pytest.raises(AttributeError):
            st.freq
        with pytest.raises(AttributeError):
            st.sideband
