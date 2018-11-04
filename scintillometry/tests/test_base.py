# Licensed under the GPLv3 - see LICENSE

import numpy as np
import astropy.units as u
import operator
import pytest
import itertools

from ..base import TaskBase, Task

from baseband import vdif
from baseband.data import SAMPLE_VDIF


class Reshape(TaskBase):
    """Dummy class to test accessing task properties and methods.

    `Reshape` simply reshapes blocks of baseband data into frames.
    """

    def __init__(self, ih, n, samples_per_frame=1, **kwargs):

        n = operator.index(n)
        samples_per_frame = operator.index(samples_per_frame)
        sample_rate = ih.sample_rate / n

        nsample = samples_per_frame * (ih.shape[0] // n // samples_per_frame)
        super().__init__(ih, shape=(nsample, n) + ih.shape[1:],
                         sample_rate=sample_rate,
                         samples_per_frame=samples_per_frame, **kwargs)

    def task(self, data):
        return data.reshape((-1,) + self.sample_shape)


class TestTaskBase:

    @staticmethod
    def convert_time_offset(offset, sample_rate):
        return int((offset * sample_rate).to(u.one).round())

    @pytest.mark.parametrize(('n', 'samples_per_frame'),
                             tuple(itertools.product([256, 337], [1, 7, 16])))
    def test_taskbase(self, n, samples_per_frame):
        """Test properties and methods of TaskBase, including
        self-consistency with varying ``n`` and ``samples_per_frame``.
        """
        fh = vdif.open(SAMPLE_VDIF)
        rt = Reshape(fh, n, samples_per_frame=samples_per_frame)

        # Check sample pointer.
        assert rt.sample_rate == fh.sample_rate / n
        nsample = samples_per_frame * (fh.shape[0] // n // samples_per_frame)
        assert rt.shape == (nsample, n) + fh.sample_shape
        assert rt.size == np.prod(rt.shape)
        assert rt.ndim == fh.ndim + 1
        assert rt.tell() == 0
        assert rt.tell(unit='time') == rt.time == rt.start_time
        assert abs(rt.stop_time - rt.start_time -
                   (nsample * n) / fh.sample_rate) < 1*u.ns

        # Get reference data.
        ref_data = fh.read(nsample * n).reshape((-1, n) + fh.sample_shape)

        # Check sequential reading.
        data1 = rt.read()
        assert rt.tell() == rt.shape[0]
        assert abs(rt.time - rt.start_time -
                   rt.shape[0] / rt.sample_rate) < 1*u.ns
        assert rt.dtype is data1.dtype
        assert np.all(ref_data == data1)

        # Check seeking and selective decode.
        rt.seek(-7, 2)
        assert rt.tell() == rt.shape[0] - 7
        data2 = rt.read()
        assert data2.shape[0] == 7
        assert np.all(data2 == ref_data[-7:])
        sec_offset = -0.25 * u.ms
        rt.seek(sec_offset, 'end')
        assert rt.tell() == rt.shape[0] + self.convert_time_offset(
            sec_offset, rt.sample_rate)
        assert rt.tell(unit=u.ms) == (rt.tell() / rt.sample_rate).to(u.ms)
        current_offset = rt.tell()
        rt.seek(2, 'current')
        assert rt.tell() == current_offset + 2
        time_offset = rt.start_time + 0.13 * u.ms
        rt.seek(time_offset, 'start')
        assert rt.tell() == self.convert_time_offset(
            time_offset - rt.start_time, rt.sample_rate)

        # Check reading to external array.
        out = np.empty((11,) + rt.sample_shape)
        rt.seek(0)
        rt.read(out=out)
        assert np.all(out == ref_data[:11])

        # Check closing.
        rt.close()
        assert fh.closed

    def test_frequency_sideband_propagation(self):
        fh = vdif.open(SAMPLE_VDIF)
        # Add frequency and sideband information by hand.
        # (Note: sideband is incorrect; just for testing purposes)
        fh.frequency = 311.25 * u.MHz + (np.arange(8.) // 2) * 16. * u.MHz
        fh.sideband = np.tile([-1, +1], 4)
        rt = Reshape(fh, 256)
        assert np.all(rt.sideband == fh.sideband)
        assert np.all(rt.frequency == fh.frequency)

    def test_frequency_sideband_setting(self):
        fh = vdif.open(SAMPLE_VDIF)
        # Add frequency and sideband information by hand, broadcasting it.
        # (Note: sideband is incorrect; just for testing purposes)
        frequency_in = 311.25 * u.MHz + (np.arange(8.) // 2) * 16. * u.MHz
        sideband_in = np.tile([-1, +1], 4)
        rt = Reshape(fh, 256, frequency=frequency_in, sideband=sideband_in)
        assert np.all(rt.sideband == sideband_in)
        assert np.all(rt.frequency == frequency_in)

    def test_taskbase_exceptions(self):
        """Test exceptions in TaskBase."""

        with vdif.open(SAMPLE_VDIF) as fh:
            rt = Reshape(fh, 1024, samples_per_frame=3)

            # Check that reading beyond the bounds of the data leads to an
            # error.
            rt.seek(0, 2)
            with pytest.raises(EOFError):
                rt.read(1)
            rt.seek(-2, 'end')
            with pytest.raises(EOFError):
                rt.read(10)
            rt.seek(-2, 'end')
            with pytest.raises(EOFError):
                rt.read(out=np.empty((5,) + rt.sample_shape))
            rt.seek(-4, 'start')
            with pytest.raises(OSError):
                rt.read(1)

            # Check invalid whence.
            with pytest.raises(ValueError):
                rt.seek(1, 'now')
            with pytest.raises(ValueError):
                rt.seek(1, 3)

            # Check external array shape mismatch raises an error.
            with pytest.raises(AssertionError):
                rt.read(out=np.empty(3))

            # Check missing frequency/sideband definitions
            with pytest.raises(AttributeError):
                rt.frequency
            with pytest.raises(AttributeError):
                rt.sideband
            with pytest.raises(ValueError):
                Reshape(fh, 1024, samples_per_frame=3,
                        frequency=np.arange(4.)*u.GHz)
            with pytest.raises(ValueError):
                Reshape(fh, 1024, samples_per_frame=3,
                        sideband=np.ones((2, 8), dtype=int))


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


class TestTasks:
    """Test applying tasks to Baseband's sample VDIF file."""

    @pytest.mark.parametrize('task, sample_factor',
                             ((zero_channel_4, 1.),
                              (zero_every_8th_sample, 1.),
                              (remove_every_other, 0.5),
                              (double_data, 2.)))
    def test_function_tasks(self, task, sample_factor):
        """Test setting a channel to zero."""

        # Load baseband file and get reference intensities.
        fh = vdif.open(SAMPLE_VDIF)
        ref_data = task(fh.read())

        ft = Task(fh, task, sample_rate=fh.sample_rate * sample_factor)

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
    def test_method_task(self, samples_per_frame):
        fh = vdif.open(SAMPLE_VDIF)
        count = fh.shape[0]
        if samples_per_frame is not None:
            count = (count // samples_per_frame) * samples_per_frame
        ref_data = zero_every_8th_sample(fh.read(count))

        ft = Task(fh, zero_every_8th_complex,
                  samples_per_frame=samples_per_frame)

        data1 = ft.read()

        assert np.all(data1 == ref_data)
