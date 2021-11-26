# Licensed under the GPLv3 - see LICENSE
import inspect
import itertools
import operator

import numpy as np
import astropy.units as u
import pytest

from baseband_tasks.base import (
    BaseTaskBase, SetAttribute, TaskBase, PaddedTaskBase, Task)
from .common import UseVDIFSample


class ReshapeTime(TaskBase):
    """Dummy class to test accessing task properties and methods.

    `ReshapeTime` simply reshapes blocks of baseband data into frames.
    """

    def __init__(self, ih, n, samples_per_frame=1, fix_shape=False,
                 **kwargs):

        self._n = n = operator.index(n)
        samples_per_frame = operator.index(samples_per_frame)
        sample_rate = ih.sample_rate / n
        sh0 = ih.shape[0] // n if fix_shape else -1
        super().__init__(ih, shape=(sh0, n) + ih.shape[1:],
                         sample_rate=sample_rate,
                         samples_per_frame=samples_per_frame, **kwargs)

    def task(self, data):
        return data.reshape((-1,) + self.sample_shape)


class Multiply(BaseTaskBase):
    """Dummy class to test accessing task properties and methods.

    `Multiply` simply multiplies data with a given factor.
    It is based on BaseTaskBase just to check that defining one's
    own ``_read_frame`` method works.
    """

    def __init__(self, ih, factor):
        self._factor = factor
        super().__init__(ih)

    def _read_frame(self, frame_index):
        self.ih.seek(frame_index * self.samples_per_frame)
        return self.ih.read(self.samples_per_frame) * self._factor


class SquareHat(PaddedTaskBase):
    """Dummy class to test accessing task properties and methods.

    `SquareHat` simply convolves with a set of 1s of given length.
    """

    def __init__(self, ih, n, offset=0, **kwargs):
        self._n = n
        super().__init__(ih, pad_start=n-1-offset, pad_end=offset,
                         **kwargs)

    def task(self, data):
        size = data.shape[0]
        result = data[:size-self._n + 1].copy()
        for i in range(1, self._n):
            result += data[i:size-self._n+i+1]
        return result


class TestSetAttribute(UseVDIFSample):
    def test_set_basics(self):
        expected = self.fh.read()
        frequency = 311.25 * u.MHz + (np.arange(8.) // 2) * 16. * u.MHz
        sideband = np.tile([-1, +1], 4)
        sa = SetAttribute(self.fh, frequency=frequency, sideband=sideband)
        assert np.all(sa.frequency == frequency)
        assert np.all(sa.sideband == sideband)
        for attr in ('start_time', 'sample_rate', 'samples_per_frame',
                     'shape', 'dtype'):
            assert getattr(sa, attr) == getattr(self.fh, attr)
        # Check that frequency does not propagate.
        frequency[...] = 0
        assert np.all(sa.frequency != 0)
        sideband[...] = 0
        assert np.all(np.abs(sa.sideband) == 1)
        # Check data can be read.
        data = sa.read()
        assert np.all(data == expected)
        # Check we didn't magically define polarization.
        with pytest.raises(AttributeError):
            sa.polarization
        sa.close()

    def test_set_start_time(self):
        expected = self.fh.read()
        offset = 0.1 * u.s
        sa = SetAttribute(self.fh, start_time=self.fh.start_time+offset)
        assert sa.start_time == self.fh.start_time+offset
        for attr in ('sample_rate', 'samples_per_frame', 'shape', 'dtype'):
            assert getattr(sa, attr) == getattr(self.fh, attr)
        data = sa.read()
        assert np.all(data == expected)
        sa.seek(10)
        data2 = sa.read(10)
        assert np.all(data2 == expected[10:20])
        sa.seek(10/sa.sample_rate)
        data3 = sa.read(10)
        assert np.all(data3 == expected[10:20])
        sa.seek(sa.start_time+10/self.fh.sample_rate)
        data4 = sa.read(10)
        assert np.all(data4 == expected[10:20])
        for attr in ('frequency', 'sideband', 'polarization'):
            with pytest.raises(AttributeError):
                getattr(sa, attr)
        sa.close()

    def test_need_both_frequency_and_sideband(self):
        frequency = 311.25 * u.MHz + (np.arange(8.) // 2) * 16. * u.MHz
        sideband = np.tile([-1, +1], 4)
        with pytest.raises(ValueError):
            SetAttribute(self.fh, frequency=frequency)
        with pytest.raises(ValueError):
            SetAttribute(self.fh, sideband=sideband)

    def test_samples_per_frame(self):
        expected = self.fh.read()
        sa = SetAttribute(self.fh, samples_per_frame=11111)
        assert sa.shape == (33333, 8)
        data = sa.read()
        assert np.all(data == expected[:33333])

        sa2 = SetAttribute(self.fh, samples_per_frame=11111,
                           shape=self.fh.shape)
        assert sa2.shape == self.fh.shape
        data2 = sa2.read()
        assert np.all(data2 == expected)

    def test_dtype(self):
        expected = self.fh.read()
        sa = SetAttribute(self.fh, dtype='f2')
        assert isinstance(sa.dtype, np.dtype)
        assert sa.dtype == 'f2'
        data = sa.read()
        assert data.dtype == 'f2'
        assert np.all(data == expected.astype('f2'))

    def test_metadata_propagation(self):
        sa = SetAttribute(self.fh)
        sa.meta['parrot'] = 'dead'
        assert not hasattr(sa, 'frequency')
        sa2 = SetAttribute(sa, frequency=300*u.MHz, sideband=1)
        assert sa2.frequency is not None
        assert sa2.sideband is not None
        assert set(sa2.meta) == {'parrot', '__attributes__'}
        assert sa2.meta['parrot'] == 'dead'
        sa2.meta['parrot'] = 'goner'
        assert sa2.meta['parrot'] == 'goner'
        assert sa.meta['parrot'] == 'dead'

    def test_fail_on_unknown_attribute(self):
        with pytest.raises(TypeError):
            SetAttribute(self.fh, freq=1.*u.MHz)


class TestTaskBase(UseVDIFSample):
    def test_basetaskbase(self):
        fh = self.fh
        mh = Multiply(fh, 2.)
        # Check sample pointer.
        assert mh.sample_rate == fh.sample_rate
        assert mh.shape == fh.shape
        assert mh.size == np.prod(mh.shape)
        assert mh.ndim == fh.ndim
        assert mh.tell() == 0
        assert mh.tell(unit='time') == mh.time == mh.start_time
        # stop_time is calculated via frame_rate in baseband and via
        # sample_rate here, so can be slightly different.
        assert abs(mh.stop_time - fh.stop_time) <= 2**-51 * u.day

        expected = fh.read() * 2.
        data = mh.read()
        assert np.all(data == expected)
        assert mh.time == mh.stop_time
        # Check closing.
        mh.close()
        with pytest.raises(ValueError):
            mh.read(1)
        with pytest.raises(AttributeError):
            mh.ih

    @staticmethod
    def convert_time_offset(offset, sample_rate):
        return int((offset * sample_rate).to(u.one).round())

    @pytest.mark.parametrize(('n', 'samples_per_frame', 'fix_shape'),
                             tuple(itertools.product(
                                 [256, 337],
                                 [1, 7, 16],
                                 [True, False])))
    def test_taskbase(self, n, samples_per_frame, fix_shape):
        """Test properties and methods of TaskBase, including
        self-consistency with varying ``n`` and ``samples_per_frame``,
        and checking that we can get the full data if we fix the shape.
        """
        fh = self.fh
        rt = ReshapeTime(fh, n, samples_per_frame=samples_per_frame,
                         fix_shape=fix_shape)
        # Check sample pointer.
        assert rt.sample_rate == fh.sample_rate / n
        nsample = fh.shape[0] // n
        if not fix_shape:
            nsample = (nsample // samples_per_frame) * samples_per_frame
        assert rt.shape == (nsample, n) + fh.sample_shape
        assert rt.size == np.prod(rt.shape)
        assert rt.ndim == fh.ndim + 1
        assert rt.tell() == 0
        assert rt.tell(unit='time') == rt.time == rt.start_time
        assert abs(rt.stop_time
                   - rt.start_time - (nsample * n) / fh.sample_rate) < 1*u.ns

        # Get reference data.
        ref_data = fh.read(nsample * n).reshape((-1, n) + fh.sample_shape)

        # Let's delete fh here, so we check that `rt` keeps it alive.
        del fh

        # Check sequential reading.
        data1 = rt.read()
        assert rt.tell() == rt.shape[0]
        assert abs(rt.time
                   - rt.start_time - rt.shape[0] / rt.sample_rate) < 1*u.ns
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
        assert rt.closed
        with pytest.raises(ValueError):
            rt.read(1)
        with pytest.raises(AttributeError):
            rt.ih

    def test_count_not_changed(self):
        # Regression test for problem reported for baseband in
        # https://github.com/mhvk/baseband/issues/370#issuecomment-577916056
        count = np.array(2)
        with Multiply(self.fh, 2.) as fm:
            fm.read(count)
            assert count == 2

    def test_frequency_sideband_propagation(self):
        fh = self.fh
        # Add frequency and sideband information by hand.
        # (Note: sideband is incorrect; just for testing purposes)
        fh.frequency = 311.25 * u.MHz + (np.arange(8.) // 2) * 16. * u.MHz
        fh.sideband = np.tile([-1, +1], 4)
        with ReshapeTime(fh, 256) as rt:
            assert np.all(rt.sideband == fh.sideband)
            assert np.all(rt.frequency == fh.frequency)
        # Check now closed
        assert rt.closed
        with pytest.raises(ValueError):
            rt.read(1)
        with pytest.raises(AttributeError):
            rt.ih

    def test_frequency_sideband_setting(self):
        fh = self.fh
        # Add frequency and sideband information by hand, broadcasting it.
        # (Note: sideband is incorrect; just for testing purposes)
        frequency_in = 311.25 * u.MHz + (np.arange(8.) // 2) * 16. * u.MHz
        sideband_in = np.tile([-1, +1], 4)
        with ReshapeTime(fh, 256, frequency=frequency_in,
                         sideband=sideband_in) as rt:
            assert np.all(rt.sideband == sideband_in)
            assert np.all(rt.frequency == frequency_in)

    def test_repr_str(self):
        fh = self.fh
        # Add frequency and sideband information by hand, broadcasting it.
        # (Note: sideband is incorrect; just for testing purposes)
        frequency_in = 311.25 * u.MHz + (np.arange(8.) // 2) * 16. * u.MHz
        sideband_in = np.tile([-1, +1], 4)
        rt = ReshapeTime(fh, 256, frequency=frequency_in,
                         sideband=sideband_in)
        r = repr(rt)
        assert r.startswith('ReshapeTime(ih,')
        assert 'n=256' in r
        assert 'frequency=' in r
        assert 'sideband=' in r
        assert 'ih: <VDIF' in r

        s = str(rt)
        # String only keeps named arguments.
        assert s == 'ReshapeTime(ih, n=256)'

    def test_taskbase_exceptions(self):
        """Test exceptions in TaskBase."""
        fh = self.fh
        rt = ReshapeTime(fh, 1024, samples_per_frame=3)

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
        frequency_in = 311.25 * u.MHz + (np.arange(8.) // 2) * 16. * u.MHz
        sideband_in = np.tile([-1, +1], 4)
        with pytest.raises(ValueError):  # sideband missing
            ReshapeTime(fh, 1024, frequency=frequency_in)
        with pytest.raises(ValueError):  # frequency missing
            ReshapeTime(fh, 1024, sideband=sideband_in)
        with pytest.raises(ValueError):  # wrong shape
            ReshapeTime(fh, 1024, sideband=sideband_in,
                        frequency=np.arange(4.)*u.GHz)
        with pytest.raises(ValueError):  # wrong shape
            ReshapeTime(fh, 1024, frequency=frequency_in,
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


class TestTasks(UseVDIFSample):
    """Test applying tasks to Baseband's sample VDIF file."""

    @pytest.mark.parametrize('task, sample_factor',
                             ((zero_channel_4, 1.),
                              (zero_every_8th_sample, 1.),
                              (remove_every_other, 0.5),
                              (double_data, 2.)))
    def test_function_tasks(self, task, sample_factor):
        """Test setting a channel to zero."""

        # Load baseband file and get reference intensities.
        ref_data = task(self.fh.read())

        ft = Task(self.fh, task, sample_rate=self.fh.sample_rate*sample_factor)

        assert ft.shape[0] == self.fh.shape[0] * sample_factor
        # Apply to everything.
        data1 = ft.read()
        assert ft.tell() == ft.shape[0]
        assert abs(ft.time
                   - ft.start_time - ft.shape[0] / ft.sample_rate) < 1*u.ns
        assert ft.dtype is ref_data.dtype is data1.dtype
        assert np.allclose(ref_data, data1)

        # Seeking and selective zeroing.
        ft.seek(-3, 2)
        assert ft.tell() == ft.shape[0] - 3
        data2 = ft.read()
        assert data2.shape[0] == 3
        assert np.allclose(ref_data[-3:], data2)
        ft.close()
        assert ft.closed
        with pytest.raises(ValueError):
            ft.read(1)

    @pytest.mark.parametrize('samples_per_frame', (None, 15, 1000))
    def test_method_task(self, samples_per_frame):
        count = self.fh.shape[0]
        if samples_per_frame is not None:
            count = (count // samples_per_frame) * samples_per_frame
        ref_data = zero_every_8th_sample(self.fh.read(count))

        with Task(self.fh, zero_every_8th_complex,
                  samples_per_frame=samples_per_frame) as ft:
            data1 = ft.read()

        assert np.all(data1 == ref_data)
        assert ft.closed

    def test_task_function_repr(self):
        ft = Task(self.fh, zero_channel_4)
        r = repr(ft)
        assert r.startswith('Task(ih')
        assert 'task=' in r
        assert 'zero_channel_4' in r
        pre, post = r.split('\nih')
        assert 'samples_per_frame' not in pre
        assert 'VDIFStream' in post

    def test_task_method_repr(self):
        ft = Task(self.fh, zero_every_8th_complex)
        r = repr(ft)
        assert r.startswith('Task(ih')
        assert 'task=' in r
        assert 'zero_every_8th' in r
        pre, post = r.split('\nih')
        assert 'samples_per_frame' not in pre
        assert 'VDIFStream' in post

    def test_invalid(self):
        with pytest.raises(Exception):  # Cannot determine function/method.
            Task(self.fh, np.add)

        def trial(data, bla=1):
            return data

        with Task(self.fh, trial) as th:
            assert not inspect.ismethod(th.task)

        def trial2(data, bla, bla2=1):
            return data

        with Task(self.fh, trial2) as th2:
            assert inspect.ismethod(th2.task)

        def trial3(data, bla, bla2, bla3=1):
            return data

        with pytest.raises(Exception):
            Task(self.fh, trial3)


class TestPaddedTaskBase(UseVDIFSample):
    @pytest.mark.parametrize('samples_per_frame', [None, 128])
    @pytest.mark.parametrize('n', [3, 1])
    def test_basics(self, n, samples_per_frame):
        fh = self.fh
        sh = SquareHat(fh, n, samples_per_frame=samples_per_frame)
        expected_size = fh.shape[0] - n + 1
        assert sh.sample_rate == fh.sample_rate
        assert sh.shape == (expected_size,) + fh.shape[1:]
        assert abs(sh.start_time
                   - fh.start_time - (n-1) / fh.sample_rate) < 1. * u.ns
        if n == 3:
            raw = fh.read(12)
            expected = raw[:-2] + raw[1:-1] + raw[2:]
        elif n == 1:
            expected = fh.read(10)
        data = sh.read(10)
        # Also try from end.
        if n == 3:
            fh.seek(-12, 2)
            raw = fh.read(12)
            expected = raw[:-2] + raw[1:-1] + raw[2:]
        elif n == 1:
            fh.seek(-10, 2)
            expected = fh.read(10)
        sh.seek(-10, 2)
        data = sh.read(10)
        assert np.all(data == expected)
        sh.close()
        assert sh.closed

    def test_invalid(self):
        with pytest.raises(ValueError):
            SquareHat(self.fh, -1)
        with pytest.raises(ValueError):
            SquareHat(self.fh, 10, offset=-1)
        with pytest.warns(UserWarning, match='inefficient'):
            SquareHat(self.fh, 10, samples_per_frame=8)


class TestSlicingAndArray(UseVDIFSample):
    @pytest.mark.parametrize('cls,kwargs', [
        (Multiply, dict(factor=2)),
        (ReshapeTime, dict(n=5, samples_per_frame=2, fix_shape=True)),
        (ReshapeTime, dict(n=5, samples_per_frame=2, fix_shape=False)),
        (SquareHat, dict(n=3)),
        (SetAttribute,
         dict(frequency=311.25*u.MHz + (np.arange(8.)//2)*16.*u.MHz,
              sideband=np.tile([-1, +1], 4))),
    ])
    @pytest.mark.parametrize('item', [
        slice(20, 40),
        slice(500),
        slice(-20, None),
        slice(None),
        slice(-20, -10)])
    def test_sample_slice(self, item, cls, kwargs):
        fh = self.fh
        instance = cls(fh, **kwargs)
        expected = instance.read()[item]
        sliced = instance[item]

        start, stop, _ = item.indices(instance.shape[0])
        assert sliced.tell() == 0
        assert sliced.time == sliced.start_time
        assert sliced.sample_rate == instance.sample_rate
        assert abs(sliced.start_time - (
            instance.start_time + start/instance.sample_rate)) < 1.*u.ns
        assert abs(sliced.stop_time - (
            instance.start_time + stop/instance.sample_rate)) < 1.*u.ns
        assert sliced.shape == expected.shape
        if isinstance(instance, SetAttribute):
            assert np.all(sliced.sideband == np.tile([-1, +1], 4))

        sliced.seek(-5, 2)
        data2 = sliced.read()
        assert np.all(data2 == expected[-5:])
        # Also test interpretation as array.
        data = np.array(sliced)
        assert np.all(data == expected)

    def test_asanyarray(self):
        expected = self.fh.read() * 4
        sh = Multiply(self.fh, factor=4.)
        data = np.asanyarray(sh)
        assert np.all(data == expected)

    def test_no_implicit_array(self):
        sh = Multiply(self.fh, factor=4.)
        # Test that we do not just transform to an array
        # (picking functions unlikely ever to be whitelisted)
        with pytest.raises(TypeError):
            np.array(1.) | sh
        with pytest.raises(TypeError):
            np.sin(sh)
        with pytest.raises(TypeError):
            np.rot90(sh)
