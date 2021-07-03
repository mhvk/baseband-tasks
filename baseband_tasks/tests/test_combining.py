# Licensed under the GPLv3 - see LICENSE

import pytest
import numpy as np
from numpy.testing import assert_array_equal

from baseband_tasks.base import SetAttribute
from baseband_tasks.shaping import GetItem, Reshape
from baseband_tasks.combining import Concatenate, Stack, CombineStreams

from .common import UseVDIFSampleWithAttrs


class TestConcatenate(UseVDIFSampleWithAttrs):
    @pytest.mark.parametrize('items',
                             ((slice(None, 4), slice(4, None)),
                              (slice(None, 3), slice(3, None)),
                              (slice(None, 2), slice(2, 6), slice(6, None)),
                              (slice(None),)))
    def test_concatenate_simple(self, items):
        fh = self.fh
        expected_data = fh.read()
        fhs = [GetItem(fh, item) for item in items]
        ch = Concatenate(fhs)
        assert ch.shape == fh.shape
        assert ch.start_time == fh.start_time
        assert ch.sample_rate == fh.sample_rate
        assert ch.dtype == fh.dtype
        assert_array_equal(ch.frequency, fh.frequency)
        assert_array_equal(ch.sideband, fh.sideband)
        assert_array_equal(ch.polarization, fh.polarization)

        data = ch.read()
        assert_array_equal(data, expected_data)

        r = repr(ch)
        assert r.startswith('Concatenate(ihs)')
        assert f"ihs: {len(fhs)} streams" in r
        ch.close()

    @pytest.mark.parametrize('samples_per_frame', [None, 10, 10000])
    def test_time_offsets(self, samples_per_frame):
        fh = self.fh
        s1 = GetItem(fh, slice(None, 4))
        s2 = SetAttribute(GetItem(fh, slice(4, None)),
                          start_time=fh.start_time+10/fh.sample_rate)
        fh_data = self.fh.read()
        expected_data = np.concatenate((fh_data[10:, :4], fh_data[:-10, 4:]),
                                       axis=1)
        ch = Concatenate([s1, s2], samples_per_frame=samples_per_frame)
        assert ch.start_time == s2.start_time
        assert ch.shape == (fh.shape[0] - 10,) + fh.sample_shape
        if samples_per_frame is None:
            assert ch.samples_per_frame == s1.samples_per_frame
        else:
            assert ch.samples_per_frame == samples_per_frame
        assert ch.sample_rate == fh.sample_rate
        assert ch.dtype == fh.dtype
        assert_array_equal(ch.frequency, fh.frequency)
        assert_array_equal(ch.sideband, fh.sideband)
        assert_array_equal(ch.polarization, fh.polarization)

        data = ch.read()
        assert_array_equal(data, expected_data)
        ch.close()

    def test_wrong_axis(self):
        with pytest.raises(ValueError):
            Concatenate((self.fh, self.fh), axis=0)
        with pytest.raises(ValueError):
            Concatenate((self.fh, self.fh), axis=-2)

    def test_wrong_time(self):
        fh = self.fh
        s1 = GetItem(fh, slice(None, 4))
        s2 = SetAttribute(GetItem(fh, slice(4, None)),
                          start_time=fh.start_time+1.5/fh.sample_rate)
        with pytest.raises(ValueError):
            Concatenate([s1, s2])


class TestStack(UseVDIFSampleWithAttrs):
    @pytest.mark.parametrize('axis', (1, 2, -1))
    def test_stack_simple(self, axis):
        fh = self.fh
        data = fh.read()
        expected_data = np.stack((data[:, :4], data[:, 4:]), axis)
        fh0 = GetItem(fh, slice(None, 4))
        fh1 = GetItem(fh, slice(4, None))
        ch = Stack((fh0, fh1), axis=axis)

        assert ch.shape == expected_data.shape
        assert ch.start_time == fh.start_time
        assert ch.sample_rate == fh.sample_rate
        assert ch.dtype == fh.dtype
        assert_array_equal(ch.sideband, fh.sideband)
        if axis == 1:
            assert_array_equal(ch.frequency, fh.frequency.reshape(2, 4))
            assert_array_equal(ch.polarization, fh.polarization[:4])
        else:
            assert_array_equal(ch.frequency, fh.frequency.reshape(2, 4).T)
            assert_array_equal(ch.polarization,
                               fh.polarization[:4].reshape(4, 1))

        data = ch.read()
        assert_array_equal(data, expected_data)
        ch.close()

    @pytest.mark.parametrize('axis', (1, 2, -1))
    def test_stack_single_entry(self, axis):
        fh = self.fh
        data = fh.read()
        expected_data = np.stack((data,), axis=axis)
        ch = Stack((fh,), axis=axis)

        assert ch.shape == expected_data.shape
        assert ch.start_time == fh.start_time
        assert ch.sample_rate == fh.sample_rate
        assert ch.dtype == fh.dtype
        assert_array_equal(ch.sideband, fh.sideband)
        if axis == 1:
            assert_array_equal(ch.frequency, fh.frequency)
            assert_array_equal(ch.polarization, fh.polarization)
        else:
            assert_array_equal(ch.frequency, fh.frequency[:, np.newaxis])
            assert_array_equal(ch.polarization, fh.polarization[:, np.newaxis])

        data = ch.read()
        assert_array_equal(data, expected_data)
        ch.close()

    def test_wrong_axis(self):
        with pytest.raises(ValueError):
            Stack((self.fh, self.fh), axis=0)
        with pytest.raises(ValueError):
            Stack((self.fh, self.fh), axis=-3)


class TestCombineStreams(UseVDIFSampleWithAttrs):
    def test_combine_callable(self):
        def doublestack(data):
            # concatenate frequencies, stack polarizations.
            data = [np.stack(data[i:i+2], axis=1) for i in range(0, 8, 2)]
            return np.stack(data, axis=1)

        fhs = [GetItem(self.fh, i) for i in range(8)]
        ch = CombineStreams(fhs, doublestack)
        expected = Reshape(self.fh, (4, 2))

        assert ch.shape == expected.shape
        assert ch.start_time == expected.start_time
        assert ch.sample_rate == expected.sample_rate
        assert ch.dtype == expected.dtype
        assert_array_equal(ch.frequency, expected.frequency)
        assert_array_equal(ch.sideband, expected.sideband)
        assert_array_equal(ch.polarization, expected.polarization)

        data = ch.read()
        expected_data = expected.read()
        assert_array_equal(data, expected_data)

        r = repr(ch)
        assert r.startswith('CombineStreams(ihs')
        assert 'ihs: 8 streams' in r

        ch.close()
        expected.close()
