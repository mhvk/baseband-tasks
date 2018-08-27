# Licensed under the GPLv3 - see LICENSE

import numpy as np
import astropy.units as u
import operator
import pytest
import itertools

from ..base import TaskBase

from baseband import vdif
from baseband.data import SAMPLE_VDIF


class ReshapeTask(TaskBase):
    """Dummy class to test accessing task properties and methods.

    `ReshapeTask` simply reshapes blocks of baseband data into frames.
    """

    def __init__(self, ih, nchan, samples_per_frame=1):

        self.nchan = operator.index(nchan)
        samples_per_frame = operator.index(samples_per_frame)
        sample_rate = ih.sample_rate / self.nchan

        nsample = samples_per_frame * (ih.shape[0] // self.nchan //
                                       samples_per_frame)
        self._raw_frame_len = self.nchan * samples_per_frame
        shape = (nsample, self.nchan) + ih.sample_shape

        super().__init__(ih, shape, sample_rate, samples_per_frame,
                         ih.dtype)

    def _read_frame(self, frame_index):
        self.ih.seek(frame_index * self._raw_frame_len)
        return self.ih.read(self._raw_frame_len).reshape((-1,) +
                                                         self.sample_shape)


class TestTaskBase(object):

    @staticmethod
    def convert_time_offset(offset, sample_rate):
        return int((offset * sample_rate).to(u.one).round())

    @pytest.mark.parametrize(('nchan', 'samples_per_frame'),
                             tuple(itertools.product([256, 337], [1, 7, 16])))
    def test_taskbase(self, nchan, samples_per_frame):
        """Test properties and methods of TaskBase, including
        self-consistency with varying ``nchan`` and ``samples_per_frame``.
        """
        fh = vdif.open(SAMPLE_VDIF)
        rt = ReshapeTask(fh, nchan, samples_per_frame=samples_per_frame)

        # Check sample pointer.
        assert rt.sample_rate == fh.sample_rate / nchan
        nsample = samples_per_frame * (fh.shape[0] // nchan //
                                       samples_per_frame)
        assert rt.shape == (nsample, nchan) + fh.sample_shape
        assert rt.size == np.prod(rt.shape)
        assert rt.ndim == fh.ndim + 1
        assert rt.tell() == 0
        assert rt.tell(unit='time') == rt.time == rt.start_time
        assert abs(rt.stop_time - rt.start_time -
                   (nsample * nchan) / fh.sample_rate) < 1*u.ns

        # Get reference data.
        ref_data = fh.read(nsample * nchan).reshape((-1, nchan) +
                                                    fh.sample_shape)
        fh.seek(0)

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

    def test_taskbase_exceptions(self):
        """Test exceptions in TaskBase."""

        with vdif.open(SAMPLE_VDIF) as fh:
            rt = ReshapeTask(fh, 1024, samples_per_frame=3)

            # Check that reading beyond the bounds of the data leads to an
            # error.
            rt.seek(0, 2)
            with pytest.raises(EOFError):
                rt.read(1)
            rt.seek(-2, 'end')
            with pytest.raises(EOFError):
                rt.read(10)
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
