# Licensed under the GPLv3 - see LICENSE

import operator
import numpy as np
import astropy.units as u


class TaskBase(object):

    def __init__(self, ih):
        self.ih = ih
        self.offset = 0
        self._frame_index = None

    @property
    def shape(self):
        """Shape of the (squeezed/subset) stream data."""
        return (self._nsample,) + self.sample_shape

    @property
    def size(self):
        """Total number of component samples in the (squeezed/subset) stream
        data.
        """
        prod = 1
        for dim in self.shape:
            prod *= dim
        return prod

    @property
    def ndim(self):
        """Number of dimensions of the data."""
        return len(self.shape)

    @property
    def sample_rate(self):
        """Number of complete samples per second."""
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, sample_rate):
        # Check if sample_rate is a time rate.
        try:
            sample_rate.to(u.Hz)
        except u.UnitsError as exc:
            exc.args += ("sample rate must have units of 1 / time.",)
            raise
        self._sample_rate = sample_rate

    @property
    def start_time(self):
        """Start time of the data.

        See also `time` and `stop_time`.
        """
        return self.ih.start_time

    @property
    def time(self):
        """Time of the sample pointer's current offset in the data.

        See also `start_time` and `stop_time`.
        """
        return self.tell(unit='time')

    @property
    def stop_time(self):
        """Time at the end of the data, just after the last sample.

        See also `start_time` and `time`.
        """
        return self.start_time + self._nsample / self.sample_rate

    def seek(self, offset, whence=0):
        """Change the stream position."""
        try:
            offset = operator.index(offset)
        except Exception:
            try:
                offset = offset - self.start_time
            except Exception:
                pass
            else:
                whence = 0

            offset = int((offset * self.sample_rate).to(u.one).round())

        if whence == 0 or whence == 'start':
            self.offset = offset
        elif whence == 1 or whence == 'current':
            self.offset += offset
        elif whence == 2 or whence == 'end':
            self.offset = self._nsample + offset
        else:
            raise ValueError("invalid 'whence'; should be 0 or 'start', 1 or "
                             "'current', or 2 or 'end'.")

        return self.offset

    def tell(self, unit=None):
        """Current offset in the file.

        Parameters
        ----------
        unit : `~astropy.units.Unit` or str, optional
            Time unit the offset should be returned in.  By default, no unit
            is used, i.e., an integer enumerating samples is returned. For the
            special string 'time', the absolute time is calculated.

        Returns
        -------
        offset : int, `~astropy.units.Quantity`, or `~astropy.time.Time`
             Offset in current file (or time at current position).
        """
        if unit is None:
            return self.offset

        if unit == 'time':
            return self.start_time + self.tell(unit=u.s)

        return (self.offset / self.sample_rate).to(unit)

    def read(self, count=None, out=None):
        # NOTE: this structure will return an EOF error when attempting to read
        # partial frames, making it identical to fh.read().

        if out is None:
            if count is None or count < 0:
                count = self._nsample - self.offset
                if count < 0:
                    raise EOFError("cannot read from beyond end of input.")
            out = np.empty((count,) + self.shape[1:], dtype=self.dtype)
        else:
            assert out.shape[1:] == self.shape[1:], (
                "'out' should have trailing shape {}".format(self.sample_shape))
            count = out.shape[0]

        offset0 = self.offset
        sample = 0
        while count > 0:
            # For current position, get frame plus offset in that frame.
            frame_index, sample_offset = divmod(self.offset,
                                                self.samples_per_frame)

            if frame_index != self._frame_index:
                self._frame = self._read_frame(frame_index)
                self._frame_index = frame_index

            nsample = min(count, len(self._frame) - sample_offset)
            data = self._frame[sample_offset:sample_offset + nsample]
            # Copy to relevant part of output.
            out[sample:sample + nsample] = data
            sample += nsample
            self.offset = offset0 + sample
            count -= nsample

        return out

    def close(self):
        self.ih.close()
