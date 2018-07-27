# Licensed under the GPLv3 - see LICENSE

import operator
import numpy as np
import astropy.units as u


class ModuleBase(object):

    def __init__(self, ih):
        self.ih = ih
        if self.ih.tell():
            self.ih.seek(0)
        self.offset = 0
        self._block_index = None

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
        # partial blocks, making it identical to fh.read().

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
            block_index, sample_offset = divmod(self.offset,
                                                self.samples_per_block)

            if block_index != self._block_index:
                self._block = self._read_block(block_index)
                self._block_index = block_index

            nsample = min(count, len(self._block) - sample_offset)
            data = self._block[sample_offset:sample_offset + nsample]
            # Copy to relevant part of output.
            out[sample:sample + nsample] = data
            sample += nsample
            self.offset = offset0 + sample
            count -= nsample

        return out

    def close(self):
        self.ih.close()
