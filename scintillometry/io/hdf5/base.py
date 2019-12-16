# Licensed under the GPLv3 - see LICENSE
"""Interfaces for reading and writing from an internal HDF5 format."""

import h5py
import numpy as np

from baseband.vlbi_base.base import (
    VLBIStreamBase, VLBIStreamReaderBase, VLBIStreamWriterBase)

from .header import HDF5Header
from .payload import HDF5Payload


class HDF5StreamBase(VLBIStreamBase):
    def __init__(self, fh_raw, header0, squeeze=True, subset=(),
                 fill_value=0., verify=True):
        if hasattr(header0, 'bps'):
            bps = header0.bps
            complex_data = header0.complex_data
        else:
            complex_data = header0.dtype.kind == 'c'
            bps = header0.dtype.itemsize * 8 // (2 if complex_data else 1)
        super().__init__(
            fh_raw=fh_raw, header0=header0, sample_rate=header0.sample_rate,
            samples_per_frame=header0.samples_per_frame,
            unsliced_shape=header0.sample_shape, bps=bps,
            complex_data=complex_data, squeeze=squeeze, subset=subset,
            fill_value=fill_value, verify=verify)

    @property
    def dtype(self):
        return self.header0.dtype

    @property
    def frequency(self):
        return self.header0.frequency

    @property
    def sideband(self):
        return self.header0.sideband

    @property
    def polarization(self):
        return self.header0.polarization

    @property
    def bps(self):
        return self.header0.bps


class HDF5StreamReader(HDF5StreamBase, VLBIStreamReaderBase):
    def __init__(self, fh_raw, squeeze=True, subset=(), fill_value=0.,
                 verify=True):
        header0 = HDF5Header.fromfile(fh_raw)
        super().__init__(fh_raw, header0=header0,
                         squeeze=squeeze, subset=subset,
                         fill_value=fill_value, verify=verify)

    @property
    def _last_header(self):
        return self.header0

    def _read_frame(self, index):
        assert index == 0
        # More logical as reading a Frame!
        return HDF5Payload.fromfile(self.fh_raw, self.header0)


class HDF5StreamWriter(HDF5StreamBase, VLBIStreamWriterBase):
    def __init__(self, fh_raw, header0=None, squeeze=True,
                 template=None, **kwargs):
        if header0 is None:
            header0 = HDF5Header.fromvalues(template=template, **kwargs)
        super().__init__(fh_raw, header0, squeeze=squeeze)

    def _make_frame(self, index):
        assert index == 0
        self.header0.tofile(self.fh_raw)
        # More logical as creating a Frame!
        return HDF5Payload.fromfile(self.fh_raw, self.header0)

    def _write_frame(self, frame, valid=True):
        assert valid, 'cannot deal with invalid data yet'


def open(filename, mode='r', **kwargs):
    """Open an HDF5 file as a stream."""
    if not isinstance(filename, h5py.File):
        filename = h5py.File(filename, mode)
    if 'r' in mode:
        return HDF5StreamReader(filename)
    elif 'w' in mode:
        return HDF5StreamWriter(filename, **kwargs)
    else:
        raise ValueError('unknown mode {}'.format(mode))
