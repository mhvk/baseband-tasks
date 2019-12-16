# Licensed under the GPLv3 - see LICENSE
"""Interfaces for reading and writing from an internal HDF5 format.

In this format, each HDF5 `~h5py:File` has 'header' and 'payload'
`h5py:Dataset` instances, with the header consisting of yaml-encoded
keywords describing the start time, sample rate, etc., and the payload
consisting of either plain numpy data, or data encoded following the
VDIF standard.
"""

import h5py

from baseband.vlbi_base.base import (
    VLBIStreamBase, VLBIStreamReaderBase, VLBIStreamWriterBase)

from .header import HDF5Header
from .payload import HDF5Payload


__all__ = ['HDF5StreamBase', 'HDF5StreamReader', 'HDF5StreamWriter',
           'open']


class HDF5StreamBase(VLBIStreamBase):
    def __init__(self, fh_raw, header0, squeeze=True, subset=(),
                 fill_value=0., verify=True):
        if hasattr(header0, 'bps'):
            bps = header0.bps
            complex_data = header0.complex_data
        else:
            complex_data = header0.dtype.kind == 'c'
            # We do need to pass on bps to the base class, even though
            # we do not need it.  We override the property below.
            bps = header0.dtype.itemsize * 8 // (2 if complex_data else 1)
        super().__init__(
            fh_raw=fh_raw, header0=header0, sample_rate=header0.sample_rate,
            samples_per_frame=header0.samples_per_frame,
            unsliced_shape=header0.sample_shape, bps=bps,
            complex_data=complex_data, squeeze=squeeze, subset=subset,
            fill_value=fill_value, verify=verify)

    @property
    def dtype(self):
        if hasattr(self.header0, 'dtype'):
            return self.header0.dtype
        else:
            return super().dtype

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
        """Bits per elementary sample.

        Only available if the HDF5 payload is encoded.
        """
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
        # Possibly more logical as reading a Frame!
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
        # Possibly more logical as creating a Frame!
        return HDF5Payload.fromfile(self.fh_raw, self.header0)

    def _write_frame(self, frame, valid=True):
        assert valid, 'cannot deal with invalid data yet'


def open(filename, mode='r', **kwargs):
    """Open an HDF5 file as a stream.

    This yields a filehandle wrapped around an HDF file that has methods
    for reading and writing to the file as if it were a stream of samples.

    Parameters
    ----------
    name : str, `~h5py:File`, of `~h5py:Group`
        File name, filehandle, or group containing header and payload.
    mode : {'r', 'w'}, optional
        Whether to open for reading (default) or writing.
    **kwargs
        Additional arguments when opening the file for writing.

    --- For reading a stream : (see :class:`~scintillometry.io.hdf5.base.HDF5StreamReader`)

    squeeze : bool, optional
        If `True` (default), remove any dimensions of length unity from
        decoded data.
    subset : indexing object, optional
        Specific components of the complete sample to decode (after possibly
        squeezing).

    --- For writing a stream : (see :class:`~scintillometry.io.hdf5.base.HDF5StreamWriter`)

    header0 : `~scintillometry.io.hdf5.HDF5Header`
        Header for the first frame, holding time information, etc.  Can instead
        give a ``template`` or keyword arguments to construct a header from.
    squeeze : bool, optional
        If `True` (default), writer accepts squeezed arrays as input, and adds
        any dimensions of length unity.
    template : header or stream template, optional
        Must have attributes defining the required header keywords (see below).
    whole : bool, optional
        If `True`, assume a header for the complete stream is wanted,
        and use 'start_time' for the 'time' and the total number of
        samples for 'samples_per_frame'.  Default: `True` if the template
        has both 'start_time' and 'shape' (i.e., for streams).  Ignored if
        ``template`` is not given.
    verify : bool, optional
        Whether to do basic verification.  Default: `True`.
    **kwargs
        Any additional values for constructing a header.  If ``template`` is
        given, these will override its values.

    --- Header keywords : (see :meth:`~scintillometry.io.hdf5.HDF5Header.fromvalues`)

    sample_shape : tuple
        Shape of the individual samples.
    samples_per_frame : int
        Number of complete samples per frame.  Typically, only one frame is
        used, so this is the total number of samples to be stored.
    sample_rate : `~astropy.units.Quantity`
        Number of complete samples per second, i.e. the rate at which each
        channel of each polarization is sampled.
    dtype : str or `~numpy.dtype`
        Data type of the raw data.  Should only be given if ``bps`` and
        ``complex_data`` are not given.
    complex_data : bool, optional
        Whether encoded data are complex (default: `False`).  Should only
        be given if ``dtype`` is not given.
    bps : int, optional
        Bits per elementary sample, i.e. per real or imaginary component for
        complex data (default: 8).  Should only be given if ``dtype`` is not
        given.

    Returns
    -------
    Filehandle
        :class:`~scintillometry.io.hdf5.base.HDF5StreamReader` or
        :class:`~scintillometry.io.hdf5.base.HDF5StreamWriter` (stream).

    """
    if mode not in {'r', 'rs', 'w', 'ws'}:
        raise ValueError('unknown mode {}'.format(mode))

    mode = mode[0]
    if not isinstance(filename, h5py.File):
        filename = h5py.File(filename, mode)
    if mode == 'r':
        return HDF5StreamReader(filename, **kwargs)
    else:
        return HDF5StreamWriter(filename, **kwargs)