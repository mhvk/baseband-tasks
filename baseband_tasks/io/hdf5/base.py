# Licensed under the GPLv3 - see LICENSE
"""Interfaces for reading and writing from an internal HDF5 format.

In this format, each HDF5 `~h5py.File` has 'header' and 'payload'
`h5py.Dataset` instances, with the header consisting of yaml-encoded
keywords describing the start time, sample rate, etc., and the payload
consisting of either plain numpy data, or data encoded following the
VDIF standard.
"""
from baseband.base.base import StreamReaderBase, StreamWriterBase
from .header import HDF5Header
from .payload import HDF5Payload
from .frame import HDF5Frame


__all__ = ['HDF5StreamBase', 'HDF5StreamReader', 'HDF5StreamWriter',
           'open']


class HDF5StreamBase:
    def __init__(self, fh_raw, header0, **kwargs):
        if hasattr(header0, 'bps'):
            bps = header0.bps
            complex_data = header0.complex_data
        else:
            complex_data = header0.dtype.kind == 'c'
            # We do need to pass on bps to the base class, even though
            # we do not need it.  We override the property below.
            bps = header0.dtype.itemsize * 8 // (2 if complex_data else 1)
        super().__init__(fh_raw=fh_raw, header0=header0,
                         complex_data=complex_data, bps=bps, **kwargs)

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

    @property
    def closed(self):
        try:
            self.fh_raw.filename
        except Exception:
            return True
        else:
            return False

    def __repr__(self):
        name = self.fh_raw.filename if not self.closed else '<closed>'
        return ("<{s.__class__.__name__} name={name} "
                "offset={s.offset}\n"
                "    sample_rate={s.sample_rate},"
                " samples_per_frame={s.samples_per_frame},\n"
                "    sample_shape={s.sample_shape}, {bps_or_dtype},\n"
                "    {sub}start_time={s.start_time.isot}>"
                .format(s=self, name=name,
                        sub=('subset={0}, '.format(self.subset)
                             if getattr(self, 'subset', None) else ''),
                        bps_or_dtype=('bps={0}'.format(self.bps)
                                      if hasattr(self, 'bps') else
                                      'dtype={0}'.format(self.dtype))))


class HDF5StreamReader(HDF5StreamBase, StreamReaderBase):
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
        return HDF5Frame.fromfile(self.fh_raw)


class HDF5StreamWriter(HDF5StreamBase, StreamWriterBase):
    def __init__(self, fh_raw, header0=None, template=None, **kwargs):
        # By default, do not squeeze if we're basing off a template.
        # Presumably, we will pass in data exactly like it!
        squeeze = kwargs.pop('squeeze', template is None)
        if header0 is None:
            header0 = HDF5Header.fromvalues(template=template, **kwargs)
        super().__init__(fh_raw, header0, squeeze=squeeze)

    def _make_frame(self, index):
        assert index == 0
        self.header0.tofile(self.fh_raw)
        payload = HDF5Payload.fromfile(self.fh_raw, self.header0)
        return HDF5Frame(self.header0, payload)

    @property
    def shape(self):
        return (self.header0.samples_per_frame,) + self.sample_shape

    def __setitem__(self, item, value):
        start, stop, step = item.indices(self.shape[0])
        assert start == self.offset, 'Can only assign right following pointer.'
        assert step == 1, 'unity step size onlyin supported'
        assert len(value) == stop-start, 'number of samples should match.'
        self.write(value)


def open(filename, mode='r', **kwargs):
    """Open an HDF5 file as a stream.

    This yields a filehandle wrapped around an HDF file that has methods
    for reading and writing to the file as if it were a stream of samples.

    Parameters
    ----------
    name : str, `~h5py.File`, of `~h5py.Group`
        File name, filehandle, or group containing header and payload.
    mode : {'r', 'w'}, optional
        Whether to open for reading (default) or writing.
    **kwargs
        Additional arguments when opening the file for writing.

    --- For reading : """ \
    """(see :class:`~baseband_tasks.io.hdf5.base.HDF5StreamReader`)

    squeeze : bool, optional
        If `True` (default), remove any dimensions of length unity from
        decoded data.
    subset : indexing object, optional
        Specific components of the complete sample to decode (after possibly
        squeezing).

    --- For writing : """ \
    """(see :class:`~baseband_tasks.io.hdf5.base.HDF5StreamWriter`)

    header0 : `~baseband_tasks.io.hdf5.HDF5Header`
        Header for the first frame, holding time information, etc.  Can instead
        give a ``template`` or keyword arguments to construct a header from.
    squeeze : bool, optional
        If `True`, writer accepts squeezed arrays as input, and adds any
        dimensions of length unity. Default: `True` unless a template is given.
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

    --- Header keywords : """ \
    """(see :meth:`baseband_tasks.io.hdf5.HDF5Header.fromvalues`)

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
    encoded_dtype : str or `~numpy.dtype`, optional
        Data type of the encoded data.  By default, equal to ``dtype``, but
        can be used to reduce the precision, e.g., to half-precision with
        'f2' for real-valued data or the custom 'c4' dtype for complex.
        Should only be given if ``bps`` and ``complex_data`` are not given.
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
        :class:`~baseband_tasks.io.hdf5.base.HDF5StreamReader` or
        :class:`~baseband_tasks.io.hdf5.base.HDF5StreamWriter` (stream).

    """
    import h5py

    if mode not in {'r', 'rs', 'w', 'ws'}:
        raise ValueError('unknown mode {}'.format(mode))

    mode = mode[0]
    if not isinstance(filename, h5py.File):
        filename = h5py.File(filename, mode)
    if mode == 'r':
        return HDF5StreamReader(filename, **kwargs)
    else:
        return HDF5StreamWriter(filename, **kwargs)
