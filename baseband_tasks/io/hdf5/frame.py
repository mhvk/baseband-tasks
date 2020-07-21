# Licensed under the GPLv3 - see LICENSE
"""Frame for HDF5 format."""
from baseband.base.frame import FrameBase

from .header import HDF5Header
from .payload import HDF5Payload


__all__ = ['HDF5Frame']


class HDF5Frame(FrameBase):
    """Representation of a HDF5 frame, consisting of a header and payload.

    Parameters
    ----------
    header : `~baseband_tasks.io.hdf5.HDF5Header`
        Wrapper around the yaml-encoded header information.
    payload : `~baseband_tasks.io.hdf5.HDF5Payload`
        Wrapper around the payload, provding mechanisms to decode it.
    valid : bool or None
        Whether the data are valid.
    verify : bool
        Whether to do basic verification of integrity (default: `True`)

    Notes
    -----
    The Frame can also be read instantiated using class methods:

      fromfile : read header and payload from a filehandle

      fromdata : encode data as payload

    Of course, one can also do the opposite:

      tofile : method to write header and payload to filehandle

      data : property that yields full decoded payload

    A number of properties are defined: `shape`, `dtype` and `size` are
    the shape, type and number of complete samples of the data array, and
    `nbytes` the frame size in bytes.  Furthermore, the frame acts as a
    dictionary, with keys those of the header.  Any attribute that is not
    defined on the frame itself, such as ``.time`` will be looked up on the
    header as well.
    """

    _header_class = HDF5Header
    _payload_class = HDF5Payload

    @property
    def valid(self):
        """Whether frame contains valid data."""
        return self._valid

    @valid.setter
    def valid(self, valid):
        assert valid, 'cannot deal with invalid data yet'
        self._valid = bool(valid)

    @classmethod
    def fromfile(cls, fh, valid=True, verify=True):
        """Read a frame from a filehandle.

        Parameters
        ----------
        fh : filehandle
            To read the header and payload from.
        valid : bool
            Whether the data are valid.  Default: `True`.
        verify : bool
            Whether to do basic checks of frame integrity (default: `True`).
        """
        header = cls._header_class.fromfile(fh, verify=verify)
        payload = cls._payload_class.fromfile(fh, header=header)
        return cls(header, payload, valid=valid, verify=verify)

    @classmethod
    def fromdata(cls, data, header=None):
        return NotImplementedError('HDF5 frames cannot yet be initialized '
                                   'from data')

    def tofile(self, fh):
        """Write encoded frame to filehandle."""
        # Payloads and headers are always memory mapped,
        # so no need to do anything.
        pass
