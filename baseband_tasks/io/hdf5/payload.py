# Licensed under the GPLv3 - see LICENSE
"""Payload for HDF5 format."""
from functools import reduce
import operator

import numpy as np

from baseband.vdif import VDIFPayload
from baseband.base.payload import PayloadBase


__all__ = ['HDF5Payload', 'HDF5RawPayload', 'HDF5CodedPayload',
           'HDF5DatasetWrapper', 'DTYPE_C4']


# Ideally, we'd use 'r' and 'i' here, to match the use for
# other complex numbers inside h5py, but unfortunately that
# needs a numpy 'c4' dtype to actually exist.
DTYPE_C4 = np.dtype([('real', '<f2'), ('imag', '<f2')])
"""Numpy dtype used to encode half-precision complex numbers."""


class HDF5Payload:
    """Container for decoding and encoding HDF5 payloads.

    The data will be taken to represent their values directly unless
    the header has a ``bps`` attribute, or ``bps`` is given explicitly.

    Parameters
    ----------
    words : `~h5py.Dataset`
        Array containg data as stored in the HDF5 file, which possibly
        are encoded similar to a VDIF payload.
    header : `~baseband_tasks.io.hdf5.HDF5Header`, optional
        Header providing information about whether, and if so, how the payload
        is encoded. If not given and if the data are encoded, then the
        following should be passed in.
    sample_shape : tuple, optional
        Shape of the samples; e.g., (nchan,).  Default: ().
    bps : int, optional
        Number of bits per sample part (i.e., per channel and per real or
        imaginary component).  No default.
    complex_data : bool, optional
        Whether data are complex.  Default: `False`.
    """

    def __new__(cls, words, header=None, **kwargs):
        if 'bps' in kwargs or hasattr(header, 'bps'):
            cls = HDF5CodedPayload
        else:
            cls = HDF5RawPayload
        return super().__new__(cls)

    @classmethod
    def fromfile(cls, fh, header=None):
        """Get payload words from HDF5 file or group.

        Parameters
        ----------
        fh : `~h5py.File` or `~h5py.Group`
            Handle to the HDF5 file/group which has an 'payload' dataset.
            If the payload does not exist, it will be created.
        header : `~baseband_tasks.io.hdf5.HDF5Header`, optional
            Must be given for encoded payloads, or to create a payload.
        """
        if 'payload' in fh:
            return cls(fh['payload'], header)

        if hasattr(header, 'bps'):
            nsample = reduce(operator.mul, header.sample_shape,
                             header.samples_per_frame)
            shape = ((header.bps * (2 if header.complex_data else 1)
                      * nsample + 31) // 32,)
        else:
            shape = (header.samples_per_frame,) + header.sample_shape

        words = fh.create_dataset('payload', shape=shape,
                                  dtype=header.encoded_dtype)

        return cls(words, header)


class HDF5DatasetWrapper:
    """Make a HDF5 Dataset look a bit more like ndarray.

    In particular, adds ``nbytes`` and ``itemsize`` properties,
    and implements a ``view`` method.
    """
    def __init__(self, words):
        self.words = words

    def __getattr__(self, attr):
        if not attr.startswith('_'):
            try:
                return getattr(self.words, attr)
            except AttributeError:
                pass

        return self.__getattribute__(attr)

    @property
    def nbytes(self):
        return self.words.size * self.itemsize

    @property
    def itemsize(self):
        return self.words.dtype.itemsize

    def __getitem__(self, item):
        return self.words[item]

    def __setitem__(self, item, value):
        self.words[item] = value

    def view(self, *args, **kwargs):
        # Needed in case a whole data set is decoded in one go.
        return self.words[:].view(*args, **kwargs)


class HDF5RawPayload(HDF5DatasetWrapper, HDF5Payload):
    def __init__(self, words, header=None):
        self.words = words
        if header is not None:
            self._dtype = header.dtype
            assert header.encoded_dtype == words.dtype
            assert header.sample_shape == self.sample_shape
            assert header.samples_per_frame == len(self)
        elif words.dtype == DTYPE_C4:
            self._dtype = np.dtype('c8')
        else:
            self._dtype = words.dtype

    @property
    def data(self):
        return self[()]

    @property
    def sample_shape(self):
        return self.words.shape[1:]

    def __len__(self):
        return len(self.words)

    def __getitem__(self, item):
        result = super().__getitem__(item)
        if result.dtype == DTYPE_C4:
            result = result.view(DTYPE_C4['real']).astype('f4').view('c8')

        return result.astype(self.dtype, copy=False)

    def __setitem__(self, item, value):
        if self.words.dtype == DTYPE_C4:
            value = (value.view(value.real.dtype)
                     .astype(DTYPE_C4['real']).view(DTYPE_C4))
        super().__setitem__(item, value)

    @property
    def dtype(self):
        """Numeric type of the decoded data array."""
        return self._dtype


class HDF5CodedPayload(HDF5Payload, PayloadBase):
    _decoders = VDIFPayload._decoders
    _encoders = VDIFPayload._encoders

    def __init__(self, words, header=None, sample_shape=(), bps=None,
                 complex_data=False):
        # Wrap the h5py.Dataset since it misses a few ndarray attributes.
        # In particular, nbytes, itemsize.
        words = HDF5DatasetWrapper(words)
        if header is not None:
            sample_shape = header.sample_shape
            bps = header.bps
            complex_data = header.complex_data
        super().__init__(words, sample_shape=sample_shape,
                         bps=bps, complex_data=complex_data)
