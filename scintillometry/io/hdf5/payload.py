# Licensed under the GPLv3 - see LICENSE
"""Payload for HDF5 format."""
from functools import reduce
import operator

import numpy as np
import h5py
from baseband.vdif import VDIFPayload
from baseband.vlbi_base.payload import VLBIPayloadBase


__all__ = ['HDF5Payload', 'HDF5RawPayload', 'HDF5CodedPayload',
           'HDF5DatasetWrapper']


class DtypeDefaultCoder(dict):
    def __getitem__(self, item):
        try:
            dtype = np.dtype(item)
        except Exception:
            pass
        else:
            return lambda x: x.astype(dtype, copy=False)

        return super().__getitem__(item)


class HDF5Payload:
    """Container for decoding and encoding HDF5 payloads.

    The data will be taken to represent their values directly unless
    the header has a ``bps`` attribute, or ``bps`` is given explicitly.

    Parameters
    ----------
    words : `~numpy.ndarray`
        Array containg data as stored in the HDF5 file, which possibly
        are encoded similar to a VDIF payload.
    header : `~scintillometry.io.hdf5.HDF5Header`, optional
        Header providing information about whether, and if so, how the payload
        is encoded. If not given, then the following need to be passed in.
    sample_shape : tuple, optional
        Shape of the samples; e.g., (nchan,).  Default: ().
    bps : int, optional
        Number of bits per sample part (i.e., per channel and per real or
        imaginary component).  If given, the data are assumed to be encoded.
    complex_data : bool, optional
        Whether data are complex.  Default: `False`.
    """

    def __new__(cls, words, header=None, **kwargs):
        if header is not None and hasattr(header, 'bps') or 'bps' in kwargs:
            cls = HDF5CodedPayload
        else:
            cls = HDF5RawPayload
        return super().__new__(cls)

    @classmethod
    def fromfile(cls, fh, header=None):
        """Get payload words from HDF5 file or group.

        Parameters
        ----------
        fh : `~h5py:File` or `~h5py:Group`
            Handle to the HDF5 file/group which has an 'payload' dataset.
            If the payload does not exist, it will be created.
        header : `~scintillometry.io.hdf5.HDF5Header`, optional
            Must be given for encoded payloads, or to create a payload.
        """
        try:
            words = fh['payload']
        except KeyError:
            if hasattr(header, 'bps'):
                nsample = reduce(operator.mul, header.sample_shape,
                                 header.samples_per_frame)
                shape = ((header.bps * (2 if header.complex_data else 1)
                          * nsample + 31) // 32,)
                dtype = '<u4'
            else:
                shape = (header.samples_per_frame,) + header.sample_shape
                dtype = header.dtype

            words = fh.create_dataset('payload', shape=shape, dtype=dtype)

        return cls(words, header)


class HDF5DatasetWrapper:
    """Make a HDF5 Dataset look a bit more like ndarray."""
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
            assert header.dtype == self.dtype
            assert header.sample_shape == self.sample_shape
            assert header.samples_per_frame == len(self)

    @property
    def data(self):
        return self[()]

    @property
    def sample_shape(self):
        return self.words.shape[1:]

    def __len__(self):
        return len(self.words)


class HDF5CodedPayload(HDF5Payload, VLBIPayloadBase):
    _decoders = VDIFPayload._decoders
    _encoders = VDIFPayload._encoders

    def __init__(self, words, header=None, sample_shape=None, bps=None,
                 complex_data=False):
        # Wrap the h5py.Dataset since it misses a few ndarray attributes.
        # In particular, nbytes, itemsize.
        if isinstance(words, h5py.Dataset):
            words = HDF5DatasetWrapper(words)
        if header is not None:
            sample_shape = header.sample_shape
            bps = header.bps
            complex_data = header.complex_data
        super().__init__(words, sample_shape=sample_shape,
                         bps=bps, complex_data=complex_data)
