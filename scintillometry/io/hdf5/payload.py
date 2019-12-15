# Licensed under the GPLv3 - see LICENSE
"""Payload for HDF format."""
from functools import reduce
import operator

import numpy as np
import h5py
from baseband.vdif import VDIFPayload
from baseband.vlbi_base.payload import VLBIPayloadBase


__all__ = ['HDF5Payload']


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

    Parameters
    ----------
    words : `~numpy.ndarray`
        Array containg data as stored in the HDF5 file, which possibly
        are encoded similar to a VDIF payload.
    header : HDF5Header
        Header that provides information about how the payload is encoded.
        If not given, the following arguments have to be passed in.
    """
    _decoders = VDIFPayload._decoders
    _encoders = VDIFPayload._encoders

    def __new__(cls, words, header=None, **kwargs):
        if header is not None and hasattr(header, 'bps') or 'bps' in kwargs:
            cls = HDF5EncodedPayload
        else:
            cls = HDF5RawPayload
        return super().__new__(cls)

    @classmethod
    def fromfile(cls, fh, header=None, **kwargs):
        """Get payload words from HDF5 file or group.

        Parameters
        ----------
        fh : `~h5py.File` or `~h5py.Group`
            Handle to the HDF5 file/group which has an 'payload' dataset.
        header : `~scintillometry.io.hdf5.HDF5Header`, optional
            If given, used to infer ``dtype``, ``bps``, ``sample_shape``,
            and ``complex_data``.  If not given, some of those may have
            to be passed in.
        **kwargs
            Additional arguments are passed on to the class initializer. These
            are only needed if ``header`` is not given.
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
        # Needs more sanity checks for reading!!!
        return cls(words, header, **kwargs)


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


class HDF5EncodedPayload(HDF5Payload, VLBIPayloadBase):
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
