# Licensed under the GPLv3 - see LICENSE
"""
Definitions for HDF5 general storage headers.

Implements a HDF5Header class used to store header definitions, and provides
methods to initialize from a stream template, and to write to and read from
an HDF5 Dataset, encoded as yaml file.
"""
import operator

import numpy as np
from astropy.units import Quantity
from astropy.time import Time

from scintillometry.base import check_broadcast_to, simplify_shape


__all__ = ['HDF5Header', 'HDF5RawHeader', 'HDF5CodedHeader']


class HDF5Header(dict):
    """HDF5 format header.

    The type of header is decided by the presence of ``bps``.  If present,
    the payload will be assumed to be encoded; if not, to be raw data with
    a given ``dtype``.

    Parameters
    ----------
    verify : bool, optional
        Whether to do minimal verification that the header is consistent with
        what is needed to interpret HDF5 payloads.  Default: `True`.
    mutable : bool, optional
        Whether to allow the header to be changed after initialisation.
        Default: `True`.
    **kwargs
        Header keywords to be set.  If this includes ``bps``, then this will
        be taken to be a header for encoded data.
    """
    _properties = ('sample_shape', 'samples_per_frame', 'sample_rate', 'time',
                   'frequency', 'sideband', 'polarization')

    def __new__(cls, *, verify=True, mutable=True, **kwargs):
        if 'bps' in kwargs:
            cls = HDF5CodedHeader
        else:
            cls = HDF5RawHeader
        return super().__new__(cls)

    def __init__(self, *, verify=True, mutable=True, **kwargs):
        super().__init__()
        self.mutable = True
        self.update(**kwargs, verify=verify)
        self.mutable = mutable

    def verify(self):
        assert {'sample_shape', 'samples_per_frame',
                'sample_rate', 'time'} <= self.keys()

    def copy(self):
        return self.__class__(verify=False, **self)

    @classmethod
    def fromfile(cls, fh, verify=True):
        """Create a header from a yaml-encoded 'header' extension."""
        from astropy.io.misc import yaml

        data = fh['header'][()]
        items = yaml.load(data)
        return cls(**items, mutable=False, verify=verify)

    def tofile(self, fh):
        """Write the header as a yaml-encoded 'header' extension."""
        from astropy.io.misc import yaml

        data = yaml.dump(dict(self))
        fh.create_dataset('header', data=data)

    @classmethod
    def fromvalues(cls, template, whole=None, verify=True, **kwargs):
        """Initialise a header from a template and/or values.

        Parameters
        ----------
        template : header or stream template, optional
            Must have attributes that define a header ('sample_shape',
            'samples_per_frame', 'sample_rate', 'time', and either 'dtype'
            or 'bps' and 'complex_data').
        whole : bool, optional
            If `True`, assume a header for the complete stream is wanted,
            and use 'start_time' for the 'time' and the total number of
            samples for 'samples_per_frame'.  Default: `True` if the template
            has both 'start_time' and 'shape' (i.e., for streams).
        verify : bool, optional
            Whether to do basic verification.  Default: `True`.
        **kwargs
            Any additional values.  These will override values inferred from
            the template.
        """
        if template is not None:
            if whole or (whole is None
                         and hasattr(template, 'shape')
                         and hasattr(template, 'start_time')):
                kwargs.setdefault('time', template.start_time)
                kwargs.setdefault('samples_per_frame', template.shape[0])

            if hasattr(template, 'bps') or 'bps' in kwargs:
                attrs = HDF5CodedHeader._properties
            else:
                attrs = HDF5RawHeader._properties

            for attr in attrs:
                value = getattr(template, attr, None)
                if value is not None:
                    kwargs.setdefault(attr, value)

        return cls(verify=verify, **kwargs)

    def update(self, *, verify=True, **kwargs):
        """Update the header with new values.

        Here, any keywords matching properties are processed as well, in the
        order set by the class (in ``_properties``), and after all other
        keywords have been processed.

        Parameters
        ----------
        verify : bool, optional
            If `True` (default), verify integrity after updating.
        **kwargs
            Arguments used to set keywords and properties.
        """
        # Remove kwargs that set properties, in correct order.
        extras = [(key, kwargs.pop(key)) for key in self._properties
                  if key in kwargs]
        # Update the normal keywords.
        super().update(kwargs)
        # Now set the properties.
        for attr, value in extras:
            setattr(self, attr, value)
        if verify:
            self.verify()

    def __setitem__(self, item, value):
        if not self.mutable:
            raise TypeError("immutable {0} does not support assignment."
                            .format(type(self).__name__))

        super().__setitem__(item, value)

    def __eq__(self, other):
        return (type(self) is type(other)
                and self.keys() == other.keys()
                and all(np.all(self[key] == other[key])
                        for key in self.keys()))


# Create properties for those that have to be present, using proper
# initializing classes for the setters.
def getter(attr):
    def fget(self):
        return self[attr]

    return fget


def setter(attr, cls):
    def fset(self, value):
        self[attr] = cls(value)

    return fset


for attr, cls in [('sample_shape', tuple),
                  ('samples_per_frame', operator.index),
                  ('sample_rate', Quantity),
                  ('time', Time)]:
    setattr(HDF5Header, attr, property(getter(attr), setter(attr, cls)))


# Create properties for the optional frequency, sideband, and polarization
# items.  Those should give AttributeError if not present, and, on setting,
# should be checked to be broadcastable.
def optional_getter(attr):
    def fget(self):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError("{} not set.".format(attr)) from None

    return fget


def optional_setter(attr):
    def fset(self, value):
        broadcast = check_broadcast_to(value, self.sample_shape)
        self[attr] = simplify_shape(broadcast)

    return fset


for attr in 'frequency', 'sideband', 'polarization':
    setattr(HDF5Header, attr,
            property(optional_getter(attr), optional_setter(attr)))


class HDF5RawHeader(HDF5Header):
    _properties = ('dtype',) + HDF5Header._properties

    def verify(self):
        super().verify()
        # Next assert proves that key exists and can be parsed.
        assert isinstance(self.dtype, np.dtype)

    # Astropy's Yaml loaded cannot encode dtype, so keep its
    # string format as a key.
    @property
    def dtype(self):
        return np.dtype(self['dtype'])

    @dtype.setter
    def dtype(self, dtype):
        self['dtype'] = str(dtype)


class HDF5CodedHeader(HDF5Header):
    _properties = ('bps', 'complex_data') + HDF5Header._properties

    def verify(self):
        super().verify()
        # Next assert proves that keys exist and can be parsed.
        assert isinstance(self.bps, int)
        assert isinstance(self.complex_data, bool)


for attr, cls in [('bps', operator.index),
                  ('complex_data', bool)]:
    setattr(HDF5CodedHeader, attr, property(getter(attr), setter(attr, cls)))
