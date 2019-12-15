"""HDF5 header"""
import operator

import numpy as np
from astropy.units import Quantity
from astropy.time import Time
from astropy.io.misc import yaml

from scintillometry.base import check_broadcast_to, simplify_shape


class HDF5Header(dict):
    _properties = ('sample_shape', 'samples_per_frame',
                   'sample_rate', 'time', 'dtype',
                   'frequency', 'sideband', 'polarization')

    def __new__(cls, *args, verify=True, mutable=True, **kwargs):
        if 'bps' in kwargs:
            cls = HDF5CodedHeader
        else:
            cls = HDF5Header
        return super().__new__(cls)

    def __init__(self, *args, verify=True, mutable=True, **kwargs):
        self.mutable = True
        super().__init__(*args)
        self.update(**kwargs, verify=verify)
        self.mutable = mutable

    def verify(self):
        assert {'sample_shape', 'samples_per_frame',
                'sample_rate', 'time'} <= self.keys()

    def copy(self):
        return self.__class__(self)

    @classmethod
    def fromfile(cls, fh, verify=True):
        data = fh['header'][()]
        items = yaml.load(data)
        return cls(**items, mutable=False, verify=verify)

    def tofile(self, fh):
        data = yaml.dump(dict(self))
        fh.create_dataset('header', data=data)

    @classmethod
    def fromvalues(cls, template=None, **kwargs):
        if template is not None:
            value = getattr(template, 'start_time', None)
            if value is not None:
                kwargs.setdefault('time', value)
            value = getattr(template, 'shape', None)
            if value is not None:
                kwargs.setdefault('samples_per_frame', value[0])
            for attr in HDF5CodedHeader._properties:
                value = getattr(template, attr, None)
                if value is not None:
                    kwargs.setdefault(attr, value)

        return cls(verify=False, **kwargs)

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

    @property
    def dtype(self):
        return np.dtype(self['dtype'])

    @dtype.setter
    def dtype(self, dtype):
        self['dtype'] = str(dtype)


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
    if not hasattr(HDF5Header, attr):
        setattr(HDF5Header, attr,
                property(getter(attr), setter(attr, cls)))


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


class HDF5CodedHeader(HDF5Header):
    _properties = ('bps', 'complex_data') + HDF5Header._properties

    @property
    def dtype(self):
        return np.dtype('c8') if self.complex_data else np.dtype('f4')

    @dtype.setter
    def dtype(self, dtype):
        if self.dtype != dtype:
            raise ValueError('dtype has to be {}'.format(self.dtype))


for attr, cls in [('bps', operator.index),
                  ('complex_data', bool)]:
    setattr(HDF5CodedHeader, attr,
            property(getter(attr), setter(attr, cls)))
