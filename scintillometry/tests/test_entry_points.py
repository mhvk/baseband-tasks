# Licensed under the GPLv3 - see LICENSE
import types
import importlib

import entrypoints

from .. import io, dispersion


def fake_module(group):
    name = group.split('.')[-1]
    entries = entrypoints.get_group_all(group)
    module = types.ModuleType(name, doc=f"fake {group} module")
    for entry in entries:
        if entry.module_name.startswith('scintillometry'):
            if entry.object_name == '__all__':
                submod = importlib.import_module(entry.module_name)
                for item in entry.load():
                    setattr(module, item, getattr(submod, item))
                if entry.name != '_':
                    setattr(module, entry.name, submod)
            else:
                setattr(module, entry.name, entry.load())

    return module


def test_io_entry_points():
    bio = fake_module('baseband.io')
    assert bio is not io
    assert hasattr(bio, 'hdf5')
    assert bio.hdf5 is io.hdf5


def test_tasks_entry_points():
    bt = fake_module('baseband.tasks')
    assert bt is not dispersion
    assert hasattr(bt, 'dispersion')
    assert bt.dispersion is dispersion
    assert hasattr(bt, 'Dedisperse')
    assert bt.Dedisperse is dispersion.Dedisperse
    # Presumably, we'll always want to import Base
    # classes from bt.base.
    assert not hasattr(bt, 'Base')
