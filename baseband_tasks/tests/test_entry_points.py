# Licensed under the GPLv3 - see LICENSE
import os
import types
import importlib

import pytest
import entrypoints

from .. import io, dispersion


# needs_entrypoints is imported in io/hdf5/tests/test_hdf5.
needs_entrypoints = pytest.mark.xfail(
    'hdf5' not in entrypoints.get_group_named('baseband.io')
    and os.path.exists(os.path.join(os.path.dirname(__file__),
                                    '..', '..', 'setup.cfg')),
    reason="Source checkout without entrypoints; needs 'egg_info'.")

pytestmark = needs_entrypoints


def fake_module(group):
    from entrypoints import get_group_all

    name = group.split('.')[-1]
    entries = get_group_all(group)
    module = types.ModuleType(name, doc=f"fake {group} module")
    for entry in entries:
        if entry.module_name.startswith('baseband_tasks'):
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
    # Note: io/hdf5/tests check that it is recognized in baseband itself.
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
