# Licensed under the GPLv3 - see LICENSE
import os
import types
import importlib

import pytest

from .. import io, dispersion


def needs_entrypoints(func=None):
    # Guaranteed to be available for baseband >= 4.0.
    try:
        import entrypoints
        skip = False
    except ImportError:
        skip = True

    # If we're in a source checkout in which not even setup.py egg_info has
    # been run, the entry points cannot be found so we should skip tests.
    if (not skip and 'hdf5' not in entrypoints.get_group_named('baseband.io')
        and os.path.exists(os.path.join(os.path.dirname(__file__),
                                        '..', '..', 'setup.cfg'))):
        mark = pytest.mark.xfail(reason=(
            "Source checkout without entrypoints; needs 'egg_info'."))
    else:
        mark = pytest.mark.skipif(skip, reason="entrypoints not available")

    return mark(func) if func else mark


pytestmark = needs_entrypoints()


def fake_module(group):
    from entrypoints import get_group_all

    name = group.split('.')[-1]
    entries = get_group_all(group)
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
