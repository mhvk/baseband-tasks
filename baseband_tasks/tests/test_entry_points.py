# Licensed under the GPLv3 - see LICENSE
import os
import sys
import types
import importlib

import pytest
if sys.version_info >= (3, 8):
    from importlib.metadata import entry_points
else:
    from importlib_metadata import entry_points

from baseband_tasks import io, dispersion


# needs_entrypoints is imported in io/hdf5/tests/test_hdf5.
def _baseband_io_entry_points():
    try:
        return entry_points(group="baseband.io")
    except TypeError:
        return entry_points().get("baseband.io", [])


needs_entrypoints = pytest.mark.xfail(
    'hdf5' not in [ep.name for ep in _baseband_io_entry_points()]
    and os.path.exists(os.path.join(os.path.dirname(__file__),
                                    '..', '..', 'setup.cfg')),
    reason="Source checkout without entrypoints; needs 'egg_info'.")

pytestmark = needs_entrypoints


def fake_module(group):
    name = group.split('.')[-1]
    module = types.ModuleType(name, doc=f"fake {group} module")
    try:
        group_entry_points = entry_points(group=group)
    except TypeError:
        group_entry_points = entry_points().get(group, [])
    for entry in group_entry_points:
        # Only on python >= 3.9 do .module and .attr exist.
        ep_module, _, ep_attr = entry.value.partition(':')
        if ep_module.startswith('baseband_tasks'):
            if ep_attr == '__all__':
                submod = importlib.import_module(ep_module)
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
    # But we do want to be able to use Task and SetAttribute
    assert hasattr(bt, 'SetAttribute')
    assert hasattr(bt, 'Task')
