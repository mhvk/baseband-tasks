# Licensed under the GPLv3 - see LICENSE.rst
"""Data files helpful for baseband_tasks analysis."""

# Use private names to avoid inclusion in the sphinx documentation.
from os import path as _path


def _full_path(name, dirname=_path.dirname(_path.abspath(__file__))):
    return _path.join(dirname, name)


PSRFITS_DOCUMENTATION = _full_path('PsrfitsDocumentation.html')
"""PSRFITS documentation as a html file.

Taken from
https://www.atnf.csiro.au/research/pulsar/psrfits_definition/PsrfitsDocumentation.html
"""
