# Licensed under the GPLv3 - see LICENSE
"""Radio baseband data reduction."""

try:
    from .version import version as __version__
except ImportError:
    __version__ = ''

# Define minima for the documentation, but do not bother to explicitly check.
__minimum_python_version__ = '3.7'
__minimum_baseband_version__ = '4.2'
__minimum_astropy_version__ = '5.1'
