.. _scintillometry_docs:

**************
Scintillometry
**************

.. warning::
   The Scintillometry project has been renamed to
   `baseband-tasks <https://baseband.readthedocs.io/project/baseband-tasks>`_.
   For reproducing old results, one can checkout the old repository with
   ``pip install git+https://github.com/mhvk/baseband-tasks.git@scintillometry#egg=scintillometry``.


Welcome to the Scintillometry documentation!  Scintillometry is a package for
reduction and analysis of radio baseband data, optimized for pulsar
scintillometry science.

.. _overview_toc:

Overview
========

.. toctree::
   :maxdepth: 2

   install

.. _tasks_toc:

Tasks
=====

At the core of Scintillometry is a collection of "tasks", filehandle-like
classes that can be linked together into a data reduction pipeline.

.. toctree::
   :maxdepth: 1

   tasks/channelize
   tasks/combining
   tasks/conversion
   tasks/convolution
   tasks/dispersion
   tasks/functions
   tasks/integration
   tasks/sampling
   tasks/shaping
   tasks/base

.. _simulation_toc:

Simulation
==========

To help simulate and debug reduction processes, Scintillometry allows
one to generate fake signals.

.. toctree::
   :maxdepth: 1

   simulation/generators

.. _input_output_toc:

Input/output
============

Most I/O is via raw files read using :func:`baseband.open`, but
Scintillometry offers options to write out intermediate products as
HDF5 files and final products, such as folded profiles, in PSRFITS format.

.. toctree::
   :maxdepth: 1

   io/hdf5/index
   io/psrfits/index

.. _helpers_toc:

Helpers
=======

Scintillometry also contains helper modules that assist with calculations and
that link it with other software, such as Fourier transform packages or pulsar
timing software.

.. toctree::
   :maxdepth: 1

   helpers/dm
   helpers/fourier
   helpers/phases

.. _project_details_toc:

Project details
===============

.. image:: https://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: https://www.astropy.org/
    :alt: Powered by Astropy Badge

.. image:: https://travis-ci.org/mhvk/scintillometry.svg?branch=master
   :target: https://travis-ci.org/mhvk/scintillometry
   :alt: Test Status

.. image:: https://codecov.io/gh/mhvk/scintillometry/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/mhvk/scintillometry
   :alt: Coverage Level

.. image:: https://readthedocs.org/projects/scintillometry/badge/?version=latest
   :target: https://scintillometry.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. toctree::
   :maxdepth: 1

   authors_for_sphinx
