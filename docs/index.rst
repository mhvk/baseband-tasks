.. _scintillometry_docs:

**************
Scintillometry
**************

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

   tasks/dedisperse
   tasks/channelize
   tasks/functions
   tasks/base

.. _simulation_toc:

Simulation
==========

To help simulate and debug reduction processes, Scintillometry allows
one to generate fake signals.

.. toctree::
   :maxdepth: 1

   simulation/generators

.. _helpers_toc:

Helpers
=======

Scintillometry also contains helper modules that assist with calculations and
that link it with other software, such as Fourier transform packages.

.. toctree::
   :maxdepth: 1

   helpers/dm
   helpers/fourier

.. _project_details_toc:

Project details
===============

.. image:: https://travis-ci.org/mhvk/scintillometry.svg?branch=master
   :target: https://travis-ci.org/mhvk/scintillometry

.. image:: https://coveralls.io/repos/github/mhvk/scintillometry/badge.svg?branch=master
   :target: https://coveralls.io/github/mhvk/scintillometry?branch=master

.. image:: https://readthedocs.org/projects/scintillometry/badge/?version=latest
   :target: https://scintillometry.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. toctree::
   :maxdepth: 1

   authors_for_sphinx
