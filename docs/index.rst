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

.. _modules_toc:

   tasks/channelize
   tasks/base

Support Modules
===============

Scintillometry also contains support modules that assist with calculations and
that link it with other software, such as Fourier transform packages.

.. toctree::
   :maxdepth: 2

   supportmodules/dm
   fourier/index

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
