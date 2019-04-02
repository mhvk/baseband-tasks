************
Installation
************

Requirements
============

Scintillometry requires:

    - `Python <https://www.python.org/>`_ v3.5 or later
    - `Astropy`_ v3.1 or later
    - `Numpy <http://www.numpy.org/>`_ v1.13.0 or later
    - `Baseband <https://pypi.org/project/baseband/>`_ v1.1 or later

In addition, you may want to install:

    - `PyFFTW <https://pypi.org/project/pyFFTW/>`_ v0.11 or later, to be able
      to use the `FFTW <http://www.fftw.org/>`_ library for fast fourier
      transforms.
    - `PINT <https://github.com/nanograv/PINT>`_ to calculate phases without
      first generating polycos.

.. _installation:

Installing Scintillometry
=========================

Obtaining Source Code
---------------------

The source code and latest development version of Scintillometry can found on
`its GitHub repo <https://github.com/mhvk/scintillometry>`_.  You can get your
own clone
using::

    git clone git@github.com:mhvk/scintillometry.git

Of course, it is even better to fork it on GitHub, and then clone your own
repository, so that you can more easily contribute!

Running Code without Installing
-------------------------------

As Scintillometry is purely Python, it can be used without being built or
installed, by appending the directory it is located in to the ``PYTHON_PATH``
environment variable.  Alternatively, you can use :obj:`sys.path` within Python
to append the path::

    import sys
    sys.path.append(SCINT_PATH)

where ``SCINT_PATH`` is the directory you downloaded or cloned
Scintillometry into.

Installing Source Code
----------------------

If you want Scintillometry to be more broadly available, either to all users on
a system, or within, say, a virtual environment, use :file:`setup.py` in
the root directory by calling::

    python3 setup.py install

For general information on :file:`setup.py`, see `its documentation
<https://docs.python.org/3.5/install/index.html#install-index>`_ . Many of the
:file:`setup.py` options are inherited from Astropy (specifically, from `Astropy
-affiliated package manager <https://github.com/astropy/package-template>`_) and
are described further in `Astropy's installation documentation
<https://astropy.readthedocs.io/en/stable/install.html>`_ .

.. _sourcebuildtest:

Testing the Installation
========================

The root directory :file:`setup.py` can also be used to test if Scintillometry
can successfully be run on your system::

    python3 setup.py test

or, inside of Python::

    import scintillometry
    scintillometry.test()

These tests require `pytest <http://pytest.org>`_ to be installed. Further
documentation can be found on the `Astropy running tests documentation
<https://astropy.readthedocs.io/en/stable/development/testguide.html#running-tests>`_
.

.. _builddocs:

Building Documentation
======================

.. note::

    As with Astropy, building the documentation is unnecessary unless you
    are writing new documentation or do not have internet access, as
    Scintillometry's documentation is available online at
    `scintillometry.readthedocs.io <https://scintillometry.readthedocs.io>`_.

The Scintillometry documentation can be built again using :file:`setup.py` from
the root directory::

    python3 setup.py build_docs

This requires to have `Sphinx <http://sphinx.pocoo.org>`_ installed (and its
dependencies).
