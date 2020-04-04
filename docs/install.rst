************
Installation
************

Requirements
============

Scintillometry requires:

    - `Python <https://www.python.org/>`_ v3.6 or later
    - `Astropy`_ v3.2 or later
    - `Numpy <http://www.numpy.org/>`_ v1.16 or later
    - `Baseband <https://pypi.org/project/baseband/>`_ v3.0 or later

In addition, you may want to install:

    - `PyFFTW <https://pypi.org/project/pyFFTW/>`_ v0.11 or later, to be able
      to use the `FFTW <http://www.fftw.org/>`_ library for fast fourier
      transforms.
    - `PINT <https://pypi.org/project/pint-pulsar/>`_ to calculate phases without
      first generating polycos.
    - `H5py <https://www.h5py.org/>`_ to read files in the HDF5 format

.. _installation:

Installing Scintillometry
=========================

Scintillometry is not yet on `PyPI <https://pypi.org/>`_ (the name may
still change...), but you can download and install it with one
`pip <https://packaging.python.org/key_projects/#pip>`_ command with::

    pip install git+https://github.com/mhvk/scintillometry.git#egg=scintillometry

Possibly with ``--user`` if you installing for yourself outside of a virtual
environment, and/or with a trailing ``[all]`` to also install the optional
dependencies (which currently excludes PINT, since it does not work with
astropy 4.0).

Obtaining Source Code
---------------------

The source code and latest development version of Scintillometry can found on
`its GitHub repo <https://github.com/mhvk/scintillometry>`_.  You can get your
own clone using::

    git clone git@github.com:mhvk/scintillometry.git

Of course, it is even better to fork it on GitHub, and then clone your own
repository, so that you can more easily contribute!  From within the cloned
repository::

    pip install .

Here, apart from the ``--user`` option and possible ``[all]`` suffix,
you may want to add the ``--editable`` option to just link to the source
repository, which means that any edit will be seen.

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

.. _sourcebuildtest:

Testing the Installation
========================

To test that the code works on your system, you need
`pytest <http://pytest.org>`_ and
`pytest-astropy <https://github.com/astropy/pytest-astropy>`_
to be installed;
this is most easily done by first installing the code together
with its test dependencies::

    pip install -e .[test]

Then, inside the root directory, simply run

    pytest

or, inside of Python::

    import scintillometry
    scintillometry.test()

For further details, see the `Astropy Running Tests pages
<https://astropy.readthedocs.io/en/latest/development/testguide.html#running-tests>`_.

.. _builddocs:

Building Documentation
======================

.. note::

    As with Astropy, building the documentation is unnecessary unless you
    are writing new documentation or do not have internet access, as
    Scintillometry's documentation is available online at
    `scintillometry.readthedocs.io <https://scintillometry.readthedocs.io>`_.

To build the Scintillometry documentation, you need
`Sphinx <http://sphinx.pocoo.org>`_ and
`sphinx-astropy <https://github.com/astropy/sphinx-astropy>`_
to be installed;
this is most easily done by first installing the code together
with its documentations dependencies::

    pip install -e .[docs]

Then, go to the ``docs`` directory and run

    make html

For further details, see the `Astropy Building Documentation pages
<http://docs.astropy.org/en/latest/install.html#builddocs>`_.
