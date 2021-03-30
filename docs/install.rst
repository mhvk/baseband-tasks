************
Installation
************

Requirements
============

Baseband-tasks requires `Baseband
<https://pypi.org/project/baseband/>`_, v3.0 or later, which in turn
requires `Python <https://www.python.org/>`_, `Astropy`_, and `Numpy
<http://www.numpy.org/>`_.

In addition, you may want to install:

    - `PyFFTW <https://pypi.org/project/pyFFTW/>`_ v0.11 or later, to be able
      to use the `FFTW <http://www.fftw.org/>`_ library for fast fourier
      transforms.
    - `PINT <https://pypi.org/project/pint-pulsar/>`_ to calculate phases without
      first generating polycos.
    - `H5py <https://www.h5py.org/>`_ to read files in the HDF5 format.

.. _installation:

Installing Baseband-tasks
=========================

To install Baseband-tasks with `pip <https://pip.pypa.io/>`_,
run::

    pip3 install baseband-tasks

Possibly with ``--user`` if you installing for yourself outside of a virtual
environment, and/or with a trailing ``[all]`` to also install the optional
dependencies (or ``[io]`` for just HDF5 support).

.. note::
   Baseband-tasks was originally developped under the name ``scintillometry``.
   It was never put on `PyPI <https://pypi.org/>`_ under that name, but
   could be downloaded from github. While we strongly recommend
   adjusting any scripts you have, if you need to check an old result
   you can install the last version under the ``scintillometry`` name
   with::

    pip install git+https://github.com/mhvk/baseband-tasks.git@scintillometry#egg=scintillometry

Obtaining Source Code
---------------------

The source code and latest development version of Baseband-tasks can found on
`its GitHub repo <https://github.com/mhvk/baseband-tasks>`_.  You can get your
own clone using::

    git clone git@github.com:mhvk/baseband-tasks.git

Of course, it is even better to fork it on GitHub, and then clone your own
repository, so that you can more easily contribute!  From within the cloned
repository::

    pip install .

Here, apart from the ``--user`` option and possible ``[all]`` or ``[io]`` suffix,
you may want to add the ``--editable`` option to just link to the source
repository, which means that any edit will be seen.

Running Code without Installing
-------------------------------

As Baseband-tasks is purely Python, it can be used without being built or
installed, by appending the directory it is located in to the ``PYTHON_PATH``
environment variable.  Alternatively, you can use :obj:`sys.path` within Python
to append the path::

    import sys
    sys.path.append(PACKAGE_PATH)

where ``PACKAGE_PATH`` is the directory you downloaded or cloned
Baseband-tasks into.

Note that for the `baseband.io` and `baseband.tasks` plugins to work, you will
need to produce ``egg_info``, which can be done with::

    python3 setup.py egg_info

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

    import baseband_tasks
    baseband_tasks.test()

For further details, see the `Astropy Running Tests pages
<https://astropy.readthedocs.io/en/latest/development/testguide.html#running-tests>`_.

.. _builddocs:

Building Documentation
======================

.. note::

    As with Astropy, building the documentation is unnecessary unless you
    are writing new documentation or do not have internet access, as
    Baseband-tasks's documentation is available online at
    `baseband.readthedocs.io/project/baseband-tasks <https://baseband.readthedocs.io/project/baseband-tasks>`_.

To build the Baseband-tasks documentation, you need
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
