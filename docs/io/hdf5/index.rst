.. _hdf5:

*******************************
HDF5 (`scintillometry.io.hdf5`)
*******************************

`~scintillometry.io.hdf5` contains interfaces for reading and writing
an internal HDF5 format that is suitable for storing intermediate
steps of pipelines.  A writer can conveniently be used as the output
argument for any reader, automatically writing in the relevant block size.
For instance::

.. doctest-requires:: h5py, pyyaml

  >>> from baseband import data, vdif
  >>> from scintillometry.functions import Square
  >>> from scintillometry.io import hdf5
  >>> fh = vdif.open(data.SAMPLE_VDIF, 'rs')
  >>> square = Square(fh)
  >>> squared = square.read()
  >>> h5w = hdf5.open('squared.hdf5', 'w', template=square)
  >>> square.seek(0)
  0
  >>> square.read(out=h5w)
  <HDF5StreamWriter name=squared.hdf5 offset=40000
       sample_rate=32.0 MHz, samples_per_frame=40000,
       sample_shape=(8,), dtype=float32,
       start_time=2014-06-16T05:56:07.000000000>
  >>> h5w.close()
  >>> fh.close()
  >>> h5r = hdf5.open('squared.hdf5', 'r')
  >>> recovered = h5r.read()
  >>> (squared == recovered).all()
  True

.. _hdf5_api:

Reference/API
=============

.. automodapi:: scintillometry.io.hdf5
.. automodapi:: scintillometry.io.hdf5.header
.. automodapi:: scintillometry.io.hdf5.payload
.. automodapi:: scintillometry.io.hdf5.base
