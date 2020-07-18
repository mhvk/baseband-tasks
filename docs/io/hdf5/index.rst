.. _hdf5:

*******************************
HDF5 (`baseband_tasks.io.hdf5`)
*******************************

`~baseband_tasks.io.hdf5` contains interfaces for reading and writing
an internal HDF5 format that is suitable for storing intermediate
steps of pipelines.  A writer can conveniently be used as the output
argument for any reader, automatically writing in the relevant block size.
For instance::

.. doctest-requires:: h5py, pyyaml

  >>> from baseband import data, vdif
  >>> from baseband_tasks.functions import Square
  >>> from baseband_tasks.io import hdf5
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

It is also possible to stored data encoding using the standard
``vdif`` schemes by passing in ``bps`` and ``complex_data``.
Alternatively, half-precision floats can be used by passing in
``encoded_dtype='f2'`` or ``'c4'``.

.. _hdf5_api:

Reference/API
=============

.. automodapi:: baseband_tasks.io.hdf5
.. automodapi:: baseband_tasks.io.hdf5.header
.. automodapi:: baseband_tasks.io.hdf5.payload
   :include-all-objects:
.. automodapi:: baseband_tasks.io.hdf5.base
