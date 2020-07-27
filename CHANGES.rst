0.2 (unreleased)
================

This release will depend on ``baseband`` 4.0 as one can then assume
(and document) the existence of ``baseband.tasks``.  Like baseband 4.0,
it will also require python 3.7, astropy 4.0, and numpy 1.17.

New Features
------------

- Streams can now be sliced, returning a new stream for a more limited
  time span and/or sample shape. [#192]

Other Changes and Additions
---------------------------

- Tasks can now be defined with a number of complete samples (``shape[0]``)
  that is not an integer multiple of ``samples_per_frame``, which can be
  used to avoid losing ends of streams for tasks that can handle dealing
  with partial frames. [#188]

0.1.1 (2020-07-19)
==================

Update that includes the DOI and for which the README.txt is clean
enough for ``twine``.


0.1 (2020-07-19)
================

Initial release.  Project renamed from original ``scintillometry``,
but similar in that the documentation still suggests that tasks be
imported from the various modules in ``baseband_tasks``, which is
the only way they can be used for ``baseband`` prior to 4.0.
