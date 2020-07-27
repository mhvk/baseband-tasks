.. _shaping:

*********************************************************
Slicing and shape manipulation (`baseband_tasks.shaping`)
*********************************************************

`~baseband_tasks.shaping` contains tasks to slice the time stream and
change the sample shape, e.g., to ensure frequency and polarization
are on different axes in preparation for calculating polarized powers
and cross products.  The changes can be done using a user-defined
function or with pre-defined implementations for slicing, reshaping,
and transposing.

.. _shaping_api:

Reference/API
=============

.. automodapi:: baseband_tasks.shaping
   :no-inherited-members:
