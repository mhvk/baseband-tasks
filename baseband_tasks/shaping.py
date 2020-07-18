# Licensed under the GPLv3 - see LICENSE
import numpy as np

from .base import TaskBase, Task, check_broadcast_to, simplify_shape


__all__ = ['ChangeSampleShapeBase', 'ChangeSampleShape',
           'Reshape', 'Transpose', 'ReshapeAndTranspose', 'GetItem']


class ChangeSampleShapeBase(TaskBase):
    """Base class for sample shape operations.

    Similar to `~baseband_tasks.base.TaskBase`, where a subclass can define
    a ``task`` method to operate on data, but specifically for methods that
    change the shape of the samples, yet do not affect the time axis.  This
    class ensures the operation is possible and that the ``frequency``,
    ``sideband``, and ``polarization`` attributes are adjusted similarly.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream.
    """

    def __init__(self, ih):
        # Check operation is possible
        a = np.empty((7,) + ih.sample_shape, dtype='?')
        try:
            a = self.task(a)
        except Exception as exc:
            exc.args += ("stream samples with shape {} cannot be changed as "
                         "required".format(ih.sample_shape),)
            raise
        if a.shape[0] != 7:
            raise ValueError("change in shape affected the sample axis (0).")

        super().__init__(ih, shape=ih.shape[:1] + a.shape[1:])

    def _check_shape(self, value):
        """Broadcast value to the sample shape and apply shape changes.

        After application, axes in which all values are identical are removed.
        """
        # This overrides Base._check_value. Here, an actual check is not really
        # necessary since the values are guaranteed to come from the underlying
        # stream so should have been checked already. But we still do it.  With
        # the fully broadcast data, we then apply the shape changing operation,
        # and then remove axes in which all values are the same.
        broadcast = check_broadcast_to(value, (1,) + self.ih.sample_shape)
        # Remove sample time axis but ensure we do not decay to a scalar.
        value = self.task(broadcast)[0, ...]
        return simplify_shape(value)


class ChangeSampleShape(Task, ChangeSampleShapeBase):
    """Change sample shape using a callable.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream.
    task : callable
        The function or method-like callable. The task must work with
        any number of data samples and change the sample shape only.
        It will also be applied to the ``frequency``, ``sideband``, and
        ``polarization`` attributes of the underlying stream (if present).
    method : bool, optional
        Whether ``task`` is a method (two arguments) or a function
        (one argument).  Default: inferred by inspection.

    See Also
    --------
    Reshape : to reshape the samples
    Transpose : to transpose sample axes
    ReshapeAndTranspose : to reshape the samples and transpose the axes
    GetItem : index or slice the samples

    Examples
    --------
    The VDIF example file from ``Baseband`` has 8 threads which contain
    4 channels and 2 polarizations, with very little data in the last channel.
    To produce a stream in which the sample axes are frequency and polarization
    and only the first three channels are kept, one could do::

        >>> import numpy as np, astropy.units as u, baseband
        >>> from baseband_tasks.shaping import ChangeSampleShape
        >>> fh = baseband.open(baseband.data.SAMPLE_VDIF)
        >>> fh.frequency = 311.25 * u.MHz + (np.arange(8.) // 2) * 16. * u.MHz
        >>> fh.sideband = 1
        >>> fh.polarization = np.tile(['L', 'R'], 4)
        >>> sh = ChangeSampleShape(
        ...    fh, lambda data: data.reshape(-1, 4, 2)[:, :3])
        >>> sh.read(2).shape
        (2, 3, 2)
        >>> sh.polarization
        array(['L', 'R'], dtype='<U1')
        >>> sh.frequency  # doctest: +FLOAT_CMP
        <Quantity [[311.25],
                   [327.25],
                   [343.25]] MHz>
        >>> sh.sideband
        array(1, dtype=int8)
        >>> fh.close()
    """
    # Override __init__ only to get rid of kwargs of Task, since these cannot
    # be passed on to ChangeSampleShapeBase anyway.

    def __init__(self, ih, task, method=None):
        super().__init__(ih, task, method=method)


class Reshape(ChangeSampleShapeBase):
    """Reshapes the sample shape of a stream.

    Useful to ensure, e.g., frequencies and polarizations are on separate axes
    before feeding a stream to `~baseband_tasks.functions.Power`.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream.
    sample_shape : tuple of int
        Output sample shape.

    See Also
    --------
    Transpose : to transpose sample axes
    ReshapeAndTranspose : to reshape the samples and transpose the axes
    GetItem : index or slice the samples
    ChangeSampleShape : to change the samples with a user-supplied function.

    Examples
    --------
    The VDIF example file from ``Baseband`` has 8 threads which contain
    4 channels and 2 polarizations.  To produce a stream in which the sample
    axes are frequency and polarization, one could do::

        >>> import numpy as np, astropy.units as u, baseband
        >>> from baseband_tasks.shaping import ChangeSampleShape
        >>> fh = baseband.open(baseband.data.SAMPLE_VDIF)
        >>> fh.frequency = 311.25 * u.MHz + (np.arange(8.) // 2) * 16. * u.MHz
        >>> fh.sideband = 1
        >>> fh.polarization = np.tile(['L', 'R'], 4)
        >>> rh = Reshape(fh, (4, 2))
        >>> rh.read(2).shape
        (2, 4, 2)
        >>> rh.polarization
        array(['L', 'R'], dtype='<U1')
        >>> rh.frequency  # doctest: +FLOAT_CMP
        <Quantity [[311.25],
                   [327.25],
                   [343.25],
                   [359.25]] MHz>
        >>> rh.sideband
        array(1, dtype=int8)
        >>> fh.close()
    """

    def __init__(self, ih, sample_shape):
        self._new_shape = (-1,) + sample_shape
        super().__init__(ih)

    def task(self, data):
        """Reshape the data."""
        return data.reshape(self._new_shape)


class Transpose(ChangeSampleShapeBase):
    """Reshapes the axes of the samples of a stream.

    Useful to ensure bring, e.g., frequencies and polarizations in groups
    before dechannelizing.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream.
    sample_axes : tuple of int
        Where the input sample shape axes should end up in the output sample
        shape (as for `~numpy.transpose`).  Should contain all axes of the
        sample shape, starting at ``1`` (time axis 0 always stays in place).

    See Also
    --------
    Reshape : to reshape the samples
    ReshapeAndTranspose : to reshape the samples and transpose the axes
    GetItem : index or slice the samples
    ChangeSampleShape : to change the samples with a user-supplied function.

    Examples
    --------
    The VDIF example file from ``Baseband`` has 8 threads which contain
    4 channels and 2 polarizations.  To produce a stream in which the sample
    axes are polarization and frequency, one could do::

        >>> import numpy as np, astropy.units as u, baseband
        >>> from baseband_tasks.shaping import ChangeSampleShape
        >>> fh = baseband.open(baseband.data.SAMPLE_VDIF)
        >>> fh.frequency = 311.25 * u.MHz + (np.arange(8.) // 2) * 16. * u.MHz
        >>> fh.sideband = 1
        >>> fh.polarization = np.tile(['L', 'R'], 4)
        >>> rh = Reshape(fh, (4, 2))
        >>> th = Transpose(rh, (2, 1))
        >>> th.read(2).shape
        (2, 2, 4)
        >>> th.polarization
        array([['L'],
               ['R']], dtype='<U1')
        >>> th.frequency  # doctest: +FLOAT_CMP
        <Quantity [311.25, 327.25, 343.25, 359.25] MHz>
        >>> th.sideband
        array(1, dtype=int8)
        >>> fh.close()

    Note that the example above could also be done in one go using
    `~baseband_tasks.shaping.ReshapeAndTranspose`.
    """

    def __init__(self, ih, sample_axes):
        self._axes = (0,) + sample_axes
        super().__init__(ih)

    def task(self, data):
        """Transpose the axes of data."""
        return data.transpose(self._axes)


class ReshapeAndTranspose(Reshape):
    """Reshapes the sample shape of a stream and transpose its axes.

    Useful to ensure, e.g., frequencies and polarizations are on separate axes
    before feeding a stream to, e.g., `~baseband_tasks.functions.Power`.

    This is just the combination of `~baseband_tasks.shaping.Reshape` and
    `~baseband_tasks.shaping.Transpose` (avoiding intermediate results).

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream.
    sample_shape : tuple of int
        Output sample shape.
    sample_axes : tuple of int
        Where the input sample shape axes should end up in the output sample
        shape (as for `~numpy.transpose`).  Should contain all axes of the
        sample shape, starting at ``1`` (time axis 0 always stays in place).

    See Also
    --------
    Reshape : to just reshape the samples
    Transpose : to just transpose sample axes
    GetItem : index or slice the samples
    ChangeSampleShape : to change the samples with a user-supplied function.

    Examples
    --------
    The VDIF example file from ``Baseband`` has 8 threads which contain
    4 channels and 2 polarizations.  To produce a stream in which the sample
    axes are polarization and frequency, one could do::

        >>> import numpy as np, astropy.units as u, baseband
        >>> from baseband_tasks.shaping import ChangeSampleShape
        >>> fh = baseband.open(baseband.data.SAMPLE_VDIF)
        >>> fh.frequency = 311.25 * u.MHz + (np.arange(8.) // 2) * 16. * u.MHz
        >>> fh.sideband = 1
        >>> fh.polarization = np.tile(['L', 'R'], 4)
        >>> rth = ReshapeAndTranspose(fh, (4, 2), (2, 1))
        >>> rth.read(2).shape
        (2, 2, 4)
        >>> rth.polarization
        array([['L'],
               ['R']], dtype='<U1')
        >>> rth.frequency  # doctest: +FLOAT_CMP
        <Quantity [311.25, 327.25, 343.25, 359.25] MHz>
        >>> rth.sideband
        array(1, dtype=int8)
        >>> fh.close()
    """

    def __init__(self, ih, sample_shape, sample_axes):
        self._axes = (0,) + sample_axes
        super().__init__(ih, sample_shape=sample_shape)

    def task(self, data):
        """Reshape and transpose the axes of data."""
        return data.reshape(self._new_shape).transpose(self._axes)


class GetItem(ChangeSampleShapeBase):
    """Index or slice the samples of a stream.

    Useful to select, e.g., a specific frequency band or polariazation.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream.
    item : int, slice, list of int, or array of int
        Anything that can slice a numpy array.  Should only attempt to slice
        the samples, not the time axis.

    See Also
    --------
    Reshape : to reshape the samples
    Transpose : to transpose sample axes
    ReshapeAndTranspose : to reshape the samples and transpose the axes
    ChangeSampleShape : to change the samples with a user-supplied function.

    Examples
    --------
    The VDIF example file from ``Baseband`` has 8 threads which contain
    4 channels and 2 polarizations, with very little data in the last channel.
    To produce a stream with just the first three channels kept, one could do::

        >>> import numpy as np, astropy.units as u, baseband
        >>> from baseband_tasks.shaping import GetItem
        >>> fh = baseband.open(baseband.data.SAMPLE_VDIF)
        >>> fh.frequency = 311.25 * u.MHz + (np.arange(8.) // 2) * 16. * u.MHz
        >>> fh.sideband = 1
        >>> fh.polarization = np.tile(['L', 'R'], 4)
        >>> gih = GetItem(fh, slice(0, 6))
        >>> gih.read(2).shape
        (2, 6)
        >>> gih.polarization
        array(['L', 'R', 'L', 'R', 'L', 'R'], dtype='<U1')
        >>> gih.frequency  # doctest: +FLOAT_CMP
        <Quantity [311.25, 311.25, 327.25, 327.25, 343.25, 343.25] MHz>
        >>> gih.sideband
        array(1, dtype=int8)
        >>> fh.close()
    """

    def __init__(self, ih, item):
        if isinstance(item, tuple):
            self._item = (slice(None),) + item
        else:
            self._item = (slice(None), item)
        super().__init__(ih)

    def task(self, data):
        """Get the preset item from the data."""
        return data[self._item]
