# Licensed under the GPLv3 - see LICENSE
import numpy as np

from .base import TaskBase, Task


__all__ = ['SampleShapeChange', 'Reshape', 'Transpose', 'ReshapeAndTranspose']


class SampleShapeChangeBase(TaskBase):
    """Base class for sample shape operations.

    Assumes the subclass has a ``task`` method that will change the shape.
    Adjusts ``frequency``, ``sideband``, and ``polarization`` attributes
    similarly.

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
        """Broadcast value to the sample shape and apply shape changes

        After application, axes in which all values are identical are removed.
        """
        # This overrides Base._check_value. Here, an actual check is not
        # necessary since the values are guaranteed to come from the underlying
        # stream so should have been checked already. Instead, we apply the
        # shape changing operation on the fully broadcast data, and then remove
        # axes in which all values are the same.
        broadcast = np.broadcast_to(value, (1,) + self.ih.sample_shape,
                                    subok=True)
        value = self.task(broadcast)[0]
        for axis in range(value.ndim):
            # Get first element of the sample in the current axis.
            value_0 = value[(slice(None),) * axis + (slice(0, 1),)]
            # If all samples are the same in this axis, just keep the first
            # sample; numpy broadcasting rules will ensure operations are OK.
            if value.strides[axis] == 0 or np.all(value == value_0):
                value = value_0

        # Remove leading ones, which are not needed in numpy broadcasting.
        first_not_unity = next((i for (i, s) in enumerate(value.shape)
                                if s > 1), value.ndim)
        value.shape = value.shape[first_not_unity:]
        return value


class SampleShapeChange(Task, SampleShapeChangeBase):
    """Change sample shape using a callable.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream.
    task : callable
        The function or method-like callable, The function be applicable
        to any number of data samples and change the sample shape only.
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

    Examples
    --------
    The VDIF example file from ``Baseband`` has 8 threads which contain
    4 channels and 2 polarizations.  To produce a stream in which the sample
    axes are polarization and frequency, one could do::

        >>> import numpy as np, astropy.units as u, baseband
        >>> from scintillometry.shaping import SampleShapeChange
        >>> fh = baseband.open(baseband.data.SAMPLE_VDIF)
        >>> fh.frequency = 311.25 * u.MHz + (np.arange(8.) // 2) * 16. * u.MHz
        >>> fh.sideband = 1
        >>> fh.polarization = np.tile(['L', 'R'], 4)
        >>> sh = SampleShapeChange(
        ...    fh, lambda data: data.reshape(-1, 4, 2).swapaxes(1, 2))
        >>> sh.read(2).shape
        (2, 2, 4)
        >>> sh.polarization
        array([['L'],
               ['R']], dtype='<U1')
        >>> sh.frequency
        <Quantity [311.25, 327.25, 343.25, 359.25] MHz>
        >>> sh.sideband
        array(1, dtype=int8)
    """
    # Override __init__ only to get rid of kwargs, which cannot be
    # passed on to SampleShapeChangeBase.
    def __init__(self, ih, task, method=None):
        super().__init__(ih, task, method=method)


class Reshape(SampleShapeChangeBase):
    """Reshapes the sample shape of a stream.

    Useful to ensure, e.g., frequencies and polarizations are on separate axes
    before feeding a stream to `~scintillometry.functions.Power`.

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
    SampleShapeChange : to change the samples with a user-supplied function.

    Examples
    --------
    The VDIF example file from ``Baseband`` has 8 threads which contain
    4 channels and 2 polarizations.  To produce a stream in which the sample
    axes are frequency and polarization, one could do::

        >>> import numpy as np, astropy.units as u, baseband
        >>> from scintillometry.shaping import SampleShapeChange
        >>> fh = baseband.open(baseband.data.SAMPLE_VDIF)
        >>> fh.frequency = 311.25 * u.MHz + (np.arange(8.) // 2) * 16. * u.MHz
        >>> fh.sideband = 1
        >>> fh.polarization = np.tile(['L', 'R'], 4)
        >>> rh = Reshape(fh, (4, 2))
        >>> rh.read(2).shape
        (2, 4, 2)
        >>> rh.polarization
        array(['L', 'R'], dtype='<U1')
        >>> rh.frequency
        <Quantity [[311.25],
                   [327.25],
                   [343.25],
                   [359.25]] MHz>
        >>> rh.sideband
        array(1, dtype=int8)
    """

    def __init__(self, ih, sample_shape):
        self._new_shape = (-1,) + sample_shape
        super().__init__(ih)

    def task(self, data):
        """Reshape the data."""
        return data.reshape(self._new_shape)


class Transpose(SampleShapeChangeBase):
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
    SampleShapeChange : to change the samples with a user-supplied function.

    Examples
    --------
    The VDIF example file from ``Baseband`` has 8 threads which contain
    4 channels and 2 polarizations.  To produce a stream in which the sample
    axes are polarization and frequency, one could do::

        >>> import numpy as np, astropy.units as u, baseband
        >>> from scintillometry.shaping import SampleShapeChange
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
        >>> th.frequency
        <Quantity [311.25, 327.25, 343.25, 359.25] MHz>
        >>> th.sideband
        array(1, dtype=int8)

    Note that the example above could also be done in one go using
    `~scintillometry.shaping.ReshapeAndTranspose`.
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
    before feeding a stream to, e.g., `~scintillometry.functions.Power`.

    This is just the combination of `~scintillometry.functions.Reshape` and
    `~scintillometry.functions.Transpose` (avoiding intermediate results).

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
    SampleShapeChange : to change the samples with a user-supplied function.

    Examples
    --------
    The VDIF example file from ``Baseband`` has 8 threads which contain
    4 channels and 2 polarizations.  To produce a stream in which the sample
    axes are polarization and frequency, one could do::

        >>> import numpy as np, astropy.units as u, baseband
        >>> from scintillometry.shaping import SampleShapeChange
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
        >>> rth.frequency
        <Quantity [311.25, 327.25, 343.25, 359.25] MHz>
        >>> rth.sideband
        array(1, dtype=int8)
    """

    def __init__(self, ih, sample_shape, sample_axes):
        self._axes = (0,) + sample_axes
        super().__init__(ih, sample_shape=sample_shape)

    def task(self, data):
        """Reshape and transpose the axes of data."""
        return data.reshape(self._new_shape).transpose(self._axes)
