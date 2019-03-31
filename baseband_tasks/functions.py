# Licensed under the GPLv3 - see LICENSE
import numpy as np

from .base import TaskBase, check_broadcast_to, simplify_shape


__all__ = ['Square', 'Power']


def complex_square(z):
    return np.square(z.real) + np.square(z.imag)


class Square(TaskBase):
    """Converts samples to intensities by squaring.

    Note that `Square` does not keep full polarization information;
    it simply calculates the power for each polarization.  Use
    `~baseband_tasks.functions.Power` to also calculate cross products.

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream.
    polarization : array or (nested) list of char, optional
        Polarization labels.  Should broadcast to the sample shape,
        i.e., the labels are in the correct axis.  For instance,
        ``['X', 'Y']``, or ``[['L'], ['R']]``.  By default, taken
        from the underlying stream (and ignored if not given).
        Output labels will have the polarization labels doubled,
        i.e., ``['XX', 'YY']``, ``[['LL'], ['RR']]``, etc.
    """

    def __init__(self, ih, polarization=None):
        ih_dtype = np.dtype(ih.dtype)
        self.task = complex_square if ih_dtype.kind == 'c' else np.square
        dtype = self.task(np.zeros(1, dtype=ih_dtype)).dtype
        super().__init__(ih, dtype=dtype, polarization=polarization)
        if self._polarization is not None:
            self._polarization = np.core.defchararray.add(
                self._polarization, self._polarization)


class Power(TaskBase):
    """Calculate powers and cross terms for two polarizations.

    For polarizations X and Y, 4 terms are produced:

    ==== ========= ========================= ==========
    Term Value     Expanded                  Other name
    ==== ========= ========================= ==========
    XX   Re(X X*)  Re(X)**2 + Im(X)**2       AA
    YY   Re(Y Y*)  Re(Y)**2 + Im(Y)**2       BB
    XY   Re(X Y*)  Re(X)*Re(Y) + Im(X)*Im(Y) CR
    YX   Im(X Y*)  Im(X)*Re(Y) - Re(X)*Im(Y) CI
    ==== ========= ========================= ==========

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream.
    polarization : array or (nested) list of char, optional
        Polarization labels.  Should broadcast to the sample shape,
        i.e., the labels are in the correct axis.  For instance,
        ``['X', 'Y']``, or ``[['L'], ['R']]``.  By default, taken
        from the underlying stream.

    Raises
    ------
    AttributeError
        If no polarization information is given.
    ValueError
        If the underlying stream is not complex, the number of polarizations
        not equal to two, or the polarization labels not unique.
    """

    def __init__(self, ih, polarization=None):
        if polarization is None:
            polarization = ih.polarization
        else:
            # Check that input has consistent shape
            broadcast = check_broadcast_to(polarization, ih.sample_shape)
            polarization = simplify_shape(broadcast)

        if polarization.size != 2:
            raise ValueError("need exactly 2 polarizations.  Reshape stream "
                             "appropriately.")
        # polarization is guaranteed to have 2 distinct items.
        pol_axis = polarization.shape.index(2)
        pol_swap = polarization.swapaxes(0, pol_axis)
        pol_swap = np.core.defchararray.add(pol_swap[[0, 1, 0, 1]],
                                            pol_swap[[0, 1, 1, 0]])
        polarization = pol_swap.swapaxes(0, pol_axis)

        self._axis = ih.ndim - polarization.ndim + pol_axis
        shape = ih.shape[:self._axis] + (4,) + ih.shape[self._axis+1:]

        ih_dtype = np.dtype(ih.dtype)
        if ih_dtype.kind != 'c':
            raise ValueError("{} only works on a complex timestream.")
        dtype = np.zeros(1, ih_dtype).real.dtype

        super().__init__(ih, shape=shape, polarization=polarization,
                         dtype=dtype)

    def task(self, data):
        """Calculate the polarization powers and cross terms for one frame."""
        result = np.empty(data.shape[:1] + self.shape[1:], self.dtype)
        # Get views in which the axis with the polarization is first.
        in_ = data.swapaxes(0, self._axis)
        out = result.swapaxes(0, self._axis)
        out[0] = complex_square(in_[0])
        out[1] = complex_square(in_[1])
        c = in_[0] * in_[1].conj()
        out[2] = c.real
        out[3] = c.imag
        return result


class Digitize(TaskBase):
    """Digitize a stream to a given number of bits.

    Output values are between -(2**(bps-1)) and 2**(bps-1) -1.
    For instance, between -8 and 7 for bps=4.

    Parameters
    ----------
    ih : stream handle
        Handle of a stream reader or another task.
    nbits : int
        Number of bits to digitize too.  For complex data, the real
        and imaginary components are digitized separately.
    """
    def __init__(self, ih, bps, scale=1.):
        super().__init__(ih)
        self._low = - (1 << (bps-1))
        self._high = (1 << (bps-1)) - 1
        if self.complex_data:
            real_dtype = np.zeros(1, self.dtype).real.dtype
            self.task = lambda data: self._digitize(
                data.view(real_dtype)).view(self.dtype)
        else:
            self.task = self._digitize

    def _digitize(self, data):
        return np.clip(data.round(), self._low, self._high)
