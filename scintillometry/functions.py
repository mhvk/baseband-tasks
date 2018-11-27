# Licensed under the GPLv3 - see LICENSE
import numpy as np

from .base import TaskBase


__all__ = ['Square', 'Power']


class Square(TaskBase):
    """Converts samples to intensities by squaring.

    Note that `Square` does not keep full polarization information;
    it simply calculates the power for each polarization.  Use
    `~scintillometry.functions.Power` to also calculate cross products.

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
        if ih_dtype.kind != 'c':
            square = np.square
        else:
            def square(x):
                return np.square(x.real) + np.square(x.imag)

        self.task = square
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
        ih_dtype = np.dtype(ih.dtype)
        if ih_dtype.kind != 'c':
            raise ValueError("{} only works on a complex timestream.")
        dtype = np.zeros(1, ih_dtype).real.dtype
        super().__init__(ih, polarization=polarization, dtype=dtype)
        polarization = self.polarization
        if polarization.size != 2:
            raise ValueError("need exactly 2 polarizations.  Reshape stream "
                             "appropriately.")
        pol_axis = polarization.shape.index(2)
        polarization = polarization.swapaxes(0, pol_axis)
        if polarization[0] == polarization[1]:
            raise ValueError("need 2 unique polarizations.")

        polarization = np.core.defchararray.add(polarization[[0, 1, 0, 1]],
                                                polarization[[0, 1, 1, 0]])
        self._polarization = polarization.swapaxes(0, pol_axis)
        self._axis = ih.ndim - polarization.ndim + pol_axis
        self._shape = self._shape[:self._axis] + (4,) + self._shape[self._axis+1:]

    def task(self, data):
        """Calculate the polarization powers and cross terms for one frame."""
        result = np.empty(data.shape[:1] + self.shape[1:], self.dtype)
        # Get views in which the axis with the polarization is first.
        in_ = data.swapaxes(0, self._axis)
        out = result.swapaxes(0, self._axis)
        r0 = in_[0].real
        i0 = in_[0].imag
        r1 = in_[1].real
        i1 = in_[1].imag
        out[0] = r0 ** 2 + i0 ** 2
        out[1] = r1 ** 2 + i1 ** 2
        out[2] = r0 * r1 + i0 * i1
        out[3] = i0 * r1 - r0 * i1
        return result
