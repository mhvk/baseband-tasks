# Licensed under the GPLv3 - see LICENSE
import numpy as np

from .base import TaskBase, simplify_shape


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
        Output polarization labels.  Should broadcast to the sample shape,
        i.e., the labels are in the correct axis.  For instance,
        ``['XX', 'YY']``, or ``[['LL'], ['RR']]``.  By default, doubled
        labels from the underlying stream (and ignored if not given).
    """

    def __init__(self, ih, polarization=None):
        if polarization is None:
            polarization = self._default_polarization(ih)

        ih_dtype = np.dtype(ih.dtype)
        self.task = complex_square if ih_dtype.kind == 'c' else np.square
        dtype = self.task(np.zeros(1, dtype=ih_dtype)).dtype

        super().__init__(ih, dtype=dtype, polarization=polarization)

    def _default_polarization(self, ih):
        if not hasattr(ih, 'polarization'):
            return None

        return np.core.defchararray.add(ih.polarization, ih.polarization)

    def _repr_item(self, key, default, value=None):
        if key == 'polarization':
            default = self._default_polarization(self.ih)
        return super()._repr_item(key, default=default, value=value)


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
        Output polarization labels.  Should broadcast to the output sample
        shape, i.e., the labels are in the correct axis.  For instance,
        ``['LL', 'RR', 'LR', 'RL']``.  By default, inferred from the
        underlying stream, using the scheme described above.

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
            polarization = self._default_polarization(ih)
        else:
            polarization = simplify_shape(np.asanyarray(polarization))
            if not (polarization.size == 4 == len(np.unique(polarization))
                    and 4 in polarization.shape):
                raise ValueError('output polarizations should have 4 unique '
                                 'elements along one axis.')

        self._axis = ih.ndim - polarization.ndim + polarization.shape.index(4)
        if ih.shape[self._axis] != 2:
            raise ValueError(f"input shape should be 2 along polarization axis"
                             f" ({self._axis}), not {ih.shape[self._axis]}.")
        shape = ih.shape[:self._axis] + (4,) + ih.shape[self._axis+1:]

        ih_dtype = np.dtype(ih.dtype)
        if ih_dtype.kind != 'c':
            raise ValueError("{} only works on a complex timestream.")
        dtype = np.zeros(1, ih_dtype).real.dtype

        super().__init__(ih, shape=shape, polarization=polarization,
                         dtype=dtype)

    def _default_polarization(self, ih):
        if ih.polarization.size != 2:
            raise ValueError("stream should have exactly 2 polarizations. "
                             "Reshape appropriately.")

        # Given check_shape, ih.polarization is guaranteed to have 2 distinct
        # items, which are guaranteed to be along first axis.
        return np.core.defchararray.add(ih.polarization[[0, 1, 0, 1]],
                                        ih.polarization[[0, 1, 1, 0]])

    def _repr_item(self, key, default, value=None):
        if (key == 'polarization' and hasattr(self.ih, 'polarization')
                and default is None):
            default = self._default_polarization(self.ih)
        return super()._repr_item(key, default=default, value=value)

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
