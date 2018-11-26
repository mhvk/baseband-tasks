# Licensed under the GPLv3 - see LICENSE

import numpy as np
from numpy.testing import assert_array_equal
import astropy.units as u
import pytest

from ..base import BaseData


class TestBaseData:
    def setup(self):
        self.data = np.array([[1., 2., 2.],
                              [0., 3., 2.]])
        self.count = np.array([[1, 2, 2],
                               [0, 3, 2]])
        self.base_data = self.data.view(BaseData)
        self.base_data.count = self.count

    def test_copy(self):
        bd_copy = self.base_data.copy()
        assert_array_equal(bd_copy, self.base_data)
        assert_array_equal(bd_copy.count, self.count)

    @pytest.mark.parametrize('item', [0, (1, 0), (slice(None), 1),
                                      (slice(1, 2), slice(0, 2))])
    def test_get_item(self, item):
        bd_item = self.base_data[item]
        d_item = self.data[item]
        c_item = self.count[item]
        assert bd_item.shape == d_item.shape == c_item.shape
        assert_array_equal(bd_item.view(np.ndarray), d_item)
        assert_array_equal(bd_item.count, c_item)

    @pytest.mark.parametrize('item', [0, (1, 0), (slice(None), 1),
                                      (slice(1, 2), slice(0, 2))])
    def test_set_item(self, item):
        bd = self.base_data.copy()
        d = self.data.copy()
        c = self.count.copy()
        value = bd[1, 2]
        bd[item] = value
        d[item] = np.array(value)
        c[item] = value.count
        assert_array_equal(bd.view(np.ndarray), d)
        assert_array_equal(bd.count, c)
        bd[item] = self.base_data[item]
        assert_array_equal(bd.view(np.ndarray), self.data)
        assert_array_equal(bd.count, self.count)
