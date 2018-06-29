#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `mvport` package."""


import unittest
import numpy as np

from mvport.stock import Stock


class TestStock(unittest.TestCase):
    """Tests for `mvport` package."""

    def setUp(self):
        """SetUp."""
        self.ticker = 'AAPL'
        self.returns = [-2, -1, 0, 1, 2]
        self.stock = Stock(self.ticker, self.returns)

    def test_get_ticker(self):
        """Test get_ticker."""
        self.assertEqual(self.stock.get_ticker(), self.ticker)

    def test_set_ticker(self):
        """Test set_ticker."""
        self.stock.set_ticker('new_ticker')
        self.assertEqual(self.stock.get_ticker(), 'new_ticker')

    def test_get_returns(self):
        """Test get_returns."""
        np.testing.assert_array_equal(self.stock.get_returns(), np.array(self.returns))

    def test_set_returns(self):
        """Test set_ticker."""
        self.stock.set_returns([-1, 0, 1])
        np.testing.assert_array_equal(self.stock.get_returns(), np.array([-1, 0, 1]))

    def test_get_mean(self):
        """Test get_mean."""
        self.assertEqual(self.stock.get_mean(), 0)
        self.stock.set_returns([0, 1, 2])
        self.assertEqual(self.stock.get_mean(), 1)

    def test_get_variance(self):
        """Test get_variance."""
        self.assertEqual(self.stock.get_variance(), 2)
        self.stock.set_returns([-3,-1,0,1,3])
        self.assertEqual(self.stock.get_variance(), 4)


if __name__ == '__main__':
    sys.exit(unittest.main())
