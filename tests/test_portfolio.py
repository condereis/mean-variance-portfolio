#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `mvport` package."""


import unittest
import numpy as np

from mvport.portfolio import Portfolio


class TestPortfolio(unittest.TestCase):
    """Tests for `mvport` package."""

    def setUp(self):
        """SetUp."""
        self.ticker1 = 'AAPL'
        self.ticker2 = 'AMZN'
        self.returns1 = [.1,.2,.3]
        self.returns2 = [.1,.3,.5]
        self.portfolio = Portfolio()
        self.portfolio.add_stock(self.ticker1, self.returns1)
        self.portfolio.add_stock(self.ticker2, self.returns2)

    def test_get_stock(self):
        """Test get_stock."""
        stock = self.portfolio.get_stock(self.ticker1)
        self.assertEqual(stock.get_ticker(), self.ticker1)
        np.testing.assert_array_equal(stock.get_returns(), np.array(self.returns1))

    def test_get_stock_list(self):
        """Test get_stock_list."""
        stock_list = self.portfolio.get_stock_list()
        self.assertEqual(len(stock_list), 2)
        self.assertEqual(stock_list[0].get_ticker(), self.ticker1)
        np.testing.assert_array_equal(stock_list[0].get_returns(), np.array(self.returns1))
        self.assertEqual(stock_list[1].get_ticker(), self.ticker2)
        np.testing.assert_array_equal(stock_list[1].get_returns(), np.array(self.returns2))

    def test_add_stock(self):
        """Test add_stock."""
        self.portfolio.add_stock('new_ticker', [1, 2, 3])
        stock = self.portfolio.get_stock('new_ticker')
        self.assertEqual(stock.get_ticker(), 'new_ticker')
        np.testing.assert_array_equal(stock.get_returns(), np.array([1, 2, 3]))

    def test_remove_stock(self):
        """Test remove_stock."""
        stock_list = self.portfolio.get_stock_list()
        self.assertEqual(len(stock_list), 2)
        self.portfolio.remove_stock(self.ticker1)
        stock_list = self.portfolio.get_stock_list()
        self.assertEqual(len(stock_list), 1)

    def test_get_means(self):
        """Test get_means."""
        mean_list = self.portfolio.get_means()
        # self.assertEqual(len(mean_list), 2)
        # self.assertEqual(mean_list[0], np.mean(self.returns1))
        # self.assertEqual(mean_list[1], np.mean(self.returns2))
        np.testing.assert_array_equal(mean_list, np.matrix([np.mean(self.returns1), np.mean(self.returns2)]))

    def test_get_covariance(self):
        """Test get_covariance."""
        cov = self.portfolio.get_covariance()
        cov = np.cov([self.returns1, self.returns2])
        np.testing.assert_array_equal(cov, np.cov([self.returns1, self.returns2]))

    def test_generate_return_series(self):
        """Test generate_return_series."""
        returns = self.portfolio.generate_return_series(length=1000000)
        self.assertEqual(round(np.mean(returns), 1), 0)
        self.assertEqual(round(np.var(returns), 1), 1)

    def test_evaluate(self):
        """Test evaluate."""
        risk_free = 0
        mean, variance, sharp_ratio, w = self.portfolio.evaluate([.5, .5], risk_free)
        self.assertEqual(mean, .25)
        self.assertEqual(variance, .0225)
        self.assertEqual(sharp_ratio, (mean - risk_free) / variance)
        np.testing.assert_array_equal(w, np.matrix([.5, .5]))

    def test_get_return(self):
        """Test get_return."""
        risk_free = 0
        mean, variance, sharp_ratio, w = self.portfolio.evaluate([.5, .5], risk_free)
        p_return = self.portfolio.get_return({
            self.ticker1: 0.0,
            self.ticker2: 1.0
        })
        self.assertEqual(p_return, 0.5)

    def test_get_minimum_variance_portfolio(self):
        """Test get_minimum_variance_portfolio."""
        risk_free = 0
        expected_return = .25
        mean, variance, sharp_ratio, w = self.portfolio.get_minimum_variance_portfolio(expected_return, risk_free)
        self.assertEqual(round(mean, 2), .25)
        self.assertEqual(round(variance, 4), .0225)
        self.assertEqual( round(sharp_ratio, 2), round(((mean - risk_free) / variance), 2))
        for i in w.tolist()[0]:
            for j in [.5, .5]:
                self.assertEqual(round(i, 2), j)

    def test_get_efficient_frontier(self):
        """Test get_efficient_frontier."""
        risk_free = 0
        n_points = 100
        means = []
        variances = []
        for _ in range(1000):
            mean, variance, s_, _ = self.portfolio.evaluate(mode='random')
            means.append(mean)
            variances.append(variance)
        e_means, e_variances = self.portfolio.get_efficient_frontier()
        for mean, variance in zip(means, variances):
            for e_mean, e_variance in zip(e_means, e_variances):
                self.assertEqual((e_mean > mean or e_variance < variance), True)

    def test_tangency_portfolio(self):
        """Test get_tangency_portfolio."""
        risk_free = 0
        n_points = 100
        means, variances = self.portfolio.get_efficient_frontier()
        t_mean, t_variance, _, _ = self.portfolio.get_tangency_portfolio(risk_free)
        tline_a = (t_mean - risk_free) / t_variance
        tline_b = risk_free
        for mean, variance in zip(means, variances):
            t_mean = tline_a * variance + tline_b
            self.assertGreaterEqual(t_mean, mean)


if __name__ == '__main__':
    sys.exit(unittest.main())
