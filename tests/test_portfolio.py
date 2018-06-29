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

    # def get_means(self):
    #     """Get a list of the returns' mean for all the stocks on the portfolio.

    #     >>> stock.get_means()
    #     array([ 0.78030572, 0.45237186, 0.59878088, 0.83043576])

    #     :returns: Returns' mean of all the stocks.
    #     :rtype: list
    #     """
    #     return self.__stock_list

    # def get_covariance(self):
    #     """Get a covariance matrix of the stock's returns on the portfolio.

    #     >>> stock.get_covariance()
    #     array([ 0.78030572, 0.45237186], [0.45237186, 0.78030572])

    #     :returns: Returns' mean of all the stocks.
    #     :rtype: list
    #     """
    #     return self.__stock_list

    # def generate_return_series(self, mean=0, variance=1, length=1000):
    #     """Generate a random return series.

    #     >>> portfolio.generate_return_series(length=10, mean=0, variance=1)
    #     array([ 0.78030572, -0.45237186, -0.59878088, -0.83043576, -0.58860494,
    #     0.05050412, -1.31361197,  1.31865382,  1.88060814,  2.01899647])

    #     :param mean: Returns mean.
    #     :type mean: float
    #     :param variance: Return's variance.
    #     :type variance: float
    #     :param length: Number of samples.
    #     :type length: integer

    #     :returns: Ticker.
    #     :rtype: array
    #     """
    #     import numpy as np
    #     return np.random.normal(mean, np.sqrt(variance), length)

    # def evaluate(self, weights=None, rf_rate=0.0, mode='weights'):
    #     """Evaluate portfolio with a given or random set of weights.

    #     >>> portfolio.evaluate(mode='random')
    #     0.78030572, 1.45237186, 0.803431562395, [-0.59878088, -0.83043576, -0.58860494]

    #     :param weights: List of weiths.
    #     :type weights: list (optional. Default: None)
    #     :param rf_rate: Risk free return rate.
    #     :type rf_rate: float (optional. Default: 0.0)
    #     :param mode: Evaluation mode, either by 'weigths' given or using 'random' weights.
    #     :type mode: string

    #     :returns: Portfolio's mean.
    #     :rtype: float
    #     :returns: Portfolio's variance.
    #     :rtype: float
    #     :returns: Portfolio's Sharp Ratio.
    #     :rtype: float
    #     :returns: Portfolio's weights.
    #     :rtype: list
    #     """
    #     # Covariance, returns and weights
    #     if mode == 'random':
    #         weights = get_random_weights(len(self.__stock_list))

    #     # Calculate expected portfolio return and variance
    #     w = np.matrix(weights)
    #     mean = float(w * self.R.T)
    #     variance = float(w * self.cov * w.T)
    #     sharp_ratio = (mean - rf_rate) / variance
    #     return mean, variance, sharp_ratio, w

    # def get_minimum_variance_portfolio(self, mean, rf_rate=0):
    #     """ Get the portfolio that reduces variance for a given expected return.

    #     >>> portfolio.get_minimum_variance_portfolio(0.4)
    #     0.4, 1.45237186, 0.803431562395, [-0.59878088, -0.83043576, -0.58860494]

    #     :param mean: Portfolio's expected return.
    #     :type mean: list (optional. Default: None)
    #     :param rf_rate: Risk free return rate.
    #     :type rf_rate: float (optional. Default: 0.0)

    #     :returns: Portfolio's mean.
    #     :rtype: float
    #     :returns: Portfolio's variance.
    #     :rtype: float
    #     :returns: Portfolio's Sharp Ratio.
    #     :rtype: float
    #     :returns: Portfolio's weights.
    #     :rtype: list
    #     """
    #     cov = self.cov
    #     N = len(self.__stock_list)
    #     pbar = self.R
        
    #     # define list of optimal / desired mus for which we'd like to find the optimal sigmas
    #     optimal_mus = []
    #     r_min = pbar.mean()    # minimum expected return
    #     for i in range(50):
    #         optimal_mus.append(r_min)
    #         r_min += (pbar.mean() / 100)
        
    #     # constraint matrices for quadratic programming
    #     P = opt.matrix(cov)
    #     q = opt.matrix(np.zeros((N, 1)))
    #     G = opt.matrix(np.concatenate((-np.array(pbar), -np.identity(N)), 0))
    #     A = opt.matrix(1.0, (1,N))
    #     b = opt.matrix(1.0)
    #     h = opt.matrix(np.concatenate((-np.ones((1, 1)), np.zeros((N, 1))), 0))
        
    #     # hide optimization
    #     opt.solvers.options['show_progress'] = False
        
    #     # calculate portfolio weights, every weight vector is of size Nx1
    #     # find optimal weights with qp(P, q, G, h, A, b)
    #     optimal_weights = solvers.qp(P, q, G, h * mean, A, b)['x']

    #     return self.evaluate(list(optimal_weights))

    # def get_efficient_frontier(self, n_points=100):
    #     """ Get points that belong to the Efficient Frontier.

    #     >>> portfolio.get_efficient_frontier(5)
    #     [0.24942349584788953, 0.24942349967976762, 0.2795250781144858, 0.3340090122172212, 0.38899556405336044]
    #     [0.23681240830982317, 0.23681240830982359, 0.2515909827391488, 0.35350569620087896, 0.5596628149840878]

    #     :param n_points: Portfolio's expected return.
    #     :type n_points: int (optional. Default: 100)

    #     :returns: Points' means.
    #     :rtype: list
    #     :returns: Points' variances.
    #     :rtype: list
    #     """
    #     h_mean = 0
    #     l_mean= np.inf
    #     for stock in self.__stock_list:
    #         mean = stock.get_mean()
    #         if mean > h_mean:
    #             h_mean = mean
    #         if mean < l_mean:
    #             l_mean = mean

    #     mean_list = []
    #     variance_list = []
    #     delta = (h_mean - l_mean) / float(n_points)
    #     for i in range(n_points):
    #         expected_mean = l_mean + i * delta
    #         mean, variance, _, _ = self.get_minimum_variance_portfolio(expected_mean)
    #         mean_list.append(mean)
    #         variance_list.append(variance)
    #     return mean_list, variance_list

    # def get_tangency_portfolio(self, rf_rate=0.0):
    #     """ Get the tangency portfolio.

    #     The tangency portfolio is the portfolio that maximizes the sharp ratio for a given
    #     risk free return rate. It is used SLSQP optimization in order to find the tangency portfolio.

    #     >>> portfolio.get_tangency_portfolio(0.2)
    #     0.3, 1.25237186, 0.883431562395, [-0.59878088, -0.83043576, -0.58860494]

    #     :param rf_rate: Risk free return rate.
    #     :type rf_rate: float (optional. Default: 0.0)

    #     :returns: Portfolio's mean.
    #     :rtype: float
    #     :returns: Portfolio's variance.
    #     :rtype: float
    #     :returns: Portfolio's Sharp Ratio.
    #     :rtype: float
    #     :returns: Portfolio's weights.
    #     :rtype: list
    #     """
    #     # Function to be minimized
    #     def fitness(W, R, C, rf):
    #         _, _, sharp_ratio, _ = self.evaluate(W, rf)
    #         return 1/sharp_ratio

    #     # Set uniform initial weights
    #     n = len(self.__stock_list)
    #     W = np.ones([n])/n
    #     # Define constraints and bounds
    #     # Sum of weights = 1
    #     # Weights between 0 and 1 (no shorting)
    #     constraints = ({'type':'eq', 'fun': lambda W: sum(W)-1. })
    #     bounds = [(0.,1.) for i in range(n)]

    #     # Run optimizer
    #     optimized = optimize.minimize(fitness, W, (self.R, self.cov, rf_rate), 
    #                 method='SLSQP', constraints=constraints, bounds=bounds)  
    #     if not optimized.success: 
    #         raise BaseException(optimized.message)
    #     return self.evaluate(optimized.x, rf_rate)
