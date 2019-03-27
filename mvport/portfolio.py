"""
Portfolio module.
"""

from __future__ import unicode_literals

import numpy as np
from numpy.linalg import pinv

from scipy import optimize

from .utils import get_random_weights
from .stock import Stock


class Portfolio(object):
    """Class for calculating mean-variance portfolios.

    Portifolio optimization is done using mean-variance analysis. It is
    mathematical framework, developed by Harry Markowitz on 1952, for
    assembling a portfolio of assets such that the expected return is
    maximized for a given level of risk.

    >>> import mvport as mv
    >>>
    >>> portfolio = mv.Portfolio()
    """

    def __init__(self):
        """Instantiate Stock class."""
        self.__stock_list = []

    def __recalculate_parameters(self):
        self.cov = np.matrix(np.cov(
            [s.get_returns() for s in self.__stock_list]))
        self.R = np.matrix([s.get_mean() for s in self.__stock_list])

    def add_stock(self, ticker, returns):
        """Add stock to portfolio.

        >>> stock.add_stock('YHOO', [ 0.78030572, -0.45237186, -0.59878088])

        :param ticker: Stock's ticker.
        :type ticker: string
        :param returns: Stock's returns.
        :type returns: list
        """
        if ticker in self.__stock_list:
            raise ValueError("Stock is already part of this portfolio.")
        self.__stock_list.append(Stock(ticker, returns))
        self.__recalculate_parameters()

    def get_stock(self, ticker):
        """Get stock from portfolio.

        >>> stock.add_stock('YHOO')

        :returns: Stock.
        :rtype: Stock
        """
        for stock in self.__stock_list:
            if ticker == stock.get_ticker():
                return stock
        raise ValueError("Stock is not part of this portfolio.")

    def get_stock_list(self):
        """Get stock from portfolio.

        >>> stock.add_stock('YHOO')

        :returns: Portfolio's stock list.
        :rtype: list
        """
        return self.__stock_list

    def remove_stock(self, ticker):
        """Get stock from portfolio.

        >>> stock.add_stock('YHOO')

        :returns: Stock.
        :rtype: Stock
        """
        self.__stock_list.remove(ticker)

    def get_means(self):
        """Get a list of the returns' mean for all the stocks on the portfolio.

        >>> stock.get_means()
        array([ 0.78030572, 0.45237186, 0.59878088, 0.83043576])

        :returns: Returns' mean of all the stocks.
        :rtype: matrix
        """
        return self.R

    def get_covariance(self):
        """Get a covariance matrix of the stock's returns on the portfolio.

        >>> stock.get_covariance()
        array([ 0.78030572, 0.45237186], [0.45237186, 0.78030572])

        :returns: Returns' mean of all the stocks.
        :rtype: matrix
        """
        return self.cov

    def generate_return_series(self, mean=0, variance=1, length=1000):
        """Generate a random return series.

        >>> portfolio.generate_return_series(length=10, mean=0, variance=1)
        array([ 0.78030572, -0.45237186, -0.59878088, -0.83043576, -0.58860494,
        0.05050412, -1.31361197,  1.31865382,  1.88060814,  2.01899647])

        :param mean: Returns mean.
        :type mean: float
        :param variance: Return's variance.
        :type variance: float
        :param length: Number of samples.
        :type length: integer

        :returns: Ticker.
        :rtype: array
        """
        import numpy as np
        return np.random.normal(mean, np.sqrt(variance), length)

    def evaluate(self, weights=None, rf_rate=0.0, mode='weights'):
        """Evaluate portfolio with a given or random set of weights.

        >>> portfolio.evaluate(mode='random')
        0.78030572, 1.45237186, 0.803431562395,
        [-0.59878088, -0.83043576, -0.58860494]

        :param weights: List of weiths.
        :type weights: list (optional. Default: None)
        :param rf_rate: Risk free return rate.
        :type rf_rate: float (optional. Default: 0.0)
        :param mode: Evaluation mode, either by 'weigths'
        given or using 'random' weights.
        :type mode: string

        :returns: Portfolio's mean.
        :rtype: float
        :returns: Portfolio's variance.
        :rtype: float
        :returns: Portfolio's Sharp Ratio.
        :rtype: float
        :returns: Portfolio's weights.
        :rtype: matrix
        """
        # Covariance, returns and weights
        if mode == 'random':
            weights = get_random_weights(len(self.__stock_list))

        # Calculate expected portfolio return and variance
        w = np.matrix(weights)
        mean = float(w * self.R.T)
        variance = float(w * self.cov * w.T)
        sharp_ratio = (mean - rf_rate) / variance

        # Save weights on each stock
        for i, stock in enumerate(self.__stock_list):
            stock.set_portfolio_weight(w[0, i])

        return mean, variance, sharp_ratio, w

    def get_return(self, return_per_stock):
        """Evaluate portfolio return.

        >>> portfolio.get_return({
            'AAPL': 0.2544,
            'YHOO': -0.0245
        })
        0.19878088

        :param return_per_stock: Dictionary with returns of each stock.
        :type return_per_stock: dict 

        :returns: Portfolio's return.
        :rtype: float
        """
        total_return = 0
        for stock in  self.__stock_list:
            ticker = stock.get_ticker()
            total_return += return_per_stock[ticker] * stock.get_portfolio_weight()

        return total_return


    def get_minimum_variance_portfolio(self, mean, rf_rate=0):
        """ Get the portfolio that reduces variance for a given return.

        >>> portfolio.get_minimum_variance_portfolio(0.4)
        0.4, 1.45237186, 0.803431562395,
        [-0.59878088, -0.83043576, -0.58860494]

        :param mean: Portfolio's expected return.
        :type mean: list (optional. Default: None)
        :param rf_rate: Risk free return rate.
        :type rf_rate: float (optional. Default: 0.0)

        :returns: Portfolio's mean.
        :rtype: float
        :returns: Portfolio's variance.
        :rtype: float
        :returns: Portfolio's Sharp Ratio.
        :rtype: float
        :returns: Portfolio's weights.
        :rtype: matrix
        """
        N = len(self.__stock_list)
        one_vec = np.ones((N, 1))
        try:
            cov_inv = np.linalg.inv(self.cov)
        except Exception as e:
            cov_inv = np.linalg.pinv(self.cov)
        a = one_vec.T * cov_inv * one_vec
        b = one_vec.T * cov_inv * self.R.T
        c = self.R * cov_inv * self.R.T
        delta = a * c - b**2
        l1 = (c - b * mean) / delta
        l2 = (a * mean - b) / delta

        optimal_weights = cov_inv * (one_vec * l1 + self.R.transpose() * l2)
        optimal_weights = optimal_weights.reshape((-1)).tolist()[0]

        return self.evaluate(list(optimal_weights))

    def get_efficient_frontier(self, n_points=100, max_mean=None):
        """ Get points that belong to the Efficient Frontier.

        >>> portfolio.get_efficient_frontier(5)
        [0.24942349584788953, 0.24942349967976762, 0.2795250781144858,
        0.3340090122172212, 0.38899556405336044]
        [0.23681240830982317, 0.23681240830982359, 0.2515909827391488,
        0.35350569620087896, 0.5596628149840878]

        :param n_points: Portfolio's expected return.
        :type n_points: int (optional. Default: 100)
        :param max_mean: Efficient Frontier's maximum mean.
        :type max_mean: int (optional. Default: Maximum mean among R)

        :returns: Points' means.
        :rtype: list
        :returns: Points' variances.
        :rtype: list
        """

        N = len(self.__stock_list)
        one_vec = np.ones((N, 1))
        try:
            cov_inv = np.linalg.inv(self.cov)
        except Exception as e:
            cov_inv = np.linalg.pinv(self.cov)
        a = one_vec.T * cov_inv * one_vec
        b = one_vec.T * cov_inv * self.R.T
        c = self.R * cov_inv * self.R.T
        delta = a * c - b**2
        min_mean = float(b/a)
        if not max_mean:
            max_mean = np.max(self.R)
        mean = np.linspace(min_mean, max_mean, n_points)
        # mean = mean.append(max_r)
        var = (a * mean**2 - 2 * b * mean + c) / delta

        return mean, var.T


    def get_tangency_portfolio(self, rf_rate=0.0):
        """ Get the tangency portfolio.

        The tangency portfolio is the portfolio that maximizes the sharp ratio
        for a given risk free return rate. It is used SLSQP optimization in
        order to find the tangency portfolio.

        >>> portfolio.get_tangency_portfolio(0.2)
        0.3, 1.25237186, 0.883431562395,
        [-0.59878088, -0.83043576, -0.58860494]

        :param rf_rate: Risk free return rate.
        :type rf_rate: float (optional. Default: 0.0)

        :returns: Portfolio's mean.
        :rtype: float
        :returns: Portfolio's variance.
        :rtype: float
        :returns: Portfolio's Sharp Ratio.
        :rtype: float
        :returns: Portfolio's weights.
        :rtype: matrix
        """
        # Function to be minimized
        N = len(self.__stock_list)
        one_vec = np.ones((N, 1))
        try:
            cov_inv = np.linalg.inv(self.cov)
        except Exception as e:
            cov_inv = np.linalg.pinv(self.cov)
        a = one_vec.T * cov_inv * one_vec
        b = one_vec.T * cov_inv * self.R.T
        optimal_weights = (cov_inv * (self.R.T - rf_rate * one_vec)) / (b - a * rf_rate)
        optimal_weights = optimal_weights.reshape((-1)).tolist()[0]

        return self.evaluate(list(optimal_weights))

