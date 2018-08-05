"""
Portfolio module.
"""

from __future__ import unicode_literals

import cvxopt as opt
import numpy as np

from cvxopt import solvers
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
        cov = self.cov
        N = len(self.__stock_list)
        pbar = self.R

        # define list of optimal / desired mus for which we'd like to find the
        # optimal sigmas
        optimal_mus = []
        r_min = pbar.mean()    # minimum expected return
        for i in range(50):
            optimal_mus.append(r_min)
            r_min += (pbar.mean() / 100)

        # constraint matrices for quadratic programming
        P = opt.matrix(cov)
        q = opt.matrix(np.zeros((N, 1)))
        G = opt.matrix(np.concatenate((-np.array(pbar), -np.identity(N)), 0))
        A = opt.matrix(1.0, (1, N))
        b = opt.matrix(1.0)
        h = opt.matrix(np.concatenate((-np.ones((1, 1)), np.zeros((N, 1))), 0))

        # hide optimization
        opt.solvers.options['show_progress'] = False

        # calculate portfolio weights, every weight vector is of size Nx1
        # find optimal weights with qp(P, q, G, h, A, b)
        optimal_weights = solvers.qp(P, q, G, h * mean, A, b)['x']

        return self.evaluate(list(optimal_weights))

    def get_efficient_frontier(self, n_points=100):
        """ Get points that belong to the Efficient Frontier.

        >>> portfolio.get_efficient_frontier(5)
        [0.24942349584788953, 0.24942349967976762, 0.2795250781144858,
        0.3340090122172212, 0.38899556405336044]
        [0.23681240830982317, 0.23681240830982359, 0.2515909827391488,
        0.35350569620087896, 0.5596628149840878]

        :param n_points: Portfolio's expected return.
        :type n_points: int (optional. Default: 100)

        :returns: Points' means.
        :rtype: list
        :returns: Points' variances.
        :rtype: list
        """
        h_mean = 0
        l_mean = np.inf
        for stock in self.__stock_list:
            mean = stock.get_mean()
            if mean > h_mean:
                h_mean = mean
            if mean < l_mean:
                l_mean = mean

        mean_list = []
        variance_list = []
        delta = (h_mean - l_mean) / float(n_points)
        for i in range(n_points):
            expected_mean = l_mean + i * delta
            mean, variance, _, _ = self.get_minimum_variance_portfolio(
                expected_mean)
            mean_list.append(mean)
            variance_list.append(variance)
        return mean_list, variance_list

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
        def fitness(W, R, C, rf):
            _, _, sharp_ratio, _ = self.evaluate(W, rf)
            return 1 / sharp_ratio

        # Set uniform initial weights
        n = len(self.__stock_list)
        W = np.ones([n]) / n
        # Define constraints and bounds
        # Sum of weights = 1
        # Weights between 0 and 1 (no shorting)
        constraints = ({'type': 'eq', 'fun': lambda W: sum(W) - 1.})
        bounds = [(0., 1.) for i in range(n)]

        # Run optimizer
        optimized = optimize.minimize(
            fitness, W, (self.R, self.cov, rf_rate),
            method='SLSQP', constraints=constraints, bounds=bounds)
        if not optimized.success:
            raise BaseException(optimized.message)

        return self.evaluate(optimized.x, rf_rate)
