"""
Stock module.
"""

from __future__ import unicode_literals

import numpy as np


class Stock(object):
    """Class for handling stocks.

    >>> import mvport as mv
    >>>
    >>> stock = mv.Stock('AAPL')
    >>> print(stock)
    <Stock AAPL>

    :param ticker: Stock ticker.
    :type ticker: string
    :param ticker: Stock's returns.
    :type ticker: list
    """

    def __init__(self, ticker, returns):
        """Instantiate Stock class."""
        self.__ticker = ticker
        self.set_returns(returns)

    def __repr__(self):
        """An unambiguous representation of a Stock's instance."""
        return '<Stock {ticker}>'.format(ticker=self.__ticker)

    def __hash__(self):
        """Hash representation of a Stock's instance."""
        return self.__ticker

    def __eq__(self, other):
        """Equality comparison operator."""
        if isinstance(other, Stock):
            return self.__hash__() == other.__hash__()
        if isinstance(other, str):
            return self.__hash__() == other
        return False

    def __ne__(self, other):
        """Inquality comparison operator."""
        return self.__hash__() != other.__hash__()

    def get_ticker(self):
        """Get stock's ticker.

        >>> stock.get_ticker()
        'AAPL'

        :returns: Ticker.
        :rtype: string
        """
        return self.__ticker

    def set_ticker(self, ticker):
        """Set stock's ticker.

        >>> stock.set_ticker('YHOO')
        >>> print(stock)
        <Stock YHOO>

        :param ticker: Stock ticker.
        :type ticker: string
        """
        self.__ticker = ticker

    def get_returns(self):
        """Get stock's returns.

        >>> stock.get_returns()
        array([ 0.78030572, -0.45237186, -0.59878088, -0.83043576, -0.58860494,
        0.05050412, -1.31361197,  1.31865382,  1.88060814,  2.01899647])

        :returns: Stock returns.
        :rtype: array
        """
        return self.__returns

    def set_returns(self, returns):
        """Set stock's returns.

        >>> stock.set_returns([
        0.78030572, -0.45237186, -0.59878088, -0.83043576, -0.58860494,
        0.05050412, -1.31361197,  1.31865382,  1.88060814,  2.01899647]))

        :param ticker: Stock returns.
        :type ticker: list
        """
        self.__returns = np.array(returns)
        self.__mean = np.mean(self.__returns)
        self.__variance = np.var(self.__returns)

    def get_mean(self):
        """Get stock's mean return.

        >>> stock.get_returns()
        array([ 0.78030572, -0.45237186, -0.59878088, -0.83043576, -0.58860494,
        0.05050412, -1.31361197,  1.31865382,  1.88060814,  2.01899647])

        :returns: Stock returns.
        :rtype: array
        """
        return self.__mean

    def get_variance(self):
        """Get stock's mean return.

        >>> stock.get_returns()
        array([ 0.78030572, -0.45237186, -0.59878088, -0.83043576, -0.58860494,
        0.05050412, -1.31361197,  1.31865382,  1.88060814,  2.01899647])

        :returns: Stock returns.
        :rtype: array
        """
        return self.__variance

    def get_portfolio_weight(self):
        """Get stock's weight on the portfolio.

        >>> stock.get_portfolio_weight()
        0.2

        :returns: Stock weight.
        :rtype: float
        """
        return self.__weight

    def set_portfolio_weight(self, weight):
        """Set stock's weight on the portfolio.

        >>> stock.set_returns(0.2)

        :param ticker: Stock weight.
        :type ticker: float
        """
        self.__weight = weight
