=======================
Mean Variance Portfolio
=======================


.. image:: https://img.shields.io/pypi/v/mvport.svg
        :target: https://pypi.python.org/pypi/mvport

.. image:: https://img.shields.io/travis/condereis/mean-variance-portfolio.svg
        :target: https://travis-ci.org/condereis/mean-variance-portfolio

.. image:: https://readthedocs.org/projects/mean-variance-portfolio/badge/?version=latest
        :target: https://mean-variance-portfolio.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/condereis/mean-variance-portfolio/shield.svg
     :target: https://pyup.io/repos/github/condereis/mean-variance-portfolio/
     :alt: Updates



MV Port is a Python package to perform Mean-Variance Analysis. It provides a Portfolio class with a variety of methods to help on your portfolio optimization tasks.


* Free software: MIT license
* Documentation: https://mvport.readthedocs.io.

.. Modern portfolio theory (MPT), or mean-variance analysis, is a mathematical framework for assembling a portfolio of assets such that the expected return is maximized for a given level of risk. It is a formalization and extension of diversification in investing, the idea that owning different kinds of financial assets is less risky than owning only one type. Its key insight is that an asset's risk and return should not be assessed by itself, but by how it contributes to a portfolio's overall risk and return. It uses the variance of asset prices as a proxy for risk.

Features
--------

* Easy portfolio setup
* Portfolio evaluation
* Random portfolio allocation
* Minimum Variance Portfolio optimization
* Efficient Frontier evaluation
* Tangency Portfolio for a given risk free return rate


Installation
------------
To install MV Port, run this command in your terminal:

.. code:: bash

    $ pip install mvport

Check `here <https://mvport.readthedocs.io/en/latest/installation.html>`_  for further information on installation.

Basic Usage
-----------

Instantiate a portfolio and add some stock and evaluate it given a set of weights:

.. code:: python

    >>> import mvport as mv
    >>> p = mv.Portfolio()
    >>> p.add_stock('AAPL', [.1,.2,.3])
    >>> p.add_stock('AMZN', [.1,.3,.5])
    >>> mean, variance, sharp_ratio, weights = p.evaluate([.5, .5])
    >>> print '{} +- {}'.format(mean, variance)
    0.25 +- 0.0225

Check `here <https://mvport.readthedocs.io/en/latest/usage.html>`_  for further information on usage.