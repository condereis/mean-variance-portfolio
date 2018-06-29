=====
Usage
=====

To use Mean Variance Portfolio in a project::

.. code:: python

    >>> import mvport as mv

Instantiate a portfolio and add some stocks:

.. code:: python

    >>> p = mv.Portfolio()
    >>> p.add_stock('AAPL', [.1,.2,.3])
    >>> p.add_stock('AMZN', [.1,.3,.5])

Evaluate a portfolio given a set of weights:

.. code:: python

    >>> mean, variance, sharp_ratio, weights = p.evaluate([.5, .5])
    >>> print '{} +- {}'.format(mean, variance)
    0.25 +- 0.0225

Get the portfolio that minimizes risk for a given expected return:

.. code:: python

    >>> expected_return = 0.25
    >>> mean, variance, _, w = p.get_minimum_variance_portfolio(expected_return)
    >>> print 'weights: {} \n {} +- {}'.format(w, mean, variance)
    weights: [[0.49999993 0.50000007]] 
     0.25000000746 +- 0.022500002238

Get tangency portfolio for a given risk free asset:

.. code:: python

    >>> risk_free_rate = 0.2
    >>> mean, variance, _, w = p.get_tangency_portfolio(risk_free_rate)
    >>> print 'weights: {} \n {} +- {}'.format(w, mean, variance)
    weights: [[2.64767716e-04 9.99735232e-01]] 
     0.299973523228 +- 0.0399894099924
