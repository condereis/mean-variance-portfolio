#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    project_root = path.dirname(path.dirname(path.abspath(__file__)))
    sys.path.append(project_root)


import matplotlib.pyplot as plt
import mvport as mv
import numpy as np

NUM_SAMPLES = 1000
NUM_PORTFOLIOS = 10000

# Define stock's statistics
stocks = [
    {'ticker': 'STOCK1', 'mean': 0.15, 'variance': 0.5},
    {'ticker': 'STOCK2', 'mean': 0.25, 'variance': 1.0},
    {'ticker': 'STOCK3', 'mean': 0.4, 'variance': 1.2},
    {'ticker': 'STOCK4', 'mean': 0.5, 'variance': 1.6}
]

# Instantiate portfolio and add stocks
portfolio = mv.Portfolio()
# for stock in stocks:
#     returns = portfolio.generate_return_series(stock['mean'], stock['variance'], NUM_SAMPLES)
#     portfolio.add_stock(stock['ticker'], returns)
portfolio.add_stock('t1', [.1,.2,.3])
portfolio.add_stock('t1', [.1,.3,.5])
# Evaluate random weight allocation
mean_list = []
variance_list = []
for _ in range(NUM_PORTFOLIOS):
    mean, variance, sr, w = portfolio.evaluate(mode='random')
    mean_list.append(mean)
    variance_list.append(variance)

# Get efficient frontier
efficient_means, efficient_variances = portfolio.get_efficient_frontier()
print efficient_variances
# Plot random allocation and efficient frontier
plt.plot(variance_list, mean_list, 'o', markersize=3)
plt.plot(efficient_variances, efficient_means, 'y-o', markersize=3, color='orange')
plt.ylim(ymin=0)  
plt.xlim(xmin=0)
plt.xlabel('Variance')
plt.ylabel('Mean')
plt.title('Efficient Frontier')
plt.show()