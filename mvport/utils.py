"""
Utility functions.
"""

from __future__ import unicode_literals

import numpy as np


def get_random_weights(total):
    """Create an array of weights adding to 1"""
    good_sample = False
    while not good_sample:
        w = np.random.uniform(-1,1,total)
        w /= np.sum(w)
        if np.sum(np.abs(w)) / total < 2:
            good_sample = True
    return w
