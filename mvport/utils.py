"""
Utility functions.
"""

from __future__ import unicode_literals

import numpy as np


def get_random_weights(total):
    """Create an array of weights adding to 1"""
    w = np.random.rand(total)
    w /= sum(w)
    return w
