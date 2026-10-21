"""Small statistics helpers shared by the correlation scripts."""
import numpy as np


def fisher_ci(r, n, z=1.96):
    """95% confidence interval for a Pearson correlation r from n days (Fisher z-transform).

    The interval assumes the days are independent. That is close enough here: the abnormal
    returns are essentially uncorrelated from one day to the next, so strong autocorrelation in
    the daily sentiment barely changes the width of the interval.
    """
    if n <= 3 or abs(r) >= 1:
        return np.nan, np.nan
    centre, half = np.arctanh(r), z / np.sqrt(n - 3)
    return float(np.tanh(centre - half)), float(np.tanh(centre + half))
