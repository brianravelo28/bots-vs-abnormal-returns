import numpy as np
import pytest

from stats_utils import fisher_ci


def test_known_interval():
    # n = 103 gives a standard error of exactly 0.1 on the Fisher z scale, so the half-width is 0.196
    lo, hi = fisher_ci(0.5, 103)
    centre = np.arctanh(0.5)
    assert lo == pytest.approx(np.tanh(centre - 0.196))
    assert hi == pytest.approx(np.tanh(centre + 0.196))


def test_interval_contains_r_and_is_symmetric_at_zero():
    lo, hi = fisher_ci(0.0, 480)
    assert lo == pytest.approx(-hi)
    lo, hi = fisher_ci(0.31, 480)
    assert lo < 0.31 < hi


def test_more_days_give_a_narrower_interval():
    narrow = np.subtract(*fisher_ci(0.2, 1000)[::-1])
    wide = np.subtract(*fisher_ci(0.2, 100)[::-1])
    assert narrow < wide


def test_about_480_days_gives_roughly_plus_minus_0_09():
    lo, hi = fisher_ci(0.0, 480)
    assert hi == pytest.approx(0.09, abs=0.005)


@pytest.mark.parametrize("r, n", [(0.3, 3), (0.3, 2), (1.0, 50), (-1.0, 50)])
def test_undefined_cases_return_nan(r, n):
    lo, hi = fisher_ci(r, n)
    assert np.isnan(lo) and np.isnan(hi)
