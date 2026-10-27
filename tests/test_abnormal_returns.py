import numpy as np
import pandas as pd
import pytest

from abnormal_returns import abnormal_returns

DATES = pd.bdate_range("2011-01-03", "2017-12-29")


def prices(market_returns, stock_returns):
    """Price table from log returns (the first day has no return)."""
    def px(r):
        return 100 * np.exp(np.concatenate([[0.0], np.cumsum(r[1:])]))
    return pd.DataFrame({"AAA": px(stock_returns), "IXIC": px(market_returns)}, index=DATES)


@pytest.fixture
def market_returns():
    return np.random.default_rng(0).normal(0.0004, 0.01, len(DATES))


def run(px):
    return abnormal_returns(px, companies=["AAA"], markets=["IXIC"])


def test_risk_adjusted_return_is_zero_when_the_stock_follows_the_market_model(market_returns):
    stock = 0.0003 + 1.2 * market_returns
    ar, params = run(prices(market_returns, stock))
    assert params.beta.iloc[0] == pytest.approx(1.2)
    assert params.alpha.iloc[0] == pytest.approx(0.0003)
    assert np.allclose(ar.RAR, 0, atol=1e-9)


def test_market_adjusted_return_is_zero_when_the_stock_equals_the_market(market_returns):
    ar, _ = run(prices(market_returns, market_returns))
    assert np.allclose(ar.MKAR, 0, atol=1e-12)


def test_mean_adjusted_return_is_zero_when_the_stock_return_is_constant(market_returns):
    ar, params = run(prices(market_returns, np.full(len(DATES), 0.0005)))
    assert np.allclose(ar.MAR, 0, atol=1e-12)
    assert params.est_mean.iloc[0] == pytest.approx(0.0005)


def test_mean_adjusted_return_is_return_minus_the_estimation_mean(market_returns):
    stock = np.random.default_rng(1).normal(0.0005, 0.015, len(DATES))
    ar, params = run(prices(market_returns, stock))
    assert np.allclose(ar.MAR, ar.ret - params.est_mean.iloc[0])


def test_analysis_days_start_in_2016_and_the_estimation_window_ends_in_2015(market_returns):
    stock = np.random.default_rng(2).normal(0, 0.012, len(DATES))
    ar, params = run(prices(market_returns, stock))
    assert ar.date.min() >= pd.Timestamp("2016-01-01")
    n_est = int((DATES[1:] <= "2015-12-31").sum())   # returns exist from the second day
    assert params.est_days.iloc[0] == n_est


@pytest.mark.parametrize("measure", ["MAR", "MKAR", "RAR"])
def test_significant_days_are_the_outer_quartiles(measure, market_returns):
    stock = np.random.default_rng(3).normal(0, 0.012, len(DATES))
    ar, _ = run(prices(market_returns, stock))
    q1, q3 = ar[measure].quantile(0.25), ar[measure].quantile(0.75)
    flagged = ar[f"{measure}_sig"]
    assert ((ar[measure] < q1) | (ar[measure] > q3)).equals(flagged)
    assert flagged.mean() == pytest.approx(0.5, abs=0.02)   # about half of all days, by construction
    assert (ar[f"{measure}_dir"] == np.sign(ar[measure])).all()
