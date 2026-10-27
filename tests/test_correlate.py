"""The three correlation tests and the session split, on small datasets where the answer is known by construction."""
import numpy as np
import pandas as pd
import pytest

from correlate import correlation_tests, session_split

DAYS = pd.bdate_range("2018-01-02", periods=40)
COLUMNS = ["company", "group", "measure", "method", "n", "r", "p", "ci_low", "ci_high"]


def noise(seed, scale=1.0):
    return np.random.default_rng(seed).normal(0, scale, len(DAYS))


def ar_frame(returns, company="AAA"):
    """Abnormal-return table for one measure (RAR): significant days are the outer quartiles."""
    ar = pd.DataFrame({"company": company, "date": DAYS, "RAR": returns})
    q1, q3 = ar.RAR.quantile(0.25), ar.RAR.quantile(0.75)
    ar["RAR_sig"] = (ar.RAR < q1) | (ar.RAR > q3)
    ar["RAR_dir"] = np.sign(ar.RAR).astype(int)
    return ar


def sent_frame(values, group="all", company="AAA", tweets=10, session=None):
    df = pd.DataFrame({"company": company, "date": DAYS, "author_class": group, "tweets": tweets, "mean_sent": values})
    if session:
        df["session"] = session
    return df


def run(ar, sent, **kw):
    return correlation_tests(ar, sent, companies=["AAA"], groups=kw.pop("groups", ["all"]), measures=["RAR"], **kw)


def cell(res, method, group="all"):
    return res[(res.method == method) & (res.group == group)].iloc[0]


def test_output_has_the_expected_columns_and_three_tests():
    r = noise(0)
    res = run(ar_frame(r), sent_frame(0.5 * r + noise(1)))
    assert list(res.columns) == COLUMNS
    assert set(res.method) == {"A_sig_days_direction", "B_all_days_signed_AR", "C_prev_day_sentiment"}
    assert ((res.ci_low <= res.r) & (res.r <= res.ci_high)).all()


def test_every_day_test_pairs_each_days_sentiment_with_that_days_return():
    r = noise(0)
    s = 0.5 * r + noise(1)
    row = cell(run(ar_frame(r), sent_frame(s)), "B_all_days_signed_AR")
    assert row.n == 40 and row.r == pytest.approx(np.corrcoef(s, r)[0, 1])


def test_previous_day_test_uses_yesterdays_sentiment():
    # sentiment that LEADS the return by one day: yesterday's sentiment equals today's return (plus a little noise)
    r = noise(0)
    s = np.roll(r, -1) + noise(2, 0.05)
    res = run(ar_frame(r), sent_frame(s))
    assert cell(res, "C_prev_day_sentiment").r > 0.95      # found by the previous-day test
    assert abs(cell(res, "B_all_days_signed_AR").r) < 0.5   # invisible to the same-day test


def test_same_day_sentiment_does_not_register_as_a_lead():
    # sentiment that REACTS to the return on the same day
    r = noise(0)
    res = run(ar_frame(r), sent_frame(r + noise(2, 0.05)))
    assert cell(res, "B_all_days_signed_AR").r > 0.95
    assert abs(cell(res, "C_prev_day_sentiment").r) < 0.5
    assert cell(res, "C_prev_day_sentiment").n == 39        # the first day has no previous day


def test_significant_days_test_uses_only_those_days_and_the_direction():
    r = noise(0)
    ar = ar_frame(r)
    s = np.sign(r) + noise(3, 0.3)                          # sentiment follows the direction of the move
    row = cell(run(ar, sent_frame(s)), "A_sig_days_direction")
    sig = ar.RAR_sig.to_numpy()
    assert row.n == sig.sum() and 0 < row.n < 40
    assert row.r == pytest.approx(np.corrcoef(s[sig], np.sign(r)[sig])[0, 1])


def test_days_with_too_few_tweets_are_dropped():
    r = noise(0)
    s = 0.5 * r + noise(1)
    s_outlier = s.copy(); s_outlier[5] = 50.0                # an extreme value on a day with only 4 tweets
    sent = sent_frame(s_outlier)
    sent.loc[5, "tweets"] = 4
    row = cell(run(ar_frame(r), sent), "B_all_days_signed_AR")
    keep = np.arange(40) != 5
    assert row.n == 39 and row.r == pytest.approx(np.corrcoef(s[keep], r[keep])[0, 1])
    assert cell(run(ar_frame(r), sent, min_tweets=4), "B_all_days_signed_AR").n == 40   # the threshold is respected


def test_author_groups_are_kept_apart():
    r = noise(0)
    sent = pd.concat([sent_frame(r + noise(2, 0.05), group="organic"), sent_frame(noise(4), group="bot")])
    res = run(ar_frame(r), sent, groups=["bot", "organic"])
    assert cell(res, "B_all_days_signed_AR", "organic").r > 0.95
    assert abs(cell(res, "B_all_days_signed_AR", "bot").r) < 0.5


def test_tiny_samples_and_flat_sentiment_are_skipped():
    r = noise(0)
    assert run(ar_frame(r).head(10), sent_frame(r).head(10)).empty                      # 10 days or fewer
    flat = run(ar_frame(r), sent_frame(np.full(40, 0.1)))      # 0.1 repeated has a rounding-noise spread of ~1e-17 on some subsets
    assert flat.empty                                                                    # no variation, no correlation


def test_session_split_keeps_the_two_sessions_apart():
    r = noise(0)
    ar = ar_frame(r)
    parts = []
    for group in ("bot", "organic"):
        # during the session the sentiment tracks the return; before the open it is unrelated noise
        parts.append(sent_frame((r + noise(2, 0.05)) if group == "organic" else noise(6), group, session="intraday"))
        parts.append(sent_frame(noise(7 if group == "organic" else 8), group, session="before_open"))
    res = session_split(ar, pd.concat(parts), companies=["AAA"])
    assert len(res) == 8                                                                 # 2 groups x 2 sessions x 2 tests
    assert list(res.columns) == ["company", "group", "session", "method", "n", "r", "p", "ci_low", "ci_high"]
    pick = lambda g, s, m: res[(res.group == g) & (res.session == s) & (res.method == m)].iloc[0]
    assert pick("organic", "intraday", "B_all_days_signed_AR").r > 0.95
    assert pick("organic", "intraday", "B_all_days_signed_AR").r == pytest.approx(np.corrcoef(r + noise(2, 0.05), r)[0, 1])
    assert abs(pick("organic", "before_open", "B_all_days_signed_AR").r) < 0.5
    assert abs(pick("bot", "intraday", "B_all_days_signed_AR").r) < 0.5
    # the significant-days test uses only the outer-quartile days and the direction of the move
    a = pick("organic", "intraday", "A_sig_days_direction")
    sig = ar.RAR_sig.to_numpy()
    assert a.n == sig.sum() < 40
    assert a.r == pytest.approx(np.corrcoef((r + noise(2, 0.05))[sig], np.sign(r)[sig])[0, 1])
