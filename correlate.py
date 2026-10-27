"""Sentiment vs abnormal returns, by author class.

Method A (significant days): on 'significant' days (outside Q25-Q75 of the AR measure), Pearson r
    between mean daily sentiment and the direction of the abnormal return (+1 / -1).
Method B (full sample):    on every day, Pearson r between mean daily sentiment and the signed AR.
Method C (lead-lag):       Method B with sentiment from the PREVIOUS day (does sentiment predict?).

Restricted to 2016-2019 (tweet coverage), market = NASDAQ Composite, group-days with fewer than
MIN_TWEETS tweets dropped. A day's sentiment covers the tweets between the previous close and
this close (see market_calendar.py), so Method C uses only information from before the return window.
"""
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr

from stats_utils import fisher_ci

DATA = Path(__file__).parent / "data"
MIN_TWEETS = 5
COMPANIES = ["AMZN", "AAPL", "GOOG", "MSFT"]
GROUPS = ["all", "bot", "organic"]
MEASURES = ["MAR", "MKAR", "RAR"]


def correlation_tests(ar, sent, companies=COMPANIES, groups=GROUPS, measures=MEASURES, min_tweets=MIN_TWEETS):
    """Methods A, B and C for every company x author group x abnormal-return measure.

    `ar` has one row per company and day (columns company, date, and for each measure M: M, M_sig, M_dir);
    `sent` has one row per company, day and author group (company, date, author_class, tweets, mean_sent).
    Returns one row per test with n, r, p and the 95% interval; a test with 10 days or fewer, or with
    no variation in its x values, is skipped.
    """
    sent = sent[sent.tweets >= min_tweets]
    rows = []
    for co in companies:
        a = ar[ar.company == co].sort_values("date")
        for g in groups:
            s = sent[(sent.company == co) & (sent.author_class == g)].set_index("date").mean_sent
            d = a.set_index("date").join(s.rename("sent"), how="left")
            d["sent_prev"] = d.sent.shift(1)   # previous trading day's sentiment
            for m in measures:
                for method, sub, x, y in [
                    ("A_sig_days_direction", d[d[f"{m}_sig"]].dropna(subset=["sent"]), "sent", f"{m}_dir"),
                    ("B_all_days_signed_AR", d.dropna(subset=["sent"]), "sent", m),
                    ("C_prev_day_sentiment", d.dropna(subset=["sent_prev"]), "sent_prev", m),
                ]:
                    if len(sub) > 10 and sub[x].std() > 0:
                        r, p = pearsonr(sub[x], sub[y])
                        lo, hi = fisher_ci(r, len(sub))
                        rows.append(dict(company=co, group=g, measure=m, method=method, n=len(sub), r=r, p=p,
                                         ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows)


def session_split(ar, sess, companies=COMPANIES, min_tweets=MIN_TWEETS):
    """The same two tests as A and B (risk-adjusted return only), but using only tweets posted before
    the open (evenings, overnight, weekends, holidays, pre-market) or only tweets posted during the
    09:30-16:00 session. `sess` is like `sent` with an extra `session` column."""
    sess = sess[sess.tweets >= min_tweets]
    srows = []
    for co in companies:
        a = ar[ar.company == co].set_index("date")
        for g in ["bot", "organic"]:
            for session in ["before_open", "intraday"]:
                s = sess[(sess.company == co) & (sess.author_class == g) & (sess.session == session)]
                x = a.join(s.set_index("date").mean_sent.rename("sent"), how="inner")
                for method, sub, y in [("A_sig_days_direction", x[x.RAR_sig], "RAR_dir"),
                                       ("B_all_days_signed_AR", x, "RAR")]:
                    r, p = pearsonr(sub.sent, sub[y])
                    lo, hi = fisher_ci(r, len(sub))
                    srows.append(dict(company=co, group=g, session=session, method=method, n=len(sub), r=r, p=p,
                                      ci_low=lo, ci_high=hi))
    return pd.DataFrame(srows)


if __name__ == "__main__":
    ar = pd.read_parquet(DATA / "abnormal_returns.parquet")
    ar = ar[(ar.market == "IXIC") & (ar.date <= "2019-12-31")]

    res = correlation_tests(ar, pd.read_parquet(DATA / "daily_sentiment.parquet"))
    res.to_csv(DATA / "correlations.csv", index=False)
    session_split(ar, pd.read_parquet(DATA / "daily_sentiment_session.parquet")).to_csv(DATA / "session_split.csv", index=False)

    pd.set_option("display.width", 220)
    for method in ["A_sig_days_direction", "B_all_days_signed_AR", "C_prev_day_sentiment"]:
        print(f"\n===== {method} =====   (r; * = p<0.05)")
        t = res[res.method == method].copy()
        t["cell"] = t.apply(lambda x: f"{x.r:+.2f}{'*' if x.p < .05 else ' '}(n={x.n})", axis=1)
        print(t.pivot_table(index=["company", "group"], columns="measure", values="cell", aggfunc="first").to_string())
