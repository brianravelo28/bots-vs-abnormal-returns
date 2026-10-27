"""Checks on the committed deploy_data/ files that the dashboard and the README numbers come from."""
import json
import re
from pathlib import Path

import pandas as pd
import pytest

D = Path(__file__).parent.parent / "deploy_data"


@pytest.fixture(scope="module")
def correlations():
    return pd.read_csv(D / "correlations.csv")


def interval_contains_estimate(df):
    return bool(((df.ci_low <= df.r) & (df.r <= df.ci_high)).all())


def test_correlation_table_is_complete(correlations):
    c = correlations
    assert len(c) == 108                                    # 3 measures x 3 tests x 4 companies x 3 groups
    assert set(c.group) == {"all", "bot", "organic"}
    assert set(c.measure) == {"MAR", "MKAR", "RAR"}
    assert set(c.method) == {"A_sig_days_direction", "B_all_days_signed_AR", "C_prev_day_sentiment"}
    assert c.notna().all().all()
    assert not c.duplicated(["company", "group", "measure", "method"]).any()


def test_correlations_are_valid(correlations):
    c = correlations
    assert c.r.between(-1, 1).all() and c.p.between(0, 1).all()
    assert c.n.between(400, 1006).all()
    assert interval_contains_estimate(c)


def test_robustness_and_session_tables():
    rob = pd.read_csv(D / "robustness.csv")
    assert len(rob) == 40 and rob[["bot_at", "min_tweets"]].drop_duplicates().shape[0] == 5
    assert rob.notna().all().all() and interval_contains_estimate(rob)
    ses = pd.read_csv(D / "session_split.csv")
    assert len(ses) == 32 and set(ses.session) == {"before_open", "intraday"}
    assert ses.notna().all().all() and interval_contains_estimate(ses)


def test_summary_matches_the_class_table():
    s = json.loads((D / "summary.json").read_text())
    cs = pd.read_parquet(D / "class_summary.parquet").set_index("author_class")
    total = cs.tweets.sum()
    assert s["unique_tweets_scored"] == total
    assert s["authors"] == cs.authors.sum()
    assert s["bot_authors"] == cs.loc["bot", "authors"]
    assert s["bot_tweet_share"] == pytest.approx(cs.loc["bot", "tweets"] / total)
    assert s["organic_tweet_share"] == pytest.approx(cs.loc["organic", "tweets"] / total)
    for group in ("bot", "organic"):
        assert sum(s["session_share"][group].values()) == pytest.approx(1.0, abs=1e-3)


def test_published_accounts_are_anonymous():
    accounts = pd.read_parquet(D / "top_accounts.parquet")
    assert "writer" not in accounts.columns
    assert accounts.account_id.str.fullmatch(r"(bot|organic)-\d{2}").all()
    assert accounts.account_id.is_unique
    assert (accounts.account_id.str.split("-").str[0] == accounts.author_class).all()
    assert accounts.author_class.value_counts().to_dict() == {"bot": 60, "organic": 40}


def test_example_tweets_carry_no_handles_dates_or_links():
    ex = pd.read_parquet(D / "example_bot_tweets.parquet")
    assert list(ex.columns) == ["account_id", "body"]
    assert len(ex) == 30 and ex.account_id.nunique() == 15
    assert ex.account_id.str.fullmatch(r"bot-\d{2}").all()
    assert not ex.body.str.contains("http", case=False).any()
    assert not ex.body.str.replace("@user", "", regex=False).str.contains("@").any()


def test_abnormal_returns_cover_the_analysis_window():
    ar = pd.read_parquet(D / "ar_daily.parquet")
    assert ar.date.min() == pd.Timestamp("2016-01-04") and ar.date.max() == pd.Timestamp("2019-12-31")
    per_company = ar.groupby("company")
    assert (per_company.size() == 1006).all()
    assert not ar.duplicated(["company", "date"]).any()
    share = per_company.RAR_sig.mean()
    assert share.between(0.45, 0.52).all()                  # about half of all days by construction


def test_daily_sentiment_is_in_range():
    ds = pd.read_parquet(D / "daily_sentiment.parquet")
    assert set(ds.author_class) == {"all", "bot", "organic", "unclassified"}
    assert ds.mean_sent.between(-1, 1).all() and (ds.tweets >= 1).all()
