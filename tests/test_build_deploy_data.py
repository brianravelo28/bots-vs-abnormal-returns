"""Packaging for the hosted dashboard: anonymity of the published accounts, masking of example tweets,
the headline numbers, and an end-to-end build on a small synthetic data folder."""
import json

import duckdb
import pandas as pd
import pytest

from build_deploy_data import (KEEP, anonymous_accounts, build, example_tweets, headline_summary, mask_tweet,
                               session_shares)


def test_session_shares_sum_to_one_per_group_and_ignore_other_groups():
    ss = pd.DataFrame({
        "author_class": ["bot", "bot", "organic", "organic", "all", "unclassified"],
        "session": ["before_open", "intraday", "before_open", "intraday", "intraday", "intraday"],
        "tweets": [60, 40, 1, 3, 999, 999],
    })
    out = session_shares(ss)
    assert set(out) == {"bot", "organic"}
    assert out["bot"] == {"before_open": 0.6, "intraday": 0.4}
    assert out["organic"] == {"before_open": 0.25, "intraday": 0.75}


def scores(n_bots=5, n_organic=5):
    rows = [(f"botname{i}", "bot", 100 - i) for i in range(n_bots)]
    rows += [(f"humanname{i}", "organic", 50 - i) for i in range(n_organic)]
    rows += [("quietname", "unclassified", 2)]
    return pd.DataFrame(rows, columns=["writer", "author_class", "n_tweets"])


def test_anonymous_ids_are_ranked_by_volume_within_each_class():
    bots, org = anonymous_accounts(scores(), n_bots=3, n_organic=2)
    assert bots.account_id.tolist() == ["bot-01", "bot-02", "bot-03"]
    assert bots.writer.tolist() == ["botname0", "botname1", "botname2"]       # biggest first
    assert org.account_id.tolist() == ["organic-01", "organic-02"]
    assert set(bots.author_class) == {"bot"} and set(org.author_class) == {"organic"}


def test_ids_do_not_depend_on_the_row_order_of_the_input():
    s = scores().assign(n_tweets=7)                                           # everyone ties
    a, _ = anonymous_accounts(s, n_bots=5, n_organic=5)
    b, _ = anonymous_accounts(s.sample(frac=1, random_state=3), n_bots=5, n_organic=5)
    assert a.set_index("account_id").writer.to_dict() == b.set_index("account_id").writer.to_dict()


def test_published_columns_do_not_include_the_handle():
    assert "writer" not in KEEP and KEEP[0] == "account_id"


@pytest.mark.parametrize("raw, expected", [
    ("buy now http://t.co/abc and https://x.io/y?z=1", "buy now [link] and [link]"),
    ("see www.example.com/page", "see [link]"),
    ("thanks @someone_else and @other", "thanks @user and @user"),
    ("  padded  ", "padded"),
    ("$AAPL #stocks stay", "$AAPL #stocks stay"),
])
def test_example_tweets_have_links_and_mentions_masked(raw, expected):
    assert mask_tweet(raw) == expected


def test_example_tweets_are_capped_in_length():
    assert len(mask_tweet("x" * 500)) == 220 and len(mask_tweet("x" * 500, limit=10)) == 10


@pytest.fixture
def tweets_file(tmp_path):
    rows = []
    for w, n in [("botname0", 5), ("botname1", 1), ("smallbot", 4)]:
        for t in range(n):
            tid = hash(w) % 10_000 * 100 + t
            body = f"post number {t} @victim http://spam.example/{t}"
            rows += [(tid, "AAPL", w, body), (tid, "MSFT", w, body)]   # the same tweet under two companies
    path = tmp_path / "tweets_4co.parquet"
    pd.DataFrame(rows, columns=["tweet_id", "company", "writer", "body"]).to_parquet(path)
    return path


def test_example_tweets_use_only_the_chosen_accounts_two_each_and_publish_no_handle(tweets_file):
    big = pd.DataFrame({"writer": ["botname0", "botname1"], "account_id": ["bot-01", "bot-02"]})
    ex = example_tweets(duckdb.connect(), tweets_file, big)
    assert list(ex.columns) == ["account_id", "body"]
    assert ex.account_id.value_counts().to_dict() == {"bot-01": 2, "bot-02": 1}     # capped at 2; bot-02 has 1 tweet
    assert not ex.duplicated().any()                                                # a two-company tweet is not repeated
    assert not ex.body.str.contains("botname|victim|spam.example", regex=True).any()
    assert ex.body.str.contains(r"@user \[link\]", regex=True).all()


def test_example_tweets_are_the_same_on_every_build(tweets_file):
    big = pd.DataFrame({"writer": ["botname0"], "account_id": ["bot-01"]})
    a = example_tweets(duckdb.connect(), tweets_file, big)
    b = example_tweets(duckdb.connect(), tweets_file, big)
    assert a.body.tolist() == b.body.tolist()


def summary_inputs():
    sc = pd.DataFrame({"writer": list("abcd"), "author_class": ["bot", "organic", "organic", "unclassified"],
                       "n_tweets": [30, 50, 10, 4]})
    classes = sc.groupby("author_class").agg(authors=("writer", "count"), tweets=("n_tweets", "sum")).reset_index()
    ar = pd.DataFrame({"company": ["AAPL"] * 3 + ["MSFT"] * 3})
    return sc, classes, ar


def test_headline_numbers():
    sc, classes, ar = summary_inputs()
    s = headline_summary(sc, classes, ar, {"bot": {}}, total_unique=96, null_authors=2)
    assert s["unique_tweets_scored"] == 94 and s["null_author_tweets"] == 2 and s["authors"] == 4
    assert s["bot_authors"] == 1 and s["bot_tweet_share"] == pytest.approx(30 / 94)
    assert s["organic_tweet_share"] == pytest.approx(60 / 94) and s["unclassified_authors_share"] == pytest.approx(0.25)
    assert s["analysis_days"] == 3 and s["session_share"] == {"bot": {}}


def test_a_tweet_missing_from_the_author_scores_stops_the_build():
    sc, classes, ar = summary_inputs()
    with pytest.raises(AssertionError, match="missing from the author scores"):
        headline_summary(sc, classes, ar, {}, total_unique=97, null_authors=2)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """A tiny but complete data folder, built into a deploy folder."""
    data = tmp_path_factory.mktemp("data")
    out = tmp_path_factory.mktemp("deploy") / "deploy_data"
    days = pd.bdate_range("2018-01-02", periods=3)
    ar = pd.DataFrame([(m, c, d, 0.1) for m in ["IXIC", "GSPC"] for c in ["AAPL", "MSFT"] for d in days],
                      columns=["market", "company", "date", "RAR"])
    ar = pd.concat([ar, pd.DataFrame({"market": ["IXIC"], "company": ["AAPL"], "date": [pd.Timestamp("2020-03-02")],
                                      "RAR": [9.9]})])                      # after 2019: must be left out
    ar.to_parquet(data / "abnormal_returns.parquet")
    daily = pd.DataFrame({"company": "AAPL", "date": days, "author_class": "bot", "tweets": 6, "mean_sent": 0.1})
    daily.to_parquet(data / "daily_sentiment.parquet")
    daily.assign(session=["intraday", "before_open", "intraday"]).to_parquet(data / "daily_sentiment_session.parquet")
    for name in ["correlations", "robustness", "session_split"]:
        pd.DataFrame({"x": [1, 2]}).to_csv(data / f"{name}.csv", index=False)

    writers = {"botname0": ("bot", 5), "botname1": ("bot", 3), "humanname0": ("organic", 4), "quietname": ("unclassified", 2)}
    sc = pd.DataFrame([(w, c, 5 if c == "bot" else 0, n) for w, (c, n) in writers.items()],
                      columns=["writer", "author_class", "bot_score", "n_tweets"])
    for col in KEEP:
        if col not in sc.columns and col != "account_id":
            sc[col] = 0.5
    sc.to_parquet(data / "author_scores.parquet")

    rows, tid = [], 0
    for w, (_, n) in writers.items():
        for _ in range(n):
            tid += 1
            rows.append((tid, "AAPL", w, f"says hello @victim http://spam.example/{tid}"))
    for _ in range(3):                                                         # tweets with no author
        tid += 1
        rows.append((tid, "AAPL", None, "anonymous tweet"))
    pd.DataFrame(rows, columns=["tweet_id", "company", "writer", "body"]).to_parquet(data / "tweets_4co.parquet")

    build(data, out)
    return data, out


def test_build_writes_every_published_file(built):
    _, out = built
    assert {p.name for p in out.iterdir()} == {
        "ar_daily.parquet", "daily_sentiment.parquet", "correlations.csv", "robustness.csv", "session_split.csv",
        "top_accounts.parquet", "class_summary.parquet", "score_distribution.parquet", "example_bot_tweets.parquet",
        "summary.json"}


def test_build_keeps_only_the_nasdaq_market_up_to_2019(built):
    _, out = built
    ar = pd.read_parquet(out / "ar_daily.parquet")
    assert "market" not in ar.columns and ar.date.max() < pd.Timestamp("2020-01-01") and len(ar) == 6


def test_no_published_file_contains_an_account_handle(built):
    _, out = built
    handles = [b"botname0", b"botname1", b"humanname0", b"quietname"]
    for p in out.iterdir():
        if p.suffix == ".parquet":
            blob = pd.read_parquet(p).to_csv().encode()
        else:
            blob = p.read_bytes()
        assert not any(h in blob for h in handles), p.name
        assert b"victim" not in blob and b"spam.example" not in blob, p.name


def test_the_handle_map_stays_in_the_data_folder(built):
    data, out = built
    m = pd.read_csv(data / "account_id_map.csv")
    assert set(m.account_id) == {"bot-01", "bot-02", "organic-01"} and "botname0" in m.writer.values
    assert not (out / "account_id_map.csv").exists()


def test_build_summary_counts(built):
    _, out = built
    s = json.loads((out / "summary.json").read_text())
    assert s["unique_tweets_scored"] == 14 and s["null_author_tweets"] == 3 and s["bot_authors"] == 2
    assert s["bot_tweet_share"] == pytest.approx(8 / 14) and s["analysis_days"] == 3
    assert s["session_share"] == {"bot": {"before_open": 0.3333, "intraday": 0.6667}}
