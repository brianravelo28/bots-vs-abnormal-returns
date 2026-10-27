"""Package the small files the hosted dashboard needs into deploy_data/.

The raw study data is 1.6 GB and stays local; the dashboard ships only aggregates
(a few hundred KB): daily abnormal returns, daily sentiment by author class, the correlation
tables, account-level bot scores for the largest accounts, and a few example bot tweets.
"""
import json
import re
from pathlib import Path

import duckdb
import pandas as pd

from bot_score import top_by_volume

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "deploy_data"

URL = re.compile(r"https?://\S+|www\.\S+")
HANDLE = re.compile(r"@\w+")
KEEP = ["account_id", "author_class", "bot_score", "n_tweets", "active_days", "tweets_per_active_day",
        "max_tweets_in_a_day", "url_ratio", "cashtag_avg", "hashtag_ratio", "dup_ratio",
        "zero_eng_ratio", "eng_per_tweet"]


def session_shares(session_daily):
    """Share of each group's company-tweet rows posted before the open vs during the session."""
    ss = session_daily[session_daily.author_class.isin(["bot", "organic"])]
    ss = ss.groupby(["author_class", "session"]).tweets.sum().unstack()
    return ss.div(ss.sum(axis=1), axis=0).round(4).to_dict("index")


def anonymous_accounts(scores, n_bots=60, n_organic=40):
    """The largest bot and organic accounts by tweet volume, with anonymous IDs ranked by volume within
    their class (bot-01, organic-01, ...). Returns (top_bots, top_organic); both keep the `writer` column,
    which the caller must not publish."""
    top_bots = top_by_volume(scores[scores.author_class == "bot"], n_bots)
    top_org = top_by_volume(scores[scores.author_class == "organic"], n_organic)
    top_bots = top_bots.assign(account_id=[f"bot-{i:02d}" for i in range(1, len(top_bots) + 1)])
    top_org = top_org.assign(account_id=[f"organic-{i:02d}" for i in range(1, len(top_org) + 1)])
    return top_bots, top_org


def mask_tweet(text, limit=220):
    """Replace links and @mentions so a published example tweet identifies nobody; cap its length."""
    return HANDLE.sub("@user", URL.sub("[link]", text)).strip()[:limit]


def example_tweets(con, tweets_path, big_accounts, per_account=2):
    """Up to `per_account` masked example tweets for each account in `big_accounts` (writer, account_id).
    The sample is picked by tweet-id hash, so a rebuild gives the same examples. Only the anonymous ID and
    the text are returned, not the handle or the date."""
    id_of = dict(zip(big_accounts.writer, big_accounts.account_id))
    ex = con.execute(
        """SELECT writer, body FROM (
             SELECT writer, body, row_number() OVER (PARTITION BY writer ORDER BY hash(tweet_id)) rn
             FROM (SELECT DISTINCT ON (tweet_id) tweet_id, writer, body FROM read_parquet(?)
                   WHERE writer = ANY(?))
           ) WHERE rn <= ?""",
        [Path(tweets_path).as_posix(), list(id_of), per_account]).df()
    ex["account_id"] = ex.writer.map(id_of)
    ex["body"] = ex.body.map(mask_tweet)
    return ex.sort_values("account_id")[["account_id", "body"]]


def headline_summary(scores, classes, ar, session_share, total_unique, null_authors):
    """The numbers the dashboard and README quote. Fails if any tweet is missing from the author scores."""
    tot = int(scores.n_tweets.sum())
    assert total_unique == tot + null_authors, "some tweets are missing from the author scores"
    by_class = classes.set_index("author_class")
    return {
        "unique_tweets_scored": tot,
        "authors": int(len(scores)),
        "null_author_tweets": int(null_authors),
        "bot_authors": int(by_class.loc["bot", "authors"]),
        "bot_tweet_share": float(by_class.loc["bot", "tweets"] / tot),
        "organic_tweet_share": float(by_class.loc["organic", "tweets"] / tot),
        "unclassified_authors_share": float(by_class.loc["unclassified", "authors"] / len(scores)),
        "analysis_days": int(ar.groupby("company").size().iloc[0]),
        "session_share": session_share,
    }


def build(data=DATA, out=OUT):
    out.mkdir(exist_ok=True)

    # --- daily abnormal returns (NASDAQ Composite as the market, as in the paper) ---
    ar = pd.read_parquet(data / "abnormal_returns.parquet")
    ar = ar[(ar.market == "IXIC") & (ar.date <= "2019-12-31")].drop(columns="market")
    ar.to_parquet(out / "ar_daily.parquet", index=False)

    # --- daily sentiment by class ---
    ds = pd.read_parquet(data / "daily_sentiment.parquet")
    ds.to_parquet(out / "daily_sentiment.parquet", index=False)

    # --- correlation tables ---
    pd.read_csv(data / "correlations.csv").to_csv(out / "correlations.csv", index=False)
    pd.read_csv(data / "robustness.csv").to_csv(out / "robustness.csv", index=False)
    pd.read_csv(data / "session_split.csv").to_csv(out / "session_split.csv", index=False)

    session_share = session_shares(pd.read_parquet(data / "daily_sentiment_session.parquet"))

    # --- accounts ---
    sc = pd.read_parquet(data / "author_scores.parquet")
    classes = sc.groupby("author_class").agg(authors=("writer", "count"), tweets=("n_tweets", "sum")).reset_index()
    dist = (sc[sc.n_tweets >= 5].groupby("bot_score").agg(authors=("writer", "count"), tweets=("n_tweets", "sum"))
            .reset_index())
    # Published accounts get anonymous IDs. The handle-to-ID map stays in data/ (git-ignored) and never
    # goes into deploy_data/.
    top_bots, top_org = anonymous_accounts(sc)
    acct = pd.concat([top_bots, top_org])
    acct[["writer", "account_id"]].to_csv(data / "account_id_map.csv", index=False)
    acct[KEEP].to_parquet(out / "top_accounts.parquet", index=False)
    classes.to_parquet(out / "class_summary.parquet", index=False)
    dist.to_parquet(out / "score_distribution.parquet", index=False)

    # --- a couple of example tweets from the 15 biggest bot accounts (links and @mentions masked) ---
    con = duckdb.connect()
    tweets_path = data / "tweets_4co.parquet"
    example_tweets(con, tweets_path, top_bots.head(15)).to_parquet(out / "example_bot_tweets.parquet", index=False)

    # --- headline numbers ---
    # Tweets with no author cannot be scored per author; count them from the data instead of typing the number.
    total_unique, null_authors = con.execute(
        "SELECT count(DISTINCT tweet_id), count(DISTINCT tweet_id) FILTER (WHERE writer IS NULL) FROM read_parquet(?)",
        [tweets_path.as_posix()]).fetchone()
    summary = headline_summary(sc, classes, ar, session_share, total_unique, null_authors)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    for p in sorted(out.iterdir()):
        print(f"{p.name:32s}{p.stat().st_size / 1024:8.1f} KB")
    print(summary)


if __name__ == "__main__":
    build()
