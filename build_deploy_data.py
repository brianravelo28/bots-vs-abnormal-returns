"""Stage 8: package the small files the hosted dashboard needs into deploy_data/.

The raw study data is 1.6 GB and stays local; the dashboard ships only aggregates
(a few hundred KB): daily abnormal returns, daily sentiment by author class, the correlation
tables, account-level bot scores for the largest accounts, and a few example bot tweets.
"""
import json
import re
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "deploy_data"
OUT.mkdir(exist_ok=True)

# --- daily abnormal returns (NASDAQ Composite as the market, as in the paper) ---
ar = pd.read_parquet(DATA / "abnormal_returns.parquet")
ar = ar[(ar.market == "IXIC") & (ar.date <= "2019-12-31")].drop(columns="market")
ar.to_parquet(OUT / "ar_daily.parquet", index=False)

# --- daily sentiment by class ---
ds = pd.read_parquet(DATA / "daily_sentiment.parquet")
ds.to_parquet(OUT / "daily_sentiment.parquet", index=False)

# --- correlation tables ---
pd.read_csv(DATA / "correlations.csv").to_csv(OUT / "correlations.csv", index=False)
pd.read_csv(DATA / "robustness.csv").to_csv(OUT / "robustness.csv", index=False)
pd.read_csv(DATA / "session_split.csv").to_csv(OUT / "session_split.csv", index=False)

# --- share of each group's company-tweet rows posted before the open vs during the session ---
ss = pd.read_parquet(DATA / "daily_sentiment_session.parquet")
ss = ss[ss.author_class.isin(["bot", "organic"])].groupby(["author_class", "session"]).tweets.sum().unstack()
session_share = ss.div(ss.sum(axis=1), axis=0).round(4).to_dict("index")

# --- accounts ---
sc = pd.read_parquet(DATA / "author_scores.parquet")
classes = sc.groupby("author_class").agg(authors=("writer", "count"), tweets=("n_tweets", "sum")).reset_index()
dist = (sc[sc.n_tweets >= 5].groupby("bot_score").agg(authors=("writer", "count"), tweets=("n_tweets", "sum"))
        .reset_index())
top_bots = sc[sc.author_class == "bot"].nlargest(60, "n_tweets")
top_org = sc[sc.author_class == "organic"].nlargest(40, "n_tweets")
# Published accounts get anonymous IDs ranked by tweet volume within their class. The
# handle-to-ID map stays in data/ (git-ignored) and never goes into deploy_data/.
top_bots = top_bots.assign(account_id=[f"bot-{i:02d}" for i in range(1, len(top_bots) + 1)])
top_org = top_org.assign(account_id=[f"organic-{i:02d}" for i in range(1, len(top_org) + 1)])
acct = pd.concat([top_bots, top_org])
acct[["writer", "account_id"]].to_csv(DATA / "account_id_map.csv", index=False)
keep = ["account_id", "author_class", "bot_score", "n_tweets", "active_days", "tweets_per_active_day",
        "max_tweets_in_a_day", "url_ratio", "cashtag_avg", "hashtag_ratio", "dup_ratio",
        "zero_eng_ratio", "eng_per_tweet"]
acct[keep].to_parquet(OUT / "top_accounts.parquet", index=False)
classes.to_parquet(OUT / "class_summary.parquet", index=False)
dist.to_parquet(OUT / "score_distribution.parquet", index=False)

# --- a couple of example tweets from the 15 biggest bot accounts (links and @mentions masked) ---
# The sample is picked by tweet-id hash, so a rebuild gives the same examples. The handle and the
# date are not published, only the anonymous ID and the text.
URL = re.compile(r"https?://\S+|www\.\S+")
HANDLE = re.compile(r"@\w+")
big15 = top_bots.head(15)  # nlargest above is sorted by tweet volume
id_of = dict(zip(big15.writer, big15.account_id))
con = duckdb.connect()
ex = con.execute(
    """SELECT writer, body FROM (
         SELECT writer, body, row_number() OVER (PARTITION BY writer ORDER BY hash(tweet_id)) rn
         FROM (SELECT DISTINCT ON (tweet_id) tweet_id, writer, body FROM read_parquet(?)
               WHERE writer = ANY(?))
       ) WHERE rn <= 2""",
    [(DATA / "tweets_4co.parquet").as_posix(), list(id_of)]).df()
ex["account_id"] = ex.writer.map(id_of)
ex["body"] = ex.body.map(lambda s: HANDLE.sub("@user", URL.sub("[link]", s)).strip()[:220])
ex = ex.sort_values("account_id")[["account_id", "body"]]
ex.to_parquet(OUT / "example_bot_tweets.parquet", index=False)

# --- headline numbers ---
tot = int(sc.n_tweets.sum())
summary = {
    "unique_tweets_scored": tot,
    "authors": int(len(sc)),
    "null_author_tweets": 25842,
    "bot_authors": int(classes.set_index("author_class").loc["bot", "authors"]),
    "bot_tweet_share": float(classes.set_index("author_class").loc["bot", "tweets"] / tot),
    "organic_tweet_share": float(classes.set_index("author_class").loc["organic", "tweets"] / tot),
    "unclassified_authors_share": float(classes.set_index("author_class").loc["unclassified", "authors"] / len(sc)),
    "analysis_days": int(ar.groupby("company").size().iloc[0]),
    "session_share": session_share,
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
for p in sorted(OUT.iterdir()):
    print(f"{p.name:32s}{p.stat().st_size / 1024:8.1f} KB")
print(summary)
