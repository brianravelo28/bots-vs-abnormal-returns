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

# --- accounts ---
sc = pd.read_parquet(DATA / "author_scores.parquet")
classes = sc.groupby("author_class").agg(authors=("writer", "count"), tweets=("n_tweets", "sum")).reset_index()
dist = (sc[sc.n_tweets >= 5].groupby("bot_score").agg(authors=("writer", "count"), tweets=("n_tweets", "sum"))
        .reset_index())
top_bots = sc[sc.author_class == "bot"].nlargest(60, "n_tweets")
top_org = sc[sc.author_class == "organic"].nlargest(40, "n_tweets")
acct = pd.concat([top_bots, top_org])
keep = ["writer", "author_class", "bot_score", "n_tweets", "active_days", "tweets_per_active_day",
        "max_tweets_in_a_day", "url_ratio", "cashtag_avg", "hashtag_ratio", "dup_ratio",
        "zero_eng_ratio", "eng_per_tweet"]
acct[keep].to_parquet(OUT / "top_accounts.parquet", index=False)
classes.to_parquet(OUT / "class_summary.parquet", index=False)
dist.to_parquet(OUT / "score_distribution.parquet", index=False)

# --- a couple of example tweets from the 15 biggest bot accounts (links stripped) ---
URL = re.compile(r"https?://\S+|www\.\S+")
writers = top_bots.nlargest(15, "n_tweets").writer.tolist()
con = duckdb.connect()
ex = con.execute(
    """SELECT writer, ts::DATE AS day, body FROM (
         SELECT writer, ts, body, row_number() OVER (PARTITION BY writer ORDER BY random()) rn
         FROM (SELECT DISTINCT ON (tweet_id) tweet_id, writer, ts, body FROM read_parquet(?)
               WHERE writer = ANY(?))
       ) WHERE rn <= 2 ORDER BY writer""",
    [(DATA / "tweets_4co.parquet").as_posix(), writers]).df()
ex["body"] = ex.body.map(lambda s: URL.sub("[link]", s).strip()[:220])
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
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
for p in sorted(OUT.iterdir()):
    print(f"{p.name:32s}{p.stat().st_size / 1024:8.1f} KB")
print(summary)
