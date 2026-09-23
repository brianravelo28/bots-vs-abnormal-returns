"""Stage 4: transparent rule-based bot score per author.

No ground-truth labels exist, so this is a points system anyone can audit. Each rule adds
points; authors with fewer than MIN_TWEETS tweets are 'unclassified' (not enough behaviour
to judge), the rest are 'bot' (score >= BOT_AT) or 'organic' (= not flagged as automated; includes news publishers).

  Volume/cadence   max tweets in one day >= 50 ........ 2   (>= 20 ........ 1)
                   avg tweets per active day >= 10 .... 1
  Links            url_ratio >= 0.90 ................. 1
  Repetition       dup_ratio >= 0.50 ................. 2   (>= 0.20 ...... 1)
  Cashtag spam     avg $TICKERS per tweet >= 5 ....... 1
  Nobody engages   zero_eng_ratio >= 0.90 ............ 1
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
MIN_TWEETS = 5
BOT_AT = 4

f = pd.read_parquet(DATA / "author_features.parquet")

pts = pd.DataFrame(index=f.index)
pts["pts_volume"] = np.select([f.max_tweets_in_a_day >= 50, f.max_tweets_in_a_day >= 20], [2, 1], 0) \
    + (f.tweets_per_active_day >= 10).astype(int)
pts["pts_links"] = (f.url_ratio >= 0.90).astype(int)
pts["pts_repeat"] = np.select([f.dup_ratio >= 0.50, f.dup_ratio >= 0.20], [2, 1], 0)
pts["pts_cashtag"] = (f.cashtag_avg >= 5).astype(int)
pts["pts_noeng"] = (f.zero_eng_ratio >= 0.90).astype(int)
f = pd.concat([f, pts], axis=1)
f["bot_score"] = pts.sum(axis=1)
f["author_class"] = np.where(f.n_tweets < MIN_TWEETS, "unclassified",
                             np.where(f.bot_score >= BOT_AT, "bot", "organic"))
f.to_parquet(DATA / "author_scores.parquet")

pd.set_option("display.width", 200)
tw = f.groupby("author_class").agg(authors=("writer", "count"), tweets=("n_tweets", "sum"))
tw["pct_authors"] = (100 * tw.authors / tw.authors.sum()).round(1)
tw["pct_tweets"] = (100 * tw.tweets / tw.tweets.sum()).round(1)
print(tw.to_string())
print("\nScore distribution (authors with >= 5 tweets):")
c = f[f.n_tweets >= MIN_TWEETS]
print(c.groupby("bot_score").agg(authors=("writer", "count"), tweets=("n_tweets", "sum")).to_string())
cols = ["writer", "n_tweets", "bot_score", "author_class"]
print("\nTop 15 by volume:")
print(f.sort_values("n_tweets", ascending=False).head(15)[cols].to_string(index=False))
print("\nOrganic with highest volume (spot-check for missed bots):")
print(f[f.author_class == "organic"].sort_values("n_tweets", ascending=False).head(15)[cols + ["url_ratio", "dup_ratio", "max_tweets_in_a_day"]].round(2).to_string(index=False))
print("\nBots with lowest volume (spot-check for false positives):")
print(f[f.author_class == "bot"].sort_values("n_tweets").head(10)[cols + ["url_ratio", "dup_ratio", "max_tweets_in_a_day"]].round(2).to_string(index=False))

# Publisher note: the largest 'organic' accounts include news/alert publishers, not individuals.
top_org = f[f.author_class == "organic"].sort_values("n_tweets", ascending=False).head(50)
top_org[["writer", "n_tweets", "bot_score", "url_ratio", "max_tweets_in_a_day"]].to_csv(
    DATA / "top_organic_accounts.csv", index=False)
