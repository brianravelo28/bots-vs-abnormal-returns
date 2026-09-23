"""Stage 5: VADER sentiment for every unique tweet, then daily aggregates.

Tweets are scored once per tweet_id (URLs and @handles stripped first, $TICKER kept),
then joined back to company rows. Days are US/Eastern calendar dates. Output:

  data/tweet_sentiment.parquet   tweet_id -> compound score (local, git-ignored)
  data/daily_sentiment.parquet   company x date x author_class: tweets, mean_sent
"""
import re
from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

DATA = Path(__file__).parent / "data"
URL = re.compile(r"https?://\S+|www\.\S+")
HANDLE = re.compile(r"@\w+")

tw = pd.read_parquet(DATA / "tweets_4co.parquet")
uniq = tw.drop_duplicates("tweet_id")[["tweet_id", "body"]]
print("unique tweets to score:", len(uniq), flush=True)

an = SentimentIntensityAnalyzer()
score = an.polarity_scores
clean = uniq.body.map(lambda s: HANDLE.sub("", URL.sub("", s)))
comp = [score(t)["compound"] for t in clean]
sent = pd.DataFrame({"tweet_id": uniq.tweet_id.values, "compound": comp})
sent.to_parquet(DATA / "tweet_sentiment.parquet")

scores = pd.read_parquet(DATA / "author_scores.parquet")[["writer", "author_class"]]
df = tw[["tweet_id", "company", "writer", "ts"]].merge(sent, on="tweet_id").merge(scores, on="writer")
df["date"] = df.ts.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
df = df[(df.date >= "2016-01-01") & (df.date <= "2019-12-31")]

grp = ["company", "date", "author_class"]
daily = df.groupby(grp).agg(tweets=("compound", "size"), mean_sent=("compound", "mean")).reset_index()
alld = df.groupby(["company", "date"]).agg(tweets=("compound", "size"), mean_sent=("compound", "mean")).reset_index()
alld["author_class"] = "all"
daily = pd.concat([daily, alld], ignore_index=True)
daily.to_parquet(DATA / "daily_sentiment.parquet")

pd.set_option("display.width", 200)
print("\nCompound score distribution, by author class (2016-2019):")
print(df.groupby("author_class").compound.describe().round(3).to_string())
share = (df.compound > 0.05).groupby(df.author_class).mean().round(3)
neg = (df.compound < -0.05).groupby(df.author_class).mean().round(3)
print("\nShare positive (>0.05):", share.to_dict(), "| negative (<-0.05):", neg.to_dict())
print("\nMean daily sentiment by company/class:")
print(daily.pivot_table(index="company", columns="author_class", values="mean_sent", aggfunc="mean").round(3).to_string())
