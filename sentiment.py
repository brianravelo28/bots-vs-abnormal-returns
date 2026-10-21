"""VADER sentiment for every unique tweet, then daily aggregates.

Tweets are scored once per tweet_id (URLs and @handles stripped first, $TICKER kept),
then joined back to company rows. Each tweet is matched to a trading day by
market_calendar.assign_trading_day: tweets after the 16:00 Eastern close, on weekends or on
market holidays go to the next trading day. Output:

  data/tweet_sentiment.parquet   tweet_id -> compound score (local, git-ignored)
  data/daily_sentiment.parquet   company x date x author_class: tweets, mean_sent
  data/daily_sentiment_session.parquet   the same, split by session (before_open / intraday)
"""
import re
from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from market_calendar import MARKET_TZ, assign_session, assign_trading_day

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
df["date"] = assign_trading_day(df.ts)
in_window = (df.date >= "2016-01-01") & (df.date <= "2019-12-31")
same_day = df.date == df.ts.dt.tz_convert(MARKET_TZ).dt.normalize().dt.tz_localize(None)
print("\nCompany-tweet rows by how they were matched to a trading day:")
print(f"  total with a known author .......... {len(df):>9,}")
print(f"  matched to their own day (intraday)  {int((same_day & in_window).sum()):>9,}")
print(f"  moved to a later trading day ....... {int((~same_day & in_window).sum()):>9,}  (after close, weekend or holiday)")
print(f"  outside the 2016-2019 window ....... {int((~in_window).sum()):>9,}  (dropped)")
df = df[in_window].copy()
df["session"] = assign_session(df.ts, df.date)

grp = ["company", "date", "author_class"]
daily = df.groupby(grp).agg(tweets=("compound", "size"), mean_sent=("compound", "mean")).reset_index()
alld = df.groupby(["company", "date"]).agg(tweets=("compound", "size"), mean_sent=("compound", "mean")).reset_index()
alld["author_class"] = "all"
daily = pd.concat([daily, alld], ignore_index=True)
daily.to_parquet(DATA / "daily_sentiment.parquet")

# Same aggregates split by when the tweet was posted: before the open vs during the session.
sgrp = grp + ["session"]
by_session = df.groupby(sgrp).agg(tweets=("compound", "size"), mean_sent=("compound", "mean")).reset_index()
alls = df.groupby(["company", "date", "session"]).agg(tweets=("compound", "size"), mean_sent=("compound", "mean")).reset_index()
alls["author_class"] = "all"
by_session = pd.concat([by_session, alls], ignore_index=True)
by_session.to_parquet(DATA / "daily_sentiment_session.parquet")
print("\nShare of company-tweet rows posted before the open vs during the session:")
print(df.groupby("author_class").session.value_counts(normalize=True).round(3).unstack().to_string())

pd.set_option("display.width", 200)
print("\nCompound score distribution, by author class (2016-2019):")
print(df.groupby("author_class").compound.describe().round(3).to_string())
share = (df.compound > 0.05).groupby(df.author_class).mean().round(3)
neg = (df.compound < -0.05).groupby(df.author_class).mean().round(3)
print("\nShare positive (>0.05):", share.to_dict(), "| negative (<-0.05):", neg.to_dict())
print("\nMean daily sentiment by company/class:")
print(daily.pivot_table(index="company", columns="author_class", values="mean_sent", aggfunc="mean").round(3).to_string())
