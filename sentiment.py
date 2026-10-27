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


def clean_text(text):
    """Strip links and @handles before scoring (a $TICKER cashtag is kept)."""
    return HANDLE.sub("", URL.sub("", text))


def score_tweets(uniq, analyzer=None):
    """VADER compound score for each row of `uniq` (columns tweet_id, body), after cleaning the text."""
    score = (analyzer or SentimentIntensityAnalyzer()).polarity_scores
    comp = [score(t)["compound"] for t in uniq.body.map(clean_text)]
    return pd.DataFrame({"tweet_id": uniq.tweet_id.values, "compound": comp})


def tweet_rows(tw, sent, scores, calendar=None):
    """One row per company-tweet with a known author: its compound score, its author's class, and the
    trading day it counts toward. `tw` has tweet_id, company, writer, ts; `scores` has writer, author_class;
    `calendar` is a list of trading dates (default: the dates in data/prices.parquet)."""
    df = tw[["tweet_id", "company", "writer", "ts"]].merge(sent, on="tweet_id").merge(scores, on="writer")
    df["date"] = assign_trading_day(df.ts, calendar)
    return df


def daily_aggregates(df, extra=()):
    """Tweet count and mean compound score per company x day x author class, plus an 'all' class that
    pools every author. `extra` adds more grouping columns, for example ["session"]."""
    extra = list(extra)
    agg = dict(tweets=("compound", "size"), mean_sent=("compound", "mean"))
    by_class = df.groupby(["company", "date", "author_class"] + extra).agg(**agg).reset_index()
    pooled = df.groupby(["company", "date"] + extra).agg(**agg).reset_index()
    pooled["author_class"] = "all"
    return pd.concat([by_class, pooled], ignore_index=True)


def main():
    tw = pd.read_parquet(DATA / "tweets_4co.parquet")
    uniq = tw.drop_duplicates("tweet_id")[["tweet_id", "body"]]
    print("unique tweets to score:", len(uniq), flush=True)

    sent = score_tweets(uniq)
    sent.to_parquet(DATA / "tweet_sentiment.parquet")

    scores = pd.read_parquet(DATA / "author_scores.parquet")[["writer", "author_class"]]
    df = tweet_rows(tw, sent, scores)
    in_window = (df.date >= "2016-01-01") & (df.date <= "2019-12-31")
    same_day = df.date == df.ts.dt.tz_convert(MARKET_TZ).dt.normalize().dt.tz_localize(None)
    print("\nCompany-tweet rows by how they were matched to a trading day:")
    print(f"  total with a known author .......... {len(df):>9,}")
    print(f"  matched to their own day (intraday)  {int((same_day & in_window).sum()):>9,}")
    print(f"  moved to a later trading day ....... {int((~same_day & in_window).sum()):>9,}  (after close, weekend or holiday)")
    print(f"  outside the 2016-2019 window ....... {int((~in_window).sum()):>9,}  (dropped)")
    df = df[in_window].copy()
    df["session"] = assign_session(df.ts, df.date)

    daily = daily_aggregates(df)
    daily.to_parquet(DATA / "daily_sentiment.parquet")

    # Same aggregates split by when the tweet was posted: before the open vs during the session.
    daily_aggregates(df, extra=["session"]).to_parquet(DATA / "daily_sentiment_session.parquet")
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


if __name__ == "__main__":
    main()
