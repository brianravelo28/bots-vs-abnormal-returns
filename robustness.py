"""Stage 7: does the bot-vs-organic result depend on where we draw the bot line?

Re-classifies authors at several (BOT_AT, MIN_TWEETS) settings using the stored point scores,
re-aggregates daily sentiment per class, and reruns Method A (paper baseline: significant days,
sentiment vs direction of RAR, NASDAQ, 2016-2019). Writes data/robustness.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

DATA = Path(__file__).parent / "data"
MIN_DAY_TWEETS = 5
SETTINGS = [(3, 5), (4, 5), (5, 5), (6, 5), (4, 20)]

tw = pd.read_parquet(DATA / "tweets_4co.parquet", columns=["tweet_id", "company", "writer", "ts"])
sent = pd.read_parquet(DATA / "tweet_sentiment.parquet")
auth = pd.read_parquet(DATA / "author_scores.parquet")[["writer", "n_tweets", "bot_score"]]
df = tw.merge(sent, on="tweet_id").merge(auth, on="writer")
df["date"] = df.ts.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
df = df[(df.date >= "2016-01-01") & (df.date <= "2019-12-31")].drop(columns="ts")

ar = pd.read_parquet(DATA / "abnormal_returns.parquet")
ar = ar[(ar.market == "IXIC") & (ar.date <= "2019-12-31")]

rows = []
for bot_at, min_t in SETTINGS:
    cls = np.where(df.n_tweets < min_t, "unclassified", np.where(df.bot_score >= bot_at, "bot", "organic"))
    d = df.assign(cls=cls)
    share = d.cls.value_counts(normalize=True)
    n_bot_auth = auth[(auth.n_tweets >= min_t) & (auth.bot_score >= bot_at)].shape[0]
    daily = d.groupby(["company", "date", "cls"]).compound.agg(["size", "mean"]).reset_index()
    daily = daily[daily["size"] >= MIN_DAY_TWEETS]
    for co in ["AAPL", "AMZN", "GOOG", "MSFT"]:
        a = ar[(ar.company == co) & ar.RAR_sig].set_index("date")
        for g in ["bot", "organic"]:
            s = daily[(daily.company == co) & (daily.cls == g)].set_index("date")["mean"]
            x = a.join(s.rename("sent"), how="inner")
            r, p = pearsonr(x.sent, x.RAR_dir)
            rows.append(dict(bot_at=bot_at, min_tweets=min_t, company=co, group=g, n=len(x), r=r, p=p,
                             bot_share_of_tweets=share.get("bot", 0), bot_authors=n_bot_auth))

res = pd.DataFrame(rows)
res.to_csv(DATA / "robustness.csv", index=False)

pd.set_option("display.width", 200)
print("Share of tweets / authors labelled bot at each setting:")
print(res.drop_duplicates(["bot_at", "min_tweets"])[["bot_at", "min_tweets", "bot_authors", "bot_share_of_tweets"]]
      .round(3).to_string(index=False))
for g in ["bot", "organic"]:
    t = res[res.group == g].copy()
    t["setting"] = "bot>=" + t.bot_at.astype(str) + ", min" + t.min_tweets.astype(str)
    t["cell"] = t.apply(lambda x: f"{x.r:+.2f}{'*' if x.p < .05 else ' '}", axis=1)
    print(f"\n=== {g.upper()} sentiment vs RAR direction, significant days (r; * = p<0.05) ===")
    print(t.pivot(index="company", columns="setting", values="cell").to_string())
