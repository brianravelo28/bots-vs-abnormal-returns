"""Stage 3: per-author bot-likelihood features from the four-company tweets.

Features (the ones named in the paper's literature review: URL / hashtag ratios,
uniqueness of text, plus cadence and engagement):

  n_tweets, active_days, tweets_per_active_day, max_tweets_in_a_day
  url_ratio        share of tweets containing a link
  cashtag_avg      mean number of $TICKER mentions per tweet (spam tends to list many)
  hashtag_ratio    share of tweets containing a #hashtag
  dup_ratio        share of an author's tweets whose normalised text (URLs/mentions/digits
                   stripped) repeats an earlier tweet by the same author
  zero_eng_ratio   share of tweets with no likes, retweets or comments
  eng_per_tweet    mean likes + retweets + comments

Tweets are de-duplicated by tweet_id first (a tweet tagged with several companies is
one tweet for author-level behaviour).
"""
from pathlib import Path

import duckdb

DATA = Path(__file__).parent / "data"
SRC = (DATA / "tweets_4co.parquet").as_posix()
OUT = (DATA / "author_features.parquet").as_posix()

con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute(f"""
CREATE TEMP TABLE uniq AS
SELECT tweet_id, any_value(writer) writer, any_value(ts) ts, any_value(body) body,
       any_value(comment_num) comment_num, any_value(retweet_num) retweet_num,
       any_value(like_num) like_num
FROM '{SRC}' GROUP BY tweet_id
""")
con.execute("""
CREATE TEMP TABLE norm AS
SELECT *,
  ts::DATE AS day,
  trim(regexp_replace(regexp_replace(regexp_replace(lower(body),
        'https?://\\S+', '', 'g'), '@\\w+', '', 'g'), '[0-9]+', '', 'g')) AS norm_text
FROM uniq
""")
con.execute(f"""
COPY (
  WITH per_day AS (
    SELECT writer, day, count(*) n FROM norm GROUP BY 1, 2
  ), cad AS (
    SELECT writer, count(*) active_days, max(n) max_tweets_in_a_day FROM per_day GROUP BY 1
  ), dups AS (
    SELECT writer, sum(CASE WHEN rn > 1 THEN 1 ELSE 0 END) dup_tweets FROM (
      SELECT writer, row_number() OVER (PARTITION BY writer, norm_text ORDER BY ts) rn FROM norm
    ) GROUP BY 1
  ), base AS (
    SELECT writer,
      count(*) n_tweets,
      avg(CASE WHEN regexp_matches(body, 'https?://|www\\.') THEN 1 ELSE 0 END) url_ratio,
      avg(len(regexp_extract_all(body, '\\$[A-Za-z]{{1,5}}\\b'))) cashtag_avg,
      avg(CASE WHEN body LIKE '%#%' THEN 1 ELSE 0 END) hashtag_ratio,
      avg(CASE WHEN comment_num + retweet_num + like_num = 0 THEN 1 ELSE 0 END) zero_eng_ratio,
      avg(comment_num + retweet_num + like_num) eng_per_tweet
    FROM norm GROUP BY 1
  )
  SELECT b.writer, b.n_tweets, c.active_days, b.n_tweets * 1.0 / c.active_days tweets_per_active_day,
         c.max_tweets_in_a_day, b.url_ratio, b.cashtag_avg, b.hashtag_ratio,
         d.dup_tweets * 1.0 / b.n_tweets dup_ratio, b.zero_eng_ratio, b.eng_per_tweet
  FROM base b JOIN cad c USING (writer) JOIN dups d USING (writer)
) TO '{OUT}' (FORMAT parquet, COMPRESSION zstd)
""")

pd = con.sql(f"SELECT * FROM '{OUT}'").df()
import pandas as _pd
_pd.set_option("display.width", 200)
print("authors:", len(pd), "| unique tweets:", int(pd.n_tweets.sum()))
print(pd.drop(columns="writer").describe(percentiles=[.5, .9, .99]).round(3).T.to_string())
print("\nTop 10 authors by volume:")
print(pd.sort_values("n_tweets", ascending=False).head(10).round(3).to_string(index=False))
