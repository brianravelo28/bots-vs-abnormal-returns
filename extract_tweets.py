"""Stage 1: pull the four study companies' tweets out of the 631 MB Tweet.csv.

Streams through DuckDB (no full load into RAM) and writes a local, git-ignored
parquet at data/tweets_4co.parquet. GOOG and GOOGL are both mapped to 'GOOG'
(Alphabet share classes). A tweet mentioning several companies appears once per company.
"""
import os
from pathlib import Path

import duckdb

# Folder containing Tweet.csv; set TWEETS_DIR to override.
D = Path(os.environ.get("TWEETS_DIR", Path(__file__).parent / "data" / "raw"))
OUT = Path(__file__).parent / "data" / "tweets_4co.parquet"
OUT.parent.mkdir(exist_ok=True)

con = duckdb.connect()
con.execute(f"""
COPY (
  SELECT t.tweet_id,
         CASE c.ticker_symbol WHEN 'GOOGL' THEN 'GOOG' ELSE c.ticker_symbol END AS company,
         t.writer,
         to_timestamp(t.post_date) AS ts,
         t.body,
         t.comment_num, t.retweet_num, t.like_num
  FROM read_csv('{(D / 'Tweet.csv').as_posix()}', header=true, quote='"', escape='"',
                columns={{'tweet_id':'BIGINT','writer':'VARCHAR','post_date':'BIGINT','body':'VARCHAR',
                         'comment_num':'BIGINT','retweet_num':'BIGINT','like_num':'BIGINT'}},
                ignore_errors=true) t
  JOIN read_csv('{(D / 'Company_Tweet.csv').as_posix()}', header=true,
                columns={{'tweet_id':'BIGINT','ticker_symbol':'VARCHAR'}}) c USING (tweet_id)
  WHERE c.ticker_symbol IN ('AAPL','AMZN','GOOG','GOOGL','MSFT')
) TO '{OUT.as_posix()}' (FORMAT parquet, COMPRESSION zstd)
""")

print(con.sql(f"""
SELECT company, count(*) tweets, count(DISTINCT writer) authors,
       min(ts)::DATE first_day, max(ts)::DATE last_day
FROM '{OUT.as_posix()}' GROUP BY 1 ORDER BY 1""").df())
print("null bodies:", con.sql(f"SELECT count(*) FROM '{OUT.as_posix()}' WHERE body IS NULL").fetchone()[0])
