"""The DuckDB extraction, run on tiny CSVs written in the same layout as the real Tweet.csv and Company_Tweet.csv."""
import pandas as pd
import pytest

from extract_tweets import extract

# 1514764800 is 2018-01-01 00:00:00 UTC
TWEET_CSV = '''tweet_id,writer,post_date,body,comment_num,retweet_num,like_num
1,alice,1514764800,"plain tweet",0,1,2
2,bob,1514768400,"has a comma, and a ""quote""",3,4,5
3,carol,1514772000,"first line
second line",0,0,0
4,,1514775600,"no author",0,0,0
5,dave,1514779200,"mentions TSLA only",0,0,0
6,erin,1514782800,"mentions two companies",0,0,0
7,frank,not_a_number,"malformed date",0,0,0
8,gina,1514786400,"has no company row",0,0,0
'''
COMPANY_CSV = '''tweet_id,ticker_symbol
1,AAPL
2,GOOGL
3,MSFT
4,AMZN
5,TSLA
6,AAPL
6,AMZN
7,AAPL
'''


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    (d / "Tweet.csv").write_bytes(TWEET_CSV.encode("utf-8"))   # bytes, so Windows keeps the \n
    (d / "Company_Tweet.csv").write_bytes(COMPANY_CSV.encode("utf-8"))
    out = tmp_path_factory.mktemp("out") / "nested" / "tweets_4co.parquet"
    extract(d, out)
    return pd.read_parquet(out).sort_values(["tweet_id", "company"]).reset_index(drop=True)


def test_output_columns(result):
    assert list(result.columns) == ["tweet_id", "company", "writer", "ts", "body", "comment_num", "retweet_num", "like_num"]


def test_only_the_four_companies_are_kept_and_googl_becomes_goog(result):
    assert set(result.company) == {"AAPL", "AMZN", "GOOG", "MSFT"}
    assert result[result.tweet_id == 2].company.tolist() == ["GOOG"]
    assert 5 not in result.tweet_id.values                      # TSLA only


def test_a_tweet_about_two_companies_has_one_row_per_company(result):
    assert result[result.tweet_id == 6].company.tolist() == ["AAPL", "AMZN"]


def test_tweets_without_a_company_row_or_with_a_malformed_row_are_dropped(result):
    assert 8 not in result.tweet_id.values                      # no Company_Tweet entry
    assert 7 not in result.tweet_id.values                      # the date is not a number
    assert result.tweet_id.tolist() == [1, 2, 3, 4, 6, 6]


def test_timestamps_are_utc_times_from_the_epoch_seconds(result):
    assert str(result.ts.dt.tz) == "UTC"
    assert result.ts.iloc[0] == pd.Timestamp("2018-01-01 00:00:00", tz="UTC")
    assert result.ts.iloc[1] == pd.Timestamp("2018-01-01 01:00:00", tz="UTC")


def test_quoted_text_survives_and_a_missing_author_stays_missing(result):
    body = result.set_index("tweet_id").body
    assert body[2] == 'has a comma, and a "quote"'
    assert body[3] == "first line\nsecond line"
    assert pd.isna(result[result.tweet_id == 4].writer.iloc[0])
    assert result[result.tweet_id == 2][["comment_num", "retweet_num", "like_num"]].iloc[0].tolist() == [3, 4, 5]
