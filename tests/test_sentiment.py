"""Sentiment scoring, the match of tweets to trading days, and the daily aggregates."""
import pandas as pd
import pytest

from sentiment import clean_text, daily_aggregates, score_tweets, tweet_rows

# Deliberately unlike the real calendar: Monday 2018-06-04 is treated as a closure, so a result that uses the
# default calendar (data/prices.parquet) instead of this one would differ.
CALENDAR = pd.to_datetime(["2018-06-01", "2018-06-05"])


@pytest.mark.parametrize("raw, expected", [
    ("great quarter http://t.co/abc123 now", "great quarter  now"),
    ("see https://example.com/a?b=1 and www.example.org/x done", "see  and  done"),
    ("@someone thanks @other_one", " thanks "),
    ("$AAPL looks strong #earnings", "$AAPL looks strong #earnings"),   # cashtags and hashtags are kept
])
def test_links_and_handles_are_stripped_before_scoring(raw, expected):
    assert clean_text(raw) == expected


class Recorder:
    """Stands in for VADER and records the text it is asked to score."""
    def __init__(self):
        self.seen = []

    def polarity_scores(self, text):
        self.seen.append(text)
        return {"compound": float(len(text))}


def test_each_row_is_scored_once_on_its_cleaned_text_and_keeps_its_id():
    uniq = pd.DataFrame({"tweet_id": [30, 10, 20], "body": ["a @x", "bb http://t.co/1", "ccc"]})
    rec = Recorder()
    out = score_tweets(uniq, analyzer=rec)
    assert rec.seen == ["a ", "bb ", "ccc"]
    assert out.tweet_id.tolist() == [30, 10, 20] and out.compound.tolist() == [2.0, 3.0, 3.0]


def test_vader_scores_positive_and_negative_text_with_the_right_sign():
    uniq = pd.DataFrame({"tweet_id": [1, 2, 3],
                         "body": ["I love this, great results!", "terrible awful horrible loss", "the company reports"]})
    c = score_tweets(uniq).compound
    assert c[0] > 0.05 and c[1] < -0.05 and abs(c[2]) <= 0.05


def test_stripping_changes_nothing_when_the_rest_of_the_text_is_the_same():
    plain = score_tweets(pd.DataFrame({"tweet_id": [1], "body": ["great results"]})).compound[0]
    noisy = score_tweets(pd.DataFrame({"tweet_id": [1], "body": ["great results @x http://t.co/z"]})).compound[0]
    assert plain == noisy


def utc(eastern):
    return pd.Timestamp(eastern, tz="America/New_York").tz_convert("UTC")


def test_tweets_are_matched_to_authors_and_trading_days():
    tw = pd.DataFrame({
        "tweet_id": [1, 2, 2, 3, 4],
        "company": ["AAPL", "AAPL", "MSFT", "AAPL", "AAPL"],
        "writer": ["a", "a", "a", "b", None],             # tweet 4 has no author
        "ts": [utc("2018-06-01 10:00"), utc("2018-06-01 17:00"), utc("2018-06-01 17:00"),
               utc("2018-06-02 12:00"), utc("2018-06-01 11:00")],
    })
    sent = pd.DataFrame({"tweet_id": [1, 2, 3, 4], "compound": [0.5, -0.5, 0.1, 0.9]})
    scores = pd.DataFrame({"writer": ["a", "b"], "author_class": ["bot", "organic"]})
    out = tweet_rows(tw, sent, scores, calendar=CALENDAR).sort_values(["tweet_id", "company"]).reset_index(drop=True)
    assert out.tweet_id.tolist() == [1, 2, 2, 3]                       # the author-less tweet is dropped
    assert out.company.tolist() == ["AAPL", "AAPL", "MSFT", "AAPL"]     # a two-company tweet keeps both rows
    assert out.date.tolist() == pd.to_datetime(["2018-06-01", "2018-06-05", "2018-06-05", "2018-06-05"]).tolist()
    assert out.author_class.tolist() == ["bot", "bot", "bot", "organic"]
    assert out.compound.tolist() == [0.5, -0.5, -0.5, 0.1]


def rows():
    day1, day2 = pd.Timestamp("2018-06-04"), pd.Timestamp("2018-06-05")
    return pd.DataFrame({
        "company": ["AAPL"] * 5,
        "date": [day1, day1, day1, day2, day2],
        "author_class": ["bot", "bot", "organic", "organic", "bot"],
        "session": ["intraday", "before_open", "intraday", "intraday", "intraday"],
        "compound": [0.2, 0.4, -0.6, 0.0, 0.8],
    })


def test_daily_aggregates_count_and_average_each_class_and_pool_all_authors():
    out = daily_aggregates(rows())
    assert list(out.columns) == ["company", "date", "author_class", "tweets", "mean_sent"]
    cell = lambda d, c: out[(out.date == d) & (out.author_class == c)].iloc[0]
    day1, day2 = pd.Timestamp("2018-06-04"), pd.Timestamp("2018-06-05")
    assert (cell(day1, "bot").tweets, cell(day1, "bot").mean_sent) == (2, pytest.approx(0.3))
    assert (cell(day1, "organic").tweets, cell(day1, "organic").mean_sent) == (1, pytest.approx(-0.6))
    assert (cell(day1, "all").tweets, cell(day1, "all").mean_sent) == (3, pytest.approx(0.0))
    assert (cell(day2, "all").tweets, cell(day2, "all").mean_sent) == (2, pytest.approx(0.4))
    assert len(out) == 6                                           # 2 days x (bot, organic, all)


def test_session_aggregates_keep_the_sessions_apart():
    out = daily_aggregates(rows(), extra=["session"])
    day1 = pd.Timestamp("2018-06-04")
    pick = lambda c, s: out[(out.date == day1) & (out.author_class == c) & (out.session == s)].iloc[0]
    assert pick("bot", "intraday").mean_sent == pytest.approx(0.2) and pick("bot", "before_open").mean_sent == pytest.approx(0.4)
    assert pick("all", "intraday").tweets == 2 and pick("all", "intraday").mean_sent == pytest.approx(-0.2)
    assert out.groupby("author_class").tweets.sum().to_dict() == {"all": 5, "bot": 3, "organic": 2}
