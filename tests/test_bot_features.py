import pandas as pd
import pytest

from bot_features import build_features


def tweets(rows):
    df = pd.DataFrame(rows, columns=["tweet_id", "company", "writer", "ts", "body", "comment_num", "retweet_num", "like_num"])
    df["ts"] = pd.to_datetime(df.ts, utc=True)
    return df


@pytest.fixture(scope="module")
def features(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("features")
    rows = []
    # spam: six near-identical link tweets with five cashtags, no engagement, all on one Eastern day
    for i in range(6):
        rows.append((1000 + i, "AAPL", "spam", f"2018-06-01 15:0{i}:00",
                     f"Buy $AAPL $MSFT $GOOG $AMZN $TSLA now http://x.co/{i}", 0, 0, 0))
    # human: three different tweets, one hashtag, some engagement, no links
    rows += [(2001, "AAPL", "human", "2018-06-01 14:00:00", "Great quarter for Apple", 0, 0, 3),
             (2002, "MSFT", "human", "2018-06-04 14:00:00", "Thinking about Microsoft earnings #msft", 0, 0, 0),
             (2003, "AMZN", "human", "2018-06-05 14:00:00", "Amazon looks expensive here", 0, 1, 0)]
    # night: 23:30 and 01:30 US/Eastern on consecutive days. In UTC (both 2018-06-02) or in
    # Pacific time (both 2018-06-01) these would count as one day.
    rows += [(3001, "AAPL", "night", "2018-06-02 03:30:00", "late one", 0, 0, 0),
             (3002, "AAPL", "night", "2018-06-02 05:30:00", "later one", 0, 0, 0)]
    # multi: two tweets that each mention two companies (four rows, two tweet ids)
    for tid, body in [(4001, "first note"), (4002, "second note")]:
        for company in ["AAPL", "MSFT"]:
            rows.append((tid, company, "multi", "2018-06-01 16:00:00", body, 0, 0, 0))
    src, out = tmp / "tweets.parquet", tmp / "features.parquet"
    tweets(rows).to_parquet(src)
    return build_features(src, out).set_index("writer")


def test_one_row_per_author(features):
    assert sorted(features.index) == ["human", "multi", "night", "spam"]


def test_link_cashtag_and_engagement_ratios(features):
    spam = features.loc["spam"]
    assert spam.url_ratio == 1.0
    assert spam.cashtag_avg == 5.0
    assert spam.zero_eng_ratio == 1.0
    human = features.loc["human"]
    assert human.url_ratio == 0.0 and human.cashtag_avg == 0.0
    assert human.hashtag_ratio == pytest.approx(1 / 3)
    assert human.zero_eng_ratio == pytest.approx(1 / 3)
    assert human.eng_per_tweet == pytest.approx(4 / 3)


def test_repeated_text_ignores_links_and_digits(features):
    # the six spam tweets differ only in the link, so five of them repeat the first
    assert features.loc["spam"].dup_ratio == pytest.approx(5 / 6)
    assert features.loc["human"].dup_ratio == 0.0


def test_cadence_features(features):
    spam = features.loc["spam"]
    assert spam.n_tweets == 6 and spam.active_days == 1
    assert spam.tweets_per_active_day == 6.0 and spam.max_tweets_in_a_day == 6
    human = features.loc["human"]
    assert human.active_days == 3 and human.max_tweets_in_a_day == 1


def test_a_day_is_a_us_eastern_calendar_day(features):
    night = features.loc["night"]
    assert night.active_days == 2 and night.max_tweets_in_a_day == 1


def test_a_tweet_tagged_with_several_companies_counts_once(features):
    assert features.loc["multi"].n_tweets == 2
