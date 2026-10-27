import pandas as pd
import pytest

from market_calendar import assign_session, assign_trading_day

# A small trading calendar: Fri 2018-06-01 and Mon 2018-06-04; 2018-12-25 (a holiday) is missing;
# the last 2015 trading day and the first 2016 one; the last 2019 day and the first 2020 one.
CALENDAR = pd.to_datetime(["2015-12-31", "2016-01-04", "2018-06-01", "2018-06-04",
                           "2018-12-24", "2018-12-26", "2019-12-31", "2020-01-02"])


def utc(eastern_wall_time):
    """A one-element tz-aware UTC Series for a US/Eastern wall-clock time."""
    return pd.Series(pd.to_datetime([eastern_wall_time]).tz_localize("America/New_York").tz_convert("UTC"))


@pytest.mark.parametrize("eastern, expected", [
    ("2018-06-01 10:00", "2018-06-01"),   # Friday, during the session
    ("2018-06-01 15:59", "2018-06-01"),   # one minute before the close
    ("2018-06-01 16:00", "2018-06-04"),   # at the close: next trading day
    ("2018-06-01 20:30", "2018-06-04"),   # Friday evening -> Monday
    ("2018-06-02 12:00", "2018-06-04"),   # Saturday -> Monday
    ("2018-06-03 23:59", "2018-06-04"),   # Sunday -> Monday
    ("2018-06-04 08:00", "2018-06-04"),   # Monday pre-market stays Monday
    ("2018-12-25 10:00", "2018-12-26"),   # holiday -> next trading day
    ("2018-12-24 17:00", "2018-12-26"),   # evening before the holiday skips it
    ("2015-12-31 17:00", "2016-01-04"),   # across the year boundary
    ("2016-01-01 12:00", "2016-01-04"),   # New Year's Day
    ("2019-12-31 17:00", "2020-01-02"),
])
def test_assign_trading_day(eastern, expected):
    assert assign_trading_day(utc(eastern), calendar=CALENDAR).iloc[0] == pd.Timestamp(expected)


def test_tweet_after_the_calendar_ends_has_no_trading_day():
    assert pd.isna(assign_trading_day(utc("2020-01-02 17:00"), calendar=CALENDAR).iloc[0])


def test_uses_us_eastern_time_not_utc():
    # 2018-06-02 03:30 UTC is still Friday evening in New York, so it belongs to Monday, not Saturday's next day
    ts = pd.Series(pd.to_datetime(["2018-06-02 03:30"]).tz_localize("UTC"))
    assert assign_trading_day(ts, calendar=CALENDAR).iloc[0] == pd.Timestamp("2018-06-04")


@pytest.mark.parametrize("eastern, expected", [
    ("2018-06-01 09:29", "before_open"),
    ("2018-06-01 09:30", "intraday"),
    ("2018-06-01 15:59", "intraday"),
    ("2018-06-01 16:00", "before_open"),   # after the close it counts toward Monday, before that day's open
    ("2018-06-02 12:00", "before_open"),   # weekend
    ("2018-06-04 08:00", "before_open"),   # Monday pre-market
    ("2018-12-25 12:00", "before_open"),   # holiday
])
def test_assign_session(eastern, expected):
    ts = utc(eastern)
    day = assign_trading_day(ts, calendar=CALENDAR)
    assert assign_session(ts, day).iloc[0] == expected
