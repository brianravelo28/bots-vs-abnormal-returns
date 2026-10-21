"""Match tweets to the trading day whose return they could relate to.

A trading day's return runs from the previous close to this close (16:00 US/Eastern). A tweet is
therefore assigned to the first trading day whose close comes after it:

  - posted on a trading day before 16:00 Eastern       -> that day
  - posted on a trading day at or after 16:00 Eastern   -> the next trading day
  - posted on a weekend or market holiday               -> the next trading day

Trading days are the dates present in data/prices.parquet (the NASDAQ Composite's own calendar),
so holidays need no separate list. Early-close days (13:00) are treated as normal days.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
MARKET_TZ = "America/New_York"
OPEN_MINUTE = 9 * 60 + 30  # regular session opens 09:30 Eastern
CLOSE_HOUR = 16  # regular session ends 16:00 Eastern


def trading_days():
    """Sorted trading dates (naive, midnight) from the price file."""
    return pd.DatetimeIndex(pd.read_parquet(DATA / "prices.parquet").index).sort_values()


def assign_trading_day(ts):
    """Trading day (naive date) for each timestamp in a tz-aware Series; NaT past the calendar's end."""
    td = trading_days().values.astype("datetime64[ns]")
    et = ts.dt.tz_convert(MARKET_TZ)
    day = et.dt.normalize().dt.tz_localize(None).values.astype("datetime64[ns]")
    same_day = np.isin(day, td) & (et.dt.hour.values < CLOSE_HOUR)
    pos = np.searchsorted(td, day, side="right")  # first trading day strictly after the calendar day
    nxt = np.where(pos < len(td), td[np.minimum(pos, len(td) - 1)], np.datetime64("NaT"))
    return pd.Series(pd.to_datetime(np.where(same_day, day, nxt)), index=ts.index)


def assign_session(ts, trading_day):
    """'intraday' for tweets posted 09:30-16:00 Eastern on their own trading day; 'before_open' for
    everything else that counts toward that day (evenings, overnight, pre-market, weekends, holidays)."""
    et = ts.dt.tz_convert(MARKET_TZ)
    own_day = trading_day.values == et.dt.normalize().dt.tz_localize(None).values
    minutes = (et.dt.hour * 60 + et.dt.minute).values
    return pd.Series(np.where(own_day & (minutes >= OPEN_MINUTE), "intraday", "before_open"), index=ts.index)
