"""The price download is not tested (it needs the network), but the cleaning that follows it is."""
import numpy as np
import pandas as pd

from fetch_prices import TICKERS, tidy_prices


def raw_download():
    """Shaped like yfinance's output: (field, ticker) columns in alphabetical ticker order, a tz-aware index."""
    names = sorted(TICKERS.values())
    idx = pd.DatetimeIndex(["2018-06-01", "2018-06-02", "2018-06-04", "2018-06-05"], tz="UTC", name="Date")
    close = pd.DataFrame(np.arange(len(idx) * len(names), dtype=float).reshape(len(idx), len(names)),
                         index=idx, columns=names)
    close.iloc[1, :] = np.nan                 # a day when every series is missing
    close.iloc[2, names.index("^GSPC")] = np.nan   # a day when only one is missing
    volume = close * 1000
    return pd.concat({"Close": close, "Volume": volume}, axis=1)


def test_columns_are_renamed_and_ordered_as_in_tickers():
    out = tidy_prices(raw_download())
    assert list(out.columns) == list(TICKERS)                  # AAPL, AMZN, GOOG, MSFT, IXIC, GSPC
    assert "^IXIC" not in out.columns


def test_only_close_prices_are_kept():
    raw = raw_download()
    out = tidy_prices(raw)
    assert out.loc[pd.Timestamp("2018-06-01"), "IXIC"] == raw["Close"]["^IXIC"].iloc[0]
    assert out.loc[pd.Timestamp("2018-06-01"), "AAPL"] == raw["Close"]["AAPL"].iloc[0]
    assert (out.max() < 1000).all()                            # volume (x1000) did not leak in


def test_index_is_a_timezone_free_date_and_all_missing_days_are_dropped():
    out = tidy_prices(raw_download())
    assert out.index.name == "date" and out.index.tz is None
    assert out.index.tolist() == pd.to_datetime(["2018-06-01", "2018-06-04", "2018-06-05"]).tolist()


def test_a_partly_missing_day_is_kept_with_its_gap():
    out = tidy_prices(raw_download())
    assert out.loc["2018-06-04"].isna().tolist().count(True) == 1 and np.isnan(out.loc["2018-06-04", "GSPC"])
