"""Download daily prices for the four study companies and the market index.

Estimation window 2011-2015, analysis window 2016 - Jun 2020 (as in the paper).
Cached to data/prices.parquet so the network is only hit once.
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

OUT = Path(__file__).parent / "data" / "prices.parquet"
TICKERS = {"AAPL": "AAPL", "AMZN": "AMZN", "GOOG": "GOOG", "MSFT": "MSFT", "IXIC": "^IXIC", "GSPC": "^GSPC"}

OUT.parent.mkdir(exist_ok=True)
raw = yf.download(list(TICKERS.values()), start="2011-01-01", end="2020-07-01",
                  auto_adjust=False, progress=False, group_by="column")
close = raw["Close"].rename(columns={v: k for k, v in TICKERS.items()})
close.index = pd.to_datetime(close.index).tz_localize(None)
close.index.name = "date"
close = close[list(TICKERS)].dropna(how="all")
close.to_parquet(OUT)
print(close.shape, close.index.min().date(), close.index.max().date())
print(close.isna().sum().to_dict())
print(close.tail(3))
