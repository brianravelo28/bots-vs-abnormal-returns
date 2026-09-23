"""Stage 2: daily abnormal returns (MAR, MKAR, RAR) for the four study companies.

Follows the paper: estimation window 2011-2015, analysis window Jan 2016 - Jun 2020,
log returns (as in AR Calculations.xlsx). Market-based models are run against both the
NASDAQ Composite (the paper's text) and the S&P 500 (the paper's workbook).

  MAR   AR = R - mean(R_est)
  MKAR  AR = R - R_market
  RAR   AR = R - (alpha + beta * R_market)      alpha/beta from OLS on the estimation window

'Significant' days follow the paper: AR below Q25 or above Q75 of that measure's
analysis-window distribution.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
COMPANIES = ["AAPL", "AMZN", "GOOG", "MSFT"]
MARKETS = ["IXIC", "GSPC"]

px = pd.read_parquet(DATA / "prices.parquet")
ret = np.log(px / px.shift(1)).dropna(how="all")
est = ret.loc[:"2015-12-31"]
ana = ret.loc["2016-01-01":]

rows, params = [], []
for co in COMPANIES:
    for mk in MARKETS:
        e = est[[co, mk]].dropna()
        beta, alpha = np.polyfit(e[mk], e[co], 1)
        r2 = np.corrcoef(e[mk], e[co])[0, 1] ** 2
        mean_est = e[co].mean()
        params.append(dict(company=co, market=mk, alpha=alpha, beta=beta, r2=r2,
                           est_mean=mean_est, est_days=len(e)))
        a = ana[[co, mk]].dropna()
        df = pd.DataFrame({
            "date": a.index, "company": co, "market": mk,
            "ret": a[co].values, "mkt_ret": a[mk].values,
            "MAR": (a[co] - mean_est).values,
            "MKAR": (a[co] - a[mk]).values,
            "RAR": (a[co] - (alpha + beta * a[mk])).values,
        })
        rows.append(df)

ar = pd.concat(rows, ignore_index=True)
for m in ["MAR", "MKAR", "RAR"]:
    q1 = ar.groupby(["company", "market"])[m].transform(lambda s: s.quantile(0.25))
    q3 = ar.groupby(["company", "market"])[m].transform(lambda s: s.quantile(0.75))
    ar[f"{m}_sig"] = (ar[m] < q1) | (ar[m] > q3)
    ar[f"{m}_dir"] = np.sign(ar[m]).astype(int)

ar.to_parquet(DATA / "abnormal_returns.parquet")
pd.DataFrame(params).to_csv(DATA / "ar_parameters.csv", index=False)

pd.set_option("display.width", 200)
print("=== Estimation-window parameters (2011-2015) ===")
print(pd.DataFrame(params).round(5).to_string(index=False))
print("\n=== Analysis window ===")
print(ar.groupby(["company", "market"]).agg(days=("date", "count"), first=("date", "min"),
                                             last=("date", "max")).to_string())
print("\n=== Abnormal return distribution (daily, log) ===")
print(ar.groupby(["company", "market"])[["MAR", "MKAR", "RAR"]].agg(["mean", "std"]).round(5).to_string())
print("\n=== Days flagged 'significant' (outside Q25-Q75) ===")
print(ar.groupby(["company", "market"])[["MAR_sig", "MKAR_sig", "RAR_sig"]].sum().to_string())
print("\n=== Overlap: days flagged significant by all 3 models (IXIC) ===")
x = ar[ar.market == "IXIC"]
print(x.assign(all3=x.MAR_sig & x.MKAR_sig & x.RAR_sig).groupby("company")["all3"].sum().to_string())
print("\n=== Largest RAR days (IXIC) ===")
print(x.reindex(x.RAR.abs().sort_values(ascending=False).index).groupby("company").head(3)
      .sort_values(["company", "date"])[["company", "date", "ret", "mkt_ret", "RAR"]].round(4).to_string(index=False))
