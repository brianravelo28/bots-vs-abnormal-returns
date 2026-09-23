"""Stage 6: sentiment vs abnormal returns, by author class.

Method A (paper baseline): on 'significant' days (outside Q25-Q75 of the AR measure), Pearson r
    between mean daily sentiment and the direction of the abnormal return (+1 / -1).
Method B (full sample):    on every day, Pearson r between mean daily sentiment and the signed AR.
Method C (lead-lag):       Method B with sentiment from the PREVIOUS day (does sentiment predict?).

Restricted to 2016-2019 (tweet coverage), market = NASDAQ Composite, group-days with fewer than
MIN_TWEETS tweets dropped. Paper's reported tweet correlations: AMZN .36, AAPL .05, GOOG .32, MSFT .28.
"""
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr

DATA = Path(__file__).parent / "data"
MIN_TWEETS = 5
GROUPS = ["all", "bot", "organic"]
MEASURES = ["MAR", "MKAR", "RAR"]

ar = pd.read_parquet(DATA / "abnormal_returns.parquet")
ar = ar[(ar.market == "IXIC") & (ar.date <= "2019-12-31")]
sent = pd.read_parquet(DATA / "daily_sentiment.parquet")
sent = sent[sent.tweets >= MIN_TWEETS]

rows = []
for co in ["AMZN", "AAPL", "GOOG", "MSFT"]:
    a = ar[ar.company == co].sort_values("date")
    for g in GROUPS:
        s = sent[(sent.company == co) & (sent.author_class == g)].set_index("date").mean_sent
        d = a.set_index("date").join(s.rename("sent"), how="left")
        d["sent_prev"] = d.sent.shift(1)   # previous trading day's sentiment
        for m in MEASURES:
            for method, sub, x, y in [
                ("A_sig_days_direction", d[d[f"{m}_sig"]].dropna(subset=["sent"]), "sent", f"{m}_dir"),
                ("B_all_days_signed_AR", d.dropna(subset=["sent"]), "sent", m),
                ("C_prev_day_sentiment", d.dropna(subset=["sent_prev"]), "sent_prev", m),
            ]:
                if len(sub) > 10 and sub[x].std() > 0:
                    r, p = pearsonr(sub[x], sub[y])
                    rows.append(dict(company=co, group=g, measure=m, method=method, n=len(sub), r=r, p=p))

res = pd.DataFrame(rows)
res.to_csv(DATA / "correlations.csv", index=False)

pd.set_option("display.width", 220)
paper = {"AMZN": .36, "AAPL": .05, "GOOG": .32, "MSFT": .28}
for method in ["A_sig_days_direction", "B_all_days_signed_AR", "C_prev_day_sentiment"]:
    print(f"\n===== {method} =====   (r; * = p<0.05)")
    t = res[res.method == method].copy()
    t["cell"] = t.apply(lambda x: f"{x.r:+.2f}{'*' if x.p < .05 else ' '}(n={x.n})", axis=1)
    print(t.pivot_table(index=["company", "group"], columns="measure", values="cell", aggfunc="first").to_string())
print("\nPaper's reported tweet correlations:", paper)
