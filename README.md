# Bots vs Abnormal Returns

A revisit of my graduate research paper (MIS 581, Colorado State University Global, 2022) on whether Twitter sentiment lines up with abnormal stock returns for Apple, Amazon, Google and Microsoft. The paper's own conclusion was that it couldn't tell real users from bots, and it recommended adding that step. This project is that step.

**Live dashboard:** https://bots-vs-abnormal-returns.onrender.com/ (Render free tier, so the first visit after idle takes about 30 seconds to wake). To run it locally: `python app.py`, then http://localhost:8060. It is a Dash app in the same style as my Rossmann forecasting dashboard.

## Result

Bot accounts wrote about a third of the tweets and their sentiment carries no price signal. Organic accounts' sentiment does line up with price moves.

| Company | All tweets | Bot accounts | Organic accounts | Original paper |
|---|---|---|---|---|
| Apple | +0.12 | -0.03 | **+0.39** | 0.05 |
| Amazon | +0.26 | +0.02 | **+0.35** | 0.36 |
| Google | +0.11 | -0.03 | **+0.24** | 0.32 |
| Microsoft | +0.20 | +0.04 | **+0.25** | 0.28 |

Pearson r between daily sentiment and the direction of the risk-adjusted abnormal return on significant days (the paper's method), NASDAQ Composite, 2016-2019.

- **Bots:** 499 accounts (0.4% of authors) wrote 36% of the 2.73M tweets that have a known author. With the paper's method their correlation is between -0.06 and +0.05 for every company and abnormal-return model, none significant. The every-day test agrees (-0.03 to +0.05, none significant).
- **Robust to the bot definition.** Across five threshold settings (risk-adjusted return, significant days) the bot correlation stays between -0.04 and +0.06, none significant, and organic stays between +0.19 and +0.45, all significant.
- **Mostly reaction, not prediction.** Using the previous day's sentiment, organic correlations fall to between -0.01 and +0.14 (6 of 12 are still significant, but small) and bot correlations to between 0.00 and +0.07 (2 of 12 are significant, both Microsoft).
- **Not an exact replication.** I used VADER where the paper used R's SentimentAnalysis, so the all-tweets numbers differ (Apple 0.12 vs 0.05, Google 0.11 vs 0.32). The pattern is consistent; the values are not identical.
- **Correlation, not causation, and no ground-truth bot labels.** The score is a transparent heuristic, not a trained classifier.

## Method

1. **Abnormal returns** (`abnormal_returns.py`): daily log returns; mean-adjusted, market-adjusted and risk-adjusted (CAPM) models, estimated on 2011-2015 and measured from 2016. A day is 'significant' outside the 25th-75th percentile of a measure, as in the paper.
2. **Bot features** (`bot_features.py`): per author, burst size, tweets per active day, link ratio, cashtag spam, repeated text, engagement.
3. **Bot score** (`bot_score.py`): a points system anyone can audit. 4+ points = bot; under 5 tweets = unclassified; everyone else = organic (which includes news publishers).
4. **Sentiment** (`sentiment.py`): VADER compound score per tweet, averaged per company per day.
5. **Correlations** (`correlate.py`) and **threshold robustness** (`robustness.py`).

## Reproduce

The raw data is 1.6 GB and is not in the repo. To rebuild it from the Kaggle stock-tweet dataset:

```
python -m venv .venv && .venv\Scripts\pip install -r requirements.txt
python fetch_prices.py        # Yahoo Finance prices -> data/prices.parquet
python extract_tweets.py      # stream the 631 MB Tweet.csv through DuckDB
python abnormal_returns.py
python bot_features.py
python bot_score.py
python sentiment.py           # ~5 minutes for 2.7M tweets
python correlate.py
python robustness.py
python build_deploy_data.py   # packages ~0.5 MB of aggregates into deploy_data/
python app.py
```

`extract_tweets.py` points at the local dataset folder; edit `D` at the top for your path. The hosted dashboard only needs `deploy_data/` (`wsgi.py`, `render.yaml` and `requirements-render.txt` are set up for Render).

## Notes

- Tweets end 2019-12-31, so sentiment analysis covers 2016-2019 (1,006 trading days per company) even though returns are computed to June 2020.
- 2,759,594 unique tweets were scored with VADER. 2,733,752 of them have a known author and feed the bot analysis; the other 25,842 have no author.
- Timing is crude. Each tweet is assigned to its US/Eastern calendar day and matched to that day's return. Tweets after the 4 pm close are not moved to the next day, and weekend tweets have no return to match, so they are dropped. Same-day correlations therefore partly reflect people reacting to a move that already happened, which is why the previous-day test matters. An intraday or event-window version would be the next refinement.
- Correlations were run against the NASDAQ Composite only, as in the paper. The S&P 500 was tried for the abnormal-return step (similar betas and distributions) but not carried through to the correlations.
- See [DATA_QUIRKS.md](DATA_QUIRKS.md) for the odd things found in the data.
- Scope: the four companies from the paper. The tweet data also covers Tesla, and the headlines dataset (about 850 MB, not yet used) covers thousands of tickers.

## Data and credits

- Tweets: the Kaggle stock-tweet dataset (Dogan et al., 2020), as used in the original paper. Raw files are not redistributed here.
- Prices: Yahoo Finance, downloaded with `yfinance`.
- Sentiment: VADER (Hutto & Gilbert, 2014).
