# Bots vs Abnormal Returns

This project extends my graduate paper (MIS 581, Colorado State University Global, 2022), which tested whether Twitter sentiment lines up with abnormal stock returns for Apple, Amazon, Google and Microsoft. I couldn't separate real users from bots in that paper, and I named doing so as the next step. This project flags accounts that behave like bots and compares their sentiment, and that of the remaining accounts, against the same returns.

**Live dashboard:** https://bots-vs-abnormal-returns.onrender.com/ (Render free tier, so the first visit after idle takes about 30 seconds to wake). To run it locally: `python app.py`, then http://localhost:8060.

## Result

Who is tweeting, and when, matters. Bot-flagged accounts wrote about a third of the tweets, and their sentiment shows no detectable correlation with price moves. The other ("organic") accounts' sentiment does move with prices, mostly on the same day and mostly in tweets posted while the market is open.

An abnormal return is a stock's return beyond what the market and the stock's usual behavior predict. The table uses the risk-adjusted version (CAPM, the standard market model), measured against the NASDAQ Composite over 2016-2019, on the days when the abnormal return was in the top or bottom quartile.

| Company | All tweets | Bot accounts | Organic accounts |
|---|---|---|---|
| Apple | +0.08 | -0.01 | +0.31\* |
| Amazon | +0.25\* | +0.04 | +0.33\* |
| Google | +0.07 | 0.00 | +0.15\* |
| Microsoft | +0.13\* | +0.04 | +0.15\* |

Each cell is the Pearson correlation (r) between a day's average tweet sentiment and the direction of that day's abnormal return. A correlation of 0 means no link; 1 means sentiment and price moves always agree. \* = significant at p < 0.05.

- **Bots.** 498 accounts (0.4% of authors) wrote 36% of the 2.73M tweets that have a known author. Their correlation is between -0.03 and +0.06 for every company and abnormal-return model, none significant (0 of 12). The every-day test agrees (-0.02 to +0.06, 0 of 12). With about 480 days per company, only a correlation of roughly 0.13 or more would be reliably detected, so this means no detectable correlation, not proof of none.
- **Timing matters.** About 66% of bot-account tweets and 53% of organic-account tweets were posted before the market opened (evenings, weekends, holidays, pre-market). For organic accounts, tweets posted during the 9:30 am to 4 pm session correlate much more with that day's return (average r +0.31, significant for all four companies) than tweets posted before the open (+0.07, significant only for Apple and Amazon at +0.11 and +0.15). Bot-account tweets show nothing in either group. Most of the link looks like reaction to price moves as they happen. Tweets before the open can still be reacting to the previous day's move.
- **No lead.** Using the previous day's sentiment (tweets posted before that day's close), no correlation is significant: organic -0.03 to +0.05 and bot -0.02 to +0.05 (0 of 12 each).
- **Robust to the bot definition.** Across five threshold settings the bot correlation stays between -0.01 and +0.06, none significant (0 of 20), and organic stays between +0.11 and +0.37, all significant (20 of 20).
- **Correlation, not causation, and no ground-truth bot labels.** The bot score is a transparent heuristic, not a trained classifier, and "organic" means not flagged as automated: it includes news publishers as well as individuals.

**Relation to the paper.** The paper's figures are not shown beside these because they were computed on a different set of days, with a different up-or-down rule and a different sentiment tool, so they are not directly comparable. Running the paper's own day selection on this project's sentiment gives similar all-tweet values for Amazon (0.34 against the paper's 0.36) and Apple (0.07 against 0.05) and lower values for Google (0.11 against 0.32) and Microsoft (0.20 against 0.28). The details are in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Method

1. **Abnormal returns** (`abnormal_returns.py`): daily log returns; mean-adjusted, market-adjusted and risk-adjusted models, estimated on 2011-2015 and measured from 2016. A day is "significant" when its abnormal return falls outside the 25th-75th percentile of that measure (so about half of all days).
2. **Bot features** (`bot_features.py`): per author, burst size, tweets per active day, link ratio, cashtag spam, repeated text, engagement.
3. **Bot score** (`bot_score.py`): a points system anyone can audit. 4+ points = bot; under 5 tweets = unclassified; everyone else = organic.
4. **Sentiment** (`sentiment.py`, `market_calendar.py`): VADER (a word-list sentiment scorer built for social media) gives each tweet a score from -1 to +1, averaged per company per trading day. Each tweet counts toward the trading day whose close-to-close return window it falls in.
5. **Correlations** (`correlate.py`): three tests (significant days against the return's direction, every day against the return, previous day's sentiment against the return), plus the before-open versus during-session split. **Robustness** (`robustness.py`): the significant-days test at five bot thresholds.

## Reproduce

The raw data is 1.6 GB and is not in the repo. To rebuild it from the Kaggle stock-tweet dataset, in an activated virtual environment (Windows: `.venv\Scripts\activate`; macOS/Linux: `source .venv/bin/activate`), Python 3.11 or newer:

```
python -m venv .venv
python -m pip install -r requirements.txt
python fetch_prices.py        # Yahoo Finance prices -> data/prices.parquet
python extract_tweets.py      # stream the 631 MB Tweet.csv through DuckDB (folder via TWEETS_DIR, default data/raw)
python abnormal_returns.py
python bot_features.py
python bot_score.py
python sentiment.py           # VADER-scores 2.76M tweets; about 3.5 minutes
python correlate.py
python robustness.py
python build_deploy_data.py   # packages ~0.5 MB of aggregates into deploy_data/
python app.py
```

`extract_tweets.py` reads `Tweet.csv` and `Company_Tweet.csv` from the folder named by the `TWEETS_DIR` environment variable (default `data/raw`). The hosted dashboard only needs `deploy_data/` (`wsgi.py`, `render.yaml` and `requirements-render.txt` are set up for Render).

## Tests

About 100 tests check that the calculations are correct and that the published numbers are consistent. If a change breaks either, the tests catch it. A few examples:

- **Bot score:** each rule gives the points in the table, and an author with fewer than 5 tweets is "unclassified" whatever their score.
- **Calculations:** abnormal returns are zero when a stock follows its expected return; after-close, weekend and holiday tweets go to the next trading day; author features use US/Eastern days, not the machine's time zone; confidence intervals match known values.
- **Published data:** the committed files in `deploy_data/` are complete and consistent (108 correlations, every interval contains its estimate, account names are anonymous).
- **Dashboard:** every chart has its hint line, dropdowns with fewer than 7 options are not searchable, and every chart callback returns a figure.
- **Documentation:** the tables and statements in this README and in the Methodology match the output files, so rerunning the pipeline with different results fails the tests until the text is updated.

They use small hand-built fixtures and the committed `deploy_data/`, so they need no downloaded data. They run in about ten seconds, and automatically on every push through GitHub Actions on Python 3.11, 3.13 and 3.14. They do not cover the full pipeline on the real 1.6 GB of data, the VADER scoring, the price download, the comparison with the paper (it uses files outside this repository), or how the live page looks.

```
python -m pip install -r requirements-dev.txt
python -m pytest
```

## Notes

- Analysis window: 2016-2019 (1,006 trading days per company), set by where the tweet data ends (2019-12-31).
- 2,759,594 unique tweets were scored with VADER. 2,733,752 of them have a known author and feed the bot analysis; the other 25,842 have no author.
- Timing rule: each tweet counts toward the first trading day whose 4 pm Eastern close comes after it. Tweets after the close, on weekends and on market holidays go to the next trading day (`market_calendar.py`). Early-close days (1 pm) are treated as normal days. Tweets during the session can still react to the price as it moves, so same-day correlations are not evidence that sentiment leads returns.
- Correlations use the NASDAQ Composite only. The S&P 500 was tried for the abnormal-return step (similar betas and distributions) but not carried through to the correlations.
- The dashboard shows anonymous IDs for the largest accounts. The "bot" label is a heuristic flag, not a verified one.
- Each correlation carries a 95% confidence interval (`ci_low` and `ci_high` in the CSVs in `deploy_data/`), shown as whiskers on the dashboard charts.
- [docs/METHODOLOGY.md](docs/METHODOLOGY.md) has the full method, confidence intervals, the before-open split, the limitations and a detailed comparison with the paper.
- Scope: Apple, Amazon, Google and Microsoft.

## Data and credits

- **Tweets:** ["Tweets about the Top Companies from 2015 to 2020"](https://www.kaggle.com/datasets/omermetinn/tweets-about-the-top-companies-from-2015-to-2020) by Ömer Metin on Kaggle, released under CC0: Public Domain. It was created for Dogan, Metin, Tek, Yumusak and Oztoprak (2020), ["Speculator and influencer evaluation in stock market by using social media"](https://ieeexplore.ieee.org/document/9378170), IEEE International Conference on Big Data, and the tweets were collected from Twitter with a Selenium-based script ([TweetCollector](https://github.com/omer-metin/TweetCollector)). The raw tweets are not redistributed here. The dashboard shows 30 sample tweet texts with links, @mentions and account names removed. The CC0 dedication is the dataset author's; the tweet texts were written by other people and remain subject to Twitter's terms.
- **Prices:** Yahoo Finance, downloaded with `yfinance`, which is intended for research and educational use and is not affiliated with Yahoo; see Yahoo's terms of use for the rights to the data itself. The repository contains derived daily returns, not the downloaded prices.
- **Sentiment:** VADER (Hutto & Gilbert, 2014).
- **Code:** MIT license (see `LICENSE`).
- **Earlier paper:** my final research paper for MIS 581 (Colorado State University Global, 2022), unpublished, which this project extends.
