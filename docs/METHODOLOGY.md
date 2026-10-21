# Methodology

How the data is prepared, how the numbers are calculated, what the results can and cannot show, and how they compare with my 2022 paper (MIS 581, Colorado State University Global). The README has the headline results; this document has the detail behind them.

## 1. Data

**Tweets.** The Kaggle stock-tweet dataset (Dogan et al., 2020) holds 3,717,964 tweets about six tickers. This project uses four companies: Apple, Amazon, Microsoft and Google. Google's two share classes (GOOG and GOOGL) are merged under GOOG. That gives 3,239,577 company-tweet rows, because a tweet that mentions several companies appears once per company, and 2,759,594 unique tweets. Of these, 25,842 have no author and are left out of the bot analysis, which leaves 2,733,752 tweets from 120,969 authors. The tweets end on 2019-12-31, so the analysis window is 2016-01-04 to 2019-12-31, which is 1,006 trading days per company. How the tweets were collected is not documented.

**Prices.** Daily closing prices from Yahoo Finance (via `yfinance`), January 2011 to June 2020, for the four stocks, the NASDAQ Composite and the S&P 500. The `Close` column is adjusted for splits but not for dividends, so returns are price returns. GOOG is Yahoo's Class C series.

## 2. Abnormal returns

An abnormal return is a stock's return beyond what would be expected. Returns are daily log returns, and three models define "expected":

| Model | Abnormal return = |
|---|---|
| Mean-adjusted (MAR) | return minus the stock's average return in 2011-2015 |
| Market-adjusted (MKAR) | return minus the NASDAQ Composite's return that day |
| Risk-adjusted (RAR) | return minus (alpha + beta x market return), where alpha and beta come from a regression on 2011-2015 (CAPM, the standard market model) |

The estimation window is 2011-01-03 to 2015-12-31 (1,257 days per stock). Betas against the NASDAQ Composite are 0.93 (Apple), 1.10 (Amazon), 0.94 (Google) and 0.89 (Microsoft), with R-squared from 0.34 to 0.43.

A **significant day** is a day whose abnormal return is below the 25th or above the 75th percentile of that measure. The percentiles come from the January 2016 to June 2020 distribution (1,131 days), so 566 of those days are flagged for every company and model by construction. Within 2016-2019 the flagged share is 48-49%. The flag marks the larger moves; it is not a statistical test that a day was unusual.

All correlations use the NASDAQ Composite. The S&P 500 was also computed for this step and gives similar betas and distributions, but it was not carried through to the correlations.

## 3. Bot score

Each author earns points from their behavior across the four-company tweets (2015-2019), with days counted as US/Eastern calendar days:

| Behavior | Points |
|---|---|
| 20+ tweets in one day / 50+ tweets in one day | 1 / 2 |
| 10+ tweets per active day on average | 1 |
| 90%+ of tweets contain a link | 1 |
| Repeated text (after removing links, mentions and digits) in 20%+ / 50%+ of tweets | 1 / 2 |
| 5+ cashtags ($TICKER) per tweet on average | 1 |
| 90%+ of tweets get no likes, replies or retweets | 1 |

Four points or more is **bot**. Authors with fewer than 5 tweets are **unclassified**. Everyone else is **organic**. The thresholds are rules I chose after looking at the feature distributions and the highest-volume accounts.

| Class | Authors | Tweets | Share of tweets |
|---|---|---|---|
| Bot | 498 (0.4%) | 984,247 | 36.0% |
| Organic | 23,366 (19.3%) | 1,608,637 | 58.8% |
| Unclassified | 97,105 (80.3%) | 140,868 | 5.2% |

Activity is concentrated: the median author wrote one tweet, and the ten highest-volume authors wrote 19.5% of all tweets.

**What the labels mean.** "Bot" describes high-volume, link-heavy, repetitive accounts that spam cashtags and draw little engagement, mostly promotional or news-aggregator feeds rather than accounts imitating people. "Organic" means an author with 5+ tweets who is not flagged. It includes news publishers and is not the same as "human investors". The ten largest organic accounts write 5.1% of organic tweets (the largest 50 write 15.1%), so a few publishers do not dominate the group. There are no ground-truth bot labels, so the score is a heuristic and has not been checked against known bots.

## 4. Sentiment and timing

**Scoring.** VADER (a word-list sentiment scorer built for social-media text) gives each tweet a compound score from -1 (most negative) to +1 (most positive), after links and @handles are removed. Tweets with no scored words get 0. VADER is not tuned for financial vocabulary.

**Daily value.** For each company, trading day and author class, sentiment is the mean score of that day's tweets. A cell with fewer than 5 tweets is dropped.

**Which day a tweet belongs to.** A day's abnormal return runs from the previous close to this close (4 pm Eastern). A tweet therefore counts toward the first trading day whose close comes after it:

- posted on a trading day before 4 pm Eastern: that day;
- posted on a trading day at or after 4 pm, or on a weekend or market holiday: the next trading day.

Trading days are the NASDAQ dates in the price file. Early-close days (1 pm) are treated as normal days. Under this rule 928,844 of 2,487,308 in-window company-tweet rows (37%) belong to a later day than their calendar date.

## 5. Correlation tests

Every test is a Pearson correlation (r), computed per company, author class and abnormal-return measure. An r of 0 means no linear link and 1 means perfect agreement.

- **A. Significant days:** sentiment against the direction (+1 or -1) of the abnormal return, on significant days only.
- **B. Every day:** sentiment against the signed abnormal return, on all days.
- **C. Previous day:** the previous trading day's sentiment (tweets posted before that day's close) against the abnormal return.

The headline result is test A on the risk-adjusted return. The other tests and measures are robustness checks. In total the code computes 108 correlations (3 measures x 3 tests x 4 companies x 3 author groups) without correcting for multiple tests. The three measures overlap heavily, so the number of significant cells across the grid is not a count of independent results.

**Precision.** Each cell uses about 480 days, so one standard error is about 0.046. A correlation of roughly 0.09 is needed for p < 0.05, and roughly 0.13 for 80% power. The p-values and the confidence intervals assume independent days. Daily sentiment is strongly autocorrelated (about 0.5 from one day to the next for organic accounts and 0.7 for bot accounts), but daily abnormal returns are not (autocorrelations within +/-0.03 at lags 1 to 5). A correlation's uncertainty depends on the autocorrelation of both series together, and the estimated inflation from this is about 1.0, so the assumption costs little here. The intervals are in `correlations.csv`, `session_split.csv` and `robustness.csv` (`ci_low`, `ci_high`) and appear as whiskers on the dashboard charts.

Headline results (test A, risk-adjusted), with 95% confidence intervals (Fisher z):

| Company | All tweets | Bot accounts | Organic accounts |
|---|---|---|---|
| Apple | +0.08 [-0.01, +0.17] | -0.01 [-0.09, +0.08] | +0.31 [+0.23, +0.39] |
| Amazon | +0.25 [+0.17, +0.33] | +0.04 [-0.05, +0.13] | +0.33 [+0.25, +0.41] |
| Google | +0.07 [-0.02, +0.16] | 0.00 [-0.09, +0.08] | +0.15 [+0.06, +0.23] |
| Microsoft | +0.13 [+0.04, +0.22] | +0.04 [-0.05, +0.13] | +0.15 [+0.06, +0.23] |

The bot intervals include zero and extend to about +/-0.13, so a small correlation for bot accounts cannot be ruled out. Across all 36 test-A cells (3 measures x 4 companies x 3 groups) none of the 12 bot cells is significant and all 12 organic cells are. With the previous day's sentiment (test C), none of the 36 cells is significant.

## 6. Before the open and during the session

Tweets are split by when they were posted. **Before the open** covers evenings, overnight, weekends, holidays and pre-market. **During the session** is 9:30 am to 4 pm Eastern. Before the open accounts for 66% of bot-account tweets and 53% of organic-account tweets. The tests are run on each part separately, for the risk-adjusted return:

| Test A, correlation | Organic, before open | Organic, during session | Bot, before open | Bot, during session |
|---|---|---|---|---|
| Apple | +0.11* | +0.45* | +0.01 | -0.03 |
| Amazon | +0.15* | +0.37* | +0.04 | +0.02 |
| Google | -0.02 | +0.26* | +0.02 | -0.04 |
| Microsoft | +0.05 | +0.18* | +0.04 | +0.04 |
| Average | +0.07 | +0.31 | +0.03 | 0.00 |

(\* = significant at p < 0.05.) The every-day test gives the same pattern (organic averages +0.08 before the open and +0.24 during the session; bot +0.04 and +0.02). Most of the organic link therefore sits in tweets posted while the price is moving. Tweets posted before the open can still be reacting to the previous day's move, so a small correlation there does not show that sentiment leads returns.

## 7. Robustness to the bot definition

Test A is rerun at five bot definitions (risk-adjusted return):

| Definition | Bot authors | Bot share of tweet rows | Bot r | Organic r |
|---|---|---|---|---|
| 3+ points | 1,521 | 40.3% | 0.00 to +0.04 | +0.15 to +0.35 |
| 4+ points (baseline) | 498 | 33.5% | -0.01 to +0.04 | +0.15 to +0.33 |
| 4+ points, 20-tweet minimum | 315 | 33.4% | -0.01 to +0.04 | +0.14 to +0.37 |
| 5+ points | 100 | 25.2% | -0.01 to +0.06 | +0.11 to +0.31 |
| 6+ points | 31 | 18.1% | +0.01 to +0.04 | +0.11 to +0.29 |

In all five, no bot cell is significant and every organic cell is. The share here counts company-tweet rows in 2016-2019, so it differs slightly from the 36.0% above, which counts unique tweets from 2015-2019. This check changes only the cutoff; the features and point values stay the same.

## 8. Relation to the paper

The paper's Table 1 reports one all-tweet correlation per company. The R script that produced it still exists, and the comparison below is built from it.

**How the paper's Table 1 was produced** (from the script):

- Returns are percent price changes. Mean-adjusted, market-adjusted and risk-adjusted returns use the same 2011-2015 estimation window as here (1,257 days).
- A day qualifies only if it is outside the 25th-75th percentile on all four measures at once (the raw price change and the three abnormal returns). The percentiles cover January 2016 to May 2020, and the days are then cut to those with tweet data: 329 (Amazon), 345 (Apple), 314 (Google) and 332 (Microsoft).
- The up-or-down variable is the direction (+1 or -1) of the raw price change.
- Sentiment is the QDAP dictionary score from R's SentimentAnalysis package, for tweets selected by company name or ticker in the tweet text (across the whole tweet file) and matched to the UTC calendar date. The daily mean on qualifying days is correlated with the direction.

Rebuilt this way, with current Yahoo prices and the paper's saved daily sentiment files, the script gives 0.356, 0.051, 0.316 and 0.280, which match Table 1 to two decimals.

**Where the paper's text and script differ.**

| Item | Paper's text | Paper's script | Effect |
|---|---|---|---|
| Days that qualify | Reads as outside the quartiles on any of the four measures | All four at once, which matches the roughly 330 days the paper reports | A reader following the text would get a different day set |
| Market-adjusted return | Stock return minus the market's return that day | Stock return minus the market's average return over 2011-2015 | Using the formula as written removes 3-12% of the days; all-tweet correlations change by at most 0.02 and bot correlations by at most 0.04 (with this project's sentiment) |

The direction variable and the sentiment package are the same in the text and the script.

**How this project differs from the script.**

| Item | The paper's script | This project |
|---|---|---|
| Qualifying days | Outside the quartiles on all four measures at once | Outside the quartiles of one measure at a time (headline: risk-adjusted) |
| Up-or-down variable | Direction of the raw price change | Direction of the abnormal return |
| Market-adjusted return | Minus the market's average return | Minus the market's return that day |
| Returns | Percent change | Log change |
| Sentiment | QDAP dictionary (R) | VADER |
| Tweet selection | Company name or ticker in the text | The dataset's ticker tags |
| Matching tweets to days | UTC calendar date | The trading-day rule (section 4) |
| Author classes | None | Bot, organic, unclassified |

Because of these differences, the paper's figures are not shown beside the headline table in the README.

**The same days, side by side.** On the paper's exact days and with its direction variable, this project's sentiment gives the following. Difference is the paper's value minus this project's, with a 95% interval from a paired bootstrap over days (4,000 resamples; 10-day blocks give a similar picture).

| Company | Days | Paper, all tweets | This project, all tweets | Difference [95% CI] | This project, UTC dates | Organic | Bot |
|---|---|---|---|---|---|---|---|
| Amazon | 329 | 0.36 | 0.34 | +0.02 [-0.08, +0.12] | 0.40 | 0.42 | 0.06 |
| Apple | 345 | 0.05 | 0.07 | -0.02 [-0.08, +0.04] | 0.13 | 0.37 | -0.04 |
| Google | 314 | 0.32 | 0.11 | +0.21 [+0.10, +0.32] | 0.17 | 0.25 | -0.03 |
| Microsoft | 332 | 0.28 | 0.20 | +0.08 [-0.03, +0.20] | 0.25 | 0.22 | 0.08 |

(Microsoft's script keeps 342 days, but 10 fall after the tweet data ends.)

- Amazon and Apple agree within noise. Google differs. Microsoft differs by 0.08, with an interval that includes zero.
- Matching tweets to UTC dates raises every all-tweet correlation, so part of the gap comes from the day-matching rule. The remaining Google gap is not isolated: the paper's daily series and this project's correlate at 0.85 for Apple but only 0.47 for Google and 0.37 for Microsoft, which points to differences in sentiment tool and tweet selection.
- On these days the organic correlations are higher than the paper's all-tweet figures for Amazon and Apple and lower for Google and Microsoft. Bot correlations stay near zero (-0.04 to +0.08).

This comparison was computed outside the repository, from the paper's R script and its saved daily sentiment files, which are not part of it.

## 9. Limitations

- **The bot label is a heuristic.** No ground-truth labels exist, the thresholds are my choices, and the score has not been validated against known bots. Each author gets one fixed label for 2015-2019, based only on their tweets about these four companies.
- **The bot group is mostly automated news and link feeds.** A lack of correlation may reflect what those feeds post (headlines on a schedule) as much as any attempt to influence prices. The analysis does not test for manipulation.
- **"Organic" includes publishers.** Part of its correlation can be news reacting to price moves rather than individual opinion. The two cannot be separated without labels.
- **"All tweets" is not bot plus organic.** It also includes the unclassified group (5.2% of tweets), whose tweets are more positive on average.
- **Same-session tweets can react to the price.** Correlation is not causation, and a same-day correlation is not evidence that sentiment leads returns. The previous-day test (section 5) is the check on that, and it finds no lead.
- **Multiple tests.** 108 correlations are computed without correction for multiple tests, and the three abnormal-return measures overlap heavily, so the number of significant cells across the grid is not a count of independent results (section 5).
- **Abnormal returns are simple.** One fixed estimation window (2011-2015) is used through 2019, so betas that drifted are not captured. Returns exclude dividends. Early-close days are treated as normal.
- **VADER is not tuned for finance** and has not been checked against labeled financial text. Sarcasm and emoji are not handled.
- **Rows that are dropped:** 25,842 tweets with no author, and any company-day-group cell with fewer than 5 tweets.
