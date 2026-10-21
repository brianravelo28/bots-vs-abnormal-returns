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

A **significant day** is a day whose abnormal return is below the 25th or above the 75th percentile of that measure, as in the paper. The percentiles come from the January 2016 to June 2020 distribution (1,131 days), so 566 of those days are flagged for every company and model by construction. Within 2016-2019 the flagged share is 48-49%. The flag marks the larger moves; it is not a statistical test that a day was unusual.

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

- **A. Significant days (the paper's method):** sentiment against the direction (+1 or -1) of the abnormal return, on significant days only.
- **B. Every day:** sentiment against the signed abnormal return, on all days.
- **C. Previous day:** the previous trading day's sentiment (tweets posted before that day's close) against the abnormal return.

The headline result is test A on the risk-adjusted return. The other tests and measures are robustness checks. In total the code computes 108 correlations (3 measures x 3 tests x 4 companies x 3 author groups) without correcting for multiple tests. The three measures overlap heavily, so the number of significant cells across the grid is not a count of independent results.

**Precision.** Each cell uses about 480 days, so one standard error is about 0.046. A correlation of roughly 0.09 is needed for p < 0.05, and roughly 0.13 for 80% power. The p-values assume independent days; daily sentiment is autocorrelated, so p-values and intervals describe the data rather than prove anything.

Headline results (test A, risk-adjusted), with 95% confidence intervals (Fisher z, which assumes independent days):

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

The paper's test is rerun at five bot definitions (risk-adjusted return):

| Definition | Bot authors | Bot share of tweet rows | Bot r | Organic r |
|---|---|---|---|---|
| 3+ points | 1,521 | 40.3% | 0.00 to +0.04 | +0.15 to +0.35 |
| 4+ points (baseline) | 498 | 33.5% | -0.01 to +0.04 | +0.15 to +0.33 |
| 4+ points, 20-tweet minimum | 315 | 33.4% | -0.01 to +0.04 | +0.14 to +0.37 |
| 5+ points | 100 | 25.2% | -0.01 to +0.06 | +0.11 to +0.31 |
| 6+ points | 31 | 18.1% | +0.01 to +0.04 | +0.11 to +0.29 |

In all five, no bot cell is significant and every organic cell is. The share here counts company-tweet rows in 2016-2019, so it differs slightly from the 36.0% above, which counts unique tweets from 2015-2019. This check changes only the cutoff; the features and point values stay the same.

## 8. Differences from the paper

| Company | Paper, all tweets | This project, all tweets | Difference | Bot | Organic |
|---|---|---|---|---|---|
| Apple | 0.05 | +0.08 | +0.03 | -0.01 | +0.31 |
| Amazon | 0.36 | +0.25 | -0.11 | +0.04 | +0.33 |
| Google | 0.32 | +0.07 | -0.25 | 0.00 | +0.15 |
| Microsoft | 0.28 | +0.13 | -0.15 | +0.04 | +0.15 |

The paper's figures are from Table 1 (tweets column). The table does not say which abnormal-return measure it reports, so the comparison uses the risk-adjusted one. The ranking is similar but not identical: the paper orders the companies Amazon, Google, Microsoft, Apple; the all-tweets column here orders them Amazon, Microsoft, Apple, Google.

**What differs in method.**

- Sentiment tool: VADER here; R's SentimentAnalysis package in the paper.
- Matching tweets to days: here a tweet counts toward the trading day whose return window contains it (section 4). The paper does not describe how tweets were assigned to days.
- Abnormal returns: the Excel workbook of abnormal-return calculations kept with the paper's data is a monthly template on the S&P 500, so it cannot be the source of daily results. Here the daily calculation is rebuilt from prices (section 2).
- Window: the paper describes 2016 to June 2020, but its tweet data ends in 2019, as it does here, so the correlations cover 2016-2019 in both.

**What can be concluded.** The like-for-like evidence is the three columns computed the same way in this project (all, bot, organic). The organic correlations are not higher than the paper's all-tweet figures for Amazon (0.33 against 0.36), Google (0.15 against 0.32) or Microsoft (0.15 against 0.28), and are higher only for Apple (0.31 against 0.05). The differences from the paper therefore cannot be attributed to bots; only the split within this project can.

**Precision.** One standard error is about 0.05 with roughly 480 days, so small differences such as Apple's 0.08 against 0.05 are within noise. The paper reports about 330 abnormal-return days per company on average but not the number behind each cell of Table 1. No test of the paper-versus-project differences was run.

## 9. Limitations

- **The bot label is a heuristic.** No ground-truth labels exist, the thresholds are my choices, and the score has not been validated against known bots. Each author gets one fixed label for 2015-2019, based only on their tweets about these four companies.
- **The bot group is mostly automated news and link feeds.** A lack of correlation may reflect what those feeds post (headlines on a schedule) as much as any attempt to influence prices. The analysis does not test for manipulation.
- **"Organic" includes publishers.** Part of its correlation can be news reacting to price moves rather than individual opinion. The two cannot be separated without labels.
- **"All tweets" is not bot plus organic.** It also includes the unclassified group (5.2% of tweets), whose tweets are more positive on average.
- **Same-session tweets can react to the price.** Correlation is not causation, and a same-day correlation is not evidence that sentiment leads returns. The previous-day test (section 5) is the check on that, and it finds no lead.
- **The statistics are descriptive.** Days are not independent, and 108 correlations are computed without correction (section 5).
- **Abnormal returns are simple.** One fixed estimation window (2011-2015) is used through 2019, so betas that drifted are not captured. Returns exclude dividends. Early-close days are treated as normal.
- **VADER is not tuned for finance** and has not been checked against labeled financial text. Sarcasm and emoji are not handled.
- **Rows that are dropped:** 25,842 tweets with no author, and any company-day-group cell with fewer than 5 tweets.
