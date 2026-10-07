# Data Quirks Catalog: Bots vs Abnormal Returns

Weird things found in the data, kept as interview stories: what it was, how it was found, what it would have broken, what we did.
"Verified" means counted or reproduced in the data. "Observed" means seen but not explained.

## Stock tweets (Kaggle, 3,717,964 tweets across 6 tickers)

| # | Quirk | Status | Impact / handling |
|---|---|---|---|
| 1 | **The tweets stop at 2019-12-31**, but the study window (and the paper's text) runs to June 2020. The last `post_date` is epoch 1577836553 (2019-12-31 UTC). The paper's results section says 2016-2019, so the text and the results disagree. | Verified | All sentiment analysis uses 2016-2019 (1,006 trading days per company). Abnormal returns are computed to June 2020 but not used past 2019. |
| 2 | **Only six tickers exist**: AAPL, TSLA, AMZN, GOOG, GOOGL, MSFT. The paper describes "hundreds of thousands of entities"; that is the headlines dataset, not the tweets. | Verified | GOOG and GOOGL (Alphabet share classes) are merged as GOOG: 392,569 + 327,569 = 720,138 tweets. TSLA is out of scope for now and is the cheap fifth company when expanding. |
| 3 | **A tweet can carry several tickers**, so `Company_Tweet` has 3,239,577 rows for the four companies but only 2,759,594 unique tweets. | Verified | Author-level features (bursts, duplicates) use unique tweets; company-level sentiment uses the company rows. Bot share is quoted as 36% on unique tweets and 34% on company rows, same score. |
| 4 | **25,842 tweets (0.9%) have no author** (`writer` is NULL). They can't be attributed to an account. | Verified | Left out of the author-level analysis (2,759,594 unique tweets scored, 2,733,752 analyzed). The count is recorded in `deploy_data/summary.json` and the README. |
| 5 | **Authorship is extremely concentrated.** The median author wrote 1 tweet; 80% of authors wrote fewer than 5. The 10 biggest accounts wrote 19.5% of all tweets; the 4 biggest (App_sw_, _peripherals, computer_hware, It_c0nsulting) alone wrote 349,389. Those four post 164-218 times per active day with peaks of 1,036-1,653 in one day. | Verified | Authors under 5 tweets are 'unclassified' (80% of authors, about 5% of tweets). 499 bot accounts wrote 36% of the tweets, so a plain average over all tweets is heavily weighted toward a few hundred accounts. |
| 6 | **Low-activity accounts are far more positive.** Mean VADER compound 0.38 and 63% positive for unclassified authors, versus 0.12-0.14 and about 41% for bot and organic accounts. | Observed (cause not investigated; could be promotional or first-time posters) | Kept as a separate bucket rather than folded into 'organic' so it can't inflate that group. |
| 7 | **Sentiment scoring drops nothing but assumes a lot.** VADER was built for social-media text but not for finance ('short', 'bull', '$AAPL puts'). The paper used R's SentimentAnalysis dictionary, so the numbers differ (for example AAPL all-tweets r is 0.12 here vs 0.05 in the paper). | Verified (the difference) | Reported as 'consistent with' the paper, not a replication. |

## Prices and the paper's calculation workbook

| # | Quirk | Status | Impact / handling |
|---|---|---|---|
| 8 | **`AR Calculations.xlsx` is not the daily calculation.** It is a monthly, AMZN-only template using the S&P 500, log returns and a hardcoded beta of 1.12 / intercept 0.0094, while the paper's text says daily NASDAQ closing prices. | Verified | Daily abnormal returns were rebuilt from Yahoo Finance prices. Both indexes were tried for the abnormal-return step (betas and return distributions are similar); the correlations were run against NASDAQ only, as in the paper. |
| 9 | **The paper's "significant day" rule flags half of all days by construction.** Outside the 25th-75th percentile of a measure means exactly 566 of 1,131 days (Jan 2016 - Jun 2020) for every company and model; within the 2016-2019 analysis window it is 48-49%. Between 338 and 384 days per company (full window) are flagged by all three models. | Verified | Kept as the baseline because it is the paper's definition (the goal was notable moves, not extremes); a full-sample correlation on every day is reported next to it. |
| 10 | **The biggest abnormal days are earnings days**: AAPL 2019-01-03 (-10.5%, iPhone revenue warning), AMZN 2016-01-29 / 2016-04-29 / 2017-10-27, GOOG 2019-04-30 / 2019-07-26. | Verified | A good validity check on the abnormal-return code. |

## Headlines (not used yet)

| # | Quirk | Status | Impact / handling |
|---|---|---|---|
| 11 | **Mojibake in `raw_partner_headlines.csv`**, e.g. `$5â€¦â€¦ Million of Senior Notes`. UTF-8 text was decoded as Windows-1252 somewhere upstream. | Verified (seen in the first rows) | Will need repair before any text analysis of headlines. |
| 12 | **The two headline files overlap.** `raw_analyst_ratings.csv` (313 MB) and `analyst_ratings_processed.csv` (150 MB) start with the same headlines; the processed one drops URL and publisher. | Observed (only the first rows were compared) | Use one of them, not both, when headlines are added. |

## The bot label itself

| # | Quirk | Status | Impact / handling |
|---|---|---|---|
| 13 | **There is no ground truth for 'bot'.** The score is a points system (bursts, links, repeated text, cashtag spam, no engagement); nothing was checked against labelled accounts. | Verified (none exist in the data) | Tested with five threshold settings (3/4/5/6 points, and a 20-tweet minimum). Bot correlation stays between -0.04 and +0.06 at every setting (risk-adjusted return, significant days), and organic between +0.19 and +0.45. |
| 14 | **News publishers look fast but aren't automated spam.** SeekingAlpha, Benzinga and theflynews post thousands of tweets (5,272-12,055) with many links (78-97%), but almost no repeated text (1-5%) and more engagement than the typical account: 20-40% of their tweets get no reaction, against 60% for the median organic account and 78-97% for the 15 largest flagged accounts. They score 2 points each. | Verified (their scores and ratios) | The non-bot bucket is called 'organic', not 'human', and the dashboard says it includes publishers. |
