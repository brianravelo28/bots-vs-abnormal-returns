"""The robustness check re-draws the bot line at five settings; here it runs on a small dataset with known authors."""
import numpy as np
import pandas as pd
import pytest

from robustness import robustness

DAYS = pd.bdate_range("2018-01-02", periods=40)
TWEETS_PER_AUTHOR_PER_DAY = 6          # every group then clears the 5-tweets-a-day minimum

# O1: always organic. A: bot at every setting. B: a bot only when 3 points is enough. C: under 5 tweets, so unclassified.
AUTHORS = pd.DataFrame({"writer": ["O1", "A", "B", "C"], "n_tweets": [100, 100, 100, 3], "bot_score": [0, 6, 3, 7]})
SETTINGS = [(3, 5), (4, 5), (5, 5), (6, 5), (4, 20)]


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 1, len(DAYS))
    ar = pd.DataFrame({"company": "AAA", "date": DAYS, "RAR_dir": np.sign(returns).astype(int)})
    ar["RAR_sig"] = (returns < np.quantile(returns, 0.25)) | (returns > np.quantile(returns, 0.75))
    rows = []
    for i, day in enumerate(DAYS):
        for w in AUTHORS.writer:
            # O1's tweets follow the direction of the day's move; everyone else's are noise
            base = 0.5 * ar.RAR_dir[i] if w == "O1" else 0.0
            for k in range(TWEETS_PER_AUTHOR_PER_DAY):
                rows.append(dict(company="AAA", date=day, writer=w, compound=base + rng.normal(0, 0.3)))
    df = pd.DataFrame(rows).merge(AUTHORS, on="writer")
    res = robustness(df, AUTHORS, ar, settings=SETTINGS, companies=["AAA"])
    return df, ar, res


def row(res, bot_at, min_tweets, group):
    return res[(res.bot_at == bot_at) & (res.min_tweets == min_tweets) & (res.group == group)].iloc[0]


def test_one_row_per_setting_and_group(world):
    _, _, res = world
    assert len(res) == 10 and set(res.group) == {"bot", "organic"}
    assert res[["bot_at", "min_tweets"]].drop_duplicates().apply(tuple, axis=1).tolist() == SETTINGS
    assert ((res.ci_low <= res.r) & (res.r <= res.ci_high)).all()


def test_number_of_authors_labelled_bot_at_each_setting(world):
    _, _, res = world
    counts = {(s[0], s[1]): int(row(res, *s, "bot").bot_authors) for s in SETTINGS}
    # 3 points: A and B; 4, 5 and 6 points: A; 4 points with a 20-tweet minimum: A. C (3 tweets) is never counted.
    assert counts == {(3, 5): 2, (4, 5): 1, (5, 5): 1, (6, 5): 1, (4, 20): 1}


def test_share_of_tweet_rows_labelled_bot(world):
    _, _, res = world
    shares = {(s[0], s[1]): row(res, *s, "bot").bot_share_of_tweets for s in SETTINGS}
    # four authors post the same number of tweets: 2 of 4 are bots at 3 points, 1 of 4 at the others
    assert shares == pytest.approx({(3, 5): 0.5, (4, 5): 0.25, (5, 5): 0.25, (6, 5): 0.25, (4, 20): 0.25})


def test_moving_an_author_into_the_bot_group_changes_the_organic_result(world):
    df, ar, res = world
    # At 3 points B (noise) is a bot, so the organic group is just O1, whose tweets follow the move. At 4 points B
    # joins the organic group and dilutes it.
    sig = ar.set_index("date")[ar.set_index("date").RAR_sig]
    o1 = df[df.writer == "O1"].groupby("date").compound.mean().reindex(sig.index)
    expected = np.corrcoef(o1, sig.RAR_dir)[0, 1]
    assert row(res, 3, 5, "organic").r == pytest.approx(expected)
    assert row(res, 3, 5, "organic").r > row(res, 4, 5, "organic").r
    assert row(res, 3, 5, "organic").n == int(ar.RAR_sig.sum())


def test_group_days_with_too_few_tweets_are_dropped(world):
    df, ar, res = world
    first_sig_day = ar[ar.RAR_sig].date.iloc[0]
    # at 3 points the organic group is O1 alone; leave O1 only 2 tweets on that day
    drop = df[(df.writer == "O1") & (df.date == first_sig_day)].index[2:]
    thin = df.drop(drop)
    kept = robustness(thin, AUTHORS, ar, settings=[(3, 5)], companies=["AAA"])
    organic = kept[kept.group == "organic"].iloc[0]
    assert organic.n == row(res, 3, 5, "organic").n - 1
    loose = robustness(thin, AUTHORS, ar, settings=[(3, 5)], companies=["AAA"], min_day_tweets=2)
    assert loose[loose.group == "organic"].iloc[0].n == row(res, 3, 5, "organic").n


def test_bot_group_stays_unrelated_to_returns_at_every_setting(world):
    _, _, res = world
    assert (res[res.group == "bot"].r.abs() < 0.5).all()
    assert (res[res.group == "organic"].r > 0.3).all()
