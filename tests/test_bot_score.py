import pandas as pd
import pytest

from bot_score import score_authors


def author(**overrides):
    """One author with benign defaults (scores 0 on every rule); override a field to trigger a rule."""
    row = dict(writer="a", n_tweets=10, max_tweets_in_a_day=1, tweets_per_active_day=1.0,
               url_ratio=0.0, dup_ratio=0.0, cashtag_avg=1.0, zero_eng_ratio=0.5)
    row.update(overrides)
    return pd.DataFrame([row])


def scored(**overrides):
    return score_authors(author(**overrides)).iloc[0]


def test_defaults_score_zero_and_are_organic():
    row = scored()
    assert row.bot_score == 0 and row.author_class == "organic"


@pytest.mark.parametrize("field, value, column, points", [
    ("max_tweets_in_a_day", 19, "pts_volume", 0),
    ("max_tweets_in_a_day", 20, "pts_volume", 1),
    ("max_tweets_in_a_day", 49, "pts_volume", 1),
    ("max_tweets_in_a_day", 50, "pts_volume", 2),
    ("tweets_per_active_day", 9.99, "pts_volume", 0),
    ("tweets_per_active_day", 10.0, "pts_volume", 1),
    ("url_ratio", 0.89, "pts_links", 0),
    ("url_ratio", 0.90, "pts_links", 1),
    ("dup_ratio", 0.19, "pts_repeat", 0),
    ("dup_ratio", 0.20, "pts_repeat", 1),
    ("dup_ratio", 0.49, "pts_repeat", 1),
    ("dup_ratio", 0.50, "pts_repeat", 2),
    ("cashtag_avg", 4.99, "pts_cashtag", 0),
    ("cashtag_avg", 5.0, "pts_cashtag", 1),
    ("zero_eng_ratio", 0.89, "pts_noeng", 0),
    ("zero_eng_ratio", 0.90, "pts_noeng", 1),
])
def test_each_rule_gives_the_points_in_the_table(field, value, column, points):
    row = scored(**{field: value})
    assert row[column] == points
    assert row.bot_score == points  # nothing else fires


def test_volume_rules_add_up():
    # 50+ tweets in a day (2 points) and 10+ per active day (1 point) both count
    assert scored(max_tweets_in_a_day=60, tweets_per_active_day=12).pts_volume == 3


def test_maximum_score_is_eight():
    row = scored(max_tweets_in_a_day=100, tweets_per_active_day=50, url_ratio=1.0,
                 dup_ratio=0.9, cashtag_avg=8.0, zero_eng_ratio=1.0)
    assert row.bot_score == 8 and row.author_class == "bot"


def test_four_points_is_a_bot_and_three_is_organic():
    four = scored(max_tweets_in_a_day=50, url_ratio=1.0, zero_eng_ratio=1.0)    # 2 + 1 + 1
    three = scored(max_tweets_in_a_day=50, url_ratio=1.0)                       # 2 + 1
    assert (four.bot_score, four.author_class) == (4, "bot")
    assert (three.bot_score, three.author_class) == (3, "organic")


def test_authors_under_five_tweets_are_unclassified_whatever_their_score():
    row = scored(n_tweets=4, max_tweets_in_a_day=100, url_ratio=1.0, dup_ratio=0.9, zero_eng_ratio=1.0)
    assert row.bot_score >= 4 and row.author_class == "unclassified"
    assert scored(n_tweets=5, max_tweets_in_a_day=100, url_ratio=1.0, dup_ratio=0.9).author_class == "bot"


def test_thresholds_can_be_changed_for_the_robustness_check():
    f = author(max_tweets_in_a_day=50, url_ratio=1.0)  # 3 points
    assert score_authors(f, bot_at=3).author_class.iloc[0] == "bot"
    assert score_authors(f, bot_at=4).author_class.iloc[0] == "organic"


def test_input_table_is_not_modified():
    f = author(url_ratio=1.0)
    before = list(f.columns)
    score_authors(f)
    assert list(f.columns) == before
