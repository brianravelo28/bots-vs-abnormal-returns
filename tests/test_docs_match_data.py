"""The numbers quoted in the README and docs/METHODOLOGY.md must match the committed output files.

If the pipeline is rerun and the data changes, these tests fail until the text is updated. Not covered: the
comparison with the paper (section 8 of the Methodology), which uses files outside this repository.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
D = ROOT / "deploy_data"
README = (ROOT / "README.md").read_text(encoding="utf-8")
METHOD = (ROOT / "docs" / "METHODOLOGY.md").read_text(encoding="utf-8")
TICKER = {"Apple": "AAPL", "Amazon": "AMZN", "Google": "GOOG", "Microsoft": "MSFT"}


def fmt(v):
    """Two decimals with a sign, and no negative zero (the docs print 0.00)."""
    s = f"{v:+.2f}"
    return "0.00" if s in ("+0.00", "-0.00") else s


def star(p):
    return "*" if p < 0.05 else ""


def table(text, header_start):
    """Cells of the markdown table whose header row starts with `header_start`, one list per body row."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(header_start))
    rows = []
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        rows.append([c.strip().replace("\\*", "*") for c in line.strip().strip("|").split("|")])
    return rows


def span(values):
    return f"{fmt(values.min())} to {fmt(values.max())}"


@pytest.fixture(scope="module")
def corr():
    return pd.read_csv(D / "correlations.csv")


@pytest.fixture(scope="module")
def session():
    return pd.read_csv(D / "session_split.csv")


@pytest.fixture(scope="module")
def robust():
    return pd.read_csv(D / "robustness.csv")


@pytest.fixture(scope="module")
def summary():
    return json.loads((D / "summary.json").read_text())


def headline(corr):
    a = corr[(corr.method == "A_sig_days_direction") & (corr.measure == "RAR")]
    return a.set_index(["company", "group"])


# ----------------------------------------------------------------------------------------- README

def test_readme_results_table(corr):
    h = headline(corr)
    rows = table(README, "| Company | All tweets | Bot accounts | Organic accounts |")
    assert [r[0] for r in rows] == list(TICKER)
    for name, *cells in rows:
        expected = [fmt(h.loc[(TICKER[name], g), "r"]) + star(h.loc[(TICKER[name], g), "p"]) for g in ("all", "bot", "organic")]
        assert cells == expected, name


def test_readme_bot_statements(corr, summary):
    bot = lambda m: corr[(corr.method == m) & (corr.group == "bot")]
    a, b = bot("A_sig_days_direction"), bot("B_all_days_signed_AR")
    assert (a.p < 0.05).sum() == 0 and (b.p < 0.05).sum() == 0 and len(a) == len(b) == 12
    s = summary
    assert (f"{s['bot_authors']} accounts ({s['bot_authors'] / s['authors']:.1%} of authors) wrote "
            f"{s['bot_tweet_share']:.0%} of the {s['unique_tweets_scored'] / 1e6:.2f}M tweets") in README
    assert f"between {fmt(a.r.min())} and {fmt(a.r.max())} for every company and abnormal-return model, none significant (0 of 12)" in README
    assert f"({span(b.r)}, 0 of 12)" in README


def test_readme_no_lead_statement(corr):
    c = corr[corr.method == "C_prev_day_sentiment"]
    organic, bot = c[c.group == "organic"], c[c.group == "bot"]
    assert (organic.p < 0.05).sum() == 0 and (bot.p < 0.05).sum() == 0
    assert f"organic {span(organic.r)} and bot {span(bot.r)} (0 of 12 each)" in README


def test_readme_robustness_statement(robust):
    bot, organic = robust[robust.group == "bot"], robust[robust.group == "organic"]
    assert (bot.p < 0.05).sum() == 0 and (organic.p < 0.05).all() and len(bot) == len(organic) == 20
    assert (f"the bot correlation stays between {fmt(bot.r.min())} and {fmt(bot.r.max())}, none significant (0 of 20), "
            f"and organic stays between {fmt(organic.r.min())} and {fmt(organic.r.max())}, all significant (20 of 20)") in README


def test_readme_timing_statement(session, summary):
    share = summary["session_share"]
    assert f"About {share['bot']['before_open']:.0%} of bot-account tweets and {share['organic']['before_open']:.0%} of organic-account tweets" in README
    a = session[(session.method == "A_sig_days_direction") & (session.group == "organic")]
    during, before = a[a.session == "intraday"], a[a.session == "before_open"].set_index("company")
    assert (during.p < 0.05).all()
    assert set(before.index[before.p < 0.05]) == {"AAPL", "AMZN"}
    assert f"(average r {fmt(during.r.mean())}, significant for all four companies)" in README
    assert (f"({fmt(before.r.mean())}, significant only for Apple and Amazon at "
            f"{fmt(before.loc['AAPL', 'r'])} and {fmt(before.loc['AMZN', 'r'])})") in README


def test_readme_counts(summary):
    s = summary
    assert f"{s['analysis_days']:,} trading days per company" in README
    assert f"{s['unique_tweets_scored'] + s['null_author_tweets']:,} unique tweets were scored" in README
    assert f"{s['unique_tweets_scored']:,} of them have a known author" in README
    assert f"the other {s['null_author_tweets']:,} have no author" in README


# ------------------------------------------------------------------------------------ Methodology

def test_methodology_headline_table_with_intervals(corr):
    h = headline(corr)
    rows = table(METHOD, "| Company | All tweets | Bot accounts | Organic accounts |")
    assert [r[0] for r in rows] == list(TICKER)
    for name, *cells in rows:
        expected = []
        for g in ("all", "bot", "organic"):
            r = h.loc[(TICKER[name], g)]
            expected.append(f"{fmt(r.r)} [{fmt(r.ci_low)}, {fmt(r.ci_high)}]")
        assert cells == expected, name


def test_methodology_class_table_and_counts():
    cs = pd.read_parquet(D / "class_summary.parquet").set_index("author_class")
    rows = {r[0]: r[1:] for r in table(METHOD, "| Class | Authors | Tweets | Share of tweets |")}
    for label, cls in (("Bot", "bot"), ("Organic", "organic"), ("Unclassified", "unclassified")):
        authors, tweets = int(cs.loc[cls, "authors"]), int(cs.loc[cls, "tweets"])
        assert rows[label] == [f"{authors:,} ({authors / cs.authors.sum():.1%})", f"{tweets:,}", f"{tweets / cs.tweets.sum():.1%}"], label
    s = json.loads((D / "summary.json").read_text())
    assert f"{s['unique_tweets_scored']:,} tweets from {s['authors']:,} authors" in METHOD
    assert f"{s['analysis_days']:,} trading days per company" in METHOD


def test_methodology_session_table_and_statements(session, summary):
    for method in ("A_sig_days_direction", "B_all_days_signed_AR"):
        assert session[session.method == method].shape[0] == 16
    a = session[session.method == "A_sig_days_direction"].set_index(["company", "group", "session"])
    rows = table(METHOD, "| Test A, correlation |")
    cols = [("organic", "before_open"), ("organic", "intraday"), ("bot", "before_open"), ("bot", "intraday")]
    for name, *cells in rows[:-1]:
        assert cells == [fmt(a.loc[(TICKER[name], g, s), "r"]) + star(a.loc[(TICKER[name], g, s), "p"]) for g, s in cols], name
    average = rows[-1]
    assert average[0] == "Average"
    assert average[1:] == [fmt(a.xs((g, s), level=("group", "session")).r.mean()) for g, s in cols]
    share = summary["session_share"]
    assert (f"Before the open accounts for {share['bot']['before_open']:.0%} of bot-account tweets "
            f"and {share['organic']['before_open']:.0%} of organic-account tweets") in METHOD
    b = session[session.method == "B_all_days_signed_AR"].groupby(["group", "session"]).r.mean()
    assert (f"organic averages {fmt(b['organic', 'before_open'])} before the open and {fmt(b['organic', 'intraday'])} during the session; "
            f"bot {fmt(b['bot', 'before_open'])} and {fmt(b['bot', 'intraday'])}") in METHOD


def test_methodology_robustness_table(robust):
    labels = {(3, 5): "3+ points", (4, 5): "4+ points (baseline)", (4, 20): "4+ points, 20-tweet minimum",
              (5, 5): "5+ points", (6, 5): "6+ points"}
    rows = table(METHOD, "| Definition | Bot authors |")
    assert [r[0] for r in rows] == list(labels.values())
    for row, ((bot_at, min_tweets), label) in zip(rows, labels.items()):
        g = robust[(robust.bot_at == bot_at) & (robust.min_tweets == min_tweets)]
        bot, organic = g[g.group == "bot"], g[g.group == "organic"]
        assert row == [label, f"{int(g.bot_authors.iloc[0]):,}", f"{g.bot_share_of_tweets.iloc[0]:.1%}",
                       span(bot.r), span(organic.r)], label
        assert (bot.p < 0.05).sum() == 0 and (organic.p < 0.05).all()
