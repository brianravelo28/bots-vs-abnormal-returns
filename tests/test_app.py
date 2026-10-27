"""Dashboard checks: the dashboard conventions and that every chart callback produces a figure."""
import pytest
from dash import dcc, html
from dash.development.base_component import Component

import app

LAYOUTS = {
    "Bots vs Organic": app.tab1_layout,
    "Timeline": app.tab2_layout,
    "Tweet timing": app.tab5_layout,
    "The Accounts": app.tab3_layout,
    "Robustness": app.tab4_layout,
}


def walk(node, parent=None):
    """Yield (component, parent) for every Dash component under `node`."""
    yield node, parent
    children = getattr(node, "children", None)
    if children is None:
        return
    for child in children if isinstance(children, (list, tuple)) else [children]:
        if isinstance(child, Component):
            yield from walk(child, node)


@pytest.mark.parametrize("name", LAYOUTS)
def test_every_chart_has_the_hint_line_and_no_mode_bar(name):
    graphs = [(n, p) for n, p in walk(LAYOUTS[name]()) if isinstance(n, dcc.Graph)]
    assert graphs, f"{name} should contain a chart"
    for graph, parent in graphs:
        assert graph.config == {"displayModeBar": False}
        siblings = parent.children
        assert any(isinstance(s, html.P) and s.className == "chart-hint" and s.children == app.CHART_HINT
                   for s in siblings)


@pytest.mark.parametrize("name", LAYOUTS)
def test_dropdowns_with_fewer_than_seven_options_are_not_searchable(name):
    for node, _ in walk(LAYOUTS[name]()):
        if isinstance(node, dcc.Dropdown) and len(node.options) < 7:
            assert node.searchable is False


def text_labels(fig):
    return [t for trace in fig.data if getattr(trace, "mode", None) == "text" for t in trace.text]


def assert_clean_figure(fig):
    assert len(fig.data) > 0
    assert not any(str(t).lstrip("+").startswith("-0.00") or str(t).startswith("+0.00") for t in text_labels(fig))
    assert not any("paper" in (trace.name or "").lower() and "threshold" not in (trace.name or "").lower()
                   for trace in fig.data)


@pytest.mark.parametrize("method", list(app.METHODS))
@pytest.mark.parametrize("measure", list(app.MEASURES))
def test_bots_vs_organic_chart(measure, method):
    fig, note = app.update_tab1(measure, method)
    assert_clean_figure(fig)
    bars = [t for t in fig.data if t.type == "bar"]
    assert len(bars) == 3 and all(b.error_y is not None and b.error_y.array is not None for b in bars)
    assert "Whiskers are 95% confidence intervals" in note and "Correlation, not causation" in note


@pytest.mark.parametrize("method", list(app.TIMING_METHODS))
def test_tweet_timing_chart(method):
    fig = app.update_tab5(method)
    assert_clean_figure(fig)
    assert len([t for t in fig.data if t.type == "bar"]) == 4


@pytest.mark.parametrize("company", app.COMPANIES)
def test_robustness_and_timeline_charts(company):
    assert_clean_figure(app.update_tab4(company))
    assert_clean_figure(app.update_tab2(company, "2016-01-01", "2019-12-31"))


def test_bar_values_match_the_published_table():
    fig, _ = app.update_tab1("RAR", "A_sig_days_direction")
    organic = next(t for t in fig.data if t.name == "Organic accounts")
    expected = app.corr[(app.corr.measure == "RAR") & (app.corr.method == "A_sig_days_direction")
                        & (app.corr.group == "organic")].set_index("company").r.reindex(app.COMPANIES)
    assert list(organic.y) == pytest.approx(list(expected))
