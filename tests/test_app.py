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


def test_accounts_table_is_an_ag_grid_with_every_account():
    import dash_ag_grid as dag
    grids = [n for n, _ in walk(app.tab3_layout()) if isinstance(n, dag.AgGrid)]
    assert len(grids) == 1
    grid = grids[0]
    rows = grid.rowData
    assert [c["field"] for c in grid.columnDefs] == ["account", "class", "score", "tweets", "per day", "peak day",
                                                      "links", "$tags", "repeats", "no engagement"]
    assert len(rows) == 100 and sum(r["class"] == "bot" for r in rows) == 60 and {r["class"] for r in rows} == {"bot", "organic"}
    # percentages are stored as numbers (so they sort and filter numerically) and only formatted for display
    assert all(isinstance(r[c], int) and 0 <= r[c] <= 100 for r in rows for c in ("links", "repeats", "no engagement"))
    formatters = {c["field"]: c["valueFormatter"]["function"] for c in grid.columnDefs if "valueFormatter" in c}
    assert "toLocaleString" in formatters["tweets"] and "'%'" in formatters["links"]
    assert grid.rowClassRules == {"bot-row": "params.data['class'] === 'bot'"}
    assert grid.dashGridOptions["pagination"] is True and grid.dashGridOptions["paginationPageSize"] == 15


def test_dashboard_does_not_use_the_deprecated_datatable():
    from pathlib import Path
    assert "dash_table" not in Path(app.__file__).read_text(encoding="utf-8")
