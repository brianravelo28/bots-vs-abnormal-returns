"""
Bots vs Abnormal Returns — Interactive Dash App

Revisits a grad-school study (MIS 581, Colorado State University Global): does Twitter sentiment
line up with abnormal stock returns for AAPL, AMZN, GOOG and MSFT? The original could not tell
bots from real users. This dashboard adds that step.

Tabs:
1. Bots vs Organic   (sentiment/abnormal-return correlation, by author class)
2. Timeline          (abnormal returns and sentiment through time, per company)
3. The Accounts      (how the bot score works, who scored highest, example bot tweets)
4. Robustness        (does the result survive moving the bot threshold?)

Run: python app.py  ->  http://localhost:8060
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html
from plotly.subplots import make_subplots

BASE_DIR = Path(__file__).resolve().parent
DEPLOY = BASE_DIR / "deploy_data"

# Design tokens (same system as the Rossmann dashboard)
SURFACE = "#fcfcfb"
PAGE_BG = "#f4f4f2"
BORDER = "#e5e4df"
GRID = "#ecebe7"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#6b6a66"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
GROUP_COLOR = {"all": INK_3, "bot": ORANGE, "organic": BLUE, "unclassified": AQUA}
GROUP_LABEL = {"all": "All tweets", "bot": "Bot accounts", "organic": "Organic accounts", "unclassified": "Unclassified"}
COMPANIES = ["AAPL", "AMZN", "GOOG", "MSFT"]
COMPANY_NAME = {"AAPL": "Apple", "AMZN": "Amazon", "GOOG": "Google", "MSFT": "Microsoft"}
PAPER_R = {"AMZN": 0.36, "AAPL": 0.05, "GOOG": 0.32, "MSFT": 0.28}
METHODS = {
    "A_sig_days_direction": "Significant days: sentiment vs direction of the return (the paper's method)",
    "B_all_days_signed_AR": "Every day: sentiment vs the signed abnormal return",
    "C_prev_day_sentiment": "Every day: PREVIOUS day's sentiment vs the abnormal return (does it predict?)",
}
MEASURES = {"MAR": "Mean-adjusted (MAR)", "MKAR": "Market-adjusted (MKAR)", "RAR": "Risk-adjusted (RAR)"}

# ---------------------------------------------------------------- load ----
ar = pd.read_parquet(DEPLOY / "ar_daily.parquet")
sent = pd.read_parquet(DEPLOY / "daily_sentiment.parquet")
corr = pd.read_csv(DEPLOY / "correlations.csv")
robust = pd.read_csv(DEPLOY / "robustness.csv")
accounts = pd.read_parquet(DEPLOY / "top_accounts.parquet")
class_summary = pd.read_parquet(DEPLOY / "class_summary.parquet")
score_dist = pd.read_parquet(DEPLOY / "score_distribution.parquet")
examples = pd.read_parquet(DEPLOY / "example_bot_tweets.parquet")
S = json.loads((DEPLOY / "summary.json").read_text())

DATE_MIN, DATE_MAX = ar["date"].min(), ar["date"].max()


# ==================================================================== app ==
CSS = f"""
:root {{ color-scheme: light; }}
body {{ margin: 0; background: {PAGE_BG}; color: {INK};
  font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
.wrap {{ font-size: 15px; max-width: 1180px; margin: 0 auto; padding: 24px 16px 48px; }}
.card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px; padding: 16px 18px; }}
.kpi-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }}
.kpi {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px; padding: 12px 16px; }}
.kpi .v {{ font-size: 28px; font-weight: 650; letter-spacing: -0.01em; }}
.kpi .l {{ font-size: 15px; color: {INK_2}; margin-bottom: 2px; }}
.kpi .s {{ font-size: 15px; color: {INK_3}; margin-top: 2px; }}
.controls {{ display: flex; flex-wrap: wrap; gap: 16px 20px; align-items: flex-end; }}
.controls label {{ display: block; font-size: 15px; color: {INK_2}; margin-bottom: 4px; }}
.note {{ font-size: 15px; color: {INK_2}; line-height: 1.5; margin: 10px 2px 0; }}
.two {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 12px; margin-top: 12px; }}
details.about {{ margin-top: 12px; }}
details.about summary {{ cursor: pointer; font-weight: 600; color: {INK_2}; }}
details.about ul {{ margin: 8px 0 0; padding-left: 20px; color: {INK_2}; line-height: 1.55; font-size: 15px; }}
.tweet {{ border-left: 3px solid {ORANGE}; padding: 4px 10px; margin: 8px 0; color: {INK_2}; font-size: 14px; }}
.tweet b {{ color: {INK}; }}
.dash-dropdown, .dash-dropdown *, .dash-datepicker-input, .DateInput_input, .Select-value-label, .Select-input input {{ font-size: 15px !important; }}
.dash-options-list:not(.dash-checklist) .dash-options-list-option {{ display: flex !important; align-items: center; gap: 10px; width: 100%; box-sizing: border-box; padding: 8px 12px; margin: 0; cursor: pointer; font-size: 15px; }}
.dash-options-list:not(.dash-checklist) .dash-options-list-option:hover {{ background: {PAGE_BG}; }}
"""

app = Dash(__name__, suppress_callback_exceptions=True)  # tab contents are rendered on demand
app.title = "Bots vs Abnormal Returns"
server = app.server  # WSGI entry point for gunicorn
app.index_string = f"""<!DOCTYPE html>
<html>
<head>
{{%metas%}}
<title>{{%title%}}</title>
{{%favicon%}}
{{%css%}}
<style>{CSS}</style>
</head>
<body>
{{%app_entry%}}
<footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
</body>
</html>"""


def kpi(label, value, sub=None):
    return html.Div(
        [html.Div(label, className="l"), html.Div(value, className="v")] + ([html.Div(sub, className="s")] if sub else []),
        className="kpi",
    )


def style_fig(fig, title=None, height=None, legend=True):
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0, font=dict(size=17, color=INK)) if title else None,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(color=INK_2, size=15),
        hoverlabel=dict(font_size=15),
        margin=dict(l=70, r=20, t=60 if title else 30, b=90 if legend else 40),
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center", yanchor="top", font=dict(color=INK_2)),
        showlegend=legend,
        height=height,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=BORDER, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=BORDER, zeroline=False, automargin=True)
    return fig


def header():
    return html.Div(
        [
            html.H1("Bots vs Abnormal Returns", style={"margin": "0 0 4px", "fontSize": "28px", "letterSpacing": "-0.02em"}),
            html.Div(
                "Does Twitter sentiment line up with big stock moves for Apple, Amazon, Google and Microsoft — "
                "and how much of that sentiment comes from bots?",
                style={"color": INK_2, "marginBottom": "16px"},
            ),
            html.Div(
                [
                    kpi("Tweets scored", f"{S['unique_tweets_scored'] / 1e6:.2f}M", f"{S['authors']:,} authors, 2015–2019"),
                    kpi("Written by bot accounts", f"{S['bot_tweet_share']:.0%}", f"{S['bot_authors']} accounts ({S['bot_authors'] / S['authors']:.1%} of authors)"),
                    kpi("Bot sentiment vs returns", _range_label("bot"), "correlation, every company & model"),
                    kpi("Organic sentiment vs returns", _range_label("organic"), "same test, same days"),
                ],
                className="kpi-row",
            ),
            html.Details(
                [
                    html.Summary("About this project"),
                    html.Ul(
                        [
                            html.Li("A revisit of a graduate research paper (MIS 581, Colorado State University Global, 2022). It found weak-to-moderate links between tweet sentiment and abnormal returns but could not separate real users from bots; its own recommendation was to add that step. This is that step."),
                            html.Li("Abnormal returns (log, daily): mean-adjusted, market-adjusted and risk-adjusted (CAPM), estimated on 2011–2015 and measured Jan 2016 onward against the NASDAQ Composite. A day is 'significant' when its abnormal return falls outside that measure's 25th–75th percentile, as in the paper."),
                            html.Li("Sentiment: VADER compound score per tweet, averaged per company per day (US/Eastern dates). The paper used R's SentimentAnalysis package, so numbers are close in spirit, not identical."),
                            html.Li("Bots: a transparent points score per author (tweet bursts, link ratio, repeated text, cashtag spam, zero engagement). Authors with fewer than 5 tweets are 'unclassified'. 'Organic' means not flagged as automated — it includes news publishers, not only individuals."),
                            html.Li("Tweets end 2019-12-31, so all sentiment analysis uses 2016–2019 (1,006 trading days per company). Correlation is not causation, and there are no ground-truth bot labels: the score is a defensible heuristic, tested in the Robustness tab."),
                            html.Li("Data: Kaggle stock-tweet dataset (Dogan et al., 2020) and Yahoo Finance prices."),
                        ]
                    ),
                ],
                className="about",
            ),
        ],
        className="card",
        style={"marginBottom": "16px"},
    )


def _range_label(group):
    d = corr[(corr.group == group) & (corr.method == "A_sig_days_direction")]
    return f"{d.r.min():+.2f} to {d.r.max():+.2f}"


TAB_STYLE = {"padding": "10px 14px", "fontSize": "15px", "border": f"1px solid {BORDER}", "backgroundColor": PAGE_BG, "color": INK_2, "fontWeight": "500"}
TAB_SELECTED = {**TAB_STYLE, "backgroundColor": SURFACE, "color": INK, "fontWeight": "650", "borderTop": f"2px solid {BLUE}"}

app.layout = html.Div(
    className="wrap",
    children=[
        header(),
        dcc.Tabs(
            id="tabs",
            value="tab1",
            colors={"border": BORDER, "primary": BLUE, "background": PAGE_BG},
            children=[
                dcc.Tab(label="Bots vs Organic", value="tab1", style=TAB_STYLE, selected_style=TAB_SELECTED),
                dcc.Tab(label="Timeline", value="tab2", style=TAB_STYLE, selected_style=TAB_SELECTED),
                dcc.Tab(label="The Accounts", value="tab3", style=TAB_STYLE, selected_style=TAB_SELECTED),
                dcc.Tab(label="Robustness", value="tab4", style=TAB_STYLE, selected_style=TAB_SELECTED),
            ],
        ),
        html.Div(id="tab-content", style={"marginTop": "16px"}),
    ],
)


# ------------------------------------------------------------- tab layouts

def dropdown(label, id_, options, value, width="220px"):
    return html.Div(
        [html.Label(label), dcc.Dropdown(id=id_, options=options, value=value, clearable=False, style={"width": width})]
    )


def tab1_layout():
    return html.Div(
        [
            html.Div(
                [
                    dropdown("Abnormal-return model", "t1-measure", [{"label": v, "value": k} for k, v in MEASURES.items()], "RAR"),
                    dropdown("Test", "t1-method", [{"label": v, "value": k} for k, v in METHODS.items()], "A_sig_days_direction", "560px"),
                ],
                className="controls card",
            ),
            html.Div(dcc.Graph(id="t1-chart"), className="card", style={"marginTop": "12px"}),
            html.Div(id="t1-note", className="note"),
        ]
    )


def tab2_layout():
    return html.Div(
        [
            html.Div(
                [
                    dropdown("Company", "t2-company", [{"label": f"{COMPANY_NAME[c]} ({c})", "value": c} for c in COMPANIES], "AAPL"),
                    html.Div(
                        [
                            html.Label("Date range"),
                            dcc.DatePickerRange(id="t2-dates", min_date_allowed=DATE_MIN, max_date_allowed=DATE_MAX,
                                                start_date=DATE_MIN, end_date=DATE_MAX),
                        ]
                    ),
                ],
                className="controls card",
            ),
            html.Div(dcc.Graph(id="t2-chart"), className="card", style={"marginTop": "12px"}),
            html.Div(
                "Top: risk-adjusted return (RAR) each day; darker dots are 'significant' days (outside the 25th–75th percentile). "
                "Bottom: 14-day rolling average of daily sentiment (VADER, −1 to +1). Rolling averages smooth out the day-to-day link; the Bots vs Organic tab has the day-level correlations.",
                className="note",
            ),
        ]
    )


def tab3_layout():
    tot = class_summary.tweets.sum()
    order = ["bot", "organic", "unclassified"]
    cs = class_summary.set_index("author_class").loc[order]
    bar = go.Figure(
        [
            go.Bar(name="Share of authors", x=order, y=(cs.authors / cs.authors.sum()).values,
                   marker_color=[GROUP_COLOR[o] for o in order], opacity=0.45,
                   hovertemplate="%{x}: %{y:.1%} of authors<extra></extra>"),
            go.Bar(name="Share of tweets", x=order, y=(cs.tweets / tot).values,
                   marker_color=[GROUP_COLOR[o] for o in order],
                   hovertemplate="%{x}: %{y:.1%} of tweets<extra></extra>"),
        ]
    )
    bar.update_layout(barmode="group", yaxis_tickformat=".0%")
    style_fig(bar, "A tiny share of accounts, a third of the tweets", 360)

    sd = score_dist.copy()
    dist = go.Figure(go.Bar(x=sd.bot_score, y=sd.tweets, marker_color=[ORANGE if s >= 4 else BLUE for s in sd.bot_score],
                            customdata=sd.authors, hovertemplate="score %{x}: %{y:,} tweets, %{customdata:,} authors<extra></extra>"))
    dist.update_xaxes(title="Bot score (points)", dtick=1)
    dist.update_yaxes(title="Tweets", tickformat=",")
    style_fig(dist, "Tweets by bot score (authors with 5+ tweets)", 360, legend=False)

    tbl = accounts.copy()
    tbl["author_class"] = tbl.author_class.replace({"organic": "organic"})
    tbl = tbl.rename(columns={"writer": "account", "author_class": "class", "bot_score": "score", "n_tweets": "tweets",
                              "active_days": "active days", "tweets_per_active_day": "per day",
                              "max_tweets_in_a_day": "peak day", "url_ratio": "links", "cashtag_avg": "$tags",
                              "dup_ratio": "repeats", "zero_eng_ratio": "no engagement"})
    for c in ["links", "repeats", "no engagement"]:
        tbl[c] = (tbl[c] * 100).round(0).astype(int).astype(str) + "%"
    tbl["per day"] = tbl["per day"].round(1)
    tbl["$tags"] = tbl["$tags"].round(1)
    cols = ["account", "class", "score", "tweets", "per day", "peak day", "links", "$tags", "repeats", "no engagement"]

    ex = []
    for w, g in examples.groupby("writer"):
        ex.append(html.Div([html.B(f"@{w}"), *[html.Div(f"“{t}”") for t in g.body]], className="tweet"))

    return html.Div(
        [
            html.Div(
                [
                    html.B("How the score works. "),
                    "Each author earns points: a burst of 20+ tweets in a day (1) or 50+ (2); averaging 10+ tweets per active day (1); "
                    "90%+ of tweets contain a link (1); repeated text in 20%+ of tweets (1) or 50%+ (2); an average of 5+ $cashtags "
                    "per tweet (1); 90%+ of tweets get no likes, replies or retweets (1). Four points or more = bot. Authors with fewer "
                    "than 5 tweets are left unclassified. "
                    f"{S['unclassified_authors_share']:.0%} of authors fall there, but they wrote only about 5% of tweets.",
                ],
                className="card",
            ),
            html.Div([html.Div(dcc.Graph(figure=bar), className="card"), html.Div(dcc.Graph(figure=dist), className="card")], className="two"),
            html.Div(
                [
                    html.H3("Highest-volume flagged and organic accounts", style={"margin": "0 0 8px", "fontSize": "17px"}),
                    dash_table.DataTable(
                        data=tbl[cols].to_dict("records"),
                        columns=[{"name": c, "id": c} for c in cols],
                        page_size=15,
                        sort_action="native",
                        filter_action="native",
                        style_header={"backgroundColor": PAGE_BG, "fontWeight": "650", "border": f"1px solid {BORDER}"},
                        style_cell={"padding": "6px 10px", "border": f"1px solid {BORDER}", "fontSize": "14px", "color": INK_2,
                                    "backgroundColor": SURFACE, "textAlign": "left"},
                        style_data_conditional=[{"if": {"filter_query": '{class} = "bot"'}, "color": ORANGE, "fontWeight": "600"}],
                        style_table={"overflowX": "auto"},
                    ),
                    html.Div(
                        "The largest 'organic' accounts include news and alert publishers (e.g. SeekingAlpha, Benzinga, theflynews): "
                        "not individuals and not flagged as automated. That is why the label is 'organic' rather than 'human'.",
                        className="note",
                    ),
                ],
                className="card",
                style={"marginTop": "12px"},
            ),
            html.Div(
                [html.H3("What the top bots post", style={"margin": "0 0 4px", "fontSize": "17px"}),
                 html.Div("Random examples from the 15 largest bot accounts (links removed).", className="note", style={"margin": "0 0 6px"}), *ex],
                className="card",
                style={"marginTop": "12px"},
            ),
        ]
    )


def tab4_layout():
    return html.Div(
        [
            html.Div(
                [dropdown("Company", "t4-company", [{"label": f"{COMPANY_NAME[c]} ({c})", "value": c} for c in COMPANIES], "AAPL")],
                className="controls card",
            ),
            html.Div(dcc.Graph(id="t4-chart"), className="card", style={"marginTop": "12px"}),
            html.Div(
                "Each x-position re-draws the bot line: 'bot ≥ N' means an author needs at least N points to be flagged (baseline is 4); "
                "'min 20' requires 20+ tweets before an author is judged at all. Correlation is sentiment vs the direction of the risk-adjusted "
                "return on significant days. Bot correlation stays near zero at every setting; organic stays clearly positive.",
                className="note",
            ),
        ]
    )


@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    return {"tab1": tab1_layout, "tab2": tab2_layout, "tab3": tab3_layout, "tab4": tab4_layout}[tab]()


# --------------------------------------------------------------- callbacks

@app.callback(Output("t1-chart", "figure"), Output("t1-note", "children"), Input("t1-measure", "value"), Input("t1-method", "value"))
def update_tab1(measure, method):
    d = corr[(corr.measure == measure) & (corr.method == method)]
    fig = go.Figure()
    for g in ["all", "bot", "organic"]:
        x = d[d.group == g].set_index("company").reindex(COMPANIES)
        fig.add_bar(name=GROUP_LABEL[g], x=[f"{COMPANY_NAME[c]} ({c})" for c in COMPANIES], y=x.r, marker_color=GROUP_COLOR[g],
                    text=[f"{v:+.2f}{'*' if p < .05 else ''}" for v, p in zip(x.r, x.p)], textposition="outside",
                    customdata=np.stack([x.n, x.p], axis=-1),
                    hovertemplate="%{x}<br>r = %{y:+.3f}  (n = %{customdata[0]:.0f}, p = %{customdata[1]:.3g})<extra>" + GROUP_LABEL[g] + "</extra>")
    if method == "A_sig_days_direction":
        fig.add_trace(go.Scatter(name="Original paper (all tweets)", x=[f"{COMPANY_NAME[c]} ({c})" for c in COMPANIES],
                                 y=[PAPER_R[c] for c in COMPANIES], mode="markers",
                                 marker=dict(symbol="diamond", size=12, color=INK, line=dict(color="white", width=1)),
                                 hovertemplate="Paper: r = %{y:.2f}<extra></extra>"))
    fig.add_hline(y=0.3, line_dash="dot", line_color=INK_3)
    fig.add_trace(go.Scatter(name="0.3 = paper's threshold for a meaningful relationship", x=[None], y=[None], mode="lines",
                             line=dict(color=INK_3, dash="dot")))
    fig.update_layout(barmode="group", yaxis_title="Pearson correlation (r)", yaxis_range=[-0.15, 0.6])
    style_fig(fig, f"Sentiment vs abnormal returns — {MEASURES[measure]}", 470)
    note = ("* = statistically significant (p < 0.05). "
            + ("The diamonds are the original paper's tweet correlations. Bot tweets sit near zero and dilute the all-tweets bars; organic tweets alone correlate clearly more strongly, and reach the paper's 0.3 line for some companies and models. "
               if method == "A_sig_days_direction" else "")
            + ("With the previous day's sentiment, nearly everything falls to about zero: sentiment reacts to price moves rather than predicting them. "
               if method == "C_prev_day_sentiment" else "")
            + "Correlation, not causation.")
    return fig, note


@app.callback(Output("t2-chart", "figure"), Input("t2-company", "value"), Input("t2-dates", "start_date"), Input("t2-dates", "end_date"))
def update_tab2(company, start, end):
    a = ar[(ar.company == company) & (ar.date >= start) & (ar.date <= end)]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.06)
    for flag, label, color in [(False, "Other days (RAR)", "#c9c8c3"), (True, "Significant days (RAR)", INK_2)]:
        d = a[a.RAR_sig == flag]
        fig.add_scatter(x=d.date, y=d.RAR, name=label, mode="markers", marker=dict(size=5, color=color),
                        hovertemplate="%{x|%b %d, %Y}<br>RAR %{y:.2%}<extra>" + label + "</extra>", row=1, col=1)
    for g in ["bot", "organic"]:
        s = sent[(sent.company == company) & (sent.author_class == g) & (sent.tweets >= 5)].set_index("date").mean_sent
        s = s.reindex(pd.date_range(start, end)).rolling(14, min_periods=3).mean().dropna()
        fig.add_scatter(x=s.index, y=s.values, name=GROUP_LABEL[g] + " sentiment", line=dict(color=GROUP_COLOR[g], width=2),
                        hovertemplate="%{x|%b %d, %Y}<br>%{y:+.2f}<extra>" + GROUP_LABEL[g] + "</extra>", row=2, col=1)
    fig.update_yaxes(title_text="RAR", tickformat=".0%", row=1, col=1)
    fig.update_yaxes(title_text="Sentiment (14-day avg)", row=2, col=1)
    style_fig(fig, f"{COMPANY_NAME[company]} — abnormal returns and sentiment", 640)
    fig.update_layout(hovermode="x unified", showlegend=True)
    return fig


@app.callback(Output("t4-chart", "figure"), Input("t4-company", "value"))
def update_tab4(company):
    d = robust[robust.company == company].copy()
    d["setting"] = np.where(d.min_tweets == 20, "bot ≥ " + d.bot_at.astype(str) + ", min 20",
                            "bot ≥ " + d.bot_at.astype(str))
    d["order"] = d.bot_at * 100 + d.min_tweets
    d = d.sort_values("order")
    settings = list(dict.fromkeys(d.setting))
    fig = go.Figure()
    for g in ["bot", "organic"]:
        x = d[d.group == g].set_index("setting").reindex(settings)
        fig.add_bar(name=GROUP_LABEL[g], x=settings, y=x.r, marker_color=GROUP_COLOR[g],
                    text=[f"{v:+.2f}" for v in x.r], textposition="outside",
                    hovertemplate="%{x}<br>r = %{y:+.3f}<extra>" + GROUP_LABEL[g] + "</extra>")
    share = d[d.group == "bot"].set_index("setting").reindex(settings).bot_share_of_tweets
    fig.add_trace(go.Scatter(name="Bot share of tweets (right axis)", x=settings, y=share, yaxis="y2", mode="lines+markers",
                             line=dict(color=INK_3, dash="dot"), marker=dict(color=INK_3),
                             hovertemplate="%{x}: %{y:.0%} of tweets flagged<extra></extra>"))
    fig.update_layout(barmode="group", yaxis_title="Pearson correlation (r)", yaxis_range=[-0.15, 0.6],
                      yaxis2=dict(overlaying="y", side="right", tickformat=".0%", range=[0, 0.6], showgrid=False, title="Bot share of tweets"))
    style_fig(fig, f"{COMPANY_NAME[company]} — correlation at different bot thresholds", 470)
    return fig


if __name__ == "__main__":
    app.run(debug=False, port=8060)
