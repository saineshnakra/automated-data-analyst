"""Presentation components for ADA's Streamlit interface."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas.api.types import is_datetime64_any_dtype

from aggregation import build_trend, driver_frame, heatmap_frame, segment_frame
from anomalies import detect_anomalies
from autovis import fold_small_series, recommend_chart
from business_insights import BusinessBrief
from forecasting import build_forecast, describe_backtest
from formatting import format_number, format_period
from nlq import QueryAnswer
from schema import ColumnRoles
from chart_summaries import (
    summarize_categories,
    summarize_distribution,
    summarize_heatmap,
    summarize_movement,
    summarize_relationship,
    summarize_trend,
)

if TYPE_CHECKING:  # The AI layer is optional; ui must import without it.
    from ai_insights import AINarrative

ACCENT = "#635BFF"
LIME = "#C7F36B"  # Brand accent for surfaces and text. Too light to be a data mark.

# Data-mark colours, checked against the light chart surface for the lightness
# band, chroma floor, colour-vision separation and 3:1 contrast. Assigned in
# this order and never cycled -- a generated fifth hue reads as one of these
# four to a colour-blind reader, so series past the fourth fold into "Other".
SERIES_COLORS = ("#635BFF", "#0E8F6E", "#B5761B", "#D64A73")
LEAF = "#5C8A1B"  # single-hue magnitude, replacing LIME which sat at 1.24:1
OTHER_GRAY = "#98A2B3"
# Polarity, for up-versus-down. Green/red is the one pair a deuteranope cannot
# read -- the previous #26A17B/#E35D6A sat at 5.4 separation against a 8 floor.
RISE, FALL = "#1B6FB5", "#B5761B"
INK = "#101114"
MUTED = "#667085"


def inject_styles() -> None:
    stylesheet = Path(__file__).with_name("assets").joinpath("styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{stylesheet}</style>", unsafe_allow_html=True)


def render_nav() -> None:
    st.markdown(
        """
        <nav class="ada-nav" aria-label="Primary navigation">
          <div class="ada-brand">
            <div class="ada-mark">A</div>
            <div><div class="ada-wordmark">ADA</div><div class="ada-nav-note">Automated Data Analyst</div></div>
          </div>
          <div class="nav-actions">
            <a class="nav-link" href="https://github.com/saineshnakra/automated-data-analyst" target="_blank">GitHub ↗</a>
            <span class="trust-chip"><span class="trust-dot"></span>Deterministic core · AI optional</span>
          </div>
        </nav>
        """,
        unsafe_allow_html=True,
    )


def render_landing() -> None:
    st.markdown(
        """
        <section class="hero">
          <div class="hero-copy">
            <div class="eyebrow">Zero-config business intelligence</div>
            <h1>Drop a file.<br><span class="grad">Get the business story.</span></h1>
            <p>ADA turns CSV and Excel data into an executive dashboard, explains what changed, identifies the driver, and recommends the next move—without making you configure a BI tool.</p>
            <div class="proof-row">
              <span class="proof-pill"><strong>01</strong> Automatic schema detection</span>
              <span class="proof-pill"><strong>02</strong> Traceable calculations</span>
              <span class="proof-pill"><strong>03</strong> Decision-ready actions</span>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_how_it_works() -> None:
    st.markdown(
        """
        <section class="how-grid">
          <article class="how-card"><div class="how-number">01 · DROP</div><h3>Any business file</h3><p>Upload CSV or Excel. ADA cleans common issues and detects the metric, date, segment, and identifiers.</p></article>
          <article class="how-card"><div class="how-number">02 · TRACE</div><h3>Facts before opinions</h3><p>Every trend, driver, concentration signal, and exception exposes its calculation.</p></article>
          <article class="how-card"><div class="how-number">03 · DECIDE</div><h3>Actions, not chart clutter</h3><p>Interpretation stays separate from evidence, with the highest-value investigation first.</p></article>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_section_heading(kicker: str, title: str, description: str) -> None:
    st.markdown(
        f'<header class="section-heading"><div class="section-kicker">{escape(kicker)}</div>'
        f'<h2>{escape(title)}</h2><p>{escape(description)}</p></header>',
        unsafe_allow_html=True,
    )


def render_dataset_bar(
    source_name: str,
    dataframe: pd.DataFrame,
    roles: ColumnRoles,
    focus: str | None = None,
) -> None:
    chips = "".join(
        f'<span class="role-chip">{escape(label)} · {escape(value)}</span>'
        for label, value in (
            ("Focus", focus),
            ("Metric", roles.measure),
            ("Segment", roles.dimension),
            ("Date", roles.date),
        )
        if value
    )
    st.markdown(
        f'<div class="dataset-bar"><div class="dataset-main"><span class="dataset-name">{escape(source_name)}</span>'
        f'{chips}</div><div>{len(dataframe):,} rows · {len(dataframe.columns):,} columns · local calculations</div></div>',
        unsafe_allow_html=True,
    )


def render_brief(brief: BusinessBrief) -> None:
    st.markdown(
        f"""
        <section class="brief">
          <div class="brief-top"><span class="signal-orb"></span><div class="eyebrow">Executive signal</div></div>
          <h1>{escape(brief.headline)}</h1>
          <p>{escape(brief.summary)}</p>
          <div class="brief-trust">CALCULATED FROM THE FILE · INTERPRETATION SHOWN SEPARATELY · NO CAUSAL CLAIMS</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_kpis(brief: BusinessBrief) -> None:
    cards = [
        f'<article class="kpi-card {escape(item.tone)}"><div class="kpi-label">{escape(item.label)}</div>'
        f'<div class="kpi-value">{escape(item.value)}</div><div class="kpi-context">{escape(item.context)}</div></article>'
        for item in brief.kpis
    ]
    st.markdown(f'<section class="kpi-grid">{"".join(cards)}</section>', unsafe_allow_html=True)


def render_recommendations(brief: BusinessBrief) -> None:
    for item in brief.recommendations:
        st.markdown(
            f"""
            <article class="recommendation">
              <div class="recommendation-top"><span class="priority">{escape(item.priority)}</span><h3>{escape(item.title)}</h3></div>
              <p>{escape(item.action)}</p>
              <p class="why"><strong>Evidence:</strong> {escape(item.rationale)}</p>
            </article>
            """,
            unsafe_allow_html=True,
        )


def render_evidence(brief: BusinessBrief, *, limit: int | None = None) -> None:
    evidence = brief.evidence[:limit] if limit else brief.evidence
    if not evidence:
        st.markdown(
            '<div class="empty-state">No strong signal yet. Open “Tune detection” and select a date, metric, and segment.</div>',
            unsafe_allow_html=True,
        )
        return
    cards = [
        f'<article class="evidence {escape(item.tone)}"><div class="evidence-value">{escape(item.value)}</div>'
        f'<h3>{escape(item.title)}</h3><p>{escape(item.statement)}</p>'
        f'<p class="calculation">CALC · {escape(item.calculation)}</p></article>'
        for item in evidence
    ]
    st.markdown(f'<section class="evidence-grid">{"".join(cards)}</section>', unsafe_allow_html=True)


def render_ai_narrative(narrative: AINarrative, *, model: str) -> None:
    # Each action names the evidence it rests on, and the model's own caveats
    # are shown rather than dropped: both were in the typed response and
    # neither reached the page, which left the confident parts of the read
    # standing without the parts that qualified them.
    actions = "".join(
        f'<article class="ai-action"><span class="confidence">{escape(item.confidence)} confidence</span>'
        f'<strong>{escape(item.title)}</strong><p>{escape(item.recommendation)}</p>'
        f'<p class="ai-evidence"><em>Evidence:</em> {escape(item.evidence)}</p></article>'
        for item in narrative.actions
    )
    watchouts = "".join(f"<li>{escape(item)}</li>" for item in narrative.watchouts)
    watchout_block = (
        f'<div class="ai-watchouts"><strong>Watch-outs</strong><ul>{watchouts}</ul></div>' if watchouts else ""
    )
    st.markdown(
        f"""
        <section class="ai-panel">
          <span class="ai-badge">AI STRATEGIC READ · {escape(model)}</span>
          <h2>{escape(narrative.executive_summary)}</h2>
          <p>{escape(narrative.strategic_read)}</p>
          <div class="ai-actions">{actions}</div>
          {watchout_block}
          <p class="ai-caveat">Written by a model from the computed evidence above. It can misread that evidence; the calculations are authoritative, the read is not.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def style_chart(figure: go.Figure, *, height: int = 390) -> go.Figure:
    figure.update_layout(
        height=height,
        margin={"l": 22, "r": 18, "t": 58, "b": 22},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif", "color": INK, "size": 12},
        title={"font": {"size": 16, "color": INK, "family": "Space Grotesk, Inter, sans-serif"}, "x": 0.02},
        hoverlabel={"bgcolor": INK, "font_color": "white"},
        legend={"orientation": "h", "y": 1.08, "x": 0},
    )
    figure.update_xaxes(showgrid=False, linecolor="#E7E9EE", tickfont={"color": MUTED})
    figure.update_yaxes(gridcolor="#EEF0F3", zeroline=False, tickfont={"color": MUTED})
    return figure


def render_dashboard(dataframe: pd.DataFrame, roles: ColumnRoles) -> None:
    series = build_trend(dataframe, roles)
    trend = series.frame
    segments = segment_frame(dataframe, roles)
    chart_columns = st.columns(2, gap="medium")
    with chart_columns[0]:
        if not trend.empty:
            figure = px.area(
                trend,
                x="Period",
                y="Value",
                markers=True,
                title=f"{roles.measure or 'Records'} over time",
                color_discrete_sequence=[ACCENT],
            )
            figure.update_traces(line={"width": 3}, fillcolor="rgba(99,91,255,.11)")
            anomalies = detect_anomalies(trend)
            if anomalies:
                figure.add_trace(
                    go.Scatter(
                        x=[anomaly.period for anomaly in anomalies],
                        y=[anomaly.value for anomaly in anomalies],
                        mode="markers",
                        name="Anomaly",
                        marker={
                            "symbol": "diamond",
                            "size": 11,
                            "color": "#E35D6A",
                            "line": {"width": 2, "color": "white"},
                        },
                        hovertemplate="%{x|%b %Y}: %{y:,.0f} — outside the expected band<extra>Anomaly</extra>",
                    )
                )
            forecast = build_forecast(trend)
            if forecast:
                figure.add_trace(
                    go.Scatter(
                        x=[*forecast.periods, *reversed(forecast.periods)],
                        y=[*forecast.upper, *reversed(forecast.lower)],
                        mode="lines",
                        fill="toself",
                        fillcolor="rgba(139,92,246,.09)",
                        line={"width": 0},
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
                figure.add_trace(
                    go.Scatter(
                        x=[trend.iloc[-1]["Period"], *forecast.periods],
                        y=[float(trend.iloc[-1]["Value"]), *forecast.values],
                        mode="lines",
                        name="Forecast",
                        line={"width": 2.5, "dash": "dash", "color": "#8B5CF6"},
                        hovertemplate="%{x|%b %Y}: %{y:,.0f} — baseline forecast<extra></extra>",
                    )
                )
            st.plotly_chart(style_chart(figure), width="stretch", config={"displayModeBar": False})
            if forecast:
                st.caption(
                    f"Baseline forecast: {forecast.method} · {describe_backtest(forecast.backtest)}."
                )
            for note in series.notes:
                st.caption(f"Timeline: {note}")
        else:
            st.markdown('<div class="empty-state">Select a date column to reveal movement over time.</div>', unsafe_allow_html=True)

    with chart_columns[1]:
        if not segments.empty:
            figure = px.bar(
                segments.sort_values("Value"),
                x="Value",
                y="Segment",
                orientation="h",
                title=f"{roles.measure or 'Records'} by {roles.dimension}",
                color_discrete_sequence=[LEAF],
            )
            figure.update_traces(marker_line_width=0, hovertemplate="%{y}: %{x:,.2f}<extra></extra>")
            st.plotly_chart(style_chart(figure), width="stretch", config={"displayModeBar": False})
        else:
            st.markdown('<div class="empty-state">Select a segment column to reveal contribution.</div>', unsafe_allow_html=True)

    movement_columns = st.columns(2, gap="medium")
    with movement_columns[0]:
        drivers = driver_frame(dataframe, roles)
        if not drivers.empty:
            waterfall = go.Figure(
                go.Waterfall(
                    x=[*drivers["Segment"], "Net change"],
                    y=[*drivers["Change"], 0],
                    measure=[*(["relative"] * len(drivers)), "total"],
                    connector={"line": {"color": "#E5E7EB"}},
                    increasing={"marker": {"color": RISE}},
                    decreasing={"marker": {"color": FALL}},
                    totals={"marker": {"color": ACCENT}},
                    hovertemplate="%{x}: %{delta:+,.0f}<extra></extra>",
                )
            )
            waterfall.update_layout(title=f"What moved {roles.measure} — latest vs previous period")
            st.plotly_chart(style_chart(waterfall), width="stretch", config={"displayModeBar": False})
        else:
            st.markdown(
                '<div class="empty-state">A date, metric, and segment together unlock the movement waterfall.</div>',
                unsafe_allow_html=True,
            )

    with movement_columns[1]:
        heat = heatmap_frame(dataframe, roles, frequency=series.frequency)
        if not heat.empty and len(heat.columns) >= 2:
            heatmap = go.Figure(
                go.Heatmap(
                    z=heat.to_numpy(),
                    x=[format_period(period, series.frequency) for period in heat.columns],
                    y=[str(segment) for segment in heat.index],
                    colorscale=[[0, "#F6F7F9"], [0.5, "#B9B1FF"], [1, "#4E43C7"]],
                    hovertemplate="%{y} · %{x}: %{z:,.0f}<extra></extra>",
                    showscale=False,
                )
            )
            heatmap.update_layout(
                title=f"{roles.measure or 'Records'} intensity by {roles.dimension} and period"
            )
            heatmap.update_yaxes(autorange="reversed")
            st.plotly_chart(style_chart(heatmap), width="stretch", config={"displayModeBar": False})
        else:
            st.markdown(
                '<div class="empty-state">A date and a segment together unlock the intensity heatmap.</div>',
                unsafe_allow_html=True,
            )

    numeric = [column for column in roles.numeric if dataframe[column].nunique(dropna=True) > 2]
    lower_columns = st.columns(2, gap="medium")
    with lower_columns[0]:
        if roles.measure:
            figure = px.histogram(
                dataframe,
                x=roles.measure,
                nbins=35,
                title=f"Distribution of {roles.measure}",
                color_discrete_sequence=["#0E8F6E"],
            )
            st.plotly_chart(style_chart(figure), width="stretch", config={"displayModeBar": False})
    with lower_columns[1]:
        partner = next((column for column in numeric if column != roles.measure), None)
        if roles.measure and partner:
            figure = px.scatter(
                dataframe,
                x=partner,
                y=roles.measure,
                color=roles.dimension if roles.dimension else None,
                opacity=0.62,
                title=f"{roles.measure} vs {partner}",
                color_discrete_sequence=list(SERIES_COLORS),
            )
            st.plotly_chart(style_chart(figure), width="stretch", config={"displayModeBar": False})


def _chat_answer_figure(result: QueryAnswer) -> go.Figure | None:
    table = result.table
    if table is None or table.empty or result.chart is None:
        return None
    if result.chart == "line" and {"Period", "Value"}.issubset(table.columns):
        figure = px.area(
            table,
            x="Period",
            y="Value",
            markers=True,
            title=result.plan.measure or "Records",
            color_discrete_sequence=[ACCENT],
        )
        figure.update_traces(line={"width": 3}, fillcolor="rgba(99,91,255,.11)")
        return style_chart(figure, height=320)
    category = table.columns[0]
    value = "Change %" if "Change %" in table.columns else table.columns[1]
    figure = px.bar(
        table.sort_values(value),
        x=value,
        y=category,
        orientation="h",
        title=f"{value} by {category}",
        color_discrete_sequence=[LEAF if "Change" not in value else ACCENT],
    )
    figure.update_traces(marker_line_width=0, hovertemplate="%{y}: %{x:,.2f}<extra></extra>")
    return style_chart(figure, height=320)


def render_chat_answer(result: QueryAnswer, *, key: str) -> None:
    """Render one answered question with its table, chart, and calculation.

    ``key`` identifies this answer within the transcript. Two questions can
    resolve to the same plan and therefore render an identical chart and
    table, and Streamlit refuses to give two identical elements the same
    auto-generated id, so each answer names its own.
    """
    if result.plan.source == "ai":
        st.markdown(
            '<span class="ai-plan-badge">AI-planned · executed locally · schema only</span>',
            unsafe_allow_html=True,
        )
    st.markdown(result.answer)
    figure = _chat_answer_figure(result)
    if figure is not None:
        st.plotly_chart(
            figure,
            width="stretch",
            config={"displayModeBar": False},
            key=f"chat_chart_{key}",
        )
    if result.table is not None and not result.table.empty:
        with st.expander("See the numbers"):
            st.dataframe(result.table, hide_index=True, width="stretch", key=f"chat_table_{key}")
    st.markdown(
        f'<p class="calculation chat-calc">CALC · {escape(result.calculation)}</p>',
        unsafe_allow_html=True,
    )


def render_chat_fallback(suggestions: list[str]) -> None:
    """Shown when a question cannot be mapped to a local calculation."""
    st.markdown(
        "I map questions to transparent calculations, and I could not map that one. "
        "Try naming a metric, a segment, or a time scope — for example:"
    )
    st.markdown("\n".join(f"- {suggestion}" for suggestion in suggestions))


def _cap(frame: pd.DataFrame, spec, value: str, limit: int) -> pd.DataFrame:
    """Trim to a drawable size by dropping the least interesting rows.

    Taking the first N groups drops the newest periods off a trend and the
    biggest categories off a ranking -- exactly the rows the reader opened the
    chart for. Time keeps its most recent end; anything else keeps its largest.
    """
    if len(frame) <= limit:
        return frame
    if spec.x and spec.x in frame.columns and is_datetime64_any_dtype(frame[spec.x]):
        return frame.nlargest(limit, spec.x).sort_values(spec.x).reset_index(drop=True)
    return frame.nlargest(limit, value).reset_index(drop=True)


def _explore_frame(
    dataframe: pd.DataFrame, spec, *, limit: int = 400
) -> pd.DataFrame:
    """Aggregate the raw rows into what the recommended chart plots."""
    if spec.form in ("scatter", "histogram"):
        columns = [column for column in (spec.x, spec.y) if column]
        return dataframe[columns].dropna().head(20_000)

    grouping = [column for column in (spec.x, spec.y, spec.color) if column]
    # A heatmap, and the table it falls back to when the grid is too large,
    # are both "measure across two categories".
    if spec.color and spec.x and spec.y and spec.form in ("heatmap", "table"):
        keys = [spec.y, spec.x]
        grid = dataframe.groupby(keys, dropna=True, observed=True)[spec.color].sum().reset_index()
        if spec.form == "table":
            return grid.nlargest(limit, spec.color).reset_index(drop=True)
        return grid

    keys = [column for column in (spec.x, spec.color) if column]
    if not keys:
        return dataframe[grouping].dropna()

    if spec.aggregation == "count" or not spec.y:
        # A column the file calls "Records" collides with the count column
        # built here, so the count takes a name the frame is not using.
        value = next(name for name in ("Records", "Record count", "Rows") if name not in dataframe.columns)
        frame = dataframe.groupby(keys, dropna=True, observed=True).size().reset_index(name=value)
    else:
        frame = dataframe.groupby(keys, dropna=True, observed=True)[spec.y].sum().reset_index()
        value = spec.y

    if spec.color and spec.color in frame.columns:
        frame = fold_small_series(frame, spec.color, value)
        frame = frame.groupby(keys, dropna=True, observed=True)[value].sum().reset_index()
    return _cap(frame, spec, value, limit)


def _explore_figure(frame: pd.DataFrame, spec) -> go.Figure | None:
    """Draw exactly the form the recommendation asked for."""
    # The grouped frame's last column is always the value being plotted,
    # whatever it had to be called.
    value = spec.y if (spec.y and spec.y in frame.columns) else str(frame.columns[-1])

    if spec.form in ("line", "area"):
        if spec.color:
            figure = px.line(
                frame, x=spec.x, y=value, color=spec.color, markers=True,
                color_discrete_sequence=list(SERIES_COLORS),
            )
            figure.update_traces(line={"width": 2})
        else:
            figure = px.area(frame, x=spec.x, y=value, color_discrete_sequence=[ACCENT])
            figure.update_traces(line={"width": 2}, fillcolor="rgba(99,91,255,.11)")
    elif spec.form == "column":
        figure = px.bar(frame, x=spec.x, y=value, color_discrete_sequence=[LEAF])
    elif spec.form == "bar":
        figure = px.bar(
            frame.sort_values(value), x=value, y=spec.x, orientation="h",
            color_discrete_sequence=[LEAF],
        )
    elif spec.form == "histogram":
        figure = px.histogram(frame, x=spec.x, nbins=35, color_discrete_sequence=[ACCENT])
    elif spec.form == "scatter":
        figure = px.scatter(frame, x=spec.x, y=spec.y, color_discrete_sequence=[ACCENT])
        figure.update_traces(marker={"size": 8, "opacity": 0.7})
    elif spec.form == "heatmap":
        grid = frame.pivot_table(index=spec.y, columns=spec.x, values=spec.color, aggfunc="sum")
        figure = px.imshow(grid, color_continuous_scale="Purples", aspect="auto")
    else:
        return None

    if spec.form == "heatmap":
        # The colour is the measurement; naming only the two axes describes the
        # grid and not what is in it.
        title = f"{spec.color} by {spec.y} and {spec.x}"
    elif spec.form == "histogram":
        title = f"Distribution of {spec.x}"
    else:
        title = f"{value} by {spec.x}" if spec.x else value
    figure.update_layout(title=title)
    if spec.form in ("column", "bar"):
        figure.update_traces(marker_line_width=0)
    return style_chart(figure, height=380)


def render_explore(dataframe: pd.DataFrame, roles: ColumnRoles) -> None:
    """Pick any columns; ADA picks the chart and says why it picked it."""
    render_section_heading(
        "Explore",
        "Chart any columns you like",
        "Choose columns and ADA works out which chart form the data calls for. "
        "The reasoning is printed under every chart, the same way calculations are.",
    )

    default = [column for column in (roles.date, roles.measure) if column][:2]
    chosen = st.multiselect(
        "Columns to chart",
        list(dataframe.columns),
        default=default,
        help="A date and a measure make a trend. A category and a measure make a comparison. "
        "Two measures make a relationship.",
    )
    if not chosen:
        st.markdown(
            '<div class="empty-state">Pick one or more columns above and ADA will '
            "choose a chart for them.</div>",
            unsafe_allow_html=True,
        )
        return

    spec = recommend_chart(dataframe, chosen)
    if spec.form == "none":
        st.info(spec.rationale)
        return

    frame = _explore_frame(dataframe, spec)
    if spec.form == "stat":
        present = dataframe[spec.y].dropna() if spec.y else pd.Series(dtype="float64")
        if present.empty:
            st.markdown(
                f'<div class="empty-state">{escape(str(spec.y))} has no values in this '
                "view, so there is nothing to chart.</div>",
                unsafe_allow_html=True,
            )
            return
        st.metric(spec.y or "Value", format_number(float(present.iloc[0]), spec.y))
    elif spec.form == "table":
        st.dataframe(frame, hide_index=True, width="stretch")
    else:
        figure = _explore_figure(frame, spec)
        if figure is not None:
            st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

    st.markdown(
        f'<p class="calculation">WHY THIS CHART · {escape(spec.rationale)}</p>',
        unsafe_allow_html=True,
    )
    for note in spec.notes:
        st.caption(note)


def render_chat_rejected() -> None:
    """Render the transcript message for a proposed calculation that was declined."""
    st.markdown("I did not run that calculation. You can ask another question whenever you’re ready.")


def render_footer(*, build: str = "") -> None:
    # The build tag is what lets a mixed or stale deployment be recognised
    # from the page itself, instead of from a redacted traceback.
    tag = f' &nbsp;·&nbsp; <span class="build">build {escape(build)}</span>' if build else ""
    st.markdown(
        f"""
        <footer class="footer"><span>ADA · Automated Data Analyst · Built by Sainesh Nakra</span>
        <span><a href="https://sainesh.com/" target="_blank">Portfolio ↗</a> &nbsp;·&nbsp; <a href="https://github.com/saineshnakra/automated-data-analyst" target="_blank">Contribute ↗</a>{tag}</span></footer>
        """,
        unsafe_allow_html=True,
    )
