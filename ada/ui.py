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

from .aggregation import build_trend, driver_frame, heatmap_frame, segment_frame
from .anomalies import detect_anomalies
from .autovis import fold_small_series, recommend_chart
from .business_insights import BusinessBrief
from .forecasting import build_forecast, describe_backtest
from .formatting import format_number, format_period, group_label
from .metrics import metric_for
from .nlq import QueryAnswer
from .pipeline import analyze_selection
from .schema import ColumnRoles, is_design_name, is_entity_name
from .selection import comparison_frame, selection_from_event, selection_modes

if TYPE_CHECKING:  # The AI layer is optional; ui must import without it.
    from .ai_insights import AINarrative

ACCENT = "#635BFF"
LIME = "#C7F36B"  # Brand accent for surfaces and text. Too light to be a data mark.

# Data-mark colours, checked on the light chart surface for lightness, chroma,
# colour-vision separation and 3:1 contrast. Assigned in order, never cycled: a
# fifth hue would read as one of these four to colour-blind readers, so series
# past the fourth fold into "Other".
SERIES_COLORS = ("#635BFF", "#0E8F6E", "#B5761B", "#D64A73")
LEAF = "#5C8A1B"  # single-hue magnitude (LIME is only 1.24:1 on the chart surface)
# Up versus down: blue/orange, since deuteranopes cannot separate green/red.
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
            <div class="eyebrow">Exploratory analysis for any table</div>
            <h1>Drop in the messy table.<br><span class="grad">Every number shows its math.</span></h1>
            <p>Lab results, field measurements, survey exports, sales. ADA reads the columns, finds the structure, flags what looks odd, and shows the arithmetic behind every figure. No guessing which column is which, no groupby after groupby, no chart redone because the dates parsed as text.</p>
            <div class="proof-row">
              <span class="proof-pill"><strong>01</strong> No model computes a number</span>
              <span class="proof-pill"><strong>02</strong> No API key, no setup</span>
              <span class="proof-pill"><strong>03</strong> It says when it cannot</span>
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
          <article class="how-card"><div class="how-number">01 · DROP</div><h3>Any table you have</h3><p>Upload CSV or Excel. ADA detects the date, the measure, the segment and the identifiers, cleans common problems, and reports what it changed.</p></article>
          <article class="how-card"><div class="how-number">02 · TRACE</div><h3>Arithmetic, not assertion</h3><p>Trend, breakdown, anomaly flags, correlation with a confidence interval. Every finding shows the calculation that produced it. All local, no network call.</p></article>
          <article class="how-card"><div class="how-number">03 · REFUSE</div><h3>It says when it cannot</h3><p>The forecast will not run on thin history. Ask ADA declines a question it cannot represent. Recommendations are labelled interpretation, never cause.</p></article>
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
          <div class="brief-top"><span class="signal-orb"></span><div class="eyebrow">Headline</div></div>
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
            '<div class="empty-state">No strong signal yet. Open “Tune ADA\'s schema detection” and pick a date, metric and grouping column.</div>',
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
    # Show each action's evidence and the model's watch-outs, so the caveats sit
    # beside the recommendations they qualify.
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


GRAIN_NAMES = {"D": "Day", "W": "Week", "M": "Month", "Q": "Quarter", "Y": "Year"}


def axis_names(roles: ColumnRoles, frequency: str | None = None) -> dict[str, str]:
    """Map the internal Period, Value and Segment columns to user-facing axis labels.

    Value takes the measure name (or "Records"); with frequency "M", Period reads "Month".
    """
    return {
        "Value": roles.measure or "Records",
        "Segment": roles.dimension or "Segment",
        "Period": GRAIN_NAMES.get(frequency or "", "Period"),
    }


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


def _hover_format(values) -> str:
    """Plotly number format for hover labels: whole numbers for large values, decimals for small."""
    finite = pd.Series(values, dtype="float64").abs().dropna()
    return ",.0f" if not finite.empty and finite.median() >= 100 else ",.2f"


def group_names(dataframe: pd.DataFrame, dimension: str | None, values) -> list[str]:
    """Chart labels for groups: "Status 2" for a numeric code, the text itself otherwise.

    The aggregations may hand back a numeric code as text ("2"), so it is read
    back as a number when the grouping column is numeric. A label that isn't a
    number, such as "Other segments", is kept as it is.
    """
    numeric = bool(dimension) and pd.api.types.is_numeric_dtype(dataframe[dimension])
    if not numeric:
        return [str(value) for value in values]
    codes = pd.to_numeric(pd.Series(list(values), dtype=object), errors="coerce")
    return [
        group_label(dimension, float(code)) if pd.notna(code) else str(value)
        for value, code in zip(values, codes, strict=True)
    ]


def _trend_figure(frame: pd.DataFrame, averaged: bool, **options) -> go.Figure:
    """Period values as a filled area for totals, or a plain line for averages."""
    if averaged:
        figure = px.line(frame, x="Period", y="Value", markers=True, color_discrete_sequence=[ACCENT], **options)
        figure.update_traces(line={"width": 3})
        return figure
    figure = px.area(frame, x="Period", y="Value", markers=True, color_discrete_sequence=[ACCENT], **options)
    figure.update_traces(line={"width": 3}, fillcolor="rgba(99,91,255,.11)")
    return figure


def render_dashboard(dataframe: pd.DataFrame, roles: ColumnRoles) -> None:
    metric = metric_for(roles.measure, roles)
    series = build_trend(dataframe, roles)
    trend = series.frame
    segments = segment_frame(dataframe, roles)
    if not segments.empty:
        segments["Segment"] = group_names(dataframe, roles.dimension, segments["Segment"])
    chart_columns = st.columns(2, gap="medium")
    with chart_columns[0]:
        if not trend.empty:
            figure = _trend_figure(
                trend,
                not metric.additive,
                title=f"{roles.measure or 'Records'} over time",
                labels=axis_names(roles, series.frequency),
            )
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
                        hovertemplate=f"%{{x|%b %Y}}: %{{y:{_hover_format(trend['Value'])}}}, outside the expected band<extra>Anomaly</extra>",
                    )
                )
            forecast = build_forecast(trend, floor_at_zero=metric.floors_at_zero)
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
                        hovertemplate=f"%{{x|%b %Y}}: %{{y:{_hover_format(trend['Value'])}}} (forecast)<extra></extra>",
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
                labels=axis_names(roles),
                color_discrete_sequence=[LEAF],
            )
            figure.update_traces(marker_line_width=0, hovertemplate="%{y}: %{x:,.2f}<extra></extra>")
            st.plotly_chart(style_chart(figure), width="stretch", config={"displayModeBar": False})
        else:
            st.markdown('<div class="empty-state">Pick a grouping column to compare groups.</div>', unsafe_allow_html=True)

    movement_columns = st.columns(2, gap="medium")
    with movement_columns[0]:
        drivers = driver_frame(dataframe, roles)
        if not drivers.empty and not metric.additive:
            # Changes in group averages don't add up to the overall change, so
            # they are shown side by side rather than stacked into a total.
            changes = go.Figure(
                go.Bar(
                    x=group_names(dataframe, roles.dimension, drivers["Segment"]),
                    y=list(drivers["Change"]),
                    marker_color=[RISE if change >= 0 else FALL for change in drivers["Change"]],
                    hovertemplate=f"%{{x}}: %{{y:+{_hover_format(drivers['Change'])}}}<extra></extra>",
                )
            )
            changes.update_layout(
                title=f"{roles.measure}: change in average per {roles.dimension}"
            )
            st.plotly_chart(style_chart(changes), width="stretch", config={"displayModeBar": False})
        elif not drivers.empty:
            waterfall = go.Figure(
                go.Waterfall(
                    x=[*group_names(dataframe, roles.dimension, drivers["Segment"]), "Net change"],
                    y=[*drivers["Change"], 0],
                    measure=[*(["relative"] * len(drivers)), "total"],
                    connector={"line": {"color": "#E5E7EB"}},
                    increasing={"marker": {"color": RISE}},
                    decreasing={"marker": {"color": FALL}},
                    totals={"marker": {"color": ACCENT}},
                    hovertemplate=f"%{{x}}: %{{delta:+{_hover_format(drivers['Change'])}}}<extra></extra>",
                )
            )
            waterfall.update_layout(title=f"What moved {roles.measure}: latest vs previous period")
            st.plotly_chart(style_chart(waterfall), width="stretch", config={"displayModeBar": False})
        else:
            st.markdown(
                '<div class="empty-state">A date, a measure and a grouping column together show what changed between the last two periods.</div>',
                unsafe_allow_html=True,
            )

    with movement_columns[1]:
        heat = heatmap_frame(dataframe, roles, frequency=series.frequency)
        if not heat.empty and len(heat.columns) >= 2:
            heatmap = go.Figure(
                go.Heatmap(
                    z=heat.to_numpy(),
                    x=[format_period(period, series.frequency) for period in heat.columns],
                    y=group_names(dataframe, roles.dimension, heat.index),
                    colorscale=[[0, "#F6F7F9"], [0.5, "#B9B1FF"], [1, "#4E43C7"]],
                    hovertemplate=f"%{{y}} · %{{x}}: %{{z:{_hover_format(heat.to_numpy().ravel())}}}<extra></extra>",
                    showscale=False,
                )
            )
            heatmap.update_layout(
                title=f"{roles.measure or 'Records'} intensity by {roles.dimension} and period"
            )
            heatmap.update_yaxes(autorange="reversed")
            # A categorical axis doesn't thin its labels, so show about five;
            # the hover still names every period.
            labels = [format_period(period, series.frequency) for period in heat.columns]
            step = max(1, -(-len(labels) // 5))
            heatmap.update_xaxes(
                tickmode="array",
                tickvals=labels[::step],
                ticktext=labels[::step],
                tickangle=0,
            )
            st.plotly_chart(style_chart(heatmap), width="stretch", config={"displayModeBar": False})
        else:
            st.markdown(
                '<div class="empty-state">A date and a grouping column together show each group over time.</div>',
                unsafe_allow_html=True,
            )

    numeric = [
        column
        for column in roles.numeric
        if dataframe[column].nunique(dropna=True) > 2
        and not is_design_name(column)
        and not is_entity_name(column)
    ]
    lower_columns = st.columns(2, gap="medium")
    with lower_columns[0]:
        if roles.measure:
            figure = px.histogram(
                dataframe,
                x=roles.measure,
                nbins=35,
                title=f"Distribution of {roles.measure}",
                labels={"count": "Records"},
                color_discrete_sequence=["#0E8F6E"],
            )
            figure.update_yaxes(title_text="Records")
            st.plotly_chart(style_chart(figure), width="stretch", config={"displayModeBar": False})
    with lower_columns[1]:
        candidates = [column for column in numeric if column != roles.measure]
        # Plot the measure against the column it correlates with most strongly.
        def strength(column: str) -> float:
            value = dataframe[roles.measure].corr(dataframe[column])
            return 0.0 if pd.isna(value) else abs(float(value))

        partner = max(candidates, key=strength) if roles.measure and candidates else None
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
    if result.chart == "scatter" and len(table.columns) >= 2:
        x, y = table.columns[:2]
        figure = px.scatter(table, x=x, y=y, opacity=0.62, title=f"{y} vs {x}", color_discrete_sequence=[ACCENT])
        return style_chart(figure, height=320)
    if result.chart == "line" and {"Period", "Value"}.issubset(table.columns):
        # The plan carries the aggregation the trend was built with.
        averaged = bool(result.plan.measure) and result.plan.aggregation == "mean"
        figure = _trend_figure(table, averaged, title=result.plan.measure or "Records")
        return style_chart(figure, height=320)
    if result.chart == "line":
        # Groups of a numbered column (timepoint 0, 1, 2...), in the table's order.
        category, value = table.columns[:2]
        figure = px.line(
            table, x=category, y=value, markers=True, title=f"{value} by {category}",
            color_discrete_sequence=[ACCENT],
        )
        figure.update_traces(line={"width": 3})
        return style_chart(figure, height=320)
    category = table.columns[0]
    value = next((name for name in ("Change %", "Change") if name in table.columns), table.columns[1])
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

    ``key`` must be unique within the transcript: two questions can resolve to the
    same plan, and Streamlit rejects identical elements without explicit keys.
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


def render_chat_fallback(suggestions: list[str], hint: str | None = None) -> None:
    """Render the reply for a question that maps to no local calculation.

    ``hint`` is the reason from ``rephrase_hint``: the part of the question ADA
    could not use, or the column the question may have meant.
    """
    st.markdown(
        "I map questions to transparent calculations, and I could not map that one. "
        + (f"{hint} " if hint else "")
        + "Try naming a metric, a segment, or a time scope, for example:"
    )
    st.markdown("\n".join(f"- {suggestion}" for suggestion in suggestions))


# Points the Explore chart draws along one axis.
EXPLORE_POINTS = 400
# A long date axis steps up through these until the whole range fits.
EXPLORE_GRAINS = ("D", "W", "M", "Q", "Y")


def _fit_date_axis(
    dataframe: pd.DataFrame, spec, *, limit: int = EXPLORE_POINTS
) -> tuple[pd.DataFrame, str | None]:
    """Group a date axis by the finest period that fits ``limit`` points.

    Drawing only the latest dates would drop the start of a long series without
    a word, so dates step up through days, weeks, months, quarters and years
    until the whole range fits. Returns the rows with each date moved to its
    period's start, and a caption saying what a point is, or, when even years
    don't fit, how many are shown.
    """
    if not (spec.x and spec.x in dataframe.columns and is_datetime64_any_dtype(dataframe[spec.x])):
        return dataframe, None
    dates = dataframe[spec.x]
    distinct = int(dates.nunique())
    if distinct <= limit:
        return dataframe, None
    for grain in EXPLORE_GRAINS:
        periods = dates.dt.to_period(grain).dt.to_timestamp()
        count = int(periods.nunique())
        if count <= limit:
            break
    name = GRAIN_NAMES[grain].lower()
    if count > limit:
        caption = f"Showing the latest {limit:,} of {count:,} {name}s; earlier ones are not drawn."
    else:
        caption = f"{spec.x} has {distinct:,} dates, so each point is one {name} and the whole range fits."
    return dataframe.assign(**{spec.x: periods}), caption


def _cap(frame: pd.DataFrame, spec, value: str, limit: int) -> pd.DataFrame:
    """Trim to ``limit`` points for drawing.

    A datetime x axis keeps its most recent ``limit`` dates, each with all its
    series; any other chart keeps its largest values.
    """
    if len(frame) <= limit:
        return frame
    if spec.x and spec.x in frame.columns and is_datetime64_any_dtype(frame[spec.x]):
        latest = frame[spec.x].drop_duplicates().nlargest(limit)
        return frame[frame[spec.x].isin(latest)].sort_values(spec.x).reset_index(drop=True)
    return frame.nlargest(limit, value).reset_index(drop=True)


def _explore_frame(
    dataframe: pd.DataFrame, spec, *, limit: int = EXPLORE_POINTS
) -> pd.DataFrame:
    """Aggregate the raw rows into what the recommended chart plots."""
    if spec.form in ("scatter", "histogram"):
        columns = [column for column in (spec.x, spec.y) if column]
        return dataframe[columns].dropna().head(20_000)

    grouping = [column for column in (spec.x, spec.y, spec.color) if column]
    how = "mean" if spec.aggregation == "mean" else "sum"
    # A heatmap and its fallback table (for grids too large to draw) both
    # aggregate a measure across two categories.
    if spec.color and spec.x and spec.y and spec.form in ("heatmap", "table"):
        keys = [spec.y, spec.x]
        grid = dataframe.groupby(keys, dropna=True, observed=True)[spec.color].agg(how).reset_index()
        if spec.form == "table":
            return grid.nlargest(limit, spec.color).reset_index(drop=True)
        return grid

    keys = [column for column in (spec.x, spec.color) if column]
    if not keys:
        return dataframe[grouping].dropna()

    if spec.aggregation == "count" or not spec.y:
        # Use a count column name the frame lacks, so a user column "Records" can't collide.
        value = next(name for name in ("Records", "Record count", "Rows") if name not in dataframe.columns)
        frame = dataframe.groupby(keys, dropna=True, observed=True).size().reset_index(name=value)
    elif how == "mean" and spec.color:
        # Fold the small series on the raw rows, so "Other" is the mean of its rows
        # rather than an unweighted mean of each series' mean.
        rows = fold_small_series(dataframe[[*keys, spec.y]].assign(__rows=1), spec.color, "__rows")
        frame = rows.groupby(keys, dropna=True, observed=True)[spec.y].mean().reset_index()
        return _cap(frame, spec, spec.y, limit)
    else:
        frame = dataframe.groupby(keys, dropna=True, observed=True)[spec.y].agg(how).reset_index()
        value = spec.y

    if spec.color and spec.color in frame.columns:
        frame = fold_small_series(frame, spec.color, value)
        frame = frame.groupby(keys, dropna=True, observed=True)[value].sum().reset_index()
    return _cap(frame, spec, value, limit)


def _explore_figure(frame: pd.DataFrame, spec) -> go.Figure | None:
    """Draw the form named by ``spec``, or return None for forms it does not plot."""
    # The grouped frame's last column is the plotted value, whatever its name.
    value = spec.y if (spec.y and spec.y in frame.columns) else str(frame.columns[-1])

    if spec.form in ("line", "area"):
        if spec.color:
            figure = px.line(
                frame, x=spec.x, y=value, color=spec.color, markers=True,
                color_discrete_sequence=list(SERIES_COLORS),
            )
            figure.update_traces(line={"width": 2})
        elif spec.form == "line":
            figure = px.line(frame, x=spec.x, y=value, markers=True, color_discrete_sequence=[ACCENT])
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
        # Name the measure the colour encodes, not just the two axes.
        title = f"{spec.color} by {spec.y} and {spec.x}"
    elif spec.form == "histogram":
        title = f"Distribution of {spec.x}"
    else:
        title = f"{value} by {spec.x}" if spec.x else value
    figure.update_layout(title=title)
    if spec.form in ("column", "bar"):
        figure.update_traces(marker_line_width=0)
    return style_chart(figure, height=380)


def render_explore(dataframe: pd.DataFrame, roles: ColumnRoles, brief: BusinessBrief | None = None) -> None:
    """Let the user pick columns, then draw the recommended chart and its rationale."""
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

    spec = recommend_chart(dataframe, chosen, roles=roles)
    if spec.form == "none":
        st.info(spec.rationale)
        return

    rows, axis_note = _fit_date_axis(dataframe, spec)
    frame = _explore_frame(rows, spec)
    event = None
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
            event = _selectable_chart(figure, spec, chosen)

    st.markdown(
        f'<p class="calculation">WHY THIS CHART · {escape(spec.rationale)}</p>',
        unsafe_allow_html=True,
    )
    for note in (*spec.notes, *([axis_note] if axis_note else [])):
        st.caption(note)
    if event:
        render_selection(dataframe, roles, selection_from_event(event, spec, dataframe), chosen, brief)


SELECTION_HINTS = {
    "scatter": "Drag a box around points to analyze just those records, or switch to the lasso in the chart's toolbar.",
    "line": "Drag across the time axis to analyze just that stretch of time.",
    "area": "Drag across the time axis to analyze just that stretch of time.",
    "column": "Click a bar to analyze just those records. Shift-click to add more bars.",
    "bar": "Click a bar to analyze just those records. Shift-click to add more bars.",
    "histogram": "Drag across the axis to analyze just the records in that range.",
}


def _selection_key(chosen: list[str]) -> str:
    # The key changes when the columns change, and when the reader clears the
    # selection, so the chart remounts with nothing selected either way.
    generation = int(st.session_state.get("explore_selection_generation", 0))
    return f"explore_chart_{generation}_{'|'.join(chosen)}"


def _selectable_chart(figure: go.Figure, spec, chosen: list[str]):
    """Draw the chart, and return its selection event when its form supports one."""
    modes = selection_modes(spec)
    if not modes:
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
        return None
    if "box" in modes:
        # Clicks are left to Streamlit's own defaults; a drawn area needs its
        # tool active from the start, since the toolbar is usually hidden.
        figure.update_layout(dragmode="select")
    if spec.form in ("line", "area", "histogram"):
        figure.update_layout(selectdirection="h")
    config = {
        # The mode bar is only how a reader switches between box and lasso, so
        # it is shown when there is a choice to make and nothing else is on it.
        "displayModeBar": "lasso" in modes,
        "displaylogo": False,
        "modeBarButtonsToRemove": [
            "zoom2d", "pan2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d", "toImage",
        ],
    }
    st.caption(SELECTION_HINTS.get(spec.form, ""))
    state = st.plotly_chart(
        figure,
        width="stretch",
        config=config,
        key=_selection_key(chosen),
        on_select="rerun",
        selection_mode=modes,
    )
    return state.get("selection") if state else None


def render_selection(
    dataframe: pd.DataFrame,
    roles: ColumnRoles,
    selection,
    chosen: list[str],
    brief: BusinessBrief | None = None,
) -> None:
    """The slice the reader picked, analyzed, and always set against the whole view."""
    if selection.is_empty:
        return
    result = analyze_selection(dataframe, roles, selection, whole_brief=brief)
    comparison = result.comparison
    adjustment = (
        f'<p class="selection-adjustment">{escape(selection.adjustment)}</p>' if selection.adjustment else ""
    )
    st.markdown(
        f'''<section class="selection-panel"><div class="section-label">The slice you picked</div>
        <p class="selection-description">{escape(comparison.description)}</p>{adjustment}
        <p class="calculation">This is a filter you chose on the chart, not a grouping ADA found. Every figure below sits beside the same figure for the whole view.</p></section>''',
        unsafe_allow_html=True,
    )
    if st.button("Clear selection", key=f"clear_{_selection_key(chosen)}"):
        st.session_state.explore_selection_generation = (
            int(st.session_state.get("explore_selection_generation", 0)) + 1
        )
        st.rerun()

    st.dataframe(comparison_frame(comparison), hide_index=True, width="stretch")
    for note in comparison.notes:
        if note != selection.adjustment:  # already stated beside the description
            st.caption(note)
    if result.evidence:
        st.markdown(
            '<div class="section-label" style="margin-top:1rem">Evidence within your slice, beside the whole view</div>',
            unsafe_allow_html=True,
        )
        render_evidence_pairs(result.evidence)


def render_evidence_pairs(pairs) -> None:
    """Slice evidence cards, each carrying the same check's result for the whole view."""
    cards = []
    for item, whole in pairs[:4]:
        if whole is not None:
            beside = f"WHOLE VIEW · {escape(whole.value)} · {escape(whole.statement)}"
        else:
            beside = "WHOLE VIEW · not among the evidence for the whole view"
        cards.append(
            f'<article class="evidence {escape(item.tone)}"><div class="evidence-value">{escape(item.value)}</div>'
            f'<h3>{escape(item.title)}</h3><p>{escape(item.statement)}</p>'
            f'<p class="calculation">{beside}</p>'
            f'<p class="calculation">CALC · {escape(item.calculation)}</p></article>'
        )
    st.markdown(f'<section class="evidence-grid">{"".join(cards)}</section>', unsafe_allow_html=True)


def render_chat_rejected() -> None:
    """Render the transcript message for a proposed calculation that was declined."""
    st.markdown("I did not run that calculation. You can ask another question whenever you’re ready.")


def render_footer(*, build: str = "") -> None:
    # The build tag makes a stale or mixed deploy visible on the page itself.
    tag = f' &nbsp;·&nbsp; <span class="build">build {escape(build)}</span>' if build else ""
    st.markdown(
        f"""
        <footer class="footer"><span>ADA · Automated Data Analyst · Built by Sainesh Nakra</span>
        <span><a href="https://sainesh.com/" target="_blank">Portfolio ↗</a> &nbsp;·&nbsp; <a href="https://github.com/saineshnakra/automated-data-analyst" target="_blank">Contribute ↗</a>{tag}</span></footer>
        """,
        unsafe_allow_html=True,
    )
