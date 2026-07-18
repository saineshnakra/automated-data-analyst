"""Presentation components for ADA's Streamlit interface."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ai_insights import AINarrative
from business_insights import BusinessBrief, ColumnRoles, segment_frame, trend_frame
from nlq import QueryAnswer

ACCENT = "#635BFF"
LIME = "#C7F36B"
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
            <h1>Drop a file.<br><span>Get the business story.</span></h1>
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
) -> None:
    chips = "".join(
        f'<span class="role-chip">{escape(label)} · {escape(value)}</span>'
        for label, value in (("Metric", roles.measure), ("Segment", roles.dimension), ("Date", roles.date))
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
    actions = "".join(
        f'<article class="ai-action"><span class="confidence">{escape(item.confidence)} confidence</span>'
        f'<strong>{escape(item.title)}</strong><p>{escape(item.recommendation)}</p></article>'
        for item in narrative.actions
    )
    st.markdown(
        f"""
        <section class="ai-panel">
          <span class="ai-badge">AI STRATEGIC READ · {escape(model)}</span>
          <h2>{escape(narrative.executive_summary)}</h2>
          <p>{escape(narrative.strategic_read)}</p>
          <div class="ai-actions">{actions}</div>
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
        title={"font": {"size": 16, "color": INK}, "x": 0.02},
        hoverlabel={"bgcolor": INK, "font_color": "white"},
        legend={"orientation": "h", "y": 1.08, "x": 0},
    )
    figure.update_xaxes(showgrid=False, linecolor="#E7E9EE", tickfont={"color": MUTED})
    figure.update_yaxes(gridcolor="#EEF0F3", zeroline=False, tickfont={"color": MUTED})
    return figure


def render_dashboard(dataframe: pd.DataFrame, roles: ColumnRoles) -> None:
    trend = trend_frame(dataframe, roles)
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
            st.plotly_chart(style_chart(figure), width="stretch", config={"displayModeBar": False})
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
                color_discrete_sequence=[LIME],
            )
            figure.update_traces(marker_line_width=0, hovertemplate="%{y}: %{x:,.2f}<extra></extra>")
            st.plotly_chart(style_chart(figure), width="stretch", config={"displayModeBar": False})
        else:
            st.markdown('<div class="empty-state">Select a segment column to reveal contribution.</div>', unsafe_allow_html=True)

    numeric = [column for column in roles.numeric if dataframe[column].nunique(dropna=True) > 2]
    lower_columns = st.columns(2, gap="medium")
    with lower_columns[0]:
        if roles.measure:
            figure = px.histogram(
                dataframe,
                x=roles.measure,
                nbins=35,
                title=f"Distribution of {roles.measure}",
                color_discrete_sequence=["#26A17B"],
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
                color_discrete_sequence=[ACCENT, "#26A17B", "#F2B84B", "#EC6F91"],
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
        color_discrete_sequence=[LIME if "Change" not in value else ACCENT],
    )
    figure.update_traces(marker_line_width=0, hovertemplate="%{y}: %{x:,.2f}<extra></extra>")
    return style_chart(figure, height=320)


def render_chat_answer(result: QueryAnswer) -> None:
    """Render one answered question with its table, chart, and calculation."""
    st.markdown(result.answer)
    figure = _chat_answer_figure(result)
    if figure is not None:
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    if result.table is not None and not result.table.empty:
        with st.expander("See the numbers"):
            st.dataframe(result.table, hide_index=True, width="stretch")
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


def render_footer() -> None:
    st.markdown(
        """
        <footer class="footer"><span>ADA · Automated Data Analyst · Built by Sainesh Nakra</span>
        <span><a href="https://sainesh.com/" target="_blank">Portfolio ↗</a> &nbsp;·&nbsp; <a href="https://github.com/saineshnakra/automated-data-analyst" target="_blank">Contribute ↗</a></span></footer>
        """,
        unsafe_allow_html=True,
    )
