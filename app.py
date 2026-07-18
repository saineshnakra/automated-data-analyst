"""ADA: upload a business file and receive a decision-ready dashboard."""

from __future__ import annotations

from html import escape

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis import clean_dataframe, column_profile
from business_insights import (
    BusinessBrief,
    ColumnRoles,
    analyze_business,
    build_business_report,
    detect_roles,
    segment_frame,
    trend_frame,
)
from demo_data import make_demo_data
from file_io import read_tabular_file

APP_TITLE = "ADA"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_ANALYSIS_ROWS = 250_000
ACCENT = "#6D5EF7"
LIME = "#B8F24A"
INK = "#111318"
MUTED = "#6B7280"

st.set_page_config(
    page_title="ADA — a dashboard that explains itself",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
          :root {
            --ink: #111318;
            --muted: #687080;
            --line: #E7E9EE;
            --paper: #FFFFFF;
            --canvas: #F6F7F9;
            --accent: #6D5EF7;
            --accent-soft: #EFEDFF;
            --lime: #B8F24A;
            --green: #118A5B;
            --red: #C43D4B;
            --amber: #9A6700;
          }

          html, body, [class*="css"] {font-family: Inter, ui-sans-serif, system-ui, sans-serif;}
          .stApp {background: var(--canvas); color: var(--ink);}
          .block-container {max-width: 1240px; padding-top: 1.25rem; padding-bottom: 5rem;}
          header[data-testid="stHeader"] {background: transparent;}
          footer {visibility: hidden;}

          .ada-nav {
            display: flex; align-items: center; justify-content: space-between;
            padding: .35rem 0 1.35rem; border-bottom: 1px solid var(--line);
          }
          .ada-brand {display: flex; align-items: center; gap: .72rem;}
          .ada-mark {
            display: grid; place-items: center; width: 2.25rem; height: 2.25rem;
            border-radius: .72rem; background: var(--ink); color: white;
            font-weight: 900; transform: rotate(-3deg);
          }
          .ada-wordmark {font-size: 1.03rem; font-weight: 800; letter-spacing: -.02em;}
          .ada-nav-note {font-size: .8rem; color: var(--muted);}
          .privacy-chip {
            display: inline-flex; align-items: center; gap: .35rem; padding: .42rem .7rem;
            border: 1px solid var(--line); border-radius: 999px; background: white;
            color: #46505f; font-size: .74rem; font-weight: 700;
          }

          .landing {padding: 4.2rem 0 2rem; max-width: 900px;}
          .eyebrow {
            color: var(--accent); font-size: .72rem; font-weight: 850;
            letter-spacing: .14em; text-transform: uppercase;
          }
          .landing h1 {
            max-width: 850px; margin: .75rem 0 1rem; color: var(--ink);
            font-size: clamp(3rem, 7vw, 5.8rem); line-height: .98;
            letter-spacing: -.065em; font-weight: 850;
          }
          .landing p {max-width: 700px; color: #596171; font-size: 1.2rem; line-height: 1.65;}
          .source-shell {
            margin: 1rem 0 1.25rem; padding: .8rem 1rem; border: 1px solid var(--line);
            border-radius: 1rem; background: rgba(255,255,255,.72);
          }
          .how-grid {display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 2rem;}
          .how-card {
            min-height: 150px; padding: 1.25rem; border: 1px solid var(--line);
            border-radius: 1.2rem; background: white;
          }
          .how-number {color: var(--accent); font-size: .72rem; font-weight: 850; letter-spacing: .1em;}
          .how-card h3 {margin: 1.3rem 0 .45rem; font-size: 1.05rem;}
          .how-card p {margin: 0; color: var(--muted); font-size: .9rem; line-height: 1.55;}

          [data-testid="stFileUploader"] {
            background: white; border: 1px solid var(--line); border-radius: 1.2rem;
            padding: .7rem 1rem .2rem;
          }
          [data-testid="stFileUploaderDropzone"] {border: 1.5px dashed #B9BDCA; border-radius: .9rem;}
          [data-testid="stFileUploaderDropzone"] button {background: var(--ink); color: white; border: 0;}

          .dataset-bar {
            display: flex; align-items: center; justify-content: space-between; gap: 1rem;
            margin: 1.15rem 0 1rem; color: var(--muted); font-size: .82rem;
          }
          .dataset-name {color: var(--ink); font-weight: 750;}
          .role-chip {
            display: inline-block; margin-left: .35rem; padding: .27rem .5rem;
            background: var(--accent-soft); border-radius: .45rem; color: #4E43C7;
            font-size: .7rem; font-weight: 750;
          }

          .brief {
            position: relative; overflow: hidden; margin: .7rem 0 1.1rem; padding: 2rem;
            border-radius: 1.5rem; background: var(--ink); color: white;
            box-shadow: 0 16px 50px rgba(17,19,24,.13);
          }
          .brief:after {
            content: ""; position: absolute; width: 280px; height: 280px; right: -90px; top: -120px;
            background: radial-gradient(circle, rgba(109,94,247,.85), rgba(109,94,247,0) 70%);
          }
          .brief .eyebrow {color: var(--lime);}
          .brief h1 {
            position: relative; z-index: 1; max-width: 920px; margin: .65rem 0 .7rem;
            font-size: clamp(2rem, 4vw, 3.5rem); line-height: 1.06;
            letter-spacing: -.045em; font-weight: 820;
          }
          .brief p {position: relative; z-index: 1; max-width: 920px; margin: 0; color: #C7CBD4; line-height: 1.65;}
          .brief-trust {margin-top: 1.25rem; color: #9299A8; font-size: .72rem; font-weight: 700;}

          .kpi-grid {display: grid; grid-template-columns: repeat(4, 1fr); gap: .85rem; margin: 1rem 0 1.4rem;}
          .kpi-card {padding: 1.15rem; border: 1px solid var(--line); border-radius: 1.1rem; background: white;}
          .kpi-label {color: var(--muted); font-size: .72rem; font-weight: 750; text-transform: uppercase; letter-spacing: .06em;}
          .kpi-value {margin: .48rem 0 .32rem; color: var(--ink); font-size: 1.75rem; font-weight: 820; letter-spacing: -.04em;}
          .kpi-context {color: #89909D; font-size: .72rem; line-height: 1.35;}
          .kpi-card.positive .kpi-value {color: var(--green);}
          .kpi-card.negative .kpi-value {color: var(--red);}

          .section-label {margin: .7rem 0 .8rem; font-size: .75rem; font-weight: 850; color: var(--muted); text-transform: uppercase; letter-spacing: .12em;}
          .recommendation {
            margin-bottom: .72rem; padding: 1rem 1.05rem; border: 1px solid var(--line);
            border-radius: 1rem; background: white;
          }
          .recommendation-top {display: flex; align-items: center; gap: .6rem;}
          .priority {
            padding: .25rem .48rem; border-radius: .4rem; background: var(--ink);
            color: white; font-size: .64rem; font-weight: 850; text-transform: uppercase;
          }
          .recommendation h3 {margin: 0; font-size: 1rem; letter-spacing: -.015em;}
          .recommendation p {margin: .65rem 0 0; color: #596171; font-size: .86rem; line-height: 1.55;}
          .recommendation .why {padding-top: .55rem; border-top: 1px solid #F0F1F4; color: #8A909B; font-size: .72rem;}

          .evidence-grid {display: grid; grid-template-columns: repeat(2, 1fr); gap: .72rem;}
          .evidence {
            min-height: 168px; padding: 1.05rem; border: 1px solid var(--line);
            border-radius: 1rem; background: white;
          }
          .evidence-value {font-size: 1.55rem; font-weight: 850; letter-spacing: -.04em; color: var(--accent);}
          .evidence.positive .evidence-value {color: var(--green);}
          .evidence.negative .evidence-value {color: var(--red);}
          .evidence.warning .evidence-value {color: var(--amber);}
          .evidence h3 {margin: .28rem 0 .5rem; font-size: .88rem;}
          .evidence p {margin: 0; color: #646C7A; font-size: .78rem; line-height: 1.5;}
          .calculation {margin-top: .65rem !important; color: #A0A5AF !important; font-size: .66rem !important;}

          .chart-card {padding: .65rem .75rem .2rem; border: 1px solid var(--line); border-radius: 1.15rem; background: white;}
          .empty-state {padding: 2rem; border: 1px dashed #C8CBD4; border-radius: 1rem; color: var(--muted); background: white;}
          .trust-note {margin-top: 1rem; padding: .85rem 1rem; border-radius: .85rem; background: #F0F8E3; color: #496224; font-size: .77rem;}

          div[data-baseweb="tab-list"] {gap: .35rem; background: transparent;}
          button[data-baseweb="tab"] {height: 2.8rem; border-radius: .75rem; padding: 0 1rem;}
          button[data-baseweb="tab"][aria-selected="true"] {background: white; border: 1px solid var(--line);}
          div[data-testid="stExpander"] {border: 1px solid var(--line); border-radius: .9rem; background: white;}
          [data-testid="stDataFrame"] {border: 1px solid var(--line); border-radius: .8rem; overflow: hidden;}
          .stDownloadButton button {background: var(--ink); color: white; border: 0; border-radius: .75rem; font-weight: 750;}
          .stDownloadButton button:hover {background: #2A2E36; color: white;}

          @media (max-width: 850px) {
            .how-grid, .kpi-grid, .evidence-grid {grid-template-columns: 1fr 1fr;}
            .ada-nav-note {display: none;}
          }
          @media (max-width: 560px) {
            .block-container {padding-left: 1rem; padding-right: 1rem;}
            .landing {padding-top: 2.5rem;}
            .how-grid, .kpi-grid, .evidence-grid {grid-template-columns: 1fr;}
            .dataset-bar {align-items: flex-start; flex-direction: column;}
            .brief {padding: 1.4rem;}
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def read_uploaded_file(contents: bytes, filename: str) -> pd.DataFrame:
    return read_tabular_file(contents, filename)


def render_nav() -> None:
    st.markdown(
        """
        <div class="ada-nav">
          <div class="ada-brand">
            <div class="ada-mark">A</div>
            <div><div class="ada-wordmark">ADA</div><div class="ada-nav-note">Automated Data Analyst</div></div>
          </div>
          <div class="privacy-chip">● Private by design · no API key</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_landing() -> None:
    st.markdown(
        """
        <div class="landing">
          <div class="eyebrow">Decision intelligence in seconds</div>
          <h1>Drop a file.<br>Get the business story.</h1>
          <p>ADA turns a messy spreadsheet into a dashboard, explains what changed, and tells you what to investigate next. No analyst required.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_how_it_works() -> None:
    st.markdown(
        """
        <div class="how-grid">
          <div class="how-card"><div class="how-number">01 · UPLOAD</div><h3>Any business file</h3><p>CSV or Excel. ADA cleans common issues and detects dates, metrics, and segments automatically.</p></div>
          <div class="how-card"><div class="how-number">02 · UNDERSTAND</div><h3>Facts before opinions</h3><p>Every trend, driver, and exception links back to a transparent calculation.</p></div>
          <div class="how-card"><div class="how-number">03 · DECIDE</div><h3>Clear next actions</h3><p>ADA separates what happened from what it recommends you investigate or change.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpis(brief: BusinessBrief) -> None:
    cards = []
    for item in brief.kpis:
        cards.append(
            f'<div class="kpi-card {escape(item.tone)}">'
            f'<div class="kpi-label">{escape(item.label)}</div>'
            f'<div class="kpi-value">{escape(item.value)}</div>'
            f'<div class="kpi-context">{escape(item.context)}</div></div>'
        )
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_recommendations(brief: BusinessBrief) -> None:
    st.markdown('<div class="section-label">What ADA would do next</div>', unsafe_allow_html=True)
    for item in brief.recommendations:
        st.markdown(
            f"""
            <div class="recommendation">
              <div class="recommendation-top"><span class="priority">{escape(item.priority)}</span><h3>{escape(item.title)}</h3></div>
              <p>{escape(item.action)}</p>
              <p class="why"><strong>Evidence:</strong> {escape(item.rationale)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_evidence(brief: BusinessBrief, *, limit: int | None = None) -> None:
    evidence = brief.evidence[:limit] if limit else brief.evidence
    if not evidence:
        st.markdown(
            '<div class="empty-state">ADA could not find a strong business signal. Use “Tune detection” to select a date, metric, and segment.</div>',
            unsafe_allow_html=True,
        )
        return
    cards = []
    for item in evidence:
        cards.append(
            f'<div class="evidence {escape(item.tone)}">'
            f'<div class="evidence-value">{escape(item.value)}</div>'
            f'<h3>{escape(item.title)}</h3><p>{escape(item.statement)}</p>'
            f'<p class="calculation">↳ {escape(item.calculation)}</p></div>'
        )
    st.markdown(f'<div class="evidence-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def style_chart(figure: go.Figure, *, height: int = 390) -> go.Figure:
    figure.update_layout(
        height=height,
        margin={"l": 18, "r": 18, "t": 55, "b": 20},
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
    chart_columns = st.columns(2)
    with chart_columns[0]:
        with st.container(border=False):
            if not trend.empty:
                figure = px.area(
                    trend,
                    x="Period",
                    y="Value",
                    markers=True,
                    title=f"{roles.measure or 'Records'} over time",
                    color_discrete_sequence=[ACCENT],
                )
                figure.update_traces(line={"width": 3}, fillcolor="rgba(109,94,247,.12)")
                st.plotly_chart(style_chart(figure), width="stretch", config={"displayModeBar": False})
            else:
                st.markdown('<div class="empty-state">Add or select a date column to see movement over time.</div>', unsafe_allow_html=True)

    with chart_columns[1]:
        if not segments.empty:
            ordered = segments.sort_values("Value")
            figure = px.bar(
                ordered,
                x="Value",
                y="Segment",
                orientation="h",
                title=f"{roles.measure or 'Records'} by {roles.dimension}",
                color_discrete_sequence=[LIME],
            )
            figure.update_traces(marker_line_width=0, hovertemplate="%{y}: %{x:,.2f}<extra></extra>")
            st.plotly_chart(style_chart(figure), width="stretch", config={"displayModeBar": False})
        else:
            st.markdown('<div class="empty-state">Add or select a segment column to see the strongest contributors.</div>', unsafe_allow_html=True)

    numeric = [column for column in roles.numeric if dataframe[column].nunique(dropna=True) > 2]
    lower_columns = st.columns(2)
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


inject_styles()
render_nav()
render_landing()

source_mode = st.segmented_control(
    "Choose a source",
    ["Explore the live demo", "Upload your file"],
    default="Explore the live demo",
    label_visibility="collapsed",
)

uploaded_file = None
business_context = ""
if source_mode == "Upload your file":
    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel workbook",
        type=["csv", "xlsx", "xlsm"],
        help="Maximum file size: 25 MB. The first worksheet is analyzed.",
    )
    business_context = st.text_input(
        "Optional business context",
        placeholder="Example: Subscription revenue by customer, product, and month",
        max_chars=500,
    )
    if uploaded_file is None:
        render_how_it_works()
        st.stop()

try:
    if source_mode == "Explore the live demo":
        raw_dataframe = make_demo_data()
        source_name = "Acme operating data · demo"
        business_context = "Two years of orders across products, regions, and sales channels."
    else:
        assert uploaded_file is not None
        if uploaded_file.size > MAX_UPLOAD_BYTES:
            st.error("That file is larger than the 25 MB analysis limit.")
            st.stop()
        raw_dataframe = read_uploaded_file(uploaded_file.getvalue(), uploaded_file.name)
        source_name = uploaded_file.name

    if len(raw_dataframe) > MAX_ANALYSIS_ROWS:
        st.warning(
            f"This file has {len(raw_dataframe):,} rows. ADA analyzed the first "
            f"{MAX_ANALYSIS_ROWS:,} for predictable performance."
        )
        raw_dataframe = raw_dataframe.head(MAX_ANALYSIS_ROWS).copy()
    dataframe, cleaning_report = clean_dataframe(raw_dataframe)
except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, ValueError, ImportError) as error:
    st.error(f"ADA could not read this file: {error}")
    st.stop()

detected = detect_roles(dataframe)
date_options = ["None", *[column for column in dataframe.columns if pd.api.types.is_datetime64_any_dtype(dataframe[column])]]
measure_options = ["None", *detected.numeric]
dimension_options = ["None", *detected.dimensions]

with st.expander("Tune detection", expanded=False):
    st.caption("ADA chose these columns automatically. Change them only if the dashboard misunderstood your file.")
    selectors = st.columns(3)
    selected_date = selectors[0].selectbox(
        "Date",
        date_options,
        index=date_options.index(detected.date) if detected.date in date_options else 0,
    )
    selected_measure = selectors[1].selectbox(
        "Primary metric",
        measure_options,
        index=measure_options.index(detected.measure) if detected.measure in measure_options else 0,
    )
    selected_dimension = selectors[2].selectbox(
        "Business segment",
        dimension_options,
        index=dimension_options.index(detected.dimension) if detected.dimension in dimension_options else 0,
    )

roles = ColumnRoles(
    date=None if selected_date == "None" else selected_date,
    measure=None if selected_measure == "None" else selected_measure,
    dimension=None if selected_dimension == "None" else selected_dimension,
    identifier=detected.identifier,
    numeric=detected.numeric,
    dimensions=detected.dimensions,
)
brief = analyze_business(dataframe, roles)

role_chips = "".join(
    f'<span class="role-chip">{escape(label)}: {escape(value)}</span>'
    for label, value in (("Metric", roles.measure), ("Segment", roles.dimension), ("Date", roles.date))
    if value
)
st.markdown(
    f'<div class="dataset-bar"><div><span class="dataset-name">{escape(source_name)}</span>{role_chips}</div>'
    f'<div>{len(dataframe):,} rows · {len(dataframe.columns):,} columns · analyzed locally</div></div>',
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="brief">
      <div class="eyebrow">Executive brief</div>
      <h1>{escape(brief.headline)}</h1>
      <p>{escape(brief.summary)}</p>
      <div class="brief-trust">CALCULATED FROM YOUR FILE · RECOMMENDATIONS ARE SHOWN SEPARATELY · NO EXTERNAL AI CALL</div>
    </div>
    """,
    unsafe_allow_html=True,
)
render_kpis(brief)

executive_tab, dashboard_tab, evidence_tab, data_tab = st.tabs(
    ["Executive view", "Dashboard", "Evidence", "Data room"]
)

with executive_tab:
    executive_columns = st.columns([1.12, 0.88], gap="large")
    with executive_columns[0]:
        render_recommendations(brief)
    with executive_columns[1]:
        st.markdown('<div class="section-label">What the data says</div>', unsafe_allow_html=True)
        render_evidence(brief, limit=4)
        st.markdown(
            '<div class="trust-note"><strong>How to read this:</strong> evidence cards are calculations. Recommendations are ADA’s rule-based interpretation of those calculations—not causal proof.</div>',
            unsafe_allow_html=True,
        )

with dashboard_tab:
    render_dashboard(dataframe, roles)

with evidence_tab:
    st.markdown('<div class="section-label">Trace every conclusion</div>', unsafe_allow_html=True)
    render_evidence(brief)
    st.markdown('<div class="section-label" style="margin-top:1.5rem">Detected business schema</div>', unsafe_allow_html=True)
    schema = pd.DataFrame(
        [
            ["Primary metric", roles.measure or "Not detected"],
            ["Business segment", roles.dimension or "Not detected"],
            ["Date", roles.date or "Not detected"],
            ["Identifier", roles.identifier or "Not detected"],
        ],
        columns=["Role", "Column"],
    )
    st.dataframe(schema, hide_index=True, width="stretch")

with data_tab:
    report = build_business_report(
        dataframe,
        brief,
        source_name=source_name,
        context=business_context,
    )
    downloads = st.columns(2)
    downloads[0].download_button(
        "Download executive brief",
        data=report,
        file_name="ada_executive_brief.md",
        mime="text/markdown",
        width="stretch",
    )
    downloads[1].download_button(
        "Download cleaned data",
        data=dataframe.to_csv(index=False).encode("utf-8"),
        file_name="ada_cleaned_data.csv",
        mime="text/csv",
        width="stretch",
    )

    quality_columns = st.columns(4)
    quality_columns[0].metric("Rows analyzed", f"{len(dataframe):,}")
    quality_columns[1].metric("Columns", f"{len(dataframe.columns):,}")
    quality_columns[2].metric("Duplicates removed", f"{cleaning_report.duplicate_rows_removed:,}")
    quality_columns[3].metric("Missing cells", f"{int(dataframe.isna().sum().sum()):,}")

    with st.expander("Cleaning audit"):
        audit = pd.DataFrame(
            [
                ["Empty rows removed", cleaning_report.empty_rows_removed],
                ["Empty columns removed", cleaning_report.empty_columns_removed],
                ["Exported index columns removed", cleaning_report.index_columns_removed],
                ["Duplicate rows removed", cleaning_report.duplicate_rows_removed],
                ["Numeric columns inferred", cleaning_report.numeric_columns_inferred],
                ["Datetime columns inferred", cleaning_report.datetime_columns_inferred],
            ],
            columns=["Operation", "Count"],
        )
        st.dataframe(audit, hide_index=True, width="stretch")

    st.subheader("Cleaned data")
    st.dataframe(dataframe.head(1_000), width="stretch", height=420)
    st.caption("Preview limited to 1,000 rows. The download includes every analyzed row.")
    st.subheader("Data dictionary")
    st.dataframe(column_profile(dataframe), hide_index=True, width="stretch")

st.caption("ADA · Automated Data Analyst · Built by Sainesh Nakra")
