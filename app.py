"""Streamlit entry point for Automated Data Analyst."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from analysis import (
    build_markdown_report,
    clean_dataframe,
    column_profile,
    generate_insights,
)

APP_TITLE = "Automated Data Analyst"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_ANALYSIS_ROWS = 250_000
DEMO_DATA_PATH = Path(__file__).parent / "data" / "sample_sales.csv"

st.set_page_config(
    page_title=f"{APP_TITLE} — private CSV insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1200px; padding-top: 2.2rem; padding-bottom: 4rem;}
      [data-testid="stMetric"] {
        background: #f8fafc; border: 1px solid #e2e8f0;
        padding: 1rem; border-radius: 1rem;
      }
      [data-testid="stSidebar"] {border-right: 1px solid #e2e8f0;}
      .privacy-note {
        padding: .8rem 1rem; border-radius: .8rem; background: #ecfdf5;
        color: #065f46; font-size: .9rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def read_csv_bytes(contents: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(contents), low_memory=False)


@st.cache_data(show_spinner=False)
def read_demo_data() -> pd.DataFrame:
    return pd.read_csv(DEMO_DATA_PATH, low_memory=False)


def render_insights(insights: list) -> None:
    columns = st.columns(2)
    for index, insight in enumerate(insights):
        with columns[index % 2]:
            with st.container(border=True):
                st.markdown(f"**{insight.title}**")
                st.write(insight.detail)


with st.sidebar:
    st.title("📊 ADA")
    st.caption("Exploratory analysis without the API bill.")
    st.markdown(
        '<div class="privacy-note">🔒 Your CSV is processed in this Streamlit session. '
        "No OpenAI key is used and no rows are sent to an AI API.</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    data_source = st.radio("Data source", ["Try the demo", "Upload a CSV"], horizontal=False)
    uploaded_file = None
    if data_source == "Upload a CSV":
        uploaded_file = st.file_uploader("CSV file", type=["csv"], help="Maximum file size: 25 MB")
    description = st.text_area(
        "Dataset context (optional)",
        placeholder="Example: 2019 ecommerce orders by product and city",
        max_chars=500,
    )
    st.divider()
    st.caption(
        "Built by [Sainesh Nakra](https://sainesh.com/) · "
        "[Source](https://github.com/saineshnakra/automated-data-analyst)"
    )

st.title(APP_TITLE)
st.markdown(
    "Upload a CSV and get a quality audit, reproducible observations, interactive charts, "
    "and exportable results—without sending your data to an LLM."
)

if data_source == "Upload a CSV" and uploaded_file is None:
    st.info("Upload a CSV from the sidebar to begin, or switch to the included demo dataset.")
    st.stop()

try:
    if data_source == "Try the demo":
        raw_dataframe = read_demo_data()
        source_name = "sample_sales.csv"
    else:
        assert uploaded_file is not None
        if uploaded_file.size > MAX_UPLOAD_BYTES:
            st.error("That file is larger than the 25 MB analysis limit.")
            st.stop()
        raw_dataframe = read_csv_bytes(uploaded_file.getvalue())
        source_name = uploaded_file.name

    if len(raw_dataframe) > MAX_ANALYSIS_ROWS:
        st.warning(
            f"The file contains {len(raw_dataframe):,} rows. Analysis is limited to the first "
            f"{MAX_ANALYSIS_ROWS:,} rows for predictable performance."
        )
        raw_dataframe = raw_dataframe.head(MAX_ANALYSIS_ROWS).copy()

    dataframe, cleaning_report = clean_dataframe(raw_dataframe)
except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
    st.error(f"I could not analyze this CSV: {error}")
    st.stop()

insights = generate_insights(dataframe)
missing_cells = int(dataframe.isna().sum().sum())
numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
datetime_columns = dataframe.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
category_columns = [
    column
    for column in dataframe.columns
    if column not in numeric_columns
    and column not in datetime_columns
    and dataframe[column].nunique(dropna=True) <= 100
]

st.caption(f"Analyzing **{source_name}**")
metric_columns = st.columns(4)
metric_columns[0].metric("Rows", f"{len(dataframe):,}")
metric_columns[1].metric("Columns", f"{len(dataframe.columns):,}")
metric_columns[2].metric("Missing cells", f"{missing_cells:,}")
metric_columns[3].metric("Duplicates removed", f"{cleaning_report.duplicate_rows_removed:,}")

overview_tab, charts_tab, data_tab, export_tab = st.tabs(
    ["Overview", "Visual explorer", "Data & quality", "Export"]
)

with overview_tab:
    st.subheader("Reproducible observations")
    st.caption(
        "Every statement below is computed directly from the uploaded data. Nothing is model-generated."
    )
    render_insights(insights)
    st.subheader("Cleaning audit")
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

with charts_tab:
    if numeric_columns:
        left, right = st.columns(2)
        with left:
            distribution_column = st.selectbox("Distribution", numeric_columns, key="distribution")
            distribution = px.histogram(
                dataframe,
                x=distribution_column,
                marginal="box",
                color_discrete_sequence=["#059669"],
                title=f"Distribution of {distribution_column}",
            )
            st.plotly_chart(distribution, width="stretch")
        with right:
            if category_columns:
                category_column = st.selectbox("Category", category_columns, key="category")
                counts = (
                    dataframe[category_column]
                    .fillna("Missing")
                    .astype(str)
                    .value_counts()
                    .head(20)
                    .reset_index()
                )
                counts.columns = [category_column, "Rows"]
                category_chart = px.bar(
                    counts,
                    x="Rows",
                    y=category_column,
                    orientation="h",
                    color_discrete_sequence=["#0f766e"],
                    title=f"Top values in {category_column}",
                )
                category_chart.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(category_chart, width="stretch")
            else:
                st.info("No low-cardinality category was found for a bar chart.")

        if len(numeric_columns) >= 2:
            st.subheader("Relationships")
            relationship_columns = st.columns(2)
            x_column = relationship_columns[0].selectbox("X axis", numeric_columns, key="scatter_x")
            y_options = [column for column in numeric_columns if column != x_column]
            y_column = relationship_columns[1].selectbox("Y axis", y_options, key="scatter_y")
            scatter = px.scatter(
                dataframe,
                x=x_column,
                y=y_column,
                opacity=0.65,
                render_mode="webgl",
                color_discrete_sequence=["#7c3aed"],
                title=f"{y_column} vs {x_column}",
            )
            st.plotly_chart(scatter, width="stretch")

            correlation = dataframe[numeric_columns].corr().round(2)
            heatmap = px.imshow(
                correlation,
                text_auto=True,
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Numeric correlation matrix",
            )
            st.plotly_chart(heatmap, width="stretch")
    else:
        st.info("No numeric columns were detected, so numeric charts are unavailable.")

with data_tab:
    st.subheader("Cleaned data preview")
    st.dataframe(dataframe.head(1_000), width="stretch", height=420)
    st.caption("The preview is capped at 1,000 rows; downloads include the full analyzed dataset.")
    st.subheader("Data dictionary")
    st.dataframe(column_profile(dataframe), hide_index=True, width="stretch")

with export_tab:
    report = build_markdown_report(dataframe, cleaning_report, insights, description)
    csv_bytes = dataframe.to_csv(index=False).encode("utf-8")
    st.subheader("Take the results with you")
    st.download_button(
        "Download cleaned CSV",
        data=csv_bytes,
        file_name="cleaned_data.csv",
        mime="text/csv",
        width="stretch",
    )
    st.download_button(
        "Download Markdown report",
        data=report,
        file_name="analysis_report.md",
        mime="text/markdown",
        width="stretch",
    )
    with st.expander("Preview report"):
        st.markdown(report)
