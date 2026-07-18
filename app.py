"""ADA: zero-configuration business intelligence for CSV and Excel data."""

from __future__ import annotations

import hashlib
import os
import secrets

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from ai_insights import (
    DEFAULT_PRESET,
    MODEL_PRESETS,
    AINarrative,
    build_ai_payload,
    generate_ai_narrative,
    narrative_to_markdown,
)
from analysis import column_profile
from business_insights import BusinessBrief, build_business_report
from demo_data import make_demo_data
from file_io import read_tabular_file
from nlq import answer_question, suggested_questions
from pipeline import apply_role_selection, cleaning_audit_frame, prepare_analysis, schema_frame
from ui import (
    inject_styles,
    render_ai_narrative,
    render_brief,
    render_chat_answer,
    render_chat_fallback,
    render_dashboard,
    render_dataset_bar,
    render_evidence,
    render_footer,
    render_how_it_works,
    render_kpis,
    render_landing,
    render_nav,
    render_recommendations,
    render_section_heading,
)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_ANALYSIS_ROWS = 250_000

st.set_page_config(
    page_title="ADA | AI Business Dashboard from CSV & Excel",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get Help": "https://github.com/saineshnakra/automated-data-analyst/issues",
        "Report a bug": "https://github.com/saineshnakra/automated-data-analyst/issues/new",
        "About": "ADA turns business spreadsheets into evidence-backed dashboards and actions.",
    },
)


@st.cache_data(show_spinner=False)
def read_uploaded_file(contents: bytes, filename: str) -> pd.DataFrame:
    return read_tabular_file(contents, filename)


def get_openai_api_key() -> str:
    environment_key = os.getenv("OPENAI_API_KEY", "").strip()
    if environment_key:
        return environment_key
    try:
        return str(st.secrets.get("OPENAI_API_KEY", "")).strip()
    except (FileNotFoundError, StreamlitSecretNotFoundError):
        return ""


def get_safety_identifier() -> str:
    if "ada_session_id" not in st.session_state:
        st.session_state.ada_session_id = secrets.token_urlsafe(24)
    session_id = str(st.session_state.ada_session_id)
    return hashlib.sha256(f"ada:{session_id}".encode()).hexdigest()


def render_sidebar(*, server_api_key: str) -> str:
    with st.sidebar:
        st.title("ADA")
        st.caption("A dashboard that explains itself.")
        st.markdown("---")
        st.markdown("**Analysis contract**")
        st.markdown(
            "- Calculations happen locally\n"
            "- Evidence is shown before interpretation\n"
            "- Raw rows are never sent to the strategy model\n"
            "- Recommendations are not causal proof"
        )
        st.markdown("---")
        if server_api_key:
            st.success("Optional strategy agent is available on this deployment.")
            api_key = server_api_key
        else:
            st.info("Deterministic mode is free and complete. No model call is required.")
            api_key = st.text_input(
                "Optional OpenAI API key",
                type="password",
                placeholder="Session only · not persisted by ADA",
                help=(
                    "Use your own key to enable the optional strategic read. The key remains in this "
                    "Streamlit session and is sent only to the OpenAI API."
                ),
            ).strip()
        st.link_button(
            "Contribute on GitHub",
            "https://github.com/saineshnakra/automated-data-analyst",
            width="stretch",
        )
    return api_key


def maybe_generate_narrative(
    *,
    api_key: str,
    brief: BusinessBrief,
    business_context: str,
) -> tuple[AINarrative | None, str | None]:
    if not api_key:
        return None, None

    payload = build_ai_payload(brief, context=business_context)
    fingerprint = hashlib.sha256(payload.encode()).hexdigest()
    cached = st.session_state.get("ai_narrative")
    cached_fingerprint = st.session_state.get("ai_narrative_fingerprint")
    selected_preset = st.selectbox(
        "Strategy model",
        list(MODEL_PRESETS),
        index=list(MODEL_PRESETS).index(DEFAULT_PRESET),
        help="Luna is the cost-efficient default. Terra spends more reasoning on ambiguous decisions.",
    )
    config = MODEL_PRESETS[selected_preset]

    if st.button("Generate AI strategic read", type="primary", width="stretch"):
        try:
            with st.spinner("Connecting the evidence into a strategic read…"):
                cached = generate_ai_narrative(
                    brief,
                    api_key=api_key,
                    config=config,
                    context=business_context,
                    safety_identifier=get_safety_identifier(),
                )
            st.session_state.ai_narrative = cached
            st.session_state.ai_narrative_fingerprint = fingerprint
            st.session_state.ai_narrative_model = config.model
            cached_fingerprint = fingerprint
        except Exception:  # API failures should never take down the deterministic product.
            st.error("The optional strategy agent is temporarily unavailable. Try again or switch models.")
            return None, None

    if cached_fingerprint != fingerprint or not isinstance(cached, AINarrative):
        return None, None
    model = str(st.session_state.get("ai_narrative_model", config.model))
    return cached, model


def render_ask_ada(dataframe: pd.DataFrame, roles, source_name: str) -> None:
    """Chat over the analyzed dataset; every answer is a local calculation."""
    fingerprint = f"{source_name}:{len(dataframe)}:{','.join(dataframe.columns)}"
    if st.session_state.get("chat_fingerprint") != fingerprint:
        st.session_state.chat_fingerprint = fingerprint
        st.session_state.chat_history = []

    suggestions = suggested_questions(dataframe, roles)
    chips = st.columns(len(suggestions))
    question = None
    for chip, suggestion in zip(chips, suggestions, strict=True):
        if chip.button(suggestion, key=f"chip_{suggestion}", width="stretch"):
            question = suggestion

    typed = st.chat_input("Ask about this data — try “top 5 by revenue” or “which segment grew fastest?”")
    question = typed or question
    if question:
        st.session_state.chat_history.append(
            {"question": question, "result": answer_question(question, dataframe, roles)}
        )

    if not st.session_state.chat_history:
        st.markdown(
            '<div class="empty-state">Ask anything about the analyzed table. '
            "Answers are computed locally and every one shows its calculation.</div>",
            unsafe_allow_html=True,
        )
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(entry["question"])
        with st.chat_message("assistant"):
            if entry["result"] is not None:
                render_chat_answer(entry["result"])
            else:
                render_chat_fallback(suggestions)


inject_styles()
render_nav()
render_landing()

api_key = render_sidebar(server_api_key=get_openai_api_key())

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
        help="Maximum file size: 25 MB. ADA analyzes the first worksheet.",
    )
    business_context = st.text_input(
        "Optional business context",
        placeholder="Example: Subscription revenue by customer, product, and month",
        max_chars=500,
    )
    if uploaded_file is None:
        render_how_it_works()
        render_footer()
        st.stop()

try:
    if source_mode == "Explore the live demo":
        raw_dataframe = make_demo_data()
        source_name = "Acme operating data · demo"
        business_context = "Two years of orders across products, regions, and sales channels."
    else:
        assert uploaded_file is not None
        if uploaded_file.size > MAX_UPLOAD_BYTES:
            st.error("That file is larger than ADA's 25 MB analysis limit.")
            st.stop()
        raw_dataframe = read_uploaded_file(uploaded_file.getvalue(), uploaded_file.name)
        source_name = uploaded_file.name

    prepared = prepare_analysis(raw_dataframe, row_limit=MAX_ANALYSIS_ROWS)
except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, ValueError, ImportError) as error:
    st.error(f"ADA could not read this file: {error}")
    st.stop()

if prepared.truncated_rows:
    st.warning(
        f"ADA analyzed the first {MAX_ANALYSIS_ROWS:,} rows for predictable performance "
        f"and skipped {prepared.truncated_rows:,}."
    )

dataframe = prepared.dataframe
detected = prepared.detected_roles
date_options = [
    "None",
    *[
        column
        for column in dataframe.columns
        if pd.api.types.is_datetime64_any_dtype(dataframe[column])
    ],
]
measure_options = ["None", *detected.numeric]
dimension_options = ["None", *detected.dimensions]

with st.expander("Tune ADA's schema detection", expanded=False):
    st.caption("ADA selected these roles automatically. Override them only when the source schema needs context.")
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

roles = apply_role_selection(
    detected,
    date=selected_date,
    measure=selected_measure,
    dimension=selected_dimension,
)
brief = prepared.analyze(roles)

render_dataset_bar(source_name, dataframe, roles)
render_brief(brief)
render_kpis(brief)

executive_tab, ask_tab, dashboard_tab, evidence_tab, data_tab = st.tabs(
    ["Executive brief", "Ask ADA", "Live dashboard", "Evidence ledger", "Data room"]
)

with executive_tab:
    render_section_heading(
        "Decision layer",
        "The next move, with receipts",
        "ADA keeps recommendations beside the evidence that triggered them so judgment never masquerades as a metric.",
    )
    executive_columns = st.columns([1.08, 0.92], gap="large")
    with executive_columns[0]:
        st.markdown('<div class="section-label">What ADA would do next</div>', unsafe_allow_html=True)
        render_recommendations(brief)
    with executive_columns[1]:
        st.markdown('<div class="section-label">What the data says</div>', unsafe_allow_html=True)
        render_evidence(brief, limit=4)
        st.markdown(
            '<div class="trust-note"><strong>Trust contract:</strong> evidence cards are calculations. Recommendations are interpretations—not causal proof.</div>',
            unsafe_allow_html=True,
        )

    narrative = None
    narrative_model = None
    if api_key:
        render_section_heading(
            "Optional strategy agent",
            "Connect the signals into a strategic read",
            "Only the computed evidence and supplied business context are sent. Raw uploaded rows stay out of the model prompt.",
        )
        control_column, note_column = st.columns([.42, .58], gap="large")
        with control_column:
            narrative, narrative_model = maybe_generate_narrative(
                api_key=api_key,
                brief=brief,
                business_context=business_context,
            )
        with note_column:
            st.info(
                "Luna is the efficient default. Terra is available when ambiguity justifies more reasoning. "
                "The calculated dashboard remains authoritative either way."
            )
        if narrative and narrative_model:
            render_ai_narrative(narrative, model=narrative_model)
    else:
        narrative = None
        narrative_model = None

with ask_tab:
    render_section_heading(
        "Conversational analyst",
        "Ask this data anything",
        "Questions become transparent pandas calculations that run locally. "
        "No question or answer leaves the session, and every reply shows its math.",
    )
    render_ask_ada(dataframe, roles, source_name)

with dashboard_tab:
    render_section_heading(
        "Operating view",
        "The shape of the business",
        "Trend, contribution, distribution, and the strongest measurable relationship—generated without chart configuration.",
    )
    render_dashboard(dataframe, roles)

with evidence_tab:
    render_section_heading(
        "Evidence ledger",
        "Trace every conclusion",
        "Every displayed signal exposes the calculation behind it. Adjust the detected schema when a business-specific field was misunderstood.",
    )
    render_evidence(brief)
    st.markdown('<div class="section-label" style="margin-top:1.5rem">Detected business schema</div>', unsafe_allow_html=True)
    st.dataframe(schema_frame(roles), hide_index=True, width="stretch")

with data_tab:
    render_section_heading(
        "Data room",
        "Clean, inspect, and take it with you",
        "Review ADA's cleaning audit, inspect the normalized table, and export both the executive brief and analysis-ready data.",
    )
    report = build_business_report(
        dataframe,
        brief,
        source_name=source_name,
        context=business_context,
    )
    if narrative and narrative_model:
        report += "\n\n" + narrative_to_markdown(narrative, model=narrative_model)

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
    quality_columns[2].metric(
        "Duplicates removed",
        f"{prepared.cleaning_report.duplicate_rows_removed:,}",
    )
    quality_columns[3].metric("Missing cells", f"{int(dataframe.isna().sum().sum()):,}")

    with st.expander("Cleaning audit"):
        st.dataframe(
            cleaning_audit_frame(prepared.cleaning_report),
            hide_index=True,
            width="stretch",
        )

    st.subheader("Cleaned data")
    st.dataframe(dataframe.head(1_000), width="stretch", height=420)
    st.caption("Preview limited to 1,000 rows. The download includes every analyzed row.")
    st.subheader("Data dictionary")
    st.dataframe(column_profile(dataframe), hide_index=True, width="stretch")

render_footer()
