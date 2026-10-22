# Architecture

## Overview

ADA is a pure-Python analysis engine (the `ada` package) with two front ends: a
Streamlit app (`app.py` at the repository root) and a Python API with a command
line (`ada/api.py`, `ada/cli.py`). The engine does not import Streamlit, so it
runs from a script or a test with no browser and no API key.

```
   app.py  (Streamlit page, session state)      ada/cli.py  (the ada command)
      │                                              │
   ada/ui.py  (components, Plotly styling)       ada/api.py  (ada.analyze, Analysis, Refusal)
      │                                              │
      └──────────────────┬───────────────────────────┘
                         │
                  ada/pipeline.py  (cleaning, roles, row limit, drill-down)
                         │
   ┌─────────────────────┼──────────────────────────────┐
   │                     │                              │
 file_io.py      business_insights.py                 nlq.py
 analysis.py      │        │         │                  │
 schema.py    anomalies  forecasting  aggregation ──────┘
                  │        │          metrics.py
                  │        │          quality.py
                  └────────┴─ timeseries.py
                              formatting.py

 compare.py      group comparisons and the relationship scan (used by business_insights.py and nlq.py)
 autovis.py      chart-form choice for the Explore tab (used by ui.py)
 ai_insights.py  optional; the only module that makes a network call
```

Only `app.py`, `ada/ui.py` and the `ada app` subcommand import Streamlit, and
`ada app` imports it only when run. The installed package depends on pandas,
numpy, scipy and openpyxl; Streamlit and Plotly come with the `app` extra, and the
OpenAI client and Pydantic with the `ai` extra.

## Layers

| Layer | Files | Rule |
|---|---|---|
| Presentation | `app.py`, `ada/ui.py` | Draws results. Does not calculate. |
| Public interface | `ada/api.py`, `ada/cli.py` | Wraps the engine for Python and the shell. No analysis of its own. |
| Orchestration | `ada/pipeline.py` | Sequences cleaning, role detection and the row limit. |
| Engine | `analysis.py`, `schema.py`, `metrics.py`, `aggregation.py`, `timeseries.py`, `anomalies.py`, `forecasting.py`, `compare.py`, `quality.py`, `business_insights.py`, `nlq.py`, `autovis.py`, `selection.py` | Functions over DataFrames. No network, no global state. |
| Input | `ada/file_io.py`, `ada/demo_data.py` | Parses and validates input. No analysis. |
| Output formatting | `ada/formatting.py` | Decides how a value is displayed, never what it is. |
| Optional AI | `ada/ai_insights.py` | The only module that makes a network call. `app.py` imports it inside a `try`, so a missing dependency disables the AI features and nothing else. |

## Module responsibilities

| File | Owns |
|---|---|
| `app.py` | Streamlit page setup, session state, sidebar, upload flow, the Ask ADA approval step |
| `ada/ui.py` | Render functions, Plotly styling, the chart palette |
| `ada/api.py` | `analyze`, `Analysis`, `Refusal` and its codes, `sample`, `read_file` |
| `ada/cli.py` | The `ada` command: `analyze`, `schema`, `ask`, `check`, `samples`, `app` |
| `ada/pipeline.py` | `prepare_analysis`, role overrides, drill-down focus, chart-selection analysis, the audit and schema tables |
| `ada/file_io.py` | CSV and Excel parsing, worksheet listing, bundled samples, formula-safe CSV export |
| `ada/analysis.py` | Cleaning, the cleaning report, column profiling |
| `ada/schema.py` | `ColumnRoles` and role detection |
| `ada/metrics.py` | Whether a measure is summed or averaged, and whether a rise is good news |
| `ada/aggregation.py` | The definition of a period; trend, segment, driver and heatmap frames |
| `ada/timeseries.py` | Theil-Sen trendline, calendar positions, robust scale |
| `ada/anomalies.py` | Anomaly detection and the calibrated threshold table |
| `ada/forecasting.py` | Baseline forecast, prediction band, backtest |
| `ada/quality.py` | The data checks: impossible values, codes, caps, fill values, extreme values, copied columns, patterns in missing data; setting a clear code in the measure aside |
| `ada/business_insights.py` | Evidence cards, KPIs, recommendations, the executive brief |
| `ada/nlq.py` | Question to `QueryPlan` to local execution |
| `ada/autovis.py` | Chart form for a set of columns, and the sentence explaining it |
| `ada/selection.py` | Turns a selection on the Explore chart into filters, states them in words, and compares the slice with the whole |
| `ada/formatting.py` | `format_number`, `format_period`, `normalized_name` |
| `ada/ai_insights.py` | Typed Responses API calls for the query planner and the AI summary |
| `ada/demo_data.py` | The seeded synthetic demo dataset |
| `tools/` | Offline calibration script, not imported by the app or the package |

## Design rules

**1. One module defines a period.**
`ada/aggregation.py` chooses the grain and the timeline adjustments. The trend
chart, anomaly detection, the forecast and Ask ADA all read the same
`TrendSeries`, so they agree on what a period is.

**2. Formatting does not change values.**
`ada/formatting.py` turns a number into a string. It does not rescale, round or
reinterpret stored data, so a display change cannot alter a result.

**3. Calculations live in the engine.**
A number computed inside a Streamlit callback can only be tested by running the
app. Engine functions can be tested directly.

**4. Evidence and recommendations are separate types.**
Evidence is calculated; recommendations are interpretation. They are separate
dataclasses (`Evidence`, `Recommendation`) and are rendered and worded
differently. See [Concepts](concepts.md#evidence-vs-recommendation).

**5. Raw rows are not sent to a model.**
`ada/ai_insights.py` builds its payloads from the column schema and
already-computed evidence. Model output is parsed into Pydantic models, and a
model-produced query plan is validated, shown to the user for approval, and run
by the same local executor as a rule-parsed plan. Generated code is never
executed. See [Privacy](privacy.md).

**6. Each analytical rule has a test**, including the degraded cases: short
history, all-zero series, a single segment, missing dates.

**7. Everything except the AI features works without an API key.**

## Dependency injection at the model boundary

`generate_ai_narrative` and `plan_query_with_ai` accept a `client` argument and
construct an `OpenAI` client only when it is `None`. The expected shape is a
`Protocol` (`_Client`, `_Responses`). Tests pass a fake client and inspect the
exact request, with no network access or API credits.

## Where to add things

| To add | Put it in | Also |
|---|---|---|
| An evidence card | `ada/business_insights.py` | A `calculation` string and a test |
| A question shape | `ada/nlq.py` | Extend `QueryPlan` only if needed; test parsing and execution |
| A dashboard chart | `ada/ui.py` | Build its frame in `ada/aggregation.py`, not in the render function |
| An Explore chart form | `ada/autovis.py` | A matching branch in `ui._explore_figure` |
| A file format | `ada/file_io.py` | Validation at least as strict as the existing formats |
| A detection heuristic | `ada/schema.py` | A synthetic fixture that fails without it |
| A public API method | `ada/api.py` | A `Refusal` code from `REFUSAL_CODES` for each case it cannot answer |
| Anything model-related | `ada/ai_insights.py` | A test showing no cell values enter the payload |
