# Architecture

## The shape of it

ADA is a Streamlit app wrapped around a pure-Python analysis engine. The engine
does not know Streamlit exists. That is the main structural idea, and most of
the rules below exist to protect it.

```
                 app.py          ← Streamlit page, session state
                    │
                 ui.py           ← components, Plotly styling
                    │
                pipeline.py      ← bundles prep, cleaning, roles, drill-down
                    │
   ┌────────────────┼───────────────────────────────┐
   │                │                               │
file_io.py    business_insights.py                nlq.py
analysis.py     ↑        ↑        ↑                  │
schema.py    anomalies forecasting aggregation ──────┘
                  └────────┴──── timeseries.py
                                 formatting.py
                                       │
                                 ai_insights.py   ← optional, isolated
```

Nothing below `pipeline.py` imports Streamlit. You can run the entire analysis
engine from a plain Python script or a test, with no browser and no API key.

## Layers

| Layer | Files | Rule |
|---|---|---|
| **Presentation** | `app.py`, `ui.py` | Draws things. Calculates nothing. |
| **Orchestration** | `pipeline.py` | Sequences the engine. Holds no business logic of its own. |
| **Engine** | `analysis.py`, `schema.py`, `aggregation.py`, `timeseries.py`, `anomalies.py`, `forecasting.py`, `business_insights.py`, `nlq.py` | Pure functions over DataFrames. No I/O, no globals. |
| **Input** | `file_io.py`, `demo_data.py` | Parses and validates. No analysis. |
| **Output formatting** | `formatting.py` | Decides how a number *looks*. Never changes what it *is*. |
| **Optional AI** | `ai_insights.py` | The only file allowed to make a network call. |

## What each file owns

| File | Owns |
|---|---|
| `app.py` | Streamlit page setup, session state, sidebar, the upload flow |
| `ui.py` | Reusable render functions and consistent Plotly styling |
| `pipeline.py` | `prepare_analysis`, role overrides, drill-down focus, audit frames |
| `file_io.py` | Validated CSV/Excel parsing, worksheet listing |
| `analysis.py` | Cleaning, the cleaning report, column profiling, Markdown report |
| `schema.py` | `ColumnRoles` and the role-detection scoring |
| `aggregation.py` | **The only definition of a period.** Trend, segment, driver, heatmap frames |
| `timeseries.py` | Theil–Sen trendline and robust scale |
| `anomalies.py` | Anomaly detection and the calibrated threshold table |
| `forecasting.py` | Baseline forecast, prediction band, backtest |
| `business_insights.py` | Evidence cards, KPIs, recommendations, the executive brief |
| `nlq.py` | Question → `QueryPlan` → local execution |
| `formatting.py` | `format_number`, `format_period`, column-name helpers |
| `ai_insights.py` | Typed Responses API calls for the query planner and strategic read |
| `demo_data.py` | The deterministic synthetic dataset |
| `tools/` | Offline calibration scripts, not imported by the app |

## Rules that hold the design together

These are not style preferences. Breaking one causes a specific, known problem.

**1. One place decides what a period is.**
`aggregation.py` owns it. If the trend chart, the anomaly detector, and the
forecast each decided their own grain, they would disagree on screen and nobody
would be able to tell which was right.

**2. Formatting never changes values.**
`formatting.py` turns a number into a string. It never rescales, rounds, or
reinterprets the stored data. A display bug should never become a data bug.

**3. Calculations live in the engine, not in UI callbacks.**
If a number is computed inside a Streamlit callback, it cannot be tested without
running a browser session.

**4. Evidence and recommendations stay separate.**
Evidence is calculated. Recommendations are interpretation. They are different
types (`Evidence`, `Recommendation`), rendered differently, and worded
differently. See [Concepts](concepts.md#evidence-vs-recommendation).

**5. Raw rows never reach a model.**
`ai_insights.py` builds its payloads from column schema and already-computed
evidence only. Model output is parsed into a typed Pydantic schema, and a
model-produced query plan runs through the same local executor as a rule-parsed
one. Generated code is never executed. See [Privacy](privacy.md).

**6. Every analytical rule has a test.**
Including the edge cases where it degrades — not enough history, all-zero
series, a single segment, missing dates.

**7. The product works with no API key.**
The deterministic path is the product. The AI layer is an extra, and the app
must be fully usable without it.

## Dependency injection at the model boundary

`generate_ai_narrative` and `plan_query_with_ai` take a client object rather
than constructing one. The expected shape is declared as a `Protocol`
(`_Client`, `_Responses`), so tests pass a fake and assert on the payload
without network access or API credits.

This is how the privacy-contract tests work: they inspect exactly what would
have been sent.

## Where to add things

| You want to add | Put it in | Also do |
|---|---|---|
| A new evidence card | `business_insights.py` | Add a `calculation` string and a test |
| A new question shape | `nlq.py` | Extend `QueryPlan` if needed; test parse *and* execute |
| A new chart | `ui.py` | Build its frame in `aggregation.py`, not in the render function |
| A new file format | `file_io.py` | Keep the validation as strict as it is now |
| A new detection heuristic | `schema.py` | Add a synthetic fixture that would fail without it |
| Anything model-related | `ai_insights.py` | Confirm no rows enter the payload; add a privacy test |
