# Architecture

## Overview

ADA is a pure-Python analysis engine (the `ada` package) with a Python API, a
command line and tools for language models (`ada/api.py`, `ada/cli.py`,
`ada/tools.py`, `ada/mcp.py`, `ada/chat.py`). It runs from a script or a test
with no browser and no API key.

```
   ada/cli.py  (the ada command)      ada/tools.py, ada/mcp.py, ada/chat.py  (language models)
        │                                  │
        └────────────────┬─────────────────┘
                         │
   ada/api.py  (ada.analyze, Analysis, Refusal)   ada/claims.py  (ada.check_claim)
   ada/folder.py  (ada.scan)                      ada/joins.py  (ada.ask_folder)
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
 autovis.py      chart-form choice for a set of columns
 selection.py    a selection on a chart turned into filters
```

The installed package depends on pandas, numpy, scipy, statsmodels, ruptures,
python-calamine, PyYAML and SymPy.
[Credits](credits.md) says what each library does for ADA and how to cite it.

The research workflow, the modelling mode, the browser app and the optional AI
layer are in ADA Lab, which builds on this package. `app.py` at the repository
root only runs ADA Lab's app, for the hosted demo. The commands and names that
moved (`ada init`, `ada build`, `ada model`, `ada app`, `ada.model`,
`ada.research` and the rest) are listed in `MOVED_COMMANDS` in `ada/cli.py` and
`MOVED` in `ada/__init__.py`, which hand them to ADA Lab when it is installed.

## Layers

| Layer | Files | Rule |
|---|---|---|
| Public interface | `ada/api.py`, `ada/cli.py`, `ada/tools.py`, `ada/mcp.py`, `ada/chat.py` | Wraps the engine for Python, the shell and language models' tool calls. No analysis of its own. |
| Orchestration | `ada/pipeline.py` | Sequences cleaning, role detection and the row limit. |
| Engine | `analysis.py`, `schema.py`, `claims.py`, `folder.py`, `aggregates.py`, `decisions.py`, `joins.py`, `metrics.py`, `aggregation.py`, `timeseries.py`, `anomalies.py`, `forecasting.py`, `survival.py`, `timepoints.py`, `compare.py`, `quality.py`, `business_insights.py`, `nlq.py`, `autovis.py`, `selection.py` | Functions over DataFrames. No network, no global state. |
| Input | `ada/file_io.py`, `ada/numeric.py`, `ada/sources.py`, `ada/demo_data.py` | Parses and validates input and records where it came from. No analysis. |
| Output formatting | `ada/formatting.py` | Decides how a value is displayed, never what it is. |
| Network | `ada/chat.py`, `ada/mcp.py` | `chat` calls the model server it is pointed at, on this computer by default; `mcp --http` listens on 127.0.0.1 unless told otherwise. Nothing else opens a connection, and both use only the standard library. |

## Module responsibilities

| File | Owns |
|---|---|
| `ada/api.py` | `analyze`, `Analysis`, `Refusal` and its codes, `sample`, `read_file` |
| `ada/cli.py` | The `ada` command: `analyze`, `schema`, `ask`, `claim`, `scan`, `check`, `skill`, `tools`, `chat`, `mcp`, `samples`; the commands that moved to ADA Lab are handed to it |
| `ada/compute.py` | `compute`: an arithmetic expression read by a small grammar (no Python is run), computed with exact fractions and integers, each step recorded, conventions stated, and refusals for what reads two ways |
| `ada/algebra.py` | `simplify`, `solve` and `check_solution`: expressions with one-letter variables as quotients of polynomials with fraction coefficients, linear equations and systems solved by elimination with each step, solutions put back into the equations; `work_out` sends a text to compute, simplify or solve |
| `ada/latex.py` | Reading arithmetic written in LaTeX for `check_arithmetic`: fractions, roots, braced powers, binomial coefficients and boxed answers read into `compute`'s notation; every step of a chain checked, exactly when both sides are rational and to 30 digits otherwise; steps with letters after the first of a chain checked as identities with ADA's algebra; what reads two ways, such as a mixed number, skipped |
| `ada/formulas.py` | `solve_with_letters`: an equation with powers of the unknown and other letters, factored by SymPy over the rationals and checked by multiplying back; each factor's roots by solving for the unknown, the quadratic formula with square factors taken out, or root(t, n); the conditions each answer needs; each root checked in its factor with exact arithmetic |
| `ada/systems.py` | `solve_system`: systems of polynomial equations with finitely many solutions, brought down by SymPy's Gröbner basis to one equation in one letter with each unknown a polynomial in it; real solutions exact where the root of that equation is, otherwise inside an interval worked out with exact fractions; each checked by dividing every equation by the factor its root comes from |
| `ada/roots.py` | `real_roots` and `factor_over_rationals`: the real roots of a polynomial in one letter, factored over the rationals with SymPy; each factor's roots by the quadratic formula with simplified square roots, root(t, n) for a*x^n + c, or SymPy's isolating intervals (counted again by Sturm's theorem up to degree 30) narrowed by halving; exact roots checked by exact substitution, the rest bracketed by a sign change |
| `ada/tools.py` | `definitions`, `run` and `brief`: the tools a language model calls (`ada_ask`, `ada_check_claim`, `ada_scan`, `ada_compute`), all read-only, for OpenAI-compatible servers, the Anthropic API and MCP, and the slips of small models it takes in stride; `langchain` gives the same tools as LangChain tools when langchain-core is installed |
| `ada/mcp.py` | `Server`, `serve_stdio` and `http_server`: the Model Context Protocol over standard input and output or Streamable HTTP, on the standard library |
| `ada/chat.py` | `Chat` and `chat`: a conversation with a model on an OpenAI-compatible server (Ollama by default) that calls ADA's tools; `ada chat` |
| `ada/claims.py` | `check_claim`: reads a claim's number and interval, asks the question, compares, and returns `ClaimCheck` with its reason |
| `ada/sources.py` | `Source`: the file, its sha256, rows and columns an answer was computed from |
| `ada/folder.py` | `scan`: profiles every table in a folder, proposes links from their key values, and lists the questions to confirm |
| `ada/aggregates.py` | `aggregate_rows`: the World, region and Total rows that add up other rows; `total_levels`: the "Both sexes" and "All Ages" levels of a cross-tabulated table, and the whole of a column of geographic levels |
| `ada/joins.py` | `ask_folder`: a question about a folder answered from the file it fits, or from two files joined along a link; `joined_tables` |
| `ada/decisions.py` | `Decision`: the choices an answer rests on, ADA's default for each, and the options a caller may pick |
| `ada/numeric.py` | `to_numbers`: text read as numbers; every module uses it instead of `pd.to_numeric`, which crashes on some text |
| `ada/pipeline.py` | `prepare_analysis`, role overrides, drill-down focus, chart-selection analysis, the audit and schema tables |
| `ada/file_io.py` | CSV, TSV and Excel parsing, worksheet listing, bundled samples, formula-safe CSV export |
| `ada/analysis.py` | Cleaning, the cleaning report, column profiling |
| `ada/schema.py` | `ColumnRoles` and role detection |
| `ada/metrics.py` | Whether a measure is summed or averaged, and whether a rise is good news |
| `ada/aggregation.py` | The definition of a period; trend, segment, driver and heatmap frames |
| `ada/timeseries.py` | Theil-Sen trendline, calendar positions, robust scale, trend test |
| `ada/anomalies.py` | Anomaly detection and the calibrated threshold table |
| `ada/forecasting.py` | Baseline forecast, prediction band, backtest |
| `ada/survival.py` | Follow-up time and event pairs, Kaplan-Meier survival, log-rank tests, a Cox model |
| `ada/timepoints.py` | Numeric time-point axes, profiles in order and groups' reach along them, series laid out across numbered columns |
| `ada/quality.py` | The data checks: impossible values, codes, caps, fill values, extreme values, swapped, copied and stuck columns, copied rows, simulated-looking data, patterns in missing data; setting clear missing-value codes to missing during cleaning; the frame statistics are computed on |
| `ada/business_insights.py` | Evidence cards, KPIs, recommendations, the executive brief |
| `ada/nlq.py` | Question to `QueryPlan` to local execution |
| `ada/autovis.py` | Chart form for a set of columns, and the sentence explaining it; draws nothing |
| `ada/selection.py` | Turns a selection on a chart into filters, states them in words, and compares the slice with the whole |
| `ada/formatting.py` | `format_number`, `format_period`, `normalized_name` |
| `ada/demo_data.py` | A seeded synthetic sales dataset, used by the tests |
| `app.py` | Runs ADA Lab's browser app for the hosted demo; no analysis of its own |
| `tools/` | Offline calibration scripts, not imported by the package |

## Design rules

**1. One module defines a period.**
`ada/aggregation.py` chooses the grain and the timeline adjustments. The
trend, anomaly detection, the forecast and Ask ADA all read the same
`TrendSeries`, so they agree on what a period is.

**2. Formatting does not change values.**
`ada/formatting.py` turns a number into a string. It does not rescale, round or
reinterpret stored data, so a display change cannot alter a result.

**3. Calculations live in the engine.**
The command line, the tools and the MCP server only pass a request to the
engine and its result back, so every number can be tested by calling an engine
function directly.

**4. Evidence and recommendations are separate types.**
Evidence is calculated; recommendations are interpretation. They are separate
dataclasses (`Evidence`, `Recommendation`) and are reported and worded
differently. See [Concepts](concepts.md#evidence-vs-recommendation).

**5. A model picks a tool; ADA does the arithmetic.**
A language model calls `ada_ask`, `ada_check_claim` or `ada_scan` with a path
and a question or claim. ADA turns the question or claim into a `QueryPlan`,
runs it with pandas and returns the result with its calculation. Generated
code is never executed. See [Privacy](privacy.md).

**6. Each analytical rule has a test**, including the degraded cases: short
history, all-zero series, a single segment, missing dates.

**7. Nothing needs an API key.**

## Testing the model boundary

`ada chat` reaches a model only through `Chat._request`, at the `base_url` it
is given. `tests/test_chat.py` points it at a fake OpenAI-compatible server on
127.0.0.1 and inspects the exact requests, with no network access and no
model.

## Where to add things

| To add | Put it in | Also |
|---|---|---|
| An evidence card | `ada/business_insights.py` | A `calculation` string and a test |
| A question shape | `ada/nlq.py` | Extend `QueryPlan` only if needed; test parsing and execution |
| A chart form | `ada/autovis.py` | A rationale sentence and a test asserting the form for a representative selection |
| A file format | `ada/file_io.py` | Validation at least as strict as the existing formats |
| A detection heuristic | `ada/schema.py` | A synthetic fixture that fails without it |
| A public API method | `ada/api.py` | A `Refusal` code from `REFUSAL_CODES` for each case it cannot answer |
| A tool for language models | `ada/tools.py` | A definition in `_TOOLS` and a branch in `run`; `ada mcp` and `ada chat` take their tools from there |
| A change to the AI skill | `ada/skills/ada/` | The same change in `plugins/automated-data-analyst/skills/ada/`; a test checks the two match |
