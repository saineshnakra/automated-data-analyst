# ADA documentation

ADA computes numbers from tables and checks numbers stated about them. `ada
ask` answers a question with the calculation and the rows behind it, `ada
claim` checks a stated number and says verified, or unverified with a reason
(see [Checking claims](reference/claims.md)), and `ada scan` maps a folder of
related files and the keys that join them (see [Reading a folder of
tables](reference/folder.md)). For a study, `ada init`, `ada build` and `ada
verify` make a paper's assets from the data and a `study.yml`, each traceable
to the data: see [Paper assets for a study](research.md). For any table, ADA
also takes a first look: it checks the data for problems, tests which groups
differ, finds what relates to what and flags changes over time. Each number is
shown with the calculation that produced it.

The engine is the `ada` package. The Streamlit app is `app.py` at the
repository root, and the bundled sample files are in `ada/samples/`.

## Start here

| # | Page | Contents |
|---|---|---|
| 1 | [Concepts](concepts.md) | The terms ADA uses: measure, segment, period, evidence, plan |
| 2 | [What ADA does](features.md) | Everything each step does, in detail |
| 3 | [How it works](how-it-works.md) | The path from uploaded file to dashboard |
| 4 | [Architecture](architecture.md) | Which module does what, and the rules between them |

## Reference

One page per pipeline step: what it does, where the code is, the rules it
follows and its edge cases.

| Page | Module | Covers |
|---|---|---|
| [Reading files](reference/reading-files.md) | `ada/file_io.py` | Accepted formats, encodings, limits, rejections |
| [Cleaning](reference/cleaning.md) | `ada/analysis.py` | What ADA changes before analyzing |
| [Schema detection](reference/schema-detection.md) | `ada/schema.py` | How columns are assigned roles |
| [Periods](reference/periods.md) | `ada/aggregation.py`, `ada/timeseries.py` | Period grain, timeline adjustments, the trendline |
| [Anomalies](reference/anomalies.md) | `ada/anomalies.py` | When a period is flagged |
| [Forecasting](reference/forecasting.md) | `ada/forecasting.py` | The forecast model, its guards and backtest |
| [Data checks](reference/quality.md) | `ada/quality.py` | Impossible values and physical limits, codes, caps, fill values, extreme values, clusters, swapped and stuck readings, copied rows, simulated-looking data and patterns in missing data |
| [Evidence and recommendations](reference/evidence.md) | `ada/business_insights.py` | Evidence cards, KPIs, recommendations |
| [Comparing groups and finding relationships](reference/compare.md) | `ada/compare.py` | Group comparisons with tests and effect sizes, the scan of every reading, and what relates to the measure |
| [Ask ADA](reference/ask-ada.md) | `ada/nlq.py` | How a question becomes a pandas calculation |
| [Checking claims](reference/claims.md) | `ada/claims.py` | How a stated number is checked against the table: verified, or unverified with a reason |
| [Decisions](reference/decisions.md) | `ada/decisions.py`, `ada/aggregates.py` | The choices an answer rests on (World and region rows, a total row), ADA's defaults and how a caller picks another |
| [Reading a folder of tables](reference/folder.md) | `ada/folder.py` | Profiles each table in a folder, proposes the keys that join them and lists what to confirm |
| [Formatting](reference/formatting.md) | `ada/formatting.py` | How numbers and dates are displayed |
| [Choosing the chart](reference/autovis.md) | `ada/autovis.py` | How Explore picks a chart form, and the colour rules |
| [Selecting a slice](reference/selection.md) | `ada/selection.py` | How a selection on the Explore chart becomes a filter, and how the slice is compared with the whole |
| [The optional AI layer](reference/ai-layer.md) | `ada/ai_insights.py` | What the model receives and how its output is checked |

## Working on ADA

| Page | Contents |
|---|---|
| [Development](development.md) | Setup, tests, linting, CI, conventions |
| [Privacy](privacy.md) | What stays local and what can leave the machine |
| [FAQ](faq.md) | Short answers to common questions |
| [Credits](credits.md) | The libraries ADA builds on, their licences, and how to cite them |

Project files at the repository root: [CONTRIBUTING](../CONTRIBUTING.md),
[ROADMAP](../ROADMAP.md), [SECURITY](../SECURITY.md),
[CODE_OF_CONDUCT](../CODE_OF_CONDUCT.md).
