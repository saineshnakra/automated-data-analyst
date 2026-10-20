# ADA documentation

ADA reads a CSV or Excel file and produces a dashboard, an executive brief and
answers to plain-English questions. Each number is shown with the calculation
that produced it.

The engine is the `ada` package. The Streamlit app is `app.py` at the
repository root, and the bundled sample files are in `ada/samples/`.

## Start here

| # | Page | Contents |
|---|---|---|
| 1 | [Concepts](concepts.md) | The terms ADA uses: measure, segment, period, evidence, plan |
| 2 | [How it works](how-it-works.md) | The path from uploaded file to dashboard |
| 3 | [Architecture](architecture.md) | Which module does what, and the rules between them |

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
| [Evidence and recommendations](reference/evidence.md) | `ada/business_insights.py` | Evidence cards, KPIs, recommendations |
| [Ask ADA](reference/ask-ada.md) | `ada/nlq.py` | How a question becomes a pandas calculation |
| [Formatting](reference/formatting.md) | `ada/formatting.py` | How numbers and dates are displayed |
| [Choosing the chart](reference/autovis.md) | `ada/autovis.py` | How Explore picks a chart form, and the colour rules |
| [The optional AI layer](reference/ai-layer.md) | `ada/ai_insights.py` | What the model receives and how its output is checked |

## Working on ADA

| Page | Contents |
|---|---|
| [Development](development.md) | Setup, tests, linting, CI, conventions |
| [Privacy](privacy.md) | What stays local and what can leave the machine |
| [FAQ](faq.md) | Short answers to common questions |

Project files at the repository root: [CONTRIBUTING](../CONTRIBUTING.md),
[ROADMAP](../ROADMAP.md), [SECURITY](../SECURITY.md),
[CODE_OF_CONDUCT](../CODE_OF_CONDUCT.md).
