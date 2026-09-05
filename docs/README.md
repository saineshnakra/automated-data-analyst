# ADA documentation

ADA reads a CSV or Excel file and returns a dashboard, an executive brief, and
answers to plain-English questions. Every number it shows comes with the
calculation that produced it.

These docs explain how that works, one piece at a time.

## Start here

Read these three in order. About 15 minutes total.

| # | Page | What you get |
|---|---|---|
| 1 | [Concepts](concepts.md) | The words ADA uses: measure, segment, period, evidence, plan |
| 2 | [How it works](how-it-works.md) | The full journey from uploaded file to dashboard |
| 3 | [Architecture](architecture.md) | Which file does what, and the rules that keep it that way |

## Reference

One page per step of the pipeline. Each page says what the step does, where the
code lives, the rules it follows, and what happens at the edges.

| Page | Module | Question it answers |
|---|---|---|
| [Reading files](reference/reading-files.md) | `file_io.py` | What files are accepted, and what is rejected |
| [Cleaning](reference/cleaning.md) | `analysis.py` | What ADA changes about your data before analyzing it |
| [Schema detection](reference/schema-detection.md) | `schema.py` | How ADA guesses which column means what |
| [Periods](reference/periods.md) | `aggregation.py`, `timeseries.py` | How rows become a timeline, and how the trendline is fitted |
| [Anomalies](reference/anomalies.md) | `anomalies.py` | When a period counts as unusual |
| [Forecasting](reference/forecasting.md) | `forecasting.py` | What the forecast is, and when it refuses to make one |
| [Evidence and recommendations](reference/evidence.md) | `business_insights.py` | How findings and next steps are produced |
| [Ask ADA](reference/ask-ada.md) | `nlq.py` | How a question becomes a pandas calculation |
| [Formatting](reference/formatting.md) | `formatting.py` | How numbers and dates are written down |
| [The optional AI layer](reference/ai-layer.md) | `ai_insights.py` | What the model does, and what it is never given |

## Working on ADA

| Page | What you get |
|---|---|
| [Development](development.md) | Setup, tests, linting, CI, and the conventions to follow |
| [Privacy](privacy.md) | What stays local and what can leave the machine |
| [FAQ](faq.md) | Short answers to the questions people ask most |

Project-level files live at the repository root: [CONTRIBUTING](../CONTRIBUTING.md),
[ROADMAP](../ROADMAP.md), [SECURITY](../SECURITY.md), [CODE_OF_CONDUCT](../CODE_OF_CONDUCT.md).
