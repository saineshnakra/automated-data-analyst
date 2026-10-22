# ADA roadmap

The roadmap favors depth of analysis over more chart types. Each item is scoped
so one contributor can take it from start to finish.

## Shipped

- Calibrated anomaly threshold: a residual band per history length with a 5%
  false-alarm rate on stable series
- Timeline adjustments: partial trailing periods excluded, empty periods
  filled, grain chosen from span and recording cadence, each adjustment noted
  under the chart
- Forecast with a band that widens with the horizon and a holdout backtest
  (MAPE, MASE against no change, and seasonal MASE against the same period a
  year earlier)
- Evidence with volatility context, and correlations with sample size, a
  confidence interval and a rank check
- Ask ADA: plain-English questions parsed into query plans and run locally,
  with refusal of questions the parser cannot read
  ([#22](https://github.com/saineshnakra/automated-data-analyst/issues/22))
- Optional AI query planner using the schema only, with user approval before a
  plan runs ([#21](https://github.com/saineshnakra/automated-data-analyst/issues/21))
- Anomaly flags on the chart, in evidence and in recommendations
- Drill-down focus on one segment value
- Movement waterfall and segment × period heatmap
- Worksheet selection for multi-sheet workbooks
- Explore tab: chart form chosen from the selected columns, with the reason
  printed (see [docs/reference/autovis.md](docs/reference/autovis.md))
- Select points on the Explore chart and analyze that slice against the whole
  ([#30](https://github.com/saineshnakra/automated-data-analyst/issues/30), see
  [docs/reference/selection.md](docs/reference/selection.md))
- Currency and percentage formatting from column names
  ([#7](https://github.com/saineshnakra/automated-data-analyst/issues/7))
- Accounting-style negatives and locale-formatted numbers
  ([#1](https://github.com/saineshnakra/automated-data-analyst/issues/1))
- Rates and measured levels averaged rather than summed
- Installable package with a Python API (`ada.analyze`) and the `ada` command

## Near term

- Rolling-origin backtesting: score the forecast from several origins instead
  of one holdout split
- Detection sensitivity control in the app: let a user trade the calibrated
  false-alarm rate for sensitivity and show the cost (the library already
  accepts `detect_anomalies(threshold=...)`)
- Heavy-tailed calibration: the false-alarm rate assumes roughly normal noise;
  spiky measures need their own calibration or a warning
- [Cohort and retention analysis](https://github.com/saineshnakra/automated-data-analyst/issues/3):
  detect customer and event-time fields, build a cohort matrix, and explain
  retention changes with visible calculations
- Metric semantics beyond rates and levels: balances and other non-additive
  measures
- Cross-sheet relationships: joins and shared keys across worksheets of one
  workbook
- Accessibility: keyboard-first controls,
  [text summaries for charts](https://github.com/saineshnakra/automated-data-analyst/issues/2),
  stronger focus states, signals that do not rely on colour
- Ask ADA grammar:
  [comparisons](https://github.com/saineshnakra/automated-data-analyst/issues/5)
  ("West vs East"),
  [shares](https://github.com/saineshnakra/automated-data-analyst/issues/4)
  ("what % of revenue is Enterprise") and date ranges

## Contributor-sized

- Synthetic fixtures for finance, subscription, support, marketplace and
  operations schemas
- [Export the evidence ledger as JSON](https://github.com/saineshnakra/automated-data-analyst/issues/6)
  from the app (the library has `Analysis.to_dict()` and `ada analyze --json`)
- Chart download controls with accessible filenames
- Mixed-locale dates, including dot-separated day-first dates
  ([#23](https://github.com/saineshnakra/automated-data-analyst/issues/23))
- [Pass column values to `format_number` from Ask ADA answers](https://github.com/saineshnakra/automated-data-analyst/issues/26)
- Document a self-hosted deployment path and its privacy tradeoffs

## Decision tools

**Templates.** A dashboard shaped for a known kind of file.

- [Prebuilt dashboard template library](https://github.com/saineshnakra/automated-data-analyst/issues/11) (the other templates build on this)
- [AR aging dashboard](https://github.com/saineshnakra/automated-data-analyst/issues/12)
- [Profit and loss statement and P&L KPIs](https://github.com/saineshnakra/automated-data-analyst/issues/13)
- [Cash flow statement and cash flow KPIs](https://github.com/saineshnakra/automated-data-analyst/issues/14)
- [Balance sheet KPIs](https://github.com/saineshnakra/automated-data-analyst/issues/15)

**Connected data.**

- [Analyze several files together](https://github.com/saineshnakra/automated-data-analyst/issues/16), showing the join match rate and any fan-out
- [Read from a data source, not just an upload](https://github.com/saineshnakra/automated-data-analyst/issues/17): a read-only SQL or URL source before any vendor API
- [Refresh a saved analysis on a schedule](https://github.com/saineshnakra/automated-data-analyst/issues/18), saving the specification and never the rows

A remote source or a scheduled refresh changes what the privacy statement can
say. Either one ships with an updated README and
[docs/privacy.md](docs/privacy.md).

**Broader answers**, each still showing its calculation.

- [Build a whole dashboard from one question](https://github.com/saineshnakra/automated-data-analyst/issues/19)
- [Proactive risk and opportunity alerts](https://github.com/saineshnakra/automated-data-analyst/issues/20)

## Exploration

- [Manual pivot builder](https://github.com/saineshnakra/automated-data-analyst/issues/27): shelves for x, y, colour and filter, starting from the recommended chart
- [Search column pairs for findings](https://github.com/saineshnakra/automated-data-analyst/issues/28), with a correction for the number of candidates examined
- [Suggest cleaning steps](https://github.com/saineshnakra/automated-data-analyst/issues/29): preview a transform and apply it only on confirmation

## Longer horizon

- User-defined metric definitions without SQL
- Multi-file comparison and period-over-period uploads
- Pluggable deterministic insight rules
- An evaluation corpus for narrative faithfulness and action quality
- Saved analysis specifications without storing uploaded data

## Out of scope

- **A catalogue of vendor integrations.** Maintaining hundreds of connectors is
  an ongoing operation, not a feature. One generic source that can point at
  anything fits this project better.
- **Analyst services.** ADA is software; its calculations are meant to be
  readable without someone to interpret them.
- **Causal discovery, editable causal graphs and what-if simulation.** ADA
  labels recommendations as interpretation and evidence as observed
  calculation, and [CONTRIBUTING.md](CONTRIBUTING.md) rules out claims of
  cause. A causal graph inferred from observational data is a hypothesis, not
  a finding. If this is ever built, it needs its own labelling: stated
  assumptions, a stated identification strategy, and wording that separates
  "these move together" from "this causes that".

## Definition of done

A roadmap item is complete when it has a clear outcome for a non-technical
user, deterministic evidence where applicable, defined behaviour on
insufficient data, automated tests, and documentation of any privacy or cost
change.
