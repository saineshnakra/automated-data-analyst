# ADA roadmap

The roadmap favors analytical depth and trust over a larger pile of charts. Items are deliberately scoped so contributors can own one outcome end to end.

## Recently delivered

- **Calibrated anomaly threshold** — the residual band is simulated per history length so a stable series raises a false flag about once in twenty analyses, instead of the one-in-four a fixed three deviations produced
- **Honest timelines** — an in-progress trailing period is excluded, periods with no rows count as zero, the grain respects how often the data is actually recorded, and every adjustment is stated
- **Forecast that admits its limits** — a band that widens with the horizon and a scaled error that says outright when the baseline did not beat assuming no change
- **Qualified evidence** — movement measured against the series' own volatility, and correlations reported with sample size, a confidence interval, and a rank check for outlier-driven relationships
- **Ask ADA** — plain-English questions parsed into auditable query plans and executed locally, with an optional schema-only AI planner fallback
- **Anomaly radar** — robust trendline detection with flagged periods on the chart, in evidence, and in recommendations
- **Forecast guardrails** — a baseline offered only with sufficient history, capped horizon, seasonality, and a backtested error shown beside the chart
- **Drill-down focus** — one segment value filters the whole product and regroups by the next useful dimension
- **Movement waterfall and intensity heatmap** — the latest change reconciled by segment, and the measure over segment × period
- **Worksheet selection** — analyze any sheet of a multi-sheet workbook

## Near term

- **Rolling-origin backtesting** — score the forecast across several origins rather than one holdout split, which uses short histories far better
- **Seasonal-naive comparison** — for seasonal series, the bar to clear is last year's same period, not last period
- **Detection sensitivity control** — let a user trade the calibrated false-alarm rate for sensitivity deliberately, and say what it costs
- **Heavy-tailed calibration** — the current false-alarm guarantee assumes roughly normal noise; spiky measures need their own calibration or an explicit warning

- **Cohort and retention analysis** — detect customer and event-time fields, produce a cohort matrix, and explain retention changes with visible calculations
- **Metric semantics** — distinguish additive measures, rates, balances, and identifiers so aggregation choices remain valid
- **Cross-sheet relationships** — surface joins and shared keys across compatible worksheets of one workbook
- **Accessibility pass** — keyboard-first controls, chart descriptions, stronger focus states, and color-independent signals
- **Richer question grammar** — comparisons ("West vs East"), shares ("what % of revenue is Enterprise"), and date ranges in Ask ADA

## Contributor-sized improvements

- Add synthetic fixtures for finance, subscription, support, marketplace, and operations schemas
- Add currency and percentage formatting based on column semantics
- Export the evidence ledger as JSON for downstream workflows
- Add chart download controls with accessible filenames
- Add tests for mixed locale dates and accounting-style negative numbers
- Document a self-hosted deployment path and its privacy tradeoffs

## Decision intelligence

A deliberate push toward what mid-market finance and operations teams expect
from a decision-intelligence platform, tracked against HeronAI as the
reference point. The order matters: templates are useful immediately, and
everything under "connected data" depends on ADA being able to reach a source
at all.

**Templates** — a file gets the dashboard its shape deserves rather than the
generic one.

- [Prebuilt dashboard template library](https://github.com/saineshnakra/automated-data-analyst/issues/11) — the framework the rest depend on
- [AR aging dashboard](https://github.com/saineshnakra/automated-data-analyst/issues/12)
- [Profit and loss statement and P&L KPIs](https://github.com/saineshnakra/automated-data-analyst/issues/13)
- [Cash flow statement and cash flow KPIs](https://github.com/saineshnakra/automated-data-analyst/issues/14)
- [Balance sheet KPIs](https://github.com/saineshnakra/automated-data-analyst/issues/15)

**Connected data** — the manual export is the stale step in every analysis.

- [Analyze several files together](https://github.com/saineshnakra/automated-data-analyst/issues/16) — joins across uploads, with the match rate and any fan-out shown
- [Read from a data source, not just an upload](https://github.com/saineshnakra/automated-data-analyst/issues/17) — a read-only SQL or URL source before any vendor API
- [Refresh a saved analysis on a schedule](https://github.com/saineshnakra/automated-data-analyst/issues/18) — save the specification, never the rows

**Answers that go further** — without giving up the calculation under each one.

- [Build a whole dashboard from one question](https://github.com/saineshnakra/automated-data-analyst/issues/19)
- [Proactive risk and opportunity alerts](https://github.com/saineshnakra/automated-data-analyst/issues/20)
- [Approve an AI-planned query before it runs](https://github.com/saineshnakra/automated-data-analyst/issues/21)
- [Refuse questions ADA cannot actually read](https://github.com/saineshnakra/automated-data-analyst/issues/22)

### Where this stops

Two things a platform like HeronAI offers are not roadmap items here, and
saying so keeps the rest honest:

- **A catalogue of vendor integrations.** Hundreds of maintained connectors is
  an operating commitment, not a feature. One good generic source that a
  contributor can point at anything is worth more to this project than a long
  list that rots.
- **A human analytics team.** Part of that product is people who meet you and
  interpret the findings. ADA cannot ship that, so its answer has to be that
  the calculation is legible enough to interpret without them.

**A privacy claim that needs restating.** Today the README and
[docs/privacy.md](docs/privacy.md) say data never leaves the machine, and that
is exactly true because every analysis starts with a local file. A remote
source and a scheduled refresh both weaken it. Neither ships before the claim
is rewritten to say precisely what is true instead — that is a condition of
those issues, not a follow-up to them.

## Longer horizon

- User-defined metric contracts without SQL
- Multi-file comparison and period-over-period uploads
- Pluggable deterministic insight rules
- Evaluation corpus for narrative faithfulness and action quality
- Saved analysis specifications without persisting uploaded datasets

## Definition of done

A roadmap item is complete when it has a clear non-technical user outcome, deterministic evidence where applicable, graceful behavior on insufficient data, automated tests, and documentation of any privacy or cost change.
