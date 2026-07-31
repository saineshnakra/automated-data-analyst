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

## Longer horizon

- User-defined metric contracts without SQL
- Multi-file comparison and period-over-period uploads
- Pluggable deterministic insight rules
- Evaluation corpus for narrative faithfulness and action quality
- Saved analysis specifications without persisting uploaded datasets

## Definition of done

A roadmap item is complete when it has a clear non-technical user outcome, deterministic evidence where applicable, graceful behavior on insufficient data, automated tests, and documentation of any privacy or cost change.
