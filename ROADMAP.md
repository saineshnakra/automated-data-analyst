# ADA roadmap

The roadmap favors analytical depth and trust over a larger pile of charts. Items are deliberately scoped so contributors can own one outcome end to end.

## Recently delivered

- **Ask ADA** — plain-English questions parsed into auditable query plans and executed locally, with an optional schema-only AI planner fallback
- **Anomaly radar** — robust trendline detection with flagged periods on the chart, in evidence, and in recommendations
- **Forecast guardrails** — a baseline offered only with sufficient history, capped horizon, seasonality, and a backtested error shown beside the chart
- **Drill-down focus** — one segment value filters the whole product and regroups by the next useful dimension
- **Movement waterfall and intensity heatmap** — the latest change reconciled by segment, and the measure over segment × period
- **Worksheet selection** — analyze any sheet of a multi-sheet workbook

## Near term

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
