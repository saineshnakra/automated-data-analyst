# Concepts

The terms used throughout the docs.

## Column roles

ADA assigns columns to four roles before it analyzes a file. All four are
detected automatically and can be overridden in the app or in `ada.analyze`.

| Role | Meaning | Example columns |
|---|---|---|
| Measure | The number being analyzed | `Revenue`, `Units`, `Viability %` |
| Date | When each row happened | `Order Date`, `Created At` |
| Segment (also *dimension*) | How the measure is split into groups | `Region`, `Product`, `Channel` |
| Identifier | Names a row; not a quantity | `Order ID`, `Invoice Number` |

The identifier role mainly keeps columns such as `Order Number` out of the
measure slot: they are numeric, but their sum means nothing.

See [Schema detection](reference/schema-detection.md).

## Amounts, rates and levels

A measure is either summed or averaged, decided from its column name by
`resolve_metric` in `ada/metrics.py`:

- A rate (a name with a `%` sign, or ending in `rate`, `margin` or `ratio`) is
  averaged.
- A measured level (a temperature, pH, concentration, score and similar, or a
  unit annotation such as `(mg/L)` or `(°C)`) is averaged. `LEVEL_WORDS` in
  `ada/metrics.py` holds the word list.
- Everything else is an amount and is summed.
- With no measure, ADA counts rows.

An averaged measure has no total, so ADA quotes no shares, concentration card
or "Other segments" bar for it. A change is reported as a percentage for an
amount or count, in percentage points for a rate, and as a plain difference for
a level.

## Period and grain

A period is one bucket on the timeline. The grain is its size: `W` (week), `M`
(month), `Q` (quarter) or `Y` (year). Callers can also fix the grain, including
`D` (day).

ADA picks the grain once per analysis, from the span of the dates and how often
rows are recorded, and every later step reuses it.

Two adjustments are made before any calculation:

- A final period that the data only partly covers is excluded, so an export cut
  on the 12th of a month does not read as a drop.
- A period with no rows is filled, so the timeline stays evenly spaced: with
  zero for a summed measure, and left empty for an averaged one, which has
  nothing to average.

Both are reported in a note under the chart. See [Periods](reference/periods.md).

## Trend series

The measure per period, plus the notes about those adjustments
(`TrendSeries` in `ada/aggregation.py`). It is the input to the trend chart,
anomaly detection, the forecast and Ask ADA's trend answers.

## Trendline

A straight line through the periods, used as the expected level. ADA fits a
Theil-Sen line: the median of the slopes between every pair of points, so one
extreme period has little effect on it.

## Anomaly

A period whose distance from the trendline exceeds a calibrated band. The band
width comes from simulation rather than a fixed number of deviations, set so
that a stable series raises a false alarm in about one analysis in twenty. See
[Anomalies](reference/anomalies.md).

## Evidence vs recommendation

- **Evidence** is a calculated fact about the data, shown with the calculation
  that produced it. Example: "Revenue fell 12.4% in July."
- **A recommendation** is a suggestion about what to look at next. It is
  labelled as interpretation and never presented as a cause. Example: "Check
  what changed in the West region."

ADA does not claim that one thing caused another.

## Query plan

A plain-English question is turned into a `QueryPlan`: a fixed structure naming
the measure, grouping, filters and aggregation. pandas executes the plan
locally, and the answer shows the calculation. No generated code runs. See
[Ask ADA](reference/ask-ada.md).

## Deterministic path and optional AI

- **Deterministic path.** Everything is computed in-process with pandas, with
  no network call and no API key. On the hosted demo, the uploaded file is sent
  to the Streamlit server and analyzed there.
- **Optional AI.** Two features that need an API key: a query planner for
  questions the rules cannot parse, and a written AI summary. They receive
  column names, types and already-computed evidence, which can include the
  segment names an evidence sentence mentions. They do not receive rows. A
  planned query runs only after the user approves it.

See [The optional AI layer](reference/ai-layer.md) and [Privacy](privacy.md).
