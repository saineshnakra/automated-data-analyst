# Concepts

The words ADA uses, in plain English. Everything else in the docs assumes these.

## The four column roles

ADA cannot analyze a file until it decides what the columns mean. It looks for
four roles. All four are guesses, and all four can be overridden in the app.

| Role | Plain meaning | Example columns |
|---|---|---|
| **Measure** | The number you care about | `Revenue`, `Units`, `Cost` |
| **Date** | When each row happened | `Order Date`, `Created At` |
| **Segment** (also called *dimension*) | How you split the measure into parts | `Region`, `Product`, `Channel` |
| **Identifier** | Names a row, but is not worth adding up | `Order ID`, `Invoice Number` |

The identifier role exists mostly to keep things *out* of the measure slot.
`Order Number` is numeric, but summing it is meaningless.

See [Schema detection](reference/schema-detection.md) for how the guess is made.

## Period and grain

A **period** is one bucket on the timeline: one week, one month, one quarter.
The **grain** is which of those ADA picked — `W`, `M`, or `Q`.

ADA picks the grain once, from your dates, and every later step reuses it. That
matters: if the trend chart says months and the forecast says weeks, the two
disagree without ever saying so.

Two things happen to the timeline before any maths runs:

- A final period the data only partly covers is **excluded**. An export cut on
  the 12th of the month would otherwise look like revenue collapsed.
- A period with no rows is counted as **zero**, not skipped, so the timeline
  stays evenly spaced.

Both adjustments are written under the chart rather than applied silently.

See [Periods](reference/periods.md).

## Trend series

The measure totalled per period, plus notes about the two adjustments above.
This is the single input to the trend chart, anomaly detection, and the
forecast, so those three can never drift apart.

## Trendline

A straight line fitted through the periods, used as the "expected" level.

ADA uses a **Theil–Sen** line: take every pair of points, work out the slope
between them, and use the median of all those slopes. One wild month cannot
drag it around the way an ordinary best-fit line can.

## Anomaly

A period whose distance from the trendline is bigger than a calibrated band.

"Calibrated" means the band width was *measured* by simulation, not picked by
hand, so that a genuinely stable series raises a false alarm about once in
twenty analyses. See [Anomalies](reference/anomalies.md).

## Evidence vs recommendation

ADA keeps these apart on purpose.

- **Evidence** is something ADA calculated. It is a fact about your data, and it
  ships with the exact calculation behind it. "Revenue fell 12.4% in July."
- **Recommendation** is ADA's interpretation. It is a suggestion about what to
  look at next, and it is never presented as proof of cause. "Check what changed
  in the West region."

Nothing in ADA claims that A caused B.

## Query plan

When you ask a question in plain English, ADA does not run generated code. It
turns your question into a **QueryPlan** — a small, fixed structure saying which
measure, which grouping, which filters, which aggregation.

That plan is then executed locally with pandas. You can read the plan, and the
answer shows the calculation. See [Ask ADA](reference/ask-ada.md).

## Deterministic vs optional AI

- **Deterministic** — everything computed with pandas in-process. No network,
  no API key. This is the whole product. (On the hosted demo the upload itself
  reaches a Streamlit server; the analysis still runs there and nowhere else.)
- **Optional AI** — two narrow extras that need an API key. They receive column
  names, types, and already-computed evidence — including the segment names an
  evidence sentence mentions. They never receive your rows.

See [The optional AI layer](reference/ai-layer.md) and [Privacy](privacy.md).
