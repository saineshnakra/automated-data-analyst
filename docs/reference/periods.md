# Periods and the trendline

**Code:** `ada/aggregation.py`, `ada/timeseries.py`
**Tests:** `tests/test_aggregation.py`, `tests/test_timeseries.py`

`ada/aggregation.py` turns rows into a timeline, and `ada/timeseries.py` fits a
robust line through it.

`ada/aggregation.py` is the only module that decides what a period is. The
trend chart, anomaly detection, the forecast and Ask ADA all read that
decision.

---

## Choosing the grain

The grain is `W` (weekly), `M` (monthly), `Q` (quarterly) or `Y` (yearly). Two
rules each suggest a grain, and the coarser one wins.

**1. Span of the dates**

| Span | Suggests |
|---|---|
| Up to 120 days | Weekly |
| Up to 900 days | Monthly |
| Longer | Quarterly |

**2. Recording cadence** (median gap between distinct dates)

| Typical gap | Suggests |
|---|---|
| Fewer than 3 distinct dates | Weekly |
| Up to 10 days | Weekly |
| Up to 45 days | Monthly |
| Up to 140 days | Quarterly |
| Longer | Yearly |

Five monthly readings over four months would be seventeen weeks by span alone,
twelve of them empty. Taking the coarser grain keeps them monthly. Annual
readings stay yearly rather than becoming quarters with three empty periods
each year. That includes a table with a `Year` column and no date, which
cleaning dates to 1 January of each year (see
[Cleaning](cleaning.md#dates-from-years)).

Callers can fix the grain instead: `build_trend(..., frequency="W")`,
`ada.analyze(..., frequency="W")`, or a grain word in an Ask ADA question
("daily", "monthly"). A fixed grain is useful for scheduled jobs, so it does not
change between runs. With `ada.analyze`, the fixed grain applies everywhere:
the headline, the trend, driver and anomaly findings, `anomalies` and
`forecast()` all use the same periods.

## How each period's value is computed

| Measure | Period value |
|---|---|
| Summed (an amount) | Sum of the period's rows; a period whose values are all missing stays empty |
| Averaged (a rate or level) | Mean of the period's rows |
| None | Row count, or distinct values of a `count_column` |

See [Concepts](../concepts.md#amounts-rates-and-levels) for which measures are
averaged.

## Timeline adjustments

Both are made before any calculation and recorded on the `TrendSeries` as
`notes`, which the app shows under the chart.

### Excluding a partial trailing period

An export cut on the 12th of a month contains about a third of that month, and
charted as-is it looks like a sharp drop. ADA compares how far into each
period the data reaches:

1. For each period, compute the fraction of it the data covers (the last date
   seen, relative to the period's start and end).
2. Take the median coverage of all periods except the last.
3. If the last period's coverage is more than 0.2 below that median
   (`PARTIAL_COVERAGE_MARGIN`), drop it, provided at least two periods remain.

This needs at least 4 periods (`MIN_PERIODS_FOR_PARTIAL_CHECK`). With fewer,
the check does not run and the latest period is described as "so far" rather
than "complete".

The note reads: *"Aug 2026 is still in progress (12 of 31 days) and is
excluded, so a half-finished period cannot read as a collapse."*

If the last period is more than 0.05 below the median but within 0.2
(`SHORT_COVERAGE_MARGIN`), it is kept and a note says part of its shortfall may
be missing data: *"The last period only reaches 26 of 30 days, ..."*. A cut
late in the month and a few quiet days look the same at this level.

### Filling empty periods

A period with no rows is inserted so the timeline stays evenly spaced:

- with zero for a summed measure or a row count, since no rows means none
  happened
- left empty (NaN) for an averaged measure, which has no observation to
  average; anomaly detection and the forecast skip these periods

Filling runs only when there are at least 3 distinct dates. Filled periods are
counted in a note, for example *"3 periods with no rows counted as zero,
keeping the timeline evenly spaced."*

## Frames built here

| Function | Returns | Default limit |
|---|---|---|
| `build_trend` / `trend_frame` | The measure per period (`TrendSeries` / its frame) | – |
| `segment_frame` | The measure per segment value | Top 12 |
| `segment_period_change` | Latest period-over-period change per segment | – |
| `driver_frame` | Movement waterfall inputs, with an "Other segments" bar for summed measures | Top 9 |
| `heatmap_frame` | Segment × period matrix; empty cells are zero for a summed measure and blank for an averaged one | Top 8 segments |

Rows with no segment value are kept under the label `(not recorded)`, so the
segment values add up to the total. The limits are arguments and can be
raised.

---

## The trendline

**Code:** `ada/timeseries.py`

### Theil-Sen slope

Compute the slope between every pair of points and take the median. One
extreme period cannot pull the line the way it pulls a least-squares fit, and
every observation is used, not only adjacent pairs.

The pairwise search is O(n²), so above 800 points (`MAX_PAIRWISE_POINTS`) the
positions are thinned by even sampling first.

### Calendar positions

Periods are numbered by the calendar, not by row order. Consecutive periods
are 1 apart, and a skipped period leaves a gap in the fit. Numbering by row
order would shorten a timeline with gaps and flatten the slope; dividing
elapsed days would drift, because months and quarters differ in length.

`period_grain` classifies the median spacing between periods as `D` (up to 1.5
days), `W` (up to 10), `M` (up to 45), `Q` (up to 130) or `Y`. Using the median
means one gap in a regular series does not change the result.

### Robust scale

`robust_scale` estimates the typical distance of points from the line. Anomaly
thresholds and forecast bands are expressed in this unit.

It is the median absolute deviation multiplied by 1.4826, which matches a
standard deviation for normal data. If more than half the residuals are
identical, the median absolute deviation is zero, so the mean absolute
deviation multiplied by 1.2533 is used instead (the same scale). If that is
zero too, as for a perfectly flat series, the scale is zero and anomaly
detection flags nothing.

## Edge cases

| Situation | Behavior |
|---|---|
| No date column | Empty trend frame; trend, anomalies and forecast are skipped |
| Fewer than 4 periods | Partial-period check does not run |
| Fewer than 3 distinct dates | Empty periods are not filled |
| Fewer than 2 points | Slope 0, intercept equal to the value |
| All points identical | Slope 0, scale 0, no anomalies |
| More than 800 periods | Slope search thinned by even sampling |
