# Periods and the trendline

**Code:** `aggregation.py`, `timeseries.py`
**Tests:** `tests/test_trend_series.py`, `tests/test_timeseries.py`

Two modules, one job: turn rows into an honest timeline, then fit a line through
it that a single bad month cannot bend.

`aggregation.py` is **the only place in ADA that decides what a period is.** The
trend chart, the anomaly detector, and the forecast all read that one decision,
so they cannot disagree on screen.

---

## Choosing the grain

The grain is `W` (weekly), `M` (monthly), or `Q` (quarterly). Two questions are
asked, and the **coarser** answer wins.

**1. How long is the span?**

| Span of your dates | Suggests |
|---|---|
| Up to 120 days | Weekly |
| Up to 900 days | Monthly |
| Longer | Quarterly |

**2. How often is data actually recorded?** (median gap between distinct dates)

| Typical gap | Suggests |
|---|---|
| Up to 10 days | Weekly |
| Up to 45 days | Monthly |
| Longer | Quarterly |

Why both? Five monthly readings spanning four months would be charted as
seventeen weeks by the span rule, twelve of them empty. Taking the coarser
answer stops ADA from inventing periods nobody measured.

## The two timeline adjustments

Both happen before any maths sees the numbers, and both are recorded on the
`TrendSeries` as human-readable `notes` shown under the chart.

### Excluding a partial trailing period

An export cut on the 12th of the month contains a third of that month. Charted
as-is, it looks like the business fell off a cliff.

ADA compares how far into each period the data reaches:

1. For every period, work out what fraction of it the data covers.
2. Take the median coverage of all periods **except the last**.
3. If the last period's coverage is more than **0.2 below** that median, drop it.

Needs at least **4 periods** to make the comparison. A merely quiet final week
is nowhere near 20 points below typical, so it is left alone — this only fires
on a genuinely truncated extract.

The note reads: *"Aug 2026 is still in progress (12 of 31 days) and is excluded,
so a half-finished period cannot read as a collapse."*

### Filling empty periods with zero

A month with no rows is a month with zero, not a month that did not happen.
Skipping it would shorten the timeline and flatten the slope. Filled periods are
counted and reported: *"3 periods with no rows counted as zero, keeping the
timeline evenly spaced."*

## The frames built here

| Function | Returns | Default limit |
|---|---|---|
| `build_trend` / `trend_frame` | Measure totalled per period | — |
| `segment_frame` | Measure per segment value | Top 12 |
| `segment_period_change` | Latest period-over-period change per segment | — |
| `driver_frame` | Movement waterfall inputs | Top 9 |
| `heatmap_frame` | Segment × period intensity | Top 8 segments |

The limits exist so charts stay readable. They are arguments, not constants —
callers can raise them.

---

## The trendline

**Code:** `timeseries.py`

### Theil–Sen slope

Take every pair of points, compute the slope between them, use the **median** of
all those slopes.

Compared with an ordinary least-squares line, one wild period cannot drag it.
Compared with the median of *consecutive* differences, it uses information from
every observation rather than adjacent pairs only — which matters on the short,
noisy histories a business file usually contains.

The pairwise search is O(n²), so above **800 points** the positions are thinned
by even sampling first. That bounds the work without changing the answer
meaningfully.

### Calendar positions

Periods are numbered by the calendar, not by row order. Consecutive periods are
1 apart, and a skipped period leaves a real hole in the fit.

Numbering by row order would silently shorten a timeline with gaps and flatten
the slope. Dividing elapsed days would drift, because months and quarters are
not equal lengths.

`period_grain` classifies observed spacing into `D`, `W`, `M`, `Q`, or `Y` using
the **median** gap, so one hole in an otherwise regular series cannot change the
verdict.

### Robust scale

`robust_scale` measures the typical size of the residuals — how far points
normally sit from the line. This is the unit anomalies and forecast bands are
expressed in.

It uses the median absolute deviation, multiplied by **1.4826** so the number is
comparable to a standard deviation for normal data.

There is one fallback. If more than half the residuals are identical, the median
absolute deviation is exactly zero, and every remaining period would look
infinitely surprising. In that case ADA uses the mean absolute deviation
instead. If that is zero too — a perfectly flat series — the scale is zero and
anomaly detection returns nothing rather than flagging everything.

## Edge cases

| Situation | Behavior |
|---|---|
| No date column | Empty trend frame; downstream steps skip |
| Fewer than 4 periods | Partial-period check does not run |
| Fewer than 2 points | Slope is 0, intercept is the value |
| All points identical | Slope 0, scale 0, no anomalies |
| More than 800 periods | Slope search thinned by even sampling |
