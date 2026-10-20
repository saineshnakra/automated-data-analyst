# Selecting a slice on a chart

**Code:** `selection.py` · orchestrated by `pipeline.analyze_selection` · rendered by `ui.render_explore` · **Tests:** `tests/test_selection.py`

The drill-down dropdown answers "show me West". It cannot answer "those ones":
the cluster in the scatter, the four months on the trend. On the Explore tab a
reader can now select directly on the chart, and ADA analyzes just that slice,
set against the whole view.

## A selection is a filter, not a finding

The reader chose the slice; ADA did not discover it. Every sentence on the
panel says so — "the slice you picked", "records you selected" — and nothing
implies the grouping was found in the data. One is a filter, the other would be
a claim.

## Gestures

| Chart form | Gesture | Becomes |
|---|---|---|
| Scatter | Box, or lasso from the chart toolbar | A range on each axis, or the lasso polygon |
| Line / area over time | Drag across the time axis | A date range, widened to whole periods |
| Column / bar | Click a bar, shift-click for more | The clicked categories |
| Histogram | Drag across the axis | A value range |
| Heatmap, table, stat | — | No selection; none maps back to rows honestly |

The filter is built from the geometry the reader drew, not from the indices of
the plotted points. A trend or a bar chart plots aggregates, so a point index
says nothing about which rows are behind it, while "dates from March to June"
does — and that is also what the reader can check against the axis.

A box drawn over a trend only uses its horizontal extent. The vertical extent
is a range of period totals, not of records.

## Rules

**The slice is stated in words.** `describe_selection` writes the count and
every filter in the column's own units — "447 of 1,800 records you selected:
Order Date from 01 Apr 2025 to 30 Sep 2025" — so a screenshot still explains
itself.

**Time windows are widened to whole periods, and say so.** A window from the
19th of one month to the 8th of another leaves both end months part-covered,
and a third of a month reads as a collapse to every trend and anomaly check.
The window is widened to the grain `aggregation.preferred_frequency` picks for
the rows inside it, so the slice and its trend agree on what a period is. The
dates the reader drew are printed beside the widened ones.

**The same evidence machinery, not a parallel path.** The slice goes through
`analyze_business` like any view. Each evidence card for the slice carries the
result of the same check on the whole view.

**Never shown alone.** Every slice figure — records, total, average, the most
common segment — sits beside the same figure for the whole view, with its
calculation.

**Too small says so.** Below `MIN_SLICE_RECORDS` (5) an average or a segment
mix is mostly noise. The count and total are still reported, because they are
arithmetic; the statistics are replaced by a note saying how many records
would be needed. Evidence is not computed at all.

**A rate is averaged, never totalled.** `metrics.resolve_metric` decides, as it
does everywhere else.

**Clearing restores the whole exactly.** An empty selection returns the frame
untouched, and the rest of the product never sees the slice: the brief,
dashboard, chat and exports keep analyzing the whole view throughout.

## Changing this

A new chart form that should accept a selection needs an entry in
`SELECTION_MODES`, a branch in `selection_from_event` that maps its gesture to
filters over raw rows, and a test asserting the rows it keeps. Keep the
wording rule: if a sentence could be read as ADA having found the grouping,
rewrite it.
