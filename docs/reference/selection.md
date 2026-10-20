# Selecting a slice on a chart

**Code:** `ada/selection.py`, run by `pipeline.analyze_selection` and drawn by `ui.render_explore` · **Tests:** `tests/test_selection.py`

On the Explore tab you can select part of a chart (a cluster in a scatter, a
few months on a trend, some bars) and ADA analyzes just those rows beside the
whole view. The drill-down dropdown filters by one value; a selection filters
by what you drew.

A selection is a filter you chose, not a grouping ADA found, and the panel
words it that way ("the slice you picked", "records you selected").

## Gestures

| Chart form | Gesture | Becomes |
|---|---|---|
| Scatter | Box, or lasso from the chart toolbar | A range on each axis, or the lasso polygon |
| Line or area over time | Drag across the time axis | A date range, narrowed to whole periods |
| Column or bar | Click a bar; shift-click adds more | The clicked categories |
| Histogram | Drag across the axis | A value range |
| Heatmap, table, stat | None | A cell doesn't map back to rows |

Filters come from the drawn geometry, not from the indices of plotted points: a
trend or bar chart plots aggregates, so an index doesn't identify the rows
behind it. A box over a trend uses only its horizontal extent; its vertical
extent is a range of plotted totals.

## Rules

**The slice is stated in words.** `describe_selection` gives the count and each
filter in the column's own units, for example "447 of 1,800 records you
selected: Order Date from 01 May 2025 to 31 Aug 2025".

**A drawn time window is narrowed to whole periods.** A window from the 19th of
one month to the 8th of another leaves both end months part-covered, and a
part-covered month reads as a fall in the trend checks. The window is narrowed
to the whole periods inside it, at the grain `aggregation.preferred_frequency`
picks for those rows, and the panel prints the dates you drew. Rows outside
the drawn window are never added. If no whole period fits, the window is kept
as drawn. Clicked points keep the exact range clicked.

**The slice uses the same evidence code.** It goes through `analyze_business`
like any view, and each evidence card for the slice shows the same card for the
whole view.

**Every figure has the whole view beside it.** Records, total, average and the
most common segment are shown for both, with the calculation. The average of a
rate differs from the whole in percentage points, and a level (temperature,
pH) by a plain difference. The segment mix is left out when the selection
itself filters that column, since it would always be 100%.

**Small slices get a note instead of statistics.** Below `MIN_SLICE_RECORDS`
(5) only the count and the total are reported, and no evidence is computed.

**Clearing restores the whole view.** An empty selection returns the frame
unchanged, and the brief, dashboard, chat and exports always use the whole view.

## Changing this

A new chart form that accepts a selection needs an entry in `SELECTION_MODES`,
a branch in `selection_from_event` that maps its gesture to filters over raw
rows, and a test for the rows it keeps. Keep the wording rule: no sentence
should suggest ADA found the grouping.
