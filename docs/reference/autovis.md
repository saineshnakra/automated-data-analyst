# Choosing the chart

**Code:** `ada/autovis.py`, rendered by `ui.render_explore` · **Tests:** `tests/test_autovis.py`

The Explore tab lets the user pick any columns. ADA chooses a chart form for
them, draws it, and prints the reason underneath.

`recommend_chart` draws nothing and chooses no colours. It returns a
`ChartSpec`: a form, the column for each channel, and a sentence explaining the
choice, so the decision can be tested without a browser.

## Form by selection

Columns are classified as date, numeric (measure) or category, and the first
matching row below applies. Values are aggregated the way the measure is
(a mean for a rate or level, a sum otherwise; see
[Concepts](../concepts.md#amounts-rates-and-levels)).

| Selection | Form | Reason |
|---|---|---|
| Date + measure + category | Line per category | Series have to be told apart over time |
| Date + measure | Area | One series; the shape of the movement matters |
| Two categories + measure | Heatmap; a table above 4,000 cells | Magnitude across a grid |
| Category + measure | Columns; horizontal bars above 6 categories or for labels over 12 characters; a table above 25 categories | Magnitude comparison |
| Two measures | Scatter | Whether the two move together |
| Date only | Records per period (area) | With no measure, frequency is what can be plotted |
| One measure | Histogram; a single stat when there is at most one value | How the values are spread |
| One category | Count per category; a table above 25 categories | How many records fall in each |

## Rules

**One y-axis.** When two measures are selected with a date, only the first is
charted and a note says so. Two scales on one chart make lines look comparable
when they are not.

**At most four series.** `fold_small_series` keeps the four largest series by
total and groups the rest as "Other" (or "Other (2)" if the column already has
an "Other"). A fifth hue would be hard to tell from the existing four for a
colour-blind reader.

**One hue for magnitude, several only for identity.** Bars and columns that
compare sizes use a single hue. Distinct hues are used only when the series
themselves are being compared.

**The whole date range.** A date axis carries at most 400 points
(`EXPLORE_POINTS` in `ada/ui.py`). When there are more dates than that, they
are grouped by the finest period that fits, stepping up from days to weeks,
months, quarters and years (`_fit_date_axis`), and a caption says what each
point is. A monthly series from 1958 to 2020 is drawn by quarter from its first
reading, rather than as its latest 400 months. If even years don't fit, the
latest 400 are drawn and the caption says how many periods are shown out of how
many. A line per category keeps every series at each date it draws.

## Colours

The data-mark palette in `ada/ui.py` is checked against the light chart
surface for lightness, a minimum chroma, colour-vision separation and 3:1
contrast.

| Slot | Value | Use |
|---|---|---|
| `SERIES_COLORS` | `#635BFF` `#0E8F6E` `#B5761B` `#D64A73` | Series identity, assigned in order, never cycled |
| `LEAF` | `#5C8A1B` | Single-hue magnitude (columns and bars) |
| `RISE` / `FALL` | `#1B6FB5` / `#B5761B` | Up and down in the waterfall; blue and orange stay distinct under deuteranopia |
| `ACCENT` | `#635BFF` | Single-series area, histogram and scatter |

`LIME` (`#C7F36B`) is the brand accent for surfaces and text. At 1.24:1
against the chart surface it is too light for a data mark.

## Changing this

A new form needs a branch in `recommend_chart`, a matching branch in
`ui._explore_figure`, a rationale sentence, and a test asserting the form for a
representative selection. A new data colour must pass the same checks as the
existing palette.
