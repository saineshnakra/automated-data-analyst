# Choosing the chart

**Code:** `ada/autovis.py` · **Tests:** `tests/test_autovis.py`

For any set of columns a reader picks, ADA chooses a chart form and says why.
The browser app in ADA Lab draws the chart on its Explore tab, with the reason
underneath.

`recommend_chart` draws nothing and chooses no colours. It returns a
`ChartSpec`: a form, the column for each channel, and a sentence explaining the
choice, so the decision can be tested without a browser.

## Form by selection

Columns are classified as date, numeric (measure) or category, and the first
matching row below applies. Values are aggregated the way the measure is
(a sum for money and counts, a mean otherwise, or the `aggregation` override
when the charted column is the active measure; see
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

## Changing this

A new form needs a branch in `recommend_chart`, a rationale sentence, and a
test asserting the form for a representative selection. ADA Lab's app draws
each form, so a new one needs a matching change there.
