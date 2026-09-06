# Choosing the chart

**Code:** `autovis.py` · rendered by `ui.render_explore` · **Tests:** `tests/test_autovis.py`

The Explore tab lets a reader pick any columns. ADA works out which chart form
the data calls for, draws it, and prints the reason underneath — the same
contract the numbers have.

`recommend_chart` decides nothing about colour and draws nothing. It returns a
`ChartSpec` — a form, the columns to put on each channel, and a sentence
explaining the choice — so the decision can be tested without a browser.

## The rule: the data's job picks the form

| What was selected | Form | Because |
|---|---|---|
| Date + measure | Area | One thing to follow; the shape of the movement is the point |
| Date + measure + segment | Line per segment | The reader has to tell series apart over time |
| Category + measure | Column, or **bar** past 6 categories or long names | Magnitude comparison; a vertical axis cannot hold long labels |
| Two categories + measure | Heatmap | Magnitude across a grid |
| Two measures | Scatter | The question two measures ask is whether they move together |
| One measure | Distribution | A lone measure asks how its values are spread |
| One category | Count | How many records fall in each |
| Date only | Records per period | With no measure, frequency is the only honest thing to plot |
| A single value | **Stat, not a chart** | A one-bar bar chart carries no more than the figure |
| More than 25 categories | **Table** | Past that nobody reads the bars |

## Three rules that are not negotiable

**Never two y-axes.** Selecting two measures alongside a date charts the first
and says so in a note. Two scales on one chart make the lines comparable by
accident rather than by fact.

**Series past the fourth fold into "Other".** `fold_small_series` keeps the
largest four and groups the tail. A generated fifth hue is indistinguishable
from an existing one to a colour-blind reader, so the honest move is to stop
drawing series rather than to invent colours.

**One hue for magnitude, several only for identity.** A bar chart comparing
sizes uses a single hue. Distinct hues are spent only when the series
themselves are the subject.

## Colours

The data-mark palette in `ui.py` is checked rather than chosen by eye, against
the light chart surface, for the lightness band, a chroma floor, colour-vision
separation and 3:1 contrast:

| Slot | Value | Job |
|---|---|---|
| Series | `#635BFF` `#0E8F6E` `#B5761B` `#D64A73` | Identity, assigned in order, never cycled |
| `LEAF` | `#5C8A1B` | Single-hue magnitude |
| `RISE` / `FALL` | `#1B6FB5` / `#B5761B` | Polarity, up versus down |

Two of these replaced values that failed. The old chart lime sat at **1.24:1**
against the surface, and the old up/down green and red separated by **5.4**
under deuteranopia against a floor of 8 — the one pair a red-green colour-blind
reader cannot tell apart. `LIME` still exists for surfaces and text, where the
contrast rule for data marks does not apply.

## Changing this

A new form needs a branch in `recommend_chart`, a matching branch in
`ui._explore_figure`, a rationale sentence, and a test asserting the form for a
representative selection. Adding a colour means running the palette validator
first — the current values pass, and a slot that fails is a defect rather than
a preference.
