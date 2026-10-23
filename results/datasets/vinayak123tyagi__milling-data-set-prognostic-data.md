# NASA Milling Dataset

[vinayak123tyagi/milling-data-set-prognostic-data](https://www.kaggle.com/datasets/vinayak123tyagi/milling-data-set-prognostic-data) on Kaggle · sensor · 167 rows × 14 columns · licence: U.S. Government Works · round 3

NASA tool-wear experiments: 16 cases from a 2 x 2 x 2 design (depth of cut, feed and material, each run twice), each followed over successive runs (167 rows) with cutting time, measured flank wear VB (21 missing) and one summary value per run from six sensors.

**First run:** poor. The arithmetic is right but the report compares design settings and experiment numbers and never looks at tool wear, the one outcome in the table.

**Now:** partly useful. The report now studies wear against cutting time with experiments as units and its model finds the material effect, but the headline compares the weaker factor and the direct question about material gets a misleading 'no clear difference'.

## What ADA finds now

This table follows tool wear over cutting time in 16 milling experiments that vary depth of cut, feed and material. The tool now treats wear as the outcome, recognises each experiment as a unit followed over time and compares groups only over the stretch of time they share. It shows that wear rises with cutting time, strongly within each material, and its combined model finds that the second material wears about 0.32 units more at the same cutting time. But the headline compares depth of cut, a minor factor, and asked directly whether wear differs by material it answers 'no clear difference', because that answer does not line up cutting times. It never reports a wear rate per experiment, the natural summary for this kind of data.

Headline: “The DOC groups were followed for different lengths along time (DOC 0.75 to 105 and DOC 1.5 to 57); over the stretch they share (time 0 to 57), VB averages DOC 0.75 0.305 and DOC 1.5 0.326.”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| A design setting (feed) was chosen as the measure although the measured outcome VB is in the table. | high | fixed |
| The experiment number (case) was read as a numeric reading and became the headline. | high | fixed |
| The table is a set of wear curves over cutting time, but rows were treated as independent and averaged over time, hiding the effects of the cutting conditions. | high | partly fixed |
| The joint model took regressors only from the correlation screen, so it included the case number and left out the design factors. | medium | partly fixed |
| A difference in run length caused by the stopping rule was reported as a difference between materials. | medium | fixed |
| Adjusted comparisons and 'which columns relate most to VB' were refused. | low | fixed |

## Still open

- Asked 'does VB differ by material', it answers 'no clear difference' (per-case medians 0.328 vs 0.265, p = 0.33) without lining up cutting time, although material-1 experiments ran about four times longer; at matched time material 2 shows about three times the wear, and the time-adjusted answer is +0.32 (p < 0.001).
- The headline compares depth of cut, which explains 9% of wear, instead of material, which explains 61%, and gives two averages with no test.
- Wear is never summarised as a rate per experiment, so the main result (the second material wears about four times faster per minute) is only implied by a model coefficient.
- The scan's adjustment for 20 tests on 16 experiments labels the material gap (g = 2.18) 'no clear difference'.
- The joint model leaves out feed and depth of cut and does not say its 146 rows come from 16 experiments.
- The 'VB over time' card averages different experiments at each time point, so its peak at time 42 mixes experiments.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "mill.csv"`. The data is not stored in this repository.
