# Steel Strength Data 2018

[fuarresvij/steel-test-data](https://www.kaggle.com/datasets/fuarresvij/steel-test-data) on Kaggle · materials · 312 rows × 17 columns · licence: Other (specified in description) · round 2

312 steel alloys, one row per composition, with the weight percent of 13 alloying elements and three measured properties: yield strength and tensile strength (MPa) and elongation (%). Most are high-strength maraging-type steels (cobalt, nickel, molybdenum and titanium together).

**First run:** partly useful. Roles and correlations are right and the strength-ductility trade-off is found, but the top data checks are false alarms, nothing goes beyond pairwise r, and natural 'what predicts' questions are refused.

**Now:** good. All figures match, the false copy alarms are gone, and the joint model points to carbon, cobalt, aluminium, titanium and nitrogen as clear contributors; the quibble is that elongation, a co-measured property, sits in that model and leads the headline.

## What ADA finds now

This file lists 312 steel compositions with their measured strength and ductility. ADA finds the expected trade-off between strength and ductility (tensile strength vs elongation, r = -0.62), shows that tensile and yield strength track each other (r = 0.82), and fits a combined model in which carbon, cobalt, aluminium, titanium and nitrogen each go with higher tensile strength; every figure checked is correct. Its data checks sensibly note that several elements are reported at a floor of 0.01 and that some are zero in most alloys. The one weakness is that the combined model also uses elongation, which is a result of the same test rather than an ingredient, so it says less about composition than it appears to.

Headline: “Together, 10 readings explain 79% of the variance in tensile strength (n = 303), against 38% for elongation alone. With the others held fixed, elongation, co, c, al, ti and n stay clear.”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| The copied-values check fired on columns that only share a 0.01 reporting floor (v = nb, mn = si) and was ranked first. | medium | fixed |
| 'what predicts' and 'what drives' questions were refused. | medium | fixed |
| Descriptive statistics such as the standard deviation were refused. | medium | fixed |
| Only pairwise correlations were given, though elements travel together by alloy family. | medium | partly fixed |
| There was no outlier question, and the brief was silent when it found none. | low | fixed |
| Two high vanadium values were called 'often entry or unit errors' though they sit in a plausible tail. | low | fixed |

## Still open

- The headline model uses elongation, a result of the same tensile test, as a predictor of tensile strength, so the element coefficients are 'with ductility held fixed' rather than the effect of composition; the 13 elements alone explain 69%, not 79%.
- 'what predicts yield strength' ranks ti second (r = 0.51) although its rank correlation is 0.06, because 38% of ti values are zero; the data checks warn about this, but the answer does not.
- 'average tensile strength by co' says its three bins are 'of about equal size' when they hold 156, 81 and 75 rows (118 rows share co = 0.01), and calls the second bin highest though it ties the third (1,817.1 vs 1,815.1).

## Reproduce

Download the dataset from Kaggle and run `ada analyze "steel_strength.csv"`. The data is not stored in this repository.
