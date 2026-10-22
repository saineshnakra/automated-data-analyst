# Anemia Dataset

[biswaranjanrao/anemia-dataset](https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset) on Kaggle · clinical · 1,421 rows × 6 columns · licence: Unknown · round 2

1,421 rows of haemoglobin and red-cell indices (MCH, MCHC, MCV) with gender (0/1) and an anaemia label (Result, 0/1); only 534 rows are distinct. The label is a haemoglobin cut-off: below 12.0 for gender 0 and below 13.5 for gender 1.

**First run:** partly useful. The numbers are right and the null result for MCV is fair, but the headline is circular (Result is a haemoglobin cut-off) and the fact that 62% of the rows are copies is waved through.

**Now:** good. The brief states that the label is a haemoglobin cut-off, sets aside the copied rows, flags the table as probably simulated and correctly finds nothing else that differs; the quibbles are a bland headline and Ask answers that still treat the cut-off as a finding.

## What ADA finds now

This file has haemoglobin and red-cell measurements with an anaemia label for 1,421 rows, of which only 534 are distinct. ADA works out that the anaemia label is simply a haemoglobin cut-off (12.0 for one gender, 13.5 for the other), sets aside the 887 copied rows before testing, and warns that three of the four blood measurements are spread flat and unrelated, as generated data would be. It reports that the anaemia rate differs by gender (57% vs 35%) and correctly explains that this gap is built into the different cut-offs. Its headline is only the average haemoglobin, and when asked directly it still presents the haemoglobin gap between anaemic and non-anaemic rows as a large difference, although that gap holds by definition.

Headline: “Hemoglobin averages 13.41 across all rows.”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| A 0/1 label that is a fixed haemoglobin cut-off (per gender) was treated as an independent grouping, so 'Hemoglobin is higher in Result 0' led the brief though it holds by definition. | high | partly fixed |
| 62% of rows are exact copies, kept with only a cleaning note, so every test used an inflated n. | high | fixed |
| 'share of Result 1 by Gender' was answered as the gender split of anaemic rows, not the rate within each gender. | medium | fixed |
| A 0/1 outcome was never compared across a category as rates with a chi-squared test. | medium | fixed |
| The entropy tie-break picked MCV rather than Hemoglobin, the reading the label is built from. | medium | fixed |
| Flat distributions and readings that should correlate but don't (signs of generated data) were not mentioned. | low | fixed |

## Still open

- Asked directly, ADA still reports the haemoglobin gap between anaemic and non-anaemic rows as a large finding (Hedges' g = 2.62), although the label is a haemoglobin cut-off, as the brief itself says.
- The headline is a bare average (13.41) that counts the 887 copied rows, while the rest of the brief uses the 534 distinct rows (mean 13.29); the real story, a probably simulated table where nothing differs beyond the definition, is in the summary line instead.
- The duplicate check says 'one row appears 3 times'; in fact 415 distinct rows appear three times and 57 twice.
- Simple counts in Ask still include the copies ('how many patients have Result 1' -> 620 of 1,421; 247 of the 534 distinct rows).

## Reproduce

Download the dataset from Kaggle and run `ada analyze "anemia.csv"`. The data is not stored in this repository.
