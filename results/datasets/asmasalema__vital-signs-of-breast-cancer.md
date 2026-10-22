# vital signs of breast cancer

[asmasalema/vital-signs-of-breast-cancer](https://www.kaggle.com/datasets/asmasalema/vital-signs-of-breast-cancer) on Kaggle · clinical · 2,435 rows × 23 columns · licence: Other (specified in description) · round 2

2,435 patients from an oncology centre, with age, home town, 19 blood tests (glucose, kidney and liver markers, electrolytes, blood counts, ESR, LDH), the tumour markers CA15-3 and CEA, and a 0/1 class label (1,084 in class 1). Many rows hold physiologically impossible values, almost all of them in class 1.

**First run:** partly useful. The roles and arithmetic are right and the group comparison is where a clinician would start, but ADA reports huge effects without noticing that 44% of patients have impossible sodium, potassium, red-cell or albumin values, which is the main thing to know about this file.

**Now:** partly useful. The arithmetic is right and the data checks now catch the impossible sodium, potassium, calcium, red-cell and albumin values, but the brief still presents near-perfect class separations as ordinary large differences without saying the impossible values sit almost entirely in class 1.

## What ADA finds now

This file holds blood tests and tumour markers for 2,435 patients split into two classes. The analysis reports that the tumour markers CA15-3 and CEA, white cells, platelets and haemoglobin all differ sharply between the classes, and it shows that the CA15-3 and CEA correlation is much weaker within each class than overall. It now flags thousands of impossible readings, such as red-cell counts up to 34 and sodium up to 279, and leaves them out of its comparisons. What it does not say is that almost all of these impossible readings belong to class 1, or that blood counts separating the classes this cleanly are not credible, so a reader may take the large differences at face value.

Headline: “CA15 is higher in CLASS 1 than in CLASS 0: mean 51.26 vs 16.01 (n = 1,084 and 1,351).”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| Physiologically impossible lab values were not flagged, so the main findings rested on them unnoticed. | high | partly fixed |
| A near-perfect separation was reported as a plain 'large difference', with no caution. | medium | not fixed |
| Pooled correlations inflated by the class split were reported without a within-group check. | medium | fixed |
| Breakdown answers led with a group of one and did not show n. | medium | fixed |
| 'Which X best separates Y' and 'within' as a filter word were not understood. | low | fixed |
| No discrimination summary (AUC) for a labelled 0/1 clinical table. | low | not fixed |

## Still open

- The checks flag the impossible values but never say that 94% of the affected patients (1,027 of 1,097) are in class 1, which is the main reason to doubt the class comparison.
- Near-perfect separations (CA15, CEA, white cells, platelets and haemoglobin all with AUC about 0.95) are reported as plain 'large differences' with no caution that real blood counts do not split cancer classes this cleanly.
- No AUC is given for a 0/1 clinical label, the figure clinicians expect.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "BC.csv"`. The data is not stored in this repository.
