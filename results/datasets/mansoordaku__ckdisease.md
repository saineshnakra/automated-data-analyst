# Chronic KIdney Disease dataset

[mansoordaku/ckdisease](https://www.kaggle.com/datasets/mansoordaku/ckdisease) on Kaggle · clinical · 400 rows × 26 columns · licence: Unknown · round 3

The classic chronic kidney disease table: 400 patients (250 with the disease, 150 without) with age, blood pressure, urine dipstick results, blood chemistry and yes/no comorbidities. Values carry stray tabs and '?' markers.

**First run:** good. The reading, cleaning and arithmetic are right and the scan finds the strongest separating readings, but it takes an obviously constructed control group at face value.

**Now:** good. The headline now uses the reading that best separates the groups, every figure checks out, and the tool itself warns that the healthy group looks constructed; the main gap is the missing ranking or model for predicting the diagnosis.

## What ADA finds now

This is the classic kidney-disease table of 400 patients. The tool cleans its stray tabs and question marks correctly and leads with the clearest difference: haemoglobin is far lower in patients with kidney disease (10.7 against 15.2). It now warns that every healthy patient sits inside narrow normal ranges and that haemoglobin and packed cell volume are unrelated in that group, so the healthy group may have been built from reference values and the differences partly restate how it was made. It flags impossible creatinine, potassium and sodium entries and treats blood pressure as a measurement. It does not rank the readings by how well they predict the diagnosis or fit a combined model, and it refuses the question 'what predicts classification'.

Headline: “hemo is higher in notckd than in ckd: mean 15.19 vs 10.65 (n = 144 and 204).”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| The notckd group looks constructed (every value inside a normal band, haemoglobin unrelated to packed cell volume) and the large differences were reported at face value. | medium | fixed |
| Diastolic blood pressure (multiples of 10 mm Hg) was read as a category. | medium | fixed |
| The rate card used anaemia as its outcome and compared it by hypertension, a comparison confounded by CKD. | medium | partly fixed |
| The headline measure was blood glucose rather than the readings that separate the groups most. | low | fixed |
| Implausible creatinine values (76 and 48.1 mg/dL) were not flagged. | low | fixed |
| No ranking of readings by how well they separate the classes, and no combined model. | low | not fixed |

## Still open

- There is no ranking of readings by how well they predict the diagnosis and no combined model; 'what predicts classification' is refused.
- The rate card still leads with anaemia by hypertension, which mostly follows kidney disease, though it now shows the smaller gap within the kidney-disease group.
- The headline card does not itself repeat the warning that the healthy group looks built from reference ranges; the warning sits in the summary, the data checks and the answers.
- 'percentage of ckd patients with diabetes' is refused because the column is named dm.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "kidney_disease.csv"`. The data is not stored in this repository.
