# Heart Attack

[pritsheta/heart-attack](https://www.kaggle.com/datasets/pritsheta/heart-attack) on Kaggle · clinical · 303 rows × 14 columns · licence: Database: Open Database, Contents: © Original Authors · round 3

The Cleveland heart-disease table: 303 patients with age, sex, chest-pain type, blood pressure, cholesterol, exercise-test results, vessel count, thal and a 0/1 label 'target'. In this copy target 1 is commoner in women and in people without exercise angina, which matches 'no disease' in the original coding.

**First run:** partly useful. Every number checks out and the target-rate card is the right idea, but the headline is cholesterol by sex, and the strongest links to the outcome (chest-pain type, thal, vessel count) were never looked at because of a six-grouping cap.

**Now:** good. The report now centres on the outcome label, scans every factor and ranks thal, chest-pain type and vessel count correctly; it lacks a combined model for the label and misses one missing-value code.

## What ADA finds now

This is the classic Cleveland heart-disease table of 303 patients with a yes/no label. The tool now builds its report around that label: maximum heart rate is higher in the label-1 group (158 against 139), and the gap holds within each chest-pain type. Its rate card correctly finds thal, chest-pain type and vessel count to be the factors most tied to the label, and plain questions about rates by group now work. It flags the thal value 0 as a likely missing code but not the vessel count 4, which it then quotes as the highest rate. It offers no combined model for the label, so it cannot say which factors still matter once the others are taken into account.

Headline: “thalach is higher in target 1 than in target 0: mean 158.47 vs 139.10 (n = 165 and 138).”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| The difference and rate scans used only the first 6 groupings, so chest-pain type, thal and vessel count were never looked at and a weaker factor was called the clearest. | high | fixed |
| The headline was a side comparison (cholesterol by sex) rather than the 0/1 outcome. | medium | fixed |
| No combined model for the 0/1 outcome. | medium | not fixed |
| Natural ways of asking for a rate were refused. | medium | fixed |
| Rare codes that stand for missing values (ca = 4, thal = 0) were kept as real levels. | low | partly fixed |
| The outlier card checked age within vessel-count groups, which is hard to interpret. | low | partly fixed |

## Still open

- There is no combined model for the yes/no outcome, so a reader cannot tell which factors matter once the others are allowed for; 'what predicts target' looks only at the five numeric columns.
- ca = 4, a missing-value code in this copy, is not flagged and is quoted as the highest target rate (80%, 4 of 5).
- The joint model of heart rate uses the outcome label as a predictor and sets ST depression, blood pressure and cholesterol aside as 'candidate outcomes'.
- The scan's top finding (ST depression by ST slope) links two descriptions of the same ECG segment.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "Heart Attack Data Set.csv"`. The data is not stored in this repository.
