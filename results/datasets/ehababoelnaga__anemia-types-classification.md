# Anemia Types Classification

[ehababoelnaga/anemia-types-classification](https://www.kaggle.com/datasets/ehababoelnaga/anemia-types-classification) on Kaggle · clinical · 1,281 rows × 15 columns · licence: Apache 2.0 · round 1

1,281 complete blood count records (WBC, RBC, haemoglobin, haematocrit, MCV, MCH, MCHC, platelets and others), each labelled with one of 9 diagnoses such as iron deficiency anaemia, leukaemia or healthy. Most rows have gaps filled with column means, and a few hold impossible values.

**First run:** poor. The arithmetic is right, but the headline ranking and the strongest correlation both come from one or two impossible records, true strong correlations are reported as absent, and 61% mean-imputed rows and negative haemoglobin go unflagged.

**Now:** good. The report leads with haemoglobin as the clearest separator of diagnoses, ranks the other indices with tests, flags the mean-filled gaps in 61% of rows, the impossible values and the duplicates, and recovers the true red-cell correlations; asking for average haematocrit by diagnosis still gives an outlier-driven ranking.

## What ADA finds now

ADA finds that haemoglobin separates the diagnoses most clearly, lowest in iron deficiency anaemia and highest in the leukaemia and thrombocytopenia groups, and that MCH, MCV and MCHC follow, each with a rank test. It warns that 61% of rows hold the same non-round value in six columns, which means missing results were filled in with averages, and it sets those cells aside. It flags impossible readings such as negative haemoglobin and a neutrophil percentage of 5,317, notes 49 duplicated rows, and shows that the strong links between haemoglobin and haematocrit and between MCV and MCH were hidden by a handful of bad records. Asked for average haematocrit by diagnosis, it still ranks a group first on the strength of two impossible values.

Headline: “HGB differs across Diagnosis: Leukemia with thrombocytopenia has the highest median, 14.60 (n = 11), and Iron deficiency anemia the lowest, 9.80 (n = 189); ε² = 0.76.”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| Group means and Pearson correlations were driven by one or two impossible records (HCT 3715 decided both the headline and the strongest relationship). | high | partly fixed |
| A Spearman much larger than Pearson was explained as 'not a straight line' when a few gross outliers hid a strong linear relationship. | high | fixed |
| No check for physically impossible values or sentinels. | high | fixed |
| Mean imputation not detected (61% of rows share column-mean constants in 6 columns). | medium | fixed |
| Only one arbitrary measure reported; which measures separate the diagnoses not shown. | medium | fixed |
| Fully duplicated rows not reported. | low | fixed |
| Count-per-group and outlier questions refused. | medium | fixed |

## Still open

- Asking for average haematocrit by diagnosis still ranks Normocytic hypochromic anaemia first at 53.90, a mean carried by two impossible values (316 and 3,715); without them its mean is 39.7, among the lowest, and the answer only adds a generic caution.
- The relationships card lists haemoglobin vs MCH as r = 0.00 (rank 0.59) because the 7 extreme MCH values are flagged but not set aside; without them r = 0.57.
- The strongest other pair, lymphocyte % vs neutrophil % (r = -0.86), is largely built in, since both are shares of the same white-cell count.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "diagnosed_cbc_data_v4.csv"`. The data is not stored in this repository.
