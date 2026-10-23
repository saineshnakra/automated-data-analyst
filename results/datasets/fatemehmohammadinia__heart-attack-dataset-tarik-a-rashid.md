# Heart Attack Dataset

[fatemehmohammadinia/heart-attack-dataset-tarik-a-rashid](https://www.kaggle.com/datasets/fatemehmohammadinia/heart-attack-dataset-tarik-a-rashid) on Kaggle · clinical · 1,319 rows × 9 columns · licence: Attribution 4.0 International (CC BY 4.0) · round 1

1,319 patients from one hospital in Erbil, Iraq, with age, sex (0/1), heart rate, blood pressure, blood sugar and two cardiac markers (CK-MB, troponin), each labelled positive or negative for a heart attack. The natural question is which measurements separate the two groups.

**First run:** partly useful. The numbers are right, but the headline is a non-difference picked by alphabetical order, and the real story (troponin, CK-MB and age separate positive from negative, plus impossible heart-rate entries) only shows up if you already know what to ask.

**Now:** good. The headline is the right finding (troponin separates positive from negative patients, AUC 0.89), with CK-MB and age as the next clear differences and the impossible heart-rate entries flagged; the CK-MB ceiling is misnamed as a missing-value code and dropped.

## What ADA finds now

ADA leads with the right result: troponin is much higher in heart-attack patients than in others (median 0.044 vs 0.006), with a test and an effect size, and it names CK-MB and age as the next clear differences while showing that blood sugar does not differ. It flags three heart-rate entries of 1,111 as likely typing errors and reads the 0/1 sex column as a category. It notices that troponin rises with age in rank order even though the straight-line correlation is weak. It treats the 19 CK-MB readings of exactly 300, all from heart-attack patients, as missing values when they look like an upper reporting limit, and it still lists systolic and diastolic blood pressure moving together, which is expected.

Headline: “Troponin is higher in positive than in negative: median 0.044 vs 0.006 (n = 810 and 509).”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| The measure came from an alphabetical tie-break (Blood sugar), the reading that least separates positive from negative patients. | high | fixed |
| Group differences reported as highest/lowest with no test or effect size. | high | fixed |
| Impossible and capped values in other columns missed (heart rate 1111, CK-MB stuck at 300). | medium | fixed |
| 0/1 Gender treated as a numeric measure and never offered as a grouping. | medium | fixed |
| Pairs ranked by Pearson only, so Age vs Troponin was missed while the expected systolic/diastolic pair was reported. | low | partly fixed |
| K abbreviation hid an impossible reading ('Maximum Heart rate is 1.1K'). | low | fixed |
| Count-by-value and two-group comparison questions were refused. | medium | fixed |

## Still open

- The 19 CK-MB readings of exactly 300 are called 'a code for a missing reading' and dropped from the analysis, but all 19 are positive patients and the next value is 297.5, so they look like a reporting ceiling; dropping them lowers the positive group's CK-MB median from 3.775 to 3.59.
- The differences card says 'Of the variance in Troponin, Result explains 5%' (eta squared on raw values, pandas 0.053), while the same comparison on ranks gives 0.44 and the headline says positive is larger in 89% of pairs; the 5% figure undersells the strongest marker in the data.
- A conditional share question is answered as a share of all rows: 'what share of patients with Gender 1 are positive' gives 563 of 1,319 (42.7%) instead of 563 of 870 (64.7%).
- The relationships card still reports systolic vs diastolic blood pressure (r = 0.59) as the strongest pair, which is expected and says nothing new.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "Medicaldataset.csv"`. The data is not stored in this repository.
