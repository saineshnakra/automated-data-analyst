# Age Prediction Using Genomic Information

[thedevastator/age-prediction-for-individuals-using-multi-omic](https://www.kaggle.com/datasets/thedevastator/age-prediction-for-individuals-using-multi-omic) on Kaggle · biology · 104 rows × 13 columns · licence: CC0: Public Domain · round 2

DNA methylation levels, in percent, at 13 age-linked CpG sites (genes such as ELOVL2, TRIM59 and KLF14) for 104 people. Only the test split is included, and the ages the sites are meant to predict are not in the file.

**First run:** partly useful. Every number is right and the caveats are honest, but with no outcome column the brief centres on an arbitrary site and misses that all 13 sites form one correlated block.

**Now:** partly useful. Every number is right and Ask now handles column comparisons and distributions, but the brief still centres on one arbitrarily chosen site and never says that the 13 sites form one correlated block.

## What ADA finds now

This file holds DNA methylation levels at 13 age-linked sites for 104 people, without their ages. ADA reports that the site it focuses on, DDO_1, tracks LDB2_3 most closely (r = 0.52), that TRIM59_5 and ELOVL2_6 are the most tightly linked pair (r = 0.87), and fits a model of DDO_1 on the other sites; every figure matches an independent check. It also answers which column has the highest average, how a column is spread and how two sites relate. Its main shortfall is that it builds the summary around one site picked almost at random and never says that all 13 sites rise and fall together as one block, which is the main feature of this table.

Headline: “DDO_1 and LDB2_3 move together: Pearson r = 0.52 (95% CI 0.36 to 0.65, n = 104).”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| With no outcome column, the brief centres on one site picked on a tie-break (DDO_1) and misses that all 13 sites move together as one correlated block. | medium | not fixed |
| 'which column has the highest average' was answered as the mean of the current measure. | medium | fixed |
| An outlier card was raised for 1 value outside 1.5 x IQR, no more than chance gives, with plural wording. | low | fixed |
| Ranked correlates are ordered by a robust score the reader does not see, so the printed r values look out of order. | low | partly fixed |
| Questions about outliers or a distribution were refused. | low | partly fixed |

## Still open

- The brief is built around one site chosen on a tie-break (DDO_1) and never says that all 13 sites rise and fall together as one block (the first principal component holds 55.5% of the variance).
- The headline is the best of 12 correlations against that arbitrary site, shown without the 'picked from many' caveat that its card carries; the strongest pair in the table is TRIM59_5 and ELOVL2_6 (r = 0.87).
- The joint model of DDO_1 leaves out ELOVL2_6, its second-strongest correlate (r = -0.48), as 'another candidate outcome', and its adjusted R² (0.26) is no better than LDB2_3 alone (0.27); the card does say the other sites add little.
- Outlier answers switch to log-scale fences for a bounded percentage, so 'no KLF14_2 value' is called unusual while the ordinary rule finds 3 (8.73, 8.75, 9.23).
- The correlate list in the brief is ordered by a score it does not show, so TRIM59_5 (r = -0.42) is listed before ELOVL2_6 (r = -0.48).
- 'which columns are most correlated' is refused.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "test_rows.csv"`. The data is not stored in this repository.
