# Prediction of Mohs Hardness with Machine Learning

[jocelyndumlao/prediction-of-mohs-hardness-with-machine-learning](https://www.kaggle.com/datasets/jocelyndumlao/prediction-of-mohs-hardness-with-machine-learning) on Kaggle · materials · 622 rows × 13 columns · licence: CC0: Public Domain · round 3

622 minerals with their Mohs hardness (the outcome) and 11 descriptors computed from the chemical formula: totals over the formula (electron count, density) and per-atom averages (valence electrons, atomic weight, ionisation energy, electronegativity, atomic radii, density). Ten rows have every descriptor equal to 0, and 23 rows are exact duplicates.

**First run:** poor. The arithmetic is right, but it analyses a computed descriptor instead of hardness, and the brief describes how the descriptors relate to each other; with measure='Hardness' it gives a sound, honest answer (weak single correlations, R² = 0.28 jointly).

**Now:** partly useful. It now studies hardness, sets aside the all-zero rows and reports an honest R2 of 0.41, but the headline names 'clear' predictors that depend on two near-identical columns, and opposite-sign correlations are still given a direction.

## What ADA finds now

This table gives the Mohs hardness of 622 minerals with 11 descriptors computed from their chemical formulas. ADA now treats hardness as the outcome, spots the 10 rows where every descriptor is zero and leaves them out, and reports that no single descriptor relates strongly to hardness (the strongest correlation is -0.32) while all of them together explain about 41% of its variance, figures that check out. Its headline names six descriptors as clear predictors, but two of them are near-copies of each other, so that list changes when either is removed. Where the ordinary and rank correlations point in opposite directions, it still describes a single direction.

Headline: “Together, 10 readings explain 41% of the variance in Hardness (n = 612), against 10% for zaratio_Average alone. With the others held fixed, zaratio_Average, ionenergy_Average, atomicweight_Average, allelectrons_Average, allelectrons_Total and density_Average stay clear.”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| A keyword in a descriptor's name ('Total') beat the real outcome, Hardness. | high | fixed |
| Rows where every descriptor is exactly 0 were reported as extreme values, not as missing data coded as zero. | medium | fixed |
| When Pearson and Spearman have opposite signs, a direction is stated anyway. | medium | not fixed |
| The headline was a by-construction correlation between two '_Total' columns. | medium | fixed |
| Rows identical in every input but different in hardness are not reported. | low | not fixed |
| Heavy tails around a narrow middle were called 'exceptional records'. | low | fixed |

## Still open

- The headline's list of predictors that 'stay clear' rests on collinearity: atomicweight_Average and allelectrons_Average are near-copies (r = 0.998, coefficients +0.415 and -1.26), and dropping one lowers R2 from 0.41 to 0.37 and makes density_Average non-significant (p < 0.001 to p = 0.70). The model card's warning about unstable coefficients does not reach the headline.
- Where Pearson and Spearman disagree in sign (density_Average, allelectrons_Average, atomicweight_Average), answers still state a direction.
- Minerals with identical descriptors but different hardness, which cap how well any model can do, are not reported.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "jm79zfps6b-1/Mineral_Dataset_Supplementary_Info.csv"`. The data is not stored in this repository.
