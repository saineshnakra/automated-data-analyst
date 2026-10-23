# Soil Nutrient Dataset of Southern Indian States

[manojk2002/soil-nutrient-dataset-of-southern-indian-states](https://www.kaggle.com/datasets/manojk2002/soil-nutrient-dataset-of-southern-indian-states) on Kaggle · agriculture · 875 rows × 5 columns · licence: Apache 2.0 · round 2

875 soil samples from 101 districts (3 to 20 samples each), mostly in southern India with some in Gujarat, with nitrogen, phosphorus and potassium values and pH. The nutrient values look clipped to a range of about 20 to 80.

**First run:** partly useful. The numbers and the cap/floor checks are right, but a 101-value cut-off hides District, which explains 45-90% of the variance in every reading; with District set by hand the brief is good.

**Now:** good. District is the grouping, the brief leads with how much of each nutrient it explains, all figures match, and the floor, ceiling and piles in the readings are flagged; the remaining quibbles are the outlier answers and an effect size inflated by the ceiling.

## What ADA finds now

This file has nitrogen, phosphorus, potassium and pH values for 875 soil samples from 101 districts. ADA finds that the district explains most of the variation in every reading (69% for potassium, about 75% for nitrogen and phosphorus, 45% for pH) and names the highest and lowest districts; the figures match an independent check. It also spots that the nutrient values look clipped, with 182 nitrogen values at exactly 20 and many phosphorus and potassium values bunched just under 80, and that half the nitrogen and pH values sit in narrow piles. It does not connect the top district's near-identical values to that ceiling, and its answers about outliers still count values around those piles as unusual.

Headline: “Potassium value differs across District: Kalaburagi has the highest mean, 79.87 (n = 11), and Thiruvallur the lowest, 38.78 (n = 14); η² = 0.69.”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| District (101 values) was dropped by a hard 100-value limit, so the brief said no grouping existed and missed the main pattern. | high | fixed |
| Outliers were counted only for the measure; nitrogen's many values outside 1.5 x IQR were not mentioned. | medium | fixed |
| The headline was a bare overall mean of one of three equal nutrients. | low | fixed |
| Outlier questions were refused. | low | fixed |

## Still open

- The headline's top district, Kalaburagi (mean 79.87), sits against the apparent ceiling at 80 (its 11 values run 79.38 to 80.00, SD 0.17), which inflates the stated effect size (Hedges' g = 8.40); the card does not connect this to the ceiling the data checks report.
- Ask's outlier answers for nitrogen (105 values) and pH (190) count values around narrow piles without the warning the data checks give, and put pH, already a log measure, on a log scale.
- 'correlation between Nitrogen Value and pH' gives Pearson -0.06 and rank -0.20 without saying the floor at 20 is why they disagree.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "Soil data.csv"`. The data is not stored in this repository.
