# Crop Health and Environmental Stress Dataset

[datasetengineer/crop-health-and-environmental-stress-dataset](https://www.kaggle.com/datasets/datasetengineer/crop-health-and-environmental-stress-dataset) on Kaggle · agriculture · 212,019 rows × 32 columns · licence: CC BY-NC-SA 4.0 · round 1

212,019 rows by 32 columns of crop records: image-availability flags, vegetation indices, weather, soil and pest measures, expected yield, crop type and a 0/1 crop health label. The values look generated: no column relates to any other.

**First run:** poor. The arithmetic is correct, but ADA presents identical-looking groups as a ranking and misses the obvious signs that this is generated data (impossible values, zero correlation everywhere).

**Now:** good. The headline correctly finds no yield difference by health label, the report says no column relates to any other and that several look generated, and four kinds of impossible values are flagged; the outlier card on skewed columns is the one misleading item.

## What ADA finds now

ADA correctly reports that expected yield does not differ between healthy and unhealthy crops, and that none of 125 comparisons across the label, crop type and growth stage shows a real difference. It points out that no column relates to any other, which is unusual for measured data, names seven columns that look generated, and flags impossible values such as negative yields, canopy cover above 100% and vegetation indices above 1. Its outlier card treats the ordinary tails of skewed columns such as rainfall as thousands of exceptional records. The full report takes about 12 seconds on this 212,000-row table.

Headline: “Expected_Yield shows no clear difference between Crop_Health_Label 0 and Crop_Health_Label 1: mean 3,004.5 vs 3,003.0 (n = 63,821 and 148,198).”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| Named a highest and lowest group for a negligible gap and rounded both values to the same '3.0K'. | high | fixed |
| No plausibility check: impossible values and the table-wide absence of any correlation went unmentioned. | high | fixed |
| The relationship finding silently disappeared when no pair passed the threshold. | medium | fixed |
| The 0/1 health label and other integer codes were treated as measures; the label was never a grouping. | medium | partly fixed |
| GPS_Coordinates (a single 6-digit integer) used as a numeric measure. | low | partly fixed |
| Filtered count and share-of-rows questions refused. | medium | fixed |
| Slowest dataset of the set. | low | not fixed |

## Still open

- The 'Exceptional records' card reports 5,978 Rainfall values (and similar counts for Canopy_Coverage and Organic_Matter) as beyond chance, but these columns are exponential-shaped (Rainfall mean 20.0, SD 19.95), and nearly all the flagged values are near-zero readings in the low tail after a log transform (pandas: 5,810 of 5,811 log-scale outliers are low), not exceptional records; a recommendation then asks the reader to review them.
- The whole report takes about 12.6 seconds, over the 10-second budget.
- GPS_Coordinates and Field_Boundaries are still treated as measured quantities.
- 'is NDVI higher in healthy crops' is refused, although 'what share of crops are healthy' correctly maps 'healthy' to the health label; NDVI group means still print as '0.5'.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "agriculture_dataset.csv"`. The data is not stored in this repository.
