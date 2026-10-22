# Air Quality from Multiple Districts of Bangladesh.

[mdshobuj/air-quality-dataset-of-bangladesh](https://www.kaggle.com/datasets/mdshobuj/air-quality-dataset-of-bangladesh) on Kaggle · sensor · 4,566 rows × 13 columns · licence: Attribution 4.0 International (CC BY 4.0) · round 2

4,566 rows of low-cost sensor readings (six MQ gas sensors in raw counts, PM1, PM2.5, PM10, temperature and humidity), a Gas_Index that is the average of the six gas sensors, and a four-level AQI_Level label graded from particulate readings. It is the processed file, whose four classes appear balanced to about 1,140 rows each.

**First run:** partly useful. The tests and Ask answers are accurate, but the brief leads with a composite and with circular PM-by-AQI differences, and it misses the 138-row mq135 fault that drives every mean.

**Now:** partly useful. The figures are right and the sensor faults are now caught, but the brief still leads with an average of raw sensor counts, its correlation card misreads opposite-sign group correlations driven by the faulty rows, and Ask still reports the by-definition PM differences.

## What ADA finds now

This file holds low-cost sensor readings from Bangladesh with a four-level air-quality label. ADA reports that the combined gas index is higher on 'Unhealthy' days than on 'Good' days (median 115 vs 82) and that humidity differs strongly between levels, and it correctly sets aside the particulate readings, from which the label is graded. Its data checks catch the main faults: about 140 gas-sensor readings that are hundreds of times too high, extreme values in two other sensors, and a -1 code for missing readings. It does not say plainly that the gas index is just the average of the six sensors, one of its correlation statements is driven by the faulty readings, and its answers to direct questions still present the particulate differences between levels as findings.

Headline: “Gas_Index differs across AQI_Level: Unhealthy has the highest median, 115.33 (n = 1,158), and Good the lowest, 82.38 (n = 1,140); ε² = 0.34.”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| A detached out-of-scale cluster (the mq135 sensor fault) was not reported, though it drives every mean. | high | fixed |
| PM readings were compared across AQI_Level, a label graded from PM, and reported as findings. | high | partly fixed |
| The composite Gas_Index (mean of the six MQ sensors) was not recognised, and its top 'relationship' was with its own part. | medium | partly fixed |
| The card used medians because the measure is skewed, but the key figure, group table and Ask used means without warning. | medium | partly fixed |
| A -1 no-reading code shared by every sensor was not recognised as missing. | low | fixed |
| The exceptional-records card flagged 12% of values, mixing the real fault with ordinary tail values. | low | fixed |
| Ordered levels (Good < Moderate < USG < Unhealthy) were treated as unordered, with no trend test. | low | not fixed |

## Still open

- The correlation card says Gas_Index and pm10 are 'stronger within each AQI_Level: r = -0.32', but the per-level correlations run from -0.32 to +0.35, with opposite signs, and without the 139 flagged mq135 rows the within-level figure is -0.14.
- Asked directly, ADA still reports PM differences across AQI_Level as findings ('larger in 100% of pairs'), though the brief sets them aside as holding by definition.
- The measure and headline are Gas_Index, an average of six raw gas-sensor counts, which the text never describes as such; the group table shows its means (Unhealthy 938) driven by the faulty rows.
- It does not notice that the four classes appear to have been balanced to about 1,140 rows each (the raw file has 121 Unhealthy rows), so its tests count resampled rows as separate records.
- The ordered air-quality levels are treated as unordered categories, with no test for a trend across them.
- A rank correlation of 0.43 is described as moving together 'only weakly'.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "air_quality_processed.csv"`. The data is not stored in this repository.
