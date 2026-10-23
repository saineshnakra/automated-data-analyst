# Air Quality Index in Jakarta

[senadu34/air-quality-index-in-jakarta-2010-2021](https://www.kaggle.com/datasets/senadu34/air-quality-index-in-jakarta-2010-2021) on Kaggle · environmental · 5,538 rows × 11 columns · licence: Database: Open Database, Contents: Database Contents · round 2

One row per day for Jakarta from January 2010 to February 2025 (5,538 days): pollutant index values (ISPU) for PM2.5, PM10, SO2, CO, O3 and NO2, the day's maximum, the pollutant that set it ('critical'), its health band ('categori') and the reporting station. PM2.5 is recorded only from 2021.

**First run:** partly useful. The arithmetic, the Ask answers and the forecast caveat are right, but the headline and the largest differences are true by construction, and the step change and the time-driven pm25 gap are missed.

**Now:** good. It finds the 2021 step change in ozone and ties it to PM2.5 entering the index, keeps the columns built from the readings out of its comparisons, and every figure checked is right; what remains is a by-construction group table and one correlation explained by the wrong cause.

## What ADA finds now

This file holds daily air-quality index values for Jakarta from 2010 to early 2025. ADA's headline is the main event in the record: ozone index values fell from about 93 to 35 per quarter from 2021, when PM2.5 began to be recorded and became the day's main pollutant, and it flags that PM2.5 is missing for every day before 2021. It recognises that the daily maximum, the main-pollutant label and the health band are all built from the six readings and keeps them out of its comparisons, finds real differences between stations, and answers seasonal and yearly questions correctly. It still groups its summary table by the main-pollutant label, which only restates the readings, and it explains the negative link between ozone and sulphur dioxide by that label rather than by the 2021 change.

Headline: “o3 fell to a new level from Q1 2021: from 93.39 to 35.28 per quarter on average.”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| A category computed from the readings (critical = which sub-index is largest; categori = a band of max) was chosen as the grouping and scanned as an independent factor, so the headline and the largest difference were true by definition. | high | partly fixed |
| A column that is the row maximum of the others (max) was not recognised: the checks called it possible copying and the scan treated max vs pm25 as a lead. | high | fixed |
| Missing values that follow time were blamed on a group; there was no check for a column that starts at a date. | medium | fixed |
| A lasting step change in o3 was not detected; the trend compared only the first and last periods. | medium | fixed |
| 'month of the year' questions were answered as calendar months. | medium | fixed |
| The Indonesian 'no data' label TIDAK ADA DATA was kept as a group and its placeholder 0 entered means. | medium | partly fixed |
| The 'critical' label read as jargon and the like-for-like note about groups present in both quarters was hard to follow. | low | partly fixed |
| The headline led with a quarter-on-quarter change for a 15-year daily record. | low | fixed |

## Still open

- critical, which only names the highest of the six sub-indices, is still the brief's grouping, so its group table ranks pollutants by mean o3 (O3 first at 110.02), a comparison that holds by definition.
- The relationship card says the o3 and so2 correlation (r = -0.35) comes from 'the difference between the critical groups', a grouping built from those readings; the driver is the 2021 break (o3 fell from 93.8 to 34.6 while so2 rose from 15.7 to 40.1; within each era r is 0.01 and 0.07).
- Ask still averages the 3 placeholder zeros in max: 'average max per year' gives 63.33 for 2010 instead of 63.86.
- The o3 trend answer reports 'no clear trend' (-4.97 per year, 95% CI -12.2 to +3.35) without mentioning the 2021 step that the headline reports.
- 'when did pm25 start being recorded' is refused, although the data checks know the answer (1 January 2021).

## Reproduce

Download the dataset from Kaggle and run `ada analyze "ispu_dki_all.csv"`. The data is not stored in this repository.
