# IAQ Baquba Hospital dataset – University of Diyala

[haiderqassim/iaq-baquba-hospital-dataset-university-of-diyala](https://www.kaggle.com/datasets/haiderqassim/iaq-baquba-hospital-dataset-university-of-diyala) on Kaggle · environmental · 523,524 rows × 12 columns · licence: Attribution 4.0 International (CC BY 4.0) · round 1

Minute-by-minute indoor air-quality sensor logs (CO2, TVOC, PM10, PM2.5, eCO2, CO, an air-quality index, light, O3, temperature, humidity) from a teaching hospital in Iraq, May 2024 to January 2025, 523,524 rows. The file is two sensor logs stacked one after the other with no device column.

**First run:** partly useful. The numbers are right for the window it used and the wording is careful, but it missed that the file holds two stacked sensor logs, analysed only the newest 45% of rows for no speed gain, and printed a self-contradictory '7 of 7 days' caveat.

**Now:** good. It separates the two stacked logs, analyses the whole file, leads with the jump in particle readings from July 2024 (PM10 about 5 in May, 103 in June, about 800 from July) and flags the eCO2 floor, the CO2 code and a stuck humidity sensor; the new regression card is over-confident and the run takes 18 seconds.

## What ADA finds now

ADA spots that this file is two sensor logs stacked end to end and keeps them apart, then leads with the most important feature of the data: particle readings jump about fifteen-fold between May-June and July 2024, which usually means a change of instrument or setup. It flags an eCO2 sensor resting at its 400 floor in almost half its readings, a CO2 value of 964 that appears in 13% of rows and looks like a code, a humidity sensor stuck for five days, and a log that never records eCO2. It shows that particle readings track each other much more closely within each log than when the two are pooled, and it can profile readings by hour of day. A new regression card reports intervals far too narrow for minute-by-minute readings that follow each other closely, and the whole report takes about 18 seconds.

Headline: “PM10 rose to a new level from Jul 2024: from 54.47 to 852.66 per month on average.”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| Two stacked sensor logs (timestamps restart mid-file, 227k repeated timestamps) were averaged together. | high | fixed |
| The row limit dropped the oldest 55% of rows with no speed gain. | medium | fixed |
| Off-by-one partial-period caveat ('7 of 7 days') next to 'latest complete period'. | medium | fixed |
| No check for level shifts, sensor floors or ceilings. | medium | fixed |
| No hour-of-day view for minute-level data. | low | fixed |
| Measure picked alphabetically among tied sensor columns. | low | fixed |

## Still open

- A new 'Joint model of PM10' card gives regression coefficients with very tight intervals (light +0.546 per unit, CI 0.529 to 0.562) on minute readings whose residuals are almost perfectly autocorrelated (Durbin-Watson about 0.03), so the intervals and p-values are far too confident; it is also fitted on Block 1 only, because Block 2 never records eCO2, which the card does not say.
- The full report takes 18 seconds, over the 10-second budget.
- Asking for average CO2 by month reports a significant fall from 862 in June to 475 in December, but 74% of June readings are the flagged 964 code; without them the months run from 576 to 434. The answer adds a caution but still reports the trend.
- Group sizes from a 50,000-row sample are printed as if they were the groups' sizes ('Air Quality ... n = 26,328 and 23,672'; the blocks have 276,872 and 246,652 rows).
- 'when did PM10 jump' is refused, though the report itself names July 2024.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "IAQ Baqubah Teaching Hospital .csv"`. The data is not stored in this repository.
