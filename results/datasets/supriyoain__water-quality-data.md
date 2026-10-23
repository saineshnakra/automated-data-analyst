# Water Quality Data 🌊

[supriyoain/water-quality-data](https://www.kaggle.com/datasets/supriyoain/water-quality-data) on Kaggle · environmental · 2,371 rows × 8 columns · licence: Apache 2.0 · round 3

Volunteer water-quality readings at a wildlife refuge from May 1989 to Nov 2019 (2,371 rows): salinity, dissolved oxygen, pH, Secchi depth, water depth, water and air temperature. Until 1999 there is one row per date; from mid-1999 usually 4-5 rows per date (several sites), with no site column.

**First run:** partly useful. It picks the right reading, finds the real 1999 step and the expected oxygen-temperature link, but misses a 0 °F missing-value code and several impossible readings, and doesn't connect the step to the change in sampling that explains it.

**Now:** good. It finds the 1999 drop and ties it to the change in sampling, tests the long-run trend properly, shows the seasonal cycle and catches the missing-value code and impossible readings; the salinity link is the one misleading leftover.

## What ADA finds now

This is 30 years of volunteer water-quality readings at a wildlife refuge. ADA finds that dissolved oxygen dropped from about 9.5 to 5.9 mg/L in mid-1999 and points out that this is exactly when the sampling switched from one reading per date to several sites per date, so the drop may reflect where samples were taken. Around that step it finds no clear long-term trend, shows the seasonal cycle (highest in winter, lowest in late summer), and links oxygen to water temperature as expected. It catches the air temperatures of -17.78 °C that are blank Fahrenheit readings and several impossible values, and warns when an answer involves them. It still lists salinity as related to oxygen, a link that comes from both changing at the same 1999 sampling change.

Headline: “DissolvedOxygen (mg/L) fell to a new level from Q3 1999: from 9.54 to 5.94 per quarter on average.”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| A unit-converted zero (0 °F = -17.78 °C) used as a missing-value code was not recognised. | high | fixed |
| Impossible values (water 43-74 °C, pH 0.3 and 0.7, Secchi depth greater than water depth) were not flagged and were repeated in answers. | medium | fixed |
| The 1999 level shift was attributed to method or conditions without noting that sampling changed at the same time. | medium | fixed |
| Trend answers fitted one slope across the detected step. | medium | fixed |
| Model coefficients confounded by the period change (salinity with a positive sign). | medium | partly fixed |
| 'by month' gave a 365-month series; before/after and season splits were refused. | medium | fixed |
| The brief led with the change in the latest quarter. | low | fixed |
| Rows with no date were counted without comment. | low | fixed |

## Still open

- Salinity is still presented as relating to oxygen (r = 0.30), a link produced by both changing at the 1999 sampling change.
- The headline sentence states the 1999 drop in oxygen without the sampling caveat, which appears only in the card below it.
- The brief's seasonal figures (January 8.71, September 5.47, averages of monthly means) differ from the asked month-of-year answer (8.35 and 4.86, averages of rows) without explanation.
- In the joint model, air temperature takes a positive coefficient only because it travels with water temperature (variance inflation noted), and water readings of 42.5 and 43 °C pass the checks.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "waterquality.csv"`. The data is not stored in this repository.
