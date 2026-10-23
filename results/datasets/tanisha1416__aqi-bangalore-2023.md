# Smog City Diaries: Unveiling Air Quality Trends📈

[tanisha1416/aqi-bangalore-2023](https://www.kaggle.com/datasets/tanisha1416/aqi-bangalore-2023) on Kaggle · environmental · 147 rows × 12 columns · licence: CC0: Public Domain · round 2

Daily readings for one Bangalore monitoring station from 1 January to 31 May 2023 (147 days): NO2, SO2, CO, PM10, temperature, humidity, wind speed and direction, solar radiation and an air quality index. The last 12 rows, added in an update, have wind speed and direction swapped, and the columns behave like independent random numbers.

**First run:** partly useful. Dates, roles and arithmetic are right and the correlation caution is good, but ADA missed the swapped wind columns and the impossible temperature, and its headline is a change within the noise.

**Now:** good. The headline correctly says nothing changed, the data checks find the swapped wind columns, the impossible temperature and the signs of simulated data, and every answer checked matches pandas; the only quibble is a joint-model card that explains nothing.

## What ADA finds now

This file holds five months of daily air-quality readings for one Bangalore station. The analysis correctly reports that the air quality index did not change clearly in the last month, and its data checks find the three things a careful analyst would: wind speed and direction swapped on the last 12 days, a temperature of 80.6 °C, and columns that look like independent random numbers, which suggests the table is simulated. It keeps the swapped values out of its correlations and warns when an answer touches a flagged value. Its one weak point is a model of the index on the pollutants that explains almost nothing, which is the expected result for data like this.

Headline: “No clear change in AQI in May 2023 (+2.27, within the usual range).”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| The data checks missed 12 rows with wind speed and direction swapped (wind speeds of 136-280 m/s), which drove the strongest correlations. | high | fixed |
| An impossible temperature (80.6 C) was not flagged. | medium | fixed |
| The headline reported a +2.27 monthly change that is within the noise, without saying so. | medium | fixed |
| The anomaly check was refused (5 monthly periods) and no other outlier check ran. | medium | fixed |
| No warning that the table looks simulated (an index unrelated to its own pollutants, flat distributions). | medium | fixed |
| Ask returned impossible values without caution. | low | fixed |

## Still open

- The brief still fits a joint model of AQI on five pollutants and temperature that explains 4% of its variance (adjusted R^2 = 0.01); it says they 'add little', but for a table it has just called possibly simulated, this card adds length without information.
- That model sets humidity and solar radiation aside as 'another candidate outcome' rather than as possible predictors.
- 'average PM10 by month' reports May at 72.12, a figure pulled down by the 12 appended rows (PM10 24.8-67.4), without noting that those rows came from the flagged block.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "AQI_Bangalore_2023_Update_01.csv"`. The data is not stored in this repository.
