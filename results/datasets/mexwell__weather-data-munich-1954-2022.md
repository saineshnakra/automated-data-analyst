# 🌤️ Weather Data Munich 1954-2022

[mexwell/weather-data-munich-1954-2022](https://www.kaggle.com/datasets/mexwell/weather-data-munich-1954-2022) on Kaggle · environmental · 25,051 rows × 19 columns · licence: CC BY-SA 3.0 · round 2

Daily observations from the German Weather Service station in Munich, 1 June 1954 to 31 December 2022 (25,051 days): mean, maximum and minimum temperature, rain and its form, sunshine, snow depth, cloud, humidity, pressure and wind. The date is stored as a YYYYMMDD number and -999 marks a missing reading.

**First run:** poor. ADA read the file and spotted the -999 code, but it took the date as the measure, could not read the date even when told, and left -999 in every other column, so nearly every figure in the brief is an artefact.

**Now:** good. The headline warming trend (+0.47 °C per decade in daily maximum temperature, Theil-Sen on complete years) and every figure checked match SciPy and pandas, and the cold winters of 1963 and 1987 are found; what remains is a meaningless trend test across rain-type codes and a correlation card that restates max, mean and min temperature.

## What ADA finds now

This file holds daily weather readings for Munich from 1954 to 2022. The analysis now reads the date, treats -999 as a missing value everywhere, and sets the quality-flag columns aside, so its figures are real readings. It leads with a clear warming of the daily maximum temperature, about 0.47 °C per decade with an interval, picks out the cold winters of 1963 and 1987, and forecasts with the seasons. When asked, it gives the 0.33 °C per decade rise in mean temperature and the warmest year. Its weak spots are minor: it tests a trend across rain-type codes that have no order, and its list of related readings is led by other versions of temperature.

Headline: “The yearly average of TXK rose by 0.468 per decade from 1955 to 2022 (95% CI +0.352 to +0.554; Mann-Kendall p < 0.001).”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| A date stored as a YYYYMMDD whole number was not recognised, even when named with date=. | high | fixed |
| The -999 missing-value code was masked only in the measure and left in every other column, corrupting correlations, comparisons and answers. | high | fixed |
| The date key (unique, rising whole numbers) was picked as the measure. | high | fixed |
| Quality-level flags were used as groupings, so most 'clear differences' tracked the decade. | medium | fixed |
| Trend answers compared the first and last (partial) periods; 'which year was the warmest' was refused. | medium | fixed |
| Constant columns were dropped without a note. | low | fixed |

## Still open

- The group card still runs a trend test across the precipitation-form code as if its codes were ordered: 'TXK falls as RSKF increases across its 6 levels (Spearman rho = -0.19)'. RSKF 0, 1, 4, 6, 7, 8 mean none, rain (before 1979), unknown, rain, snow, rain and snow, so the order means nothing; ADA itself reads RSKF as categories.
- 'What relates to TXK' leads with TMK (r = 0.98), TNK (0.91) and TGK (0.87): daily mean, minimum and ground temperature are versions of the same reading, so the card mostly restates that.
- The 'Largest change by RSKF' card says rain days (RSKF 6) moved most from Q3 to Q4 2022 (23.23 to 12.47); that is the season, and comparing precipitation types as if they were reporting units adds little.
- Refusals still call the date column text: 'MESS_DATUM holds text, not numbers, so it can't be averaged'.
- 'which rows have the highest TXK' gives only the value (37.50), not the day; asking 'which day' does give 27 Jul 1983.
- 'which month has the lowest TMK' ranks single calendar months (Feb 1956, -10.32) rather than months of the year (January, 0.17 on average); the question is ambiguous, but the month-of-year reading is the likelier one.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "weatherdata.csv"`. The data is not stored in this repository.
