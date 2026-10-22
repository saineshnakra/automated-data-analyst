# Beach Weather Stations 🏖️🌊👓

[sanjanchaudhari/beach-weather-stations](https://www.kaggle.com/datasets/sanjanchaudhari/beach-weather-stations) on Kaggle · environmental · 156,051 rows × 18 columns · licence: ODC Public Domain Dedication and Licence (PDDL) · round 1

Hourly readings from three Chicago Park District beach weather stations (63rd Street, Oak Street, Foster), April 2015 to June 2023, 156,051 rows: air and wet-bulb temperature, humidity, rain per reading, a cumulative rain counter, wind, pressure, solar radiation and battery. One station stops in August 2022 and Foster has no rain gauge.

**First run:** poor. It summed a cumulative rain counter, so the headline, station shares, anomalies, correlation and forecast are all meaningless, and it reported a station with no rain gauge as 0 rain and a station dropout as a change.

**Now:** good. It sets aside the cumulative rain counter, analyses rain per reading with seasonal comparisons, removes the sensor codes and flags a stuck wind sensor, the missing gauges and the gaps in the log; the headline itself is weak and yearly rain totals cannot be asked for.

## What ADA finds now

ADA recognises that the 'Total Rain' column is a running counter that cannot be added up, and works with the rain recorded at each reading instead, comparing the latest quarter with the same season in earlier years. Its data checks are thorough: it removes wind and solar codes such as 999.9 and -100,000, finds a wind sensor at Foster stuck on one value for about two and a half years, notes that Foster has no rain gauge, and reports a gap of 176 days in the log. The headline itself says little, a change in the latest partial quarter and no real difference between stations, and its largest reported difference, more rain when the sensor reports rain, is close to true by definition. It also refuses to give yearly rainfall totals.

Headline: “-0.0853 latest movement; no clear difference in Interval Rain between station name groups.”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| A cumulative rain counter (Total Rain) was picked as the measure and summed. | high | fixed |
| A station with no rain gauge was shown as 0 rain, and a station dropping out was read as a change. | high | fixed |
| Sentinel and impossible sensor values not detected. | high | fixed |
| Latest movement was mostly seasonality; anomaly bands and forecast ignored the annual cycle. | medium | fixed |
| 'total X' returned the mean; 'maximum Wind Speed' was read as the Maximum Wind Speed column. | medium | fixed |
| Market-style concentration metrics and share wording on physical measurements. | low | fixed |
| Missingness reported only as a table-wide percentage. | low | fixed |

## Still open

- The headline is weak: it opens with the change in the latest, partial quarter and a station comparison that finds nothing, while the most useful content (a wind sensor stuck for two and a half years at Foster, gauges missing at Foster, one station stopping in 2022) sits in the data checks.
- The 'largest difference found' is rain by precipitation type (type 60 mean 1.99 vs type 0 0.03, epsilon squared 0.50), which is largely true by construction, since the precipitation-type sensor reports rain when rain is falling.
- Annual rainfall totals cannot be asked for: 'total Interval Rain by year' is refused because 'Periods of Interval Rain are averaged, as on the dashboard'.
- The station comparison gives p = 0.0059 in the report and p = 0.42 in the answer to the same question, because the answer tests a 50,000-row sample; and station rainfall totals are shown as shares of a total although the stations reported for different spans.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "Beach_Weather_Stations_-_Automated_Sensors.csv"`. The data is not stored in this repository.
