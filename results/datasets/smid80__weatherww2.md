# Weather Conditions in World War Two

[smid80/weatherww2](https://www.kaggle.com/datasets/smid80/weatherww2) on Kaggle · environmental · 119,040 rows × 31 columns · licence: U.S. Government Works · round 1

119,040 daily weather records from 159 weather stations, 1940-1945: maximum, minimum and mean temperature, precipitation, snowfall, wind and weather-type flags. Most quantities appear twice, in metric and imperial units, and the number of stations reporting grows from 6 in 1940 to 137 in 1945.

**First run:** poor. The arithmetic is right, but ADA analysed a mostly-zero duplicate snowfall column pooled over a changing set of stations, flagged winter as an anomaly, and lost precipitation to unparsed 'T' markers.

**Now:** good. The station comparison, the matched-station seasonal change and the data checks are right and useful; the remaining problems are a circular correlation card and a trend answer that pools a changing set of stations.

## What ADA finds now

ADA picks daily maximum temperature as the reading and the weather station as the grouping, and correctly reports that stations differ hugely (station explains 68% of the variation, from 31 °F to 99 °F on average). Its latest-change card compares only the 88 stations that reported in both quarters and correctly judges the autumn 1945 drop as normal for the season. It catches the dataset's traps: 'T' trace markers in precipitation, 0 °F stored as a missing-value code, duplicated metric and imperial columns, and days where the maximum is below the minimum. Its top correlation, however, is between the maximum and the daily mean, which is calculated from the maximum, and its multi-year trend answer still pools a set of stations that grows from 6 to 137.

Headline: “MAX differs across STA: STA 32603 has the highest mean, 99.11 (n = 276), and STA 16504 the lowest, 31.04 (n = 1,492); η² = 0.68.”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| A mostly-zero duplicate snowfall column (SNF) was picked as the measure. | high | fixed |
| The numeric station code STA was never a grouping and was offered for correlation. | high | fixed |
| Pooled period means were compared across periods while the set of reporting stations changed. | high | partly fixed |
| Trace markers ('T') left precipitation as text, and it dropped out silently. | high | fixed |
| Ask silently answered about SNF when asked about Precip. | high | fixed |
| Winter quarters were flagged as anomalies, with a band that allowed negative snowfall. | medium | fixed |
| The 0 °F (-17.78 °C) missing-value code was not detected, creating impossible max < min days. | medium | fixed |
| Outliers were computed pooled across tropical and arctic stations, and the top correlation was with a derived column. | medium | partly fixed |
| Positional 0/1 weather-flag codes were read as numbers, and spaced forms were dropped silently. | medium | partly fixed |
| Distinct counts and groupings with more than 50 groups were refused. | low | fixed |
| A useless forecast was shown, and the 9 dropped empty columns were not mentioned. | low | fixed |

## Still open

- The relationship card's top result, MAX vs MEA at r = 0.97, is true by construction: MEA is the average of MAX and MIN.
- The yearly trend answer pools stations although the number reporting grows from 6 to 137; only the latest-change card matches stations.
- The 'largest change by station' card (one station's maximum falling 33.8 °F from Q3 to Q4 1945) has no seasonal context; at that station the maximum runs from 64 °F in Q1 to 109 °F in Q3, so this is its usual autumn drop.
- The weather-flag columns are still read as numbers, and their runs of 0 (no poor weather) are flagged as a stopped instrument.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "Summary of Weather.csv"`. The data is not stored in this repository.
