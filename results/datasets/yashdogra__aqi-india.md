# AQI India

[yashdogra/aqi-india](https://www.kaggle.com/datasets/yashdogra/aqi-india) on Kaggle · environmental · 3,327 rows × 11 columns · licence: GNU Affero General Public License 3.0 · round 3

A single snapshot (all rows stamped 14 Feb 2025 17:00) of about 500 Indian air-quality monitoring stations, one row per station and pollutant (PM2.5, PM10, NO2, CO, ozone, SO2, NH3), giving the minimum, maximum and average reading. The values behave like air-quality sub-index values (PM readings stop at 500).

**First run:** poor. The arithmetic is right, but the main figures average seven different pollutants together and the relationships found are min/avg/max of the same series.

**Now:** partly useful. Every number checks out and pollutants are no longer mixed, but the brief is about carbon monoxide alone (without naming it in the headline) and misses the station-level air-quality summaries this snapshot is for.

## What ADA finds now

This file is one snapshot of about 500 Indian air-quality stations, with one row per station and pollutant. ADA now recognises that each row is a different pollutant and never mixes them: its brief covers carbon monoxide, the pollutant with the most readings, and finds Delhi highest (49.45) and Andhra Pradesh lowest (22.27) among states, figures that check out. Asked about PM2.5 directly it answers correctly (West Bengal highest by state at 138.73, Gurugram the single highest reading at 333). It does not lead with PM2.5 or give each station's overall air-quality index or main pollutant, and on a default run its data checks see only the carbon monoxide rows.

Headline: “pollutant_avg differs across state: Delhi has the highest mean, 49.45 (n = 38), and Andhra_Pradesh the lowest, 22.27 (n = 11); η² = 0.12.”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| Long table treated as flat, so readings of seven different pollutants were averaged, ranked and correlated together. | high | fixed |
| Min, average and max of the same reading correlated with each other and fed into a joint model (true by construction). | high | partly fixed |
| Headline ranked states by mean of station means, while segments and answers ranked by row means, naming two different 'highest' states. | medium | fixed |
| No per-station air-quality index (highest sub-index), dominant pollutant or per-pollutant ranking of stations. | medium | not fixed |
| pollutant_max picked as the measure instead of pollutant_avg. | low | fixed |
| No check for an average outside its own min-max range. | low | partly fixed |
| A one-value timestamp reported as 'No date column was found'. | low | partly fixed |
| The 500 ceiling on PM readings is not mentioned. | low | not fixed |

## Still open

- The brief leads with carbon monoxide, picked only because it has the most recorded rows (474 vs 456 for PM2.5); for an air-quality table PM2.5 or the overall index is the obvious lead, and the headline sentence itself does not say it is about CO.
- No station-level air-quality index (highest sub-index) or dominant pollutant, the first summaries a researcher would want from this snapshot.
- The generated question still correlates pollutant_avg with pollutant_min (r = 0.76) without saying they are two summaries of the same series.
- Data checks run only on the CO rows, so the one row whose average lies outside its min-max (PM2.5, Keonjhar) and the 500 ceiling are not reported on the default run.
- The anomalies and forecast sections still say 'no date column' while the summary correctly calls the file a snapshot.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "Data Export.csv"`. The data is not stored in this repository.
