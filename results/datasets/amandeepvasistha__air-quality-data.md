# Air Quality Data - India

[amandeepvasistha/air-quality-data](https://www.kaggle.com/datasets/amandeepvasistha/air-quality-data) on Kaggle · environmental · 737,406 rows × 16 columns · licence: CC0: Public Domain · round 1

Hourly pollutant concentrations (PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, benzene, toluene, xylene) and AQI for 26 Indian cities from January 2015 to July 2020, from the national monitoring network. The file is two tables stacked: 707,875 hourly rows followed by 29,531 daily-average rows, and the number of cities reporting grows from 7 in 2015 to 26 in 2020.

**First run:** partly useful. Dates, city grouping and the arithmetic are right, but it read NO as text, headlined a city whose CO is a copy-paste error, analysed only the newest third of the file, and missed the lockdown drop.

**Now:** partly useful. The reading and data checks are now strong and the headline is correct, but the brief leads with a weak ozone result, misses the lockdown, and some answers mix changing city coverage or repeat Ahmedabad's faulty values without a caveat.

## What ADA finds now

ADA now reads this large file correctly: it separates the hourly readings from the daily summaries appended at the end, uses all 700,000 hourly rows, flags long runs of exact zeros, and notices that Ahmedabad's carbon monoxide column copies its nitric oxide column. Its headline, that ozone differs between cities (highest in Bhopal, lowest in Kochi), is correct and notes that those two cities were only monitored for a short period. But ozone explains little of the picture, and the summary never mentions the sharp fall in particulates and nitrogen dioxide during the 2020 lockdown. Some answers still mix years with very different numbers of cities, and the city ranking for the air-quality index puts Ahmedabad first on the strength of its faulty readings.

Headline: “O3 differs across City: Bhopal has the highest mean, 59.94 (n = 6,579), and Kochi the lowest, 3.99 (n = 3,556); η² = 0.09.”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| The pollutant column 'NO' (nitric oxide) was forced to text because its name looked like 'No.' for number. | high | fixed |
| Ahmedabad's CO, a copy of its NO column, was reported as the leading city. | high | partly fixed |
| Daily summary rows appended to the hourly data were not recognised. | high | fixed |
| The row limit left out two thirds of the file (only Apr 2019 to Jul 2020 analysed). | high | fixed |
| Seasonality and the 2020 lockdown were not detected, and the forecast was worse than naive. | medium | partly fixed |
| City rankings and pooled trends mixed cities with different coverage periods. | medium | partly fixed |
| Analysis took 10-13 s, mostly in cleaning. | medium | partly fixed |
| Exact zeros and flatlines in pollutant columns were not reported. | low | fixed |
| Comparing the same period across years was not supported. | low | not fixed |

## Still open

- The 2020 lockdown, the largest event in the data (April PM2.5 down a median 38% and NO2 down 45% within cities), is never mentioned, and a same-month year-on-year comparison cannot be asked.
- 'average AQI by City' ranks Ahmedabad first at 453.5 with no caveat, although its AQI is driven by the CO column the tool itself flags as a copy of NO.
- Pooled monthly and yearly answers mix a changing set of cities: the 'highest PM2.5 month' (Jan 2015) rests on only Delhi and Ahmedabad.
- The comparisons card's top finding is PM2.5 by AQI category (ε² = 0.52), which is close to circular because AQI is computed mainly from PM2.5 and PM10.
- The measure chosen, ozone, varies little between cities (9% of variance) and is not what most readers of Indian air-quality data would lead with; PM2.5 is offered only as an alternative.
- The brief takes about 18 seconds.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "air_quality_data.csv"`. The data is not stored in this repository.
