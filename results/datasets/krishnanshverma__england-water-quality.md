# England Water Quality Sampling Dataset (2025)

[krishnanshverma/england-water-quality](https://www.kaggle.com/datasets/krishnanshverma/england-water-quality) on Kaggle · environmental · 81,026 rows × 13 columns · licence: CC0: Public Domain · round 1

Environment Agency water-quality samples from England, 2 January to 6 February 2025, in long format: each of the 81,026 rows is one of 355 determinands (ammonia, pH, temperature, E. coli and so on) measured at a sampling point, with a result and its unit (30 units). Each determinand has to be analysed on its own.

**First run:** poor. Every number is computed correctly but averages pH, temperature, E. coli counts and grid coordinates together, so nothing in the report means anything, and the natural per-determinand questions are refused or answered with the ungrouped mean.

**Now:** partly useful. It now handles the long format properly, analyses ammonia on its own with tests and answers questions about any other determinand, but it silently merges different units and determinands during cleaning and leads with a weak latest-week change.

## What ADA finds now

ADA recognises that this file mixes 355 different measurements in one result column and analyses the most common one, ammonia, on its own, while answering questions about any other measurement such as pH or water temperature. For ammonia it finds that samples taken in response to pollution incidents run far higher than routine monitoring, with a rank test, and it notes when a figure rests on only a few samples. Its headline still opens with last week's change in the average, which is driven by a few very high readings while the typical value barely moves. During cleaning it wrongly merges depth readings in metres with volumes in millilitres, and two different chloroform measurements, so some answers mix units. Its anomaly list also flags single days with only one or two samples.

Headline: “-0.329 latest movement; result differs across sample.purpose.label: UNPLANNED REACTIVE MONITORING FORMAL (POLLUTION INCIDENTS) has the highest median of per-sample.samplingPoint.notation medians, 3.77 (n = 22), and MONITORING  (UK GOVT POLICY - NOT GQA OR RE) and STATUTORY FAILURES (FOLLOW UPS AT NON-DESIGNATED POINTS) the lowest, 0.03 (n = 14); ε² = 0.22.”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| The long-format table was analysed as if 'result' were one quantity, mixing pH, temperature, E. coli counts and grid coordinates. | high | fixed |
| A 'by X' question on a column with more than 100 values silently dropped the grouping. | high | fixed |
| Filtering to one category value ('for pH', 'Ammonia(N)') not supported. | medium | fixed |
| Change drivers and period changes reported without the rows behind them. | medium | fixed |
| Code and coordinate columns treated as quantities (determinand.notation, easting/northing). | low | fixed |

## Still open

- Two cleaning merges are wrong: unit 'm' (614 rows of sample depth, water depth and transparency in metres) is merged into 'ml' (filtered volume), and 'CHLOROFORM' (notation 7558, a load in t/d) is merged into 'Chloroform' (notation 3373, a concentration in ug/l), so 'average result for Chloroform' gives 2.59 from 26 rows instead of 2.93 from the 23 concentration rows.
- The headline still opens with the latest week's change in the mean (0.66 to 0.33), on a reading so skewed that the median barely moves (0.051 to 0.042) and the weekly trend answer finds no clear trend.
- The anomaly card flags single days backed by one or two samples (12 January: 2 samples, mean 17.5; 25 January: 1 sample) without giving the counts.
- The headline is hard to read: long internal column names and two tied lowest groups given one sample size.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "water_quality.csv"`. The data is not stored in this repository.
