# Indoor Laboratory Fire Dataset

[saurabhshahane/indoor-laboratory-fire-dataset](https://www.kaggle.com/datasets/saurabhshahane/indoor-laboratory-fire-dataset) on Kaggle · sensor · 2,974 rows × 9 columns · licence: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) · round 1

One of eight controlled indoor fire experiments (electrical_4.csv): 2,974 one-second readings over about 50 minutes from humidity, temperature, MQ139 gas, TVOC and eCO2 sensors, with a smoke-detector ON/OFF flag and a 0/1/2 fire-phase label (Status). The research interest is how and when each sensor responds as the fire develops.

**First run:** poor. The numbers are right, but ADA invents a date, can't look at a 1 Hz series over time, groups by the wrong label, and calls the fire itself a batch of entry errors.

**Now:** partly useful. The fire phase, the jump in level at 15:48 and the differences across phases are right and tested, but the headline opens with the last minute's change and a data check wrongly suggests the detector-ON readings were filled in.

## What ADA finds now

ADA now follows this one-second fire experiment minute by minute: it finds the gas reading jumping to a new level at 15:48, when the fire phase begins, and shows that every sensor differs strongly across the three phases, with tests and effect sizes. It reads the clock times honestly without inventing a date, spots the eCO2 sensor resting at its 400 floor, and correctly explains why two gas readings agree on a straight line but not in rank. The headline still opens with the change in the final minute, which says little about an experiment that has ended. One data check wrongly suggests that the readings taken while the smoke detector was on were filled in, when they are just the last 51 seconds of the run. It still cannot say when each sensor first responded, and it looks at only one of the eight experiment files.

Headline: “+26.20 latest movement; MQ139 differs across Status: Status 2 has the highest mean, 308.52 (n = 442), and Status 0 the lowest, 104.22 (n = 215); η² = 0.87.”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| Clock times with no calendar date were given today's date, inventing a date range. | high | fixed |
| No sub-daily grain, so a 1-second series got no trend, anomalies or forecast, and no view of when signals left baseline. | high | partly fixed |
| The numeric phase label Status was not offered as a grouping. | high | fixed |
| IQR outliers in time-ordered data presented as suspect records when they are one contiguous fire event. | medium | fixed |
| A Pearson/Spearman gap was always explained as 'a few extreme records'. | medium | fixed |
| Numeric group label printed without its column name; row counter called 'unique entities'. | low | fixed |
| Only one of eight same-schema experiment files is analysed. | low | not fixed |
| Event, minute-trend and filtered-count questions refused. | medium | partly fixed |

## Still open

- A data check says the 51 detector-ON rows 'may have been filled in from reference ranges' because their readings sit in narrow bands, but they are simply the last 51 seconds of the run (15:55:02 to 15:55:52); it is the one item the summary names under 'Check these first', and it is attached as a caution to every Detector answer.
- The headline and two cards still open with the change in the last minute ('+26.20 latest movement'), and the 'Largest change by status' card is true by construction ('It is the only status with figures in both periods'), followed by advice to check whether the change 'persists or is a one-off' for a run that has ended.
- It cannot say when a sensor first responded or when the detector first turned on: 'when does TVOC start rising' and 'when did the Detector first turn ON' are refused.
- Fire phases are described as groups that 'cover a different stretch of time', which reads as a data problem when the phases are stretches of time by design.
- Only one of the eight experiment files is analysed, so the comparison across fire sources is not possible.
- The MQ139 baseline (31% of readings between 116 and 119) is flagged as a possible 'default entry, rounding' pile; it is the sensor at rest before the fire.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "Indoor Laboratory Fire Dataset /electrical_4.csv"`. The data is not stored in this repository.
