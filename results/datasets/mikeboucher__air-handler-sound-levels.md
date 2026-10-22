# Air Handler Sound Levels

[mikeboucher/air-handler-sound-levels](https://www.kaggle.com/datasets/mikeboucher/air-handler-sound-levels) on Kaggle · sensor · 178,280 rows × 15 columns · licence: CC0: Public Domain · round 3

Three years (June 2022 to May 2025) of uplinks from one sound-level sensor inside a home HVAC air handler: 178,280 rows at a nominal 5-minute interval with sound level (dBA, valid range 40-100 per the header), onboard temperature, battery voltage and radio diagnostics.

**First run:** poor. It reports totals of a running packet counter and misses the sound level entirely; the one correct check (the temperature code) stops short of the same failed rows in sound and battery.

**Now:** good. The report now reads the sound level as an on/off fan signal with a clear seasonal pattern, sets aside the counters and failed records, and reports the logging gaps; what remains are small inconsistencies and a confusing latest-quarter card.

## What ADA finds now

This is three years of five-minute readings from a sound sensor inside a home air handler. The tool now leads with what matters: the sound level sits either around 41 dBA (fan off) or around 81 dBA (fan on), and the fan is running in 80% of January readings but only 8% of April ones. It sets aside the two running radio counters, flags the 1,500 failed records where sound, temperature and battery all read impossible values, applies the 40-100 dBA range given in the column header and reports that only 58% of the expected readings were logged. Monthly and quarterly averages still include the failed zero readings, which pulls them down by under 1 dBA, and a card about the latest quarter treats the usual winter rise as news before calling it ordinary.

Headline: “Sound Level (dBA) (40 - 100) is either around 41 or around 81; the share of readings at 60.25 or above runs from 7.9% in April to 80.1% in January.”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| A running counter (Packet Count) was chosen as the measure and summed per period. | high | fixed |
| The physical reading the dataset is about (sound level) was not picked or even listed. | high | fixed |
| The valid range in the header was ignored, and a failed-record pattern was caught in temperature but not in sound and battery. | high | fixed |
| Five-minute data was summarised by quarter and a regular seasonal rise was called unusual. | medium | partly fixed |
| Gaps in the regular sensor log were not reported. | medium | fixed |
| The run took about 22 s, almost all of it reading the workbook. | medium | partly fixed |
| A question asking for a share per period was answered with counts. | medium | fixed |
| Share and concentration cards used business language for a radio relay. | low | partly fixed |
| A measure override with the column name exactly as in the file was rejected. | low | not fixed |

## Still open

- Averages by month, quarter and gateway still include the 1,500 failed records (sound = 0) that the checks set aside, with only a caution; January reads 72.5 dBA instead of 73.4.
- The 'latest movement' card treats the usual winter rise as news before concluding it is ordinary for the time of year, which is confusing.
- A 'largest change by gateway' card reports a radio relay as if it were a business segment.
- Reading the large workbook still takes most of the 12-second run.
- A measure named exactly as in the file header (with its line break) is rejected.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "20250526_Datacake Outbound Webhook Log - TBSL100 - CLEANED.xlsx" --sheet "Data"`. The data is not stored in this repository.
