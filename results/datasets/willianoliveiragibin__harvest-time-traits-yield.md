# Harvest Time_Traits_Yield

[willianoliveiragibin/harvest-time-traits-yield](https://www.kaggle.com/datasets/willianoliveiragibin/harvest-time-traits-yield) on Kaggle · agriculture · 107 rows × 10 columns · licence: CC0: Public Domain · round 2

107 plot-year records (10 plots, 2005 to 2015) from the Shenyang agro-ecological station in China, with bilingual Chinese and English headers: plot code and name, farmland type, rotation system, the crop grown that year and a multiple cropping index. Despite the title there are no yield or trait values, and the only number, the cropping index, is 100 on every row.

**First run:** poor. The file was read correctly, but every finding is about a constant column (0.0-point changes, a flat forecast), and the real content (which crop each plot grew each year) can only be reached by asking.

**Now:** partly useful. The brief is now honest (the constant index is flagged and record counts are not trended) and Ask reaches the plots, crops and crop-by-year table, but the brief itself reports only the dry-field share and misses the rotation pattern.

## What ADA finds now

This file records which crop each of 10 plots at a Chinese field station grew each year from 2005 to 2015; despite its title it contains no yields. ADA reads the bilingual headers and decimal commas correctly, notices that the only numeric column, the multiple cropping index, is 100 on every row and sets it aside, and recognises that each plot appears once a year, so it does not trend or forecast the record counts. Asked directly, it correctly counts 10 plots and 3 crops and gives the crop-by-year table. But its summary only reports that 79% of the records are dry fields, it describes a clear crop rotation as 'no clear association', and it does not flag plot codes that carry two different names.

Headline: “旱地 Dry field has the most rows: 85 of 107 (79.4%).”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| A constant column (the cropping index, 100 everywhere) was taken as the measure, then trended, compared and forecast. | high | fixed |
| 'how many plots' and 'how many crops' counted rows instead of the matching column. | medium | fixed |
| No cross-tab of categories over time (crop by year), the file's main content. | medium | partly fixed |
| No consistency check between a code column and its name column. | low | not fixed |
| Bilingual headers are repeated in full in every sentence, and their case changes. | low | not fixed |

## Still open

- The brief says almost nothing: its one finding is that 79% of records are dry fields, while the crop rotation over the years (soybean only in 2007 and 2010 to 2014, maize every year, rice on 1 or 2 plots) is shown only if asked.
- The crop-by-year answer calls the pattern 'no clear association' (p = 0.053), using a test that treats the same 10 plots each year as independent records.
- Four plot codes carry two different plot names each, and this is not flagged.
- Bilingual headers are repeated in full in every sentence.
- Ask breakdowns of the constant cropping index report a 'tie for the highest' at 100% instead of saying the column never changes.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "Harvest Time_Traits_Yield new.csv"`. The data is not stored in this repository.
