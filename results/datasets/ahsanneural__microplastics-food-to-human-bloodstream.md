# Microplastics: Food to Human Bloodstream

[ahsanneural/microplastics-food-to-human-bloodstream](https://www.kaggle.com/datasets/ahsanneural/microplastics-food-to-human-bloodstream) on Kaggle · environmental · 354 rows × 53 columns · licence: Attribution 4.0 International (CC BY 4.0) · round 2

A literature compilation that stacks five different tables in one wide file (354 rows, 53 columns, 75% empty): microplastic counts in foods and drinks (301 rows), human organs (20), other biological samples (14), blood by particle type (12) and coagulation markers in low- and high-exposure groups (7). The main food reading is written as text with mixed units, such as '5.66 ± 1.69 items/10 g'.

**First run:** poor. It no longer crashes, but the headline averages a p-value over rows with no group, the main measurement is taken as an ID, and two of the data checks are false alarms; nothing a researcher could use.

**Now:** partly useful. It now reads the particle counts with their units, keeps the five stacked tables apart and reports an honest 'no clear difference' on the rows it covers, but those are a fifth of the file, ranges and below-detection results are dropped, and the data checks are swamped by other tables' empty columns.

## What ADA finds now

This file stacks five tables from published studies on microplastics in foods, organs and blood, with counts written as text in many units. ADA now recognises the stacked tables and keeps them apart, reads counts such as '1.8 ± 0.34 items/individual' as numbers, and never mixes units; on the 75 foods counted per individual animal or item it finds no clear difference between food categories (median 1.8 for seafood, with only 4 and 2 items in the other groups). It also answers questions about the other tables, such as the coagulation markers, and says which table each answer comes from. It still drops readings written as ranges and results below the detection limit, so the summary rests on a fifth of the file, and its list of data issues is filled with 'missing' warnings for columns that belong to the other tables.

Headline: “particle_number shows no clear difference across the 3 food_category groups (Kruskal-Wallis, p = 0.15; ε² = 0.05): medians run from 1.80 (Marine animal origin, n = 69) to 7 (Food Containers, n = 4).”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| A file that stacks several tables was analysed as one flat table, so the measure and grouping came from blocks that never share a row. | high | fixed |
| Test statistics (p-values) could be picked as the measure, and a column filled on 2% of rows won. | high | fixed |
| A text column of 'number ± spread unit' cells was not read as a measurement and was taken as the identifier. | high | fixed |
| 'ND' was read as missing and described as 'not stated', though it means 'not detected'. | medium | partly fixed |
| The 'max below min' check read high/low exposure groups as upper/lower bounds and applied to SDs. | medium | fixed |
| Answers ended with 'Based on all N rows' when only a handful of rows held the measure. | medium | fixed |
| A grouping whose rows carry several units was ranked on one scale. | medium | fixed |
| A constant compilation date (last_updated = '2025-02') was used as the date. | low | fixed |
| The cap check fired on a study's round sample size (2,000). | low | fixed |

## Still open

- 40 of the 41 data checks are '100% missing' warnings for columns that belong to the other four stacked tables, and completeness is given as 25.1%, so the real gaps in the food table are buried.
- The brief covers only 75 of the 354 rows (foods counted per individual); 115 readings written as ranges and 20 below-detection results are dropped, and below-detection results are real low values that bias the medians up when left out.
- The headline names Food Containers (n = 4) as the highest median although the method says groups under 5 are not named, and comparing food groups of 69, 4 and 2 items says little.
- 'how many rows per source_table' now answers only for the 75 rows in focus ('food_water_beverages ... at 75 (100.0% of the total)') instead of listing the five tables (301, 20, 14, 12, 7).
- 'average particle_number for particle_number unit items/g' gives a mean of 17,018 that rests on five values from one study (50,550 to 195,500; the other 34 are under 3), with no warning.
- 'how many particle_number values are ND' is still refused.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "master_microplastics_human_body.csv"`. The data is not stored in this repository.
