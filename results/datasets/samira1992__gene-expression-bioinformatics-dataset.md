# 🧬 Gene Expression - Bioinformatics 🗃️ Dataset

[samira1992/gene-expression-bioinformatics-dataset](https://www.kaggle.com/datasets/samira1992/gene-expression-bioinformatics-dataset) on Kaggle · biology · 4,381 rows × 24 columns · licence: Other (specified in description) · round 3

The Spellman yeast cell-cycle microarray time course (cdc15 synchronisation): 4,381 genes, each with log-ratio expression at 23 times from 40 to 260 minutes. The first column is named 'time' but holds gene names, and the 40, 60 and 260 minute columns look interpolated from their neighbours.

**First run:** poor. Everything it reports is either a normalisation constant or an interpolation artefact, and it misses that the columns are a time course.

**Now:** good. It now reads the table as a gene-by-time course and leads with the cell-cycle periodicity (about 112 minutes, 562 genes), which reproduces exactly; the leftovers are a side card built on an interpolated column and some labelling slips.

## What ADA finds now

This is the classic yeast cell-cycle experiment: expression of 4,381 genes measured every 10 minutes from 40 to 260 minutes. ADA now recognises the columns as a time course and the first column as gene names, and its headline is the biology the data is known for: 562 genes follow a repeating pattern with a period of about 112 minutes, a figure that reproduces exactly. It also names the most variable genes and the average expression profile over time. Its 'strongest relationship' is between the 250 and 260 minute columns, which looks like an interpolation artefact in the file rather than a finding, and a couple of labels are wrong (it calls a standard deviation a mean).

Headline: “The columns 40 to 260 form a series; 562 of 4,381 rows follow a repeating pattern across it with a period of about 112.”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| Evenly spaced numbered columns (time points) were treated as an unordered matrix, so there was no time-course analysis. | high | fixed |
| The row mean was the measure although every row is centred, so the headline described a normalisation constant. | high | fixed |
| The gene-name column headed 'time' was not recognised as the identifier; rows were labelled by number. | medium | fixed |
| Interpolated columns were not recognised; the strongest 'relationship' was a real column against one averaged from it, and the one identity was stated backwards. | medium | partly fixed |
| Threshold questions on number-named columns failed, and ADA's own summary column leaked into column comparisons. | low | partly fixed |

## Still open

- The 'Strongest measurable relationship' card is 250 vs 260 minutes (r = 0.94), where 260 appears to be interpolated from 250; the 40 and 260 columns are not recognised as interpolated, and the periodicity test treats them as independent samples.
- The relationship card says 'Row SD (23 columns) = mean of the 23 columns 40 … 260', which is wrong: it is their standard deviation.
- ADA's own Row SD column still appears in column rankings ('which column has the highest average' puts it first) and in the generated question 'correlation between Row SD and 40' (r = -0.15), which means nothing.
- Summary text still says '4,381 rows, one per time' though it has recognised the column as gene names.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "Spellman.csv"`. The data is not stored in this repository.
