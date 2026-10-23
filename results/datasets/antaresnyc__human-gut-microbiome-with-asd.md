# Human gut microbiome of the children with ASD

[antaresnyc/human-gut-microbiome-with-asd](https://www.kaggle.com/datasets/antaresnyc/human-gut-microbiome-with-asd) on Kaggle · biology · 5,619 rows × 61 columns · licence: CC0: Public Domain · round 1

A species-by-sample abundance grid: 5,619 rows are bacterial taxa and 60 columns are stool samples, 30 named A* (children with autism) and 30 named B* (typically developing children). Cells are read counts, and two thirds of them are zero.

**First run:** poor. Every number is arithmetically right, but ADA analysed one sample column as if it were the outcome and missed the ASD vs control design encoded in the column names.

**Now:** poor. Every number is arithmetically right and it now recognises the grid and the A/B groups, but the brief has no usable finding: the headline is the grand mean of all counts and the autism vs control comparison that the data exists for is not attempted.

## What ADA finds now

This file is a grid of bacterial species (rows) by 60 stool samples (columns), half from autistic children and half from typically developing children. ADA recognises that it is a grid, uses the species names to label rows, and points out that the sample names split into an A group and a B group. It answers questions such as which species is most abundant in a given sample correctly. But it does not compare the two groups of children, which is the point of this data, and its headline is only the average count over the whole grid. Its correlation and outlier cards describe the usual shape of microbiome data (a few species dominate every sample) rather than anything a researcher would act on.

Headline: “Row mean (60 columns) averages 15.13 across all rows.”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| The table was not recognised as a samples-by-taxa grid, so one sample column (A101) was headlined as the measure. | high | partly fixed |
| The ASD vs control comparison encoded in the A*/B* column prefixes was missed. | high | partly fixed |
| 1.5x IQR outliers on zero-inflated counts flagged 14% of values and suggested entry errors. | medium | fixed |
| 'Which <row label> has the highest X' returned the value without naming the row. | medium | fixed |
| Unmapped parts of a question were silently dropped ('top 5 Taxonomy by A3' answered as the mean of A3). | medium | fixed |
| The strongest-pair card (two samples correlated across taxa) is uninformative, because the same dominant taxa top every sample. | low | not fixed |
| The unique Taxonomy label was not used as the row label. | low | fixed |
| Boilerplate summary line with no content. | low | fixed |

## Still open

- It does not compare the autism (A*) and control (B*) samples, although it names the two groups; after normalising for read depth, 286 taxa differ at a 5% false discovery rate.
- The headline is the average of every count in the grid (15.13), which says nothing about the children or the species.
- The only relationship card is a 0.97 correlation between two samples, with no note that nearly all sample pairs correlate highly (median 0.78) because the same few species dominate every sample.
- The 'exceptional records' card flags the 123 most abundant species, which is the expected long tail of microbiome data rather than anything unusual.
- The 'are there outliers' answer names only 2 columns with values outside the fences without saying that the other 59 sample columns were skipped because they are mostly zeros.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "ASD meta abundance.csv"`. The data is not stored in this repository.
