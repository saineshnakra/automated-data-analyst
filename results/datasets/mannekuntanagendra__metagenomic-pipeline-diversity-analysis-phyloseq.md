# Metagenomic Pipeline Diversity Analysis Phyloseq

[mannekuntanagendra/metagenomic-pipeline-diversity-analysis-phyloseq](https://www.kaggle.com/datasets/mannekuntanagendra/metagenomic-pipeline-diversity-analysis-phyloseq) on Kaggle · biology · 2,000 rows × 27 columns · licence: MIT · round 2

A bacterial abundance table from a published microbiome survey: 2,000 OTUs (rows, ID in an unnamed first column) by 26 samples (columns) from soil, faeces, skin, tongue, freshwater, ocean, sediment and mock communities. Values are raw read counts; sample depths differ 60-fold and 14-57% of each sample's cells are zero.

**First run:** poor. The arithmetic is right, but ADA treated a sample-by-OTU count matrix as an ordinary table and headlined the mean of one arbitrary sample, half of which is a single OTU, so a microbiome researcher gets nothing usable.

**Now:** poor. Ask now answers sample-level questions correctly, but the brief a new user sees headlines the grand mean of raw counts across samples of very different depth and offers no per-sample summary, so it says nothing useful about this microbiome table.

## What ADA finds now

This file is a microbiome abundance table: read counts for 2,000 bacterial types across 26 samples from soil, water, gut and skin. The analysis recognises it as a matrix and correctly warns that sample depths differ about 60-fold and should be put on a common scale. Its headline, though, is the average count per bacterial type across all samples, a number that describes no sample, and the summary has no per-sample depth, share of zeros or count of types present. Asked directly, it gives correct sample totals, zero counts, the most abundant type in a sample and sensible correlations between replicate samples.

Headline: “Row mean (26 columns) averages 508.95 across all rows.”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| The sample-by-OTU count matrix was not recognised, so one sample column was picked as the measure and the other 25 became correlates. | high | partly fixed |
| The headline average was dominated by one record, with nothing said. | high | partly fixed |
| 'What correlates with X' silently dropped near-duplicate and same-stem columns. | medium | fixed |
| The near-duplicate rule used raw Pearson r on very skewed counts. | medium | fixed |
| No sequencing-depth, sparsity or diversity summary. | medium | partly fixed |
| Counting rows equal to a value was refused. | medium | fixed |
| The log-scale outlier note ignored zeros and did not name the flagged records. | low | partly fixed |

## Still open

- The headline, 'Row mean (26 columns) averages 508.95', is the grand mean of raw counts pooled over samples whose depths differ 60-fold, which the tool's own note says not to do; it describes no sample and no OTU.
- The brief has no per-sample summary (depth, share of zeros, number of OTUs present), which is what a microbiome researcher checks first; these only come out of Ask.
- The relationship card says 'Even1 and Even2 move together by rank: the rank correlation is only 0.54, while Pearson r = 0.98', which contradicts itself; it is the raw values that move together.
- The 'Exceptional records' card reports 79 OTUs with unusually high or low average counts, which is the normal shape of abundance data, not a finding, and does not name them.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "06_GlobalPatterns_OTU.csv"`. The data is not stored in this repository.
