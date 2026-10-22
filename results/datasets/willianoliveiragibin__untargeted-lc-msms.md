# Untargeted LC-MS/MS

[willianoliveiragibin/untargeted-lc-msms](https://www.kaggle.com/datasets/willianoliveiragibin/untargeted-lc-msms) on Kaggle · chemistry · 1,230 rows × 169 columns · licence: CC0: Public Domain · round 3

A feature table from untargeted LC-MS/MS metabolomics of pea seeds: 1,230 detected features with retention time, m/z, annotation fields and peak intensities in 147 injections (90 seed samples, 29 pooled QC runs and 28 other runs). Only 123 features carry a library annotation; the rest hold '-' or 0. Numbers use decimal commas.

**First run:** poor. The mass columns are misread or left as text, the headline is a by-construction artefact of the 'not annotated' placeholder, and separate samples are dropped as copies. Nothing a metabolomics researcher needs (sample groups, QC repeatability, features that differ) is attempted.

**Now:** poor. The reading problems are fixed, but the headline is still a by-construction artefact built on the 'not annotated' placeholder, and nothing a metabolomics researcher needs (sample groups, QC repeatability) is attempted.

## What ADA finds now

This is a metabolomics feature table: 1,230 detected compounds measured in 147 injections of pea-seed samples, pooled quality-control samples and other runs. The tool now reads the decimal commas correctly, recognises that the zeros and dashes on 1,107 unannotated features mean 'not recorded', and ranks named compounds properly when asked. But its headline still says that features with a library match always have a fragment spectrum, which is true by how such matches are made, and it uses the 'not annotated' group in that comparison. It summarises each compound by one average over every injection and never compares the seed groups or checks how repeatable the quality-control injections are, which is what a researcher would look at first.

Headline: “The share of rows with MS/MS 1 differs across Annotation Source: highest 100.0% in Bruker MetaboBASE Personal Library 2.0_in-silico (n = 13) against 79.9% in 0 (n = 1,107), a difference of 20.1 percentage points (95% CI -2.8 to 22.6); chi-squared test, p < 0.001 across all 4 groups; Cramér's V = 0.16.”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| Decimal commas were parsed inconsistently (some values 1,000 times too large) and the mass columns were left as text, with no note. | high | fixed |
| Placeholder zeros and dashes were treated as real values and as a group, giving circular findings and a by-construction headline. | high | partly fixed |
| The near-duplicate rule (/r/ >= 0.99 on raw values) called separate samples copies and dropped them. | high | partly fixed |
| Matrix detection took in non-sample columns and pooled every sample type into one row mean, which became the measure. | high | partly fixed |
| No per-feature comparison across the sample groups in the column names and no QC repeatability check. | medium | not fixed |
| Ranking by Name ranked the '-' placeholder and dropped every real compound; 'top 5' returned 1. | medium | fixed |
| The headline puts a confidence interval that includes zero next to p < 0.001. | low | not fixed |

## Still open

- The headline restates how the table was built: features with a library match always have an MS/MS spectrum, and it uses the 'not annotated' placeholder as one of its groups even though the tool flags that placeholder elsewhere.
- The sample design coded in the column names (seed series, pooled QC, other runs) is never used: no per-feature comparison across sample groups and no QC repeatability check.
- The measure is one average over every injection, QC and other runs alike, and the grouping is an instrument flag; the top relationship (row mean vs Boxplot, r = 0.98) is a summary of the same intensities.
- Separate biological replicates and repeat QC injections are described as probably measuring the same thing.
- 'how many rows have an annotation' returns 14 distinct annotation values instead of the 123 annotated features, and 'Annotation Source not 0' is refused.
- The headline places a confidence interval that includes zero next to p < 0.001 without saying they come from different tests.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "Untargeted LC-MSMS new.csv"`. The data is not stored in this repository.
