# Gene Expression Dataset

[ziya07/gene-expression-dataset](https://www.kaggle.com/datasets/ziya07/gene-expression-dataset) on Kaggle · biology · 1,000 rows × 1001 columns · licence: CC0: Public Domain · round 2

A table said to hold expression of 1,000 genes (Gene_1 to Gene_1000) for 1,000 samples labelled AML (409), ALL (380) or Healthy (211). Every one of the million values is a uniform random number between 0 and 5, so the data are simulated.

**First run:** partly useful. Every number is right and the null result is correct, but ADA leads with a row mean, screens only 300 genes with Holm, and never says the data look like uniform random numbers.

**Now:** good. It warns up front that the table looks simulated, screens all 1,000 genes with a proper false-discovery-rate adjustment and correctly finds nothing; the quibbles are a headline about the per-sample mean and follow-up answers on the top gene that drop the selection caution.

## What ADA finds now

This file claims to hold expression of 1,000 genes in 1,000 leukaemia and healthy samples, but every value is a uniform random number between 0 and 5. The analysis says up front that the columns look flat and unrelated and that the table may be simulated. It tests every gene across the diagnoses with a correction for 1,000 tests and correctly finds none that differ, noting that 59 nominal hits is about what chance gives. Its headline is about each sample's overall average rather than the gene screen, and when asked about the top gene on its own it reports a significant difference without repeating that the gene was the best of 1,000 chances.

Headline: “Row mean (1,000 columns) shows no clear difference across the 3 Diagnosis groups (Welch ANOVA, p = 0.77; η² = 0.00): means run from 2.501 (AML, n = 409) to 2.504 (Healthy, n = 211).”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| Only the 300 most variable genes were compared across diagnoses, with a Holm adjustment, instead of screening every gene with a false-discovery-rate adjustment. | medium | fixed |
| The brief leads with each sample's mean over all genes, a quality-control number, rather than the per-gene comparison. | medium | not fixed |
| No warning that the data look simulated. | medium | fixed |
| 'which genes differ between AML and ALL' was answered about the row mean across all three groups. | medium | fixed |
| The 'Did you mean' list in a refusal listed all 1,000 gene names. | medium | fixed |
| A follow-up test on the gene singled out as best of many is reported as a plain significant difference. | low | not fixed |
| Outlier cards gave no chance baseline; awkward 'one row per row' wording. | low | fixed |

## Still open

- The headline is about each sample's average over all 1,000 genes, a quality-control figure; the per-gene screen a researcher wants is the third card.
- Follow-up questions about the top gene (Gene_149) answer 'differs, p < 0.001' and 'Yes' without repeating that it was the best of 1,000 tests and does not survive adjustment.
- The relationship card scans only 300 of the 1,001 columns (stated in its calculation), unlike the full 1,000-gene group screen.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "leukemia_gene_expression.csv"`. The data is not stored in this repository.
