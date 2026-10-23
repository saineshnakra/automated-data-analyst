# Genomics of Drug Sensitivity in Cancer (GDSC)

[samiraalipour/genomics-of-drug-sensitivity-in-cancer-gdsc](https://www.kaggle.com/datasets/samiraalipour/genomics-of-drug-sensitivity-in-cancer-gdsc) on Kaggle · biology · 242,035 rows × 19 columns · licence: GPL 3 · round 2

Drug-screen results from a large cancer cell-line project: one row per cell line and drug (969 cell lines, 286 drugs, 242,035 rows) with the drug response (log IC50, AUC and a per-drug Z-score), cell-line annotations (cancer type, tissue, MSI status, growth type, which omics data exist) and each drug's target and pathway.

**First run:** partly useful. The statistics are sound at the cell-line level and the filtered Ask rankings are right, but the brief ignores drugs (the point of the dataset), misses that Z_SCORE is standardised, and headlines an extreme group of 2 cell lines.

**Now:** partly useful. The headline (cancer type explains 43% of a cell line's average log IC50, pancreatic lines least sensitive and lymphoma lines most) is right and now checked for the overlap with growth type, but the brief still ignores the drugs that explain most of the variation, reports conflicting variance shares, and spends a card on data-availability flags.

## What ADA finds now

This file holds about 242,000 drug-response measurements from 969 cancer cell lines and 286 drugs. The analysis averages each cell line's response, finds that cancer type explains about 43% of the differences (pancreatic lines least sensitive, lymphoma lines most), and checks this against the overlap with whether lines grow attached or in suspension. Asked directly, it ranks drugs and pathways, counts drugs and cell lines, and filters to one drug correctly. The summary never mentions the drugs, which account for most of the variation, gives three different figures for how much growth type explains, and spends part of its summary on which kinds of genomic data exist for each line.

Headline: “LN_IC50 differs across Cancer Type (matching TCGA label): PAAD has the highest mean of per-COSMIC_ID means, 3.81 (n = 27), and DLBC the lowest, 1.53 (n = 34); η² = 0.43.”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| The drug, the main design factor, never appears in the brief; many-level groupings are dropped. | high | not fixed |
| The standardised Z_SCORE was not recognised: its near-zero mean was a KPI and its correlation with its own source was reported as a lead. | medium | partly fixed |
| The headline's highest-vs-lowest comparison used a group of 2 cell lines. | medium | partly fixed |
| A 21%-missing grouping was chosen over a nearly identical complete one, and a placeholder label was kept as a group. | medium | partly fixed |
| The same card gave two different η² values for the same pairs without saying which unit each used. | medium | not fixed |
| 'How many <noun>' fell back to the row count. | medium | fixed |
| Effect-size wording and the large-sample caveat were misapplied. | low | fixed |
| Overlap between groupings was not noted (suspension lines are mostly blood cancers). | low | fixed |
| Slow on a quarter-million rows. | low | not fixed |
| Repeated cell-line and drug pairs were not reported. | low | not fixed |

## Still open

- The brief never mentions the drugs, which explain 71% of the variation in log IC50; a researcher must ask to see which drugs or pathways are most potent.
- Figures for the same comparison disagree: growth type explains '15%', 'η² = 0.24' and '28%' of log IC50 in one brief, and Ask pairs row-level means and counts (PAAD 3.79, n = 7,083) with the per-cell-line η² of 0.43, where across rows cancer type explains 4.5%.
- The within-growth comparison is computed on a 50,000-row sample, about 50 of each cell line's roughly 250 drugs, so its cell-line means are noisy; it names semi-adherent groups of 2 cell lines as extremes and gives COAD/READ (n = 5) as the highest suspension group where the full data give SCLC (n = 34).
- A whole card compares which omics data exist ('Methylation = Y' by 'Gene Expression = Y', 97% vs 58%), which says nothing about drug response.
- Ask still reports Z_SCORE vs LN_IC50 (r = 0.51) as a relationship, though the brief knows one is built from the other.
- The 9 drugs screened under two IDs are pooled silently, and the full run takes over 20 seconds.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "GDSC_DATASET.csv"`. The data is not stored in this repository.
