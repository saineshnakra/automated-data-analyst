# Gene Expression in Brain Cancer

[khushim10/gene-expression-in-brain-cancer](https://www.kaggle.com/datasets/khushim10/gene-expression-in-brain-cancer) on Kaggle · biology · 54,675 rows × 9 columns · licence: Unknown · round 2

Per-probe summary statistics from a microarray study of brain tumours: each of the 54,675 rows is a probe, and the columns are the count, mean, standard deviation, minimum, quartiles and maximum of log2 expression over 46 ependymoma samples. Four sibling files hold the same summaries for normal brain and three other tumour types.

**First run:** poor. The counts and ranks are right when asked directly, but the brief leads with a quartile printed as a percentage and trivial correlations among summary statistics, and several basic questions are refused because of the column names.

**Now:** good. It now recognises a table of per-probe summaries of 46 samples, leads with correct rankings by mean, spread and relative spread, and stops reporting correlations that hold by construction; the main limit is that the sibling tumour files cannot be brought in.

## What ADA finds now

This file summarises the expression of 54,675 microarray probes across 46 ependymoma samples, one row per probe. The analysis recognises that each row is already a summary of 46 values, uses the mean, and ranks probes by average expression, by spread and by spread relative to the mean; every figure checked is correct. It no longer reports the trivially high correlations between a probe's mean, minimum and maximum. It cannot combine this file with the matching files for normal brain and other tumour types, which is where the biology lies, and it does not point out that the top-ranked probes are array controls.

Headline: “AFFX-hum_alu_at has the highest mean, 14.52, and 1564631_at the lowest, 3.08.”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| A '%' in a column name ('25%', '50%') made values print as percentages. | high | fixed |
| The summary-statistics layout was not recognised: a quartile was the measure and the brief correlated statistics of the same values with each other. | high | fixed |
| Column names that are also statistic words (mean, std, max, min) broke correlation questions. | medium | fixed |
| The unnamed identifier column could not be referred to in plain words. | medium | fixed |
| The sibling files (one per tumour type, same probes and columns) cannot be compared. | low | not fixed |
| Small shares were printed as 0.0%. | low | fixed |

## Still open

- The four sibling files (normal brain and three other tumour types) cannot be analysed with this one, so the comparison the dataset exists for is not possible.
- The top-ranked probes by mean and by spread (AFFX-hum_alu_at, AFFX-HUMRGE/M10098_5_at) are array control probes, which the ranking does not flag.
- 'which probes vary the most' is refused; the same ranking is in the brief and 'which Unnamed: 0 has the highest std' works.
- The constant count column (46 samples on every row) is listed as a data issue with a note that it 'may be a fill value or a detection limit', although the cleaning note correctly reads it as the number of samples.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "Ependymoma_sample_data.csv"`. The data is not stored in this repository.
