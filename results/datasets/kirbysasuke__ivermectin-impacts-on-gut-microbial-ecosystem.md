# Ivermectin's Influence on Gut Microbiota

[kirbysasuke/ivermectin-impacts-on-gut-microbial-ecosystem](https://www.kaggle.com/datasets/kirbysasuke/ivermectin-impacts-on-gut-microbial-ecosystem) on Kaggle · biology · 198 rows × 14 columns · licence: CC0: Public Domain · round 3

Concentrations of 13 short-chain fatty acids in 198 samples from a gut-simulator study of the drug ivermectin. The design (reactor, colon region, stage) is coded only in the sample ID, and the full sample sheet is in a second file.

**First run:** partly useful. The correlations and model are computed correctly, but with the design hidden in the sample IDs ADA can only correlate the acids with each other, which says nothing about ivermectin or gut region, and it drops six all-zero or constant columns without saying so.

**Now:** partly useful. The correlations and new data checks are right, but with the design hidden in the sample IDs the report can only relate the acids to one another, which answers no study question.

## What ADA finds now

This table holds 13 short-chain fatty acids measured in 198 samples from a gut-simulator study of the drug ivermectin. The tool reports, correctly, that the main acids rise and fall together (acetic and propanoic acid, r = 0.64), and it now flags the five acids that were never detected or hold one fixed value, and the one that is zero in two thirds of samples. Filters on part of a sample name now work, so a user can pull out one reactor or gut region by hand. But the study design is written only into the sample names and a separate sample sheet, and the tool uses neither, so its report says nothing about the drug or the gut region. Its combined model of one acid on the others sets the most closely related acids aside and adds little.

Headline: “Acetic_acid and Propanoic_acid move together: Pearson r = 0.64 (95% CI 0.55 to 0.71, n = 198).”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| The study design is coded in the parts of the sample ID and is not split out, so there is no grouping. | high | not fixed |
| The dataset's sample sheet (treatment stage, supplementation, colon region) is not used. | medium | not fixed |
| All-zero and single-value columns were dropped without a note. | medium | fixed |
| The headline was a joint model of one fermentation product on its sibling readings. | medium | partly fixed |
| Mostly-zero readings (non-detects) enter Pearson correlations and the model as ordinary values. | low | partly fixed |
| Filters without '=' and substring filters on a text ID failed. | low | fixed |

## Still open

- The study design (reactor, colon region, treatment stage) is coded in the sample IDs and a separate sample sheet, and the tool uses neither, so it says nothing about the drug or gut region.
- The joint model sets aside propanoic and butanoic acid as 'candidate outcomes' and then compares itself with a weaker single predictor; propanoic acid alone explains as much (41%).
- Pentanoic acid, zero in two thirds of samples, still leads the model and the correlation list on its Pearson r, though the tool's own check says rank correlation is the better guide.
- All nine unusually high acetic-acid values come from one reactor (RB3); the outlier card does not notice.
- The rows are repeated samples from a few vessels, so the model's p-values are too small, and the card does not say so.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "SCFA_data.csv"`. The data is not stored in this repository.
