# Disease Recovery PLOS paper  😷🙄

[qusaybtoush1990/disease-recovery-plos-paper](https://www.kaggle.com/datasets/qusaybtoush1990/disease-recovery-plos-paper) on Kaggle · biology · 168 rows × 7 columns · licence: Unknown · round 3

Repeated measurements of 15 infected mice (indiv) at daily time points (timept 0-15), with two disease-space coordinates x and y per visit and the outcome (Dead or Live). The 11 mice that died were measured to day 8-10; the 4 survivors to day 15.

**First run:** poor. The file is refused, and once loaded by hand the headline compares mouse IDs while the group comparison mixes unequal follow-up with repeated visits, so it misses the clear survivor/non-survivor difference that per-mouse or matched-day comparisons show.

**Now:** partly useful. The file now loads, mouse IDs are treated as subjects and the headline is honest about unequal follow-up, but the profile and correlation cards still read survivorship as a trend and the clear survivor/non-survivor difference is not found.

## What ADA finds now

This is a small study of 15 infected mice measured daily, where the 11 that died were followed only to about day 9 and the 4 survivors to day 15. ADA now reads the file, treats each mouse as one subject, and leads by noting the unequal follow-up: over the days both groups were measured, mice that died averaged a higher x (4.99) than survivors (3.20), though with 4 survivors it does not call this clear. It still says x 'rises' over time and relates to time, when that comes from only survivors being measured after day 10. It misses the striking per-mouse result: survivors reach a far higher peak x than mice that died, and some plain-language questions still pool visits as if they were independent.

Headline: “The groups were followed for different lengths along timept (Dead to 10 and Live to 15); over the stretch they share (timept 0 to 10), x averages Dead 4.99 and Live 3.20.”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| The .tsv file was refused. | high | fixed |
| The mouse ID (written as decimals) was read as a numeric reading and put in the headline. | high | fixed |
| Repeated measures on 15 mice were tested as 168 independent rows. | high | partly fixed |
| Groups followed for different lengths were compared on pooled means, and the time column differing by group was reported as a finding (survivorship artefacts). | high | partly fixed |
| A numeric elapsed-time column could not serve as the time axis. | medium | fixed |
| The Pearson/Spearman gap was blamed on ties at timept 0 rather than the skewed tail of x. | medium | not fixed |
| Doubled words with a column literally named 'group', and 'highest of any group' with two groups. | low | not fixed |

## Still open

- The clear result in this data is missed: per mouse, survivors reach a much higher peak x than mice that died (median 59.4 vs 11.1, Mann-Whitney p = 0.0015), but the brief only reports a shared-stretch mean difference it calls not clear.
- The profile card says x 'rises along timept' and peaks on day 12, and the relationship card leads with timept (r = 0.51); both are produced by survivors being the only mice measured after day 10.
- Asked answers by group still treat 168 visits as independent and pool unequal follow-up (Live 12.04 vs Dead 4.99), unlike the brief.
- The explanation of the Pearson/Spearman gap is still wrong (ties at day 0 instead of the skewed tail).
- 'y trend' returns the overall mean of y instead of a trend; 'group groups' wording remains.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "Data.tsv"`. The data is not stored in this repository.
