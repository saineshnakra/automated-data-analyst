# Cancer Statistics in US States

[nancyalaswad90/cancer-statistics-in-us-states](https://www.kaggle.com/datasets/nancyalaswad90/cancer-statistics-in-us-states) on Kaggle · biology · 272 rows × 1570 columns · licence: CC0: Public Domain · round 3

Despite the title, this is the NKI breast-cancer cohort: 272 patients with follow-up time (survival, in years), a death indicator (eventdeath), time to recurrence, treatment flags, tumour features (grade, nodes, size, angioinvasion) and log-ratio expression of about 1,550 genes clipped at -2 and +2.

**First run:** poor. The headline is circular and the cohort's main question, who survives longer, is never analysed as survival data, though the grade death-rate card is sound.

**Now:** partly useful. The brief now leads with a correct survival analysis by tumour grade, but plain questions about survival still fall back to comparing follow-up times, contradicting it, and the chosen measure is an arbitrary gene.

## What ADA finds now

Despite its title, this is a breast-cancer cohort of 272 patients with follow-up, deaths, tumour features and about 1,550 gene-expression values. ADA now treats follow-up time as censored survival data and leads with the main clinical result: survival falls sharply with tumour grade (96.9%, 72.9% and 50.5% alive at 10 years for grades 1 to 3, with Cox hazard ratios of 4.3 and 9.3), figures that check out. It also reports death rates by grade correctly and recognises that the gene values are clipped at -2 and +2. But asked in plain words whether survival differs by grade, or for the median survival, it still compares raw follow-up times, which gives a weaker and misleading answer, and it takes about 16 seconds on this very wide table.

Headline: “Survival differs by grade (log-rank p < 0.001): at survival = 10, 96.9% for grade 1, 72.9% for grade 2 and 50.5% for grade 3.”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| Follow-up time and its event indicator were treated as a measure and a grouping, so the headline compared follow-up between those who died and those who did not (true by construction). | high | partly fixed |
| No survival analysis for a survival cohort. | high | partly fixed |
| A censored recurrence time equal to follow-up time was reported as a possible copying error. | medium | fixed |
| Group comparisons led with treatment allocation (posnodes by chemo) and reported over a thousand 'clear findings', mostly genes against treatment flags. | medium | partly fixed |
| Platform clipping at -2 and +2 was flagged as entry or unit errors in 155 issues. | medium | fixed |
| Rate questions about a 0/1 outcome failed or were misread, and a 'higher in A than B' question got a stray filter. | medium | fixed |
| About 14 s to analyse 272 rows because of 1,570 columns. | medium | not fixed |
| The array barcode was treated as a reading. | low | fixed |
| A rare histological type code (histtype 7) was called a survey 'refused or don't know' code. | low | fixed |

## Still open

- Questions about survival still ignore censoring: 'average survival by eventdeath' gives the circular 9.40 vs 4.73 (p < 0.001), 'does survival differ by grade' runs an ANOVA on follow-up time (p = 0.012) instead of the log-rank test in the brief, and 'median survival' gives 7.36 where the brief says the median is not reached.
- The measure is a single gene, Contig57584_RC, chosen as the one that best separates deaths from survivors; its card reports p < 0.001 without saying it was picked from about 1,550 genes (it would still survive that: Welch p = 2.6e-10), and comparing it by eventdeath ignores follow-up time.
- The 'What differs between groups' card reports 2,116 clear gene-by-grouping differences, which a reader cannot use.
- A run takes about 16 seconds on 272 rows.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "NKI_cleaned.csv"`. The data is not stored in this repository.
