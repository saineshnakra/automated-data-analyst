# Endangered fishes dataset

[harshithgupta/endangered-fish-data](https://www.kaggle.com/datasets/harshithgupta/endangered-fish-data) on Kaggle · environmental · 42,336 rows × 67 columns · licence: U.S. Government Works · round 3

A fish-occurrence table compiled from 8 US stream surveys (42,336 rows): one row per species found on a site visit, with that visit's water chemistry (conductivity, pH, hardness, ions) repeated on every species row. Conductivity and species are recorded in every survey; most other chemistry and the fish-community metrics exist only in some.

**First run:** poor. It reads the file and correctly spots that it combines several surveys, but the headline and most cards are artifacts of species rows repeating one site reading, a log copy of the measure and fits on 5 distinct visits.

**Now:** partly useful. The brief's basin and ecoregion comparisons are now correct, counting each site visit once, but it covers only one of eight surveys, ignores the species records the dataset exists for, and asked answers still count species rows as readings.

## What ADA finds now

This file combines eight US stream surveys, with one row per fish species found on a site visit and the visit's water chemistry repeated on each row. ADA now sees that conductivity is one reading per visit and counts each visit once: it finds conductivity highest in the Upper Ohio basin (512, 53 visits) and lowest in the Monongahela (170, 195 visits), and that it tracks water hardness closely (r = 0.87), all of which check out. It no longer mistakes a log copy of conductivity for a finding. But it analyses only the Pennsylvania survey, says nothing about which species live at which conductivity unless asked, and some answers to questions still treat each species row as a separate measurement, which makes the uncertainty look far smaller than it is.

Headline: “COND differs across HUCName: Upper Ohio has the highest mean over Station_Date values, 512.09 (n = 53), and Monongahela the lowest, 170.38 (n = 195); η² = 0.08.”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| Visit-level readings repeated on every species row were tested and counted as separate measurements. | high | partly fixed |
| Continuous chemistry readings with few distinct values were used as groupings. | high | partly fixed |
| A joint model and 'fixed combination' claims fitted on 53 rows that were 5 distinct visits (exact fit by construction). | high | fixed |
| lcond (log10 of COND) was not recognised as derived and led the relationship scan. | high | fixed |
| The stacked-table rule keeps only the largest survey, though conductivity and species are recorded in all 8. | medium | not fixed |
| The species-occurrence structure is not used: no per-species or per-visit views. | medium | not fixed |
| The fill-value check fired on one visit's reading repeated across its species rows. | medium | fixed |
| Refusals gave a wrong or missing reason when a column was empty or constant in the chosen survey. | medium | fixed |
| Yearly trends and the forecast track which sites were sampled, not conductivity change. | medium | partly fixed |
| Physically implausible ion values not flagged. | low | not fixed |
| A row-count answer said groups were on different scales and units. | low | partly fixed |

## Still open

- The brief covers only the Pennsylvania survey (17,100 of 42,336 rows), though conductivity and species are recorded in all eight surveys.
- Nothing in the brief uses the species records: no conductivity range per species or species richness per visit, which is what a fish-tolerance dataset is for.
- Asked answers still count each species row as a separate reading (e.g. COND-hardness n = 12,568 with a zero-width interval; the segments table names a different lowest basin than the headline).
- Yearly averages are still reported as a trend without the sampling-plan warning the forecast gives.
- The outlier card groups pH by total phosphorus, and implausible ion values (SO4 up to 6,146 at conductivity near 400) are not flagged.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "Combined_Less.csv"`. The data is not stored in this repository.
