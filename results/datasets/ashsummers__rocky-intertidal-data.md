# Rocky Intertidal Data

[ashsummers/rocky-intertidal-data](https://www.kaggle.com/datasets/ashsummers/rocky-intertidal-data) on Kaggle · environmental · 120 rows × 7 columns · licence: CC0: Public Domain · round 2

A student field survey of a rocky shore: counts of five large mobile invertebrates (nudibranchs, sea stars, crabs, sea cucumbers, octopuses) in 120 quadrats, 12 at each of 10 distances (0-45 m) along a transect, split left and right. Ten quadrats at the seaward end have no counts.

**First run:** partly useful. The numbers are right and the zero-heavy note is good, but the headline is an n = 2 artefact of counts misread as flags, and the obvious ecological summaries (abundance by taxon and by distance) are missing.

**Now:** partly useful. What it says about crabs (no clear difference by distance or side, 83% zeros, the empty cucumber column) is correct and honest, but it never summarises the five taxa side by side and still treats the rare ones as yes/no codes when asked.

## What ADA finds now

This file counts five kinds of large invertebrates in 120 quadrats along a rocky-shore transect. The analysis focuses on crabs, the only common animal, and correctly reports that their numbers do not differ clearly by distance or by side once the many empty quadrats are allowed for; it compares the share of quadrats with crabs as well as the averages. It notes that sea cucumbers were never recorded and that 10 quadrats have no counts. It does not give the basic table of all five animals side by side, cannot say which is most abundant, and treats the rarest animals as yes/no columns.

Headline: “Crabs shows no clear difference across the 10 Location_m groups (Kruskal-Wallis, p = 0.66; ε² = 0.06): means run from 0 (Location_m 45, n = 6) to 6.67 (Location_m 5, n = 12).”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| The headline was a 'clear difference' between a group of 2 quadrats and a group of 108. | high | fixed |
| Count columns with a maximum of 1 were read as 0/1 flags and used as groupings. | high | partly fixed |
| An all-zero column (Cucumbers) was offered as a measure and not flagged. | medium | fixed |
| Group summaries by median were empty because most values are zero. | medium | fixed |
| No trend test for an ordered numeric grouping. | medium | fixed |
| No summary across the taxa. | medium | not fixed |
| The missing-together check did not say that the gap follows a group. | low | not fixed |
| Answers said 'Based on all 120 rows' when only 110 have the measure. | low | partly fixed |

## Still open

- There is no summary across the five taxa (crabs 220, octopuses 3, nudibranchs 2, sea stars 2, cucumbers 0), the first table an ecologist builds, and 'which species is most abundant' and 'which column has the highest total' are refused, the second with the false reason that there are fewer than two readings.
- Rare taxa whose counts happen to be only 0 or 1 are still handled in Ask as yes/no codes ('Nudibranchs holds category codes'), so they cannot be totalled against each other or correlated.
- The heavy-tail card still advises that 'medians describe it better than means', while the headline explains that every median is 0 and compares means and non-zero shares instead.
- The missing-data note blames 'a subsample, a separate test or an instrument' for 10 empty quadrats that are all at 35-45 m, the seaward end of the transect.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "Large_Mobile_Invertebrates.csv"`. The data is not stored in this repository.
