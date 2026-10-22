# Soil Compositions

[ramakrishnanthiyagu/soil-compositions](https://www.kaggle.com/datasets/ramakrishnanthiyagu/soil-compositions) on Kaggle · chemistry · 48 rows × 14 columns · licence: Unknown · round 1

48 soil samples from a randomised block design: 3 contour positions x 4 depths x 4 blocks, each with pH, nitrogen, bulk density, P, Ca, Mg, K, Na and conductivity. Group (1 to 12) and Gp (T0 to D6) both code the contour-by-depth cell, and the 4 rows in each cell are the 4 block replicates.

**First run:** partly useful. The headline and the Na-conductivity link are correct and meaningful, but ADA misses that depth drives almost every property, correlates with a cell code, and doesn't notice the Excel-mangled depth label.

**Now:** partly useful. The headline correlation, the depth means and the repaired '10-30' label are right, but the report now says no depth or contour difference is clear when depth drives almost every soil property, because it treats each design cell as a single repeated subject.

## What ADA finds now

ADA repairs the depth label that a spreadsheet had turned into a date, reads the cell and block codes as categories, and reports that calcium and nitrogen rise and fall together, noting that the link is weaker within each depth. It also says depth accounts for over 90% of the variation in calcium between cells, while contour and block account for little. But it then states that none of 27 comparisons shows a clear difference, and answers that calcium does not clearly differ by depth, because it averages the four replicate samples in each cell into one value before testing. On the 48 samples the depth differences are overwhelming for almost every property, so a reader could come away with the opposite of the study's main result.

Headline: “Ca and N move together: Pearson r = 0.85 (95% CI 0.75 to 0.91, n = 48).”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| A depth range turned into a date by a spreadsheet ('10-30' read as 'Oct-30') was kept without warning and sorted out of order. | medium | fixed |
| The crossed design was reduced to one combined code, and ADA never said which factor drives the variation or gave effect sizes. | high | partly fixed |
| Nine chemistry outcomes, but only Ca was reported. | medium | partly fixed |
| The numbered cell code Group was treated as a quantity and correlated with outcomes. | medium | fixed |
| Chemical symbols and abbreviations lower-cased in answers. | low | partly fixed |
| Full words not matched to abbreviated column names; 'does X differ by Y' refused. | low | fixed |

## Still open

- The differences card says 'None of the 27 comparisons across Contour, Depth and Block shows a clear difference', and 'does Ca differ by Depth' says the groups are 'too few to call the difference clear'. Depth in fact separates nearly every property (Ca p = 1e-13, conductivity p = 3e-22 over 48 samples); ADA collapses each contour-by-depth cell to one mean, treating the cell code as if it were a repeatedly measured subject, which leaves 3 values per depth.
- The relationships card says the rows 'are not independent and the intervals are narrower than the evidence' because several come from each Group, but the rows within a Group are the four independent block replicates of that treatment cell.
- The 'Highest average by Gp' card says 'Too few groups have two or more values to test whether the gap is larger than the noise', though every Gp has 4 values.
- Variance shares and group effect sizes in answers are computed on cell means while the tables show 12 rows per depth (Na by Depth eta squared 0.96 shown; 0.88 on the 48 samples), so the numbers do not match the table beside them.
- 'is Conduc higher at 60-90 than 0-10' is refused because the depth labels are not recognised as values, and 'Conduc' is still lower-cased in answers.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "Soils.csv"`. The data is not stored in this repository.
