# Food Dyes RGB data for Digital Image Colorimetry

[yozeftjandra/food-dyes-rgb-data-for-digital-image-colorimetry](https://www.kaggle.com/datasets/yozeftjandra/food-dyes-rgb-data-for-digital-image-colorimetry) on Kaggle · chemistry · 1,228 rows × 6 columns · licence: CC BY-NC-SA 4.0 · round 1

1,228 photographs of nine food-dye solutions, each made up at 4 or 5 set concentrations with about 15 replicate images per level and lighting condition (Open, Closed). The readings are the R, G and B values of each image; concentration is the design variable a colorimetry model would predict.

**First run:** poor. The arithmetic is right, but the headline is about the design (which concentrations were prepared) and the big lighting effect and the per-dye calibration relationships go unmentioned.

**Now:** partly useful. The lighting effect, the dye differences and the within-dye correlation are right and well tested, but the calibration relationships the data exists for are missing from the brief and the pooled concentration answer points the wrong way.

## What ADA finds now

ADA reads R, G and B as the readings and treats concentration as a set experimental level. It correctly reports the large effect of lighting (green reads 188 in closed lighting vs 109 in open, a very large difference) and the expected differences between dyes, with proper tests and effect sizes, and notes that colour channels track each other much more closely within a dye than across dyes. Asked about a single dye, it gives clean concentration-response curves, such as blue falling from 150 to 25 across four Sunset Yellow concentrations. But the summary never shows these per-dye curves, and a question about concentration across all dyes gets a misleading answer, because the lowest concentrations were only made for the blue and green dyes. Its outlier card also flags ordinary readings of one dark red dye.

Headline: “B differs across Dye Label: Brilliant blue FCF, FD&C Blue 1 (US), E133 (EU) has the highest mean, 175.55 (n = 120), and Sunset Yellow, FD&C Yellow 6 (US), E110 (EU) the lowest, 55.68 (n = 120); η² = 0.42.”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| The set concentration (a design variable) was chosen as the measure, so the headline reported which concentrations were prepared for each dye. | high | fixed |
| The largest effect, lighting condition, was never mentioned and no group difference was tested. | high | fixed |
| A filter given as the first word of a long dye label ('for Tartrazine') was silently ignored. | high | fixed |
| Pooled correlations across dyes hide the within-dye calibration relationships a colorimetry researcher needs. | medium | partly fixed |
| 'does B change with Concentration' was answered with the mean of Concentration. | medium | fixed |
| Business wording ('Leading dye label') for a lab experiment. | low | fixed |

## Still open

- Asked about concentration across all dyes, it reports that R rises with concentration (ρ = 0.26), the opposite of what happens within each blue or green dye (ρ about -0.7), without warning that the low concentrations exist only for those dyes.
- The brief never shows the per-dye concentration-response (calibration) relationships that are the purpose of the data; they appear only when the user asks for one dye at a time (e.g. Sunset Yellow under closed lighting, B falls from 149.6 to 25.2 across four levels, ρ = -0.97).
- The 'exceptional records' card flags 24 G values that are all one red dye (Carmoisine) at its two highest concentrations in open lighting; they are flagged only on a log scale, and none falls outside the fences on the raw scale.
- The headline (blue dyes have the most blue, yellow dyes the least) is correct but is what the dye colours predict rather than a finding.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "Colorimeter Dataset - kaggle (1).csv"`. The data is not stored in this repository.
