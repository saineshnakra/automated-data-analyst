# Landslide dataset

[rajumavinmar/landslide-dataset](https://www.kaggle.com/datasets/rajumavinmar/landslide-dataset) on Kaggle · environmental · 2,000 rows × 10 columns · licence: Apache 2.0 · round 3

2,000 generated rows of rainfall, slope, soil saturation, vegetation cover, earthquake activity, distance to water and soil type (one-hot columns for three of four types), with a 0/1 landslide label split exactly 1,000/1,000.

**First run:** poor. It spots the exact cut points that define a landslide, but leads with a confounded comparison and a set of correlations that exist only because two artificial classes are mixed, and never says the data are simulated.

**Now:** partly useful. The headline, the cut points and the warning that the table looks simulated are right, but the second card leads with soil-type effects that are only the landslide split in disguise.

## What ADA finds now

This is a generated table of 2,000 sites with rainfall, slope, soil and other readings and a yes/no landslide label. The tool now leads with the right comparison (slopes are steeper at landslide sites, 42 against 18 degrees), finds the exact cut points in saturation, vegetation, earthquake activity and distance to water that define a landslide, and warns that each group looks drawn from flat, unrelated ranges, as simulated data are. It rebuilds the soil type from its separate yes/no columns and explains that the strong correlations across the whole table come only from mixing the two groups. Its second card, though, leads with differences between soil types that exist only because each soil type falls entirely in one landslide group, and it never says outright that soil type alone decides whether a site had a landslide.

Headline: “Slope_Angle is higher in Landslide 1 than in Landslide 0: mean 42.25 vs 17.72 (n = 1,000 and 1,000).”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| Pooled correlations and the joint model were reported although they exist only because two classes at different levels are mixed. | high | fixed |
| A predictor that defines the label (earthquake activity) was made the measure, and the label was only one of several groupings. | high | fixed |
| The headline compared groups (gravel soil or not) that differ in the outcome label, so the gap reflected the label. | high | partly fixed |
| One-hot soil columns were treated as unrelated 0/1 flags, so soil type was never one grouping and the all-zero level was invisible. | medium | fixed |
| The simulated-data check missed a table where each class is drawn independently from flat ranges. | medium | fixed |
| 'Share of <flag> by <group>' was answered as a count of distinct values, and 'what affects Landslide' did not reuse the cut points. | medium | partly fixed |

## Still open

- The second card leads with soil-type differences in saturation, vegetation and distance to water (eta-squared about 0.75) that exist only because each soil type falls wholly in one landslide class; within a class they vanish.
- The report never says that soil type alone decides landslide (two types always, two never); this comes out only when asked.
- 'what affects Landslide' gives correlations as 'leads to check' instead of the exact cut points the report itself found, and a direct correlation question gives the pooled r = 0.65 without the within-class caveat.
- The headline card says the landslide groups 'were not given the same inputs', wording meant for treatments that reads oddly for an outcome.
- 'is this data simulated' is refused, though the data checks say it is.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "landslide_dataset.csv"`. The data is not stored in this repository.
