# Self compacting concrete dataset

[gaming919/self-compacting-concrete-dataset](https://www.kaggle.com/datasets/gaming919/self-compacting-concrete-dataset) on Kaggle · materials · 555 rows × 8 columns · licence: Unknown · round 3

555 self-compacting concrete mixes from published studies: amounts of cement, sand, gravel, mineral admixture, superplasticiser, water and viscosity-modifying agent (kg/m3), with measured 28-day compressive strength (10 to 122 MPa).

**First run:** partly useful. Everything it reports is correct, but it stops at one weak correlation and leaves out the joint model and the water-to-cement ratio that a concrete researcher would look at first.

**Now:** partly useful. Every figure is right, but it stops at one weak correlation, and its new model card wrongly sets aside water and gravel as outcomes and so understates what the mix explains.

## What ADA finds now

This table lists 555 self-compacting concrete mixes with their 28-day strength. The tool correctly finds that strength rises only weakly with cement content (r = 0.38) and flags 10 repeated rows. It now adds a combined model, but leaves water, gravel and superplasticiser out of it as if they were outcomes, so it reports that the mix explains 15% of strength when all seven ingredients together explain about 23%. It never tries the water-to-cement ratio, the standard predictor of concrete strength, and does not flag mixes whose strength is implausible for their water content. Simple filtered questions now work however they are phrased.

Headline: “fc' 28 days and Cement (kg/m3 ) move together only weakly: Pearson r = 0.38 (95% CI 0.30 to 0.44, n = 555).”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| When every column but the outcome is a mix input, the joint model is the main result, but it was hidden unless it beat the best single input by 10 points. | medium | partly fixed |
| Ratios of recipe columns, which carry the known physics (water-to-cement), are never tried. | medium | not fixed |
| Physically implausible rows are not flagged. | low | not fixed |
| Outlier counts on design inputs were reported as 'exceptional records'. | low | fixed |
| Numeric filters failed unless phrased with 'where'. | low | fixed |

## Still open

- The joint model leaves out water, gravel and superplasticiser as 'candidate outcomes' although they are mix inputs, so it reports 15% of strength explained instead of 23% and drops water, one of the strongest terms.
- The water-to-cement ratio, the standard predictor of concrete strength (Spearman -0.44 here), is never computed and the question is refused.
- The headline is a single weak correlation with cement (r = 0.38); the more useful message, that the recorded mix explains under a quarter of strength, is not stated.
- Physically doubtful mixes (water-to-cement 2.24 at 48 MPa) are not flagged.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "scc_final_database.csv"`. The data is not stored in this repository.
