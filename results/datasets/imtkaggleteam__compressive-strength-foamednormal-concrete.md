# Compressive strength foamed/normal concrete

[imtkaggleteam/compressive-strength-foamednormal-concrete](https://www.kaggle.com/datasets/imtkaggleteam/compressive-strength-foamednormal-concrete) on Kaggle · materials · 1,133 rows × 9 columns · licence: Other (specified in description) · round 1

1,133 concrete test specimens, each with its mix recipe (cement, slag, fly ash, water, superplasticiser, coarse and fine aggregate, in kg/m3), the curing age at testing (1 to 365 days) and the measured compressive strength in MPa, which is the outcome.

**First run:** poor. The arithmetic is correct, but it summarises an ingredient instead of strength and reports a mix-design artefact as the main relationship, so the study's actual question (what drives strength) goes unanswered.

**Now:** good. It takes strength as the outcome, shows the gain with curing age with a warning that mixes differ between ages, ranks the drivers of strength correctly and answers every question tried; the main gap is a joint model that leaves out age.

## What ADA finds now

ADA treats compressive strength as the outcome and curing age as the grouping, and correctly reports that strength rises with age (about 19 MPa at 3 days, 37 at 28 days and 52 at 56 days), with the caution that different mixes were tested at different ages. It ranks the ingredients by how strongly they track strength, led by cement (r = 0.49, rising to 0.63 within a curing age), and every answer checked matched an independent calculation. Its combined model of the mix ingredients explains 43% of the variation in strength but leaves out curing age, which would roughly double that. It also does not consider the water-to-binder ratio that engineers would look at first (r = -0.62 with strength).

Headline: “Concrete compressive strength (MPa) differs across Age of testing (day): Age of testing (day) 91 has the highest mean, 69.81 (n = 22), and Age of testing (day) 3 the lowest, 18.98 (n = 134); η² = 0.40.”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| An ingredient (blast-furnace slag) was picked as the measure by name order instead of strength. | high | fixed |
| There was no 'what relates to the measure' finding; the top pair was two inputs (water vs superplasticiser). | high | fixed |
| The summary was filler text. | medium | fixed |
| Curing age was not offered as a grouping, so the strength gain with age was missed. | medium | fixed |
| Short names and value filters were refused. | low | fixed |

## Still open

- The joint model of strength leaves out curing age, the factor that explains 40% on its own: six mix inputs give R² = 0.43, while adding log age and fly ash raises it to about 0.81.
- The water-to-binder ratio, the classic driver of strength (r about -0.62, stronger than any single input), is not considered; only single columns are correlated.
- The relationship card still appends water vs superplasticiser (r = -0.59), which reflects how the mixes were designed rather than a finding.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "Dataset2.xlsx" --sheet "Data"`. The data is not stored in this repository.
