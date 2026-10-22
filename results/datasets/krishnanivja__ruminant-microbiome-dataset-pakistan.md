# Ruminant Microbiome Dataset — Pakistan

[krishnanivja/ruminant-microbiome-dataset-pakistan](https://www.kaggle.com/datasets/krishnanivja/ruminant-microbiome-dataset-pakistan) on Kaggle · biology · 10,000 rows × 18 columns · licence: CC0: Public Domain · round 2

10,000 animals (cattle, buffalo and goats) from four Pakistani provinces with breed, age, weight, antibiotic and probiotic use, diet shares that add up to 100, farm size, gut diversity, four phylum shares and a 0-1 dysbiosis risk score. It looks generated: age and weight are flat and unrelated to anything, and goats weigh up to 698 kg.

**First run:** partly useful. The measure and headline are wrong (weight, and a band-vs-acres tautology), but the real result, that antibiotics and probiotics move dysbiosis risk, is in the brief as a secondary line, and Ask's comparison answers are accurate.

**Now:** partly useful. The outcome, headline, group differences, correlations and data checks are right, but the joint-model card wrongly says diet adds little, because it fits three shares that add up to 100 together, and the first card is a null regional comparison.

## What ADA finds now

This file describes the gut microbiome and a dysbiosis risk score for 10,000 farm animals in Pakistan, and it looks generated. ADA picks the risk score as the outcome and leads with the clearest result: animals given antibiotics have a much higher risk score (median 0.81 vs 0.50), and probiotics go with a lower one; it also finds the strong link between risk and gut diversity and warns that age and weight look generated. It recognises that the diet and phylum shares add up to 100 and that the farm-size bands are cut from acreage. However, its combined model wrongly concludes that diet adds little once antibiotics and probiotics are known, when diet in fact accounts for about 30% of the variation, and its first card is spent on regions, which do not differ.

Headline: “Largest difference found: Dysbiosis_Risk is higher in Antibiotics 1 than in Antibiotics 0: median 0.81 vs 0.5 (n = 3,525 and 6,475).”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| The headline compared a band column with the number it was cut from (Farm_Size vs Farm_Size_Acres). | high | fixed |
| The measure was a body-size covariate (Weight_kg) related to nothing. | high | fixed |
| The headline and the measure's first card did not follow the measure; the default grouping explained nothing. | medium | partly fixed |
| Age_months was read as a calendar month and dropped from every scan. | medium | fixed |
| The four phylum shares that sum to 100 were not recognised; broken rows were not flagged and a closure correlation was reported. | medium | fixed |
| Causal-sounding phrasings and flag shares were refused. | medium | fixed |
| Rank and breakdown answers named a top group with no test. | medium | fixed |
| Two-way breakdowns were refused. | low | fixed |

## Still open

- The joint-model card puts all three diet shares, which add up to 100, into one model, gets unstable coefficients (forage p = 0.06, grains and silage p = 0.35) and concludes that diet 'adds little once the others are known'; in fact diet lifts R² from 0.65 (antibiotics and probiotics alone) to 0.94, and each point of forage lowers the risk score by 0.0096 (p < 0.001) once one share is dropped.
- The first card still compares regions, where nothing differs (ε² = 0.00), while antibiotics explain 43% of the risk score.
- The note that the four phylum shares exceed 100 on 866 rows sits in the correlation card rather than the data checks, and does not say these are rows where the 'other' share is 0.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "ruminant_microbiome_pakistan.csv"`. The data is not stored in this repository.
