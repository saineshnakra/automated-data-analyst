# Soil Analysis Data at Colombia

[pablorodrguezn/soil-analysis-data-at-colombia](https://www.kaggle.com/datasets/pablorodrguezn/soil-analysis-data-at-colombia) on Kaggle · chemistry · 46,745 rows × 33 columns · licence: Unknown · round 1

46,745 farm soil samples analysed by a Colombian soil laboratory, each with department, municipality, crop and site details plus about 21 lab results (pH, organic matter, phosphorus, exchangeable acidity and cations, conductivity, micronutrients). Values below a detection limit are written as '<x,xx' and values not measured as 'ND'.

**First run:** partly useful. The reading is careful and the cleaning notes are transparent, but the headline is about a 'not stated' label, crop and region comparisons are missing, and below-detection values are dropped, biasing means upward.

**Now:** partly useful. The reading, data checks and department comparisons are strong and the pH answers are right, but the joint model rests on 2% of the samples and the crop ranking is led by single-sample crops.

## What ADA finds now

ADA handles this messy lab file well: it reads 'not stated' labels and spelling variants as missing, reports how many values in each column were below the detection limit and warns that this pushes means up, notices that acidity was only measured on acidic soils, and recognises that one column is the sum of five others. Its headline, that cation exchange capacity differs strongly between departments (median 24.6 in Sucre against 1.3 in Vichada), is correct and relevant, and pH by department is right (highest in La Guajira, 7.52). Its multi-variable model is fitted on only 2% of the samples without saying how narrow that subset is, and crop rankings are led by crops with a single sample.

Headline: “capacidad de intercambio cationico (CICE) suma de bases cmol(+)/kg differs across Departamento: SUCRE has the highest median, 24.59 (n = 431), and VICHADA the lowest, 1.34 (n = 99); ε² = 0.23.”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| 'Not stated' labels and spelling variants were kept as categories, and the headline compared two of them. | high | fixed |
| Below-detection values were turned into missing, biasing means upward. | high | partly fixed |
| A grouping was silently dropped from an answer ('median organic matter by Cultivo' gave the overall median). | high | fixed |
| The grouping was chosen by 'number of levels closest to 10', so crop and region were ignored. | medium | fixed |
| Short column names such as 'pH' were refused. | medium | fixed |
| Structural missingness (acidity and Al measured only on acidic soils) was not reported. | medium | fixed |
| The strongest relationship and the outlier count were dominated by expected chemistry and skew, and the CICE sum was not detected. | medium | partly fixed |
| Physically impossible values were not flagged. | low | fixed |
| Counts with a threshold or category filter were refused. | low | fixed |

## Still open

- The joint-model card says six variables explain 30% of CICE and that pH 'adds little', but it is fit on 972 samples (2%), mostly from Meta, that had the double-acid micronutrient test; over all samples pH is the strongest correlate of CICE (r = 0.60).
- Below-detection values are still dropped and means still used, so means of phosphorus, sodium, zinc and copper run high; the iron double-acid 'ND' block is mislabelled as below detection.
- The crop ranking's sentence names a crop with 10 samples while its table is led by crops with a single sample.
- The chosen measure, CICE, is itself the sum of five other columns, and the relationship card still ends with the expected acidity vs aluminium pair.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "Resultados_de_An_lisis_de_Laboratorio_Suelos_en_Colombia.csv"`. The data is not stored in this repository.
