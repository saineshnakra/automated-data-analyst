# FI_Optimun_Frecuency

[davidhguerrero/fi-optimun-frecuency](https://www.kaggle.com/datasets/davidhguerrero/fi-optimun-frecuency) on Kaggle · sensor · 3,108,545 rows × 8 columns · licence: Unknown · round 1

Terahertz transmission sweeps from 100 to 590 GHz in 10 GHz steps on 12 plastic materials (Sample A to O, 4 specimens each), with thousands of repeated readings per specimen and frequency, 3.1 million rows in all. LG and HG are the low- and high-gain detector voltages; the aim is to find the frequencies that best tell the materials apart.

**First run:** poor. Every finding comes from the last 8% of the file, which holds one of 12 materials, and ADA misses the material-by-frequency question the dataset was built for.

**Now:** partly useful. The sampling, the frequency profile and the answers are right, but the brief reads material differences only averaged over the sweep and calls them negligible, missing the frequency-by-material question the data exists for.

## What ADA finds now

ADA now samples the 3.1 million rows fairly and shows how the high-gain signal depends on frequency, peaking at 230 GHz and dropping to near zero across 110 to 190 GHz; frequency explains 70% of the signal. It treats the row number as an identifier, measures outliers within each frequency, and answers filtered and grouped questions correctly, including the full material-by-frequency table when asked. But the summary only compares materials averaged over the whole sweep, where they look almost the same, and never shows that at single frequencies they differ sharply (at 280 GHz only one plastic transmits). It also describes the spectrum as rising with frequency when it rises and then falls, and a direct question about which frequency best separates the materials is misread.

Headline: “HG (mV) differs across Frequency (GHz): Frequency (GHz) 230 has the highest mean, 703.55 (n = 20,380), and Frequency (GHz) 130 the lowest, -0.0172 (n = 17,468); η² = 0.70.”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| The row limit kept the last 250,000 rows in file order (one material of 12) and called the dropped rows 'older'. | high | fixed |
| Filters and groupings absent from the analysed rows were silently dropped. | high | fixed |
| The material x frequency design was not recognised, so the question of which frequency separates the materials was missed. | high | partly fixed |
| 41 s to analyse 3.1 million rows. | high | fixed |
| IQR outliers flagged the normal high readings at resonant frequencies. | medium | fixed |
| The monotonic row number index_original was not recognised as an identifier. | medium | fixed |
| Business wording ('Leading sample_original'). | low | fixed |

## Still open

- It never compares the materials frequency by frequency, which is what the data was collected for, and it calls the material differences negligible (1% of variance) when, at a given frequency, material explains a median 62% of the signal.
- The question 'which frequency best separates the samples' is misread as comparing the frequency values across samples and answered with a meaningless result.
- The headline says HG 'rises as Frequency (GHz) increases' (Spearman 0.36), but the profile is not monotonic: near zero at 110 to 190 GHz, alternating peaks and zeros from 200 to 300 GHz, then falling from about 500 mV at 340 GHz to 24 mV at 590 GHz.
- The data check lists the 37% of HG readings at the detector floor as an issue to look at, when it is the expected reading at blocked frequencies.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "train_day_1_to_4_e5.csv"`. The data is not stored in this repository.
