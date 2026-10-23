# SuperCam calibration dataset 20221218

[erikkepes/supercam-calibration-dataset-20221218](https://www.kaggle.com/datasets/erikkepes/supercam-calibration-dataset-20221218) on Kaggle · sensor · 1,139 rows × 7933 columns · licence: CC0: Public Domain · round 1

1,139 laser-induced breakdown spectra recorded in the lab on calibration targets for the Mars 2020 SuperCam instrument. Each row is one spectrum, named by a .fits file whose name contains the target (325 targets, 1 to 13 spectra each), and the 7,932 numbered columns are wavelength channels from 243.8 to 852.8 nm (wavelengths are in a separate file).

**First run:** poor. Correct arithmetic on one arbitrary channel out of 7,932, a pairwise scan that took about 4 minutes and found only that neighbouring channels agree, and nothing a spectroscopist could use.

**Now:** poor. The answers are now right and name spectra, but the headline reports a meaningless 'repeating pattern' across the spectrum, the cards add little a spectroscopist could use, and the brief takes about 46 seconds.

## What ADA finds now

ADA recognises that each row is a spectrum with 7,932 wavelength channels, uses the file name as the row label, and answers questions correctly, including which spectra are brightest at a given channel and which spectra have unusually high overall intensity. It correctly finds that the average spectrum peaks at channel 703 (279.6 nm, the magnesium line) and names the most variable spectra. But its headline claims every spectrum follows a 'repeating pattern' spanning the whole wavelength range, which is a statistical artefact with no physical meaning. It does not group spectra by the calibration target named in each file, and it takes about 46 seconds to produce the summary.

Headline: “The columns 1 to 7932 form a series; 1,139 of 1,139 rows follow a repeating pattern across it with a period of about 7.93e+03.”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| The spectral matrix was analysed column by column, and one arbitrary channel (1001) was headlined. | high | partly fixed |
| 269 s run time, almost all of it an all-pairs correlation scan over 7,932 columns. | high | partly fixed |
| The 'strongest relationship' was mined from about 31 million channel pairs and only shows that neighbouring channels agree. | medium | partly fixed |
| No whole-spectrum outlier detection (saturated, misfired or very weak spectra). | medium | fixed |
| Numeric column names could not be used in questions, and 'which file has the highest' returned only the value. | medium | fixed |
| The unique file name was not used as the row label, and the target name inside it was not offered as a grouping. | low | partly fixed |
| Intensities were shown in 'B' and 'T'. | low | fixed |

## Still open

- The headline claims every spectrum 'follows a repeating pattern' with a period equal to the whole wavelength range; that is one broad sine fitted to a spectrum, explaining a median 5% of its variation, and means nothing physically.
- A user waits about 46 seconds for the brief on this 1,139 x 7,932 table.
- The relationship card reports two neighbouring emission-line channels as the strongest pair, which carries no information, and its calculation line includes a false claim that channel 1514 is capped at channel 1513.
- The target name embedded in each file name is not used, so the natural grouping (325 calibration targets) is never compared.
- Channel numbers are never mapped to wavelengths, so 'highest at 703' has to be translated by hand (279.6 nm).

## Reproduce

Download the dataset from Kaggle and run `ada analyze "data.csv"`. The data is not stored in this repository.
