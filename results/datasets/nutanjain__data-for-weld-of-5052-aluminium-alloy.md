# Data for weld of 5052 Aluminium Alloy

[nutanjain/data-for-weld-of-5052-aluminium-alloy](https://www.kaggle.com/datasets/nutanjain/data-for-weld-of-5052-aluminium-alloy) on Kaggle · materials · 31 rows × 8 columns · licence: Unknown · round 2

A designed friction-stir-welding experiment on 5052 aluminium: 30 welds, each with four set inputs (pin diameter, shoulder diameter, rotation speed, welding speed, five levels each) and three measured outputs (tensile strength, elongation and Charpy impact strength). The sheet has a two-row header, a group label above the real column names with units.

**First run:** partly useful. The numbers are right and the warnings about weak correlations are honest, but the lost header makes the output hard to read, the headline is a false positive from Welch's test on a group of 2, and the real effects of welding speed and pin diameter are missed.

**Now:** partly useful. Everything it says is right and the false positive is gone, but for a designed experiment the brief is a single correlation card with no model of each output on the four settings, so most of what the experiment shows is left for the user to ask.

## What ADA finds now

This file is a 30-weld experiment that varies four welding settings and measures three strength outputs. The analysis now reads the two-row header correctly and reports that a larger pin diameter goes with higher impact strength (r = 0.50) and that faster welding goes with lower elongation (r = -0.57); both are real and hold up in a full regression. Asked directly, it compares groups sensibly and declines to rank groups of one or two welds. It does not fit the standard model of each output on all four settings, so it misses that rotation and welding speed also affect impact strength and that tensile strength is not explained by any setting. It also does not separate the settings from the outputs, so another output can top the list of what 'affects' a result.

Headline: “Charpy Impact Strength (kg-m) and Pin Dia (mm) move together: Pearson r = 0.50 (95% CI 0.17 to 0.73, n = 30).”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| The two-row header was not detected; the real column names became a row of data, giving 31 rows, 'Unnamed' columns and false missing cells. | high | fixed |
| Welch's ANOVA on a group of 2 produced a false 'clear difference' (impact strength by shoulder diameter) that became the headline. | high | fixed |
| Ordered numeric design inputs were treated as unordered categories and left out of the correlation scan; no regression of each output on all inputs. | high | partly fixed |
| Design inputs with the same structure were classified differently (some readings, some categories). | medium | fixed |
| 'what affects X' was refused. | medium | fixed |
| Column names lower-cased mid-sentence and awkward numeric group labels. | low | partly fixed |

## Still open

- There is no joint model of each output on the four set inputs, the standard first look at a designed experiment; the brief shows one correlation card, so it misses that rotation and welding speed also matter for impact strength and that tensile strength is not explained by any input (R^2 = 0.15).
- Set inputs and measured outputs are not told apart: 'what affects UTS (MPa)' ranks % Elongation, another output, first (r = 0.29, interval includes zero), and the impact-strength ranking lists elongation beside the settings.
- Column names are still lower-cased mid-sentence ('average charpy impact strength (kg-m)') and the summary reads 'one per Weld No..'.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "Data for weld of 5052 Aluminioum alloy.xlsx" --sheet "For ChatGPT"`. The data is not stored in this repository.
