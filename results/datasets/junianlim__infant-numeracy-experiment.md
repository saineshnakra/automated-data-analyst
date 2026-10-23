# Infant Numeracy Experiment

[junianlim/infant-numeracy-experiment](https://www.kaggle.com/datasets/junianlim/infant-numeracy-experiment) on Kaggle · behavioural · 30 rows × 24 columns · licence: CC0: Public Domain · round 1

A thesis experiment with 30 infants in two conditions (13 and 17 infants), recording looking times on four 'correct' and four 'incorrect' counting outcomes (capped at 22 s), their totals and difference, engagement and exit times, and button presses. The question is whether looking differs between the conditions.

**First run:** poor. The headline is a meaningless sum, the top correlation is circular, and the actual experiment (Condition 1 vs 2) is never compared.

**Now:** good. It compares the two conditions with proper tests and correctly reports no clear difference, drops circular correlations, and flags the 22-second cap and the blocks of missing measures; only the paradigm's preference score is under-used.

## What ADA finds now

ADA compares the two experimental conditions on infants' looking times and correctly reports that none of the 21 measures differs clearly once the number of tests is allowed for, giving each comparison an interval, p-value and effect size. It warns that 52 trial looking times sit at the 22-second cap and that engagement and button measures are missing for whole blocks of infants. It no longer reports correlations between a total and its own parts. It does not lead with the correct-minus-incorrect score that this kind of experiment is built around, and it cannot test whether that score differs from zero.

Headline: “CorrectTotal shows no clear difference between Condition 2 and Condition 1: mean 69.65 vs 58.50 (n = 17 and 13).”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| A per-infant score with 'Total' in its name was summed across infants, giving a meaningless grand total as the headline. | high | fixed |
| Condition (coded 1/2) was treated as a number, so the outcome was correlated with the condition code. | high | fixed |
| The strongest relationship was a part-whole correlation (Correct3 vs CorrectTotal). | high | fixed |
| No comparison between the two conditions, the purpose of the experiment. | high | fixed |
| Missing-data note gave only a total, not which blocks of measures were missing. | medium | fixed |
| The 22-second ceiling on looking times was not flagged. | low | fixed |
| Filler sentence in the headline and K abbreviation for seconds. | low | fixed |
| Group comparison, group counts and missing-data questions refused. | medium | partly fixed |

## Still open

- The paradigm's key score, CorIncor (correct minus incorrect looking), is not the lead outcome, and 'is CorIncor different from zero' is refused; the answer is mean -1.58, one-sample p = 0.34.
- 'which columns have missing values' is refused, though the data checks already hold the answer.
- The top relationship (CorrectTotal vs IncorrectTotal, r = 0.86) is caveated as partly a by-product of totals, but it is still the lead correlation and says little about the experiment.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "Experiment Data.csv"`. The data is not stored in this repository.
