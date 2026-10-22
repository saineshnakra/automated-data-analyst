# Social Learning in House Sparrows

[thedevastator/social-learning-in-house-sparrows](https://www.kaggle.com/datasets/thedevastator/social-learning-in-house-sparrows) on Kaggle · behavioural · 351 rows × 7 columns · licence: CC0: Public Domain · round 1

A repeated-measures behaviour experiment: 24 captive house sparrows (8 in each of three phenotype groups) were each tested about 15 times over 3 weeks, and the time each took to approach a food dish (TIME, seconds, capped at 3,600) was recorded with no object (OBJECT 0) or one of nine novel objects beside it. STATUS = 0 marks the 13 trials where the bird never approached.

**First run:** poor. The headline averages an object code per bird; the actual outcome (approach latency), the phenotype comparison and the strong novel-object effect were all missed.

**Now:** good. It analyses approach time per bird, tests phenotype and object effects with the birds as the unit and the right tests, and the answers check out; the quibbles are a fragile 'slowest phenotype' ranking and some confusing wording about the cap flag.

## What ADA finds now

ADA takes approach time as the outcome, treats phenotype, object and week as groups, and correctly reduces each bird to one value before comparing groups, so the 24 birds rather than 351 trials are the sample. It finds that phenotype explains about a fifth of the variation in birds' typical approach times and that novel objects slow birds sharply (a median of 725 seconds with the most off-putting object vs 18 seconds with none). It handles repeated measures properly when asked about weeks and gives correct answers to a wide range of questions. Its headline names phenotype 2 as the slowest, but that ranking depends on one slow bird and the pairwise tests cannot separate the groups. The 3,600-second cap on approach time is only mentioned inside answers.

Headline: “TIME differs across PHENOTYPE: PHENOTYPE 2 has the highest mean of per-ID medians, 216.12 (n = 8), and PHENOTYPE 0 the lowest, 46.31 (n = 8); η² = 0.21.”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| Integer codes were treated as quantities and the object code (OBJECT) was picked as the measure by name order. | high | fixed |
| No test or effect size for group differences, and 'does TIME differ between phenotypes' was refused. | high | fixed |
| Skewed latencies were summarised by means, and outliers were flagged on the raw scale. | medium | partly fixed |
| The censoring flag STATUS paired with its own capped TIME was shown as the strongest relationship. | medium | fixed |
| The ceiling of 3,600 s (13 rows) was not noticed. | medium | partly fixed |
| Ties for highest and lowest were not mentioned. | low | fixed |
| Filters on numeric codes and 'longest' were not supported. | low | fixed |

## Still open

- The headline names phenotype 2 as the slowest, but that rests on one slow bird: by the median of the birds' medians phenotype 1 is slowest, and no pair of phenotypes is separable on its own.
- The headline card says STATUS 'sets the level of TIME (100% of its variance)', which is true only because STATUS marks the trials capped at 3,600 s.
- The 3,600 s cap is no longer reported in the brief's data checks; it appears only inside answers.
- The simplest novel-object result, that 23 of 24 birds were slower with any novel object than with none, is not stated; the brief compares only the slowest object code with the control.
- Asked about TIME and OBJECT, it says 'TIME rises as OBJECT increases across its 10 levels', a trend over object ID numbers that have no order.
- The trial comparison ('TRIAL 6 highest median, 558') does not say that the same birds appear in every trial, unlike the object and week answers.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "Approach_times_all_phenotypes_all_weeks.csv"`. The data is not stored in this repository.
