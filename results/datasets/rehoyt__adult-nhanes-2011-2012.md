# Adult NHANES 2011-2012

[rehoyt/adult-nhanes-2011-2012](https://www.kaggle.com/datasets/rehoyt/adult-nhanes-2011-2012) on Kaggle · clinical · 4,571 rows × 73 columns · licence: CC BY-SA 4.0 · round 1

4,571 US adults from the 2011-12 National Health and Nutrition Examination Survey, one row per person: integer-coded demographics, 0/1 medical-history flags, body measures, about 35 lab values, spirometry, accelerometer activity and a cognitive score. The last column is a second diabetes flag derived from HbA1c.

**First run:** poor. The headline sums income-band codes as dollars, no groupings were found because every category is integer-coded, and the only finding is the textbook hgb-hct link. The Ask answers on lab values are right.

**Now:** good. The headline is a real, correctly tested finding (diabetes is 13 times as common in people reporting poor health as in those reporting excellent health), the data checks catch the survey's subsamples, caps and codes, and the answers check out; the main quibbles are coded labels and an unfocused summary.

## What ADA finds now

ADA reads this health survey correctly: coded answers such as sex, race and yes/no history are treated as groups rather than numbers, and the participant number as an ID. Its headline is a real finding with a proper test: the HbA1c diabetes rate is 21.7% among people who rate their health as poor and 1.7% among those who rate it excellent. It flags the survey's structure well, including the cognitive test given only to older adults, the fasting labs measured on half the sample, the age cap at 80 and stray 'refused' codes. Its answers on prevalence, group differences and correlations match independent checks. Categories are shown only as numeric codes, and the summary jumps between activity, diabetes and grip strength rather than telling one story.

Headline: “The share of rows with diabetes.1 1 differs across gen_health: highest 21.7% in gen_health 5 (n = 152) against 1.7% in gen_health 1 (n = 411), a difference of 20.0 percentage points (95% CI 13.9 to 27.3); chi-squared test, Holm-adjusted p < 0.001 across all 5 groups; Cramér's V = 0.22.”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| Integer-coded categories were treated as quantities, so no grouping was found and codes were summed. | high | fixed |
| The income band code was formatted as dollars ('income totals $33.7K'). | high | fixed |
| No group comparisons and no prevalence of the 0/1 health flags. | high | fixed |
| Columns that are mostly empty by design (subsamples) were not reported. | medium | fixed |
| The top relationship was a physiological near-identity (haemoglobin vs haematocrit). | medium | partly fixed |
| The respondent ID (seqn) was not recognised and appeared in a generated correlation. | medium | fixed |
| Questions about the share or count of a 0/1 flag were refused. | medium | fixed |
| A duplicate 'diabetes' header was renamed silently. | low | fixed |
| Survey 'refused/don't know' codes were kept as values without comment. | low | fixed |
| No caveat that unweighted means are not US population estimates. | low | not fixed |

## Still open

- Categories appear only as their codes ('gender 0', 'gen_health 5'), so a reader needs the data dictionary to see that the headline compares 'poor' with 'excellent' self-rated health; the dictionary ships with the data but is not read.
- The brief's focus is split three ways: the chosen measure is weekly activity, the headline is the diabetes rate, and the summary's 'largest difference' is grip strength by sex, a textbook result.
- The relationship card still appends haemoglobin vs haematocrit (r = 0.96) as the strongest pair without noting they measure nearly the same thing.
- Averages of the income band code are reported without noting that the bands are unequal steps.
- Plain-language comparison questions ('is grip_strength different between people with and without diabetes') are refused; the 'does X differ by Y' form works.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "NHANES_2011-2012.csv"`. The data is not stored in this repository.
