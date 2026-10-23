# Secchi Depth

[jacobsharples/secchi-depth](https://www.kaggle.com/datasets/jacobsharples/secchi-depth) on Kaggle · environmental · 404 rows × 30 columns · licence: CC0: Public Domain · round 3

404 water samples from about 45 sites in Georgian Bay (Lake Huron), taken on 9 sampling trips between June 2003 and October 2005, each with Secchi depth (water clarity, m), site depth and 27 water-chemistry readings. 45 Secchi values are written as '>X' because the disk was still visible on the bottom.

**First run:** poor. It reads the file well but picks and sums the wrong reading, so the headline, KPIs, trend and forecast are about added-up nitrate concentrations.

**Now:** good. The headline and the relationship scan (turbidity, organic carbon and nitrogen go with lower clarity) are right and useful, trips and censored readings are recognised, and the main remaining flaw is that the trend is reported two different ways.

## What ADA finds now

These are water samples from Georgian Bay with water clarity (Secchi depth) and lake chemistry. ADA now studies clarity itself and finds that it falls as turbidity, organic carbon and nitrogen rise (rank correlation -0.88 with turbidity), which checks out and is the expected result. It recognises that the samples come from 9 sampling trips rather than a monthly record, explains that 45 '>X' readings mean the disk was visible on the lake bottom, and flags phosphate sitting at its reporting floor in 94% of rows. Its trend reporting is inconsistent: the brief reports clarity rising by 1.15 m a year across trips, while asking for the trend gives 'no clear trend', and neither notes that the trips fell in different seasons each year.

Headline: “secchi and turbidity move in opposite directions in rank: the rank correlation is -0.88, while Pearson r = -0.51 (95% CI -0.58 to -0.43, n = 359).”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| Concentrations named '..._total' were summed across samples for the KPI, trend and forecast. | high | fixed |
| nitrates_total was picked as the measure over secchi, the subject of the file. | high | fixed |
| A slope over sparse periods was computed per inferred quarter but labelled 'per month'. | medium | fixed |
| Repeated sampling trips were forced onto a calendar-month grid; trips are the natural unit. | medium | partly fixed |
| Censored '>X' values were called 'above a detection limit' though X is the site depth. | medium | fixed |
| Data checks missed the floor on phosphate (0.001 on 94% of rows) and others. | low | fixed |
| With secchi as the measure, the headline was a sign reversal in a collinear model. | low | fixed |

## Still open

- The brief and the answers disagree on the trend: the trip card says Secchi depth rose 1.15 m per year (p = 0.022) across 9 trips, while 'secchi trend' says no clear trend on calendar months; neither notes that the 2003 trips fell in different months from later years, or that the rise is almost all between 2003 (3.50) and 2004 (5.53), with 2005 the same (5.53).
- The trend answer drops the June 2003 trip as a 'partial month' and splits the April-May 2004 trip in two.
- The relationship card says magnesium is 'a fixed combination of calcium, chlorophyll_b_total, chloride, color, hardness and potassium'; calcium and hardness alone explain it (R2 = 0.9999, the hardness formula), so four of the six names are noise.
- The joint model keeps alkalinity, magnesium and hardness together (pairwise r 0.989 to 0.996); it says the coefficients are unstable, but the model adds little beyond nitrogen alone (R2 0.52 vs 0.47).
- Questions about trips or censored readings are refused, and the 45 '>X' readings are still called 'missing' in answers.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "georgian_bay_secchi.csv"`. The data is not stored in this repository.
