# Enhanced Gait Biomechanics Dataset

[anitarostami/enhanced-gait-biomechanics-dataset](https://www.kaggle.com/datasets/anitarostami/enhanced-gait-biomechanics-dataset) on Kaggle · sensor · 178,200 rows × 9 columns · licence: Attribution 4.0 International (CC BY 4.0) · round 3

Joint-angle curves over the gait cycle for 10 subjects walking in 3 conditions (by the dataset's description unbraced, knee brace and ankle brace), with 10 replications, 2 legs and 3 joints, 99 points per curve (178,200 rows). angle is rescaled to run exactly 0-1 across the whole file; velocity and acceleration are its first and second differences.

**First run:** partly useful. Correct numbers and a sensible per-subject headline, but it pools joints and the gait cycle, uses independent tests on a within-subject design and so misses the large, joint-specific brace effects.

**Now:** good. The condition comparison now uses the right repeated-measures test and breaks the effect down by joint, showing lower knee angles with condition 2 and lower ankle angles with condition 3; the per-curve range of motion, where the brace effect is largest, is still missing.

## What ADA finds now

This dataset holds joint-angle curves over the walking cycle for 10 people tested unbraced and with knee or ankle braces. ADA now recognises that every person walked in every condition and uses the matching repeated-measures tests; its condition differences check out exactly. It also splits the effect by joint, showing that knee angles drop in one braced condition and ankle angles in the other, and it describes the average gait curve in time order. Its headline still averages the three joints together, and it does not compute each curve's range of motion, where the braces make their biggest difference (about a third less movement at the braced joint).

Headline: “angle differs across condition: condition 1 has the highest mean of per-subject means, 0.384 (n = 10), and condition 3 the lowest, 0.361 (n = 10); partial η² = 0.59.”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| A reading recorded at several joints was averaged across them, and the joint was then reported as a group difference. | high | partly fixed |
| A within-subject design was tested with independent-groups tests. | high | fixed |
| velocity and acceleration, time derivatives of angle, were not recognised, so their correlation was reported as a relationship. | medium | partly fixed |
| No per-curve summary such as range of motion; 'range of angle by condition' was refused. | medium | partly fixed |
| 'compare angle between condition 1 and condition 2' read 'condition 1' as an extra filter. | medium | fixed |
| A breakdown by the 99-point cycle index was ranked by value instead of shown in order. | low | fixed |
| angle rescaled to exactly 0-1 is not noted. | low | not fixed |

## Still open

- The main brace effect, a 35-38% drop in each curve's range of motion at the braced joint, is not reported, and 'range of angle' gives the pooled max-minus-min, which hides it.
- The headline still averages angle over ankle, knee and hip; the joint-specific results that matter sit lower in the same card.
- The 'What differs between groups' card leads with joint differences (partial η² 0.93) and leg differences, which reflect anatomy and the shared rescaling, not findings.
- The within-joint statistics are computed on a 50,000-row sample, so they differ slightly from the full data and from asked answers (knee partial η² 0.69 in the brief vs 0.75 when asked; full data 0.75).
- velocity and acceleration are not recognised as derivatives of angle, and the 0-1 rescaling of angle is not mentioned.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "Gait_Biomechanics_Dataset.csv"`. The data is not stored in this repository.
