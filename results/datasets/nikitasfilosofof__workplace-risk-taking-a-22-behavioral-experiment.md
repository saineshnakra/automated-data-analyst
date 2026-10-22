# Workplace Risk-Taking: A 2×2 Behavioral Experiment

[nikitasfilosofof/workplace-risk-taking-a-22-behavioral-experiment](https://www.kaggle.com/datasets/nikitasfilosofof/workplace-risk-taking-a-22-behavioral-experiment) on Kaggle · behavioural · 231 rows × 35 columns · licence: Attribution 4.0 International (CC BY 4.0) · round 1

A 2x2 between-subjects vignette experiment (employee vs contractor role, low vs high firm-specific investment) with 231 participants. Each row holds rating items plus derived scores: risky_decision (the outcome), risk_perception (the mediator), risk_propensity and a realism check.

**First run:** poor. The numbers are correct, but ADA analysed the realism check instead of the outcome, headlined a non-significant 0.14-point gap, and offered a circular r = 0.97, so it missed the experiment's actual result (FSI lowers risky decisions; perceived risk r = -0.51 with them).

**Now:** good. It analyses the right outcome, tests the condition effect, separates the two factors (firm-specific investment matters, p = 0.012; role does not) and reports the perceived-risk link without circular pairs; only the interaction test is missing.

## What ADA finds now

ADA picks the study's outcome, the risky-decision score, and finds that it differs across the four experimental conditions. It then splits the design into its two factors and shows that high firm-specific investment lowers risky decisions while the employee-versus-contractor role makes no clear difference, each with a test. It reports that people who see more risk make less risky choices (r = -0.51) and leaves out correlations between a score and its own items. It does not test whether the two factors interact, and the link between investment and perceived risk only appears when asked.

Headline: “risky_decision differs across condition: Employee_Low has the highest median, 5.33 (n = 69), and Contractor_High and Employee_High the lowest, 4.67 (n = 62); ε² = 0.04.”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| The measure was picked alphabetically, giving the realism manipulation check instead of the study outcome. | high | fixed |
| Group means were ranked with no test, and a non-significant realism gap was the headline. | high | fixed |
| The strongest relationship was a part-whole artefact (risk_perception vs risk_perception_extended, r = 0.97). | medium | fixed |
| Group-difference, significance, correlates-of-X and participants-per-condition questions refused. | medium | fixed |
| Sales wording for experimental cells ('Leading condition', 'Top condition'). | low | fixed |

## Still open

- The 2x2 interaction is never tested and 'is there an interaction between contractor and fsi_high' is refused; a two-way ANOVA gives interaction p = 0.063, with the investment effect much larger for employees (5.34 vs 4.36) than contractors (4.86 vs 4.78).
- The 92-test differences card includes every rating item, so after adjustment it reports nothing clear and the mediator result (high investment raises risk_perception, 3.10 vs 3.48, Welch p = 0.014) appears only when asked.
- The headline names two tied lowest cells but gives one sample size ('Contractor_High and Employee_High ... (n = 62)'); Contractor_High has 54.
- Medians are justified by '8 values sit at its minimum of 0', though 0 is a valid answer on a 0-10 slider rather than a floor problem.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "participants.csv"`. The data is not stored in this repository.
