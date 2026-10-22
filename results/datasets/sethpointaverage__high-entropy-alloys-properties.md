# High Entropy Alloys - Properties

[sethpointaverage/high-entropy-alloys-properties](https://www.kaggle.com/datasets/sethpointaverage/high-entropy-alloys-properties) on Kaggle · materials · 1,545 rows × 20 columns · licence: CC BY-SA 4.0 · round 3

1,545 literature records of high-entropy alloys from 265 papers: composition, processing route, phase (BCC/FCC/other), test type (compression or tension), test temperature and mechanical properties such as hardness, yield and tensile strength and elongation. About half the cells are empty.

**First run:** partly useful. The data checks and per-paper averaging are good, but the headline compression-vs-tension gap mostly reflects which alloys and temperatures are tested each way, and an 'explains 100%' result comes from a measurement misread as a category.

**Now:** partly useful. The data checks, per-paper averaging and question handling are now good, but the headline compression-versus-tension gap is confounded by alloy phase and test temperature, and the one stratified check offered points the wrong way.

## What ADA finds now

This table gathers 1,545 published test results on high-entropy alloys from 265 papers. The tool averages within each paper before testing, flags the compression tests stopped at 50% strain, the duplicated rows and the heavy, patterned gaps, and now answers questions phrased with the table's long column names. Its headline, that yield strength is higher in compression than in tension (1,185 against 633 MPa), mostly reflects which alloys and temperatures are tested each way: within BCC or FCC alloys the gap disappears. The tool checks the gap only within processing method, where it holds, so a reader is wrongly reassured. Answers also pair row averages with the effect size from the per-paper comparison.

Headline: “PROPERTY: YS (MPa) is higher in C than in T: mean of per-IDENTIFIER: Reference ID means 1,184.6 vs 633.06 (n = 115 and 77).”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| The headline compression-vs-tension gap is confounded by phase and test temperature, and neither is checked. | high | not fixed |
| O content (a measurement) was read as a category and then 'explained 100%' of yield strength. | high | fixed |
| Elongation above 100% was called impossible, while the real cap (compression stopped at 50%) was missed. | medium | fixed |
| Questions with the shared 'PROPERTY:' prefix, LaTeX units or a slash in a column name failed. | medium | fixed |
| Different parts of the output use different aggregations for the same comparison and attach one effect size to both. | medium | not fixed |
| The 'not recorded' group was ranked as the leader in answers. | low | fixed |
| The composition string is treated as a 630-level label. | low | not fixed |

## Still open

- The headline gap between compression and tension tests mostly reflects which alloys and temperatures are tested each way; within BCC and FCC alloys it disappears, and the only check shown (within processing method) wrongly suggests it holds.
- Answers pair row averages (1,013 vs 607 MPa) with the per-paper effect size (g = 1.20); the row-level g is 0.75.
- Test temperature, which lowers yield strength (r = -0.42) and differs between test types, is never used as a covariate.
- Oxygen content is listed as a strong correlate of strength (r = 0.84) on 55 rows from 10 papers, driven by two extreme values.
- The composition string is treated as a label, so element content cannot be related to phase or strength.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "High Entropy Alloy Properties.csv"`. The data is not stored in this repository.
