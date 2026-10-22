# Paper assets for a study

ADA can make the assets a paper needs from your data: a data audit, a data
dictionary, a record of every exclusion, descriptive and results tables,
figures with their source data and captions, a methods draft, and a package
that lets another researcher rebuild all of it. Every number traces back to
the data through `provenance.json`, and every decision behind it is written in
one file, `study.yml`, that you can read and edit.

It is built for quantitative tabular studies: one table where each row is an
observation, from an independent, paired, repeated-measures or clustered
design.

## Three commands

Run them in your project folder, next to your data:

```
pip install automated-data-analyst
ada init data/measurements.csv     # answer a few questions; writes study.yml
ada build                          # writes paper-assets/
ada verify                         # rebuilds in a scratch folder and checks everything matches
```

`ada preview` lists what would be shared and flags anything that shouldn't be.
Running `ada` on its own says where the folder stands.

The same commands are also under `ada research …`, and in Python:

```python
from ada import research
research.init("data/measurements.csv", interactive=False)
built = research.build("study.yml")
```

## `ada init`

ADA reads the file, guesses what each column is, and asks:

1. What one row represents.
2. Which columns identify a participant, specimen, site or experiment.
3. The outcomes, the groups or conditions, the time point, and any covariates.
4. The unit of each measurement, and any values that mean "missing".
5. The design: independent, paired, repeated or clustered.
6. Which analyses were planned before you saw the data. The rest are exploratory.
7. Which data can be made public, and which columns must never be shared.

Each question shows ADA's guess; press Enter to accept it. `ada init --yes`
accepts every guess and marks every analysis exploratory, because only you
can say what was planned. The answers go into `study.yml`, which is yours from
then on: ADA reads it and never rewrites it.

## `study.yml`

```yaml
study:
  title: Effect of compound X on cell viability
  authors:
    - name: A. Researcher
      orcid: 0000-0000-0000-0000
  license: CC-BY-4.0          # for the shared data; you choose it
  code_license: MIT

data:
  path: data/assay.csv
  sha256: 3f2a…               # ADA stops if the file changes
  observation: one well
  missing_codes: [-99]

design:
  type: clustered             # independent | paired | repeated | clustered
  cluster: Plate              # repeated and paired designs name a `subject` instead
  time: null

variables:
  Viability (%):
    role: outcome             # outcome | group | time | covariate | identifier | other
    type: numeric             # numeric | categorical | binary | date | text
    unit: "%"
    range: [0, 120]           # checked in the audit, never enforced
  Treatment:
    role: group
    type: categorical
    levels: [Control, Low, High]   # the first is the reference

processing:
  - exclude:
      where: {column: Viability (%), below: 0}
    reason: Readings below zero are instrument errors (lab SOP 4.2).
  - exclude:
      ids: [W017]
      id_column: Well
    reason: Contaminated well, noted at the bench.
  - transform: {column: Dose, to: log_dose, apply: log10}
    reason: Doses span four orders of magnitude.

decisions:
  missing_data: complete-case
  outliers: keep              # keep | flag
  multiple_comparisons: holm  # holm | bonferroni | fdr_bh | none
  alpha: 0.05
  confidence: 0.95

analyses:
  - id: primary
    question: Does viability differ between treatments?
    outcome: Viability (%)
    compare: Treatment        # or relate: a numeric predictor, or change: the time variable
    covariates: []
    planned: true
    family: primary

figures:
  - id: viability
    type: groups
    y: Viability (%)
    x: Treatment
    analysis: primary

output:
  dir: paper-assets
  journal: plos
  tables: [csv, docx, tex]
  figures: [pdf, svg, tiff]

sharing:
  data: processed             # processed | raw | demo | none
  withhold: [Operator]
  row_level: true

custom: [analysis/extra.py]
```

### Processing

Steps run in order. Each needs a `reason`, and each appears in the processing
record with the rows before and after.

- `exclude` removes rows where every `where` condition holds, or rows whose
  `id_column` (by default the design's subject) has one of the listed `ids`.
  Conditions: `below`, `above`, `at_most`, `at_least`, `equals`, `not_equals`,
  `in`, `not_in`, `missing`.
- `transform` writes a new column (`to`) from `column` with one of `log`,
  `log10`, `log2`, `sqrt`, `square`, `reciprocal`, `zscore`, `center`, or
  `scale` (with `factor`).
- `recode` maps old values to new ones (`map`), in place or into `to`.

The original file is never changed. Removed rows are saved with the step that
removed them and their line in the original file.

### Statistical decisions

ADA only runs the analyses listed under `analyses`. It does not search for
significant results, and every result says whether it was planned.

- **Missing data.** Each analysis uses the rows with every value it needs
  (complete cases), and reports how many it left out. ADA does not impute.
- **Outliers.** ADA never removes a value on its own. `flag` lists extreme
  values in the audit; to remove any, add an `exclude` step with a reason that
  does not depend on the result.
- **Design.** The design decides the test: Welch's t-test or ANOVA for
  independent groups, the paired t-test for paired data, a linear mixed model
  with a random intercept per subject for repeated measures, and per cluster
  for clustered data.
- **Multiple comparisons.** p-values are adjusted within each `family`, by
  Holm's method unless you choose another.
- **Covariates.** Listed covariates enter the model as fixed effects.
- **Unsupported designs.** Crossover trials, nested random effects beyond one
  level, survival outcomes, ordinal or multinomial outcomes, and binary
  outcomes in repeated or clustered designs are not fitted. ADA says so and
  points to `custom` for your own analysis.

### Your own tables and figures

List a Python file under `custom`. In it, mark functions that take the
processed data and return a pandas table or a Matplotlib figure:

```python
from ada import research

@research.table("adjusted-model", title="Viability adjusted for passage")
def adjusted(data, study):
    ...
    return frame

@research.figure("dose-curve", title="Dose response", caption="Points are wells.")
def dose_curve(data, study):
    fig, ax = plt.subplots(figsize=research.figure_size(study))
    ...
    return fig, source_frame
```

They run on the processed data, are saved in the study's formats, and are
recorded in `provenance.json` with the script's checksum.

## What `ada build` writes

```
paper-assets/
  README.md               every asset, with its caption and files
  CHANGES.md              what changed at each build, and why
  provenance.json         where each asset came from
  audit/                  data-audit.md and .csv
  dictionary/             data dictionary (CSV, Word, LaTeX)
  processing/             processing record, excluded rows, processed data
  tables/                 descriptive and results tables (CSV, Word, LaTeX)
  figures/                figures, source data (.source.csv) and captions.md
  methods/                methods draft (Markdown and Word)
  supplement/             exploratory analyses, model output and diagnostics
  reproduce/              study.yml, pinned requirements, input checksums, instructions
  repository/             README, CITATION.cff, .zenodo.json, data and code availability drafts
```

Tables come as CSV with full-precision numbers (for re-analysis) and as
Word and LaTeX with the numbers formatted for print. Figures come as PDF and
SVG with editable text, and as PNG, TIFF or EPS at the journal's resolution.

## Traceability

For each asset, `provenance.json` records its files and their sha256, the
input file's checksum, how many rows it used and which exclusion steps
removed others, the parts of `study.yml` it depends on, the function that
made it, the calculation in words, references, and the versions of Python and
every package involved.

Each build compares itself with the previous record. `CHANGES.md` names every
asset that changed and why: which part of `study.yml`, the input data, a
custom script, or a version of ADA, Python or a package. For each it gives the
rows used and the headline numbers, old → new, and the files that differ, and
it names the assets that stayed the same. Entries are numbered, not dated.

Neither file holds a date or anything else that changes from one build to the
next, and Word files are written with fixed dates inside, so rebuilding an
unchanged study gives the same bytes and `git diff` stays quiet.

## Verifying

`ada verify` copies the study and data into a scratch folder, builds there,
and compares the result with `paper-assets/`: every table and source-data
file number by number within a tolerance, figures by their source data, and
the input checksums. It reports missing inputs, assets that are out of date
with `study.yml`, and files edited by hand since the build.
