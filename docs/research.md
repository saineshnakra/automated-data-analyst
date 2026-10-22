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

`ada preview` lists what would be shared and flags anything that shouldn't be;
`ada share` copies what may be shared into `share/`. `ada demo` writes a small
example study to try it all on. Running `ada` on its own says where the folder
stands.

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

### How each analysis is fitted

Each analysis uses its complete cases and reports how many rows it left out.

| Design | `compare:` a group | `relate:` a number | `change:` time |
|---|---|---|---|
| independent | Welch's t-test (2 groups) or Welch's ANOVA with Games–Howell contrasts (3 or more); linear regression with covariates | linear regression slope, with Pearson's r and Spearman's ρ | not fitted |
| paired | paired t-test on the per-subject differences | mixed model | mixed model |
| repeated | linear mixed model, random intercept per subject | mixed model | mixed model over time |
| clustered | linear mixed model, random intercept per cluster | mixed model | not fitted for subjects nested in clusters |

A binary outcome is compared with the risk difference (Newcombe's interval),
the odds ratio, and a χ² test, or Fisher's exact test when an expected count
is below 5; related to a number, or compared with covariates, by logistic
regression. Binary outcomes in paired, repeated and clustered designs are not
fitted.

`test:` names another test the design allows, such as `mann-whitney`,
`kruskal`, `wilcoxon`, `student-t` or `anova`; ADA refuses one that does not
fit and lists those that do. In a change analysis, time with six or fewer
points is categorical (one level per time point) and otherwise a straight line;
`time_as: categorical` or `time_as: linear` chooses. With `by:`, the key result
is the time × group interaction: whether the groups change differently.

Results go in `tables/results` (planned analyses) and
`supplement/exploratory-analyses` (the rest), with each model's full output,
diagnostics and warnings in `supplement/models/`.

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

### Figures

Every figure shows each observation, not only a summary, with the means and
their confidence intervals drawn in black beside or over the points. Each
`figures:` entry has an `id` and a `type`, and may name `x`, `y`, `by`, the
`analysis` it illustrates, a `title`, a `caption` placed before ADA's own, and
`width_mm` or `height_mm`.

| Type | What it draws |
|---|---|
| `groups` | Every observation per group, spread sideways so none hide each other, with the group mean and its confidence interval. `by` splits each group again. |
| `paired` | A line per subject joining its conditions, the condition means, and a second panel with each subject's difference and the mean difference. |
| `longitudinal` | A light line per subject over time, and the mean with its interval at each time point per group (`by`). |
| `scatter` | Points, the least-squares line and its confidence band for the mean, coloured by `by`. |
| `missingness` | The share of missing values per variable, and a map of rows by variables up to 2,000 rows. |
| `diagnostics` | Residuals against fitted values and a normal Q–Q plot, for an analysis with residuals. |
| `panel` | The figures named in `panels`, side by side, lettered A, B, C. They are not drawn again on their own. |

With an `analysis`, the points are the rows the analysis used and the means
and intervals are the analysis's own. Without a `figures:` list, ADA draws one
figure per analysis: groups for a comparison of independent or clustered
groups, paired for conditions measured within each subject, longitudinal for
change over time and scatter for a numeric predictor. Planned analyses go in
the paper and exploratory ones in the supplement, with a missing-values figure
when any variable has missing values, and model checks for each analysis with
residuals.

Each figure comes with `<id>.source.csv`, holding every point drawn (with its
`source_row`) and every mean, interval and n, without the columns under
`sharing.withhold`. Its caption says what a point, a line and an error bar
are, the confidence level, n per group, the rows left out, the linked test,
and whether the analysis was planned.

`output.journal` sets the size, resolution and fonts from the journal's
author guidelines; `output.width_mm`, `height_mm`, `dpi`, `font` and
`font_size` override them.

| Preset | Widths (mm) | Max height (mm) | dpi | Text |
|---|---|---|---|---|
| `default` | 85, 170 | 225 | 300 | Arial 8 pt |
| `plos` | 132 (text column), 190.5 (most); at least 66.8 | 222.3 | 300 | Arial 8 pt, nothing smaller |
| `nature` | 89, 120 or 136, 183 | 170 | 450 | Arial 7 pt (5 to 7), panels a, b, c |
| `cell` | 85, 114, 174 | 225 | 500 | Arial 8 pt (6 to 8) |
| `science` | 90, up to 144, 184 | 199 | 600 | Myriad or Helvetica 8 pt (at least 6), panels 9 pt |
| `elife` | 100, 200 | 200 | 300 | Arial 8 pt |

Where a guideline gives no number (the height for Cell and eLife, fonts for
eLife), the preset uses a conservative one. TIFF files are RGB without an
alpha channel and LZW-compressed. EPS cannot hold transparency, so a study
that asks for EPS gets opaque tints in every format. A rebuild gives the same
bytes in every format.

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
  methods/                methods draft (Markdown and Word) and a results draft
  supplement/             supplement.md and .docx: supplementary tables and figures, model output
  reproduce/              study.yml, pinned requirements, input checksums, instructions
  repository/             README, CITATION.cff, .zenodo.json, data and code availability drafts
```

Tables come as CSV with full-precision numbers (for re-analysis) and as
Word and LaTeX with the numbers formatted for print. Figures come as PDF and
SVG with editable text, and as PNG, TIFF or EPS at the journal's resolution.

## The methods draft and the supplement

`methods/methods.md` and `methods.docx` hold a draft Methods section written
only from what the build ran: the data file and its checksum, the design with
its counts, each processing step with its reason and the rows it removed,
each analysis with its test or model, the rows it used and left out and
whether it was planned, the multiple-comparison method and its families, what
the figures show, and the exact Python and package versions, ending with a
numbered reference list. An analysis that was not fitted is named, with the
reason. `methods/results.md` gives one sentence per analysis with its
estimate, confidence interval and p-value; exploratory analyses have a
paragraph of their own.

`supplement/supplement.md` and `supplement.docx` collect the supplementary
tables and figures, the data audit's items to check, the processing record,
each model's printed output and the data dictionary.

These are drafts for you to edit: add what ADA cannot know, such as how the
data were collected and any ethics approval. `ada build` rewrites them, so
copy the text into your manuscript before editing it.

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

## Sharing and reproduction

`ada build` ends by writing two folders for other people.

`reproduce/` holds what another researcher needs to rebuild every asset:
copies of study.yml and the custom scripts (with their paths kept), the exact
package versions (`requirements.txt`), the checksums of the inputs in
`sha256sum` format (`checksums.sha256`), plain steps (`INSTRUCTIONS.md`), and
`reproduce.sh`, which checks the inputs and runs `ada build` and `ada verify`.

`repository/` holds drafts for a data or code repository: a README,
`CITATION.cff`, Zenodo's `.zenodo.json`, and data and code availability
statements in the form journals ask for. They say what study.yml says and
leave a placeholder in [square brackets] for the rest. ADA never chooses a
licence: when `study.license` is blank, the README says so and the metadata
leave it out. To link the deposit to the paper, list the paper's DOI:

```yaml
study:
  related: [10.1234/journal.0001]   # or {identifier: ..., relation: isDocumentedBy}
```

### What may be shared

`sharing` in study.yml decides what leaves your computer:

- `data: processed` shares the processed data, and `raw` adds the original
  file. `demo` shares neither but writes a synthetic table (below), and `none`
  shares no data.
- `withhold` names columns that never leave. They are removed from every CSV.
- `row_level: false` keeps back the tables that list individual rows: the
  processed data, the excluded rows, the audit's list and figure source data.
  Reports and figures are still shared.

`ada preview` lists every file in the output folder with its size, whether it
may be shared, and why. It flags withheld columns in any CSV, files listing
rows when `row_level` is false, data files when the data are not shared,
identifier columns when the data are restricted, and any file holding a value
of a withheld column (text values of four or more characters; figures in PDF,
PNG, TIFF and EPS are not searched).

`ada share` copies only the shareable files into `share/` (or `--out`), with
withheld columns removed, and writes `SHARING.md` saying what it left out and
why. It then searches the copy again with the same rules; if anything withheld
remains, it deletes the copy and says what it found.

### Demo data

With `sharing.data: demo`, the build writes `reproduce/demo-data.csv`: a
synthetic table with the columns, types and levels of the processed data,
drawn with a fixed seed. Each subject or cluster keeps its rows and time
points under a new code, and conditions that change within a subject stay as
they are. Every other column is drawn on its own, numbers by inverse-transform
sampling from the observed values and categories by shuffling, so no
relationship between columns survives. Withheld columns are left out.
`demo-data.README.md` says how each column was made, and `demo-study.yml` runs
the whole build on it:

```
ada build --study paper-assets/reproduce/demo-study.yml
```

### An example study

`ada demo [FOLDER]` writes an example study: simulated weights of sixteen mice
on two diets, weighed weekly, with a study.yml that has authors, units, two
exclusions with their reasons, a planned and an exploratory analysis, figures,
a custom table and a withheld column.

```
ada demo
cd ada-demo && ada build
```

## Verifying

`ada verify` checks four things and reports each under its own heading:

- **Inputs.** The data file is there and has the checksum in `study.yml` and
  in `provenance.json`.
- **Out of date.** Each asset was built from the current `study.yml` (naming
  the parts that changed), custom scripts and ADA version, and nothing
  `study.yml` now asks for is missing.
- **Edited since the build.** No file has changed since the build recorded
  its checksum.
- **Rebuild in a scratch folder.** `study.yml`, the data file and the custom
  scripts, copied alone into a scratch folder as a fresh `git clone` would
  hold them, rebuild to the same assets. CSV files are compared cell by
  cell: numbers within a tolerance (`--rtol`, default 1e-9, and `--atol`,
  default 1e-12), text exactly. Figures match when their source data match,
  Word files are compared by their text, and other text files exactly.
  Each difference is shown with its file, line, column and both values.

```
$ ada verify
Inputs
  ok    data/assay.csv matches study.yml and the build (sha256 3f2a9c81d0e4…).
Out of date
  ok    All 14 assets were built from the current study.yml by this version of ADA (0.3.0).
Edited since the build
  ok    None of the 41 files recorded in provenance.json has changed.
Rebuild in a scratch folder
  ok    Rebuilt from study.yml and data/assay.csv alone in a scratch folder, and every file
        matches: …

Everything matches: 14 assets and their 41 files.
```

It exits with 0 when everything matches and 1 otherwise, so it can run in
continuous integration. `--out` checks a folder other than `output.dir`.
