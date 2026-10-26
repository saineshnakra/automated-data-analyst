# Changelog

## Unreleased

### Fixes

- Reading a table with hashed identifiers no longer crashes Python. pandas
  2.3 crashes on text such as `689e8820798347950a2649f71c734bd0`, which it
  reads as a number with an exponent past its limit; ADA 0.5.0 crashed on the
  public Olist payments table this way. Such text is now set to missing
  before pandas reads it as a number, which gives the same result without the
  crash.

- The plugin is now its own folder, `plugins/automated-data-analyst/`, named
  `automated-data-analyst`, with manifests for the Claude and ChatGPT plugin
  directories, an icon, a README and the license. It holds the same skill as
  `ada skill` and no code. Install it with
  `/plugin marketplace add saineshnakra/automated-data-analyst` then
  `/plugin install automated-data-analyst@automated-data-analyst` in Claude
  Code, or `codex plugin add automated-data-analyst@automated-data-analyst`
  in Codex. An install of `ada@automated-data-analyst` moves to the new
  name; run the install command once to fetch it.

## 0.5.0 (2026-10-23)

ADA now works as a skill in Claude Code and Codex, and the study workflow is
safer on real research data: an outside evaluation of 0.4.0 found seven
defects on public datasets, and every one is fixed.

```bash
pip install --upgrade automated-data-analyst
ada skill      # the skill, for Claude Code and Codex in this project
```

### A data-analysis skill for Claude Code and Codex

- `ada skill` installs ADA's skill where Claude Code (`.claude/skills/ada`) and
  Codex (`.agents/skills/ada`) look for skills. `--for claude` or
  `--for codex` picks one, `--user` writes to your home folder for every
  project, and `--dir` to any other tool's skills folder.
- It also installs as a plugin. In Claude Code:
  `/plugin install ada --marketplace saineshnakra/automated-data-analyst`.
  In Codex: `codex plugin marketplace add saineshnakra/automated-data-analyst`,
  then `codex plugin add ada@automated-data-analyst`.
- The skill tells the assistant to ask you for what only you know (what a row
  is, the subject column, the design, the planned analyses), to let ADA do
  the arithmetic, to report ADA's numbers with the calculation behind each,
  and to pass a refusal on instead of working around it.
- `llms.txt` at the root of the repository gives a language model a short
  map of the docs.

### Study workflow: `ada init`, `ada build`, `ada verify`

Found by an outside evaluation on UCI Wine, Wine Quality, Parkinsons
Telemonitoring and Beijing air-quality data, and fixed:

- **Repeated subjects.** `ada init` proposes a column of repeated numbered
  codes (`subject#`, `Patient ID`) as the subject of a repeated design and
  asks you to confirm it. A guessed design is written as
  `design.confirmed: false`, and the methods draft says it was not confirmed.
- **Mixed models that can't be trusted are refused**, with the reason: fewer
  than 2 subjects or clusters, a subject alone in its group, a fit that did
  not converge, or a singular fit.
- **Rows that form a series.** An independent design with a time column gets
  a Durbin–Watson check in time order, and a warning when successive rows
  stay alike, since its p-values are then too small.
- **Small numbers never print as zero** or lose their sign: `−1.20 × 10⁻⁵`,
  not `0.0000`.
- **Proposed analyses are checked** with the build's own rules before
  `study.yml` is written; groups coded as numbers are written as
  categorical.
- **`ada build` counts** the analyses requested, fitted and refused, with the
  reason for each refusal. `ada build --strict` exits with 1 when any was
  refused.
- **Failed builds keep the last good assets.** A build works in a scratch
  folder and replaces the old files only once every step has succeeded.
- **Scatter figures** draw the linked analysis's own fitted line and band,
  with covariates held at stated values, and their source data carry the
  table's slope and correctly paired confidence limits.
- **Hourly data** keeps its timestamps: year, month, day, hour, minute and
  second columns combine to the finest grain given.
- Smaller fixes: the "on the boundary" caveat is judged against the residual
  variance, captions say "1 subject", and a bracketed word becomes a unit
  only when it is a known unit.

### Web app

- The app opens on a page for researchers, with an example study that ADA
  builds on the server and you can download, and the first look on its own
  page (`/first-look`).
- Starlette is kept below 1.0, since Streamlit 1.59 cannot serve images or
  downloads on Starlette 1.x.

### Upgrading from 0.4.0

- A `study.yml` that names `year-month` as a variable stops with "no such
  column" when the file also has day or hour columns, because the time is
  now built at the finer grain. Run `ada init` again, or rename the variable.
  Your previous assets are kept.
- A mixed model that 0.4.0 reported although its fit did not converge or was
  singular is now refused. Rebuild, and `CHANGES.md` lists what changed.
- Tables print very small values in scientific notation, so rebuilding a
  study with such values changes those files.
- A `study.yml` without `design.confirmed` counts as confirmed, as before.

## 0.4.0 (2026-10-22)

- **Paper assets for a study.** `ada init` writes a `study.yml` from the data
  and your answers; `ada build` makes a data audit, a data dictionary, a
  processing record, tables in Word, LaTeX and CSV, figures sized for
  journals with their source data and captions, a methods draft, a
  supplement, a reproduction package and repository metadata, each traced in
  `provenance.json`; `ada verify` rebuilds in a scratch folder and checks
  every number. `ada preview`, `ada share` and `ada demo` handle sharing and
  an example study.
- **The first look**, tested on 60 research datasets in three rounds, with
  the fixes from each round and the results in `results/`.

## Earlier versions

See the [releases on GitHub](https://github.com/saineshnakra/automated-data-analyst/releases).
