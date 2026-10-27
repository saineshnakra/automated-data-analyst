# Changelog

## Unreleased

### Reading claims and questions

- "6,657 metres high" reads a height, and "Customers paid R$16,008,872" a
  payment: an adjective after the unit and a verb before the number name the
  quantity (high, tall, long, wide, deep, heavy, old; paid, spent, earned,
  sold, charged, cost).
- A people noun as a claim's subject that names nothing in the table, and a
  capitalised word before the rows' own noun ("the average Lego set"), stand
  for every row, said in the assumptions; a word that is a value still
  filters.
- A number with a currency symbol picks the one money column among the
  columns the words fit.
- "Sets from 2019" means 2019's sets, while "from 2019 onwards" stays a
  range; "released in 2020" and "on board" no longer name nothing.
- Words a claim uses that name nothing in the table are left out, and the
  rest is checked: "Brazil lost 2,564,000 hectares of forest to pasture in
  2004". Such a check verifies only a lookup of one row, a total, a count or
  a slope; a mismatch, an average and a bound give `missing_context`, so a
  left-out word that picks rows can never give a wrong verdict.
  `words_left_out` in the JSON lists them.
- Slopes: "Reaction time rises by about 10.5 ms per day" is compared with
  the slope (`scipy.stats.linregress`, or the mean of each subject's own
  slope), and a fall with the slope reversed. Correlation answers carry
  `slope of Y on X` figures.
- Fractions and numbers in words ("about a quarter of", "more than half
  of", "Seventeen cases"), "with 4 cylinders" as a filter, `--value "about
  25%"` keeping its qualifier, feet, inches, miles, pounds and tonnes as
  units, and columns named for a length or a mass (`height`, `wgt`) for a
  number in that unit.
- A number that counts things ("1,524 Kenyans") is checked as a total, not an
  average. "73 HR" names the `HR` column, not hours, and digits inside a code
  ("bondsba01") are not a number.
- Fixed wrong verdicts: a count of rows pinned to one row by a key ("Nairobi
  had 5.2 million residents" on a table of counties) and a column guessed
  from part of a word ("seedlings" for `Seed`) are no longer compared with
  the claim; a bound is never read in thousands or millions.

### Answers

- A share of a group is of that group: "what percentage of Adelie are
  female" gives 50% of the Adelie, not 21.9% of all penguins, and "what share
  of women survived" 73.2% of the women. Before, the base was every row,
  which made a claim such as "50% of Adelie penguins are female" fail.
- "Men", "women", "boys" and "girls" name the values of a column of sexes,
  "department" names `Dept` and "weight" `wgt`, and in a table of counts a
  plural that names nothing else ("applicants") names the cases.
- A claim's lead that names its base ("Of the 4,526 applicants,") is checked
  against the share's base, "at a rate of" is grammar, "1,835 women" counts
  women, a plural after the number that names no column counts rows ("1,950
  captures in 2007"), and "rose to 20.9 tonnes per hectare" is a level, not a
  slope.
- Tables of counts weight shares and counts by group too: Titanic's "share of
  Survived = Yes by Class" gives 62.5% for 1st class (203 of 325), not 50% of
  rows, and its chi-squared test runs on the weighted table.
- "how many trees" is refused, not answered with the row count, when each
  tree has several rows; "how many reviews in 2019" counts rows, not
  distinct years; "how many Gold medals were won" reads "were won" as
  grammar.
- A column named as a date that holds only years (`review_date`) scopes "in
  2019", a code in a column too large to index ("bondsba01") filters, and "on
  day 21" picks the time-point column's value.
- A clock time read onto one day ("1:27.452") is no longer the table's date,
  and joins leave out birth and death dates, so "in 2019" in a join of F1
  results, drivers and races is the race year: "Lewis Hamilton scored 413
  points in 2019" checks across the three files.

### Decisions

- Answers and claim checks list the choices they rest on: `decisions` in the
  JSON and a `decides:` line in the text, each with the question, the
  options, ADA's default, the option used and why the default is the
  default. `Analysis.ask(question, decisions=...)`, `ada.check_claim(...,
  decisions=...)` and `--decide KEY=OPTION` on `ada ask` and `ada claim` pick
  another option; an unknown key or option raises `UnknownDecision` (exit
  code 2). A person, or an agent that knows the person's goal, decides; with
  no one to ask, the defaults give the same answer every time. See
  [Decisions](docs/reference/decisions.md).

### Questions first, and answers across files

- `ada scan` asks what the analysis should find out first, and asks about the
  decisions too: for each file with World or region rows, and for joining
  files. It also scans a single file (`ada scan data.csv`). The answers file
  `--template` writes is read by `ada ask` and `ada claim` with `--answers`
  (`answers=` in Python): `decide:KEY` for every file, `decide:FILE:KEY` for
  one, and `--decide` over both.
- `ada ask` takes a folder (`ada.ask_folder` in Python), and a question or
  claim about a folder that no single file answers is tried on two files
  joined along a link: "Barry Bonds hit 73 home runs in 2001." is verified on
  `Batting.csv` joined to `People.csv`. Only many-to-one and one-to-one links
  are joined, by default only those that need no confirmation (`join_links`:
  `sure`, `all`, `none`), and the answer says which join it used.
- A person named by first and last name, in a column each, is found even
  when there are more names than the value index keeps.
- After a verb that adds things up (hit, scored, earned, collected,
  recorded), the words after the number name the column that is totalled:
  "scored 3,999.5 points", "hit 73 home runs". "home runs" reads `HR` by its
  initials wherever a claim names a column.

### Rows that add up others

- World, region and Total rows listed beside the rows they add up (OWID's
  World and Africa among countries, a spreadsheet's Total) are found by their
  labels and left out of averages, rankings, counts and comparisons, and the
  answer says which (`aggregate_rows`, default `leave_out`). A total over the
  scope is read from the World or Total row, with what the other rows add up
  to beside it (`total_from`, default `total_row`). "total co2 in 2022" on
  the OWID file gives 37,527.773, the World row, instead of 240,370.227.
- A claim that agrees only when one of these decisions goes the other way is
  `ambiguous_intent`, with both readings and the question to settle it.

### Fixes

- A year in a question scopes a numeric column named for years when the table
  has no date column (OWID's `year`, from 1750), and a year before 1900 is
  read when that column holds it.
- An aggregate over one row says "calculated from 1 row".

## 0.6.0 (2026-10-27)

ADA now checks numbers as well as computing them. `ada claim` says whether a
number stated about a table is verified or unverified, and why; `ada ask
--json` gives an answer's numbers and rows as data; `ada scan` maps a folder
of related files and the keys that join them. Answers count each subject once
when several rows come from one, and say when groups reverse a pooled
correlation. Evaluated on 73 cases from 20 public datasets
([results/verification](results/verification/README.md)).

```bash
pip install --upgrade automated-data-analyst
ada skill      # the updated skill: check every number with ada claim
```

### The skill

- The skill tells an assistant to compute with `ada ask`, check every number it
  states with `ada claim`, say which numbers are verified (with their
  calculation) and which are not and why, ask ADA's clarifying question
  instead of picking a reading, and run `ada scan` before combining files.
  Run `ada skill` again, or `ada skill --check`, after upgrading.

### Reading a folder of tables

- `ada.scan(folder)` and `ada scan FOLDER` profile every table in a folder,
  propose the keys that join them from their values (with the coverage both
  ways, the cardinality, the rows a join leaves unmatched and how many rows
  it repeats), find composite keys such as `(yearID, teamID)`, read a
  codebook's units, and list what a person should confirm, grouped by topic:
  what a row stands for, units, repeated measurements, links and the
  question to answer. `--template FILE` writes the questions with ADA's
  proposed answers; `--answers FILE` reads the confirmed answers back.
  `--json` prints all of it. See [Reading a folder of tables](docs/reference/folder.md).
  Whole numbers link only under matching names, since ids, positions and
  counters overlap by chance, and a link where both sides repeat needs a key
  word (`id`, `code`, `zip`, `number`) in its name. On the Formula 1 tables
  this leaves the 7 links the schema documents and one to confirm. Two bare
  `id` columns, each its own file's row id, link only when each holds 90% of
  the other's values.
- `ada.check_claim` and `ada claim` take a folder and check the claim against
  the file it fits; when no single file does, the answer names the keys that
  would join two of them.

### Checking numbers against the data

- `ada.check_claim(data, claim)` and `ada claim FILE "claim"` check a number
  someone states about a table, such as "The average departure delay was
  12.64 minutes". The result is `verified`, or `unverified` with one of six
  reasons: `missing_context`, `ambiguous_intent`, `inaccessible_data`,
  `unsupported_analysis`, `conflicting_evidence` or `failed_check`. Each check
  gives the interval the claim's number stands for ("12.64" is 12.635 to
  12.645, "about 13" is 12 to 14), the calculation, the rows used and left
  out, the file's sha256, its assumptions and warnings, and, when the claim
  can't be checked as stated, the one question that would let it go ahead.
  `--json` prints all of it; the exit code is 0 verified, 1 failed check, 2
  unreadable file, 3 unverified for another reason.
  See [Checking claims](docs/reference/claims.md).
- `ada ask FILE "question" --json` and `QueryAnswer.to_dict()` give an
  answer as data: the full-precision `value`, the `values` per group, other
  `figures` (counts, shares and their denominators, p-values, intervals), how
  the question was read, and the rows read, used and left out with a reason
  for each.
- An answer names its source: `Analysis.source` and `QueryAnswer.source`
  hold the file's name, path, sha256, rows and columns.
- A question can name the rows by the file's name: "how many flights" in
  flights.csv counts rows instead of distinct values of a `flight` column.

### Fixes

- A table of counts is counted by its counts. When each row is one
  combination of categories with a column such as `Freq` holding how many
  cases it stands for, a count or share adds up that column, and the answer
  says so: Titanic's "what share survived" is 711 of 2,201 people (32.3%),
  not half of its 32 rows. A share of a Yes/No column counts its Yes rows
  ("what share survived" was answered with the number of distinct values),
  and a percent claim that names no column ("32.3% survived") is checked as
  a share.
- Ordinals are labels. Titanic's `Class` ("1st", "2nd", "3rd", "Crew") was
  read as numbers in units called "st", "nd" and "rd", with "Crew" missing; it
  stays a column of classes.
- A mismatch is a failed check only when the claim names what the column
  measures. "Expeditions recorded 788 member deaths" read "deaths" as
  `death_height_metres`, summed the heights at which climbers died (11,050)
  and called the claim false; such a mismatch is now `missing_context`, with
  the question "Is the number about death_height_metres, or about another
  column?".
- The file's own noun for its rows names the rows, not a value: "Expeditions"
  in expeditions.csv no longer matches trekking agencies named "...
  Expeditions", and "what share of members had success" in members.csv counts
  all 76,519 members (38.2%), not the 9 whose role is "Member", which a
  question still reaches by typing "Member". In a folder, the file whose column the claim names by
  what it measures is chosen over one read from a lesser word, so the claim
  above is verified against `expeditions.csv`'s `member_deaths` (788).
- One repeated id is not repeated measurement: the warning that an average
  weights each unit by its rows needs 1.5 rows per unit on average.
- Words that a column's name shortens word by word name it: "life
  expectancy" names Gapminder's `lifeExp`, and the answer says so ('Read
  "life expectancy" as lifeExp'). The question was refused.
- A camelCase column name reads as its words, and a claim's unit can hold a
  slash. "The average fastest lap speed is 202.6 km/h" about Formula 1's
  results was refused ("lap" named `laps`, and "/h" named nothing); it is now
  verified against `fastestLapSpeed` (202.59 over 6,777 rows), with the note
  that only the numbers are compared.
- A table out of date order is no longer split into logs. Formula 1's races,
  listed season by season from 2009 back to 1950, were read as 8 logs one
  after another and given a Block column; a file with more steps back than 24
  and 1% of its rows is now left as it is.
- A year in a question is the year even when it also appears inside a longer
  text value. "how many races in 2020" on Formula 1's races table was refused
  because 2020 is part of 17 race web addresses; it now counts the 17 races
  dated 2020.
- A correlation over rows that come several to a subject counts each subject
  once. "does Reaction increase with Days" on the sleepstudy table (18
  people, 10 days each) used to be refused; it now gives the average of the
  18 people's own slopes, 10.47 per day (95% CI 7.21 to 13.73, the same
  standard error as a random-slope mixed model), with the repeated-measures
  r of 0.72. A column fixed within each subject is correlated over the
  subjects' means.
- A correlation answer says when the relation inside the groups of a
  grouping differs from the pooled one, as the brief does: over all penguins
  bill length and depth fall together (r = -0.24), within each species they
  rise together (0.39 to 0.65). The sentence also reaches a claim check's
  warnings.
- "average order value" and "average payment value per order" total each
  order first and average over orders (160.99 on the Olist payments table),
  and give the row-by-row average beside it (154.10). A claim that matches
  the row average is told so.
- Elapsed time counted within each subject (`Days`) is a time axis, not a
  running total to set aside.

- A comparison counts each subject once when the subject column is numbered
  with a sign (`subject#`, `Patient #`): `#` is now a word of its own, so such
  a column is read as the unit. On the UCI Parkinson's telemonitoring table,
  "does Jitter(Abs) differ by sex" compared 5,875 recordings as if each were a
  separate person (p = 2e-151); it now compares the 42 people (Welch's p =
  0.14 on their medians) and says the two tests disagree.
- `.data` and `.dat` files are read, as tab-separated text when the header
  splits on tabs and as comma-separated text otherwise (the UCI repository's
  `.data` files).
- Lines above the header are skipped: a data portal's "Data Source" and "Last
  Updated Date" lines as well as a report's title. A World Bank indicator file
  failed to read.
- `\N`, the null MySQL and PostgreSQL write when they export a table, is read
  as missing in every column. Formula 1's `fastestLap` and `fastestLapSpeed`,
  three quarters `\N`, stayed text.
- `NA` in a text column of words (`male`, `female`, `NA`) is read as missing,
  with a note; among codes (`EU`, `APAC`, `NA`) it stays a code. Penguins'
  `sex` had an "NA" group of 11.
- A year column shared by many rows (three years across a table sorted by
  species) no longer splits the file into "logs" with a made-up `Block`
  column.
- A comparison's rows used leave out rows with no group: "compare body mass
  between sex" rests on 333 rows, not 342.
- The data checks no longer call a share's 100 (the World row's share of world
  emissions) a cap or a missing-value code, and no longer pair a share, a rate
  or a cumulative total with the total it shares a word with as part and
  whole (`share_global_co2` beside `co2`).
- Two to four words whose initials spell a column in capitals name it, and the
  answer says so: "home runs" reads as `HR`.
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
