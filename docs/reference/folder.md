# Reading a folder of tables

`ada.scan(folder)` and `ada scan FOLDER` read every table in a folder, propose
the keys that join them, and list what a person should confirm before any
figure combines two files. Given one file, `ada scan data.csv` asks the same
questions about it alone. Code: `ada/folder.py`, and `ada/joins.py` for
answers that join two files.

```python
import ada

folder = ada.scan("data/")
folder.files        # one FileProfile per table
folder.links        # proposed joins, with how well the keys match both ways
folder.questions    # what to confirm, grouped by topic
folder.to_dict()    # all of it as JSON-safe data
```

```bash
ada scan data/
ada scan data/ --json
ada scan data/ --template answers.yml     # questions with ADA's proposed answers
ada scan data/ --answers answers.yml      # read the confirmed answers back
```

The questions are the intake before an analysis: what it should find out,
what a row is, units, which links to trust and the decisions ADA would
otherwise take by default. The answers file they go into is read back by `ada
scan`, `ada ask` and `ada claim` (`--answers`). A question or claim about a
folder is answered from the file it fits, or from two files joined along a
link (see [Answers across files](#answers-across-files)).

## Each file

Every file ADA can read (`.csv`, `.tsv`, `.txt`, `.data`, `.dat`, `.xlsx`,
`.xlsm`; the first sheet of a workbook) is analysed in full, with no row limit.
Its profile gives the rows and columns, the file's checksum, the measure, date,
grouping and identifier ADA picked, the unit of repeated rows (a subject, a
site; `compare.entity_column`), the first notes, and why a file could not be
read. Two more fields:

| Field | Meaning |
|---|---|
| `row_noun` | The plural the file's name gives its rows: "flights" for flights.csv |
| `row_key` | The fewest key columns (up to three) that tell every row apart, when no column is missing: `(origin, time_hour)` in weather.csv |

**Codebooks.** A file with a column of names (`column`, `variable`, `field`)
and a column of descriptions or units, whose names cover half or more of
another file's columns, describes that file (`describes`). The units it gives
are added to the described file's profile (`units`), so ADA does not ask for
them, and a codebook is never joined.

## Links

A link is proposed from the values. For each pair of files, ADA reads the
columns that could be keys: text, dates, and whole numbers named like an
identifier or a year (`playerID`, `customer_zip_code_prefix`, `yearID`).
Decimal readings and row-index columns (`rownames`, `Unnamed: 0`) are never
keys. Each candidate is measured both ways:

| Field | Meaning |
|---|---|
| `coverage` | Share of the first column's distinct values found in the second |
| `coverage_back` | Share of the second column's distinct values found in the first |
| `unmatched_rows`, `unmatched_values` | Rows of the first file, and distinct values, with no match |
| `kind` | `many-to-one`, `one-to-one` or `many-to-many`, from whether each side repeats |
| `fan_out` | Most rows the second file holds for one value; above 1, a join repeats rows |
| `named` | Whether the names say the columns hold the same key |
| `needs_confirmation`, `reasons` | Set when the names differ, some keys don't match, or the second side repeats |

Names match when they are the same once case, underscores and camelCase are
set aside (`playerID`), when they share two or more words and differ by at most
two (`customer_zip_code_prefix` and `geolocation_zip_code_prefix`), or when a
bare `id` belongs to the file the other column is named for. Two bare `id`
columns (or `key`, `uuid`) are each their own file's row id, so they link only
when each holds 90% or more of the other's values, as two halves of one table
do: Lego's `colors.id` (1 to 217) sits inside `inventories.id` (1 to 33,000)
and is not a link. Columns with
different names are linked only when the second is a key (one row per value),
holds 20 or more values, and 90% or more of the first column's values are
found in it; with matching names, 50% is enough, flagged to confirm. Whole
numbers written in digits (row numbers, counters, positions) overlap by
chance, so they link only under matching names: the grid positions 1 to 34 in
a Formula 1 results table sit inside every id column of the other tables.

**Composite keys.** When no single column of a pair of files is a key of the
second, ADA tries two or three columns named alike in both that together tell
the second file's rows apart: Lahman's `Batting.(yearID, teamID)` against
`Teams.(yearID, teamID)`, nycflights13's `flights.(origin, time_hour)` against
`weather`.

**What is left out.** For each column, only the best link is kept: a key on
the other side first, then matching names, then coverage. A one-to-many link
is the many-to-one read backwards and is not listed twice. A many-to-many link
is dropped when its columns already belong to a key link of either file
(`teamID` repeats in Teams because the key is `(yearID, teamID)`), when it
holds fewer than 20 values, or when its name holds no key word (`id`, `code`,
`zip`, `number` and the others in `IDENTIFIER_NAME_WORDS`): two tables that
both list nationalities share values but are not joined by them. One that
stays (customers' zip prefix against a table of locations listing each prefix
up to 1,146 times) says how many rows a join would repeat.

**Same length, no key.** Two tables with the same number of rows and no link
between them get a warning: they may describe the same records, and joining
them by row order is only safe once a person confirms the order.

## Questions

Each question has a stable `key`, ADA's proposed answer when it has one, and
the answer given, if any. They are grouped by topic:

| Topic | Asked when | Proposed answer |
|---|---|---|
| `intent` | Always, first: what the analysis should find out, and about which rows | none |
| `rows` | For each table: what one row stands for | The file name's noun ("one flight"), or the row key |
| `units` | For the measure, when it holds decimals and neither its name nor a codebook gives its unit | none |
| `repeated` | When a key with 20 or more values usually has several rows each | Each counts once in tests and correlations |
| `links` | For each link that needs confirming | "yes" for a covered key link, none otherwise |
| `decisions` | For each table with World, region or Total rows (`decide:FILE:aggregate_rows`, and `decide:FILE:total_from` when it has a total row), and for the folder when files can be joined (`decide:join_links`) | ADA's default: `leave_out`, `total_row`, `sure` |

The `intent` answer is kept for whoever reads the answers file: a person, or
an agent that uses it to pick the decisions. ADA does not interpret it.

Whole-number measures are not asked about: they are mostly counts, which have
no unit.

**Keeping answers.** `answers_template()` and `--template FILE` write every
question with its proposed or given answer to a YAML file, keyed by the
question's `key`. Edit the answers and pass the file back (`answers=` or
`--answers`): an answered question carries its answer and is no longer open
(`open_questions()`), a link answered "yes" is `confirmed`, and one answered
"no" is dropped. `ada ask` and `ada claim` take the same file with
`--answers` (`answers=` in Python) and use its `decide:` answers; a
`--decide` choice wins over them. See [Decisions](decisions.md).

## Answers across files

`ada.ask_folder(folder, question)`, `ada ask FOLDER "question"` and `ada
claim FOLDER "claim"` first try each file. When exactly one gives an answer,
it is used, and the first assumption names it ("Answered from Batting.csv,
the only file in lahman whose columns fit the question."). When several
give different answers, the question is refused, or the claim is
`ambiguous_intent`, naming them.

When no file answers it, two files are joined along a link and tried the
same way: "Barry Bonds hit 73 home runs in 2001." needs the names in
`People.csv` and the home runs in `Batting.csv`. The join:

- runs only along a `many-to-one` or `one-to-one` link, so no row of the
  first file is repeated, and only along links the decision `join_links`
  allows: by default (`sure`) those that need no confirmation and those the
  answers file confirms; `all` allows every such link, `none` none
- keeps every row of the first file (a left join); a row with no match gets
  empty columns from the second, and the note counts them
- matches keys as text, as the scan compares them, so 7 and 7.0 match
- leaves out the second file's dates and date parts (a birth date, a birth
  year) when the first file has dates of its own, so a year in the question
  still means the first file's; a first file with none takes them (a
  result's race date). A clock time read onto one day ("1:27.452", a fastest
  lap) is not a date. A birth or death date is always left out: it
  describes a person, not the row
- renames a column of the second file that the first also has: `name
  (People)`

When no pair answers it either, each file is joined to every file it links
to, and to the files those link to in turn, along the same kind of links:
"Lewis Hamilton scored 413 points in 2019" needs the names in `drivers.csv`,
the points in `results.csv` and the year in `races.csv`. The note lists each
join.

Each file is read once while it is unchanged (`ada.api.analyze_file`, keyed
on its path, modification time and size), so a claim about a folder reads a
file for its own check, for the scan and for each join only once.

The answer says which join it used ("Joined Batting.csv to People.csv on
Batting.playerID = People.playerID (many-to-one; 100.0% of keys found, 0 rows
unmatched).") and lists the `join_links` decision. Its source is the two files
joined, with no single sha256.

A person is found by first and last name in columns of their own
(`nameFirst` and `nameLast`, `forename` and `surname`, `first_name` and
`last_name`) even when there are too many names for the value index: "Barry
Bonds" is the rows with `nameFirst` Barry and `nameLast` Bonds.

## The JSON record

`FolderScan.to_dict()` and `ada scan --json` give `schema_version` (1), the
`folder`, `files` (each profile), `links` (each with `from` and `to` written as
`file.column` or `file.(a, b)`), `questions` grouped by topic, and `warnings`.

## Not covered yet

- A join covers two files; a figure that needs three (results, drivers and
  races) is not computed.
- Links are looked for between two files at a time; a key that runs through a
  third file is reported as two links.
