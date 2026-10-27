# Checking claims

`ada.check_claim` and `ada claim` check a number someone states about a table:
a sentence such as "The average departure delay was 12.64 minutes", written by
a person or by an AI assistant. The result is `verified`, or `unverified` with
one of six reasons. Code: `ada/claims.py`.

This is separate from the other two checks in ADA:

| Command | Checks |
|---|---|
| `ada claim FILE "..."` | One number stated about a table, against that table |
| `ada verify` | That a study's assets rebuild the same from `study.yml` and the data |
| `ada check FILE` | Whether the latest period of a series is unusual (anomaly detection) |

## Using it

```python
import ada

check = ada.check_claim("flights.csv", "The average departure delay was 12.64 minutes.")
check.status       # "verified" or "unverified"
check.reason       # None, or one of the six reasons below
check.to_dict()    # JSON-safe: the claim, its reading, the result, the calculation and the rows
```

```bash
ada claim flights.csv "The average departure delay was 12.64 minutes."
ada claim flights.csv --question "average dep_delay" --value 12.64 --json
```

`data` may be a path, bytes, a DataFrame or an `Analysis`. A check reads every
row of the file: the row limit that `ada.analyze` applies by default does not
apply.

When the reading is already known, pass `question` and `value` instead of a
sentence. A `value` given as a string keeps its decimals (`"12.60"` stands for
12.595 to 12.605); a float is taken at its shortest written form.

`ada claim` exits 0 when the claim is verified, 1 when it fails a check, 2
when the file can't be read, and 3 when it is unverified for any other reason.

## How a claim is read

1. **The number.** ADA takes out the number being claimed. A year ("in
   2013"), an ordinal ("25th percentile") and a value right after a column's
   name ("bill length above 45 mm") stay part of the question. A claim with
   two numbers that could each be the one claimed is not split: it gets
   `unsupported_analysis`, and each number should be sent on its own.
2. **The question.** The rest of the sentence is read as a question to
   `Analysis.ask` (see [Ask ADA](ask-ada.md)). "averaged 5050" becomes
   "average ...", and "There are 344 penguins" becomes "how many penguins",
   where the noun comes from the file's name (`row_noun` in `ada/schema.py`).
3. **The interval.** The number stands for the values its writing implies,
   worked out in decimal arithmetic:

| Written | Stands for |
|---|---|
| `12.64` | 12.635 to 12.645 |
| `13` | 12.5 to 13.5 |
| `about 13`, `roughly 13`, `~13` | 12 to 14: one significant unit either side |
| `about 100` | 90 to 110: never more than a tenth of the number |
| `about 2` | 1.5 to 2.5: never less than the number's own rounding |
| `nearly 100`, `almost 100` | 90 to 100 |
| `just over 50` | above 50, up to 55 |
| `more than 30%`, `over 30%` | above 30 |
| `at least 30` | 30 or more |
| `exactly 3` | 3 |
| `37.5 billion` | 37.45 billion to 37.55 billion |

4. **The comparison.** The computed answer is compared with the interval, in
   the claim's terms: a percent claim is compared with a share or a change in
   percent, and a fraction such as 0.32 with 32%.

## Units and scale

A unit in the column's name is used. `body_mass_g` is in grams, so "5.08 kg"
is compared with the mean divided by 1,000, and the assumption says so. A
claim in a unit that measures something else ("44 minutes" about
`bill_length_mm`) gets `conflicting_evidence`. Units are read only from
plain forms (`g`, `kg`, `mm`, `cm`, `km`, `minutes`, `hours`, `days`, and
their full names); short forms that also mean something else, such as `m`
and `t`, are not read. A unit with a slash ("202.6 km/h") is read as the
claim's unit; one ADA does not convert is noted, and only the numbers are
compared.

A claim that names no column can still name its kind of quantity: by its
unit ("4000 g" is a mass) or by a verb ("weigh", "lasted", or "measure" with
a unit). ADA then tries each column recorded in a unit of that kind. One such
column is used, and the assumptions say so; several give `ambiguous_intent`
with each column's number.

When the column's name gives no unit, ADA does not assume one. A claim that
agrees only if the column were recorded in thousands or millions gets
`missing_context`, with the question "What unit is co2 recorded in?". A claim
whose scale word is part of the unit ("37,528 million tonnes" about a column
recorded in million tonnes) agrees, and the assumption says how it was read.

## Statuses and reasons

`status` is `verified` only when the computed answer lies in the claimed
interval and no check on the analysis failed. Otherwise it is `unverified`
and `reason` is one of:

| Reason | When | Example |
|---|---|---|
| `missing_context` | The claim leaves out something the check needs: a number, a column, a group, a date column, or the unit that decides the scale | "The average is 3.7" (which column?) |
| `ambiguous_intent` | The words fit more than one column, and their answers differ | "The average delay was 13 minutes" with `dep_delay` and `arr_delay` |
| `inaccessible_data` | The data could not be read | A missing or empty file |
| `unsupported_analysis` | ADA has no way to compute what the claim describes, or the claim states several numbers | "The average was 4457 g and 3700 g" |
| `conflicting_evidence` | The claim's unit measures something other than the unit in the column's name | "44 minutes" about `bill_length_mm` |
| `failed_check` | The computed answer is outside the claimed interval, or a check on the analysis failed | "The average body mass is 4,600 g" when the data give 4,457.14 |

Unverified does not mean false. Only `failed_check` with `agreement:
"mismatch"` says the data give a different number. A verified claim is
verified as read: the `reading`, `assumptions` and `warnings` say what was
checked, and nothing beyond it. A check covers one number; an answer with
several numbers in it needs a check for each.

A mismatch is `failed_check` only when the claim names what the column
measures: the last word of its name, leaving out a unit or a counting word
("order value" for `payment_value`, "250 parts" for `num_parts`), or its
initials ("home runs" for `HR`). When the column was read from a lesser word
("deaths" for `death_height_metres`, a height), a mismatch says nothing about
the claim, so it gets `missing_context` with `agreement: "mismatch"` and the
question "Is the number about death_height_metres, or about another column?".

When the claim is unverified for a reason a person can resolve,
`clarification` holds the smallest question that would let the check go
ahead, such as "Which column is meant: dep_delay or arr_delay?".

## Checks on the analysis

Besides the number, a check records what could make the answer misleading:

- **Rows left out.** Each reason rows were left out, with its count: rows
  with the measure missing, cells the data checks flag, exact repeats of
  other rows left out of a test.
- **Repeated units.** When the table's identifier (a subject, a site) has
  several rows, 1.5 or more per unit on average, an average weights each unit
  by its number of rows, which the warnings say (one repeated id among 76,519
  climbers is not repeated measurement); a test or correlation over such rows treats them as
  independent, which fails the check.
- **Cautions from the answer.** Whatever the answer itself says may sway it,
  such as a spread too wide for the mean to be typical, or a pooled
  correlation that runs the other way within the groups of a grouping.
- **The unit of the average.** A claim about a key's average ("The average
  order value is R$154.10") is checked against each order's total averaged
  over orders (160.99). When the claim matches the average taken row by row
  instead, the message says so: the claim counted each payment row, not each
  order.

## The JSON record

`ClaimCheck.to_dict()` and `ada claim --json` give:

| Key | Contents |
|---|---|
| `schema_version` | `1`; bumped when a key changes meaning |
| `claim`, `status`, `reason`, `message` | The claim and the outcome |
| `claimed` | The number: `text`, `value`, `interval` (`null` for an open end), `interval_text`, `qualifier`, `percent`, `scale`, `unit` |
| `question` | The question the claim was read as |
| `agreement` | `match`, `mismatch` or `not_compared` |
| `result` | `value` (the computed number compared, in the claim's terms), `values` per group, `figures` |
| `reading` | The plan that ran: intent, operation, measure, grouping, filters by year or month |
| `calculation` | The operation in words, such as `mean(body_mass_g) · filtered to species = Gentoo` |
| `evidence` | `source` (name, path, `sha256` of the file's bytes, rows, columns), `rows_read`, `rows_in_scope`, `rows_used`, `filters`, `left_out` |
| `assumptions`, `warnings` | How the claim was read, and what may sway it |
| `checks` | Each check run: `name`, `passed`, `detail` |
| `alternatives` | For `ambiguous_intent`: each reading, its question, value, whether it agrees, and its calculation |
| `clarification` | The question to ask, or `null` |

`ada ask --json` gives the same `result`, `reading`, `calculation`,
`evidence`, `assumptions` and `warnings` for a question, with `status`
`answered`; a refused question gives `status: "refused"` with its `code` and
`message`, and exit code 3.

## Claims about a folder

`check_claim` and `ada claim` also take a folder. The claim is checked against
every table in it:

- one file gives a number for it: that file's check is the result, and the
  first assumption says it was the only file whose columns fit
- several give the same verified number: the first is used, and the
  assumption names them all
- several give different numbers: `ambiguous_intent`, with each file as an
  alternative and the question "Which file is the claim about?"
- none does: `missing_context`, naming the key links a scan proposes (see
  [Reading a folder of tables](folder.md)), since the claim may combine two
  files. ADA does not join files, so such a claim stays unverified

## Edge cases

| Input | Result |
|---|---|
| A folder | Checked against each file it holds (see [Claims about a folder](#claims-about-a-folder)) |
| A claim with no number | `missing_context` |
| A claim that names no column ("The average is 3.7") | `missing_context`; ADA does not fall back on the column it picks by default |
| An answer with one number per group | `missing_context`: "Which group is the number about?" |
| A year on a table with no date column | `missing_context`: "Which column holds the date or year?" |
| Too few rows for the calculation (a correlation on fewer than 12 pairs) | `unsupported_analysis`, with the answer's reason |
| A DataFrame | Checked like a file; `source.sha256` and `source.path` are `null` |

## Not covered yet

- Joins across files: a claim that needs two files of a folder is not checked; the answer names the keys that would join them.
- Verbs other than "weigh", "lasted" and "measure", or a verb whose column
  has no unit in its name ("Gentoo penguins weigh 5,076" about a `mass`
  column needs "mass" in the claim).
- Repeated-measures designs where the subject column is not detected as the
  table's identifier.
