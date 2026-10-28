# Decisions

Some steps of an answer can go more than one way: whether a World row counts
with the countries, whether a total is read from the table's own total row.
ADA takes each step the same way every time (the default), says which way it
took and why, and lets the caller pick another option. With no one to ask,
the defaults give the same answer on every run; a person, or a coding agent
that knows the person's goal, can choose instead.

The code is in `ada/decisions.py`; the rows that add up others are found in
`ada/aggregates.py`.

## The decisions

| Key | Options (default first) | When it arises | Why the default |
|---|---|---|---|
| `aggregate_rows` | `leave_out`, `keep` | The table has rows that add up other rows, and some are in the question's scope | Counting World or Africa with the countries counts some values twice |
| `total_from` | `total_row`, `sum_of_parts` | A total (`sum`) over the scope, with `aggregate_rows` left out and one total row (World, Total) in the scope | The table gives that row as the total of the others, and it can hold what no other row does |
| `total_levels` | `use_totals`, `all_rows` | A cross-tabulated table where two or more columns each hold a total level ("Both sexes", "All Ages"), or a column of geographic levels (area, county, country) | The table gives every count under each level and again under the total, so adding up all the rows counts each one once per level |
| `join_links` | `sure`, `all`, `none` | A question or claim about a folder that no single file answers | A link that needs confirmation (names that differ, keys that do not all match) may join rows that do not belong together; see [Answers across files](folder.md#answers-across-files) |

A decision that does not arise is not listed, and a choice passed for it
changes nothing: a table with no World row gives the same answer with
`aggregate_rows=keep`.

## Rows that add up others

`aggregate_rows(frame)` looks for them by their labels, never by guessing
from the numbers:

- In a text column whose name names places (country, entity, region,
  location, area, state and the like), a row is an aggregate when its label
  is a total (World, Total, Global, All countries), a continent or region as
  UN, World Bank, FAO and OWID tables write them (Africa, South Asia, "Asia,
  Central", "Europe & Central Asia", EU-27), or a group (labels with income,
  countries, excluding, OECD, IDA, dividend, small states, developing, world
  or total in them). A parenthesis is ignored for the name lists: "Africa
  (GCP)" is Africa.
- A column of codes (`USA`, `OWID_WRL`, `IDA`) is not read as names.
- A group known only by its initials (IDA, IBRD, HIPC, OECD) is an
  aggregate only beside other aggregate rows, such as World or an income
  group: "IDA" among US counties is Ida County.
- In any other text column, only a total label counts (Total, Grand total,
  Sum, Overall), and only when those rows add up to the other rows, within
  1%, in some numeric column. A theme called "World" among Lego themes is a
  theme.

The aggregates must be fewer than half of the column's labels and leave at
least two others, so a column of continents alone has none, and a table of
Kenya and World has nothing to leave out. When several columns qualify, the
one with the most aggregate labels is used.

On the OWID CO2 file this finds World and 28 regions and income groups
among 254 entities. The 225 rows left for 2022 add up to 37,527.774 million
tonnes; the World row says 37,527.773.

## What each option does

With `aggregate_rows=leave_out`, the rows are left out of the scope before the
answer: averages, rankings, counts of countries and comparisons cover the
other rows only. The answer says so ("Left out 29 rows that add up other rows
(World, Africa, Africa (GCP) and 26 others).") and `evidence.left_out` lists
them with their count. A total over the whole scope is read from the total
row when there is one (`total_from=total_row`), with what the other rows add
up to beside it: "Read the total from the World row; the other country values
add up to 37,528." With `total_from=sum_of_parts` it adds up the other rows.

With `aggregate_rows=keep`, every row counts as its own value, as in the file.

A question that picks rows of that column itself is about them, and no
decision is taken: "co2 of World in 2022" gives the World row.

## Total levels of a cross-tabulated table

A published table of counts often gives every count under each level of a
column and again under its total: CDC's injury deaths under each sex and
"Both sexes", each age band and "All Ages", each race and "All races".
`total_levels(frame)` finds such columns: one of a column's 3 to 60 labels
(up to 150 when one reads as a total) reads as a total ("All ...", "Total ...",
"Both sexes", "Both genders", "... total", "WHOLE COUNTRY", but not "All
Other Transport"), and its rows add up to at least the other levels'
rows in some count, and to no more than half as much again (groups no level
names, such as other races, add to the total alone). Two or more columns must
qualify; one total row alone is `aggregate_rows`' to find.

The words every label of a column shares are set aside first, so "Age of
mother, all ages" reads as "all ages". When some counts are blank (a total
withheld where its parts are given), the total and its parts are compared
cell by cell, matched on the other columns, over the cells that give them; a
code beside the column (a DGUID for GEO) names the same level and marks no
cell. A label with no word for a total is the total when its counts are the
others' added up, within 2%, in nine cells of ten: "Canada, place of
residence of mother" over the provinces. Counts beside percentages of them
("Number of live births", "Percentage of live births") are compared within
the counts. When the other levels overlap, so that they add up to more than
the total (cancer sites that include each other; Statistics Finland's areas
and causes of death, each with subtotals under the whole), the label that
reads as a total qualifies when its count is at least every other level's in
the cells holding nineteen twentieths of the counts.

A column of geographic levels qualifies alone: each row is at one level (a
salmon table gives each month's losses by area, again by county, and again
for the whole country), every label is a level (area, county, region,
municipality, country, national and the like), one of them is the whole,
and the whole's rows add up to the rows of one level of parts, within 2%, in
some count.

With `total_levels=use_totals`, a column the question neither filters nor
groups by is read at its total level, and a column it groups by leaves its
total level out of the groups: "total Deaths for Injury mechanism Poisoning
in 2016" reads the one row with Both sexes, All Ages, All races and All
Intentions, 68,995 deaths, where adding up every Poisoning row gives 1.1
million. A level that no row the question picks holds is left alone:
"total losses in Vestland" keeps Vestland's county rows rather than looking
for Vestland at the country level. With `total_levels=all_rows`, every row
counts, as in the file.

Each row of such a table is a cell of counts, so a count adds up the table's
one column of counts (`VALUE` in a CSO or StatCan table) rather than counting
rows, and a change or a total adds it up too unless the question says
average. A share of a level is of that column's total level: "people aged 25
to 44 made up half of immigrants in 2024" is the VALUE of 25 - 44 years over
the VALUE of All ages, with the other columns at what the question picks. A
column whose values say what kind of figure a row gives ("Number of live
births", "Percentage of live births") is read at the counts. No share is
given when two level columns are picked ("aged 15 to 19 in Nunavut": either
could be the part), and the answer is a refusal with the reason when
another column still holds several values in the rows (immigrants,
emigrants and net migration would be added up), when a count is missing or
below zero, or when the unit column gives a rate or more than one unit.

## Choosing an option

```python
analysis = ada.analyze("owid-co2-data.csv")
analysis.ask("total co2 in 2022", decisions={"total_from": "sum_of_parts"})
ada.check_claim("forest.csv", "The average net forest conversion was -76,931 hectares.",
                decisions={"aggregate_rows": "keep"})
```

```bash
ada ask owid-co2-data.csv "average co2 in 2022" --decide aggregate_rows=keep
ada claim forest.csv "The average net forest conversion was -76,931 hectares." --decide aggregate_rows=keep
```

`--decide` can be repeated. An unknown key or option raises `UnknownDecision`
in Python and exits with 2 on the command line, naming what is allowed.

## The answers file

The questions `ada scan PATH --template answers.yml` writes include the
decisions, each with ADA's default as the proposed answer:

```yaml
# decisions
"decide:owid-co2-data.csv:aggregate_rows": "leave_out"
"decide:owid-co2-data.csv:total_from": "total_row"
"decide:join_links": "sure"
```

`ada ask ... --answers answers.yml` and `ada claim ... --answers answers.yml`
(`answers=` in Python) use them: `decide:KEY` for every file, `decide:FILE:KEY`
for one file, which wins, and `--decide` over both. A wrong option in the file
is refused the same way.

## In the record

Each answer and claim check lists the decisions it took: `decisions` in the
JSON, and a `decides:` line each in the text.

| Key | Contents |
|---|---|
| `key` | `aggregate_rows`, `total_from` |
| `question` | The decision as a question: "Leave out the 29 rows that add up other rows (World, Africa, Africa (GCP) and 26 others)?" |
| `options` | Every option, the default first |
| `default`, `chosen` | ADA's default and the option used |
| `decided_by` | `default`, or `caller` when the caller picked another option |
| `reason` | Why the default is the default |

`reading.left_out_values` names the values whose rows the answer left out.

## Claims

A claim is checked with the decisions taken by default. When it disagrees,
each other option of each decision is tried; when one gives a number the claim
stands for, the claim is `ambiguous_intent`, with both readings as
`alternatives` (`aggregate_rows=leave_out` and `aggregate_rows=keep`) and the
clarification "Leave out the 4 rows that add up other rows (World)? Choose
aggregate_rows leave_out or keep." The data cannot say which way the claim's
author meant; the person can, with `--decide`. A choice the caller made is
not tried the other way.

## For coding agents

The skill tells an agent to start with `ada scan PATH --template
answers.yml` and ask the person its questions: what the analysis should find
out, what a row is, and each decision. The agent writes the answers into the
file and passes `--answers` to every `ada ask` and `ada claim`. ADA gives the
skeleton (the questions, the options, a default for each); the agent's
reasoning about the person's goal picks the options. On every answer the agent
reads `decisions`: when the goal settles one ("compare countries" leaves the
World row out; "the file's own column average" keeps it), it passes that
option with `--decide` and says so; when it does not, it asks the person the
decision's `question`. Without an agent or a person, the defaults stand and
the answer says which they were.

## Years

A year in a question ("in 2022") scopes the rows by the date column. A table
with no date column but one numeric column named for years (`year`, `yr`)
is scoped by that column instead: OWID's years start in 1750, before the
range ADA reads as dates, and "total co2 in 1751" reads `year = 1751`. A
month still needs a date column.
