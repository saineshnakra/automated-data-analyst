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

The skill tells an agent to read `decisions` on every answer and check:
when the person's goal settles one ("compare countries" leaves the World row
out; "the file's own column average" keeps it), pass that option with
`--decide` and say so; when it does not, ask the person the decision's
`question`. Without an agent or a person, the defaults stand and the answer
says which they were.

## Years

A year in a question ("in 2022") scopes the rows by the date column. A table
with no date column but one numeric column named for years (`year`, `yr`)
is scoped by that column instead: OWID's years start in 1750, before the
range ADA reads as dates, and "total co2 in 1751" reads `year = 1751`. A
month still needs a date column.
