# Checking numbers on real public data

This folder holds the evaluation behind `ada claim`, `ada ask --json` and
`ada scan`: 73 cases on 20 public datasets, the expected value of each worked
out separately with pandas, SciPy and statsmodels (never with ADA), and the
script that runs ADA on them and scores the automatic checks.

| File | Contents |
|---|---|
| `cases.yml` | The datasets (source, licence, split) and the cases: what to read, ask or check, and what the answer must be |
| `run.py` | Runs a checkout of ADA on the cases and saves what it returned with the automatic checks |
| `reference_*.py` | The independent calculations the expected values come from |
| `datasets.sha256` | The checksum of every data file used |

## Datasets

| Dataset | Split | Files | Source | What it tests |
|---|---|---|---|---|
| Palmer penguins | dev | 2 | [palmerpenguins](https://github.com/allisonhorst/palmerpenguins) (CC0) | `NA` as missing, a pooled correlation that reverses within species |
| nycflights13 | dev | 5 | [Rdatasets](https://vincentarelbundock.github.io/Rdatasets/) (CC0) | 336,776 rows, ambiguous columns, keys with unmatched values |
| Lahman baseball | dev | 4 | [Rdatasets](https://vincentarelbundock.github.io/Rdatasets/) (CC BY-SA 3.0) | composite keys, repeated stints, abbreviations |
| sleepstudy | dev | 1 | [Rdatasets](https://vincentarelbundock.github.io/Rdatasets/) | 18 subjects over 10 days |
| OWID CO2 | dev | 2 | [Our World in Data](https://github.com/owid/co2-data) (CC BY) | aggregate rows, a codebook, share columns |
| Parkinson's telemonitoring | dev | 1 | [UCI](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring) (CC BY 4.0) | 5,875 recordings of 42 people, a `.data` file |
| Olist e-commerce | dev (holdout until batch 1) | 9 | [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (CC BY-NC-SA 4.0) | hashed keys, per-order averages, zip-code fan-out |
| Formula 1 (Ergast) | dev (holdout until batch 6) | 7 | [TidyTuesday 2021-09-07](https://github.com/rfordatascience/tidytuesday/tree/main/data/2021/2021-09-07) (CC BY-NC-SA 3.0) | `\N` as missing, whole-number ids, an unsorted date column |
| World Bank life expectancy | dev (holdout until batch 6) | 3 | [World Bank WDI](https://data.worldbank.org/indicator/SP.DYN.LE00.IN) (CC BY 4.0) | metadata lines above the header, aggregate rows |
| Gapminder | dev (holdout until batch 7) | 1 | [gapminder](https://github.com/jennybc/gapminder) | camelCase names, weighted and unweighted means |
| Lego (Rebrickable) | dev (holdout until batch 7) | 7 | [TidyTuesday 2022-09-06](https://github.com/rfordatascience/tidytuesday/tree/main/data/2022/2022-09-06) | a bare `id` in every file |
| Himalayan expeditions | dev (holdout until batch 8) | 3 | [TidyTuesday 2020-09-22](https://github.com/rfordatascience/tidytuesday/tree/main/data/2020/2020-09-22) | an id that names two expeditions, a guessed column |
| Titanic | dev (holdout until batch 8) | 1 | R datasets (public domain) | a table of counts, ordinal labels |
| ChickWeight | holdout | 1 | R datasets (public domain) | repeated measures, a day filter |
| Bike sharing | holdout | 2 | [UCI](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) (CC BY 4.0) | a total that is the sum of two columns |
| Deforestation | holdout | 5 | [TidyTuesday 2021-04-06](https://github.com/rfordatascience/tidytuesday/tree/main/data/2021/2021-04-06) (CC BY 4.0) | composite keys on country and year, region rows |
| mtcars | holdout | 1 | R datasets (public domain) | abbreviated names (`mpg`, `hp`, `wt`) |
| Chocolate ratings | holdout | 1 | [TidyTuesday 2022-01-18](https://github.com/rfordatascience/tidytuesday/tree/main/data/2022/2022-01-18) | percentages written as text, a year column |
| HairEyeColor | holdout | 1 | R datasets (public domain) | a table of counts no fix looked at |
| Olympics | holdout | 2 | [TidyTuesday 2021-07-27](https://github.com/rfordatascience/tidytuesday/tree/main/data/2021/2021-07-27) (CC0) | 271,116 rows of 135,571 athletes, a region lookup |

No data was subset or synthesised; the Lego files were downloaded as
`.csv.gz` and decompressed, and both forms are in `datasets.sha256`. The data
files are not in this repository; fetch them from the sources and check them
against `datasets.sha256`.

**Dev and holdout.** Dev cases may be used to find and fix problems; holdout
cases are only scored. Whenever a holdout failure led to a fix, its dataset
moved to dev and fresh datasets joined the holdout, their cases written from
a `reference_*.py` script before ADA read them. That happened four times:
Olist (a crash), then Formula 1 and the World Bank file, then Lego and
Gapminder, then the Himalayan folder and Titanic. The holdout scored below is
the last one: two original datasets and five fresh ones.

## Results

Each case passes when every automatic check passes (a value within 0.5%, the
expected status and reason, the expected links). A share may be given as a
fraction or a percentage. Criteria that need judgement (`must:` in
`cases.yml`) are not scored here. ADA 0.5.0 and the current version are
scored with the same harness on the same cases.

| Split | Category | Cases | ADA 0.5.0 | Now |
|---|---|---|---|---|
| dev | reading | 5 | 1 | 5 |
| dev | interpretation | 11 | 1 | 9 |
| dev | calculation | 6 | 2 | 5 |
| dev | design | 6 | 2 | 4 |
| dev | evidence (claims) | 21 | 0 | 11 |
| dev | refusal | 2 | 0 | 2 |
| dev | all | 51 | 6 | 36 |
| holdout | reading | 1 | 0 | 1 |
| holdout | interpretation | 6 | 0 | 2 |
| holdout | calculation | 3 | 1 | 1 |
| holdout | design | 2 | 2 | 2 |
| holdout | evidence (claims) | 10 | 0 | 6 |
| holdout | all | 22 | 3 | 12 |

**Claims.** Of 35 claims, 26 are true and 7 false by the reference values (2
expect a clarifying question). Now: no wrong verdicts, meaning no true claim
comes back `failed_check` and no false claim comes back `verified`. 13 of the
26 true claims are verified, and 6 of the 7 false ones are caught as
`failed_check`; the rest come back unverified with a reason. ADA 0.5.0 had
no claim checking. During the work two wrong verdicts appeared and were
fixed. One read "member deaths" as the heights at which climbers died. The
other compared "52.9% of the people are female" with the share of rows in a
table of counts (50%). It was a fresh holdout case, and the fix for tables
of counts, made on Titanic, verified it without looking at it.

**Links between files.** Over the 11 folder cases ADA proposes 56 links.
Every link the cases expect is among them except three composite ones:
Formula 1's pit stops to results on race and driver (which repeats rows),
Himalayan members to expeditions on id and year, and Brazil's forest losses
to forest area on country and year. Of the 25 links the cases did not list, 17 are real (Olist's zip
prefixes, Formula 1's pit stops), 6 join the deforestation files on `code`
alone, and 2 are wrong (World Bank income groups to country names, Formula
1 car numbers). All 8 of those carry a reason to confirm. Before these
changes the same scan proposed 98 links over 9 folders, 57 of them wrong,
mostly whole-number ids that overlapped by chance.

**What still fails**, by kind:

- **Phrasing** (most misses): "horsepower" for `hp`, "rentals" for `cnt`,
  "released", "won", "lost ... to pasture", slopes ("rises by 10.5 per day"),
  "what share have blue eyes" (read as the Eye column, counted), and claims
  with several numbers ("4 cylinders ... 26.7 mpg").
- **Joins**: a number that needs two files (Barry Bonds's home runs, Lewis
  Hamilton's points) is not computed.
- **Aggregate rows**: World and region rows among countries (OWID's 2022
  total).
- **Columns**: a year column that is not read as a date (chocolate's
  `review_date`), and a value found in two columns ("U.S.A." as company
  location and bean origin), which is asked about rather than guessed.

The holdout improves less than dev (3 to 12 of 22 against 6 to 36 of 51):
the fixes are general, but unseen phrasings and layouts still defeat the
reader often. The misses come back unverified or refused with a reason
rather than wrong.

## Running it

```bash
python -I run.py /path/to/ada-checkout /path/to/dev-data /path/to/holdout-data out.json --split dev
python -I reference_dev.py /path/to/dev-data
```

Each dataset is a folder named as in `cases.yml` (`penguins`, `nycflights13`,
...) under the dev or holdout root. `--split all` runs both, and `--only`
takes a comma-separated list of case ids.
