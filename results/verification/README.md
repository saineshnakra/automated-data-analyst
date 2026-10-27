# Checking numbers on real public data

This folder holds the evaluation behind `ada claim`, `ada ask --json` and
`ada scan`: 46 cases on 13 public datasets, the expected value of each worked
out separately with pandas, SciPy and statsmodels (never with ADA), and the
script that runs ADA on them and scores the numeric checks.

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
| Olist e-commerce | dev (was holdout) | 9 | [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (CC BY-NC-SA 4.0) | hashed keys, per-order averages, zip-code fan-out |
| Gapminder | holdout | 1 | [gapminder](https://github.com/jennybc/gapminder) | weighted and unweighted means, camelCase names |
| ChickWeight | holdout | 1 | R datasets (public domain) | repeated measures, a day filter |
| Titanic | holdout | 1 | R datasets (public domain) | a frequency-weighted table |
| World Bank life expectancy | holdout | 3 | [World Bank WDI](https://data.worldbank.org/indicator/SP.DYN.LE00.IN) (CC BY 4.0) | metadata lines above the header, aggregate rows |
| Bike sharing | holdout | 2 | [UCI](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) (CC BY 4.0) | a total that is the sum of two columns |
| Formula 1 (Ergast) | holdout (fresh) | 7 | [TidyTuesday 2021-09-07](https://github.com/rfordatascience/tidytuesday/tree/main/data/2021/2021-09-07) (CC BY-NC-SA 3.0) | `\N` as missing, shared drives, many links |

No data was subset or synthesised. The data files are not in this repository;
fetch them from the sources and check them against `datasets.sha256`.

**Dev and holdout.** Dev cases may be used to find and fix problems; holdout
cases are only scored. When a holdout failure led to a fix (a crash on the
Olist payments table), Olist moved to dev and a fresh dataset (Formula 1) was
added to the holdout, its cases written from `reference_f1.py` before ADA saw
the data.

## Results

Each case passes when every automatic check passes (a value within 0.5%, the
expected status and reason, the expected links). Criteria that need judgement
(`must:` in `cases.yml`) are not scored here.

| Split | Category | ADA 0.5.0 | Now |
|---|---|---|---|
| dev | reading | 1/3 | 3/3 |
| dev | interpretation | 1/5 | 5/5 |
| dev | calculation | 1/2 | 2/2 |
| dev | design | 1/5 | 3/5 |
| dev | evidence (claims) | 0/9 | 5/9 |
| dev | refusal | 0/2 | 2/2 |
| dev | all | 4/26 | 20/26 |
| holdout | reading | 0/2 | 0/2 |
| holdout | interpretation | 0/4 | 1/4 |
| holdout | calculation | 0/2 | 0/2 |
| holdout | design | 3/3 | 3/3 |
| holdout | evidence (claims) | 0/9 | 1/9 |
| holdout | all | 3/20 | 5/20 |

**Wrong verdicts: 0 of 20 claims.** No claim the data support came back as
`failed_check`, and no claim the data do not support came back `verified`.
Every miss is an unverified claim with a reason. But only 3 of the 14 true
claims are verified: most come back `missing_context` because ADA cannot read
the phrasing ("life expectancy" for a column named `lifeExp`, "rentals" for
`cnt`), needs two files joined (a player's name and his home runs), or needs a
table the reader does not handle yet (metadata lines above the header).

The holdout improves far less than dev: the fixes were general, but the
phrasings and file layouts of unseen data still defeat the reader more often
than not.

## Running it

```bash
python -I run.py /path/to/ada-checkout /path/to/dev-data /path/to/holdout-data out.json --split dev
python -I reference_dev.py /path/to/dev-data
```

Each dataset is a folder named as in `cases.yml` (`penguins`, `nycflights13`,
...) under the dev or holdout root.
