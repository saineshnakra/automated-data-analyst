"""Infer the role of each column: measure, date, grouping, identifier or category code.

Roles are guessed from names, values and cardinality together, since none is
reliable alone: "Order Number" is numeric but not a measure, a column of 0s and
1s is a flag rather than a quantity, and "Value" may be either.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from .formatting import normalized_name
from .metrics import ADDITIVE_WORDS, LEVEL_WORDS


@dataclass(frozen=True)
class ColumnRoles:
    date: str | None
    measure: str | None
    dimension: str | None
    identifier: str | None
    #: Numeric columns that can be measured, quantities first and then factors.
    #: Identifiers are left out: a row key is a number but not a quantity.
    numeric: tuple[str, ...]
    dimensions: tuple[str, ...]
    #: "sum" or "mean" when the user has said how the measure combines; None
    #: follows its name. Read it through ``metrics.metric_for``.
    aggregation: str | None = None
    #: Numeric columns that hold category codes rather than quantities (a 0/1
    #: flag, a status code, a set dose). They are listed in ``dimensions`` too.
    factors: tuple[str, ...] = ()
    #: The factors holding only 0 and 1, whose average is the share of rows that are 1.
    flags: tuple[str, ...] = ()
    #: Columns that came close to being the measure, best first. Set only when
    #: the pick was a close call, so the reader can be told it was a judgement.
    measure_alternatives: tuple[str, ...] = ()
    #: "flat"; "long" when one column names what each row's value measures; or
    #: "matrix" when the table is mostly numeric columns of one kind, such as
    #: samples or spectral channels.
    table_shape: str = "flat"
    #: The value of ``dimension`` the brief covers. A long table's rows measure
    #: different things, often in different units, so they are never combined.
    focus: str | None = None
    #: What ADA concluded about the columns, in plain sentences.
    notes: tuple[str, ...] = ()


MEASURE_KEYWORDS = {
    "revenue": 14,
    "mrr": 14,
    "arr": 14,
    "sales": 14,
    "gmv": 14,
    "profit": 13,
    "margin": 12,
    "amount": 11,
    "value": 10,
    "income": 10,
    "spend": 9,
    "cost": 8,
    "expense": 8,
    "price": 7,
    "turnover": 14,
    "total": 9,
    "fee": 7,
    "fare": 7,
    "quantity": 6,
    "units": 6,
    "orders": 6,
    "volume": 6,
    "balance": 6,
    "count": 6,
    "abundance": 6,
    "yield": 6,
    "concentration": 5,
    # What an experiment or a study measures as its result.
    "strength": 5,
    "response": 5,
    "latency": 5,
    "result": 5,
    "outcome": 5,
    "growth": 5,
    "activity": 5,
    "viability": 5,
    "survival": 5,
    "decision": 5,
    "choice": 5,
    "score": 4,
}

DIMENSION_KEYWORDS = {
    "product": 12,
    "category": 12,
    "segment": 12,
    "region": 11,
    "country": 10,
    "state": 9,
    "city": 9,
    "channel": 10,
    "customer": 8,
    "client": 8,
    "team": 7,
    "department": 7,
    # What a study compares: its arms, its outcome classes and its sites.
    "treatment": 10,
    "diagnosis": 10,
    "result": 10,
    "outcome": 10,
    "condition": 9,
    "group": 9,
    "class": 9,
    "label": 9,
    "phenotype": 9,
    "genotype": 9,
    "arm": 9,
    "cohort": 9,
    "station": 7,
    "site": 7,
    "status": 6,
    "stage": 6,
    "phase": 6,
    "type": 5,
    "sex": 5,
    "gender": 5,
    "race": 4,
    "ethnicity": 4,
}

# What a keyword earlier in the name is worth against one at the end.
MODIFIER_WEIGHT = 2 / 3

IDENTIFIER_TOKENS = ("id", "uuid", "key", "code", "number", "invoice", "order")
TIME_PART_TOKENS = ("year", "month", "week", "day", "hour", "minute", "quarter")
# Experimental-design columns are numbered but hold labels, not quantities. Plate
# rows and columns lay out where a sample sat, not what was measured.
DESIGN_TOKENS = frozenset({
    "replicate", "rep", "batch", "plate", "well", "run", "trial", "block", "lane",
    "row", "column", "col",
})
# Conditions recorded beside an outcome, and rarely the outcome itself. Any word
# of the name counts, so "air_temp_c" matches.
COVARIATE_TOKENS = frozenset({"temp", "temperature", "humidity", "wind", "air"})
# Coordinates are continuous numbers, but never a quantity to total or trend.
COORDINATE_TOKENS = frozenset({
    "latitude", "longitude", "lat", "lon", "lng", "easting", "northing", "gps",
    "coordinate", "coordinates",
})
# Head words of what goes into a mix or a dose rather than what comes out: in a
# table of cement, water and slag in kg/m³ and a strength in MPa, the strength
# is the result.
INPUT_WORDS = frozenset({
    "cement", "water", "slag", "ash", "aggregate", "admixture", "superplasticizer",
    "superplasticiser", "plasticizer", "plasticiser", "binder", "sand", "dose", "dosage",
    "ingredient", "additive",
})
# Checks that a study worked (a realism rating, an attention check) rather than
# what it measured.
CHECK_WORDS = frozenset({"realism", "attention", "manipulation"})

# Head words that mark a column as a key rather than a quantity. Such columns stay
# text through reading and cleaning, so an account number 00042 keeps its zeros.
# Whole-word match, so "Paid Amount" is not an id.
IDENTIFIER_NAME_WORDS = frozenset({
    "id", "ids", "code", "codes", "zip", "zipcode", "postal", "postcode", "phone", "telephone",
    "mobile", "tel", "fax", "sku", "number", "no", "#", "key", "uuid", "notation",
})

# Head words that name the unit a row belongs to rather than a quantity: subject 12
# is not more of anything than subject 3, nor station 10002 than station 10001.
# These columns repeat (three readings per participant), so they fail the
# uniqueness test that identifiers must pass.
ENTITY_NAME_WORDS = frozenset({
    "subject", "participant", "patient", "animal", "mouse", "rat", "donor", "individual",
    "specimen", "station", "sta", "stn", "site", "plot",
}) | IDENTIFIER_NAME_WORDS
# The entities that are one subject each. Comparing them one by one is rarely
# the question, so they are a grouping of last resort; sites and stations are not.
INDIVIDUAL_NAME_WORDS = frozenset({
    "subject", "participant", "patient", "animal", "mouse", "rat", "donor", "individual",
    "specimen", "id", "ids", "key", "uuid",
})

# Head words of numeric columns that label groups of rows: status 2 is a group,
# not twice status 1.
LABEL_NAME_WORDS = frozenset({
    "status", "condition", "group", "treatment", "label", "class", "stage", "phase", "sex",
    "gender", "race", "ethnicity", "phenotype", "genotype", "type", "category", "diagnosis",
    "arm", "cohort",
})
# Words that make a column with few distinct values a count or an amount all the
# same: 4, 6 or 8 cylinders, 0 to 5 children. The first set counts anywhere in
# the name ("Number of Rooms"), the second only as the last word. A plural last
# word ("Visits") also marks a count.
COUNT_NAME_WORDS = frozenset({
    "count", "counts", "number", "num", "qty", "quantity", "quantities", "amount", "amounts",
    "total", "totals",
})
QUANTITY_HEAD_WORDS = frozenset({
    "size", "children", "people", "score", "rating", "price", "cost", "fee", "fare",
    "yield", "strength", "response", "latency", "result", "outcome", "growth", "activity",
    "viability", "survival",
})
# Set levels of an experiment: a dose, a concentration or a time point chosen
# in advance and repeated across samples.
DESIGN_LEVEL_WORDS = frozenset({
    "dose", "dosage", "concentration", "conc", "age", "day", "time", "timepoint", "hour",
    "week", "frequency", "freq",
})

# Head words of columns holding calendar years and months. A table with no date
# column but a year (and perhaps a month) is still a timeline; cleaning builds
# its dates from these.
YEAR_NAME_WORDS = frozenset({"year", "years", "yr"})
MONTH_NAME_WORDS = frozenset({"month", "months", "mon"})
# Parts of a date held in their own column. Beside a date column they repeat it,
# so they are not offered as the grouping.
TIME_PART_WORDS = (
    frozenset(TIME_PART_TOKENS) | YEAR_NAME_WORDS | MONTH_NAME_WORDS
    | frozenset({"mo", "da", "dy", "wk", "qtr", "doy", "dow", "weekday"})
)

# Names of a column numbering the rows. Unique whole numbers under such a name
# are a key whatever their order.
ROW_KEY_WORDS = frozenset({
    "seqn", "seq", "index", "idx", "row", "rownum", "record", "serial", "sno", "obs", "unnamed",
})
# Words that make a column a quantity however its values run: a table sorted by
# population still holds populations.
QUANTITY_WORDS = frozenset(MEASURE_KEYWORDS) | ADDITIVE_WORDS | LEVEL_WORDS

# A column naming what each row's value measures, as in a long (tidy) table of
# readings: "determinand.label", "Parameter", "Analyte".
PARAMETER_WORDS = frozenset({
    "determinand", "determinant", "parameter", "analyte", "variable", "test", "component",
    "measure", "measurement", "indicator", "characteristic", "metric", "property",
})
UNIT_WORDS = frozenset({"unit", "units", "uom"})
# Trailing words that say a column holds a name of something, not what it is.
GENERIC_NAME_WORDS = frozenset({
    "label", "labels", "name", "names", "description", "desc", "definition", "title",
})
# Unit-like markers in a unit column's values ("mg/l", "%", "°C").
_UNIT_MARKS = re.compile(r"[/%°µ]|^(?:[mµunk]?g|[mµ]?l|[mck]?m|s|min|h|ppm|ppb|mol|cel|deg[cf]?)$", re.I)

# Numbered column names: "A3", "B164", "sample_12", "1001", "243.8".
_NUMBERED_NAME = re.compile(r"^(\D*?)[\s_.-]*(\d+(?:\.\d+)?)$")

MIN_ROWS_FOR_KEY = 20
# Most levels a numeric code may have: the same cap as a text grouping.
MAX_LABEL_LEVELS = 100
MAX_GROUPING_SHARE = 0.65
MAX_CODE_LEVELS = 12
MAX_DESIGN_LEVELS = 15
MIN_ROWS_PER_LEVEL = 5
# A grouping whose largest group holds more than this splits off a sliver.
LOPSIDED_GROUPING = 0.9
# A column with one value in more than this share of rows carries little to report.
NEARLY_CONSTANT = 0.8
MIN_MATRIX_COLUMNS = 20
MATRIX_SHARE = 0.75
MIN_RUNNING_TOTAL_ROWS = 20
RUNNING_TOTAL_RISING = 0.95
# A running total that resets falls to this share of its last value or less.
RESET_FRACTION = 0.1
RUNNING_TOTAL_WORDS = frozenset({"cumulative", "cum", "running", "accumulated", "ytd", "odometer"})
TOTAL_WORDS = frozenset({"total", "totals"}) | RUNNING_TOTAL_WORDS
INTERVAL_WORDS = frozenset({"interval", "incremental", "increment", "hourly", "daily", "period", "delta"})
# Long tables are split by parameter when its values sit this many times apart.
LONG_SCALE_SPREAD = 10.0
# Runners-up within this many points of the measure's score were close calls.
CLOSE_CALL_POINTS = 3
MAX_ALTERNATIVES = 3
# Rows sampled, evenly, to measure how much information a column holds.
INFORMATION_SAMPLE = 50_000
NAMED_IN_NOTES = 3


# A trailing unit or annotation is not the head noun: "Revenue (USD)" is scored on "Revenue".
_ANNOTATION = re.compile(r"\s*[(\[][^)\]]*[)\]]\s*$")
_UNIT_ANNOTATION = re.compile(r"[(\[]([^)\]]*)[)\]]\s*$")


DATE_NAME_TOKENS = ("date", "time", "timestamp", "created", "updated")


EVENT_DATE_TOKENS = ("order", "transaction", "sale", "invoice", "posting", "created", "booking")
FOLLOW_UP_DATE_TOKENS = ("ship", "delivery", "delivered", "due", "updated", "modified", "closed")


def _date_preference(name: str) -> int:
    words = normalized_name(name).split()
    if any(token in words for token in EVENT_DATE_TOKENS):
        return 2
    if any(token in words for token in FOLLOW_UP_DATE_TOKENS):
        return 0
    return 1


def is_identifier_name(name: str) -> bool:
    """Whether the name marks a key column: its last word is in IDENTIFIER_NAME_WORDS.

    The reader and the cleaner both use this, so they agree on which columns
    (such as "Order No") stay text. "No" alone counts only with a mark ("No.",
    "No #"): a bare "NO" is as likely nitric oxide or a yes/no answer.
    """
    words = _head_words(name)
    if words == ["no"] and not re.search(r"[.#]", str(name)):
        return False
    return bool(words) and words[-1] in IDENTIFIER_NAME_WORDS


def is_design_name(name: str) -> bool:
    """Whether the name marks an experimental-design label such as "Replicate" or "Plate".

    "Position" counts only after a design word ("Well Position"), since on its
    own it is as likely a player's or an employee's role.
    """
    words = _head_words(name)
    if words[-2:-1] and words[-1] == "position":
        return words[-2] in DESIGN_TOKENS
    return bool(words) and words[-1] in DESIGN_TOKENS


def is_entity_name(name: str) -> bool:
    """Whether the name labels the unit a row belongs to, such as "Subject", "Station" or "Patient ID"."""
    words = _head_words(name)
    return bool(words) and words[-1] in ENTITY_NAME_WORDS


def is_year_name(name: str) -> bool:
    """Whether the name marks a column of calendar years, such as "Year" or "survey_yr"."""
    words = _head_words(name)
    return bool(words) and words[-1] in YEAR_NAME_WORDS


def is_month_name(name: str) -> bool:
    """Whether the name marks a column of calendar months, such as "Month"."""
    words = _head_words(name)
    return bool(words) and words[-1] in MONTH_NAME_WORDS


def _named_like_a_date(name: str) -> bool:
    return any(token in normalized_name(name).split() for token in DATE_NAME_TOKENS)


def _head_words(name: str) -> list[str]:
    """Words of a column name, with any trailing annotation and punctuation removed.

    Dots separate words too, so "determinand.unit.label" ends in "label".
    """
    text = normalized_name(_ANNOTATION.sub("", str(name))).replace(".", " ")
    words = (word.strip(",:;") for word in text.split())
    return [word for word in words if word]


def _keyword_score(name: str, keywords: dict[str, int]) -> int:
    """Score a column name against keywords, weighting its last word most.

    The last word says what the column is; earlier words only qualify it.
    "Product Category" scores as a category; "Product Container" gets only the
    reduced MODIFIER_WEIGHT score for "product".
    """
    words = _head_words(name)
    if not words:
        return 0
    head_score = keywords.get(words[-1], 0)
    anywhere = (score for token, score in keywords.items() if token in " ".join(words))
    return max(head_score, int(max(anywhere, default=0) * MODIFIER_WEIGHT))


def looks_like_identifier(name: str, series: pd.Series) -> bool:
    if is_datetime64_any_dtype(series):
        # "Order Date" has an identifier token and unique values, but is not a row key.
        return False
    words = _head_words(name)
    if words and words[-1] in MEASURE_KEYWORDS:
        # The last word decides, as in _keyword_score: "Invoice Amount" is an amount.
        return False
    if not any(token in words for token in IDENTIFIER_TOKENS):
        return False
    if is_numeric_dtype(series) and not is_bool_dtype(series) and bool((series.dropna() % 1 != 0).any()):
        # Keys are whole; a "Reynolds Number" of 2.3e5 with decimals is a measurement.
        return False
    unique_ratio = series.nunique(dropna=True) / max(int(series.notna().sum()), 1)
    return unique_ratio >= 0.8


def _is_row_key(name: str, series: pd.Series, *, first: bool) -> bool:
    """Whether a numeric column holds a different whole number on every row, as a key does.

    Unique whole numbers alone are not enough: forty order totals can all
    differ, and a dose stepped 0, 1, 2 … down the rows is unique too. They must
    also sit under a row-key name ("seqn", "index"), or be the table's first
    column and climb down the file in uneven steps, as respondent numbers with
    gaps do.
    """
    if is_bool_dtype(series) or not is_numeric_dtype(series) or _named_like_a_date(name):
        return False
    words = set(_head_words(name))
    named = bool(words & ROW_KEY_WORDS)
    if not named and (not first or words & QUANTITY_WORDS or words & TIME_PART_WORDS):
        return False
    if len(series) < MIN_ROWS_FOR_KEY or series.isna().any():
        return False
    values = series.to_numpy(dtype=float)
    if not np.all(values % 1 == 0) or pd.unique(values).size != values.size:
        return False
    if named:
        return True
    steps = np.diff(values)
    return float((steps > 0).mean()) >= RUNNING_TOTAL_RISING and np.unique(steps).size > 1


def _is_row_label(name: str, series: pd.Series) -> bool:
    """Whether a text column names each row once, as a taxon or a file name does.

    Sentences are unique too but label nothing, so values of more than a few
    words do not count.
    """
    if is_numeric_dtype(series) or is_datetime64_any_dtype(series) or _named_like_a_date(name):
        return False
    present = series.dropna()
    if len(present) < MIN_ROWS_FOR_KEY or len(present) < 0.9 * len(series):
        return False
    if present.nunique() != len(present):
        return False
    words = present.head(200).astype(str).str.split().str.len()
    return float(words.median()) <= 6


def _identifier_columns(dataframe: pd.DataFrame, columns: list[str]) -> list[str]:
    """Every one of ``columns`` that keys or labels rows, the best first.

    Key names come first ("Order ID"), then unique whole numbers that read as a
    sequence ("seqn"), then a unique text label ("Taxonomy").
    """
    named = [column for column in columns if looks_like_identifier(column, dataframe[column])]
    rest = [column for column in columns if column not in named]
    first = dataframe.columns[0] if len(dataframe.columns) else None
    keys = [
        column for column in rest if _is_row_key(column, dataframe[column], first=column == first)
    ]
    labels = [column for column in rest if column not in keys and _is_row_label(column, dataframe[column])]
    return named + keys + labels


def _places_rows(name: str, series: pd.Series) -> bool:
    """Whether the column places a row rather than measures it.

    A coordinate is a number but never a quantity. Nor are whole numbers named
    as a time part ("Order Month"), a design label ("Replicate") or an entity
    ("Station"); with decimals, a "Reynolds Number" is a measurement after all.
    """
    words = _head_words(name)
    if not words:
        return False
    if set(words) & COORDINATE_TOKENS:
        return True
    named = words[-1] in TIME_PART_WORDS or is_design_name(name) or is_entity_name(name)
    return named and not bool((series.dropna() % 1 != 0).any())


def _plural(word: str) -> bool:
    return len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is"))


def _named_as_quantity(name: str) -> bool:
    """Whether the name marks a count or an amount, whatever its values look like."""
    words = _head_words(name)
    if not words:
        return False
    return bool(set(words) & COUNT_NAME_WORDS) or words[-1] in QUANTITY_HEAD_WORDS or _plural(words[-1])


def _named_as_label(name: str) -> bool:
    """Whether the name marks group labels: a status, a condition, a site or a plate row."""
    words = _head_words(name)
    return bool(words) and (words[-1] in LABEL_NAME_WORDS or is_entity_name(name) or is_design_name(name))


def _factor_kind(name: str, series: pd.Series) -> str | None:
    """Say whether a numeric column holds category codes, and on what evidence.

    "flag" is a 0/1 column. "label" is a column of whole numbers named like a
    label (Status, Group, Station), which may have up to MAX_LABEL_LEVELS
    values. "code" is weaker evidence: two values, or up to MAX_CODE_LEVELS
    whole numbers each repeated many times, under a name that is not a count
    or an amount. None means a quantity.
    """
    if is_bool_dtype(series) or _sample(series).nunique() > MAX_LABEL_LEVELS:
        return None
    levels = int(series.nunique(dropna=True))
    if levels < 2 or levels > MAX_LABEL_LEVELS:
        return None
    values = series.dropna()
    if levels == 2 and set(values.unique()) <= {0, 1}:
        return "flag"
    if not bool((values % 1 == 0).all()):
        return None
    if _named_as_label(name) and len(values) >= 2 * levels:
        return "label"
    if _named_as_quantity(name):
        return None
    if levels == 2 and len(values) >= 3 * levels:
        return "code"
    if levels <= MAX_CODE_LEVELS and len(values) >= MIN_ROWS_PER_LEVEL * levels:
        return "code"
    return None


def _crossed(series: pd.Series, grouping: pd.Series) -> bool:
    """Whether the column takes several values within at least two groups of ``grouping``."""
    spread = series.groupby(grouping, observed=True).nunique()
    return int((spread >= 2).sum()) >= 2


def _is_design_level(name: str, series: pd.Series, groupings: list[pd.Series]) -> bool:
    """Whether the column holds a few set levels, like a dose or a time point, crossed with a grouping.

    Such a column may hold fractions (0.5, 1 and 2.5 mg) or a few more levels
    than a code, so it needs a design word in its name and levels that repeat
    within the groups of another column.
    """
    if not set(_head_words(name)) & DESIGN_LEVEL_WORDS or _named_as_quantity(name):
        return False
    values = series.dropna()
    levels = int(values.nunique())
    if not 3 <= levels <= MAX_DESIGN_LEVELS or len(values) < MIN_ROWS_PER_LEVEL * levels:
        return False
    return any(_crossed(series, grouping) for grouping in groupings)


def _top_share(series: pd.Series) -> float:
    """Share of the most common value among the present ones, judged on an even sample."""
    values = _sample(series).dropna()
    if values.empty:
        return 1.0
    return float(values.value_counts(normalize=True).iloc[0])


def _sample(series: pd.Series) -> pd.Series:
    step = max(1, len(series) // INFORMATION_SAMPLE)
    return series.iloc[::step]


def _information(series: pd.Series) -> float:
    """How much a column says, as the entropy of its values in bits.

    Many distinct, evenly used values score high; a column that is mostly one
    value, or mostly missing (missing counts as one value), scores low.
    """
    shares = _sample(series).value_counts(dropna=False, normalize=True).to_numpy(dtype=float)
    return round(float(-(shares * np.log2(shares)).sum()), 1)


def _unit(name: str) -> str | None:
    match = _UNIT_ANNOTATION.search(str(name))
    return " ".join(match.group(1).lower().split()) if match else None


def _lone_unit(columns: list[str]) -> str | None:
    """The one column whose unit differs from a unit most others share.

    Seven ingredients in kg/m³ and a strength in MPa: the strength is what came out.
    """
    units = {column: _unit(column) for column in columns}
    shared = Counter(unit for unit in units.values() if unit)
    if not shared:
        return None
    common, count = shared.most_common(1)[0]
    others = [column for column, unit in units.items() if unit and unit != common]
    return others[0] if count >= 3 and len(others) == 1 else None


def _reading_order(dataframe: pd.DataFrame, date: str | None, grouping: str | None) -> np.ndarray:
    """Row positions for following each group through time: by group, then date, else as filed."""
    keys = [key for key in (grouping, date) if key]
    if not keys:
        return np.arange(len(dataframe))
    return dataframe[keys].reset_index(drop=True).sort_values(keys, kind="stable").index.to_numpy()


def _running_total(name: str, values: np.ndarray, groups: np.ndarray) -> bool:
    """Whether a column climbs like a gauge or meter reading rather than recording each period.

    ``values`` and the matching group codes are in reading order (see
    ``_reading_order``). Within each group the column almost never falls: it
    rises many times for each fall, and the falls are resets back to near zero.
    Rainfall per hour rises and falls back as often, so it does not qualify,
    and neither does a day of the month. A name such as "Cumulative Cases" is
    enough without a reset.
    """
    words = _head_words(name)
    if words and words[-1] in TIME_PART_WORDS:
        return False
    present = ~np.isnan(values)
    values, groups = values[present], groups[present]
    if len(values) < MIN_RUNNING_TOTAL_ROWS or values.min() < 0:
        return False
    # Each value against the one before it in the same group.
    same = groups[1:] == groups[:-1]
    previous, current = values[:-1][same], values[1:][same]
    rises, falls = int((current > previous).sum()), int((current < previous).sum())
    if not rises or falls > (1 - RUNNING_TOTAL_RISING) * rises:
        return False
    if set(words) & RUNNING_TOTAL_WORDS:
        return True
    return bool(((current < previous) & (current <= RESET_FRACTION * previous)).any())


def _sibling_interval(counter: str, candidates: list[str]) -> str | None:
    """The column recording each period's share of a running total: "Interval Rain" for "Total Rain"."""
    stem = set(_head_words(counter)) - TOTAL_WORDS
    for column in candidates:
        words = set(_head_words(column))
        if column != counter and words & INTERVAL_WORDS and words - INTERVAL_WORDS == stem:
            return column
    return None


def _matrix_columns(numeric: list[str]) -> list[str]:
    """The numbered numeric columns of a table that is mostly a matrix of them, or [].

    Sixty sample columns A3 … B164, or 7,932 channels named 1 … 7932, are one
    kind of reading spread across columns; no one of them is the measure.
    """
    if len(numeric) < MIN_MATRIX_COLUMNS:
        return []
    numbered = [column for column in numeric if _NUMBERED_NAME.match(str(column).strip())]
    if len(numbered) < MIN_MATRIX_COLUMNS or len(numbered) < MATRIX_SHARE * len(numeric):
        return []
    return numbered


def row_summary_name(columns: int) -> str:
    """Name of the per-row mean a matrix table is measured by."""
    return f"Row mean ({columns:,} columns)"


def _prefix_groups(columns: list[str]) -> list[tuple[str, int]]:
    """Name prefixes that split the matrix columns into a few groups, such as A* and B*."""
    prefixes = Counter(
        _NUMBERED_NAME.match(str(column).strip()).group(1).upper() for column in columns
    )
    groups = [(prefix, count) for prefix, count in prefixes.most_common() if prefix]
    if 2 <= len(groups) <= 6 and all(count >= 3 for _, count in groups):
        return groups
    return []


def _matrix_note(columns: list[str], label: str | None) -> str:
    rows = label or "row"
    note = (
        f"This table looks like a matrix: {len(columns):,} numeric columns of one kind, named "
        f"{columns[0]} … {columns[-1]}, with one row per {rows}. No single column is the "
        f"measure, so ADA follows each row's mean across them ({row_summary_name(len(columns))}). "
        f"To compare the columns themselves, reshape the table to long form first: one row per "
        f"{rows} and column, with the value in a single column."
    )
    groups = _prefix_groups(columns)
    if groups:
        listed = ", ".join(f"{prefix}* ({count})" for prefix, count in groups)
        note += f" The column names fall into groups by prefix, {listed}, which may be groups to compare."
    return note


def _core_words(name: str) -> list[str]:
    """Head words without trailing words that only say it is a name: "Test Name" is a test."""
    words = _head_words(name)
    while len(words) > 1 and words[-1] in GENERIC_NAME_WORDS:
        words.pop()
    return words


def _looks_like_units(series: pd.Series) -> bool:
    values = series.dropna().astype(str).unique()[:50]
    return any(_UNIT_MARKS.search(value.strip()) for value in values)


def _spans_scales(dataframe: pd.DataFrame, parameter: str, measure: str) -> bool:
    """Whether the measure's typical size differs by LONG_SCALE_SPREAD or more between parameters."""
    working = dataframe[[parameter, measure]].dropna()
    working = working[working[measure] != 0]
    common = working[parameter].value_counts().head(20).index
    medians = working[working[parameter].isin(common)].groupby(parameter)[measure].median().abs()
    medians = medians[medians > 0]
    return len(medians) >= 2 and float(medians.max() / medians.min()) >= LONG_SCALE_SPREAD


def _long_layout(
    dataframe: pd.DataFrame, measure: str | None, text_columns: list[str]
) -> tuple[str | None, str | None]:
    """Find the column naming what each row measures, and the column naming its unit.

    A text column named like a parameter qualifies when the measure takes very
    different sizes across its values, or a unit column holds several units.
    A unit column alone, with several units, is used as the parameter.
    """
    if measure is None:
        return None, None
    units = next(
        (
            column
            for column in text_columns
            if (_core_words(column) or [""])[-1] in UNIT_WORDS
            and 2 <= dataframe[column].nunique() <= MAX_LABEL_LEVELS
        ),
        None,
    )
    parameter = next(
        (
            column
            for column in text_columns
            if column != units
            and (_core_words(column) or [""])[-1] in PARAMETER_WORDS
            and 2 <= dataframe[column].nunique() <= dataframe[column].notna().sum() / 2
        ),
        None,
    )
    if parameter and not units and not _spans_scales(dataframe, parameter, measure):
        parameter = None
    if units and not parameter and not _looks_like_units(dataframe[units]):
        units = None
    return parameter or units, units


def _long_focus(
    dataframe: pd.DataFrame, parameter: str, units: str | None, measure: str
) -> tuple[str, str]:
    """Pick the parameter the brief covers, the most common, and describe the choice."""
    measured = dataframe[dataframe[measure].notna()]
    counts = measured[parameter].value_counts()
    choices = counts
    if units and units != parameter:
        # A parameter recorded in two units would still mix them.
        single = measured.groupby(parameter)[units].nunique()
        choices = counts[counts.index.isin(single[single <= 1].index)]
        if choices.empty:
            choices = counts
    focus = choices.index[0]
    unit_count = int(dataframe[units].nunique()) if units else 0
    if parameter == units:
        what = (
            f"{measure} is recorded in {unit_count} different units ({units}), so values in "
            "different units are never combined."
        )
    else:
        what = (
            f"Each row holds one {parameter} measured in {measure}"
            + (f", in {unit_count} different units ({units})" if unit_count > 1 else "")
            + ", so values for different ones are never combined."
        )
    which = "the most common" if focus == counts.index[0] else "the most common recorded in one unit"
    note = f"{what} The brief covers {focus}, {which} ({counts[focus]:,} rows)"
    others = [value for value in counts.index if value != focus]
    if others:
        listed = ", ".join(f"{value} ({counts[value]:,})" for value in others[:NAMED_IN_NOTES])
        more = f", and {len(others) - NAMED_IN_NOTES:,} more" if len(others) > NAMED_IN_NOTES else ""
        note += f"; others include {listed}{more}. Filter the table to one of them to analyze it instead"
    return str(focus), note + "."


def _named(columns: list[str]) -> str:
    shown = ", ".join(str(column) for column in columns[:NAMED_IN_NOTES])
    if len(columns) > NAMED_IN_NOTES:
        return f"{shown} and {len(columns) - NAMED_IN_NOTES:,} more"
    return shown


def _factor_note(dataframe: pd.DataFrame, factors: list[str], flags: list[str]) -> str:
    them = "them" if len(factors) > 1 else "it"
    note = (
        f"Read as categories, not quantities: {_named(factors)}. ADA can group by {them} and "
        f"leaves {them} out of totals and correlations."
    )
    if flags:
        shares = [
            f"{column} {float(dataframe[column].mean()) * 100:.1f}%" for column in flags[:NAMED_IN_NOTES]
        ]
        more = f", and {len(flags) - NAMED_IN_NOTES:,} more" if len(flags) > NAMED_IN_NOTES else ""
        note += f" For a 0/1 flag the average is the share of rows that are 1: {', '.join(shares)}{more}."
    return note


def _identifier_note(column: str, series: pd.Series) -> str:
    if is_numeric_dtype(series):
        return (
            f"{column} holds a different whole number on every row, so it is read as a row "
            "identifier, not a quantity."
        )
    return f"{column} is different on every row, so it is read as the row label."


def _running_total_note(counter: str, sibling: str | None) -> str:
    note = (
        f"{counter} keeps climbing, dropping back only when it resets, like a gauge or meter "
        "reading, so adding it up or averaging it over a period would mean nothing; it is not "
        "used as the measure."
    )
    if sibling:
        note += f" {sibling} records each reading's own amount, so it is used instead."
    return note


def _choose_measure(
    dataframe: pd.DataFrame, candidates: list[str], preferred: set[str]
) -> tuple[str | None, tuple[str, ...]]:
    """Pick the measure, and the runners-up when the pick was a close call.

    The name scores first: money and outcome words up; keys, design labels,
    inputs, checks and near-constant columns down. Ties go to the column
    holding the most information, then to fractional values, completeness and
    the name, so column order never decides. ``preferred`` columns record a
    running total's own amounts and are favoured over it.
    """
    unit_outcome = _lone_unit(candidates)
    scored: list[tuple[int, float, float, float, str]] = []
    covariates: set[str] = set()
    nearly_constant: set[str] = set()
    for column in candidates:
        series = dataframe[column]
        words = _head_words(column)
        head = words[-1] if words else ""
        score = _keyword_score(column, MEASURE_KEYWORDS)
        if looks_like_identifier(column, series):
            score -= 20
        if _places_rows(column, series):
            score -= 15
        if head in INPUT_WORDS:
            score -= 4
        if set(words) & CHECK_WORDS or head == "check":
            score -= 6
        if column == unit_outcome:
            score += 4
        if column in preferred:
            score += 10
        if _top_share(series) > NEARLY_CONSTANT:
            score -= 6
            nearly_constant.add(column)
        if set(words) & COVARIATE_TOKENS:
            covariates.add(column)
        # Amounts carry fractions; postal codes and counters do not. A weak
        # signal, so it only breaks ties.
        continuous = float(not is_bool_dtype(series) and bool((series.dropna() % 1 != 0).any()))
        scored.append((score, _information(series), continuous, float(series.notna().mean()), column))
    if not scored:
        return None, ()

    # A covariate gives way only to another plausible outcome, so a table of
    # weather readings leads with a temperature rather than a mostly-zero column.
    if any(
        column not in covariates and column not in nearly_constant and score >= 0
        for score, *_, column in scored
    ):
        scored = [
            (score - 3 if column in covariates else score, *rest, column)
            for score, *rest, column in scored
        ]
    ranked = sorted(scored, key=lambda item: (-item[0], -item[1], -item[2], -item[3], item[4]))
    best = ranked[0]
    close = [item[4] for item in ranked[1:] if item[0] >= best[0] - CLOSE_CALL_POINTS]
    return best[4], tuple(close[:MAX_ALTERNATIVES])


def _factors(
    dataframe: pd.DataFrame, numeric: list[str], text_dimensions: list[str]
) -> tuple[list[str], list[str]]:
    """The numeric columns holding category codes, and the 0/1 flags among them.

    Weak evidence (a short run of whole numbers, a design level) is dropped
    when it would leave nothing to measure: marks out of 10 are the quantity
    of a table that has nothing else.
    """
    kinds = {column: _factor_kind(column, dataframe[column]) for column in numeric}
    found = {column for column, kind in kinds.items() if kind}
    rest = [column for column in numeric if column not in found]
    groupings = [dataframe[column] for column in [*text_dimensions, *found]]
    # Rows repeat a recipe or a sample across the levels of a design column.
    groupings += [
        dataframe[column]
        for column in rest
        if _sample(dataframe[column]).nunique() <= len(_sample(dataframe[column])) / 2
    ][:10]
    for column in rest:
        if _is_design_level(column, dataframe[column], groupings):
            kinds[column] = "code"
            found.add(column)
    if len(found) == len(numeric):
        found = {column for column in found if kinds[column] != "code"}
    factors = [column for column in numeric if column in found]
    return factors, [column for column in factors if kinds[column] == "flag"]


def detect_roles(dataframe: pd.DataFrame) -> ColumnRoles:
    """Infer likely roles from names, values, and cardinality."""
    numeric = dataframe.select_dtypes(include=np.number).columns.tolist()
    date_columns = [
        column for column in dataframe.columns if is_datetime64_any_dtype(dataframe[column])
    ]
    notes: list[str] = []

    # Sorted by name first because max() keeps the first of equal elements, so a
    # tie goes to the alphabetically first name regardless of column order.
    date = max(
        sorted(date_columns, key=lambda column: (normalized_name(column), str(column))),
        key=lambda column: (
            1 if any(token in normalized_name(column) for token in ("date", "time", "created")) else 0,
            # The event that created the row outranks later ones: order date over ship date.
            _date_preference(column),
            int(dataframe[column].notna().sum()),
        ),
        default=None,
    )

    matrix = _matrix_columns(numeric)
    in_matrix = set(matrix)
    identifiers = _identifier_columns(
        dataframe, [column for column in dataframe.columns if column not in in_matrix]
    )
    identifier = identifiers[0] if identifiers else None
    notes.extend(
        _identifier_note(column, dataframe[column])
        for column in identifiers
        if not looks_like_identifier(column, dataframe[column])
    )
    keys = set(identifiers)
    numeric = [column for column in numeric if column not in keys]
    in_numeric = set(numeric)

    text_columns: list[str] = []
    text_dimensions: list[str] = []
    for column in dataframe.columns:
        if column == date or column in in_numeric or column in keys:
            continue
        series = dataframe[column]
        if _named_like_a_date(column) or is_datetime64_any_dtype(series):
            # Date-named columns are never segments, even when they didn't parse as dates.
            continue
        text_columns.append(column)
        unique = int(series.nunique(dropna=True))
        non_null = int(series.notna().sum())
        if 2 <= unique <= MAX_LABEL_LEVELS and unique / max(non_null, 1) <= MAX_GROUPING_SHARE:
            text_dimensions.append(column)

    factors, flags = _factors(
        dataframe, [column for column in numeric if column not in in_matrix], text_dimensions
    )
    if factors:
        notes.append(_factor_note(dataframe, factors, flags))
    in_factors = set(factors)

    order_in_table = {column: position for position, column in enumerate(dataframe.columns)}
    dimensions = sorted([*text_dimensions, *factors], key=order_in_table.__getitem__)
    dimension_candidates: list[tuple[int, int, str]] = []
    for column in dimensions:
        words = _head_words(column)
        if date and words and words[-1] in TIME_PART_WORDS:
            # Beside a date, a month or year column only repeats the timeline.
            continue
        series = dataframe[column]
        score = _keyword_score(column, DIMENSION_KEYWORDS)
        if is_design_name(column) or (words and words[-1] in INDIVIDUAL_NAME_WORDS):
            # A plate row, a run label or one subject each groups by layout or by
            # individual, not by anything compared, so it is picked only when no
            # other column qualifies.
            score -= 15
        if _top_share(series) > LOPSIDED_GROUPING:
            score -= 5
        preferred_size = -abs(int(series.nunique(dropna=True)) - 10)
        dimension_candidates.append((score, preferred_size, column))

    dimension = (
        max(dimension_candidates, key=lambda candidate: (candidate[0], candidate[1]))[2]
        if dimension_candidates
        else None
    )

    candidates = [column for column in numeric if column not in in_factors and column not in in_matrix]
    order = _reading_order(dataframe, date, dimension)
    groups = pd.factorize(dataframe[dimension])[0][order] if dimension else np.zeros(len(order))
    counters = [
        column
        for column in candidates
        if _running_total(column, dataframe[column].to_numpy(dtype=float, na_value=np.nan)[order], groups)
    ]
    candidates = [column for column in candidates if column not in counters]
    siblings = {counter: _sibling_interval(counter, candidates) for counter in counters}
    measure, measure_alternatives = _choose_measure(
        dataframe, candidates, {sibling for sibling in siblings.values() if sibling}
    )
    notes.extend(
        _running_total_note(counter, sibling if sibling == measure else None)
        for counter, sibling in siblings.items()
    )

    table_shape = "flat"
    focus = None
    parameter, units = _long_layout(dataframe, measure, text_columns)
    if parameter and measure:
        table_shape = "long"
        dimension = parameter
        if parameter not in dimensions:
            dimensions.append(parameter)
        focus, note = _long_focus(dataframe, parameter, units, measure)
        notes.append(note)
    elif matrix:
        table_shape = "matrix"
        measure, measure_alternatives = None, ()
        notes.append(_matrix_note(matrix, identifier))

    return ColumnRoles(
        date=date,
        measure=measure,
        dimension=dimension,
        identifier=identifier,
        # Quantities first, so a list of what to measure leads with them.
        numeric=tuple(
            sorted(
                numeric,
                key=lambda column: column in in_factors or _places_rows(column, dataframe[column]),
            )
        ),
        dimensions=tuple(dimensions),
        factors=tuple(factors),
        flags=tuple(flags),
        measure_alternatives=measure_alternatives,
        table_shape=table_shape,
        focus=focus,
        notes=tuple(notes),
    )


def add_row_summary(dataframe: pd.DataFrame, roles: ColumnRoles) -> tuple[pd.DataFrame, ColumnRoles]:
    """Give a matrix table each row's mean across its numbered columns, as its measure.

    A matrix has no single column to headline, so the brief follows the row
    mean, named for what it is. Other tables are returned unchanged.
    """
    if roles.table_shape != "matrix" or roles.measure:
        return dataframe, roles
    block = _matrix_columns(list(roles.numeric))
    if not block:
        return dataframe, roles
    name = row_summary_name(len(block))
    if name not in dataframe.columns:
        dataframe = pd.concat([dataframe, dataframe[block].mean(axis=1).rename(name)], axis=1)
    numeric = roles.numeric if name in roles.numeric else (*roles.numeric, name)
    return dataframe, replace(roles, measure=name, numeric=numeric)

