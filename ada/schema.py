"""Infer the role of each column: measure, date, grouping, identifier or category code.

Roles are guessed from names, values and cardinality together, since none is
reliable alone: "Order Number" is numeric but not a measure, a column of 0s and
1s is a flag rather than a quantity, and "Value" may be either.
"""

from __future__ import annotations

import re
import warnings
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from .formatting import format_number, is_currency, is_percentage, normalized_name
from .metrics import ADDITIVE_WORDS, LEVEL_WORDS, is_additive_name, is_level_name


@dataclass(frozen=True)
class ColumnRoles:
    date: str | None
    measure: str | None
    dimension: str | None
    identifier: str | None
    #: Numeric columns that can be measured: quantities first, then codes,
    #: labels and constant columns. Identifiers are left out: a row key is a
    #: number but not a quantity.
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
    #: "flat"; "long" when one column names what each row's value measures;
    #: "matrix" when the table is mostly numeric columns of one kind, such as
    #: samples or spectral channels; or "stacked" when one column names which
    #: of several tables, each with its own columns, a row belongs to.
    table_shape: str = "flat"
    #: The value of ``dimension`` the brief covers. A long table's rows measure
    #: different things, often in different units, so they are never combined;
    #: a stacked file's tables hold different columns.
    focus: str | None = None
    #: What ADA concluded about the columns, in plain sentences.
    notes: tuple[str, ...] = ()
    #: Numeric columns that place or describe a row rather than measure it:
    #: coordinates, compass directions, quality flags, parts of a date, and
    #: running totals and counters (see ``running_total_columns``). They
    #: stay in ``numeric`` so questions can name them, but are never the
    #: measure, a grouping or a reading to compare or correlate.
    non_readings: tuple[str, ...] = ()
    #: The plural noun the file's name gives its rows, such as "penguins" for
    #: penguins.csv, so a question can name the rows the way the file does.
    row_noun: str | None = None


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
    "risk": 5,
    "diversity": 5,
    "score": 4,
    "index": 4,
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

IDENTIFIER_TOKENS = ("id", "uuid", "key", "code", "number", "invoice", "order", "barcode")
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
# The size of a subject or a plot, recorded beside the outcome of a study: a
# weight in a microbiome survey, the acres of a farm.
SIZE_COVARIATE_TOKENS = frozenset({"weight", "size", "area", "acres", "acre", "height", "age"})
# Words naming a statistic computed about some other quantity: a p-value, a
# coefficient, a standard error. Such a column is rarely what a table measures.
STATISTIC_WORDS = frozenset({
    "pvalue", "pval", "padj", "qvalue", "fdr", "stderr", "sem", "std", "stdev", "beta", "coef",
    "coefficient", "significant", "significance", "tstat", "zscore",
})
# Pairs of short words that name a statistic together, where either alone may
# be something else: "T" a temperature, "P" phosphorus, "Z" a coordinate.
STATISTIC_PAIRS = frozenset({
    ("p", "value"), ("p", "val"), ("t", "stat"), ("t", "value"), ("z", "score"), ("z", "stat"),
    ("z", "value"), ("std", "err"), ("ci", "low"), ("ci", "high"), ("ci", "lower"), ("ci", "upper"),
})
# Head words of a column naming the outcome class of each row, which the
# measure should tell apart when names alone cannot choose it.
OUTCOME_LABEL_WORDS = frozenset({
    "result", "outcome", "diagnosis", "class", "classification", "label", "disease", "case",
    "condition", "status", "target",
})
# Head words of a two-level column that a table is built to explain, as a model
# would predict it: "target", "classification", "Outcome". Weaker words such as
# "status" or "disease" also name comorbidities and other descriptors.
OUTCOME_FLAG_WORDS = frozenset({
    "target", "label", "class", "classification", "outcome", "result", "diagnosis",
})
# Parts of the name of a 0/1 column recording whether an event happened by the
# end of follow-up ("eventdeath", "dead", "relapse"), and of the column timing it
# ("survival", "os_time", "followup_months").
EVENT_PARTS = ("event", "death", "dead", "died", "deceased", "relapse", "recurrence", "censor")
DURATION_PARTS = ("survival", "time", "followup", "duration", "days", "months", "years")
# Words of a file name that say nothing about what the table holds.
GENERIC_FILE_WORDS = frozenset({
    "data", "dataset", "datasets", "database", "final", "clean", "cleaned", "export", "table",
    "file", "info", "supplementary", "combined", "sheet", "sample", "samples", "values", "copy",
    "dataframe", "upload", "demo", "the", "and", "for", "new", "raw", "all", "set", "log", "out",
    "csv", "tsv", "txt", "xls", "xlsx", "update", "version", "test", "train",
})
# A measure that tells a two-level outcome apart may give up this many points of
# name score; money and amounts score well beyond it.
OUTCOME_WINDOW = 5
# A reading sets a two-level outcome when one cut point puts this share of rows
# in their own group, as a label derived from a threshold does.
SET_BY_SHARE = 0.995
# Coordinates are continuous numbers, but never a quantity to total or trend.
COORDINATE_TOKENS = frozenset({
    "latitude", "longitude", "lat", "lon", "lng", "easting", "northing", "gps", "utm",
    "coordinate", "coordinates", "coord", "coords",
})
# Last words of a column of compass directions or angles round a circle, where
# 359° sits next to 0°: "Wind Direction", "Heading", "slope aspect", "WD".
DIRECTION_WORDS = frozenset({"direction", "dir", "bearing", "heading", "azimuth", "aspect", "wd"})
# Words of a column that flags the quality of other readings: "QN_3", "qc_code".
QUALITY_FLAG_WORDS = frozenset({"qn", "qc", "qa", "qf"})
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
    "mobile", "tel", "fax", "sku", "number", "no", "#", "key", "uuid", "notation", "barcode",
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
# Key words that end the name of a subject measured more than once (a person, an
# animal, a station) after a word saying which: "Patient ID", "Mouse #", "subject#".
# Contact details and product codes (a ZIP code, a phone number, a SKU) are keys
# too, but not subjects a study measures.
SUBJECT_KEY_WORDS = frozenset({"id", "ids", "code", "codes", "number", "no", "#", "key", "uuid"})
# Time points before a key word, beside the parts of a date and the set levels of
# an experiment: "Visit Number" counts visits, not people.
TIME_POINT_WORDS = frozenset({"visit", "wave", "session", "period", "cycle"})
# A subject measured more than once repeats its value: at least this share of the
# rows belong to subjects with two rows or more. A key with up to a fifth of its
# values repeated (see ``looks_like_identifier``) stays below it.
SUBJECT_ROW_SHARE = 0.5

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
# Words that put a column of whole numbers on a rating scale.
RATING_WORDS = frozenset({
    "rating", "ratings", "likert", "scale", "satisfaction", "agreement", "likelihood",
})
# A rating scale steps in whole numbers from 0 or 1 up to at most this, using
# at least MIN_SCALE_POINTS of its points: 0 to 10, 1 to 7, 1 to 5.
MAX_SCALE_TOP = 10
MIN_SCALE_POINTS = 5
# Columns sharing one rating scale are the items of a questionnaire.
MIN_SCALE_ITEMS = 3
MAX_SWEEP_LEVELS = 100
# A setting swept in equal steps is used about equally often at every level.
SWEEP_BALANCE = 0.5

# Head words of columns holding calendar years and months. A table with no date
# column but a year (and perhaps a month) is still a timeline; cleaning builds
# its dates from these.
YEAR_NAME_WORDS = frozenset({"year", "years", "yr"})
MONTH_NAME_WORDS = frozenset({"month", "months", "mon", "mo"})
# Head words of the parts finer than a month, coarsest first. With a year and a
# month, cleaning adds them to its dates, down to the finest one present.
FINER_PART_WORDS = {
    "day": frozenset({"day", "da", "dy"}),
    "hour": frozenset({"hour"}),
    "minute": frozenset({"minute"}),
    "second": frozenset({"second"}),
}
# Parts of a date held in their own column. Beside a date column they repeat it,
# so they are not offered as the grouping.
TIME_PART_WORDS = (
    frozenset(TIME_PART_TOKENS) | YEAR_NAME_WORDS | MONTH_NAME_WORDS
    | frozenset({"da", "dy", "wk", "qtr", "doy", "dow", "weekday"})
)
# The whole numbers each part of a date takes, by the words that name it.
CALENDAR_RANGES = (
    (MONTH_NAME_WORDS, (1, 12)),
    (frozenset({"day", "da", "dy"}), (1, 31)),
    (frozenset({"hour"}), (0, 24)),
    (frozenset({"minute"}), (0, 59)),
    (frozenset({"week", "wk"}), (0, 53)),
    (frozenset({"quarter", "qtr"}), (1, 4)),
    (frozenset({"doy"}), (1, 366)),
    (frozenset({"dow", "weekday"}), (0, 7)),
)
MONTH_ABBREVIATIONS = frozenset({
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
})

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
# Up to this many levels, any column whose values repeat can group rows; past
# it, most groups must hold MIN_ROWS_PER_GROUP rows (see ``_groups_rows``).
MAX_LABEL_LEVELS = 100
MIN_ROWS_PER_GROUP = 3
MAX_GROUPING_SHARE = 0.65
MAX_CODE_LEVELS = 12
# A short code must be recorded on at least this share of rows: a reading filled
# on a handful of rows takes few values there without being a category.
MIN_CODE_COVERAGE = 0.2
# Whole numbers on a coarse grid with at least this many levels are a reading
# rounded to that grid, such as blood pressure to the nearest 10 mm Hg.
MIN_GRID_LEVELS = 6
MAX_DESIGN_LEVELS = 15
MIN_ROWS_PER_LEVEL = 5
# A grouping whose largest group holds more than this splits off a sliver.
LOPSIDED_GROUPING = 0.9
# A column with one value in more than this share of rows carries little to report.
NEARLY_CONSTANT = 0.8
MIN_MATRIX_COLUMNS = 20
MATRIX_SHARE = 0.75
# A stacked table owns a column recorded on at least OWNED_SHARE of its rows
# that another table records on at most ABSENT_SHARE of its rows.
MAX_STACKED_TABLES = 20
OWNED_SHARE = 0.5
ABSENT_SHARE = 0.02
STACKED_COLUMN_SHARE = 1 / 3
# Columns of counts make a matrix without numbered names from this many on,
# when their typical sizes sit within this factor of one another.
MIN_COUNT_MATRIX_COLUMNS = 10
MATRIX_SCALE_SPREAD = 1000
MIN_RUNNING_TOTAL_ROWS = 20
RUNNING_TOTAL_RISING = 0.95
# A running total that resets falls to this share of its last value or less.
RESET_FRACTION = 0.1
# A counter steps up by exactly one at this share of its steps or more, and
# falls back at no more than this share.
COUNTING_STEPS = 0.5
COUNTER_FALLS = 0.01
RUNNING_TOTAL_WORDS = frozenset({"cumulative", "cum", "running", "accumulated", "ytd", "odometer"})
TOTAL_WORDS = frozenset({"total", "totals"}) | RUNNING_TOTAL_WORDS
INTERVAL_WORDS = frozenset({"interval", "incremental", "increment", "hourly", "daily", "period", "delta"})
# Long tables are split by parameter when its values sit this many times apart.
LONG_SCALE_SPREAD = 10.0
# Runners-up within this many points of the measure's score were close calls.
CLOSE_CALL_POINTS = 3
# A candidate recorded on fewer rows than these shares of the best-covered one's.
THIN_COVERAGE = 0.5
SPARSE_COVERAGE = 0.25
# A grouping and the measure must share at least this many rows, or half the
# measure's rows in a small table.
MIN_SHARED_ROWS = 12
# On a tie of names, a grouping explaining at least this share of the measure,
# and this much more than the one with the handiest number of groups, wins.
MIN_EXPLAINED = 0.05
EXPLAINED_MARGIN = 0.02
# A grouping this much more complete than its twin is preferred to it.
COMPLETENESS_GAIN = 0.05
# The most levels an outcome label may have to judge how well readings separate it.
MAX_OUTCOME_CLASSES = 20
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
    return bool(words) and words[-1] in IDENTIFIER_NAME_WORDS and not _bare_no(name, words)


def _bare_no(name: str, words: list[str]) -> bool:
    """Whether the name is "No" alone with no mark, as likely nitric oxide or a yes/no answer as a number."""
    return words == ["no"] and not re.search(r"[.#]", str(name))


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
    """Whether the name labels the unit a row belongs to, such as "Subject", "Station" or "Patient ID".

    A bare "NO" is not one (see ``is_identifier_name``): it is as likely a
    reading of nitric oxide.
    """
    words = _head_words(name)
    return bool(words) and words[-1] in ENTITY_NAME_WORDS and not _bare_no(name, words)


def is_year_name(name: str, values: pd.Series | None = None) -> bool:
    """Whether the name marks a column of calendar years, such as "Year" or "survey_yr".

    With ``values``, they must look like years too (see ``is_time_part``), so
    "Age_years" holding 1 to 80 is an age.
    """
    words = _head_words(name)
    return bool(words) and words[-1] in YEAR_NAME_WORDS and (values is None or is_time_part(name, values))


def is_month_name(name: str, values: pd.Series | None = None) -> bool:
    """Whether the name marks a column of calendar months, such as "Month".

    With ``values``, they must be months too, 1 to 12 or month names, so
    "Age_months" holding 12 to 66 is an age.
    """
    words = _head_words(name)
    return bool(words) and words[-1] in MONTH_NAME_WORDS and (values is None or is_time_part(name, values))


def finer_part_name(name: str) -> str | None:
    """The part of a date finer than a month that the name marks ("day", "hour", ...), or None.

    "Day", "DA" and "obs_hour" mark one; "Days" and "Hours" are as likely amounts of time.
    """
    words = _head_words(name)
    return next((part for part, heads in FINER_PART_WORDS.items() if words and words[-1] in heads), None)


def _calendar_range(words: list[str]) -> tuple[int, int] | None:
    """The whole numbers a part of a date named by ``words`` can take: 1 to 12 for a month."""
    head = words[-1]
    if head in YEAR_NAME_WORDS:
        # A two-digit year ("YR" 40 to 45) only under a bare singular name:
        # "Years" alone is as likely years of service.
        return (0, 2999) if words in (["yr"], ["year"]) else (1000, 2999)
    for part_words, bounds in CALENDAR_RANGES:
        if head in part_words:
            return bounds
    return None


def is_time_part(name: str, values: pd.Series | None = None) -> bool:
    """Whether the column holds a part of a date, such as a month, a day of the month or a year.

    The last word must name the part ("Month", "MO", "DA", "survey_yr"). With
    ``values``, numbers must fall within that part's range: months 1 to 12,
    days 1 to 31, hours 0 to 24, years 1000 to 2999. "Age_months" holding
    12 to 66 or "Duration Hours" up to 200 are amounts of time, not parts of a
    date. Text is judged by the name, except that a month column must hold month
    names.
    """
    words = _head_words(name)
    if not words or words[-1] not in TIME_PART_WORDS:
        return False
    if values is None:
        return True
    present = _sample(values).dropna()
    if present.empty:
        return False
    if not is_numeric_dtype(present) or is_bool_dtype(present):
        if words[-1] not in MONTH_NAME_WORDS:
            return True
        return bool(present.astype(str).str.strip().str[:3].str.lower().isin(MONTH_ABBREVIATIONS).all())
    bounds = _calendar_range(words)
    if bounds is None or not bool((present % 1 == 0).all()):
        return False
    return bool(bounds[0] <= present.min() and present.max() <= bounds[1])


def _named_like_a_date(name: str) -> bool:
    return any(token in normalized_name(name).split() for token in DATE_NAME_TOKENS)


def _head_words(name: str) -> list[str]:
    """Words of a column name, with any trailing annotation and punctuation removed.

    Dots separate words too, so "determinand.unit.label" ends in "label", and a
    number sign is a word of its own, so "subject#" reads as "subject #".
    """
    text = normalized_name(_ANNOTATION.sub("", str(name))).replace(".", " ").replace("#", " # ")
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


def is_subject_name(name: str) -> bool:
    """Whether the name labels a subject measured more than once: "Subject", "Patient ID", "subject#".

    The last word must be an entity ("Subject", "Station", "Site") or a key word.
    A key word needs a word before it that is not a time point, a count, a design
    label or a group ("Visit Number", "Run No" and "Group ID" are not subjects),
    except "ID" alone. A number mark run into a word counts as its own word.
    """
    words = []
    for word in _head_words(name):
        words += [word[:-1], "#"] if len(word) > 1 and word.endswith("#") else [word]
    if not words or _bare_no(name, words):
        return False
    if words[-1] in ENTITY_NAME_WORDS - IDENTIFIER_NAME_WORDS:
        return True
    if words[-1] not in SUBJECT_KEY_WORDS:
        return False
    qualifiers = set(words[:-1])
    if not qualifiers:
        return words[-1] in {"id", "ids"}
    times = TIME_POINT_WORDS | TIME_PART_WORDS | DESIGN_LEVEL_WORDS
    return not qualifiers & (times | DESIGN_TOKENS | COUNT_NAME_WORDS | LABEL_NAME_WORDS)


def looks_like_subject(name: str, series: pd.Series) -> bool:
    """Whether the column names the subject each row belongs to, its value repeated on that subject's rows.

    Recordings of 42 people carry "subject#" 1 to 42 on 5,875 rows. The name must
    label a subject (see ``is_subject_name``), numbers must be whole codes, and
    the values must repeat as subjects do: at least two of them, with most rows
    (SUBJECT_ROW_SHARE) on a subject that has other rows too, so a few people
    seen only once still leave a subject. That tells a subject from a key with a
    different value on (nearly) every row (see ``looks_like_identifier``), and
    from a column with one value, which tells no subjects apart. A count, a
    rating or a year repeats too, but is not named as a subject.
    """
    if is_datetime64_any_dtype(series) or is_bool_dtype(series) or not is_subject_name(name):
        return False
    present = series.dropna()
    if is_numeric_dtype(present) and not bool((present % 1 == 0).all()):
        return False
    counts = present.value_counts()
    return len(counts) >= 2 and float(counts[counts >= 2].sum()) >= SUBJECT_ROW_SHARE * len(present)


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
    if not named and (not first or words & QUANTITY_WORDS or is_time_part(name, series)):
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
    words do not count. A column named like a date counts only when its values
    are codes with letters in them rather than dates ("time" over gene names
    YAL001C ...), since the header can mislead.
    """
    if is_numeric_dtype(series) or is_datetime64_any_dtype(series):
        return False
    present = series.dropna()
    if len(present) < MIN_ROWS_FOR_KEY or len(present) < 0.9 * len(series):
        return False
    if present.nunique() != len(present):
        return False
    head = present.head(200).astype(str)
    if _named_like_a_date(name) and not bool(head.str.contains(r"[A-Za-z]{2}").mean() >= 0.9):
        return False
    words = head.str.split().str.len()
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
    named = is_time_part(name, series) or is_design_name(name) or is_entity_name(name)
    return named and not bool((series.dropna() % 1 != 0).any())


def _not_a_quantity(name: str, series: pd.Series, factor: bool) -> bool:
    """Whether a numeric column is a code, a label or a constant rather than something to measure."""
    return factor or _places_rows(name, series) or series.nunique(dropna=True) <= 1


def is_angle_column(name: str, series: pd.Series) -> bool:
    """Whether the column holds compass directions or angles round a circle, such as "Wind Direction".

    The last word must name a direction (see DIRECTION_WORDS) and the values
    fall within 0 to 360. Since 359° sits next to 0°, an average or a linear
    correlation of such a column means nothing.
    """
    words = _head_words(name)
    if not words or words[-1] not in DIRECTION_WORDS or not is_numeric_dtype(series) or is_bool_dtype(series):
        return False
    values = series.dropna()
    return not values.empty and float(values.min()) >= 0 and float(values.max()) <= 360


def is_quality_flag(name: str, columns) -> bool:
    """Whether the column flags the quality of other readings rather than recording one.

    "QN_3", "qc_code" and "quality_flag" do. A name ending in "flag" counts
    only after the name of another of ``columns`` ("TMK_flag" beside "TMK"),
    since "Fraud_Flag" alone is an outcome to compare.
    """
    words = [word for word in _head_words(name) if not word.isdigit()]
    if not words:
        return False
    if set(words) & QUALITY_FLAG_WORDS:
        return True
    if len(words) > 1 and words[0] in {"quality", "qual"} and words[-1] in {"flag", "flags", "code", "codes"}:
        return True
    if len(words) > 1 and words[-1] in {"flag", "flags"}:
        stem = words[:-1]
        return any(_head_words(other) == stem for other in columns if other != name)
    return False


def _non_reading_note(quality: list[str], places: list[str], angles: list[str], sequences: list[str]) -> str:
    def told(columns: list[str], verb: str, rest: str) -> str:
        """The columns, then "which" and the verb agreeing with how many there are."""
        return f"{_named(columns)}, which {verb + 's' if len(columns) == 1 else verb} {rest}"

    parts = []
    if sequences:
        parts.append(told(
            sequences, "hold", "a different whole number on every row in order, as a key or a date "
            "written as a number does",
        ))
    if quality:
        parts.append(told(quality, "flag", "the quality of other readings"))
    if places:
        parts.append(told(places, "place", "each row on a map"))
    if angles:
        parts.append(told(angles, "hold", "compass directions, on which 359° sits next to 0°"))
    return "Not measured, compared or correlated as readings: " + "; ".join(parts) + "."


def _plural(word: str) -> bool:
    return len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is"))


def _named_as_quantity(name: str) -> bool:
    """Whether the name marks a count or an amount, whatever its values look like."""
    words = _head_words(name)
    if not words:
        return False
    return bool(set(words) & COUNT_NAME_WORDS) or words[-1] in QUANTITY_HEAD_WORDS or _plural(words[-1])


def _named_as_label(name: str) -> bool:
    """Whether the name marks group labels: a status, a condition, a site, a plate row or "id_reef"."""
    words = _head_words(name)
    keyed = len(words) > 1 and words[0] in {"id", "code"}
    return bool(words) and (
        words[-1] in LABEL_NAME_WORDS or is_entity_name(name) or is_design_name(name) or keyed
    )


def _factor_kind(name: str, series: pd.Series, scale_item: bool = False) -> str | None:
    """Say whether a numeric column holds category codes, and on what evidence.

    "flag" is a 0/1 column. "label" is a column of whole numbers named like a
    label (Status, Group, Station), with enough rows in most groups (see
    ``_groups_rows``). "code" is weaker evidence: two values, or up to
    MAX_CODE_LEVELS whole numbers each repeated many times, under a name that
    is not a count, an amount or a unit, on a column that is not a rating item
    (``scale_item``, see ``_scale_items``), is recorded on MIN_CODE_COVERAGE of
    the rows or more, and is not a reading rounded to a grid (see
    ``_on_a_grid``). None means a quantity.
    """
    if is_bool_dtype(series):
        return None
    if _sample(series).nunique() > MAX_LABEL_LEVELS and not _named_as_label(name):
        return None
    levels = int(series.nunique(dropna=True))
    if levels < 2:
        return None
    values = series.dropna()
    if levels == 2 and set(values.unique()) <= {0, 1}:
        return "flag"
    if not bool((values % 1 == 0).all()):
        return None
    if _named_as_label(name) and len(values) >= 2 * levels and _groups_rows(values):
        return "label"
    # A reading named as one ("Humidity", "pH", "Score") or measured in a unit
    # ("O content (wppm)") is a quantity even when it was recorded in whole
    # numbers with few values, and so is an item answered on a rating scale.
    named = _named_as_quantity(name) or is_level_name(name) or is_percentage(name) or _has_unit(name)
    if named or scale_item:
        return None
    if _on_a_grid(values) or len(values) < MIN_CODE_COVERAGE * len(series):
        return None
    if levels == 2 and len(values) >= 3 * levels:
        return "code"
    if levels <= MAX_CODE_LEVELS and len(values) >= MIN_ROWS_PER_LEVEL * levels:
        return "code"
    return None


def _on_a_grid(values: pd.Series) -> bool:
    """Whether whole numbers sit on a grid coarser than one, over many of its steps.

    Blood pressure written to the nearest 10 (50, 60 … 180) is a reading
    rounded to a grid; codes 1 to 6 step by one, and a few set levels (0, 10,
    20) are too few to tell.
    """
    levels = np.unique(values.to_numpy(dtype=float)).astype(np.int64)
    if len(levels) < MIN_GRID_LEVELS:
        return False
    step = int(np.gcd.reduce(levels - levels[0]))
    return step >= 2


def _groups_rows(series: pd.Series) -> bool:
    """Whether a column's values can group its rows: at least two, most of them on several rows.

    Up to MAX_LABEL_LEVELS values, any column whose values repeat enough (see
    MAX_GROUPING_SHARE) qualifies. Beyond that, most groups need at least
    MIN_ROWS_PER_GROUP rows, so 101 districts of 3 to 20 samples each group a
    table, while names that each appear once or twice do not. Judged on an even
    sample of a large table.
    """
    values = _sample(series).dropna()
    counts = values.value_counts().to_numpy()
    if len(counts) < 2 or len(counts) > MAX_GROUPING_SHARE * len(values):
        return False
    return len(counts) <= MAX_LABEL_LEVELS or float((counts >= MIN_ROWS_PER_GROUP).mean()) >= 0.5


def _family_stem(name: str) -> str | None:
    """The stem a numbered column shares with its siblings: "rdb" for "RDB2", "rp" for "RP_1".

    Headerless columns read as "Unnamed: 3" are numbered by position, not as a family.
    """
    match = _NUMBERED_NAME.match(str(name).strip())
    if not match:
        return None
    stem = normalized_name(match.group(1)).strip(" :")
    if not re.search(r"[a-z]", stem) or stem.startswith("unnamed"):
        return None
    return stem


def _scale(series: pd.Series) -> tuple[int, int] | None:
    """The rating scale a column of whole numbers sits on, as its (bottom, top), or None.

    The bottom is 0 or 1, the top at most MAX_SCALE_TOP, and at least
    MIN_SCALE_POINTS points are used. A scale of 1 to 7 whose answers happen to
    start at 2 is still 1 to 7.
    """
    values = series.dropna()
    if values.empty or not bool((values % 1 == 0).all()):
        return None
    low, high = float(values.min()), float(values.max())
    if low < 0 or high > MAX_SCALE_TOP or values.nunique() < MIN_SCALE_POINTS:
        return None
    return (0 if low == 0 else 1), int(high)


def _scale_items(dataframe: pd.DataFrame, numeric: list[str]) -> set[str]:
    """Columns of whole numbers answered on a scale, which are quantities however few their values.

    A questionnaire's items share a numbered stem ("RDB1" to "RDB3", "RP_1" to
    "RP_3") or one rating scale with at least MIN_SCALE_ITEMS other columns
    (1 to 7), or are named as ratings ("Satisfaction"). A lone column of codes
    1 to 6, such as marital status, is none of these.
    """
    samples = {column: _sample(dataframe[column]) for column in numeric}
    whole = [
        column for column in numeric
        if not is_bool_dtype(samples[column]) and bool((samples[column].dropna() % 1 == 0).all())
    ]
    stems = Counter(_family_stem(column) for column in numeric)
    scales = {column: _scale(samples[column]) for column in whole}
    shared = Counter(scale for scale in scales.values() if scale)
    items = set()
    for column in whole:
        stem, scale = _family_stem(column), scales[column]
        in_family = stem is not None and stems[stem] >= 2 and samples[column].nunique() >= 3
        rated = scale is not None and (
            shared[scale] >= MIN_SCALE_ITEMS or bool(set(_head_words(column)) & RATING_WORDS)
        )
        if in_family or rated:
            items.add(column)
    return items


def _counts_of_things(series: pd.Series) -> bool:
    values = series.dropna()
    return not values.empty and bool((values >= 0).all() and (values % 1 == 0).all())


def _sibling_counts(dataframe: pd.DataFrame, numeric: list[str]) -> set[str]:
    """0/1 columns that count something, as the sibling columns recorded with them do.

    Five animal counts per quadrat, three of which never pass 1, are counts
    all the same: they sit in one block with a count that does, side by side
    under count names ("Crabs", "Seastars") or missing on exactly the same rows.
    A yes/no column such as "Antibiotics" beside "Probiotics" stays a flag,
    since neither ever passes 1.
    """
    counts = {
        column for column in numeric
        if not is_bool_dtype(dataframe[column])
        and _named_as_quantity(column)
        and _counts_of_things(dataframe[column])
    }
    blocks: list[list[str]] = [[]]
    for column in dataframe.columns:
        if column in counts:
            blocks[-1].append(column)
        elif blocks[-1]:
            blocks.append([])
    by_gaps: dict[bytes, list[str]] = {}
    for column in counts:
        missing = dataframe[column].isna().to_numpy()
        if 0 < missing.sum() < len(missing):
            by_gaps.setdefault(np.packbits(missing).tobytes(), []).append(column)
    blocks += by_gaps.values()
    found: set[str] = set()
    for block in blocks:
        if len(block) >= 2 and any(dataframe[column].max() > 1 for column in block):
            found |= {column for column in block if set(dataframe[column].dropna().unique()) <= {0, 1}}
    return found


def _resolution(values: np.ndarray) -> float:
    """The unit the values are recorded in: 1 for whole numbers, 0.1 for one decimal place."""
    for decimals in range(7):
        scaled = values * 10.0**decimals
        if np.allclose(scaled, np.round(scaled), rtol=0, atol=1e-6):
            return 10.0**-decimals
    return 0.0


def _is_sweep(name: str, series: pd.Series) -> bool:
    """Whether the column steps through evenly spaced settings, each used about equally often.

    A frequency swept from 100 to 590 GHz in steps of 10, or a transect read
    every 5 m, is a setting of the study rather than a reading. A reading rounded
    to its resolution (humidity in whole percent, a share to one decimal) steps
    by that resolution, and is not used equally often at every level.
    """
    if is_bool_dtype(series) or _named_as_quantity(name):
        return False
    values = _sample(series).dropna()
    counts = values.value_counts()
    if not 3 <= len(counts) <= MAX_SWEEP_LEVELS:
        return False
    if counts.min() < MIN_ROWS_PER_LEVEL or counts.min() < SWEEP_BALANCE * counts.max():
        return False
    levels = np.sort(counts.index.to_numpy(dtype=float))
    steps = np.diff(levels)
    if not np.allclose(steps, steps[0], rtol=1e-6, atol=0):
        return False
    return bool(steps[0] >= 2 * _resolution(levels) > 0)


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


# Rows sampled, evenly, for checks of how columns relate that a few thousand rows settle.
RELATION_SAMPLE = 5_000


def _thin(data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Every k-th row, so at most about RELATION_SAMPLE rows remain."""
    return data.iloc[:: max(1, len(data) // RELATION_SAMPLE)]


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


# Bracketed annotations anywhere in a name: "Sound Level (dBA) (40 - 100)".
_BRACKETED = re.compile(r"[(\[]([^)\]]{1,20})[)\]]")
# Words in brackets that describe codes rather than a unit: "(Y/N)", "(True/False)".
NOT_UNIT_WORDS = frozenset({
    "yes", "no", "y", "n", "true", "false", "code", "codes", "id", "category", "class", "flag",
    "binary", "level", "levels", "type", "group",
})


def _units_in(name: str) -> list[str]:
    """The physical units written in brackets in a name: "mpa" for "YS (MPa)", "dba" for "Sound (dBA)".

    A range ("(40 - 100)"), a list of codes ("(1=yes, 0=no)", "(Y/N)") or a
    phrase is not a unit.
    """
    units = []
    for text in _BRACKETED.findall(str(name)):
        text = text.strip()
        if not re.search(r"[A-Za-zµ°%]", text) or re.search(r"[=,;:]", text) or len(text.split()) > 2:
            continue
        if not set(re.findall(r"[a-z]+", text.lower())) & NOT_UNIT_WORDS:
            units.append(" ".join(text.lower().split()))
    return units


def _has_unit(name: str) -> bool:
    """Whether the name carries a physical unit in brackets: "(MPa)", "[mg/L]", "(wppm)", "(°F)".

    A column measured in a unit is a reading however few values it takes.
    """
    return bool(_units_in(name))


def _own_units(columns: list[str]) -> set[str]:
    """Columns measured in a unit fewer than three of ``columns`` share.

    A sensor's dBA or volts mark a physical reading; seven ingredients all in
    kg/m³ are the recipe, and the column outside it is what came out (see
    ``_lone_unit``).
    """
    units = {column: set(_units_in(column)) for column in columns}
    shared = Counter(unit for found in units.values() for unit in found)
    return {column for column, found in units.items() if found and all(shared[unit] < 3 for unit in found)}


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
    enough without a reset, and so is counting up by one at most steps while
    almost never falling, as a packet or frame counter numbers the messages a
    device sends. Money and named levels (an age) are never counted up that way.
    """
    words = _head_words(name)
    if is_time_part(name, pd.Series(values)):
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
    # A counter of messages falls back almost never and steps up by one; money
    # and named levels such as an age never count so.
    counting = (
        not is_currency(name)
        and not is_level_name(name)
        and falls <= COUNTER_FALLS * len(current)
        and bool(np.all(values % 1 == 0))
        and float(np.mean(current - previous == 1)) >= COUNTING_STEPS
    )
    return counting or bool(((current < previous) & (current <= RESET_FRACTION * previous)).any())


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


def _name_shape(name: str) -> str:
    """A name with each run of letters and digits written as "x": "x_x_x_x" for "20A_508_1_2107"."""
    return re.sub(r"[A-Za-z0-9]+", "x", str(name))


def _count_matrix_columns(dataframe: pd.DataFrame, numeric: list[str]) -> list[str]:
    """The count columns of a table that is mostly a matrix of counts, whatever their names, or [].

    Twenty-six samples of read counts named AQC1cm, CC1 … Even3, beside an id
    on each row, are one kind of reading spread across columns too. They must
    be MIN_COUNT_MATRIX_COLUMNS or more columns of whole numbers of at least
    zero, most of the numeric columns, with typical sizes within
    MATRIX_SCALE_SPREAD of one another. When most of them share a structured
    name ("20A_508_1_2107", see ``_name_shape``), only those are kept, so an
    annotation code or a plot setting beside 147 injections is not averaged in.
    """
    if len(numeric) < MIN_COUNT_MATRIX_COLUMNS:
        return []
    counts = [
        column
        for column in numeric
        if not is_bool_dtype(dataframe[column])
        and _counts_of_things(_sample(dataframe[column]))
        and _sample(dataframe[column]).nunique() > 2
    ]
    if len(counts) < MIN_COUNT_MATRIX_COLUMNS or len(counts) < MATRIX_SHARE * len(numeric):
        return []
    shapes = Counter(_name_shape(column) for column in counts)
    shape, sharing = shapes.most_common(1)[0]
    if shape != "x" and sharing >= MIN_COUNT_MATRIX_COLUMNS and sharing >= MATRIX_SHARE * len(counts):
        counts = [column for column in counts if _name_shape(column) == shape]
    means = dataframe[counts].mean().to_numpy(dtype=float)
    if not (means > 0).all() or means.max() > MATRIX_SCALE_SPREAD * means.min():
        return []
    return counts


def row_summary_name(columns: int) -> str:
    """Name of the per-row mean a matrix table is measured by."""
    return f"Row mean ({columns:,} columns)"


def row_spread_name(columns: int) -> str:
    """Name of the per-row standard deviation a matrix of centred rows is measured by."""
    return f"Row SD ({columns:,} columns)"


# Rows whose mean is typically under this share of their own standard deviation
# have been centred, as log ratios and z-scores are.
CENTRED_ROWS = 0.1


def _rows_centred(dataframe: pd.DataFrame, columns: list[str]) -> float | None:
    """How far a typical row's mean sits from zero, as a share of its spread, when rows look centred.

    A gene's log ratios across time points average near zero by
    construction, so their mean only shows the normalisation, while their
    spread says how much the gene moves. None when the rows are not centred.
    """
    values = _sample(dataframe[columns]).to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        means = np.nanmean(values, axis=1)
        spreads = np.nanstd(values, axis=1, ddof=1)
    usable = np.isfinite(means) & (spreads > 0)
    if int(usable.sum()) < MIN_ROWS_FOR_KEY:
        return None
    share = float(np.median(np.abs(means[usable]) / spreads[usable]))
    return share if share < CENTRED_ROWS else None


def row_summary_columns(roles: ColumnRoles, dataframe: pd.DataFrame | None = None) -> list[str]:
    """The columns averaged into the row mean ``add_row_summary`` made the measure, or [].

    Each of them is part of that mean, so none can be said to relate to it.
    The block is found as ``add_row_summary`` found it: by name, or, for counts
    beside an identifier, from the values in ``dataframe``.
    """
    if roles.table_shape != "matrix" or not roles.measure:
        return []
    numeric = [column for column in roles.numeric if column != roles.measure]
    block = _matrix_columns(numeric)
    if not block and roles.identifier and dataframe is not None:
        block = _count_matrix_columns(dataframe, numeric)
    return block if roles.measure in (row_summary_name(len(block)), row_spread_name(len(block))) else []


def _prefix_groups(columns: list[str]) -> list[tuple[str, int]]:
    """Name prefixes that split the matrix columns into a few groups, such as A* and B*."""
    matches = [_NUMBERED_NAME.match(str(column).strip()) for column in columns]
    if not all(matches):
        return []
    prefixes = Counter(match.group(1).upper() for match in matches)
    groups = [(prefix, count) for prefix, count in prefixes.most_common() if prefix]
    if 2 <= len(groups) <= 6 and all(count >= 3 for _, count in groups):
        return groups
    return []


def _matrix_note(dataframe: pd.DataFrame, columns: list[str], label: str | None, counts: bool) -> str:
    kind = "columns of counts" if counts else "numeric columns"
    # A label under a date-like header ("time" over gene names) is named as what it holds.
    rows = f"value of {label}" if label and _named_like_a_date(label) else label
    each = f", with one row per {rows}" if rows else ""
    centred = None if counts else _rows_centred(dataframe, columns)
    if centred is None:
        follows = f"so ADA follows each row's mean across them ({row_summary_name(len(columns))})"
    else:
        follows = (
            f"and each row's values are centred near zero, as log ratios or z-scores are (a typical "
            f"row's mean is {centred:.0%} of its standard deviation), so the row mean would only show "
            f"that normalisation. ADA follows each row's standard deviation across them instead "
            f"({row_spread_name(len(columns))}), which says how much the row moves"
        )
    note = (
        f"This table looks like a matrix: {len(columns):,} {kind} of one kind, named "
        f"{columns[0]} … {columns[-1]}{each}. No single column is the measure, {follows}. "
        f"To compare the columns themselves, reshape the table to long form first: one row per "
        f"{rows or 'original row'} and column, with the value in a single column."
    )
    if counts:
        totals = dataframe[columns].sum().to_numpy(dtype=float)
        if totals.min() > 0 and totals.max() >= 2 * totals.min():
            note += (
                f" The columns' totals differ up to {totals.max() / totals.min():,.0f}-fold, so put "
                "them on a common scale, such as shares of each column's total, before comparing them."
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


# A parameter found by how the rows behave names at most this many things measured.
MAX_PARAMETER_LEVELS = 30
# Each entity holds one row per parameter on this share of rows or more, and
# this share of entities hold more than one parameter.
PAIR_UNIQUE_SHARE = 0.95
REPEATING_ENTITIES = 0.5


def _parameter_by_rows(dataframe: pd.DataFrame, measure: str, text_columns: list[str]) -> str | None:
    """A column naming what each row measures, found by how the rows behave rather than by its name.

    In a table of stations and pollutants, "pollutant_id" holds OZONE, PM2.5,
    NO2 and the rest: each station has one row per pollutant, most stations
    have several, and the reading's typical size differs tenfold between them.
    Money and counts are never split this way, since revenue by category is
    one quantity whatever the sizes, and nor is a signed reading.
    """
    if is_additive_name(measure) or is_currency(measure) or not text_columns:
        return None
    data = dataframe[dataframe[measure].notna()]
    # Sizes that differ tenfold are sizes of different things only on a scale
    # starting at zero; a log ratio or a z-score is signed.
    if data.empty or float(data[measure].min()) < 0:
        return None
    sample = _sample(data)
    # A column with more values in the sample than a parameter may have has more in all.
    screened = [column for column in text_columns if 2 <= sample[column].nunique() <= MAX_PARAMETER_LEVELS]
    parameters = [column for column in screened if _spans_scales(sample, column, measure)]
    if not parameters:
        return None
    codes = {column: pd.factorize(data[column])[0] for column in text_columns}
    levels = {column: int(found.max(initial=-1)) + 1 for column, found in codes.items()}
    for column in parameters:
        if not 2 <= levels[column] <= MAX_PARAMETER_LEVELS:
            continue
        for entity in text_columns:
            if entity == column or levels[entity] <= levels[column]:
                continue
            both = (codes[entity] >= 0) & (codes[column] >= 0)
            if not both.any():
                continue
            pairs = codes[entity][both].astype(np.int64) * levels[column] + codes[column][both]
            distinct = np.unique(pairs)
            if distinct.size < PAIR_UNIQUE_SHARE * pairs.size:
                continue
            per_entity = np.bincount(distinct // levels[column], minlength=levels[entity])
            present = per_entity[per_entity > 0]
            if float((present >= 2).mean()) >= REPEATING_ENTITIES:
                return column
    return None


def _long_layout(
    dataframe: pd.DataFrame, measure: str | None, text_columns: list[str]
) -> tuple[str | None, str | None]:
    """Find the column naming what each row measures, and the column naming its unit.

    A text column named like a parameter qualifies when the measure takes very
    different sizes across its values, or a unit column holds several units.
    A unit column alone, with several units, is used as the parameter. Failing
    both, a column whose rows behave like parameters qualifies whatever its
    name (see ``_parameter_by_rows``).
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
    measured = dataframe[measure].notna()
    parameter = next(
        (
            column
            for column in text_columns
            if column != units
            and (_core_words(column) or [""])[-1] in PARAMETER_WORDS
            and 2 <= dataframe[column].nunique() <= dataframe[column].notna().sum() / 2
            # A parameter names what most values measure, not a note on a few of them.
            and float(dataframe.loc[measured, column].notna().mean()) >= 0.5
        ),
        None,
    )
    if parameter and not units and not _spans_scales(dataframe, parameter, measure):
        parameter = None
    if units and not parameter and not _looks_like_units(dataframe[units]):
        units = None
    if not parameter and not units:
        parameter = _parameter_by_rows(dataframe, measure, text_columns)
    return parameter or units, units


def _long_focus(
    dataframe: pd.DataFrame, parameter: str, units: str | None, measure: str
) -> tuple[str, str] | None:
    """Pick the parameter the brief covers, the most common, and describe the choice.

    None when no row has both a parameter and a value, so there is nothing to pick.
    """
    measured = dataframe[dataframe[measure].notna()]
    counts = measured[parameter].value_counts()
    if counts.empty:
        return None
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


def _listed(columns: list[str]) -> str:
    """The columns as a sentence lists them: "A", "A and B", "A, B and C", "A, B, C and 2 more"."""
    if len(columns) > NAMED_IN_NOTES:
        return _named(columns)
    names = [str(column) for column in columns]
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


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
    if _named_like_a_date(column):
        return (
            f"{column} holds a different code on every row, not times, so despite its name it is read "
            "as the row label."
        )
    return f"{column} is different on every row, so it is read as the row label."


# How the notes describe a running total; ``running_total_columns`` reads it back.
RUNNING_TOTAL_PHRASE = "keeps climbing, dropping back only when it resets"


def _running_total_note(counter: str, sibling: str | None) -> str:
    note = (
        f"{counter} {RUNNING_TOTAL_PHRASE}, like a gauge or meter "
        "reading, so adding it up or averaging it over a period would mean nothing; it is not "
        "used as the measure."
    )
    if sibling:
        note += f" {sibling} records each reading's own amount, so it is used instead."
    return note


def _names_a_statistic(name: str) -> bool:
    """Whether the name marks a statistic of another quantity: "p_value", "beta", "overall_sd", "Z_SCORE"."""
    words = _head_words(name)
    pairs = zip(words, words[1:], strict=False)
    if set(words) & STATISTIC_WORDS or any(pair in STATISTIC_PAIRS for pair in pairs):
        return True
    # "SD" alone is ambiguous; beside another word ("low_mp_sd") it is a spread.
    return len(words) > 1 and "sd" in words


def _numbers_rows(name: str, series: pd.Series) -> bool:
    """Whether the column holds a different whole number on every row, in order.

    A key or a date stored as a number (19540601, 19540602 …) does. A table
    sorted by population still holds populations, so a name that marks a
    quantity keeps the column a candidate.
    """
    if set(_head_words(name)) & QUANTITY_WORDS or is_bool_dtype(series):
        return False
    sample = _sample(series).dropna().to_numpy(dtype=float)
    steps = np.diff(sample)
    if len(sample) < MIN_ROWS_FOR_KEY or not ((steps > 0).all() or (steps < 0).all()):
        return False
    numbers = series.dropna().to_numpy(dtype=float)
    if not np.all(numbers % 1 == 0):
        return False
    # Climbing or falling at every step, so no value repeats.
    steps = np.diff(numbers)
    return bool((steps > 0).all() or (steps < 0).all())


def _constant(series: pd.Series) -> bool:
    return _sample(series).nunique() <= 1 and series.nunique() <= 1


def _explained(values: pd.Series, groups: pd.Series) -> float:
    """How much of the variation in the values' ranks the groups account for, adjusted for their number.

    Ranks keep a few extreme values from deciding, and the adjustment (that of
    an adjusted R²) keeps a column of many small groups from winning by chance.
    Judged on an even sample of the rows where both are present.
    """
    data = pd.DataFrame({"value": _sample(values).to_numpy(), "group": _sample(groups).to_numpy()}).dropna()
    rows, levels = len(data), int(data["group"].nunique())
    if levels < 2 or rows <= levels:
        return float("-inf")
    ranks = data["value"].rank().to_numpy()
    total = float(((ranks - ranks.mean()) ** 2).sum())
    if total == 0:
        return 0.0
    group_means = pd.Series(ranks).groupby(data["group"].to_numpy()).transform("mean").to_numpy()
    within = float(((ranks - group_means) ** 2).sum())
    return 1 - (within / (rows - levels)) / (total / (rows - 1))


SUMMARY_STATISTIC_NAMES = frozenset({
    "mean", "std", "sd", "min", "max", "median", "count", "25%", "50%", "75%",
})


def _summary_statistics(dataframe: pd.DataFrame, candidates: list[str]) -> tuple[str, str] | None:
    """The mean column of a table whose rows each summarise a set of values, and a note, or None.

    Columns named count, mean, std, min, the quartiles and max, as a summary of
    each probe or gene across its samples, are statistics of one set of values
    per row; the mean is what the row measures.
    """
    names = {normalized_name(column): column for column in candidates}
    if not {"mean", "min", "max"} <= names.keys() or not names.keys() & {"std", "sd", "median", "50%", "25%"}:
        return None
    mean = names["mean"]
    statistics = [column for name, column in names.items() if name in SUMMARY_STATISTIC_NAMES]
    note = (
        f"Each row summarises a set of values: {_named(statistics)} are statistics of them, so ADA "
        f"uses {mean} as the measure. The other statistics come from the same values, so they "
        "move with it by construction."
    )
    count = names.get("count")
    if count is not None and _constant(dataframe[count]) and dataframe[count].notna().any():
        size = format_number(float(dataframe[count].dropna().iloc[0]), compact=False)
        note += f" {count} is {size} on every row: each row summarises that many values."
    return mean, note


def running_total_columns(roles: ColumnRoles) -> tuple[str, ...]:
    """The columns the notes set aside as running totals, which nothing should add up or average."""
    marker = f" {RUNNING_TOTAL_PHRASE}"
    return tuple(note.split(marker)[0] for note in roles.notes if marker in note)


def _source_words(source_name: str | None) -> frozenset[str]:
    """Telling words of a file name: "secchi" in "georgian_bay_secchi.csv", none in "data.csv"."""
    if not source_name:
        return frozenset()
    stem = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", re.split(r"[\\/]", str(source_name))[-1])
    words = re.findall(r"[a-z]+", normalized_name(stem))
    return frozenset(word for word in words if len(word) >= 3 and word not in GENERIC_FILE_WORDS)


def _leading_reading(columns: list[str], candidates: list[str], groupings: set[str]) -> str | None:
    """The first of ``candidates`` in ``columns`` when only columns that lay out a design come before it.

    A table of experimental runs lists the case, the run and the material
    first and the response after them (wear, then the sensor channels), so
    this is asked only of tables with such units (see ``_block_units``).
    Keys, dates and columns set aside may come before it too, but no other
    candidate, and at least one of ``groupings``.
    """
    before = False
    for column in columns:
        if column in candidates:
            # A time point or a dose filed early is a setting, not the response.
            words = set(_head_words(column))
            timed = bool(words & (DESIGN_LEVEL_WORDS | TIME_PART_WORDS)) or _has_part(column, ("time",))
            return column if before and not timed else None
        before = before or column in groupings
    return None


def _named_in(columns: list[str], words: frozenset[str]) -> list[str]:
    """The one or two of ``columns`` whose name shares a word with the file name.

    A word most columns share ("PROPERTY: ...") says nothing about which one
    the table is about.
    """
    if not words:
        return []
    named = [column for column in columns if set(_head_words(column)) & words]
    return named if len(named) <= 2 else []


def _family_words(name: str) -> tuple[str, str] | None:
    """The first and last words of a name joined by underscores, dots or spaces: ("density", "total")."""
    parts = [part for part in re.split(r"[_.\s]+", str(name).strip().lower()) if part]
    return (parts[0], parts[-1]) if len(parts) >= 2 else None


def _lone_outside_family(columns: list[str]) -> str | None:
    """The one column outside a family of names that all the others belong to.

    Eleven descriptors named X_Average and X_Total and one "Hardness": the
    descriptors are what was computed, and the lone column is what they are
    meant to explain. Columns sharing a first or a last word are kin
    ("density_Total" and "density_Average", "density_Average" and
    "val_e_Average"), and a family is all the kin of kin.
    """
    if len(columns) < 4:
        return None
    pairs = {column: _family_words(column) for column in columns}
    kin: dict[str, set[str]] = {}
    for column, pair in pairs.items():
        if pair:
            kin.setdefault("first " + pair[0], set()).add(column)
            kin.setdefault("last " + pair[1], set()).add(column)
    family: set[str] = set()
    links = [group for group in kin.values() if len(group) >= 2]
    if links:
        family = set(max(links, key=len))
        grown = True
        while grown:
            grown = False
            for group in links:
                if group & family and not group <= family:
                    family |= group
                    grown = True
    outside = [column for column in columns if column not in family]
    return outside[0] if len(outside) == 1 else None


# Words naming a summary of a reading's values, and its extremes, as in
# pollutant_min, pollutant_avg and pollutant_max.
AVERAGE_WORDS = frozenset({"avg", "average", "mean"})
EXTREME_WORDS = frozenset({"min", "max", "minimum", "maximum", "low", "high"})


def _averages_of_extremes(columns: list[str]) -> set[str]:
    """Columns averaging a reading recorded beside its minimum or maximum.

    The average is the reading itself; the extremes are its edges.
    """
    stems: dict[tuple[str, ...], set[str]] = {}
    for column in columns:
        words = _head_words(column)
        for position, word in enumerate(words):
            if word in AVERAGE_WORDS | EXTREME_WORDS:
                stem = tuple(words[:position] + words[position + 1:])
                stems.setdefault(stem, set()).add("average" if word in AVERAGE_WORDS else "extreme")
    found = set()
    for column in columns:
        words = _head_words(column)
        for position, word in enumerate(words):
            if word in AVERAGE_WORDS and stems.get(tuple(words[:position] + words[position + 1:])) == {
                "average", "extreme",
            }:
                found.add(column)
    return found


# A column within this share of its own interquartile range of the row mean of
# the others, and moving with it this closely, is that mean.
MEAN_OF_OTHERS_TOLERANCE = 0.1
MEAN_OF_OTHERS_CORRELATION = 0.95
MAX_MEAN_CANDIDATES = 60


def _mean_of_others(dataframe: pd.DataFrame, columns: list[str]) -> str | None:
    """The column that is, row by row, the mean of all the other columns, as an annual mean is of its months.

    It is the summary the others build, so it is what the table is about.
    """
    if not 4 <= len(columns) <= MAX_MEAN_CANDIDATES:
        return None
    values = _thin(dataframe[columns]).to_numpy(dtype=float)
    if len(values) < MIN_ROWS_FOR_KEY:
        return None
    present = ~np.isnan(values)
    filled = np.where(present, values, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        # Each column against the mean of the others on its row.
        others = (filled.sum(axis=1, keepdims=True) - filled) / (present.sum(axis=1, keepdims=True) - present)
    for index, column in enumerate(columns):
        both = present[:, index] & np.isfinite(others[:, index])
        if int(both.sum()) < MIN_ROWS_FOR_KEY:
            continue
        own, rest = values[both, index], others[both, index]
        # The interquartile range, so a few huge values cannot make any gap look small.
        low, high = np.percentile(own, [25, 75])
        if not high > low or own.std() == 0 or rest.std() == 0:
            continue
        if float(np.corrcoef(own, rest)[0, 1]) < MEAN_OF_OTHERS_CORRELATION:
            continue
        if float(np.median(np.abs(own - rest))) <= MEAN_OF_OTHERS_TOLERANCE * (high - low):
            return column
    return None


_WITHOUT_TOTAL = {word: score for word, score in MEASURE_KEYWORDS.items() if word not in TOTAL_WORDS}


def _names_money(columns: list[str]) -> bool:
    """Whether any of the columns names money or a count of things in a word other than "total"."""
    counted = ADDITIVE_WORDS - TOTAL_WORDS
    return any(is_currency(column) or set(_head_words(column)) & counted for column in columns)


def _totals_of_items(columns: list[str]) -> set[str]:
    """Columns totalling numbered items beside them: "CorrectTotal" beside "Correct1" to "Correct4"."""
    stems = {_family_stem(column) for column in columns} - {None}
    found = set()
    for column in columns:
        words = _head_words(column)
        rest = " ".join(word for word in words if word not in TOTAL_WORDS)
        if len(rest.split()) < len(words) and rest in stems:
            found.add(column)
    return found


def _separations(dataframe: pd.DataFrame, columns: list[str], outcome: str) -> dict[str, float]:
    """How much of each column's rank variation the two groups of ``outcome`` account for, adjusted.

    The same figure as ``_explained``, worked out for many columns at once on
    an even sample of the rows where the outcome is recorded.
    """
    data = _thin(dataframe[[*columns, outcome]])
    data = data[data[outcome].notna()]
    codes = pd.factorize(data[outcome])[0]
    ranks = data[columns].rank().to_numpy(dtype=float)
    present = ~np.isnan(ranks)
    rows = present.sum(axis=0)
    means = np.nansum(ranks, axis=0) / np.maximum(rows, 1)
    total = np.nansum((ranks - means) ** 2, axis=0)
    within = np.zeros(len(columns))
    levels = np.zeros(len(columns))
    for code in np.unique(codes):
        member = (codes == code)[:, None] & present
        size = member.sum(axis=0)
        group_mean = np.where(member, ranks, 0).sum(axis=0) / np.maximum(size, 1)
        within += np.where(member, (ranks - group_mean) ** 2, 0).sum(axis=0)
        levels += size > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        adjusted = 1 - (within / (rows - levels)) / (total / (rows - 1))
    usable = (levels >= 2) & (rows > levels) & (total > 0)
    return {
        column: float(adjusted[index]) if usable[index] else float("-inf")
        for index, column in enumerate(columns)
    }


# Only readings whose ranks tell the outcome's two groups apart this nearly
# (the area under the curve, either way round) are checked for a cut point.
NEARLY_APART = 0.98


def _nearly_apart(dataframe: pd.DataFrame, columns: list[str], outcome: str) -> list[str]:
    """The columns whose values all but separate the two groups of ``outcome``, judged on their ranks at once.

    A cut point that places 99.5% of rows needs ranks almost entirely apart,
    so this screens the many columns of a wide table before ``_sets_outcome``
    checks each survivor.
    """
    if not columns:
        return []
    data = _thin(dataframe[[*columns, outcome]])
    data = data[data[outcome].notna()]
    codes = pd.factorize(data[outcome])[0]
    if codes.max(initial=-1) != 1:
        return []
    ranks = data[columns].rank().to_numpy(dtype=float)
    present = ~np.isnan(ranks)
    first = (codes == 1)[:, None] & present
    ones, zeros = first.sum(axis=0), (present & ~first).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        area = (np.where(first, ranks, 0).sum(axis=0) - ones * (ones + 1) / 2) / (ones * zeros)
    apart = np.maximum(area, 1 - area) >= NEARLY_APART
    return [column for index, column in enumerate(columns) if apart[index]]


def _sets_outcome(values: pd.Series, outcome: pd.Series) -> float | None:
    """The share of rows one cut point of ``values`` puts in their own group of a two-level ``outcome``.

    None when the outcome does not have two levels with rows to spare, and 0
    when the cut misplaces a tenth or more of the smaller group. A reading the
    outcome was derived from by a threshold places nearly every row; a merely
    related one does not.
    """
    data = pd.DataFrame({"x": _thin(values).to_numpy(), "y": _thin(outcome).to_numpy()}).dropna()
    codes = pd.factorize(data["y"])[0]
    if len(data) < MIN_ROWS_FOR_KEY or codes.max(initial=-1) != 1:
        return None
    ones_total = int(codes.sum())
    if min(ones_total, len(codes) - ones_total) < MIN_ROWS_PER_LEVEL:
        return None
    order = np.argsort(data["x"].to_numpy(dtype=float), kind="stable")
    x, y = data["x"].to_numpy(dtype=float)[order], codes[order]
    # Cuts fall only between distinct values, after the last row of each run.
    ends = np.r_[np.flatnonzero(x[1:] != x[:-1]), len(x) - 1]
    below = ends + 1
    ones_below = np.cumsum(y)[ends]
    zeros_below = below - ones_below
    rows = len(y)
    zeros_first = zeros_below + ones_total - ones_below
    ones_first = ones_below + (rows - below) - (ones_total - ones_below)
    wrong = rows - int(max(zeros_first.max(), ones_first.max()))
    # Putting every row in the larger group already places most of a lopsided outcome.
    if wrong > 0.1 * min(ones_total, rows - ones_total):
        return 0.0
    return 1 - wrong / rows


def _choose_measure(
    dataframe: pd.DataFrame,
    candidates: list[str],
    preferred: set[str],
    label: str | None = None,
    outcome: str | None = None,
    source_words: frozenset[str] = frozenset(),
    leading: str | None = None,
) -> tuple[str | None, tuple[str, ...]]:
    """Pick the measure, and the runners-up when the pick was a close call.

    Constant columns and whole numbers that differ on every row in order (a
    date or a key stored as a number) are never picked. The name scores first:
    money and outcome words up; keys, design labels, inputs, checks, statistics
    of other columns and near-constant or thinly recorded columns down. "Total"
    counts as a money word only beside other money or counts, or for the
    total of numbered items (see ``_totals_of_items``): among lab readings,
    total nitrogen is all forms of nitrogen. The column a file is
    named for, the one outside a family of computed descriptors (see
    ``_lone_outside_family``), the mean of the others or of a reading's
    extremes, a reading in a physical unit of its own, and ``leading``, the
    first reading after the columns that lay out a design (see
    ``_leading_reading``), score up too.

    With a two-level ``outcome``, such as a diagnosis the table is built to
    explain, every candidate within OUTCOME_WINDOW points of the best is
    judged by how well it tells the two groups apart; readings that set the
    outcome by a threshold (see ``_sets_outcome``) are left out beforehand,
    since they only restate it. Otherwise ties go to the column that best
    tells apart the groups of ``label``, an outcome class, when there is one;
    then to the column holding the most information, then to fractional
    values, completeness and the name, so column order alone never decides.
    ``preferred`` columns record a running total's own amounts and are
    favoured over it.
    """
    candidates = [
        column for column in candidates
        if not _constant(dataframe[column]) and not _numbers_rows(column, dataframe[column])
    ]
    unit_outcome = _lone_unit(candidates)
    lone = _lone_outside_family(candidates)
    named = set(_named_in(candidates, source_words))
    summaries = _averages_of_extremes(candidates) | {_mean_of_others(dataframe, candidates)}
    own_units = _own_units(candidates)
    keywords = MEASURE_KEYWORDS if _names_money(candidates) else _WITHOUT_TOTAL
    item_totals = _totals_of_items(candidates)
    coverage = {column: float(dataframe[column].notna().mean()) for column in candidates}
    best_coverage = max(coverage.values(), default=0.0)
    scored: list[tuple[int, float, float, float, float, str]] = []
    covariates: set[str] = set()
    nearly_constant: set[str] = set()
    for column in candidates:
        series = dataframe[column]
        words = _head_words(column)
        head = words[-1] if words else ""
        # A statistic of another column is no "value" or "score" of its own.
        statistic = _names_a_statistic(column)
        named_by = MEASURE_KEYWORDS if column in item_totals else keywords
        score = -6 if statistic else _keyword_score(column, named_by)
        if looks_like_identifier(column, series):
            score -= 20
        if _places_rows(column, series):
            score -= 15
        if head in INPUT_WORDS:
            score -= 4
        if set(words) & CHECK_WORDS or head == "check":
            score -= 6
        if coverage[column] < SPARSE_COVERAGE * best_coverage:
            score -= 6
        elif coverage[column] < THIN_COVERAGE * best_coverage:
            score -= 3
        if column in (unit_outcome, lone):
            score += 4
        if column in named:
            score += 4
        if column in summaries:
            score += 3
        if column in own_units:
            score += 2
        if column == leading and _top_share(series) <= NEARLY_CONSTANT:
            score += 1
        if column in preferred:
            score += 10
        if _top_share(series) > NEARLY_CONSTANT:
            score -= 6
            nearly_constant.add(column)
        if set(words) & (COVARIATE_TOKENS | SIZE_COVARIATE_TOKENS):
            covariates.add(column)
        # Amounts carry fractions; postal codes and counters do not. A weak
        # signal, so it only breaks ties.
        continuous = float(not is_bool_dtype(series) and bool((series.dropna() % 1 != 0).any()))
        scored.append((score, 0.0, _information(series), continuous, coverage[column], column))
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
    top = max(item[0] for item in scored)
    if outcome is not None:
        window = [
            column for score, *_, column in scored
            if score >= top - OUTCOME_WINDOW and column not in covariates
        ]
        if len(window) > 1:
            # The reading that best tells the outcome's two groups apart is
            # what the table is about, whatever weak keyword another carries.
            separation = _separations(dataframe, window, outcome)
            scored = [
                (top, separation[column], *rest, column)
                if column in separation
                else (score, 0.0, *rest, column)
                for score, _, *rest, column in scored
            ]
    elif label is not None and sum(item[0] == top for item in scored) > 1:
        # Names cannot choose between these, so the one that best separates
        # the outcome classes is the reading the table is about.
        scored = [
            (score, _explained(dataframe[column], dataframe[label]) if score == top else 0.0, *rest, column)
            for score, _, *rest, column in scored
        ]
    ranked = sorted(scored, key=lambda item: (-item[0], -item[1], -item[2], -item[3], -item[4], item[5]))
    best = ranked[0]
    close = [
        item[5] for item in ranked[1:]
        if item[0] >= best[0] - CLOSE_CALL_POINTS and item[5] not in nearly_constant
    ]
    return best[5], tuple(close[:MAX_ALTERNATIVES])


def _shares_rows(dataframe: pd.DataFrame, grouping: str, measured: np.ndarray) -> bool:
    """Whether a grouping and the measure are recorded together on enough rows, in two groups or more.

    ``measured`` marks the rows where the measure is recorded.
    """
    both = dataframe[grouping].notna().to_numpy() & measured
    needed = max(min(MIN_SHARED_ROWS, 0.5 * int(measured.sum())), 2)
    if int(both.sum()) < needed:
        return False
    shared = dataframe[grouping].to_numpy()[both]
    return bool((shared != shared[0]).any())


def _refines(dataframe: pd.DataFrame, fine: str, coarse: str) -> bool:
    """Whether each group of ``fine`` lies within one of ``coarse``, as specimens lie within materials."""
    data = _sample(dataframe[[fine, coarse]]).dropna()
    levels = int(data[fine].nunique())
    return levels > data[coarse].nunique() and len(data.drop_duplicates()) == levels


def _same_groups(first: np.ndarray, second: np.ndarray) -> bool:
    """Whether two columns split the rows where both are recorded into the same groups.

    Each column comes as its group codes from ``pd.factorize``, -1 where it is missing.
    """
    both = (first >= 0) & (second >= 0)
    if not both.any():
        return False
    first, second = first[both].astype(np.int64), second[both].astype(np.int64)
    pairs = np.unique(first * (int(second.max()) + 1) + second).size
    return pairs == np.unique(first).size == np.unique(second).size


def _outcome_label(dataframe: pd.DataFrame, ranked: list[str]) -> str | None:
    """The grouping its name ranks first, when it names an outcome class such as a diagnosis or a result."""
    if not ranked:
        return None
    column = ranked[0]
    named = (_core_words(column) or [""])[-1] in OUTCOME_LABEL_WORDS
    return column if named and 2 <= dataframe[column].nunique() <= MAX_OUTCOME_CLASSES else None


def _choose_dimension(
    dataframe: pd.DataFrame, candidates: list[tuple[int, int, str]], measure: str | None
) -> str | None:
    """Pick the grouping from scored candidates, each (score, closeness to ten groups, column).

    With a measure, a grouping must share rows with it (see ``_shares_rows``):
    groups that never hold a value compare nothing. The best name score wins.
    Among equal scores, the number of groups closest to ten wins, unless
    another grouping explains clearly more of the measure (MIN_EXPLAINED, and
    EXPLAINED_MARGIN more): a frequency that explains 70% of a signal beats the
    materials that explain 1%. One that only splits another's groups further,
    as specimens split materials, is left out of that comparison.
    """
    measured = dataframe[measure].notna().to_numpy() if measure else None
    # The best score among groupings that share rows with the measure, checked
    # one score at a time since most tables settle it at the first.
    for top in sorted({candidate[0] for candidate in candidates}, reverse=True):
        tied = [
            candidate
            for candidate in candidates
            if candidate[0] == top and (measured is None or _shares_rows(dataframe, candidate[2], measured))
        ]
        if tied:
            break
    else:
        return None
    sized = max(tied, key=lambda candidate: candidate[1])
    if measure is None or len(tied) == 1:
        return sized[2]
    coarse = [
        candidate
        for candidate in tied
        if not any(other is not candidate and _refines(dataframe, candidate[2], other[2]) for other in tied)
    ] or tied
    explained = {column: _explained(dataframe[measure], dataframe[column]) for *_, column in coarse}
    best = max(coarse, key=lambda candidate: (explained[candidate[2]], candidate[1]))
    if sized[2] not in explained:
        explained[sized[2]] = _explained(dataframe[measure], dataframe[sized[2]])
    baseline = explained[sized[2]]
    if explained[best[2]] >= MIN_EXPLAINED and explained[best[2]] - baseline >= EXPLAINED_MARGIN:
        return best[2]
    return sized[2]


def _twins(
    dataframe: pd.DataFrame, dimension: str | None, candidates: list[str], factors: set[str]
) -> tuple[str | None, list[str]]:
    """Swap the grouping for a twin that reads better, and name columns that split the rows alike.

    A twin splits the rows where both are recorded into the same groups. One
    recorded on clearly more rows is preferred (a cancer type left blank for a
    fifth of the rows, beside a complete one), and so is text in place of a
    numeric code (labels T0 … D6 for codes 1 … 12).
    """
    notes: list[str] = []
    described: set[str] = set()
    codes = {column: pd.factorize(_sample(dataframe[column]))[0] for column in candidates}
    coverage = {column: float((codes[column] >= 0).mean()) for column in candidates}
    if dimension:
        for other in candidates:
            if other == dimension or not _same_groups(codes[dimension], codes[other]):
                continue
            if coverage[other] >= coverage[dimension] + COMPLETENESS_GAIN:
                notes.append(
                    f"{other} holds the same groups as {dimension} where both are recorded, and is "
                    f"recorded on {coverage[other]:.0%} of rows against {coverage[dimension]:.0%}, so "
                    f"ADA groups by {other}."
                )
            elif dimension in factors and other not in factors and coverage[other] >= coverage[dimension]:
                notes.append(
                    f"{dimension} and {other} split the rows into the same groups; ADA groups by "
                    f"{other}, whose values are labels rather than numbers."
                )
            else:
                continue
            described = {dimension, other}
            dimension = other
            break
    levels: dict[int, list[str]] = {}
    for column in sorted(candidates, key=list(dataframe.columns).index):
        if coverage[column] > 0:
            levels.setdefault(int(codes[column].max()) + 1, []).append(column)
    pairs = [
        f"{first} and {second}"
        for alike in levels.values()
        for position, first in enumerate(alike)
        for second in alike[position + 1:]
        if {first, second} != described
        and abs(coverage[first] - coverage[second]) < 0.01
        and _same_groups(codes[first], codes[second])
    ]
    if pairs:
        listed = "; ".join(pairs[:NAMED_IN_NOTES])
        more = f"; and {len(pairs) - NAMED_IN_NOTES:,} more pairs" if len(pairs) > NAMED_IN_NOTES else ""
        notes.append(
            f"These columns split the rows into the same groups, so comparing by either gives the "
            f"same result: {listed}{more}."
        )
    return dimension, notes


def _factors(
    dataframe: pd.DataFrame, numeric: list[str], text_dimensions: list[str]
) -> tuple[list[str], list[str]]:
    """The numeric columns holding category codes, and the 0/1 flags among them.

    Weak evidence (a short run of whole numbers, a design level, a swept
    setting) is dropped when it would leave nothing to measure: marks out of 10
    are the quantity of a table that has nothing else. Rating items and counts
    read like their siblings are never codes (see ``_scale_items`` and
    ``_sibling_counts``).
    """
    items = _scale_items(dataframe, numeric)
    counted = _sibling_counts(dataframe, numeric)
    kinds = {
        column: None if column in counted else _factor_kind(column, dataframe[column], column in items)
        for column in numeric
    }
    found = {column for column, kind in kinds.items() if kind}
    rest = [column for column in numeric if column not in found and column not in items | counted]
    groupings = [dataframe[column] for column in [*text_dimensions, *found]]
    # Rows repeat a recipe or a sample across the levels of a design column.
    groupings += [
        dataframe[column]
        for column in rest
        if _sample(dataframe[column]).nunique() <= len(_sample(dataframe[column])) / 2
    ][:10]
    for column in rest:
        if _is_design_level(column, dataframe[column], groupings) or _is_sweep(column, dataframe[column]):
            kinds[column] = "code"
            found.add(column)
    if len(found) == len(numeric):
        found = {column for column in found if kinds[column] != "code"}
    factors = [column for column in numeric if column in found]
    return factors, [column for column in factors if kinds[column] == "flag"]


def _stacked_tables(dataframe: pd.DataFrame) -> tuple[str, str] | None:
    """The column naming which stacked table each row belongs to, and the largest table, or None.

    A file that stacks a table of foods on a table of blood markers holds each
    table's columns only on that table's rows. A text column with 2 to
    MAX_STACKED_TABLES values qualifies when its largest table and another each
    own at least two columns the other lacks, together at least
    STACKED_COLUMN_SHARE of the file's columns. A table whose rows merely leave
    a few columns blank, such as a refund amount filled only on returned
    orders, does not.
    """
    sample = _sample(dataframe)
    present = sample.notna()
    others = len(dataframe.columns) - 1
    missing = 1 - present.mean()
    if others < 4 or int((missing > 0).sum()) < 4:
        return None
    for column in dataframe.columns:
        series = sample[column]
        if is_numeric_dtype(series) or is_datetime64_any_dtype(series) or missing[column] > 0.05:
            continue
        counts = series.value_counts()
        if not 2 <= len(counts) <= MAX_STACKED_TABLES:
            continue
        # Columns another table owns are missing on nearly all of the largest
        # table's rows, so at least that share of the file's rows.
        if int((missing >= (1 - ABSENT_SHARE) * counts.iloc[0] / len(series)).sum()) < 2:
            continue
        shares = present.drop(columns=column).groupby(series).mean()
        largest = shares.loc[counts.index[0]]
        for table in counts.index[1:]:
            other = shares.loc[table]
            own = int(((largest >= OWNED_SHARE) & (other <= ABSENT_SHARE)).sum())
            theirs = int(((other >= OWNED_SHARE) & (largest <= ABSENT_SHARE)).sum())
            if own >= 2 and theirs >= 2 and own + theirs >= STACKED_COLUMN_SHARE * others:
                return column, str(counts.index[0])
    return None


def _stacked_note(dataframe: pd.DataFrame, column: str, focus: str) -> str:
    counts = dataframe[column].astype(str).value_counts()
    others = [table for table in counts.index if table != focus]
    listed = ", ".join(f"{table} ({counts[table]:,})" for table in others[:NAMED_IN_NOTES])
    more = f", and {len(others) - NAMED_IN_NOTES:,} more" if len(others) > NAMED_IN_NOTES else ""
    return (
        f"This file stacks {len(counts):,} tables, named in {column}, each with its own columns, so "
        f"their rows are never combined. The brief covers {focus}, the largest ({counts[focus]:,} "
        f"rows); the others are {listed}{more}. Filter the table to one of them to analyze it instead."
    )


# Last words of two-level columns that describe who was studied or how they
# were assigned, never the outcome a table is built around.
NOT_OUTCOME_FLAG_WORDS = frozenset({
    "sex", "gender", "race", "ethnicity", "treatment", "arm", "cohort", "group", "condition",
    "block", "batch", "phase", "site", "visit",
})


def _has_part(name: str, parts: tuple[str, ...]) -> bool:
    """Whether any of ``parts`` appears in the name, run together or not: "death" in "eventdeath"."""
    text = normalized_name(name).replace(" ", "")
    return any(part in text for part in parts)


def _outcome_flag(
    dataframe: pd.DataFrame,
    dimensions: list[str],
    flags: list[str],
    first_column: str | None,
    source_words: frozenset[str],
    candidates: list[str],
) -> tuple[str, str] | None:
    """The two-level column the table is built to explain, and how it was recognised, or None.

    In order: a column named as the label ("target", "classification"); the
    one the file is named for ("Landslide" in landslide_dataset.csv); a 0/1
    event recorded beside the time it was followed ("eventdeath" beside
    "survival"); or, when no column is named as a label, a 0/1 column filed
    straight after the identifiers (``first_column``), where a label to
    predict usually sits. Sex, treatment arms and the like
    describe who was studied, not what came of it, so they never qualify.
    """
    two = [
        column for column in dimensions
        if dataframe[column].nunique(dropna=True) == 2
        and (_core_words(column) or [""])[-1] not in NOT_OUTCOME_FLAG_WORDS
    ]
    labels = [column for column in dimensions if (_core_words(column) or [""])[-1] in OUTCOME_FLAG_WORDS]
    labelled = [column for column in labels if column in two]
    if labelled:
        return labelled[-1], "label"
    named = _named_in(two, source_words)
    if len(named) == 1:
        return named[0], "file"
    events = [column for column in two if column in flags and _has_part(column, EVENT_PARTS)]
    if len(events) == 1 and _follow_up_times(dataframe, candidates):
        return events[0], "event"
    # Only straight after a key, and only when no other column is named as the label.
    if first_column in two and first_column in flags and not labels:
        return first_column, "first"
    return None


def _follow_up_times(dataframe: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Columns timing how long each row was followed: "survival", "os_time", "followup_months"."""
    return [
        column for column in candidates
        if _has_part(column, DURATION_PARTS) and bool((dataframe[column].dropna() >= 0).all())
    ]


def _outcome_note(outcome: str, reason: str, measure: str | None) -> str:
    why = {
        "label": f"{outcome} names the label of each row",
        "file": f"{outcome} is what the file is named for",
        "event": f"{outcome} records whether the event happened",
        "first": f"{outcome}, a 0/1 column filed first after the identifier, reads as the label to explain",
    }[reason]
    then = f" and measures the reading that best tells its two groups apart, {measure}" if measure else ""
    return f"{why}, so ADA groups by it{then}."


def _settled_note(outcome: str, settled: list[tuple[str, float]]) -> str:
    column, share = settled[0]
    others = [name for name, _ in settled[1:]]
    also = f" So {'does' if len(others) == 1 else 'do'} {_listed(others)}." if others else ""
    return (
        f"{outcome} is set by {column}: one cut point puts {share:.1%} of rows in their own group, so "
        f"{column} restates {outcome} rather than differing by it, and is not the measure.{also}"
    )


def _follow_up_note(times: list[str], event: str) -> str:
    verb, pronoun = ("is", "it") if len(times) == 1 else ("are", "them")
    return (
        f"{_listed(times)} {verb} follow-up time, cut short on rows where {event} is 0, so ADA neither "
        f"averages {pronoun} nor compares {pronoun} across {event}: how long a row was followed "
        "depends on whether the event happened."
    )


MIN_BLOCK_UNITS = 5
MAX_SETTING_LEVELS = 4


def _block_units(
    dataframe: pd.DataFrame, columns: list[str], others: list[str]
) -> dict[str, list[str]]:
    """Numeric columns that number blocks of consecutive rows, each with the columns fixed within its blocks.

    An experiment number (case 1 to 16) or a subject ("indiv" 2.41) held on
    a run of rows each, under a material or a group that never changes within
    a run, names the unit the rows belong to; it is no more a quantity than a
    patient number. A reading sorted down the file repeats too, so the column
    must not be named as a quantity, must hold at least MIN_BLOCK_UNITS values
    on runs of three rows on average, and another column must stay fixed
    within every run while varying between them.
    """
    rows = len(dataframe)
    units: dict[str, list[str]] = {}
    columns = [column for column in columns if not is_bool_dtype(dataframe[column])]
    if not columns or rows < 2:
        return units
    # Each value on one unbroken run: as many runs as values. Runs are counted
    # for all columns at once, and values only where the runs are few enough.
    values = dataframe[columns].to_numpy(dtype=float)
    runs = 1 + (values[1:] != values[:-1]).sum(axis=0)
    few = [
        column for index, column in enumerate(columns)
        if MIN_BLOCK_UNITS <= runs[index] <= rows / 3 and not np.isnan(values[:, index]).any()
    ]
    for column in few:
        series = dataframe[column]
        if (
            runs[columns.index(column)] != series.nunique()
            or _named_as_quantity(column)
            or set(_head_words(column)) & (QUANTITY_WORDS | LABEL_NAME_WORDS)
            or is_level_name(column)
            or is_percentage(column)
            or _has_unit(column)
            or is_time_part(column, series)
        ):
            continue
        rest = [other for other in others if other != column]
        if not rest:
            continue
        spread = dataframe[rest].groupby(pd.factorize(series)[0]).nunique(dropna=False).max()
        fixed = [
            other for other in rest if spread[other] <= 1 and dataframe[other].nunique(dropna=False) >= 2
        ]
        if fixed:
            units[column] = fixed
    return units


def _block_note(units: dict[str, list[str]], settings: list[str]) -> str:
    unit, fixed = next(iter(units.items()))
    note = (
        f"{unit} numbers blocks of consecutive rows, within which {_listed(fixed)} "
        f"{'stays' if len(fixed) == 1 else 'stay'} fixed, so it is read as the unit each block belongs "
        "to, such as an experiment, a run or a subject, not as a quantity."
    )
    if len(units) > 1:
        note += f" So {'is' if len(units) == 2 else 'are'} {_listed(list(units)[1:])}."
    if settings:
        verb, pronoun = ("is", "it") if len(settings) == 1 else ("are", "them")
        note += (
            f" {_listed(settings)} {verb} set once per {unit}, so ADA groups by {pronoun} rather than "
            f"measuring {pronoun}."
        )
    return note


def detect_roles(dataframe: pd.DataFrame, source_name: str | None = None) -> ColumnRoles:
    """Infer likely roles from names, values, and cardinality, and the rows' noun from the file name."""
    roles = _file_roles(dataframe, source_name)
    noun = row_noun(source_name)
    # A word that is part of a column's name ("updrs" in total_UPDRS) names that column, not the rows.
    if noun and any(noun in re.findall(r"[a-z]+", normalized_name(column)) for column in dataframe.columns):
        noun = None
    return replace(roles, row_noun=noun) if noun else roles


# Plurals that don't end in s.
IRREGULAR_PLURALS = frozenset({"people", "children", "mice", "men", "women", "geese", "teeth", "feet"})


def row_noun(source_name: str | None) -> str | None:
    """The plural noun a file's name gives its rows: "penguins" for penguins.csv.

    The last plural among the telling words of the name (see ``_source_words``)
    counts, so olist_orders_dataset.csv gives "orders". A name with no plural,
    such as data.csv or ChickWeight.csv, gives None.
    """
    if not source_name:
        return None
    stem = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", re.split(r"[\\/]", str(source_name))[-1])
    words = [
        word for word in re.findall(r"[a-z]+", normalized_name(stem))
        if len(word) >= 3 and word not in GENERIC_FILE_WORDS
    ]
    plurals = [
        word for word in words
        if word in IRREGULAR_PLURALS
        or (len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "is", "us")))
    ]
    return plurals[-1] if plurals else None


def _file_roles(dataframe: pd.DataFrame, source_name: str | None = None) -> ColumnRoles:
    """Infer likely roles from names, values, and cardinality.

    ``source_name``, the file's name, counts for a column it names: the secchi
    depth of georgian_bay_secchi.csv is what that file is about. A file that
    stacks several tables (see ``_stacked_tables``) is read as its largest
    table, which becomes the ``focus`` of the column naming the tables, much as
    a long table's commonest parameter is. When that table is itself long and
    its measure is recorded nowhere else, its own parameter is the focus
    instead, which keeps readings in different units apart.
    """
    words = _source_words(source_name)
    stacked = _stacked_tables(dataframe)
    if stacked is None:
        return _detect_roles(dataframe, words)
    column, focus = stacked
    part = dataframe[dataframe[column].astype(str) == focus]
    roles = _detect_roles(part[[name for name in part.columns if part[name].notna().any()]], words)
    note = _stacked_note(dataframe, column, focus)
    measured = dataframe[roles.measure].notna() if roles.measure else None
    if roles.table_shape == "long" and measured is not None and measured.any() and bool(
        (dataframe.loc[measured, column].astype(str) == focus).all()
    ):
        # The largest table is itself long, and its measure is recorded nowhere
        # else, so its own parameter alone picks the rows the brief covers.
        return replace(roles, dimensions=(*roles.dimensions, column), notes=(note, *roles.notes))
    # The focus is applied later (``pipeline.apply_focus``) and regroups by the
    # first other dimension, so the table's own grouping goes first.
    follow = [roles.dimension] if roles.dimension else []
    return replace(
        roles,
        dimension=column,
        dimensions=(*follow, *(name for name in roles.dimensions if name not in follow), column),
        table_shape="stacked",
        focus=focus,
        notes=(note, *roles.notes),
    )


def _detect_roles(dataframe: pd.DataFrame, source_words: frozenset[str] = frozenset()) -> ColumnRoles:
    """The roles of one table, flat, long or a matrix (see ``detect_roles``)."""
    numeric = dataframe.select_dtypes(include=np.number).columns.tolist()
    # A date that never changes, such as a compilation stamp, places no row in time.
    date_columns = [
        column
        for column in dataframe.columns
        if is_datetime64_any_dtype(dataframe[column]) and dataframe[column].nunique() >= 2
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
    numbered = set(matrix)
    identifiers = _identifier_columns(
        dataframe, [column for column in dataframe.columns if column not in numbered]
    )
    identifier = identifiers[0] if identifiers else None
    # Counts spread across columns are a matrix even without numbered names,
    # when an id or a label names each row.
    counted_matrix = not matrix and identifier is not None
    if counted_matrix:
        matrix = _count_matrix_columns(dataframe, [column for column in numeric if column not in identifiers])
    in_matrix = set(matrix)
    notes.extend(
        _identifier_note(column, dataframe[column])
        for column in identifiers
        if not looks_like_identifier(column, dataframe[column])
    )
    keys = set(identifiers)
    numeric = [column for column in numeric if column not in keys]
    in_numeric = set(numeric)
    # Columns that flag the quality of other readings or place a row on a map
    # or a compass are never the measure, a grouping or a reading to compare.
    quality = [
        column for column in dataframe.columns
        if column not in keys and column not in in_matrix and is_quality_flag(column, dataframe.columns)
    ]
    places = [
        column for column in numeric
        if column not in in_matrix and set(_head_words(column)) & COORDINATE_TOKENS
    ]
    angles = [
        column for column in numeric
        if column not in in_matrix and column not in places and is_angle_column(column, dataframe[column])
    ]
    sequences = [
        column for column in numeric
        if column not in in_matrix and column not in places and _numbers_rows(column, dataframe[column])
    ]
    set_aside = {*quality, *places, *angles, *sequences}
    if set_aside:
        notes.append(_non_reading_note(quality, places, angles, sequences))

    text_columns: list[str] = []
    text_dimensions: list[str] = []
    for column in dataframe.columns:
        if column == date or column in in_numeric or column in keys or column in set_aside:
            continue
        series = dataframe[column]
        if _named_like_a_date(column) or is_datetime64_any_dtype(series):
            # Date-named columns are never segments, even when they didn't parse as dates.
            continue
        text_columns.append(column)
        if _groups_rows(series):
            text_dimensions.append(column)

    factors, flags = _factors(
        dataframe,
        [column for column in numeric if column not in in_matrix and column not in set_aside],
        text_dimensions,
    )
    if date:
        # Beside a date, a month or year column only repeats the timeline.
        factors = [column for column in factors if not is_time_part(column, dataframe[column])]
        flags = [column for column in flags if column in factors]
    order_in_table = {column: position for position, column in enumerate(dataframe.columns)}
    units = _block_units(
        dataframe,
        [column for column in numeric if column not in in_matrix and column not in set_aside | set(flags)],
        [column for column in dataframe.columns if column not in keys | set_aside and column != date],
    )
    if units:
        settings = sorted(
            {
                fixed for found in units.values() for fixed in found
                if fixed in in_numeric and fixed not in factors and fixed not in units
                and 2 <= dataframe[fixed].nunique() <= MAX_SETTING_LEVELS
            },
            key=order_in_table.__getitem__,
        )
        notes.append(_block_note(units, settings))
        factors = sorted({*factors, *units, *settings}, key=order_in_table.__getitem__)
    if factors:
        notes.append(_factor_note(dataframe, factors, flags))
    in_factors = set(factors)

    dimensions = sorted([*text_dimensions, *factors], key=order_in_table.__getitem__)
    dimension_candidates: list[tuple[int, int, str]] = []
    for column in dimensions:
        words = _head_words(column)
        if date and is_time_part(column, dataframe[column]):
            # Beside a date, a month or year column only repeats the timeline.
            continue
        series = dataframe[column]
        score = _keyword_score(column, DIMENSION_KEYWORDS)
        if is_design_name(column) or (words and words[-1] in INDIVIDUAL_NAME_WORDS) or column in units:
            # A plate row, a run label or one subject each groups by layout or by
            # individual, not by anything compared, so it is picked only when no
            # other column qualifies.
            score -= 15
        if _top_share(series) > LOPSIDED_GROUPING:
            score -= 5
        levels = int(series.nunique(dropna=True))
        if levels > MAX_LABEL_LEVELS:
            # Hundreds of groups make a poor headline, so such a column leads
            # only when nothing with as good a name has fewer.
            score -= 3
        dimension_candidates.append((score, -abs(levels - 10), column))
    ranked_dimensions = [
        column
        for *_, column in sorted(dimension_candidates, key=lambda candidate: (-candidate[0], -candidate[1]))
    ]
    # A first pick, by name alone, to follow running totals through each group.
    dimension = ranked_dimensions[0] if ranked_dimensions else None

    candidates = [
        column for column in numeric
        if column not in in_factors and column not in in_matrix and column not in set_aside
    ]
    order = _reading_order(dataframe, date, dimension)
    groups = pd.factorize(dataframe[dimension])[0][order] if dimension else np.zeros(len(order))
    counters = [
        column
        for column in candidates
        if _running_total(column, dataframe[column].to_numpy(dtype=float, na_value=np.nan)[order], groups)
    ]
    candidates = [column for column in candidates if column not in counters]
    siblings = {counter: _sibling_interval(counter, candidates) for counter in counters}
    # The column filed straight after the identifiers, when there are any.
    first_column = next(
        (column for column in dataframe.columns if column not in keys | set_aside and column != date), None
    ) if keys else None
    groupings = [candidate[2] for candidate in dimension_candidates]
    found = _outcome_flag(dataframe, groupings, flags, first_column, source_words, candidates)
    outcome, outcome_notes = (found[0] if found else None), []
    if found and found[1] == "event":
        times = _follow_up_times(dataframe, candidates)
        if [column for column in candidates if column not in times]:
            candidates = [column for column in candidates if column not in times]
            outcome_notes.append(_follow_up_note(times, outcome))
    if outcome:
        settled = [
            (column, share) for column in _nearly_apart(dataframe, candidates, outcome)
            if (share := _sets_outcome(dataframe[column], dataframe[outcome]) or 0.0) >= SET_BY_SHARE
        ]
        if settled and len(settled) < len(candidates):
            candidates = [column for column in candidates if column not in dict(settled)]
            outcome_notes.append(_settled_note(outcome, settled))
    summary = _summary_statistics(dataframe, candidates)
    if summary:
        measure, measure_alternatives = summary[0], ()
        notes.append(summary[1])
    else:
        measure, measure_alternatives = _choose_measure(
            dataframe,
            candidates,
            {sibling for sibling in siblings.values() if sibling},
            None if outcome else _outcome_label(dataframe, ranked_dimensions),
            outcome,
            source_words,
            _leading_reading(list(dataframe.columns), candidates, set(dimensions)) if units else None,
        )
    notes.extend(
        _running_total_note(counter, sibling if sibling == measure else None)
        for counter, sibling in siblings.items()
    )
    dimension = _choose_dimension(dataframe, dimension_candidates, measure)
    measured = dataframe[measure].notna().to_numpy() if measure else None
    if outcome and (measured is None or _shares_rows(dataframe, outcome, measured)):
        dimension = outcome
        notes.append(_outcome_note(outcome, found[1], measure))
        notes.extend(outcome_notes)
    dimension, twin_notes = _twins(dataframe, dimension, ranked_dimensions, in_factors)
    notes.extend(twin_notes)

    table_shape = "flat"
    focus = None
    parameter, units = _long_layout(dataframe, measure, text_columns)
    chosen = _long_focus(dataframe, parameter, units, measure) if parameter and measure else None
    if chosen:
        table_shape = "long"
        focus, note = chosen
        notes.append(note)
        # The focus is applied later (``pipeline.apply_focus``) and regroups by
        # the first other dimension, so the best grouping within it goes first.
        within = dataframe[dataframe[parameter].astype(str) == focus]
        follow = _choose_dimension(
            within,
            [candidate for candidate in dimension_candidates if candidate[2] not in (parameter, units)],
            measure,
        )
        dimensions = [*([follow] if follow else []), *(column for column in dimensions if column != follow)]
        dimension = parameter
        if parameter not in dimensions:
            dimensions.append(parameter)
    elif matrix:
        table_shape = "matrix"
        measure, measure_alternatives = None, ()
        notes.append(_matrix_note(dataframe, matrix, identifier, counted_matrix))

    return ColumnRoles(
        date=date,
        measure=measure,
        dimension=dimension,
        identifier=identifier,
        # Quantities first, so a list of what to measure leads with them.
        numeric=tuple(
            sorted(
                numeric,
                key=lambda column: column not in in_matrix and (
                    column in set_aside or _not_a_quantity(column, dataframe[column], column in in_factors)
                ),
            )
        ),
        dimensions=tuple(dimensions),
        factors=tuple(factors),
        flags=tuple(flags),
        measure_alternatives=measure_alternatives,
        table_shape=table_shape,
        focus=focus,
        notes=tuple(notes),
        # A running total or a message counter is no reading to compare or correlate either.
        non_readings=tuple(
            column for column in numeric
            if column in set_aside or column in counters
            or (column not in in_matrix and is_time_part(column, dataframe[column]))
        ),
    )


def add_row_summary(dataframe: pd.DataFrame, roles: ColumnRoles) -> tuple[pd.DataFrame, ColumnRoles]:
    """Give a matrix table each row's mean across its matrix columns, as its measure.

    A matrix has no single column to headline, so the brief follows the row
    mean, named for what it is. When the rows have been centred (see
    ``_rows_centred``) it follows each row's standard deviation instead. Other
    tables are returned unchanged.
    """
    if roles.table_shape != "matrix" or roles.measure:
        return dataframe, roles
    block = _matrix_columns(list(roles.numeric))
    if not block and roles.identifier:
        block = _count_matrix_columns(dataframe, list(roles.numeric))
    if not block:
        return dataframe, roles
    counts = bool(roles.identifier) and bool(_count_matrix_columns(dataframe, block))
    if not counts and _rows_centred(dataframe, block) is not None:
        name = row_spread_name(len(block))
        summary = dataframe[block].std(axis=1)
    else:
        name = row_summary_name(len(block))
        summary = dataframe[block].mean(axis=1)
    if name not in dataframe.columns:
        dataframe = pd.concat([dataframe, summary.rename(name)], axis=1)
    numeric = roles.numeric if name in roles.numeric else (*roles.numeric, name)
    return dataframe, replace(roles, measure=name, numeric=numeric)

