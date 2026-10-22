"""Deterministic natural-language queries over the analyzed dataset.

Each question becomes an explicit ``QueryPlan`` that pandas executes locally,
returning the answer with the calculation behind it. An optional AI planner may
emit the same plan shape for questions the rules cannot parse.

Every part of a question has to reach the plan: a column, a value, a grouping
or a condition the plan would leave out makes the parser refuse, with the
reason, rather than answer a simpler question under the original wording.
"""

from __future__ import annotations

import calendar
import re
import unicodedata
import weakref
from collections import defaultdict
from dataclasses import dataclass, field, fields, replace
from typing import Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from .aggregation import TrendSeries, build_trend, preferred_frequency, unlabelled_label
from .business_insights import (
    MIN_CORRELATION_PAIRS,
    RANK_DIVERGENCE,
    describe_correlation,
    measure_correlation,
    numeric_readings,
)
from .compare import (
    NOTABLE_CORRELATION,
    GroupComparison,
    compare_groups,
    comparison_calculation,
    comparison_table,
    correlation_interval,
    describe_comparison,
    effect_text,
    entity_column,
    format_p,
    scan_differences,
    scan_relationships,
)
from .formatting import (
    format_change,
    format_number,
    format_period,
    format_values_apart,
    group_label,
    list_names,
    normalized_name,
    sentence_case,
    with_aggregate,
)
from .metrics import metric_for
from .quality import VALUE_CHECKS, QualityIssue, prefers_log_scale, show_value
from .quality import scan as scan_quality
from .schema import ColumnRoles, running_total_columns
from .timeseries import period_positions, theil_sen

Intent = Literal[
    "aggregate", "count", "rank", "breakdown", "trend", "growth", "correlation", "compare", "correlates",
    "describe", "columns", "crosstab",
]
#: Plans answered from the column profile or the data checks rather than by combining a measure.
ProfileIntent = Literal["missing", "duplicates", "outliers"]
Aggregation = Literal[
    "sum", "mean", "median", "min", "max", "count", "std", "var", "iqr", "quantile", "mode", "share"
]
Operator = Literal["=", "<", "<=", ">", ">=", "missing"]
TimePart = Literal["hour", "weekday", "month", "quarter", "year"]

AGGREGATION_WORDS: dict[str, Aggregation] = {
    "total": "sum",
    "sum": "sum",
    "average": "mean",
    "avg": "mean",
    "mean": "mean",
    "typical": "mean",
    "median": "median",
    "max": "max",
    "maximum": "max",
    "highest": "max",
    "largest": "max",
    "biggest": "max",
    "greatest": "max",
    "longest": "max",
    "peak": "max",
    "min": "min",
    "minimum": "min",
    "lowest": "min",
    "smallest": "min",
    "shortest": "min",
}

# Words for the spread of a measure. "std" and "sd" name columns in exported
# summaries too; see _named_as_column.
SPREAD_WORDS: dict[str, Aggregation] = {
    "std": "std",
    "sd": "std",
    "stdev": "std",
    "stddev": "std",
    "variance": "var",
    "iqr": "iqr",
}
# Phrases for the same, read before single words claim theirs.
SPREAD_PHRASES: dict[str, Aggregation] = {
    "standard deviation": "std",
    "interquartile range": "iqr",
    "inter quartile range": "iqr",
}
AGGREGATION_WORDS.update(SPREAD_WORDS)

AGGREGATION_LABELS: dict[Aggregation, str] = {
    "sum": "Total",
    "mean": "Average",
    "median": "Median",
    "max": "Maximum",
    "min": "Minimum",
    "count": "Count of",
    "std": "Standard deviation of",
    "var": "Variance of",
    "iqr": "Interquartile range of",
    "quantile": "Percentile of",
    "mode": "Most common",
    "share": "Share of rows",
}
# Aggregations that describe how a measure spreads; they don't add up across groups.
SPREAD_AGGREGATIONS = ("std", "var", "iqr", "quantile")

# Words asking for the bottom of a ranking. Each is also a superlative, so
# SUPERLATIVE_WORDS includes this tuple rather than repeating it.
ASCENDING_WORDS = ("worst", "bottom", "lowest", "smallest", "least", "fewest", "shortest")
# Superlatives of temperature, read as the highest or lowest value of the
# measure, and said so in the answer.
WARM_WORDS = ("warmest", "hottest")
COLD_WORDS = ("coldest", "coolest")
ASCENDING_WORDS = (*ASCENDING_WORDS, *COLD_WORDS)
SUPERLATIVE_WORDS = (
    # "most" ranks from the top, like "highest"; "least" is in ASCENDING_WORDS.
    "best", "top", "leading", "highest", "largest", "biggest", "most", "greatest", "longest",
    "tallest", "heaviest", *WARM_WORDS,
    *ASCENDING_WORDS,
)
DECLINE_WORDS = ("slowest", "declined", "decreased", "dropped", "fell", "shrank")
ASCENDING_PATTERN = rf"\b({'|'.join((*ASCENDING_WORDS, *DECLINE_WORDS))})\b"
GROWTH_WORDS = (
    "grew", "grow", "growing", "growth", "changed", "change",
    "increase", "increased", "decrease", "decreased",
    "decline", "declined", "dropped", "drop", "shrank", "fell",
)
TREND_WORDS = ("over time", "trend", "timeline", "history", "trajectory")
CORRELATION_WORDS = (
    "correlation", "correlated", "correlate", "correlates",
    "relationship", "relate", "relates", "related",
)
GRAIN_WORDS = {
    "daily": "D",
    "day": "D",
    "weekly": "W",
    "week": "W",
    "monthly": "M",
    "month": "M",
    "quarterly": "Q",
    "quarter": "Q",
    "yearly": "Y",
    "annual": "Y",
    "year": "Y",
    # Not "hour" or "second": "by hour of day" asks for a profile, "second highest" a rank.
    "hourly": "h",
    "minute": "min",
}
MONTH_NAMES = {
    name: number
    for number, names in enumerate(
        (
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ),
        start=1,
    )
    for name in names
}

# A text column with at most this many distinct values offers them as filters
# ("for West"); beyond that it holds free text or one label per row.
MAX_FILTER_CANDIDATES = 10_000
# Longest value, in words, the parser looks for in a question.
MAX_VALUE_WORDS = 12
BREAKDOWN_LIMIT = 12
# A breakdown with more than twice this many groups shows this many highest and
# this many lowest, and says how many groups there are.
EXTREME_GROUPS = 10
# A numeric column with at most this many distinct values can group rows
# ("mean signal by timepoint"); more than that and it reads as a measure.
GROUPING_LIMIT = 50
# Most points a correlation answer carries for its scatter.
SCATTER_LIMIT = 2_000
# Words that put the column named right after them in the grouping role:
# "mean signal by timepoint". After a ranking word the column is what gets
# ranked, and a later "by" names the measure: "top 3 cylinders by mpg".
GROUPING_WORDS = r"(?:by|per|across|each)"
RANKING_WORDS = r"(?:which|top|bottom)(?:\s+\d{1,3})?"
GRAIN_NAMES = {
    "s": "second", "min": "minute", "h": "hour",
    "D": "day", "W": "week", "M": "month", "Q": "quarter", "Y": "year",
}
# Trailing unit tokens left out of a column's core name, so "body mass" names
# body_mass_g. "min" and "max" are missing because they also mean minimum and maximum.
UNIT_TOKENS = frozenset({
    "mm", "cm", "m", "km", "nm", "um", "ft", "g", "kg", "mg", "ug", "ng", "lb", "lbs", "oz",
    "l", "ml", "ul", "c", "f", "k", "s", "ms", "sec", "secs", "h", "hr", "hrs",
    "pct", "percent", "ppm", "ppb", "mol", "mmol", "umol", "kda", "rfu", "au",
    "hz", "kpa", "psi", "mph", "bpm", "rpm", "kcal", "usd", "eur", "gbp",
})
_TRAILING_ANNOTATION = re.compile(r"\s*[(\[][^)\]]*[)\]]\s*$")

QUERY_STOPWORDS = {
    "a", "about", "across", "all", "also", "an", "and", "any", "are", "at", "be", "been",
    "between", "by", "can", "data", "did", "do", "does", "done", "each",
    "every", "for", "from", "get", "give", "got", "have", "had", "has", "having", "he", "how",
    "hows", "i", "if", "in", "is", "it", "just", "last", "least", "make", "made", "many", "me",
    "measured", "most", "much", "my", "next", "of", "on", "only", "or", "our", "overall",
    "please", "re", "recorded", "s", "she", "show", "tell", "tested", "than", "that", "the",
    "then", "there", "this", "to", "they", "us", "value", "values", "ve", "was", "we", "were",
    "what", "whats", "when", "where", "which", "whose", "with", "without", "you", "your",
}
# Words asking what share of the rows meet a condition: "what share have diabetes".
SHARE_WORDS = ("share", "proportion", "percentage", "percent", "fraction")
COUNT_PATTERN = rf"\b(?:how many|count|number of|{'|'.join(SHARE_WORDS)})\b"
# Words for the things a table's rows describe. In "how many patients are
# positive" they name what is counted, not a column; when a column of IDs
# numbers them, its distinct values are counted instead of rows.
ENTITY_NOUNS = frozenset({
    "participant", "participants", "patient", "patients", "people", "person", "persons",
    "subject", "subjects", "sample", "samples", "specimen", "specimens", "animal", "animals",
    "bird", "birds", "individual", "individuals", "respondent", "respondents", "case", "cases",
    "record", "records", "observation", "observations", "reading", "readings",
    "measurement", "measurements", "entry", "entries", "row",
})
# Of those, the nouns that always mean rows, whatever columns the table has.
ROW_NOUNS = frozenset({
    "record", "records", "reading", "readings", "observation", "observations", "measurement",
    "measurements", "entry", "entries", "row", "case", "cases", "sample", "samples",
})
# Words in the name of a column that numbers the entities a table describes.
ENTITY_COLUMN_WORDS = frozenset({
    "id", "ids", "subject", "participant", "patient", "individual", "animal", "specimen",
    "donor", "respondent", "person",
})
# Words that ask about missing values rather than name a column.
MISSING_WORDS = ("missing", "null", "blank", "empty")
# Words for the columns themselves. They are grammar only where the plan uses
# them: after a name ("the pH column"), in "which columns have missing values",
# or when the columns are what is ranked ("which column has the highest mean").
COLUMN_WORDS = frozenset({"column", "columns", "variable", "variables", "field", "fields"})
# Every word the parser reads as query grammar rather than as data.
QUERY_GRAMMAR = frozenset({
    *QUERY_STOPWORDS, *AGGREGATION_WORDS, *SUPERLATIVE_WORDS, *DECLINE_WORDS, *GROWTH_WORDS,
    *CORRELATION_WORDS, *GRAIN_WORDS, *MONTH_NAMES, *SHARE_WORDS,
    "bottom", "count", "breakdown", "fastest", "history", "number", "per", "rows", "sells",
    "split", "sold", "top", "trajectory", "trend", "over", "time", "timeline",
})

# Conditions on a column, read before anything else claims their words. Numbers
# keep a decimal point (see _norm) and a minus sign becomes "minus".
NUMBER = r"(?:minus\s+)?\d+(?:\.\d+)?"
COMPARISONS: tuple[tuple[Operator, str], ...] = (
    (">=", r"at least|no less than"),
    ("<=", r"at most|no more than"),
    (">", r"(?:greater|more|higher|larger|bigger) than|above|over|exceeds|exceeding|exceed"),
    ("<", r"(?:less|lower|smaller|fewer) than|below|under"),
)
COMPARISON_PATTERN = "|".join(f"(?P<op{index}>{words})" for index, (_, words) in enumerate(COMPARISONS))
LINKING_VERB = r"(?:(?:is|are|was|were)\s+)?"
OPERATOR_WORDS = {"=": "=", "<": "below", "<=": "at most", ">": "above", ">=": "at least"}
# Comparison symbols become the words the condition grammar reads, since _norm drops symbols.
SYMBOL_WORDS = (
    (">=", " at least "), ("≥", " at least "), ("<=", " at most "), ("≤", " at most "),
    ("==", " is "), (">", " above "), ("<", " below "), ("=", " is "),
)
# Words before a value that make it a filter even when it is also a query word:
# "average Ca for Top" filters to the contour "Top", "top 2 contours" ranks.
VALUE_CUES = frozenset({"for", "in", "is", "at", "where", "when", "equals", "of", "from", "within", "among"})
# Cue words that are grammar only when a filter or condition follows them.
FILTER_CUES = frozenset({"within", "among", "amongst"})
# Words before a 0/1 flag column that ask for the rows where it is set:
# "how many have hypertension". "without" asks for the rows where it is not.
FLAG_CUES = (
    r"have|has|having|had|with|without|are|were|"
    r"received|receive|receiving|given|got|took|taking|using|used"
)
# Grouping by a part of the date: "average PM2.5 by hour", "revenue by weekday",
# "which month of the year is wettest", which pools each month across the years.
TIME_PART_PATTERN = re.compile(
    r"\b(?:by|per|each|across|which|what)\s+(?:the\s+)?"
    r"(?:(?P<hour>hours?(?:\s+of\s+(?:the\s+)?day)?)|(?P<weekday>weekdays?|days?\s+of\s+(?:the\s+)?week)"
    r"|(?P<month>(?:calendar\s+)?months?\s+of\s+(?:the\s+)?year|calendar\s+months?)"
    r"|(?P<quarter>quarters?\s+of\s+(?:the\s+)?year))\b"
)
# Names of the date parts a grouping can show, for answers and tables.
QUARTER_NAMES = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}


@dataclass(frozen=True)
class ValueFilter:
    column: str
    values: tuple[str, ...]


@dataclass(frozen=True)
class Condition:
    """Rows whose ``column`` compares to ``value``: "pH below 5.5", "OBJECT is 0".

    ``"missing"`` keeps the rows where the column has no value and needs no ``value``.
    """

    column: str
    operator: Operator
    value: float | None = None


@dataclass(frozen=True)
class QueryPlan:
    intent: Intent | ProfileIntent
    aggregation: Aggregation = "sum"
    measure: str | None = None
    dimension: str | None = None
    count_column: str | None = None
    #: The second column of a correlation; ``measure`` is the first.
    partner: str | None = None
    top_n: int | None = None
    ascending: bool = False
    filters: tuple[ValueFilter, ...] = ()
    year: int | None = None
    month: int | None = None
    grain: str | None = None
    source: str = "rules"
    conditions: tuple[Condition, ...] = ()
    #: Groups by this part of the date column ``dimension`` names, such as the hour of day.
    time_part: TimePart | None = None
    #: How the parser read the question, said in the answer: "Read "pH" as pH (CaCl2)."
    notes: tuple[str, ...] = ()
    #: A second grouping, for a cross-table of ``dimension`` by this column.
    second_dimension: str | None = None
    #: Splits a numeric ``dimension`` into this many groups of about equal size: 2 at the
    #: median, 4 at the quartiles.
    bins: int | None = None
    #: The columns a cross-column answer ranks, such as every numeric reading.
    columns: tuple[str, ...] = ()
    #: The fraction a percentile asks for: 0.9 for the 90th percentile.
    quantile: float | None = None


@dataclass(frozen=True)
class WholeTable:
    """Every row of a long table, before the brief narrowed it to one parameter.

    A long table holds one parameter per row (pH, ammonia, temperature), and
    the brief follows one of them, ``focus``. A question that names another
    value of ``parameter``, or groups by it, is answered from these rows.
    """

    frame: pd.DataFrame
    roles: ColumnRoles
    parameter: str
    focus: str


@dataclass(frozen=True)
class QueryAnswer:
    question: str
    plan: QueryPlan
    answer: str
    calculation: str
    table: pd.DataFrame | None = None
    chart: Literal["bar", "line", "scatter"] | None = None


def _norm(text: str) -> str:
    """Lowercase words without accents or punctuation; "orgánica" reads as "organica".

    A decimal point between digits stays, so "5.5" and "PM2.5" are one word each.
    """
    decomposed = unicodedata.normalize("NFKD", str(text))
    lowered = "".join(char for char in decomposed if not unicodedata.combining(char)).lower()
    lowered = re.sub(r"\b([a-z]+)['’](s|re|ve|ll|d|m|t)\b", r"\1", lowered)
    cleaned = re.sub(r"(?:(?!(?<=\d)\.(?=\d))[^0-9a-z])+", " ", lowered)
    return " ".join(cleaned.split())


def _question_text(question: str) -> str:
    """The question with comparison symbols as words and a minus sign kept as "minus"."""
    text = str(question)
    for symbol, words in SYMBOL_WORDS:
        text = text.replace(symbol, words)
    return re.sub(r"(?<![\w.])-(?=\d)", " minus ", text)


def _word_regex(text: str) -> str:
    """Match ``text`` as whole words; a decimal point does not end a word, so "5" is not in "5.5"."""
    return rf"(?<![0-9a-z.]){re.escape(text)}(?![0-9a-z]|\.[0-9])"


def _blank(text: str, start: int, end: int) -> str:
    """Replace a span with spaces, keeping every other position where it was."""
    return f"{text[:start]}{' ' * (end - start)}{text[end:]}"


def _word_variants(normalized_name: str) -> set[str]:
    """The name with simple English plurals added or removed ("category", "categories")."""
    variants = {normalized_name, f"{normalized_name}s", f"{normalized_name}es"}
    if normalized_name.endswith("s"):
        variants.add(normalized_name[:-1])
    if re.search(r"[^aeiou]y$", normalized_name):
        variants.add(f"{normalized_name[:-1]}ies")
    if normalized_name.endswith("ies"):
        variants.add(f"{normalized_name[:-3]}y")
    return variants




def _core_name(column: str) -> str:
    """The normalized name without its trailing unit: "Concentration (mg/L)" gives "concentration".

    A bracketed annotation goes first, then unit tokens from the end while a word
    remains, so body_mass_g gives "body mass" and a column called "mm" stays "mm".
    """
    words = _norm(_TRAILING_ANNOTATION.sub("", str(column))).split()
    while len(words) > 1 and words[-1] in UNIT_TOKENS:
        words.pop()
    return " ".join(words)


def _column_names(columns: list[str]) -> dict[str, tuple[str, ...]]:
    """Each column's normalized name, plus its core name when that differs.

    A core name shared with another column, or found inside another column's
    name, is left out: with "Temp (°C)" and "Temp (°F)", "temp" names neither.
    """
    full = {column: _norm(column) for column in columns}
    cores = {column: _core_name(column) for column in columns}
    names: dict[str, tuple[str, ...]] = {}
    for column in columns:
        core = cores[column]
        if not core or core == full[column]:
            names[column] = (full[column],)
            continue
        # Normalized names are single-spaced words, so padding makes this a whole-word match.
        shared = any(
            other != column and (cores[other] == core or f" {core} " in f" {full[other]} ")
            for other in columns
        )
        names[column] = (full[column],) if shared else (full[column], core)
    return names




def _follows(question: str, start: int, words: str = f"(?:{GROUPING_WORDS}|{RANKING_WORDS})") -> bool:
    """Whether the text at ``start`` comes right after one of ``words``, as in "by timepoint"."""
    return bool(re.search(rf"\b{words}\s+(?:the\s+)?$", question[:start]))


def _mention_spans(question: str, names: dict[str, tuple[str, ...]]) -> dict[str, tuple[int, int]]:
    """Map each column named in the question to the span that names it, in question order.

    Longer names claim their words first, so "revenue growth" is not also read
    as "revenue". A core name that is also a query word ("time" for "Time (h)")
    names its column only right after a grouping word: "by time", not "over time".
    """
    remaining = question
    found: dict[str, tuple[int, int]] = {}
    forms = sorted(
        ((form, column) for column, column_forms in names.items() for form in column_forms if form),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for form, column in forms:
        if column in found:
            continue
        grouping_only = form in QUERY_GRAMMAR and form != names[column][0]
        for variant in sorted(_word_variants(form), key=len, reverse=True):
            match = next(
                (
                    candidate
                    for candidate in re.finditer(_word_regex(variant), remaining)
                    if not grouping_only or _follows(question, candidate.start())
                ),
                None,
            )
            if match:
                found[column] = (match.start(), match.end())
                remaining = _blank(remaining, match.start(), match.end())
                break
    return dict(sorted(found.items(), key=lambda item: item[1][0]))


def _mentions(question: str, names: dict[str, tuple[str, ...]]) -> dict[str, int]:
    """Map each column named in the question to where it is named, in question order."""
    return {column: start for column, (start, _) in _mention_spans(question, names).items()}


def _longest(columns: list[str]) -> str | None:
    """The column with the longest name; the first wins a tie."""
    return max(columns, key=lambda column: len(_norm(column))) if columns else None


def _without_columns(
    question: str, columns: tuple[str | None, ...], names: dict[str, tuple[str, ...]]
) -> str:
    """Return the question with the matched column names blanked out.

    Column names contain words the intent scan would otherwise read: "Cell
    Count" would make "total cell count" a row count, and "Total Sales" would
    outrank an explicit "average".
    """
    blanked = question
    for column in columns:
        if not column:
            continue
        for name in names.get(column, (_norm(column),)):
            if name:
                blanked = re.sub(rf"\b{re.escape(name)}\b", " ", blanked)
    return " ".join(blanked.split())


def _unit(measure: str | None, roles: ColumnRoles) -> str:
    """The measure's unit from the metric contract: amount, rate, level or count."""
    return metric_for(measure, roles).unit


def _scoped_roles(roles: ColumnRoles, measure: str | None, dimension: str | None) -> ColumnRoles:
    """The roles with a plan's own measure and grouping.

    An aggregation override belongs to the measure it was set for, so it is
    kept only when the plan asks about that measure.
    """
    aggregation = roles.aggregation if measure == roles.measure else None
    return replace(roles, measure=measure, dimension=dimension, aggregation=aggregation)


def _labels_entities(column: str, series: pd.Series) -> bool:
    """Whether a numeric column numbers entities, like a subject column holding 1 to 20.

    It needs whole numbers and a singular name: "Units" or "Visits" holds a
    quantity, so counting its distinct values would answer a different question.
    """
    values = series.dropna()
    if values.empty or bool((values % 1 != 0).any()):
        return False
    last = (_core_name(column).split() or [""])[-1]
    return not last.endswith("s") or last.endswith(("ss", "us", "is"))


def _detect_time_filter(question: str) -> tuple[int | None, int | None]:
    year_match = re.search(r"\b(19|20)\d{2}\b", question)
    year = int(year_match.group()) if year_match else None
    month = next(
        (number for name, number in MONTH_NAMES.items() if re.search(rf"\b{name}\b", question)),
        None,
    )
    return year, month


def _detect_grain(question: str) -> str | None:
    for word, grain in GRAIN_WORDS.items():
        if re.search(rf"\b{word}(ly)?\b", question):
            return grain
    return None


# Phrasings a plan cannot represent, such as "revenue != 200" or "last 30 days".
# They are refused rather than approximated, leaving them to the optional
# planner or the suggestion chips. Operators are matched on the raw text
# because _norm strips them. A threshold ("pH below 5.5") is a condition the
# parser reads, and refuses when it names no column.
UNSUPPORTED_OPERATORS = re.compile(r"!=|<>|≠")
UNSUPPORTED_PHRASES = re.compile(
    r"\b(?:"
    r"this (?:year|month|quarter|week)|last (?:year|month|quarter|week|\d+ (?:days|weeks|months|years))|"
    r"next (?:year|month|quarter|week)|previous (?:year|month|quarter|week)|"
    # A plan holds a calendar month, not one relative to today, so "last
    # december" is refused rather than read as every December in the file.
    rf"(?:last|next|previous|this) (?:{'|'.join(MONTH_NAMES)})|"
    r"year to date|ytd|today|yesterday|past \d+|"
    r"or"
    r")\b"
)


def _asks_for_several_periods(q: str) -> bool:
    """Whether the question names two or more months or years, which a plan cannot hold."""
    months = {MONTH_NAMES[name] for name in MONTH_NAMES if re.search(rf"\b{name}\b", q)}
    years = set(re.findall(r"\b(?:19|20)\d{2}\b", q))
    return len(months) >= 2 or len(years) >= 2


def unsupported_phrasing(question: str) -> bool:
    q = _norm(question)
    return bool(
        UNSUPPORTED_OPERATORS.search(str(question))
        or UNSUPPORTED_PHRASES.search(q)
        or _asks_for_several_periods(q)
    )


# Reading a question: each part is claimed by a column, a value, a condition or
# the query grammar. What nothing claims is reported, never skipped.


@dataclass(frozen=True)
class _Value:
    """One distinct value of a text column, with its normalized words."""

    column: str
    raw: str
    words: tuple[str, ...]


@dataclass(frozen=True)
class _Vocabulary:
    """The values of a table's text columns, indexed for finding them in a question."""

    values: list[_Value]
    by_text: dict[str, list[int]]
    by_word: dict[str, list[int]]


@dataclass
class _Reading:
    """What each part of a question was read as.

    ``text`` is the normalized question with values, conditions and time parts
    blanked, so intent words are read from the rest; ``remaining`` also blanks
    the column names, and whatever is left in it must be query grammar.
    """

    question: str
    #: The normalized question as asked.
    original: str
    text: str
    remaining: str
    names: dict[str, tuple[str, ...]]
    spans: dict[str, tuple[int, int]] = field(default_factory=dict)
    #: Columns a filter or condition uses, so they are neither measure nor grouping.
    consumed: set[str] = field(default_factory=set)
    filters: dict[str, tuple[str, ...]] = field(default_factory=dict)
    conditions: list[Condition] = field(default_factory=list)
    time_part: TimePart | None = None
    year: int | None = None
    month: int | None = None
    top_n: str | None = None
    entity: str | None = None
    entity_column: str | None = None
    asks_missing: bool = False
    #: A spread or percentile the question asks for ("standard deviation", "90th percentile").
    spread: Aggregation | None = None
    quantile: float | None = None
    #: "most" or "least" for "most common", which asks for the mode or the largest group.
    common: str | None = None
    #: "distribution of X", "describe X": the measure's summary statistics.
    describe: bool = False
    #: "outliers", "unusual values", "impossible values".
    outliers: bool = False
    #: "duplicate rows".
    duplicates: bool = False
    #: "distinct", "unique" or "different", which ask for a count of distinct values.
    distinct: bool = False
    #: The columns a "which column has the highest average" question ranks.
    across: tuple[str, ...] = ()
    notes: list[str] = field(default_factory=list)
    #: Why the question can't become one plan, from reading it.
    problem: str | None = None
    #: Why the planner gave up, when it says.
    refusal: str | None = None

    @property
    def mentioned(self) -> dict[str, int]:
        """Columns the plan may use as its measure or grouping, by position."""
        return {
            column: start
            for column, (start, _) in sorted(self.spans.items(), key=lambda item: item[1][0])
            if column not in self.consumed
        }

    @property
    def unnamed(self) -> str:
        """``text`` with every column name blanked.

        Intent, grain and month words are read from this, so a word inside a
        column's name ("weekly" in average_weekly_activity, "day" in "Age of
        testing (day)") is never read as part of the question's grammar.
        """
        text = self.text
        for start, end in self.spans.values():
            text = _blank(text, start, end)
        return text

    @property
    def value_filters(self) -> tuple[ValueFilter, ...]:
        return tuple(ValueFilter(column=column, values=values) for column, values in self.filters.items())

    def claim(self, start: int, end: int, *, keep_text: bool = False) -> None:
        self.remaining = _blank(self.remaining, start, end)
        if not keep_text:
            self.text = _blank(self.text, start, end)

    def flag(self, problem: str) -> None:
        self.problem = self.problem or problem


def _tokens(text: str) -> list[tuple[int, int, str]]:
    return [(match.start(), match.end(), match.group()) for match in re.finditer(r"\S+", text)]


def _runs(
    reading: _Reading, *, content_only: bool, grammar: frozenset[str] = QUERY_GRAMMAR
) -> list[list[tuple[int, int, str]]]:
    """Unclaimed words in unbroken runs; a claimed word between two words breaks the run.

    With ``content_only`` the runs hold only words that are not in ``grammar``.
    """
    runs: list[list[tuple[int, int, str]]] = [[]]
    for token in _tokens(reading.remaining):
        previous = runs[-1][-1] if runs[-1] else None
        if content_only and token[2] in grammar:
            runs.append([])
        elif previous is None or reading.original[previous[1]:token[0]].strip() == "":
            runs[-1].append(token)
        else:
            runs.append([token])
    return [run for run in runs if run]


def _raw_chunks(question: str, words: list[str] | tuple[str, ...]) -> list[str]:
    """The question's own spellings of some normalized words, one per word typed: ["pH"] for "ph"."""
    wanted = set(words)
    return [
        chunk.strip("?.,!;:'\"")
        for chunk in _question_text(question).split()
        if set(_norm(chunk).split()) & wanted
    ]


def _raw_phrase(question: str, words: list[str]) -> str:
    """The question's own spelling of consecutive normalized words, found once: "B columns"."""
    chunks = [chunk.strip("?.,!;:'\"") for chunk in _question_text(question).split()]
    normalized = [_norm(chunk).split() for chunk in chunks]
    for start in range(len(chunks)):
        taken: list[str] = []
        for end in range(start, len(chunks)):
            taken += normalized[end]
            if taken == list(words):
                return " ".join(chunks[start:end + 1])
            if len(taken) >= len(words):
                break
    return _raw_words(question, words)


def _raw_words(question: str, words: list[str] | tuple[str, ...]) -> str:
    """The question's own spelling of some normalized words: "pH" for "ph"."""
    return " ".join(_raw_chunks(question, words)) or " ".join(words)


def _vocabulary(dataframe: pd.DataFrame, roles: ColumnRoles) -> _Vocabulary:
    """Index the values of the text columns small enough to filter on."""
    values: list[_Value] = []
    by_text: dict[str, list[int]] = defaultdict(list)
    by_word: dict[str, list[int]] = defaultdict(list)
    for column in dataframe.columns:
        series = dataframe[column]
        if column == roles.date or is_numeric_dtype(series) or is_datetime64_any_dtype(series):
            continue
        uniques = series.dropna().unique()
        if len(uniques) > MAX_FILTER_CANDIDATES:
            continue
        for raw in uniques:
            words = tuple(_norm(raw).split())
            if not words:
                continue
            position = len(values)
            values.append(_Value(column=column, raw=str(raw), words=words))
            by_text[" ".join(words)].append(position)
            for word in set(words):
                by_word[word].append(position)
    return _Vocabulary(values=values, by_text=by_text, by_word=by_word)


def _holds(words: tuple[str, ...], run: tuple[str, ...]) -> bool:
    """Whether ``run`` appears in ``words`` as consecutive whole words."""
    size = len(run)
    return any(words[index:index + size] == run for index in range(len(words) - size + 1))


def _values_named(
    run: tuple[str, ...], vocabulary: _Vocabulary, *, exact_only: bool, column: str | None = None
) -> tuple[list[_Value], bool]:
    """Values the words name, and whether they name them exactly.

    An exact match wins; otherwise a value holding the words as whole words, so
    "Tartrazine" names "Tartrazine, FD&C Yellow 5 (US), E102 (EU)".
    """
    texts = {" ".join(run)}
    if len(run) == 1:
        texts |= _word_variants(run[0])
    exact = [vocabulary.values[index] for text in texts for index in vocabulary.by_text.get(text, ())]
    if column is not None:
        exact = [value for value in exact if value.column == column]
    if exact or exact_only:
        return exact, True
    postings = [vocabulary.by_word.get(word) for word in run]
    if not all(postings):
        return [], False
    shared = set(postings[0]).intersection(*postings[1:])
    found = [
        vocabulary.values[index] for index in sorted(shared) if _holds(vocabulary.values[index].words, run)
    ]
    if column is not None:
        found = [value for value in found if value.column == column]
    return found, False


def _same_rows(dataframe: pd.DataFrame, found: list[_Value], columns: list[str]) -> bool:
    """Whether the values pick the same rows in every column, as a label and its definition do."""
    masks = [
        dataframe[column].astype(str).isin([value.raw for value in found if value.column == column])
        for column in columns
    ]
    return all(mask.equals(masks[0]) for mask in masks[1:])


def _pick_values(
    run: tuple[str, ...],
    found: list[_Value],
    exact: bool,
    question: str,
    dataframe: pd.DataFrame | None = None,
) -> tuple[_Value | None, list[_Value], str | None]:
    """The column and values ``found`` stands for, or why the words don't pick one.

    A value found in several columns is ambiguous unless it picks the same
    rows in each, in which case the first column stands for all of them.
    """
    if not found:
        return None, [], None
    phrase = _raw_words(question, run)
    columns = list(dict.fromkeys(value.column for value in found))
    if len(columns) > 1 and exact and dataframe is not None and _same_rows(dataframe, found, columns):
        found = [value for value in found if value.column == columns[0]]
        columns = columns[:1]
    if len(columns) > 1:
        return None, [], (
            f'"{phrase}" is a value of {_and_listed(columns)}; name the column too, '
            f'as in "{columns[0]} {phrase}".'
        )
    if exact:
        return found[0], found, None
    starting = [value for value in found if value.words[:len(run)] == run]
    chosen = found if len(found) == 1 else starting if len(starting) == 1 else []
    if chosen:
        return chosen[0], chosen, None
    shown = _leaders([value.raw for value in found])
    return None, [], f'"{phrase}" matches several {columns[0]} values: {shown}. Name one in full.'


def _is_number(text: str) -> bool:
    return bool(re.fullmatch(r"[\d.\s]+", text))


def _plain_match(value: _Value, question: str, cued: bool) -> bool:
    """Whether a value may be matched in a question without its column named.

    Numbers are left to conditions ("ID 47"). A value under three letters must
    be written in its own case ("pH", "ON") so "on" stays a word, and a single
    letter needs a cue as well ("for A"), since a sentence may start with "A".
    """
    text = " ".join(value.words)
    if _is_number(text) or len(text) == 1 and not cued:
        return False
    if len(text) < 3:
        return bool(re.search(rf"(?<![0-9A-Za-z]){re.escape(value.raw)}(?![0-9A-Za-z])", question))
    return True


def _cued(reading: _Reading, start: int) -> bool:
    """Whether the word before ``start`` marks a filter: "for Top", "where ... is Open"."""
    before = reading.original[:start].split()
    return bool(before) and before[-1] in VALUE_CUES


def _add_filter(reading: _Reading, values: list[_Value]) -> None:
    column = values[0].column
    known = reading.filters.get(column, ())
    reading.filters[column] = tuple(dict.fromkeys((*known, *(value.raw for value in values))))


def _claim_values(
    reading: _Reading, vocabulary: _Vocabulary, dataframe: pd.DataFrame, *, exact_only: bool
) -> None:
    """Filter on values of text columns found among the unclaimed words, longest first.

    A value made only of query words or nouns for rows ("Top", "Total",
    "Patients") needs a cue before it, so "top 2 contours" still ranks while
    "average Ca for Top" filters.
    """
    for run in _runs(reading, content_only=False):
        index = 0
        while index < len(run):
            for size in range(min(MAX_VALUE_WORDS, len(run) - index), 0, -1):
                window = run[index:index + size]
                words = tuple(word for _, _, word in window)
                ordinary = [word for word in words if word not in QUERY_GRAMMAR and word not in ENTITY_NOUNS]
                if not ordinary and not _cued(reading, window[0][0]):
                    continue
                if not exact_only and sum(len(word) for word in ordinary) < 3:
                    continue
                found, exact = _values_named(words, vocabulary, exact_only=exact_only)
                cued = _cued(reading, window[0][0])
                found = [value for value in found if _plain_match(value, reading.question, cued)]
                if not found:
                    continue
                chosen, values, problem = _pick_values(words, found, exact, reading.question, dataframe)
                if problem:
                    reading.flag(problem)
                elif chosen is not None:
                    _add_filter(reading, values)
                reading.claim(window[0][0], window[-1][1])
                index += size
                break
            else:
                index += 1


def _partial_words(columns: list[str]) -> dict[str, tuple[str, ...]]:
    """Each column's words without a trailing annotation, camelCase split: "sample sampling point label"."""
    return {
        column: tuple(_norm(normalized_name(_TRAILING_ANNOTATION.sub("", str(column)))).split())
        for column in columns
    }


def _holds_variants(words: tuple[str, ...], run: tuple[str, ...]) -> bool:
    """``_holds`` with each word of ``run`` allowed in singular or plural."""
    size = len(run)
    return any(
        all(words[index + offset] in _word_variants(word) for offset, word in enumerate(run))
        for index in range(len(words) - size + 1)
    )


def _role_of_position(text: str, start: int) -> str:
    """What a word at ``start`` is likely to name: something counted, a grouping, or a measure."""
    counted = rf"{COUNT_PATTERN}\s+(?:of\s+)?(?:the\s+)?(?:distinct\s+|different\s+|unique\s+)?$"
    if re.search(counted, text[:start]):
        return "count"
    return "group" if _follows(text, start) else "measure"


def _suits(column: str, role: str, dataframe: pd.DataFrame, roles: ColumnRoles) -> bool:
    series = dataframe[column]
    numeric = is_numeric_dtype(series) and not is_bool_dtype(series)
    if role == "count":
        return column == roles.identifier
    if role == "group":
        return column in roles.dimensions or not numeric or _can_group_numeric(series)
    return numeric


def _claim_partial_names(reading: _Reading, dataframe: pd.DataFrame, roles: ColumnRoles) -> list[str]:
    """Read a word or two of a column's name as that column when only one column fits.

    "strength" names "Concrete compressive strength (MPa)", and "conductivity"
    names "Conduc", whose whole name starts the word. When several columns
    fit, the one that suits the word's place (counted, grouped or measured)
    is taken if it is the only one; otherwise the question is refused with the
    candidates. The answer says how the words were read. Returns the columns found.
    """
    words = _partial_words([column for column in dataframe.columns if column not in reading.spans])
    found: list[str] = []
    for run in _runs(reading, content_only=True):
        index = 0
        while index < len(run):
            index += _claim_partial_at(reading, run, index, words, dataframe, roles, found)
    return found


def _claim_partial_at(
    reading: _Reading,
    run: list[tuple[int, int, str]],
    index: int,
    words: dict[str, tuple[str, ...]],
    dataframe: pd.DataFrame,
    roles: ColumnRoles,
    found: list[str],
) -> int:
    """Try the longest phrase starting at ``run[index]``; return how many words were used."""
    for size in range(min(6, len(run) - index), 0, -1):
        window = run[index:index + size]
        phrase = tuple(word for _, _, word in window)
        if not any(char.isalpha() for word in phrase for char in word):
            continue
        candidates = [column for column, names in words.items() if _holds_variants(names, phrase)]
        if not candidates and size == 1:
            candidates = _prefixed_columns(phrase[0], words)
        if not candidates:
            continue
        start, end = window[0][0], window[-1][1]
        if _names_a_set(reading, candidates, start, dataframe):
            # "which gene has the highest average" ranks the Gene_ columns themselves.
            reading.across = tuple(candidates)
            reading.claim(start, end)
            return size
        extra = ""
        if len(candidates) > 1:
            role = _role_of_position(reading.text, start)
            suited = [column for column in candidates if _suits(column, role, dataframe, roles)]
            if role == "count" and len(suited) != 1:
                named, other = _counted_column(phrase, candidates, dataframe)
                suited = [named] if named else suited
                extra = other if named else ""
            if role == "group" and len(suited) > 1:
                suited = _interchangeable(suited, dataframe) or suited
            if len(suited) != 1:
                if size == 1 and (phrase[0] in ENTITY_NOUNS or role == "count"):
                    # Left for _claim_entity: "how many samples" counts rows.
                    return 1
                reading.flag(f"Did you mean {_listed_capped(candidates)}?")
                reading.claim(start, end, keep_text=True)
                return size
            candidates = suited
        column = candidates[0]
        del words[column]
        reading.spans[column] = (start, end)
        reading.claim(start, end, keep_text=True)
        reading.notes.append(f'Read "{_raw_words(reading.question, phrase)}" as {column}.')
        if extra:
            reading.notes.append(extra)
        found.append(column)
        return size
    return 1


# Name words that identify the thing a column is named for: "Plot Code" and
# "DRUG_NAME" name plots and drugs; "Plot Type" names a kind of plot.
IDENTIFYING_WORDS = frozenset({
    "id", "ids", "code", "codes", "name", "names", "label", "labels", "number", "no", "num", "key",
})


def _counted_column(
    phrase: tuple[str, ...], candidates: list[str], dataframe: pd.DataFrame
) -> tuple[str | None, str]:
    """The column "how many <noun>" counts, and a note naming any other column that counts differently.

    The column must be named for the noun: "drugs" names DRUG_NAME and DRUG_ID,
    "plots" names "Plot Code" and "Plot Name", "crops" names "Current Crop" but
    not "Crop Rotation System". A name wins over a serial number, so DRUG_NAME
    counts drugs, and a code over a name when they disagree, since names drift
    in spelling.
    Nouns for rows ("samples", "records") count rows, not a column.
    """
    if len(phrase) == 1 and phrase[0] in ROW_NOUNS:
        return None, ""
    words = _partial_words(candidates)

    def named_for(column: str) -> bool:
        core = [word for word in words[column] if word not in IDENTIFYING_WORDS]
        tail = core[-len(phrase):]
        return len(tail) == len(phrase) and all(
            word in _word_variants(asked) for word, asked in zip(tail, phrase, strict=True)
        )

    named = [column for column in candidates if named_for(column)]
    named = [column for column in named if not _serial(dataframe[column])] or named
    if not named:
        return None, ""
    counts = {column: int(dataframe[column].nunique(dropna=True)) for column in named}
    key_words = IDENTIFYING_WORDS - {"name", "names", "label", "labels"}
    codes = [column for column in named if set(words[column]) & key_words]
    if len(set(counts.values())) == 1:
        return named[0], ""
    if len(codes) != 1:
        return None, ""
    chosen = codes[0]
    others = [f"{column} has {counts[column]:,}" for column in named if column != chosen]
    return chosen, f"{_and_listed(others)} distinct values." if others else ""


def _serial(series: pd.Series) -> bool:
    """Whether a column holds serial numbers: numbers, or text made only of digits."""
    if _is_measurement(series):
        return True
    sample = series.dropna().head(1_000).astype(str)
    return not sample.empty and bool(sample.str.fullmatch(r"\d+").all())


def _interchangeable(columns: list[str], dataframe: pd.DataFrame) -> list[str]:
    """One column standing for several that pick the same groups of rows, as a code and its label do.

    The text column with a name word such as "label" or "name" is preferred.
    Returns [] when the columns group the rows differently.
    """
    present = dataframe[columns].dropna()
    if present.empty:
        return []
    pairs = len(present.drop_duplicates())
    if any(int(present[column].nunique()) != pairs for column in columns):
        return []
    words = _partial_words(columns)
    texts = [column for column in columns if not _is_measurement(dataframe[column])] or columns
    labelled = [column for column in texts if set(words[column]) & {"label", "name", "names"}]
    return [(labelled or texts)[0]]


# Fewest columns a word must name for "which <word>" to rank the columns themselves.
MIN_COLUMN_SET = 3


def _names_a_set(reading: _Reading, candidates: list[str], start: int, dataframe: pd.DataFrame) -> bool:
    """Whether "which <word>" names numeric columns to rank, as "which gene" names Gene_1 to Gene_1000."""
    if len(candidates) < MIN_COLUMN_SET or reading.across:
        return False
    if not re.search(r"\b(?:which|what)\s+(?:the\s+)?$", reading.original[:start]):
        return False
    return all(
        is_numeric_dtype(dataframe[column]) and not is_bool_dtype(dataframe[column]) for column in candidates
    )


def _prefixed_columns(word: str, words: dict[str, tuple[str, ...]]) -> list[str]:
    """Columns with a name word that starts ``word``: "Conduc" for "conductivity".

    The name word needs three letters and a third of the question word, so a
    short abbreviation can't swallow an unrelated word.
    """
    if not word.isalpha() or len(word) < 4:
        return []
    return [
        column
        for column, names in words.items()
        if any(
            len(name) >= 3 and 3 * len(name) >= len(word) and word.startswith(name) and name != word
            and name not in QUERY_GRAMMAR
            for name in names
        )
    ]


def _number(text: str) -> float:
    words = text.split()
    return -float(words[-1]) if words[0] == "minus" else float(words[-1])


def _is_flag(series: pd.Series) -> bool:
    """Whether a column holds only 0 and 1 (or True and False): a yes/no flag."""
    if is_bool_dtype(series):
        return True
    if not is_numeric_dtype(series):
        return False
    values = series.dropna().unique()
    return 0 < len(values) <= 2 and set(values.tolist()) <= {0, 1}


def _code_like(column: str, series: pd.Series, roles: ColumnRoles) -> bool:
    """Whether a numeric column holds codes, so "Status 2" means Status equal to 2."""
    if column in roles.dimensions:
        return True
    values = series.dropna()
    return bool((values % 1 == 0).all()) and int(values.nunique()) <= GROUPING_LIMIT


def _condition_after(
    column: str, after: str, series: pd.Series, roles: ColumnRoles
) -> tuple[list[Condition], int] | None:
    """A condition written after a numeric column: "pH below 5.5", "OBJECT is 0", "Status 2".

    "values" or "readings" may come between: "how many AQC1cm values are 0".
    """
    noun = re.match(
        r"\s+(?:values?|readings?|measurements?)(?=\s+(?:is|are|was|were|equals?|at|no|greater|more|higher|"
        r"larger|bigger|less|lower|smaller|fewer|above|over|exceeds?|exceeding|below|under|between|negative|"
        r"positive|zero)\b)",
        after,
    )
    found = _condition_words(column, after[noun.end():] if noun else after, series, roles)
    if found is None:
        return None
    conditions, length = found
    return conditions, length + (noun.end() if noun else 0)


def _condition_words(
    column: str, after: str, series: pd.Series, roles: ColumnRoles
) -> tuple[list[Condition], int] | None:
    number = rf"(?P<number>{NUMBER})(?![0-9a-z])"
    if (match := re.match(rf"\s+{LINKING_VERB}(?:{COMPARISON_PATTERN})\s+{number}", after)):
        operator = next(
            operator for index, (operator, _) in enumerate(COMPARISONS) if match.group(f"op{index}")
        )
        return [Condition(column, operator, _number(match.group("number")))], match.end()
    if (match := re.match(
        rf"\s+{LINKING_VERB}between\s+(?P<low>{NUMBER})\s+and\s+(?P<high>{NUMBER})(?![0-9a-z])", after
    )):
        low, high = sorted((_number(match.group("low")), _number(match.group("high"))))
        return [Condition(column, ">=", low), Condition(column, "<=", high)], match.end()
    if (match := re.match(rf"\s+{LINKING_VERB}(?P<sign>negative|positive|zero)\b", after)):
        return [_sign_condition(column, match.group("sign"))], match.end()
    if (match := re.match(rf"\s+(?:is|are|was|were|equals|equal\s+to)\s+{number}", after)):
        return [Condition(column, "=", _number(match.group("number")))], match.end()
    if (match := re.match(rf"\s+{number}", after)):
        number = _number(match.group("number"))
        if _code_like(column, series, roles) and bool((series == number).any()):
            return [Condition(column, "=", number)], match.end()
    return None


def _code_before(
    column: str, before: str, series: pd.Series, roles: ColumnRoles
) -> tuple[Condition, int] | None:
    """A code written before its column, "cars with 6 cyl", unless it is a "top 5"."""
    match = re.search(rf"(?<![0-9a-z.])(?P<number>{NUMBER})\s+$", before)
    if match is None or re.search(r"\b(?:top|bottom)\s+$", before[:match.start()]):
        return None
    number = _number(match.group("number"))
    if not _code_like(column, series, roles) or not bool((series == number).any()):
        return None
    return Condition(column, "=", number), match.start()


def _sign_condition(column: str, sign: str) -> Condition:
    return Condition(column, {"negative": "<", "positive": ">", "zero": "="}[sign], 0.0)


def _claim_conditions(
    reading: _Reading,
    columns: list[str],
    dataframe: pd.DataFrame,
    roles: ColumnRoles,
    vocabulary: _Vocabulary,
) -> None:
    """Read conditions written around a named column, and claim their words.

    After a numeric column: a comparison, a range, a sign or a value. After a
    text column: one of its values ("Condition is Open", "Detector is ON",
    "Region West"). Before any column: "missing". Before a 0/1 flag: "have",
    "with" or "are" for the rows where it is 1, "without" for 0.
    """
    for column in columns:
        if column in reading.consumed or column not in reading.spans:
            continue
        start, end = reading.spans[column]
        after, before = reading.remaining[end:], reading.remaining[:start]
        series = dataframe[column]
        numeric = is_numeric_dtype(series) and not is_bool_dtype(series)
        claimed: tuple[int, int] | None = None
        if (match := re.match(r"\s+(?:is|are|was|were)\s+(?:missing|blank|empty|null)\b", after)):
            reading.conditions.append(Condition(column, "missing"))
            claimed = (start, end + match.end())
        elif (match := re.search(
            r"\b(?:missing|blank|empty|null)\s+(?:values?\s+(?:for|in|of)\s+(?:the\s+)?)?$", before
        )):
            reading.conditions.append(Condition(column, "missing"))
            claimed = (match.start(), end)
        elif numeric and (found := _condition_after(column, after, series, roles)):
            conditions, length = found
            reading.conditions.extend(conditions)
            claimed = (start, end + length)
        elif numeric and (match := re.search(
            r"\b(?P<sign>negative|positive|zeros?|zero)\s+(?:values?\s+|readings?\s+)?"
            r"(?:(?:in|of|for)\s+(?:the\s+)?)?$", before
        )):
            sign = match.group("sign")
            reading.conditions.append(_sign_condition(column, "zero" if sign.startswith("zero") else sign))
            claimed = (match.start(), end)
        elif numeric and (found := _code_before(column, before, series, roles)):
            reading.conditions.append(found[0])
            claimed = (found[1], end)
        elif _is_flag(series) and (match := re.search(rf"\b(?P<cue>{FLAG_CUES})\s+(?:a\s+|an\s+)?$", before)):
            # "correlated with diabetes" pairs two columns rather than keeping the diabetics.
            paired = r"\b(?:correlat\w*|relat\w*|compared?|vary|varies|differs?)\s+$"
            if re.search(paired, before[:match.start()]):
                continue
            reading.conditions.append(Condition(column, "=", 0.0 if match.group("cue") == "without" else 1.0))
            claimed = (match.start(), end)
        elif not numeric and (found_value := _attached_value(reading, column, end, vocabulary)):
            values, value_end = found_value
            _add_filter(reading, values)
            claimed = (start, value_end)
        if claimed:
            reading.consumed.add(column)
            reading.claim(*claimed)


def _attached_value(
    reading: _Reading, column: str, end: int, vocabulary: _Vocabulary
) -> tuple[list[_Value], int] | None:
    """A value of ``column`` written right after its name, with or without "is"."""
    lead = re.match(r"\s+(?:(?:is|are|was|were|equals|equal\s+to)\s+)?", reading.remaining[end:])
    if lead is None:
        return None
    tokens = [token for token in _tokens(reading.remaining[end + lead.end():])][:MAX_VALUE_WORDS]
    offset = end + lead.end()
    if not tokens or tokens[0][0] != 0:
        return None
    for size in range(len(tokens), 0, -1):
        window = tokens[:size]
        if not _unbroken_window(reading, window, offset):
            continue
        words = tuple(word for _, _, word in window)
        found, exact = _values_named(words, vocabulary, exact_only=False, column=column)
        if not found or not exact and all(word in QUERY_GRAMMAR for word in words):
            continue
        chosen, values, problem = _pick_values(words, found, exact, reading.question)
        if chosen is None:
            if problem:
                reading.flag(problem)
            return None
        return values, offset + window[-1][1]
    return None


def _unbroken_window(reading: _Reading, window: list[tuple[int, int, str]], offset: int) -> bool:
    """Whether nothing was claimed between the words of ``window``, which start at ``offset``."""
    return all(
        reading.original[offset + left[1]:offset + right[0]].strip() == ""
        for left, right in zip(window, window[1:], strict=False)
    )


def _unit_words(column: str) -> set[str]:
    """Units a column's name gives: the words in brackets, and trailing unit words."""
    words = set(_norm(" ".join(re.findall(r"[(\[]([^)\]]*)[)\]]", str(column)))).split())
    trailing = _norm(_TRAILING_ANNOTATION.sub("", str(column))).split()
    while len(trailing) > 1 and trailing[-1] in UNIT_TOKENS:
        words.add(trailing.pop())
    return words


def _claim_unit_conditions(reading: _Reading, dataframe: pd.DataFrame, roles: ColumnRoles) -> None:
    """Read "at 28 days" as a column whose unit is days being 28, when one numeric column has that unit."""
    units = {
        column: _unit_words(column)
        for column in roles.numeric
        if column in dataframe.columns
    }
    pattern = rf"\b(?:(?:at|after|on|for|of|with)\s+)?(?P<number>{NUMBER})\s+(?P<unit>[a-z]+)\b"
    for match in list(re.finditer(pattern, reading.remaining)):
        if reading.original[match.end("number"):match.start("unit")].strip():
            continue
        variants = _word_variants(match.group("unit"))
        columns = [column for column, words in units.items() if words & variants]
        if len(columns) != 1:
            continue
        reading.conditions.append(Condition(columns[0], "=", _number(match.group("number"))))
        reading.claim(match.start(), match.end())


def _claim_numbers(reading: _Reading, dataframe: pd.DataFrame, roles: ColumnRoles) -> None:
    """Claim the number of a "top N", a year, and numbers that are whole column names ("1001")."""
    if (top := re.search(r"\b(?:top|bottom)\s+(?P<n>\d{1,3})\b", reading.remaining)):
        reading.top_n = top.group(0)
        reading.claim(top.start("n"), top.end("n"), keep_text=True)
    numbered = {forms[0]: column for column, forms in reading.names.items() if _is_number(forms[0])}
    for start, end, word in _tokens(reading.remaining):
        if not _is_number(word):
            continue
        is_year = bool(re.fullmatch(r"(?:19|20)\d{2}", word))
        if word in numbered and not (is_year and roles.date):
            reading.spans[numbered[word]] = (start, end)
            reading.claim(start, end, keep_text=True)
        elif is_year and reading.year is None:
            reading.year = int(word)
            reading.claim(start, end, keep_text=True)


def _entity_column(noun: str, dataframe: pd.DataFrame, roles: ColumnRoles) -> str | None:
    """The column whose distinct values count ``noun``, when one repeats across rows.

    Nouns for rows ("readings", "records", "samples") always count rows. Other
    nouns ("birds", "patients") count the identifier, or the one column named
    like an entity ID, when its values repeat; when every row has its own
    value, counting rows is the same thing.
    """
    if noun in ROW_NOUNS:
        return None
    candidates = [roles.identifier] if roles.identifier in dataframe.columns else []
    if not candidates:
        candidates = [
            column
            for column in dataframe.columns
            if set(_norm(normalized_name(column)).split()) & ENTITY_COLUMN_WORDS
            and not is_datetime64_any_dtype(dataframe[column])
        ]
    if len(candidates) != 1:
        return None
    values = dataframe[candidates[0]].dropna()
    return candidates[0] if values.nunique() < len(values) else None


def _claim_row_label(reading: _Reading, roles: ColumnRoles) -> None:
    """Read a noun after "which" or "top 10" as an unnamed row label: "which probes" names "Unnamed: 0".

    A first column with no header is read as the row label (the notes say so),
    so the only word a question can name it by is what it labels. Only a noun
    that names nothing else, right where a grouping goes, is read this way.
    """
    label = roles.identifier
    if not label or not re.match(r"unnamed\b", str(label), re.IGNORECASE) or label in reading.spans:
        return
    for start, end, word in _tokens(reading.remaining):
        if word in QUERY_GRAMMAR or word in ENTITY_NOUNS or not word.isalpha() or len(word) < 3:
            continue
        if not _follows(reading.original, start, rf"(?:{RANKING_WORDS}|{GROUPING_WORDS})"):
            continue
        reading.spans[label] = (start, end)
        reading.claim(start, end, keep_text=True)
        reading.notes.append(f'Read "{_raw_words(reading.question, [word])}" as {label}, the row label.')
        return


def _claim_entity(reading: _Reading, dataframe: pd.DataFrame, roles: ColumnRoles) -> None:
    """Read a noun for the rows ("patients", "birds") as what is counted."""
    for start, end, word in _tokens(reading.remaining):
        if word in QUERY_GRAMMAR:
            continue
        counted = re.search(rf"{COUNT_PATTERN}\s+(?:of\s+)?(?:the\s+)?$", reading.original[:start])
        plural = word.endswith("s") and word.isalpha() and len(word) > 3
        if word not in ENTITY_NOUNS and not (counted and plural):
            continue
        reading.claim(start, end)
        if reading.entity is None:
            reading.entity = word
            reading.entity_column = _entity_column(word, dataframe, roles)


def _has_times(dates: pd.Series) -> bool:
    values = dates.dropna()
    return bool((values != values.dt.normalize()).any())


def _claim_time_part(reading: _Reading, dataframe: pd.DataFrame, roles: ColumnRoles) -> None:
    """Read "by hour", "by weekday" or "month of the year" as a grouping by that part of the date column."""
    match = TIME_PART_PATTERN.search(reading.remaining)
    if match is None:
        return
    part: TimePart = next(name for name in ("hour", "weekday", "month", "quarter") if match.group(name))
    reading.claim(*match.span(part))
    if not roles.date or roles.date not in dataframe.columns:
        reading.flag(f"This table has no date column, so there is no {part} to group by.")
    elif part == "hour" and not _has_times(dataframe[roles.date]):
        reading.flag(f"{roles.date} holds dates without times of day, so there is no hour to group by.")
    else:
        reading.time_part = part


def _reread_aggregation_names(reading: _Reading) -> None:
    """Resolve a column name that starts with an aggregation word.

    With columns "Wind Speed" and "Maximum Wind Speed", "maximum wind speed"
    asks for the maximum of Wind Speed unless the question has another
    aggregation word ("average maximum wind speed"); the answer says how it was
    read. A column called just "MAX" or "mean" is the aggregation word unless
    the question reads it as a column some other way (see ``_named_as_column``).
    """
    by_form = {form: column for column, forms in reading.names.items() for form in forms}
    outside = reading.remaining
    elsewhere = any(re.search(rf"\b{word}\b", outside) for word in AGGREGATION_WORDS)
    for column, (start, end) in list(reading.spans.items()):
        words = reading.text[start:end].split()
        if words[0] not in AGGREGATION_WORDS or elsewhere:
            continue
        if len(words) == 1:
            if not _named_as_column(reading, column, start, end):
                del reading.spans[column]
                restored = reading.text[start:end]
                reading.remaining = f"{reading.remaining[:start]}{restored}{reading.remaining[end:]}"
            continue
        other = by_form.get(" ".join(words[1:]))
        if other is None or other in reading.spans:
            continue
        word_end = start + len(words[0])
        del reading.spans[column]
        reading.spans[other] = (word_end + 1, end)
        reading.remaining = f"{reading.remaining[:start]}{words[0]}{reading.remaining[word_end:]}"
        label = AGGREGATION_LABELS[AGGREGATION_WORDS[words[0]]].lower()
        reading.notes.append(
            f'Read "{_raw_words(reading.question, words)}" as the {label} of {other}; '
            f'ask for "average {column}" to use the {column} column.'
        )


DESCRIBE_PATTERN = re.compile(
    r"\b(?:distribution|describe|summary(?:\s+statistics)?|summarise|summarize|"
    r"descriptive\s+statistics|statistics|stats)\b"
)
OUTLIER_PATTERN = re.compile(
    r"\b(?:outliers?|outlying|unusual|impossible|implausible|anomalous|suspicious|extreme)\b"
)
DUPLICATE_PATTERN = re.compile(r"\b(?:duplicates?|duplicated)\b")
DISTINCT_PATTERN = re.compile(
    r"\b(?:distinct|unique)\b|(?:(?<=how many )|(?<=number of )|(?<=count of ))different\b"
)
COMMON_PATTERN = re.compile(r"\b(?P<most>most|least)\s+(?:common|frequent|frequently\s+occurring)\b|\bmode\b")
PERCENTILE_PATTERN = re.compile(
    r"\b(?:(?P<n>\d{1,2}(?:\.\d+)?)(?:st|nd|rd|th)?\s+percentile"
    r"|percentile\s+(?P<m>\d{1,2})|p(?P<p>\d{1,2}))\b"
)
QUARTILE_PATTERN = re.compile(r"\b(?P<which>first|lower|1st|third|upper|3rd)\s+quartile\b")
ACROSS_PATTERN = re.compile(r"\b(?:which|what)\s+(?:the\s+)?(?:column|variable|field)s?\b")


def _claim_analysis_words(reading: _Reading, dataframe: pd.DataFrame, roles: ColumnRoles) -> None:
    """Read the words that ask for a summary rather than name data.

    Spreads and percentiles ("standard deviation", "90th percentile"), "most
    common", "distribution of", "outliers", "duplicate rows", "how many
    distinct", and "which column", which ranks the numeric readings themselves.
    """
    text = reading.remaining
    for phrase, aggregation in SPREAD_PHRASES.items():
        for match in re.finditer(rf"\b{phrase}\b", text):
            reading.spread = aggregation
            reading.claim(*match.span())
    if (match := PERCENTILE_PATTERN.search(reading.remaining)):
        share = float(match.group("n") or match.group("m") or match.group("p")) / 100
        if 0 < share < 1:
            reading.quantile = share
            reading.claim(*match.span())
    if (match := QUARTILE_PATTERN.search(reading.remaining)):
        reading.quantile = 0.25 if match.group("which") in ("first", "lower", "1st") else 0.75
        reading.claim(*match.span())
    if (match := COMMON_PATTERN.search(reading.remaining)):
        reading.common = match.group("most") or "most"
        reading.claim(*match.span())
    for pattern, name in (
        (DESCRIBE_PATTERN, "describe"), (DUPLICATE_PATTERN, "duplicates"), (DISTINCT_PATTERN, "distinct"),
    ):
        for match in pattern.finditer(reading.remaining):
            setattr(reading, name, True)
            reading.claim(*match.span())
    for match in OUTLIER_PATTERN.finditer(reading.remaining):
        reading.outliers = True
        reading.claim(*match.span())
        # "outlier spectra", "unusual patients": the noun names the rows.
        following = re.match(r"\s+([a-z]{3,})\b", reading.remaining[match.end():])
        if following and following.group(1) not in QUERY_GRAMMAR:
            reading.claim(match.end() + following.start(1), match.end() + following.end(1))
            reading.entity = following.group(1)
    relating = CORRELATES_PATTERN.search(text) or set(text.split()) & RESEARCH_WORDS
    if not relating and (match := ACROSS_PATTERN.search(reading.remaining)):
        reading.across = tuple(numeric_readings(dataframe, roles))
        reading.claim(*match.span())


def _filter_follows(reading: _Reading, end: int) -> bool:
    """Whether a filter or condition was read right after ``end``, as in "within CLASS 0"."""
    following = re.match(r"\s+(\S+)", reading.original[end:])
    if following is None:
        return False
    start = end + following.start(1)
    stop = end + following.end(1)
    return not reading.remaining[start:stop].strip() and not reading.text[start:stop].strip()


def _after_name(reading: _Reading, start: int) -> bool:
    """Whether the word at ``start`` comes right after a column's name: "column" in "the pH column"."""
    return any(
        end <= start and reading.original[end:start].strip() == "" for _, end in reading.spans.values()
    )


def _named_as_column(reading: _Reading, column: str, start: int, end: int) -> bool:
    """Whether a column named by one aggregation word ("MAX", "mean", "std") is meant as the column.

    It is when it follows "of", a grouping or a ranking word ("average of max",
    "top 10 genes by mean"); when it is typed exactly as the column is spelled
    and that spelling is not plain lower case ("MAX trend"); in a correlation
    ("correlation between mean and std"); when a condition follows ("std above
    2"); and when "trend" follows ("max trend"), since a timeline of the
    maximum is not something ADA builds.
    """
    if _follows(reading.text, start, rf"(?:of|{GROUPING_WORDS}|{RANKING_WORDS})"):
        return True
    name = str(column)
    if name != name.lower() and name in _raw_chunks(reading.question, reading.text[start:end].split()):
        return True
    if any(re.search(rf"\b{word}\b", reading.remaining) for word in CORRELATION_WORDS):
        return True
    after = reading.text[end:]
    condition = rf"\s+{LINKING_VERB}(?:{COMPARISON_PATTERN}|between|negative|positive|zero)\b"
    return bool(re.match(condition, after) or re.match(r"\s+(?:trend|over time|timeline)\b", after))


def _read_question(
    question: str,
    dataframe: pd.DataFrame,
    roles: ColumnRoles,
    rows_not_analyzed: int = 0,
    grammar: frozenset[str] = QUERY_GRAMMAR,
    numbers_ok: bool = False,
) -> _Reading:
    """Claim every word of the question for a column, a value, a condition or the grammar.

    ``grammar`` is the set of words the plan being built can use as grammar;
    any other word left over is reported as naming nothing in the table. With
    ``numbers_ok`` a leftover number is allowed, as a numbered group may be.
    """
    q = _norm(_question_text(question))
    names = _column_names(list(dataframe.columns))
    reading = _Reading(question=question, original=q, text=q, remaining=q, names=names)
    worded = {column: forms for column, forms in names.items() if not _is_number(forms[0])}
    reading.spans = _mention_spans(q, worded)
    # "which month had the highest wind" names a period, not a column called month.
    period = re.search(r"\b(?:which|what)\s+(day|week|month|quarter|year)\b", q) if roles.date else None
    if period:
        reading.spans = {column: span for column, span in reading.spans.items() if span[0] != period.start(1)}
    for start, end in reading.spans.values():
        reading.claim(start, end, keep_text=True)
    _reread_aggregation_names(reading)
    _claim_time_part(reading, dataframe, roles)
    vocabulary = _vocabulary(dataframe, roles)
    _claim_conditions(reading, list(reading.spans), dataframe, roles, vocabulary)
    _claim_values(reading, vocabulary, dataframe, exact_only=True)
    _claim_analysis_words(reading, dataframe, roles)
    partial = _claim_partial_names(reading, dataframe, roles)
    _claim_conditions(reading, partial, dataframe, roles, vocabulary)
    _claim_values(reading, vocabulary, dataframe, exact_only=False)
    _claim_unit_conditions(reading, dataframe, roles)
    _claim_numbers(reading, dataframe, roles)
    _claim_row_label(reading, roles)
    _claim_entity(reading, dataframe, roles)
    for start, end, word in _tokens(reading.remaining):
        if word in MISSING_WORDS:
            reading.asks_missing = True
            reading.claim(start, end)
    for start, end, word in _tokens(reading.remaining):
        if word in COLUMN_WORDS and (reading.asks_missing or _after_name(reading, start)):
            reading.claim(start, end)
        elif word in FILTER_CUES and _filter_follows(reading, end):
            reading.claim(start, end)
    _, reading.month = _detect_time_filter(reading.unnamed)
    unknown = [
        words
        for run in _runs(reading, content_only=True, grammar=grammar)
        if (words := [word for _, _, word in run if not (numbers_ok and _is_number(word))])
    ]
    if unknown:
        reading.flag(_unknown_reason(question, unknown, len(dataframe), rows_not_analyzed))
    return reading


def _unknown_reason(question: str, runs: list[list[str]], rows: int, rows_not_analyzed: int) -> str:
    """Why words were left over: each unbroken run quoted as typed, or what an analysis word lacks.

    Words that ask for a comparison or a relationship ("differ", "affect") are
    known; left over, they mean ADA could not tell which columns they are about.
    """
    unknown = [run for run in runs if not set(run) <= ANALYSIS_WORDS]
    if not unknown:
        words = _raw_words(question, [word for run in runs for word in run])
        return (
            f'"{words}" asks to compare groups or relate columns, and ADA could not tell which '
            'columns: name the measure and the grouping, as in "does <measure> differ by <group>".'
        )
    phrases = list(dict.fromkeys(f'"{_raw_phrase(question, run)}"' for run in unknown))
    reason = f"ADA found no column or value called {_listed(phrases)} in this table"
    if rows_not_analyzed:
        reason += (
            f" (it looked in the {rows:,} rows analysed; the row limit left out "
            f"{rows_not_analyzed:,} others)"
        )
    return f"{reason}."


def _can_group_text(dataframe: pd.DataFrame, column: str, roles: ColumnRoles) -> bool:
    """Whether a text column can group rows when the question names it after "by" or "which".

    Any text column with two or more values can, besides the detected segments:
    a "time" column holding "1 min" and "15 min", a label on every row
    ("which taxonomy has the highest A101" names the row), or a column with
    hundreds of values, which the answer shows as its highest and lowest groups.
    """
    if column in roles.dimensions:
        return True
    series = dataframe[column]
    if column == roles.date or is_datetime64_any_dtype(series):
        return False
    if is_numeric_dtype(series) and not is_bool_dtype(series) and column != roles.identifier:
        return False
    return int(series.nunique(dropna=True)) >= 2


def _can_group_numeric(series: pd.Series) -> bool:
    """Whether a numeric column can group rows: a few values, or whole-number codes that repeat.

    A station code with 159 values repeated across thousands of rows groups;
    a weight with a different value on almost every row does not.
    """
    if is_bool_dtype(series):
        return True
    values = series.dropna()
    distinct = int(values.nunique())
    if distinct <= GROUPING_LIMIT:
        return distinct >= 1
    return bool((values % 1 == 0).all()) and distinct * 2 <= len(values)


def _choose_columns(
    question: str,
    dataframe: pd.DataFrame,
    roles: ColumnRoles,
    names: dict[str, tuple[str, ...]],
    mentioned: dict[str, int],
) -> tuple[str | None, str | None] | None:
    """Pick the measure and the grouping column, or None when the question can't be one plan.

    A numeric column named right after a grouping or ranking word groups the rows
    when no segment is named and it holds few values or repeating codes (see
    ``_can_group_numeric``): in "mean signal by timepoint" the measure is signal.
    A continuous column cannot group, and a question naming another measure is
    refused rather than answered without its "by". Two groupings ("by timepoint
    by region") are refused too. A numeric column listed among the segments
    (a 0/1 flag, a status code) groups like a text one.
    """
    numeric_columns = [column for column in roles.numeric if column in dataframe.columns]
    ranked = [column for column, start in mentioned.items() if _follows(question, start, RANKING_WORDS)]
    grouped = [column for column, start in mentioned.items() if _follows(question, start, GROUPING_WORDS)]
    text_groupings = list(dict.fromkeys(
        column for column in ranked + grouped
        if (column not in numeric_columns or column in roles.dimensions)
        and _can_group_text(dataframe, column, roles)
    ))
    numbers = [column for column in numeric_columns if column in mentioned and column not in text_groupings]
    numeric_groupings = [column for column in ranked if column in numbers] or [
        column for column in grouped if column in numbers
    ]
    if len(text_groupings) > 1 or len(numeric_groupings) > 1:
        return None
    segments = [
        column for column in roles.dimensions if column in mentioned and column not in numeric_columns
    ]
    dimension = text_groupings[0] if text_groupings else _longest(segments)
    if not numeric_groupings:
        return _longest(numbers), dimension

    grouping = numeric_groupings[0]
    others = [column for column in numbers if column != grouping]
    if dimension:
        # "top 3 products by revenue": the column after "by" is the measure.
        return None if others else (grouping, dimension)
    unnamed = _without_columns(question, tuple(mentioned), names)
    counting = re.search(r"\b(how many|count|number of)\b", unnamed)
    if others or counting or roles.measure not in (None, grouping):
        # A continuous column after "by" is grouped into bins (see _plan_from_reading).
        if _can_group_numeric(dataframe[grouping]) or (
            others and _follows(question, mentioned[grouping], GROUPING_WORDS)
        ):
            return _longest(others), grouping
        if others:
            return None
    return _longest(numbers), None


def _correlation_plan(
    question: str,
    numeric_columns: list[str],
    mentioned: dict[str, int],
    filters: tuple[ValueFilter, ...],
    year: int | None,
    month: int | None,
) -> QueryPlan | None:
    """Plan a correlation of the two numeric columns the question names, in the order named.

    Refused when it names anything else to group by ("by species", "per month"),
    since that asks for one r per group. A column named only to filter on is fine.
    """
    pair = [column for column in mentioned if column in numeric_columns]
    filtered = {item.column for item in filters}
    if len(pair) != 2 or _detect_grain(question):
        return None
    if any(column not in pair and column not in filtered for column in mentioned):
        return None
    return QueryPlan(
        intent="correlation", measure=pair[0], partner=pair[1], filters=filters, year=year, month=month
    )


def parse_question(question: str, dataframe: pd.DataFrame, roles: ColumnRoles) -> QueryPlan | None:
    """Turn a plain-English question into an explicit plan, or None if unsupported.

    The plan uses every column, value, grouping and condition the question
    names. When it can't, the question is refused and ``rephrase_hint`` says why.
    Questions that compare groups or ask what correlates with a column are
    planned first (see ``_comparison_plan``).
    """
    return _plan_question(question, dataframe, roles)[0]


def _plan_question(
    question: str, dataframe: pd.DataFrame, roles: ColumnRoles, rows_not_analyzed: int = 0
) -> tuple[QueryPlan | None, str | None]:
    """The plan for a question, or None with the reason it was refused when there is one."""
    if not _norm(question):
        return None, None
    compared = _comparison_plan(question, dataframe, roles)
    if compared is None and unsupported_phrasing(question):
        return None, None
    plan, reason = (compared, None) if compared else _rule_plan(question, dataframe, roles, rows_not_analyzed)
    if plan is not None and (problem := _running_total_problem(plan, roles)):
        return None, problem
    return plan, reason


def _running_total_problem(plan: QueryPlan, roles: ColumnRoles) -> str | None:
    """Why a plan may not add up, average or correlate a running total the brief set aside."""
    counters = set(running_total_columns(roles))
    used = [column for column in (plan.measure, plan.partner, *plan.columns) if column in counters]
    if not used or plan.intent in ("missing", "duplicates", "outliers", "count"):
        return None
    if plan.intent == "aggregate" and plan.aggregation in ("max", "min"):
        return None
    return (
        f"{used[0]} keeps climbing like a meter reading and drops back only when it resets, so the brief "
        "set it aside: adding it up, averaging it or correlating it would mean nothing. Ask for its "
        "maximum, or use the column that records each reading's own amount."
    )


def _rule_plan(
    question: str, dataframe: pd.DataFrame, roles: ColumnRoles, rows_not_analyzed: int = 0
) -> tuple[QueryPlan | None, str | None]:
    """The plan for a question that is not a comparison, or None with the reason."""
    reading = _read_question(question, dataframe, roles, rows_not_analyzed)
    if reading.problem:
        return None, reading.problem
    if reading.asks_missing:
        plan = _missing_plan(reading)
    elif reading.duplicates:
        plan = _duplicates_plan(reading)
    elif reading.outliers:
        plan = _outliers_plan(reading, dataframe, roles)
    elif reading.describe:
        plan = _describe_plan(reading, dataframe, roles)
    elif reading.across:
        plan = _columns_plan(reading)
    else:
        plan = _plan_from_reading(reading, dataframe, roles)
    if plan is None:
        return None, reading.refusal
    problem = _unused_part(plan, reading, dataframe, roles, rows_not_analyzed)
    if problem:
        return None, problem
    notes = (*plan.notes, *reading.notes, *_entity_note(reading, plan), *_weather_note(reading, plan))
    return replace(plan, notes=notes), None


def _weather_note(reading: _Reading, plan: QueryPlan) -> tuple[str, ...]:
    """Say which measure "warmest" or "coldest" was read as, when the question names none."""
    word = next(
        (word for word in (*WARM_WORDS, *COLD_WORDS) if re.search(rf"\b{word}\b", reading.unnamed)), None
    )
    if word is None or not plan.measure or plan.measure in reading.spans:
        return ()
    side = "lowest" if word in COLD_WORDS else "highest"
    return (f'Read "{word}" as the {side} {plan.measure}; name another column to rank by it instead.',)


def _entity_note(reading: _Reading, plan: QueryPlan) -> tuple[str, ...]:
    """Say what a noun for the rows was counted as: distinct IDs, or one row each."""
    if not reading.entity or plan.aggregation != "count":
        return ()
    if plan.count_column and plan.count_column == reading.entity_column:
        return (f"Counted the {reading.entity} as distinct {plan.count_column} values.",)
    if plan.count_column is None and reading.entity not in ROW_NOUNS:
        return (f"Each row counts as one of the {reading.entity}.",)
    return ()


def _missing_plan(reading: _Reading) -> QueryPlan | None:
    """Plan "which columns have missing values", or "how much of X is missing" for one column."""
    named = list(reading.mentioned)
    if len(named) > 1:
        reading.refusal = "Ask about missing values in one column, or in every column."
        return None
    return QueryPlan(
        intent="missing", aggregation="count", measure=named[0] if named else None,
        filters=reading.value_filters, conditions=tuple(reading.conditions),
        year=reading.year, month=reading.month,
    )


def _scope(reading: _Reading) -> dict:
    """The filters, conditions and time scope a reading sets, as plan fields."""
    return {
        "filters": reading.value_filters, "conditions": tuple(reading.conditions),
        "year": reading.year, "month": reading.month,
    }


def _duplicates_plan(reading: _Reading) -> QueryPlan | None:
    """Plan "how many duplicate rows", or "duplicate values of X" for one column."""
    named = list(reading.mentioned)
    if len(named) > 1:
        reading.refusal = "Ask about duplicate rows, or duplicate values in one column."
        return None
    return QueryPlan(
        intent="duplicates", aggregation="count", count_column=named[0] if named else None, **_scope(reading)
    )


def _outliers_plan(reading: _Reading, dataframe: pd.DataFrame, roles: ColumnRoles) -> QueryPlan | None:
    """Plan "outliers in X", or "are there outliers" over every numeric reading."""
    named = list(reading.mentioned)
    numeric = [column for column in named if _is_measurement(dataframe[column])]
    if len(named) > 1 or len(numeric) != len(named):
        reading.refusal = "Ask about outliers in one numeric column, or in every column."
        return None
    if not numeric and reading.entity and roles.table_shape == "matrix" and roles.measure:
        # "outlier spectra": each row of a matrix is judged by the row summary the brief follows.
        note = f"Judged each row by {roles.measure}, the summary the brief follows."
        return QueryPlan(
            intent="outliers", aggregation="count", measure=roles.measure, notes=(note,), **_scope(reading)
        )
    columns = () if numeric else tuple(numeric_readings(dataframe, roles))
    return QueryPlan(
        intent="outliers", aggregation="count", measure=numeric[0] if numeric else None,
        columns=columns, **_scope(reading),
    )


def _describe_plan(reading: _Reading, dataframe: pd.DataFrame, roles: ColumnRoles) -> QueryPlan | None:
    """Plan "distribution of X", optionally by a grouping: count, mean, SD, quartiles, range and skew."""
    chosen = _choose_columns(reading.text, dataframe, roles, reading.names, reading.mentioned)
    measure, dimension = chosen if chosen else (None, None)
    measure = measure or (roles.measure if not reading.mentioned else None)
    if measure is None or not _is_measurement(dataframe[measure]):
        reading.refusal = "A distribution needs one numeric column, as in \"distribution of <measure>\"."
        return None
    if dimension and not _follows(reading.text, reading.mentioned.get(dimension, 0)):
        dimension = None
    return QueryPlan(
        intent="describe", aggregation="mean", measure=measure, dimension=dimension, **_scope(reading)
    )


def _columns_plan(reading: _Reading) -> QueryPlan | None:
    """Plan "which column has the highest average": the columns themselves, ranked by one aggregation."""
    if reading.mentioned or len(reading.across) < 2:
        reading.refusal = (
            "Ranking columns compares every numeric reading at once, so it can't also name one column."
            if reading.mentioned else "This table has fewer than two numeric readings to rank."
        )
        return None
    aggregation = reading.spread or ("quantile" if reading.quantile else None) or next(
        (
            AGGREGATION_WORDS[word]
            for word in AGGREGATION_WORDS
            if word not in SUPERLATIVE_WORDS and re.search(rf"\b{word}\b", reading.unnamed)
        ),
        "mean",
    )
    top = re.search(r"\b(?:top|bottom)\s+(\d{1,3})\b", reading.unnamed)
    return QueryPlan(
        intent="columns", aggregation=aggregation, columns=reading.across, quantile=reading.quantile,
        ascending=bool(re.search(rf"{ASCENDING_PATTERN}|\bbottom\b", reading.unnamed)),
        top_n=int(top.group(1)) if top else 1, **_scope(reading),
    )


def _is_measurement(series: pd.Series) -> bool:
    return is_numeric_dtype(series) and not is_bool_dtype(series)


def _plan_from_reading(reading: _Reading, dataframe: pd.DataFrame, roles: ColumnRoles) -> QueryPlan | None:
    """Choose the intent, measure and grouping for a question whose words are all read."""
    q = reading.text
    # Grammar is read with every column name blanked (see _Reading.unnamed).
    unnamed = reading.unnamed
    numeric_columns = [column for column in roles.numeric if column in dataframe.columns]
    names = reading.names
    mentioned = reading.mentioned
    # "which month had the highest wind" ranks the periods of the date column;
    # the grain word names the period, not a column that happens to be called "month".
    period = re.search(r"\b(?:which|what)\s+(day|week|month|quarter|year)\b", unnamed) if roles.date else None
    year, month = reading.year, reading.month
    grain = _detect_grain(unnamed)
    filters_including_dimension = reading.value_filters
    conditions = tuple(reading.conditions)

    if any(re.search(rf"\b{word}\b", unnamed) for word in CORRELATION_WORDS):
        correlation = _correlation_plan(
            unnamed, numeric_columns, mentioned, filters_including_dimension, year, month
        )
        if correlation is None:
            return None
        assert correlation.measure is not None and correlation.partner is not None
        # A category code is compared across its groups, or cross-tabulated with another code.
        return _relation_plan(
            correlation.measure, correlation.partner, dataframe, roles,
            {**_scope(reading), "filters": filters_including_dimension},
        )

    pair = _grouping_pair(reading, dataframe, roles)
    if pair:
        return _crosstab_plan(reading, dataframe, roles, pair)
    chosen = _choose_columns(q, dataframe, roles, names, mentioned)
    groupings = [column for column, start in mentioned.items() if _follows(q, start)]
    if chosen is None:
        if len(groupings) > 1:
            reading.refusal = _two_groupings(groupings)
        return None
    measure, dimension = chosen
    time_part = reading.time_part
    by_year = re.search(r"\b(?:by|per|each|every)\s+(?:the\s+)?(?:years?|yearly)\b", unnamed)
    if dimension and roles.date and (time_part or by_year) and dimension != roles.date:
        # "count of crop by year", "average PM2.5 by station and month of the year": a cross-table.
        pair = (roles.date, dimension)
        return _crosstab_plan(reading, dataframe, roles, pair, time_part or "year", measure)
    if time_part:
        dimension = roles.date
    # "revenue by order date" asks for the trend; with a segment as well it asks for two groupings.
    by_date = roles.date in mentioned and _follows(q, mentioned[roles.date], GROUPING_WORDS)
    if by_date and dimension:
        reading.refusal = _two_groupings([dimension, str(roles.date)])
        return None
    # A continuous grouping is split at its quartiles: "average growth by temperature".
    bins = (
        4 if dimension and not time_part and _is_measurement(dataframe[dimension])
        and dimension not in roles.dimensions and not _can_group_numeric(dataframe[dimension]) else None
    )
    # A grouped answer groups by the matched dimension, so its values are not
    # filters. Ungrouped answers keep them: "revenue for Region West" filters
    # to West even though "Region" also matched as the dimension.
    filters = tuple(item for item in filters_including_dimension if item.column != dimension)

    # Read intent words with the column names removed: "Cell Count" would make
    # "total cell count" a row count, and "Total Sales" would outrank an explicit "average".
    intent_text = unnamed
    aggregation: Aggregation | None = next(
        (
            AGGREGATION_WORDS[word]
            for word in AGGREGATION_WORDS
            if re.search(rf"\b{word}\b", intent_text)
        ),
        None,
    )
    if reading.spread:
        aggregation = reading.spread
    elif reading.quantile is not None:
        aggregation = "quantile"
    top_match = re.search(r"\b(top|bottom)\s+(\d{1,3})\b", unnamed)
    if top_match and int(top_match.group(2)) < 1:
        # "top 0" is refused; a zero limit would fall back to BREAKDOWN_LIMIT rows.
        return None
    # "highest" and "largest" map to max but ask for a ranking; only the
    # literal min/max words ask for a per-group extreme.
    explicit_extreme = bool(re.search(r"\b(min|minimum|max|maximum)\b", intent_text))
    superlative = any(re.search(rf"\b{word}\b", intent_text) for word in SUPERLATIVE_WORDS)
    wants_breakdown = bool(re.search(r"\b(by|per|across|breakdown|split|each)\b", intent_text))
    # "how much" counts too when only conditions are named: "how much is below 5".
    wants_count = bool(re.search(COUNT_PATTERN, intent_text)) or (
        bool(re.search(r"\bhow much\b", intent_text)) and not measure
        and bool(conditions or filters_including_dimension)
    )
    wants_growth = any(re.search(rf"\b{word}\b", intent_text) for word in GROWTH_WORDS)
    wants_trend = by_date or grain is not None or any(phrase in intent_text for phrase in TREND_WORDS)

    base = {
        "measure": measure or roles.measure,
        "filters": filters,
        "year": year,
        "month": month,
        "grain": grain,
        "conditions": conditions,
        "quantile": reading.quantile,
        "bins": bins,
    }
    if reading.common:
        scope = {**base, "filters": filters_including_dimension}
        return _common_plan(reading, dataframe, measure, dimension, scope)

    ungrouped = {**base, "filters": filters_including_dimension, "bins": None}
    # Same default as the dashboard: rates and levels average, amounts sum. An
    # explicit "total" or "sum" still sums. It is taken from the final measure, so
    # every intent combines a given measure the same way.
    default_aggregation = metric_for(base["measure"], roles).aggregation

    if wants_count and not wants_growth:
        within = re.search(rf"\b(?:{'|'.join(SHARE_WORDS)}|rate)\b", intent_text)
        if dimension and (wants_breakdown or time_part) and within and (conditions or filters):
            # "share of Result 1 by Gender" is the share within each Gender, not the split of Result 1 rows.
            return QueryPlan(
                intent="breakdown", aggregation="share", dimension=dimension, time_part=time_part,
                **{**base, "measure": None},
            )
        if dimension and (wants_breakdown or time_part):
            # "how many participants in each condition" counts distinct ParticipantId per condition.
            counted = next(
                (
                    column for column, start in mentioned.items()
                    if column != dimension and _role_of_position(reading.original, start) == "count"
                ),
                reading.entity_column,
            )
            return QueryPlan(
                intent="breakdown", aggregation="count", dimension=dimension,
                count_column=counted, time_part=time_part, **{**base, "measure": None},
            )
        if (grain or by_date) and roles.date:
            # "how many readings per month" is a trend of row counts.
            return QueryPlan(
                intent="trend", aggregation="count", count_column=reading.entity_column,
                **{**ungrouped, "measure": None},
            )
        return _count_plan(reading, dataframe, roles, ungrouped)

    if wants_growth and roles.date:
        wants_ranked_growth = superlative or bool(re.search(r"\b(which|fastest|slowest)\b", intent_text))
        rank_dimension = dimension or (roles.dimension if wants_ranked_growth else None)
        ascending = bool(re.search(ASCENDING_PATTERN, intent_text))
        top_n = int(top_match.group(2)) if top_match else None
        return QueryPlan(
            intent="growth", aggregation=default_aggregation, dimension=rank_dimension,
            ascending=ascending, top_n=top_n, **base,
        )

    if period and superlative:
        # "which month had the most rows" ranks row counts.
        period_measure = None if not measure and re.search(r"\brows\b", intent_text) else base["measure"]
        per_period = metric_for(period_measure, roles).aggregation
        # Periods come from build_trend, which combines with the measure's default,
        # so "which month had the highest average revenue" is refused like a trend.
        if dimension:
            return None
        if _grouped_aggregation(aggregation, explicit_extreme, per_period) != per_period:
            reading.refusal = _period_aggregation_reason(period_measure, per_period)
            return None
        return QueryPlan(
            intent="rank",
            aggregation=per_period,
            top_n=1,
            ascending=bool(re.search(ASCENDING_PATTERN, intent_text)),
            **{**ungrouped, "measure": period_measure, "grain": GRAIN_WORDS[period.group(1)]},
        )

    if (top_match or superlative) and dimension:
        if top_match:
            top_n = int(top_match.group(2))
            ascending = top_match.group(1) == "bottom"
        else:
            top_n = 1
            ascending = bool(re.search(ASCENDING_PATTERN, intent_text))
        return QueryPlan(
            intent="rank",
            aggregation=_grouped_aggregation(aggregation, explicit_extreme, default_aggregation),
            dimension=dimension,
            top_n=top_n,
            ascending=ascending,
            time_part=time_part,
            **base,
        )

    if wants_trend and roles.date:
        if aggregation not in (None, default_aggregation):
            # The trend executor always uses the measure's default aggregation,
            # so "average revenue monthly" is refused (it would get sums) while
            # "average conversion rate monthly" is answered.
            reading.refusal = _period_aggregation_reason(base["measure"], default_aggregation)
            return None
        return QueryPlan(intent="trend", aggregation=default_aggregation, **ungrouped)

    if dimension and (wants_breakdown or not measure):
        return QueryPlan(
            intent="breakdown",
            aggregation=_grouped_aggregation(aggregation, explicit_extreme, default_aggregation),
            dimension=dimension,
            time_part=time_part,
            **base,
        )

    if aggregation is None and superlative and measure:
        # "the most snowfall" asks for the largest value, "the least" for the smallest.
        aggregation = "min" if re.search(ASCENDING_PATTERN, intent_text) else "max"
    if base["measure"] and (aggregation or measure):
        return QueryPlan(intent="aggregate", aggregation=aggregation or default_aggregation, **ungrouped)

    if conditions or filters_including_dimension:
        # A question that only states conditions ("pH below 5", "West") counts the rows meeting them.
        return QueryPlan(intent="count", aggregation="count", **{**ungrouped, "measure": None})
    return None


def _count_plan(
    reading: _Reading, dataframe: pd.DataFrame, roles: ColumnRoles, ungrouped: dict
) -> QueryPlan | None:
    """Plan "how many": distinct values of a named column, the rows meeting the conditions, or a total.

    "how many cities" counts distinct cities and "how many stations" a numbered
    station column. A quantity counts by its total ("how many units"); a
    measurement such as a temperature has no count, so it is refused.
    """
    counting = {**ungrouped, "measure": None}
    for column, start in reading.mentioned.items():
        series = dataframe[column]
        if _is_flag(series) and not reading.distinct and column != roles.identifier:
            # "share of Antibiotics" counts the rows where the 0/1 flag is set.
            flagged = Condition(column, "=", 1.0)
            return QueryPlan(
                intent="count", aggregation="count",
                **{**counting, "conditions": (*counting["conditions"], flagged)},
            )
        if not _is_measurement(series) or column == roles.identifier or reading.distinct:
            return QueryPlan(intent="count", aggregation="count", count_column=column, **counting)
        spelled = reading.original[start:reading.spans[column][1]]
        plural = spelled.endswith("s") and spelled not in reading.names[column]
        counted = _role_of_position(reading.original, start) == "count"
        if _labels_entities(column, series) and plural and counted:
            return QueryPlan(intent="count", aggregation="count", count_column=column, **counting)
        if metric_for(column, roles).aggregation == "sum":
            return QueryPlan(intent="aggregate", aggregation="sum", **{**ungrouped, "measure": column})
        reading.refusal = (
            f'"How many" counts rows or distinct values, and {column} is a measurement; '
            "ask for its average or its total."
        )
        return None
    return QueryPlan(intent="count", aggregation="count", count_column=reading.entity_column, **counting)


# Words joining two groupings: "by A and B", "by A by B", "A x B".
PAIR_JOINS = ("and", "by", "and by", "x", "vs", "versus", "per", "and per", "across", "and across")


def _grouping_pair(reading: _Reading, dataframe: pd.DataFrame, roles: ColumnRoles) -> tuple[str, str] | None:
    """Two groupings named together after a grouping word: "by Antibiotics and Probiotics"."""
    spans = sorted(
        (start, end, column)
        for column, (start, end) in reading.spans.items()
        if column not in reading.consumed
    )
    for (first_start, first_end, first), (second_start, _, second) in zip(spans, spans[1:], strict=False):
        joined = " ".join(reading.original[first_end:second_start].split())
        if joined not in PAIR_JOINS or not _follows(reading.original, first_start, GROUPING_WORDS):
            continue
        if _follows(reading.original, first_start, "per") and joined in ("by", "and by"):
            continue
        if all(_groups_rows(column, dataframe, roles) for column in (first, second)):
            return first, second
    return None


def _groups_rows(column: str, dataframe: pd.DataFrame, roles: ColumnRoles) -> bool:
    """Whether a column can be one side of a cross-table: text, codes or a few numbers."""
    if column == roles.date:
        return False
    series = dataframe[column]
    if _is_measurement(series):
        return column in roles.dimensions or int(series.nunique(dropna=True)) <= GROUPING_LIMIT
    return _can_group_text(dataframe, column, roles)


def _crosstab_plan(
    reading: _Reading,
    dataframe: pd.DataFrame,
    roles: ColumnRoles,
    pair: tuple[str, str],
    time_part: TimePart | None = None,
    measure: str | None = None,
) -> QueryPlan | None:
    """Plan a cross-table of two groupings: row counts, or one measure combined per cell."""
    rows, columns = pair
    others = [
        column for column in reading.mentioned
        if column not in pair and _is_measurement(dataframe[column])
    ]
    if len(others) > 1 or (measure and others and others != [measure]):
        reading.refusal = f"A cross-table combines one measure, and the question names {_and_listed(others)}."
        return None
    measure = measure or (others[0] if others else None)
    for column in pair:
        if column != roles.date and dataframe[column].nunique(dropna=True) > GROUPING_LIMIT:
            reading.refusal = (
                f"{column} has {dataframe[column].nunique(dropna=True):,} values, too many for a cross-table "
                f"(the limit is {GROUPING_LIMIT})."
            )
            return None
    text = reading.unnamed
    counting = bool(re.search(COUNT_PATTERN, text)) or measure is None
    aggregation = reading.spread or ("quantile" if reading.quantile is not None else None) or next(
        (AGGREGATION_WORDS[word] for word in AGGREGATION_WORDS if re.search(rf"\b{word}\b", text)), None
    )
    if counting and not measure:
        aggregation = "count"
    elif aggregation is None or aggregation in ("count",):
        aggregation = metric_for(measure, roles).aggregation
    filters = tuple(item for item in reading.value_filters if item.column not in pair)
    return QueryPlan(
        intent="crosstab", aggregation=aggregation, measure=None if aggregation == "count" else measure,
        dimension=rows, second_dimension=columns, time_part=time_part, quantile=reading.quantile,
        **{**_scope(reading), "filters": filters},
    )


def _common_plan(
    reading: _Reading, dataframe: pd.DataFrame, measure: str | None, dimension: str | None, scope: dict
) -> QueryPlan | None:
    """Plan "which X is most common" as the largest group of X, or "most common X" of a number as its mode."""
    ascending = reading.common == "least"
    named = list(reading.mentioned)
    if not dimension and not measure and len(named) == 1 and not _is_measurement(dataframe[named[0]]):
        dimension = named[0]
    if dimension and not measure:
        filters = tuple(item for item in scope["filters"] if item.column != dimension)
        return QueryPlan(
            intent="rank", aggregation="count", dimension=dimension, top_n=1, ascending=ascending,
            **{**scope, "measure": None, "filters": filters},
        )
    if measure and not dimension and not ascending and _is_measurement(dataframe[measure]):
        return QueryPlan(intent="aggregate", aggregation="mode", **{**scope, "measure": measure})
    reading.refusal = 'Ask "which <column> is most common" for its largest group, or "most common <number>".'
    return None


def _two_groupings(columns: list[str]) -> str:
    return f"The question asks for two groupings, {_and_listed(columns)}; ADA answers one at a time."


def _period_aggregation_reason(measure: str | None, per_period: Aggregation) -> str:
    """Why a trend or a period ranking can't combine a measure the way the question asks."""
    combined = {"mean": "averaged", "sum": "totalled"}.get(per_period, "counted")
    return (
        f"Periods of {measure or 'rows'} are {combined}, as on the dashboard, so ADA "
        "won't build a timeline that combines them another way."
    )


def _plan_columns(plan: QueryPlan, dataframe: pd.DataFrame, roles: ColumnRoles) -> set[str]:
    """Every column the plan reads, the date included when it scopes or groups by time."""
    skipped = {"intent", "aggregation", "grain", "source", "time_part", "notes"}
    used: set[str] = set()
    for item in fields(plan):
        if item.name in skipped:
            continue
        value = getattr(plan, item.name)
        for entry in value if isinstance(value, tuple) else (value,):
            column = getattr(entry, "column", entry)
            if isinstance(column, str) and column in dataframe.columns:
                used.add(column)
    timed = plan.intent in ("trend", "growth") or plan.year or plan.month or plan.time_part
    if roles.date and (timed or (plan.intent == "rank" and plan.grain)):
        used.add(roles.date)
    return used


def _unused_part(
    plan: QueryPlan, reading: _Reading, dataframe: pd.DataFrame, roles: ColumnRoles, rows_not_analyzed: int
) -> str | None:
    """Why the plan answers less than the question asks, or None when it uses every part."""
    used = _plan_columns(plan, dataframe, roles)
    for column in reading.spans:
        if column not in used:
            return _unused_column_reason(column, reading, dataframe, rows_not_analyzed)
    planned = {item.column for item in plan.filters}
    for item in reading.value_filters:
        if item.column not in planned:
            return (
                f"{_listed(list(item.values))} is a value of {item.column}, which the answer "
                "groups by, so it can't also filter the rows."
            )
    if set(reading.conditions) - set(plan.conditions):
        return "The answer would leave out a condition the question sets."
    if reading.top_n and plan.intent not in ("rank", "growth"):
        return f'"{reading.top_n}" needs a column to rank, as in "{reading.top_n} <group> by <measure>".'
    grain = _detect_grain(reading.unnamed)
    periods = plan.intent in ("trend", "growth") or (plan.intent == "rank" and not plan.dimension) or (
        plan.time_part == "year" and grain == "Y"
    )
    if grain and not periods:
        if not roles.date:
            return f"This table has no date column, so there is no {GRAIN_NAMES[grain]} view."
        return (
            f"The answer would leave out the {GRAIN_NAMES[grain]} grouping; "
            "ask for one grouping at a time."
        )
    if reading.time_part and plan.time_part != reading.time_part:
        return f"The answer would leave out the {reading.time_part} grouping."
    return _unused_analysis_word(plan, reading)


def _unused_analysis_word(plan: QueryPlan, reading: _Reading) -> str | None:
    """Why the plan would drop a summary the question asks for: a spread, a percentile, distinct values."""
    if reading.spread and plan.aggregation != reading.spread:
        label = AGGREGATION_LABELS[reading.spread].removesuffix(" of").lower()
        return f"The answer would leave out the {label} the question asks for."
    if reading.quantile is not None and plan.quantile is None:
        return "The answer would leave out the percentile the question asks for."
    if reading.distinct and not plan.count_column and plan.intent != "duplicates":
        return '"Distinct" needs a column to count, as in "how many distinct <column>".'
    if reading.across and plan.intent != "columns":
        return "The question ranks several columns, and the answer would use only one."
    return None


def _rows_scope(rows: int, rows_not_analyzed: int) -> str:
    if rows_not_analyzed:
        return f" in the {rows:,} rows analysed (the row limit left out {rows_not_analyzed:,} others)"
    return " in this table"


def _unused_column_reason(
    column: str, reading: _Reading, dataframe: pd.DataFrame, rows_not_analyzed: int
) -> str:
    """Say why a column the question names has no place in the plan."""
    series = dataframe[column]
    numeric = is_numeric_dtype(series) and not is_bool_dtype(series)
    distinct = int(series.nunique(dropna=True))
    if _follows(reading.original, reading.spans[column][0]):
        if distinct < 2:
            held = f"a single value ({series.dropna().iloc[0]})" if distinct else "no values"
            scope = _rows_scope(len(dataframe), rows_not_analyzed)
            return f"{column} has {held}{scope}, so there is nothing to group by."
        if numeric and not _can_group_numeric(series):
            return (
                f"{column} has {distinct:,} distinct values, too many to group by "
                f"(the limit is {GROUPING_LIMIT})."
            )
        return f"The answer would leave out the grouping by {column}; ADA answers one grouping at a time."
    if not numeric:
        return f"{column} holds text, not numbers, so it can't be averaged or totalled."
    measures = [
        name for name in reading.mentioned
        if is_numeric_dtype(dataframe[name]) and not is_bool_dtype(dataframe[name])
    ]
    if len(measures) > 1:
        return f"The answer can use one measure, and the question names {_and_listed(measures)}."
    return f"The answer would leave out {column}, so ADA won't answer a narrower question."


def _grouped_aggregation(
    aggregation: Aggregation | None, explicit_extreme: bool, default: Aggregation
) -> Aggregation:
    """Choose the aggregation a rank or breakdown computes per group.

    An explicit "total", "average" or "median" is honoured; "highest" asks for
    a ranking, so only the literal "min" and "max" ask for a per-group extreme.
    """
    if aggregation in ("mean", "median", "sum", *SPREAD_AGGREGATIONS):
        return aggregation
    if aggregation in ("min", "max") and explicit_extreme:
        return aggregation
    return default


class TimeScopeUnavailable(Exception):
    """Raised when a plan asks for a year or month but the file has no date column."""


# Full month names by number, for answer sentences.
MONTH_LABELS = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December",
}


def _when_label(plan: QueryPlan) -> str:
    month = MONTH_LABELS.get(plan.month) if plan.month else None
    year = str(plan.year) if plan.year else None
    return " ".join(part for part in (month, year) if part)


def _apply_filters(
    dataframe: pd.DataFrame, plan: QueryPlan, roles: ColumnRoles
) -> tuple[pd.DataFrame, list[str]]:
    working = dataframe
    applied: list[str] = []
    for value_filter in plan.filters:
        if value_filter.column not in working.columns:
            raise ValueError(f"Unknown filter column: {value_filter.column}")
        working = working.loc[_value_mask(working[value_filter.column], value_filter.values)]
        applied.append(_describe_filter(value_filter))
    for condition in plan.conditions:
        if condition.column not in working.columns:
            raise ValueError(f"Unknown condition column: {condition.column}")
        working = working.loc[_condition_mask(working[condition.column], condition)]
    applied.extend(_describe_conditions(plan.conditions))
    if plan.year or plan.month:
        if not roles.date or roles.date not in working.columns:
            # Report the scope as unavailable rather than answer with the all-time figure.
            raise TimeScopeUnavailable(_when_label(plan))
        dates = working[roles.date]
        if plan.year:
            working = working.loc[dates.dt.year == plan.year]
            dates = working[roles.date]
        if plan.month:
            working = working.loc[dates.dt.month == plan.month]
        applied.append(f"{roles.date} in {_when_label(plan)}")
    return working, applied


def _value_mask(series: pd.Series, values: tuple[str, ...]) -> pd.Series:
    """Rows whose value is one of ``values``; numbers compare as numbers, so "2" matches 2.0."""
    if is_numeric_dtype(series) and not is_bool_dtype(series):
        wanted = pd.to_numeric(pd.Series(list(values), dtype=object), errors="coerce").dropna()
        if len(wanted) == len(values):
            return series.isin(wanted.tolist())
    return series.astype(str).isin(values)


def _condition_mask(series: pd.Series, condition: Condition) -> pd.Series:
    if condition.operator == "missing":
        return series.isna()
    values = pd.to_numeric(series, errors="coerce")
    compare = {"=": values.eq, "<": values.lt, "<=": values.le, ">": values.gt, ">=": values.ge}
    return compare[condition.operator](condition.value)


def _number_text(value: float | None) -> str:
    number = float(value or 0.0)
    return f"{int(number)}" if number.is_integer() else f"{number:g}"


def _describe_filter(value_filter: ValueFilter) -> str:
    if len(value_filter.values) == 1:
        return f"{value_filter.column} = {value_filter.values[0]}"
    return f"{value_filter.column} in ({', '.join(value_filter.values)})"


def _describe_condition(condition: Condition) -> str:
    if condition.operator == "missing":
        return f"{condition.column} missing"
    return f"{condition.column} {OPERATOR_WORDS[condition.operator]} {_number_text(condition.value)}"


def _describe_conditions(conditions: tuple[Condition, ...]) -> list[str]:
    """Each condition in words, with a range on one column as "pH between 5 and 6"."""
    described: list[str] = []
    index = 0
    while index < len(conditions):
        condition, following = conditions[index], conditions[index + 1:index + 2]
        if following and condition.operator == ">=" and following[0].operator == "<=" and (
            following[0].column == condition.column
        ):
            low, high = _number_text(condition.value), _number_text(following[0].value)
            described.append(f"{condition.column} between {low} and {high}")
            index += 2
            continue
        described.append(_describe_condition(condition))
        index += 1
    return described


def _share_text(share: float) -> str:
    """A share of rows as a percentage; a share below 0.1% is not rounded to zero."""
    return "under 0.1%" if 0 < share < 0.1 else f"{share:.1f}%"


def _scoped(applied_filters: list[str]) -> str:
    return f" · filtered to {', '.join(applied_filters)}" if applied_filters else ""


def _aggregate_series(series: pd.Series, aggregation: Aggregation, quantile: float | None = None) -> float:
    """Combine a column's values; spreads are sample statistics (n − 1), a tied mode its smallest value."""
    values = series.dropna()
    if aggregation == "iqr":
        return float(values.quantile(0.75) - values.quantile(0.25))
    if aggregation == "quantile":
        return float(values.quantile(quantile if quantile is not None else 0.5))
    if aggregation == "mode":
        modes = values.mode()
        return float(modes.iloc[0]) if not modes.empty else float("nan")
    return float(getattr(values, aggregation)())


def _aggregation_label(plan: QueryPlan) -> str:
    """The aggregation in words before a column name: "Average", "Standard deviation of"."""
    if plan.aggregation == "quantile":
        return f"{_ordinal(round((plan.quantile or 0.5) * 100, 1))} percentile of"
    return AGGREGATION_LABELS[plan.aggregation]


def _ordinal(number: float) -> str:
    """90 gives "90th", 1 gives "1st", 2.5 gives "2.5th"."""
    if not float(number).is_integer():
        return f"{number:g}th"
    whole = int(number)
    suffix = "th" if 10 <= whole % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(whole % 10, "th")
    return f"{whole}{suffix}"


def _aggregation_formula(plan: QueryPlan, column: str | None) -> str:
    """The aggregation as the calculation line writes it: "std(X)", "quantile(X, 0.9)"."""
    if plan.aggregation == "quantile":
        return f"quantile({column}, {plan.quantile or 0.5:g})"
    if plan.aggregation == "iqr":
        return f"Q3 − Q1 of {column}"
    return f"{plan.aggregation}({column})"


def _group_key(plan: QueryPlan, dataframe: pd.DataFrame) -> str:
    """The column a grouped answer's table is keyed by: the grouping, or "Hour" or "Weekday"."""
    assert plan.dimension is not None
    if not plan.time_part:
        return plan.dimension
    name = plan.time_part.capitalize()
    return f"{name} of {plan.dimension}" if name in dataframe.columns else name


def _group_values(dataframe: pd.DataFrame, plan: QueryPlan) -> pd.Series:
    """Each row's group: the grouping column's value, its part of the date, or its quantile bin.

    Rows without a value form their own group, so shares add up to the full total.
    """
    assert plan.dimension is not None
    column = dataframe[plan.dimension]
    if plan.time_part:
        return _time_part_values(column, plan.time_part)
    if plan.bins:
        return _binned(column, plan.bins)
    return column.astype(object).where(column.notna(), unlabelled_label(column.dropna().unique()))


def _time_part_values(stamps: pd.Series, part: TimePart) -> pd.Series:
    """The hour, weekday, month of year, quarter of year or year of each timestamp."""
    return {
        "hour": stamps.dt.hour, "weekday": stamps.dt.dayofweek, "month": stamps.dt.month,
        "quarter": stamps.dt.quarter, "year": stamps.dt.year,
    }[part]


def _binned(values: pd.Series, bins: int) -> pd.Series:
    """Split a numeric column into ``bins`` groups of about equal size, labelled with their ranges.

    Two groups split at the median ("low" and "high"); more are quantile bins
    in order ("Q1: 12 to 18.5"). The labels are an ordered category, so groups
    keep their order in a table.
    """
    present = values.dropna()
    edges = np.unique(present.quantile(np.linspace(0, 1, bins + 1)).to_numpy())
    if len(edges) < 3:
        edges = np.unique(present.to_numpy())[[0, -1]] if present.nunique() > 1 else edges
    shown = [format_number(float(edge), str(values.name), column_values=present) for edge in edges]
    if len(edges) == 3 and bins == 2:
        labels = [f"low (up to {shown[1]})", f"high (above {shown[1]})"]
    else:
        prefix = "Q" if bins == 4 and len(edges) == 5 else "Group "
        labels = [
            f"{prefix}{position + 1}: {shown[position]} to {shown[position + 1]}"
            for position in range(len(edges) - 1)
        ]
    cut = pd.cut(values, edges, labels=labels, include_lowest=True, duplicates="drop")
    return cut.astype(pd.CategoricalDtype(labels, ordered=True))


#: Values a group needs before an average or median ranks it against the others.
MIN_RANK_VALUES = 5


def _grouped_frame(
    dataframe: pd.DataFrame, plan: QueryPlan, value_label: str
) -> pd.DataFrame:
    """The plan's value per group, with how many values each group has (``n``) when a measure is combined."""
    assert plan.dimension is not None
    key = _group_key(plan, dataframe)
    groups = _group_values(dataframe, plan).rename(key)
    if plan.time_part:
        groups = groups[dataframe[plan.dimension].notna()]
    working = dataframe.loc[groups.index]
    if plan.aggregation == "count" or not plan.measure:
        if plan.count_column:
            grouped = working.groupby(groups, observed=True)[plan.count_column].nunique()
        else:
            grouped = working.groupby(groups, observed=True).size()
        grouped = grouped.rename(value_label).reset_index()
    else:
        values = working[plan.measure].groupby(groups, observed=True)
        grouped = pd.DataFrame({value_label: values.agg(_group_function(plan)), "n": values.count()})
        grouped = grouped.reset_index()
    grouped = grouped.sort_values(value_label, ascending=plan.ascending, kind="stable")
    # Distinct counts don't add up across groups: one subject can be in several.
    if plan.aggregation in ("sum", "count") and not plan.count_column:
        values = grouped[value_label].to_numpy(dtype=float)
        total = float(values.sum())
        # Shares need every group to have the same sign; mixed signs give values
        # like 500% and -250%.
        if total and ((values >= 0).all() or (values <= 0).all()):
            grouped["Share %"] = (grouped[value_label] / total * 100).round(1)
    return grouped.reset_index(drop=True)


def _group_function(plan: QueryPlan):
    """What pandas runs per group for the plan's aggregation."""
    if plan.aggregation in ("iqr", "quantile", "mode"):
        return lambda values: _aggregate_series(values, plan.aggregation, plan.quantile)
    return plan.aggregation


def execute_plan(
    plan: QueryPlan, dataframe: pd.DataFrame, roles: ColumnRoles, *, rows_not_analyzed: int = 0
) -> QueryAnswer:
    """Run a plan locally and return the answer with its calculation trace.

    The answer ends with how the question was read, how many rows it used and,
    when the row limit left ``rows_not_analyzed`` rows out, that it covers only
    the rows analysed. Raises ``ValueError`` for an unknown column or intent.
    The returned ``question`` is empty; ``answer_question`` fills it in.
    """
    return _with_scope(_run_plan(plan, dataframe, roles), dataframe, roles, rows_not_analyzed)


def _with_scope(
    result: QueryAnswer, dataframe: pd.DataFrame, roles: ColumnRoles, rows_not_analyzed: int
) -> QueryAnswer:
    """Add the plan's readings, the rows the answer used and any rows the row limit left out."""
    plan = result.plan
    sentences = [result.answer, *plan.notes]
    # A share within groups reads its conditions as what is counted, not as a filter.
    scoped = replace(plan, filters=(), conditions=()) if plan.aggregation == "share" else plan
    try:
        working, _ = _apply_filters(dataframe, scoped, roles)
    except TimeScopeUnavailable:
        working = None
    if working is not None and not working.empty:
        sentences.extend(_flag_cautions(plan, dataframe, working, roles))
    # A count says its own rows, and an unfiltered aggregate says how many rows it used.
    if working is not None and not working.empty and plan.intent not in ("count", "missing"):
        if len(working) < len(dataframe):
            rows = _analysed_rows(dataframe, roles, rows_not_analyzed)
            sentences.append(f"Based on {len(working):,} of {rows}.")
        elif plan.intent != "aggregate" or rows_not_analyzed:
            sentences.append(f"Based on {_analysed_rows(dataframe, roles, rows_not_analyzed, every=True)}.")
    elif rows_not_analyzed:
        sentences.append(f"ADA analysed {_analysed_rows(dataframe, roles, rows_not_analyzed, every=True)}.")
    return replace(result, answer=" ".join(sentences))


# Intents whose answer combines a measure's values, so a flagged value can sway it.
CAUTION_INTENTS = (
    "aggregate", "rank", "breakdown", "trend", "growth", "compare", "describe", "correlation", "crosstab",
)


def _flag_cautions(
    plan: QueryPlan, dataframe: pd.DataFrame, working: pd.DataFrame, roles: ColumnRoles
) -> list[str]:
    """Short cautions when an answer uses values the data checks flag.

    A maximum or minimum that is itself flagged says so; any other answer says
    how many flagged values it includes. A grouped answer also names a group
    the checks flag for the measure, such as a city whose values were copied
    from another column.
    """
    if plan.intent not in CAUTION_INTENTS:
        return []
    columns = [
        column for column in (plan.measure, plan.partner)
        if column and column in dataframe.columns and _is_measurement(dataframe[column])
    ]
    if not columns:
        return []
    issues = _data_checks(dataframe, roles)
    cautions: list[str] = []
    for column in columns:
        flagged = _value_issues(issues, column, working.index)
        if not flagged:
            continue
        cells = pd.Index([])
        for issue in flagged:
            cells = cells.union(working.index.intersection(issue.cells[column]))
        summaries = "; ".join(issue.summary for issue in flagged)
        if plan.intent == "aggregate" and plan.aggregation in ("max", "min"):
            value = _aggregate_series(working[column], plan.aggregation)
            if value in set(working.loc[cells, column].dropna().astype(float)):
                cautions.append(
                    f"Caution: the data checks flag this value ({summaries}), so it is more likely an error "
                    "or a code than a reading."
                )
            continue
        noun = "value" if len(cells) == 1 else "values"
        cautions.append(
            f"Caution: this uses {len(cells):,} {column} {noun} the data checks flag ({summaries})."
        )
    if plan.dimension and plan.measure and plan.dimension in working.columns:
        present = {str(value) for value in working[plan.dimension].dropna().unique()}
        for issue in issues:
            if issue.group is None or plan.measure not in issue.columns or issue.check in VALUE_CHECKS:
                continue
            groups = issue.group if isinstance(issue.group, tuple) else (issue.group,)
            named = [group_label(plan.dimension, group) for group in groups if str(group) in present]
            if named:
                cautions.append(
                    f"Caution: the data checks flag {plan.measure} for {_and_listed(named)}: {issue.summary}."
                )
    return cautions


def _analysed_rows(
    dataframe: pd.DataFrame, roles: ColumnRoles, rows_not_analyzed: int, every: bool = False
) -> str:
    """The rows an answer could use: "all 1,421 rows", or "the 1,000,000 sampled rows (of 3,108,545)".

    A dated table keeps its newest rows under the row limit; one without a date is sampled at random.
    """
    rows = len(dataframe)
    if not rows_not_analyzed:
        return f"all {rows:,} rows" if every else f"{rows:,} rows"
    kept = "newest" if roles.date else "sampled"
    return f"the {rows:,} {kept} rows (of {rows + rows_not_analyzed:,})"


def _run_plan(plan: QueryPlan, dataframe: pd.DataFrame, roles: ColumnRoles) -> QueryAnswer:
    """Compute the answer to a plan, without the scope sentences ``_with_scope`` adds."""
    columns = (plan.measure, plan.dimension, plan.count_column, plan.partner)
    for column in (*columns, *(condition.column for condition in plan.conditions)):
        if column is not None and column not in dataframe.columns:
            raise ValueError(f"Unknown column in plan: {column}")
    if plan.time_part and not (plan.dimension and is_datetime64_any_dtype(dataframe[plan.dimension])):
        raise ValueError("Grouping by a part of the date needs a date column")
    if plan.intent == "correlation":
        pair = (plan.measure, plan.partner)
        if None in pair or plan.measure == plan.partner:
            raise ValueError("A correlation needs two different columns")
        if not all(is_numeric_dtype(dataframe[column]) for column in pair):
            raise ValueError("A correlation needs two numeric columns")

    if plan.aggregation == "share":
        return _execute_shares(plan, dataframe, roles)
    try:
        working, applied = _apply_filters(dataframe, plan, roles)
    except TimeScopeUnavailable as unavailable:
        return QueryAnswer(
            question="",
            plan=plan,
            answer=(
                f"This file has no date column, so I cannot narrow the answer to "
                f"{unavailable}. Set a date in ADA's schema detection and ask again."
            ),
            calculation=f"no date column available to scope to {unavailable}",
        )
    scope = _scoped(applied)
    if plan.intent == "count":
        return _execute_count(plan, dataframe, working, roles, scope, applied)
    if working.empty:
        return QueryAnswer(
            question="",
            plan=plan,
            answer="No rows match that scope, so there is nothing to calculate.",
            calculation=f"0 rows after filters{scope}",
        )

    if plan.intent == "missing":
        return _execute_missing(plan, working, scope, applied)

    if plan.intent == "aggregate":
        assert plan.measure is not None
        value = _aggregate_series(working[plan.measure], plan.aggregation, plan.quantile)
        label = _aggregation_label(plan)
        rows = int(working[plan.measure].notna().sum())
        if rows == 0:
            # Summing nothing gives 0, which reads as a measured zero rather
            # than as an absence.
            return QueryAnswer(
                question="",
                plan=plan,
                answer=(
                    f"{plan.measure} has no values{_phrase(applied)}, so there is "
                    "nothing to total."
                ),
                calculation=f"0 non-missing {plan.measure} values{scope}",
            )
        answer = (
            f"{with_aggregate(label, plan.measure)}{_phrase(applied)} is "
            f"{format_number(value, plan.measure, column_values=working[plan.measure].dropna())}, "
            f"calculated from {rows:,} rows."
        )
        if plan.aggregation == "mode":
            answer += _mode_note(working[plan.measure])
        return QueryAnswer(
            question="",
            plan=plan,
            answer=answer,
            calculation=f"{_aggregation_formula(plan, plan.measure)}{scope}",
        )

    if plan.intent == "describe":
        return _execute_describe(plan, working, scope, applied)

    if plan.intent == "outliers":
        return _execute_outliers(plan, dataframe, working, roles, scope, applied)

    if plan.intent == "duplicates":
        return _execute_duplicates(plan, working, roles, scope, applied)

    if plan.intent == "columns":
        return _execute_columns(plan, working, scope, applied)

    if plan.intent == "crosstab":
        return _execute_crosstab(plan, working, scope, applied)

    if plan.intent == "correlation":
        return _execute_correlation(plan, working, scope, applied)

    if plan.intent == "compare" and plan.columns:
        return _execute_column_differences(plan, working, roles, scope, applied)

    if plan.intent == "compare":
        return _execute_compare(plan, working, roles, scope, applied)

    if plan.intent == "correlates":
        return _execute_correlates(plan, working, roles, scope, applied)

    if plan.intent == "rank" and plan.dimension is None and plan.grain:
        return _rank_periods(plan, working, roles, scope, applied)

    if plan.intent in ("rank", "breakdown"):
        return _execute_grouping(plan, working, roles, scope, applied)

    if plan.intent == "trend":
        return _execute_trend(plan, working, roles, scope, applied)

    if plan.intent == "growth":
        return _execute_growth(plan, working, roles, scope, applied)

    raise ValueError(f"Unsupported intent: {plan.intent}")


# Periods in a year, for a trend slope stated per year.
PERIODS_PER_YEAR = {"D": 365.25, "W": 52.1775, "M": 12.0, "Q": 4.0, "Y": 1.0}
# A first period the data reaches less of than this is partial and left out of a trend.
LEADING_COVERAGE = 0.8
# Years of complete data before a slope is fitted on yearly averages instead of on each period.
SLOPE_YEARS = 3


def _execute_trend(
    plan: QueryPlan, working: pd.DataFrame, roles: ColumnRoles, scope: str, applied: list[str]
) -> QueryAnswer:
    """Describe the measure over time: where it started and ended, its slope, and its peak and low.

    The slope is Theil-Sen, so one wild period can't tilt it, and over several
    years it is fitted on yearly averages, so seasons don't. A first period
    the data starts part-way through is left out, as the last one is when it
    is still in progress (see ``build_trend``).
    """
    assert roles.date is not None
    scoped_roles = _scoped_roles(roles, plan.measure, None)
    grain = plan.grain or preferred_frequency(working[roles.date])
    series = build_trend(working, scoped_roles, frequency=grain, count_column=plan.count_column)
    trend, leading = _without_leading_partial(series.frame, working[roles.date], grain)
    present = trend.dropna(subset=["Value"])
    if len(present) < 2:
        return QueryAnswer(
            question="",
            plan=plan,
            answer="Not enough periods in that scope to draw a trend.",
            calculation=f"trend needs at least 2 periods{scope}",
        )
    first, last = float(present.iloc[0]["Value"]), float(present.iloc[-1]["Value"])
    grain_name = GRAIN_NAMES.get(grain, "period")
    # Rates move in points and levels by a plain difference (see format_change).
    column = working[plan.measure].dropna() if plan.measure else None
    unit = _unit(plan.measure, scoped_roles)
    size = format_change(first, last, plan.measure, unit, column, signed=True)
    movement = (
        f"{size} across {len(trend)} {grain_name}s"
        if size
        else f"across {len(trend)} {grain_name}s, from a starting period of zero"
    )
    start, end = format_values_apart(first, last, column=plan.measure)
    counted = f"Distinct {plan.count_column}" if plan.count_column else "Records"
    named = plan.measure or counted
    first_period = _period_words(present.iloc[0]["Period"], grain)
    last_period = _period_words(present.iloc[-1]["Period"], grain)
    answer = (
        f"{named} per {grain_name}{_phrase(applied)} moved from {start} to {end} ({movement}), "
        f"from {first_period} to {last_period}."
    )
    slope, slope_calculation = _trend_slope(present, grain, plan.measure, column)
    answer += slope
    if len(present) >= 4:
        peak, low = present.loc[present["Value"].idxmax()], present.loc[present["Value"].idxmin()]
        answer += (
            f" It peaked at {format_number(float(peak['Value']), plan.measure, column_values=column)} in "
            f"{_period_words(peak['Period'], grain)} and was lowest, at "
            f"{format_number(float(low['Value']), plan.measure, column_values=column)}, in "
            f"{_period_words(low['Period'], grain)}."
        )
    calculation = f"{plan.aggregation}({plan.measure or 'rows'}) grouped per {grain_name}{scope}"
    if slope_calculation:
        calculation += f"; {slope_calculation}"
    if leading:
        calculation += f"; {leading}"
    return QueryAnswer(
        question="",
        plan=plan,
        answer=answer,
        calculation=_with_notes(calculation, series),
        table=trend,
        chart="line",
    )


def _period_words(period: object, grain: str) -> str:
    """A period for a sentence: "Q4 2012", "2018", "the week of 13 Jan 2025"."""
    shown = format_period(pd.Timestamp(period), grain)
    return f"the week of {shown}" if grain == "W" else shown


def _without_leading_partial(trend: pd.DataFrame, dates: pd.Series, grain: str) -> tuple[pd.DataFrame, str]:
    """Drop a first period the data starts part-way through, with a note saying so.

    A record that starts in June makes its first year a summer, so comparing
    it with full years would read the season as a change.
    """
    present = dates.dropna()
    if len(trend) < 4 or present.empty or grain not in PERIODS_PER_YEAR:
        return trend, ""
    first = pd.Period(pd.Timestamp(trend["Period"].iloc[0]), freq=grain)
    span = (first.end_time - first.start_time).total_seconds()
    reached = (first.end_time - present.min()).total_seconds()
    if span <= 0 or reached / span >= LEADING_COVERAGE:
        return trend, ""
    note = (
        f"{format_period(first.start_time, grain)} is left out because the data starts part-way through it "
        f"({present.min():%d %b %Y})"
    )
    return trend.iloc[1:].reset_index(drop=True), note


def _trend_slope(
    present: pd.DataFrame, grain: str, measure: str | None, column: pd.Series | None
) -> tuple[str, str]:
    """The Theil-Sen slope of a trend as a sentence and its calculation, per year where that makes sense."""
    periods = pd.DatetimeIndex(present["Period"])
    values = present["Value"].to_numpy(dtype=float)
    per_year = PERIODS_PER_YEAR.get(grain)
    years = (periods[-1] - periods[0]).days / 365.25
    if per_year and grain != "Y" and years >= SLOPE_YEARS:
        frame = pd.DataFrame({"year": periods.year, "value": values})
        counts = frame.groupby("year")["value"].count()
        full = counts[counts >= 0.9 * per_year].index
        yearly = frame[frame["year"].isin(full)].groupby("year")["value"].mean()
        if len(yearly) >= SLOPE_YEARS:
            slope, _ = theil_sen(yearly.index.to_numpy(dtype=float), yearly.to_numpy())
            text = _per_year_text(slope, len(yearly), measure, column)
            return (
                f" The Theil-Sen trend is {text}, fitted on the yearly averages of {len(yearly)} "
                "complete years.",
                f"Theil-Sen slope over the yearly averages of {len(yearly)} complete years",
            )
    slope, _ = theil_sen(period_positions(periods), values)
    if grain == "Y":
        text = _per_year_text(slope, len(values), measure, column)
    else:
        text = f"{_signed(slope, measure, column)} per {GRAIN_NAMES.get(grain, 'period')}"
    return f" The Theil-Sen trend is {text}.", f"Theil-Sen slope over {len(values)} periods"


def _per_year_text(slope: float, years: int, measure: str | None, column: pd.Series | None) -> str:
    text = f"{_signed(slope, measure, column)} per year"
    if years >= 20:
        text += f" ({_signed(slope * 10, measure, column)} per decade)"
    return text


def _signed(value: float, measure: str | None, column: pd.Series | None) -> str:
    """A slope with its sign, to about three significant digits: "+0.0328", "-4.97", "+1,204"."""
    size = abs(value)
    if size >= 1_000:
        shown = f"{size:,.0f}"
    elif size >= 1:
        shown = f"{size:,.3g}" if size < 100 else f"{size:,.1f}"
    elif size == 0:
        shown = "0"
    else:
        shown = f"{size:.3g}"
        if "e" in shown:
            shown = f"{size:.{-int(np.floor(np.log10(size))) + 2}f}"
    return f"{'-' if value < 0 else '+'}{shown}"


def _execute_count(
    plan: QueryPlan,
    dataframe: pd.DataFrame,
    working: pd.DataFrame,
    roles: ColumnRoles,
    scope: str,
    applied: list[str],
) -> QueryAnswer:
    """Count rows or distinct values; a count with conditions gives its share of the rows too.

    The share is of the rows where the filtered columns are recorded, so "what
    share have diabetes" leaves out the rows with no diabetes answer, and says so.
    """
    if plan.count_column:
        distinct = int(working[plan.count_column].nunique(dropna=True))
        return QueryAnswer(
            question="",
            plan=plan,
            answer=f"There are {distinct:,} distinct {plan.count_column} values{_phrase(applied)}.",
            calculation=f"count distinct {plan.count_column}{scope}",
        )
    if not (plan.filters or plan.conditions):
        return QueryAnswer(
            question="",
            plan=plan,
            answer=f"{len(working):,} rows match{_phrase(applied)}.",
            calculation=f"row count{scope}",
        )
    base, _ = _apply_filters(dataframe, replace(plan, filters=(), conditions=()), roles)
    filtered = [item.column for item in plan.filters] + [
        condition.column for condition in plan.conditions if condition.operator != "missing"
    ]
    recorded = [column for column in dict.fromkeys(filtered) if base[column].isna().any()]
    for column in recorded:
        base = base.loc[base[column].notna()]
    described = " and ".join(
        [_describe_filter(item) for item in plan.filters] + _describe_conditions(plan.conditions)
    )
    when = f" in {_when_label(plan)}" if plan.year or plan.month else ""
    of = f"of the {len(base):,} rows{when}"
    if recorded:
        of += f" with {_and_listed(recorded)} recorded"
    share = len(working) / len(base) * 100 if len(base) else 0.0
    return QueryAnswer(
        question="",
        plan=plan,
        answer=f"{len(working):,} {of} ({_share_text(share)}) have {described}.",
        calculation=(
            f"rows with {described}, divided by the {len(base):,} rows"
            f"{' with ' + _and_listed(recorded) + ' recorded' if recorded else ''}{scope}"
        ),
    )


def _execute_missing(plan: QueryPlan, working: pd.DataFrame, scope: str, applied: list[str]) -> QueryAnswer:
    """Count the missing values of one column, or of every column that has any."""
    rows = len(working)
    if plan.measure:
        missing = int(working[plan.measure].isna().sum())
        return QueryAnswer(
            question="",
            plan=plan,
            answer=(
                f"{plan.measure} is missing on {missing:,} of {rows:,} rows"
                f" ({_share_text(missing / rows * 100)}){_phrase(applied)}."
            ),
            calculation=f"missing values of {plan.measure} over {rows:,} rows{scope}",
        )
    counts = working.isna().sum()
    gaps = pd.DataFrame({
        "Column": counts.index.astype(str),
        "Missing": counts.to_numpy(dtype=int),
        "Missing %": (counts.to_numpy(dtype=float) / rows * 100).round(2),
    })
    gaps = gaps.loc[gaps["Missing"] > 0].sort_values("Missing", ascending=False, kind="stable")
    gaps = gaps.reset_index(drop=True)
    if gaps.empty:
        answer = f"No column has missing values in the {rows:,} rows{_phrase(applied)}."
    else:
        most = int(gaps.iloc[0]["Missing"])
        tied = gaps.loc[gaps["Missing"] == most, "Column"].tolist()
        cells = int(counts.sum())
        answer = (
            f"{len(gaps)} of {len(counts)} columns have missing values{_phrase(applied)}. "
            f"{_leaders(tied)} {'are each' if len(tied) > 1 else 'is'} missing on {most:,} of "
            f"{rows:,} rows ({_share_text(most / rows * 100)}), the most of any column. "
            f"In all, {cells:,} of {rows * len(counts):,} cells are empty."
        )
    return QueryAnswer(
        question="",
        plan=plan,
        answer=answer,
        calculation=f"missing values per column over {rows:,} rows{scope}",
        table=gaps,
    )


def _mode_note(values: pd.Series) -> str:
    """How often the most common value occurs, and whether others tie with it."""
    present = values.dropna()
    counts = present.value_counts()
    if counts.empty:
        return ""
    top = int(counts.iloc[0])
    tied = int((counts == top).sum())
    note = f" It occurs on {top:,} of the {len(present):,} rows"
    if tied > 1:
        note += f", tied with {tied - 1:,} other value{'s' if tied > 2 else ''}; the smallest is shown"
    return note + "."


# Columns of a distribution table, in order.
DESCRIBE_COLUMNS = ("n", "Mean", "SD", "Median", "Q1", "Q3", "Min", "Max", "Skew")


def _describe_values(values: pd.Series) -> dict[str, float]:
    present = values.dropna().astype(float)
    q1, median, q3 = present.quantile([0.25, 0.5, 0.75]) if len(present) else (np.nan,) * 3
    return {
        "n": float(len(present)), "Mean": float(present.mean()), "SD": float(present.std()),
        "Median": float(median), "Q1": float(q1), "Q3": float(q3),
        "Min": float(present.min()), "Max": float(present.max()),
        "Skew": float(present.skew()) if len(present) > 2 else np.nan,
    }


def _skew_words(skew: float) -> str:
    if not np.isfinite(skew) or abs(skew) < 0.5:
        return "roughly symmetric"
    side = "right, with a tail of high values" if skew > 0 else "left, with a tail of low values"
    return f"{'strongly' if abs(skew) >= 1 else 'moderately'} skewed to the {side}"


def _carried_by_few(mean: float, median: float) -> bool:
    """Whether a mean is more than twice a positive median, so a few large values carry it."""
    return np.isfinite(mean) and np.isfinite(median) and median > 0 and mean > 2 * median


def _execute_describe(plan: QueryPlan, working: pd.DataFrame, scope: str, applied: list[str]) -> QueryAnswer:
    """Summarise a measure's distribution: n, mean, SD, median, IQR, range and skew, overall or per group."""
    assert plan.measure is not None
    values = working[plan.measure]
    present = values.dropna()

    def show(value: float) -> str:
        return format_number(value, plan.measure, column_values=present)

    calculation = (
        f"count, mean, sample SD (n − 1), median, quartiles, min, max and skewness of {plan.measure}{scope}"
    )
    if plan.dimension:
        keys = _group_values(working, plan)
        rows = [
            {plan.dimension: name, **_describe_values(group)}
            for name, group in values.groupby(keys, observed=True)
        ]
        table = pd.DataFrame(rows).sort_values("Median", ascending=False, kind="stable")
        table = table.reset_index(drop=True)
        table["n"] = table["n"].astype(int)
        top, bottom = table.iloc[0], table.iloc[-1]
        lowest = f"{group_label(plan.dimension, bottom[plan.dimension])}, n = {int(bottom['n']):,}"
        highest = f"{group_label(plan.dimension, top[plan.dimension])}, n = {int(top['n']):,}"
        answer = (
            f"{plan.measure} by {plan.dimension}{_phrase(applied)}: medians run from "
            f"{show(bottom['Median'])} ({lowest}) to {show(top['Median'])} ({highest}). "
            "The table gives each group's n, mean, SD, quartiles, range and skew."
        )
        return QueryAnswer(
            question="", plan=plan, answer=answer, calculation=f"{calculation}, per {plan.dimension}",
            table=table.round(4),
        )
    if present.empty:
        return QueryAnswer(
            question="", plan=plan, answer=f"{plan.measure} has no values{_phrase(applied)}.",
            calculation=f"0 non-missing {plan.measure} values{scope}",
        )
    stats = _describe_values(values)
    answer = (
        f"{plan.measure}{_phrase(applied)} (n = {len(present):,}): mean {show(stats['Mean'])}, "
        f"SD {show(stats['SD'])}; median {show(stats['Median'])}, with the middle half between "
        f"{show(stats['Q1'])} and {show(stats['Q3'])} (IQR {show(stats['Q3'] - stats['Q1'])}); "
        f"range {show(stats['Min'])} to {show(stats['Max'])}. "
    )
    skew = stats["Skew"]
    answer += (
        f"It is {_skew_words(skew)} (skewness {skew:.2f})." if np.isfinite(skew) else ""
    )
    if _carried_by_few(stats["Mean"], stats["Median"]):
        answer += (
            " The mean is more than twice the median, so a few large values carry it; the median is "
            "the better typical value."
        )
    missing = int(values.isna().sum())
    if missing:
        answer += f" {missing:,} rows have no value."
    table = pd.DataFrame({"Statistic": list(stats), "Value": [round(value, 6) for value in stats.values()]})
    return QueryAnswer(question="", plan=plan, answer=answer, calculation=calculation, table=table)


#: Most flagged values an outlier answer lists in its table.
OUTLIER_ROWS = 50
#: Most columns an outlier answer over the whole table checks against the 1.5×IQR rule.
OUTLIER_COLUMNS = 300
#: Share of zeros at which a column is heavy-tailed and no outlier rule applies.
HEAVY_TAIL_ZEROS = 0.5


@dataclass(frozen=True)
class _Fences:
    """The 1.5×IQR rule applied to one column, on a log scale when it is right-skewed and positive."""

    flagged: pd.Series
    low: float
    high: float
    log: bool


def _fences(values: pd.Series) -> _Fences | None:
    """Values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR, judged as the brief judges its measure.

    None when there are too few values to judge, or when most are zeros and any
    rule would flag ordinary values.
    """
    present = values.dropna().astype(float)
    if len(present) < 12 or present.nunique() < 4 or float((present == 0).mean()) >= HEAVY_TAIL_ZEROS:
        return None
    log = prefers_log_scale(np.sort(present.to_numpy()))
    scaled = np.log10(present.where(present > 0)) if log else present
    first, third = scaled.quantile([0.25, 0.75])
    if not third > first:
        return None
    low, high = first - 1.5 * (third - first), third + 1.5 * (third - first)
    flagged = (scaled < low) | (scaled > high)
    if log:
        low, high = 10**low, 10**high
    return _Fences(flagged=flagged[flagged], low=float(low), high=float(high), log=log)


# The data checks of the last few tables asked about, so a run of questions scans each table once.
_CHECKS_CACHE: dict[int, tuple[weakref.ref, ColumnRoles, tuple[QualityIssue, ...]]] = {}


def _data_checks(dataframe: pd.DataFrame, roles: ColumnRoles) -> tuple[QualityIssue, ...]:
    """The issues ``quality.scan`` finds in the analysed table, scanned once per table and roles."""
    cached = _CHECKS_CACHE.get(id(dataframe))
    if cached and cached[0]() is dataframe and cached[1] == roles:
        return cached[2]
    issues = scan_quality(dataframe, roles)
    if len(_CHECKS_CACHE) >= 8:
        _CHECKS_CACHE.clear()
    _CHECKS_CACHE[id(dataframe)] = (weakref.ref(dataframe), roles, issues)
    return issues


def _value_issues(issues: tuple[QualityIssue, ...], column: str, index: pd.Index) -> list[QualityIssue]:
    """Data-check issues that flag cells of ``column`` among the rows in ``index``."""
    return [
        issue for issue in issues
        if issue.check in VALUE_CHECKS
        and column in issue.cells
        and len(index.intersection(issue.cells[column]))
    ]


def _row_names(working: pd.DataFrame, roles: ColumnRoles, index: pd.Index) -> list[str]:
    """How a flagged row is named: its identifier when the table has one, else its row number."""
    if roles.identifier and roles.identifier in working.columns:
        return [str(value) for value in working.loc[index, roles.identifier]]
    return [f"row {position + 1:,}" for position in working.index.get_indexer(index)]


def _execute_outliers(
    plan: QueryPlan, dataframe: pd.DataFrame, working: pd.DataFrame, roles: ColumnRoles, scope: str,
    applied: list[str],
) -> QueryAnswer:
    """List the values the 1.5×IQR rule and the data checks flag, in one column or across the readings."""
    issues = _data_checks(dataframe, roles)
    if plan.measure:
        return _column_outliers(plan, working, roles, issues, scope, applied)
    columns = [column for column in plan.columns if column in working.columns][:OUTLIER_COLUMNS]
    rows = []
    for column in columns:
        fences = _fences(working[column])
        flagged = _value_issues(issues, column, working.index)
        outside = len(fences.flagged) if fences else 0
        if outside or flagged:
            rows.append({
                "Column": column, "Outside 1.5×IQR": outside,
                "Data checks": "; ".join(issue.summary for issue in flagged),
            })
    capped = len(plan.columns) > len(columns)
    tested = f"{len(columns):,} numeric columns"
    if capped:
        tested += f" (the first {len(columns):,} of {len(plan.columns):,})"
    if not rows:
        answer = (
            f"None of the {tested} has values outside 1.5×IQR{_phrase(applied)}, and the data checks "
            "flag none."
        )
        return QueryAnswer(question="", plan=plan, answer=answer, calculation=_outlier_calculation(scope))
    table = pd.DataFrame(rows).sort_values("Outside 1.5×IQR", ascending=False, kind="stable")
    table = table.reset_index(drop=True)
    checked = table.loc[table["Data checks"] != "", "Data checks"].tolist()
    outside = [
        f"{column} ({count:,})"
        for column, count in zip(table["Column"], table["Outside 1.5×IQR"], strict=True) if count
    ]
    if checked:
        more = f", and {len(checked) - 3:,} more columns" if len(checked) > 3 else ""
        answer = (
            f"The data checks flag values in {len(checked):,} of the {tested}{_phrase(applied)}: "
            f"{'; '.join(checked[:3])}{more}."
        )
    else:
        answer = (
            "The data checks flag no impossible values, missing-value codes or extremes in the "
            f"{tested}{_phrase(applied)}."
        )
    if outside:
        answer += (
            f" By the 1.5×IQR rule, {len(outside):,} have values outside the fences: {_leaders(outside)}."
        )
    else:
        answer += " No value sits outside 1.5×IQR."
    return QueryAnswer(
        question="", plan=plan, answer=answer, calculation=_outlier_calculation(scope), table=table
    )


def _outlier_calculation(scope: str) -> str:
    return (
        "Values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR (on log10 of the values for a right-skewed "
        f"positive column), plus the data checks' impossible values, missing-value codes and extremes{scope}"
    )


def _column_outliers(
    plan: QueryPlan, working: pd.DataFrame, roles: ColumnRoles, issues: tuple[QualityIssue, ...], scope: str,
    applied: list[str],
) -> QueryAnswer:
    assert plan.measure is not None
    column = plan.measure
    values = working[column]
    present = values.dropna()
    fences = _fences(values)
    flagged = _value_issues(issues, column, working.index)
    if fences is None:
        if len(present) and float((present == 0).mean()) >= HEAVY_TAIL_ZEROS:
            why = f"most {column} values are 0, so an outlier rule would flag ordinary values"
        elif len(present) >= 12 and present.nunique() >= 4:
            why = f"at least half the {column} values are equal, so the quartiles leave no spread to judge by"
        else:
            why = f"{column} has too few distinct values to judge"
        answer = f"No values are listed as outliers: {why}."
        index = pd.Index([])
    else:
        index = fences.flagged.index
        on_log = " on a log scale, since it is right-skewed" if fences.log else ""
        bounds = f"below {show_value(fences.low)} or above {show_value(fences.high)}"
        if len(index):
            ordered = values.loc[index].sort_values(ascending=False)
            shown = [show_value(value) for value in ordered.head(8)]
            more = f" and {len(index) - 8:,} more" if len(index) > 8 else ""
            answer = (
                f"{len(index):,} of the {len(present):,} {column} values{_phrase(applied)} sit outside "
                f"1.5×IQR ({bounds}{on_log}): {', '.join(shown)}{more}."
            )
        else:
            answer = (
                f"No {column} value{_phrase(applied)} sits outside 1.5×IQR ({bounds}{on_log}), "
                f"of {len(present):,} values."
            )
    if flagged:
        answer += f" The data checks flag {'; '.join(issue.summary for issue in flagged)}."
        for issue in flagged:
            index = index.union(working.index.intersection(issue.cells[column]))
    else:
        answer += " The data checks flag no impossible values, missing-value codes or extremes in it."
    table = None
    if len(index):
        listed = working.loc[index, [column]].assign(
            Row=_row_names(working, roles, index),
            Flag=[_flag_reason(position, fences, flagged, column) for position in index],
        )
        table = listed[["Row", column, "Flag"]].sort_values(column, ascending=False).head(OUTLIER_ROWS)
        table = table.reset_index(drop=True)
    return QueryAnswer(
        question="", plan=plan, answer=answer, calculation=_outlier_calculation(scope), table=table
    )


def _flag_reason(position: object, fences: _Fences | None, flagged: list[QualityIssue], column: str) -> str:
    reasons = [issue.label for issue in flagged if position in issue.cells[column]]
    if fences is not None and position in fences.flagged.index:
        reasons.insert(0, "outside 1.5×IQR")
    return ", ".join(reasons)


def _execute_duplicates(
    plan: QueryPlan, working: pd.DataFrame, roles: ColumnRoles, scope: str, applied: list[str]
) -> QueryAnswer:
    """Count rows that repeat an earlier row exactly, or values of one column found on several rows."""
    rows = len(working)
    if plan.count_column:
        column = plan.count_column
        counts = working[column].dropna().value_counts()
        repeated = counts[counts > 1]
        answer = (
            f"{len(repeated):,} {column} values{_phrase(applied)} appear on more than one row, "
            f"{int(repeated.sum()):,} rows in all."
            if len(repeated) else f"Every {column} value{_phrase(applied)} appears on one row only."
        )
        table = repeated.head(BREAKDOWN_LIMIT).rename_axis(column).reset_index(name="Rows")
        return QueryAnswer(
            question="", plan=plan, answer=answer,
            calculation=f"{column} values counted, keeping those on two or more rows{scope}",
            table=table if len(repeated) else None,
        )
    repeats = int(working.duplicated().sum())
    if repeats:
        answer = (
            f"{repeats:,} of the {rows:,} rows{_phrase(applied)} ({_share_text(repeats / rows * 100)}) "
            "repeat an earlier row exactly."
        )
    else:
        answer = f"No row repeats another exactly{_phrase(applied)} ({rows:,} rows)."
    identifier = roles.identifier if roles.identifier in working.columns else None
    calculation = f"rows equal to an earlier row in every column{scope}"
    if identifier:
        without = int(working.drop(columns=[identifier]).duplicated().sum())
        if without != repeats:
            answer += (
                f" Leaving out {identifier}, which numbers the rows, {without:,} rows "
                f"({_share_text(without / rows * 100)}) repeat an earlier one."
            )
        calculation += f"; checked again without {identifier}"
    return QueryAnswer(question="", plan=plan, answer=answer, calculation=calculation)


def _execute_columns(plan: QueryPlan, working: pd.DataFrame, scope: str, applied: list[str]) -> QueryAnswer:
    """Rank the plan's columns by one aggregation: "which column has the highest average"."""
    columns = [column for column in plan.columns if column in working.columns]
    label = _aggregation_label(plan)
    values = pd.Series(
        {column: _aggregate_series(working[column], plan.aggregation, plan.quantile) for column in columns},
        dtype=float,
    ).dropna()
    if len(values) < 2:
        return QueryAnswer(
            question="", plan=plan, answer=f"Fewer than two columns have values{_phrase(applied)} to rank.",
            calculation=f"ranking needs two columns with values{scope}",
        )
    ordered = values.sort_values(ascending=plan.ascending, kind="stable")
    value_label = with_aggregate(label, "value").removesuffix(" value").strip() or label
    table = pd.DataFrame({
        "Column": ordered.index,
        value_label: ordered.to_numpy(),
        "n": [int(working[column].notna().sum()) for column in ordered.index],
    })
    best, last = ordered.index[0], ordered.index[-1]
    high, low = ("lowest", "highest") if plan.ascending else ("highest", "lowest")
    word = label.removesuffix(" of").lower()

    def show(column: str) -> str:
        return format_number(float(ordered[column]), column, column_values=working[column].dropna())

    answer = (
        f"{best} has the {high} {word} of the {len(ordered):,} numeric columns{_phrase(applied)}, at "
        f"{show(best)}; {last} has the {low}, at {show(last)}."
    )
    shown = table.head(plan.top_n) if plan.top_n and plan.top_n > 1 else table.head(BREAKDOWN_LIMIT)
    return QueryAnswer(
        question="", plan=plan, answer=answer,
        calculation=(
            f"{_aggregation_formula(plan, 'each column')} over {len(ordered):,} numeric columns, "
            f"{'ascending' if plan.ascending else 'descending'}; columns are compared as they are, "
            f"so this means something only when they share a unit{scope}"
        ),
        table=shown.round(6),
        chart="bar",
    )


def _leaders(names: list[str], limit: int = 3) -> str:
    """Name up to ``limit`` items, then how many more: "A, B, C and 2 more"."""
    shown = [str(name) for name in names[:limit]]
    if len(names) > limit:
        return f"{', '.join(shown)} and {len(names) - limit} more"
    return _and_listed(shown)


def _execute_grouping(
    plan: QueryPlan, working: pd.DataFrame, roles: ColumnRoles, scope: str, applied: list[str]
) -> QueryAnswer:
    """Rank or break down the measure by a grouping column, a part of the date, or quantile bins.

    Numbered groups (timepoint 0, 1, 2...) are listed in their own order, up to
    GROUPING_LIMIT of them; a ranking still sorts them by value. A breakdown
    with more than twice EXTREME_GROUPS groups shows the highest and the lowest
    and says how many there are. When every group is one row (a label on each
    row), the answer names the row and its own value. Ties at the top are named.

    An average, median or spread ranks only the groups with at least
    MIN_RANK_VALUES values, and the answer says whether the groups really
    differ (``compare.compare_groups``) and when a few large values carry a mean.
    """
    assert plan.dimension is not None
    counted = plan.aggregation == "count" or not plan.measure
    if counted:
        value_label = f"Distinct {plan.count_column}" if plan.count_column else "Rows"
    else:
        value_label = with_aggregate(_aggregation_label(plan), plan.measure)
    grouped = _grouped_frame(working, plan, value_label)
    key = grouped.columns[0]
    single_rows = not counted and not (plan.time_part or plan.bins) and 1 < len(grouped) == len(working)
    if single_rows:
        assert plan.measure is not None
        grouped = grouped.drop(columns="n").rename(columns={value_label: plan.measure})
        value_label = plan.measure
    numbered = bool(plan.time_part or plan.bins) or is_numeric_dtype(working[plan.dimension])
    ordered = numbered and plan.intent == "breakdown" and len(grouped) <= GROUPING_LIMIT
    rankable, small = grouped, grouped.iloc[0:0]
    fragile = False
    if "n" in grouped.columns and plan.aggregation != "sum":
        enough = grouped["n"] >= MIN_RANK_VALUES
        if enough.any() and not enough.all():
            rankable, small = grouped[enough], grouped[~enough]
        fragile = not enough.any()
    if plan.intent == "rank" and not plan.time_part and len(rankable) > 1:
        # Rows with no value for the grouping are counted, but they are not a group to name as the top one.
        unlabelled = unlabelled_label(working[plan.dimension].dropna().unique())
        if str(rankable[key].iloc[0]) == unlabelled:
            rankable = rankable[rankable[key].astype(str) != unlabelled]
    extremes = False
    if ordered:
        table = _in_group_order(grouped, key)
        order = f"in {plan.time_part or key} order"
    elif plan.intent == "rank":
        table = rankable.head(plan.top_n or BREAKDOWN_LIMIT).reset_index(drop=True)
        order = "ascending" if plan.ascending else "descending"
    else:
        extremes = len(grouped) > 2 * EXTREME_GROUPS
        if extremes:
            table = pd.concat([grouped.head(EXTREME_GROUPS), grouped.tail(EXTREME_GROUPS)])
        else:
            table = grouped
        table = table.reset_index(drop=True)
        order = "ascending" if plan.ascending else "descending"
    names = _part_names(plan.time_part)
    if names:
        table, grouped, rankable = (
            frame.assign(**{key: frame[key].map(names)}) for frame in (table, grouped, rankable)
        )

    best = rankable[value_label].iloc[0]
    leaders = rankable.loc[rankable[value_label] == best, key].tolist()
    tie = len(leaders) > 1
    named = list_names([
        group_label(key, value) if numbered and not (plan.bins or names) else str(value) for value in leaders
    ])
    share_note = ""
    if "Share %" in grouped.columns and not tie:
        share_note = f" ({rankable['Share %'].iloc[0]:.1f}% of the total)"
    # Format a row count as a plain number, not in the measure's unit, and pass
    # the column's values so codes in a money-named column don't print as money.
    if counted:
        value_text = format_number(float(best))
    else:
        value_text = format_number(float(best), plan.measure, column_values=working[plan.measure].dropna())
        if "n" in rankable.columns and not tie:
            share_note = share_note.replace(" (", "; ") if share_note else ""
            value_text += f" (n = {int(rankable['n'].iloc[0]):,}{share_note.removesuffix(')')})"
            share_note = ""
    high = "lowest" if plan.ascending else "highest"
    grouping = _grouping_name(plan)
    if single_rows:
        verb = "have" if tie else "has"
        answer = f"{named} {verb} the {high} {plan.measure}{_phrase(applied)}, at {value_text}."
    else:
        if counted:
            most = "fewest" if plan.ascending else "most"
            most = f"{most} rows" if not plan.count_column else f"{most} distinct {plan.count_column}"
        else:
            most = f"{high} {sentence_case(value_label)}"
        verb = "tie for" if tie else "has"
        answer = (
            f"{named} {verb} the {most} of any {grouping}{_phrase(applied)}, "
            f"at {value_text}{share_note}."
        )
    if extremes:
        first, last = ("lowest", "highest") if plan.ascending else ("highest", "lowest")
        answer += (
            f" {plan.dimension} has {len(grouped):,} groups; the table shows the "
            f"{EXTREME_GROUPS} {first} and the {EXTREME_GROUPS} {last}."
        )
    if single_rows and plan.intent == "breakdown":
        answer += f" Each {plan.dimension} is one row, so these are the rows' own values."
    if len(small):
        answer += (
            f" {len(small):,} {grouping} group{'s' if len(small) > 1 else ''} with fewer than "
            f"{MIN_RANK_VALUES} values {'are' if len(small) > 1 else 'is'} left out of the ranking."
        )
    if fragile:
        answer += (
            f" Every {grouping} group has fewer than {MIN_RANK_VALUES} values, so this ranking is fragile."
        )
    if not counted and not single_rows:
        answer += _group_difference_line(plan, working, roles, rankable)
        answer += _skew_note(plan, working, leaders)
    if plan.bins:
        where = " at its median" if len(grouped) == 2 else " at its quartiles" if len(grouped) == 4 else ""
        answer += (
            f" {plan.dimension} is continuous, so it is split into {len(grouped)} groups of about "
            f"equal size{where}."
        )
    if extremes:
        showing = f"the {EXTREME_GROUPS} highest and {EXTREME_GROUPS} lowest of {len(grouped):,}"
    else:
        cut = len(table) < len(grouped) and ordered
        showing = f"{len(table)} of {len(grouped)}" if cut else str(len(table))
    by = f"{plan.time_part} of {plan.dimension}" if plan.time_part else plan.dimension
    if plan.bins:
        by = f"{len(grouped)} equal-sized bins of {plan.dimension}"
    if counted:
        # Name what ran: row counts use .size(), which differs from
        # count(<measure>) when the measure has missing values.
        what = f"count distinct {plan.count_column}" if plan.count_column else "row count"
    else:
        what = _aggregation_formula(plan, plan.measure)
    calculation = f"{what} by {by}, {order}, showing {showing}{scope}"
    if len(small):
        calculation += f"; groups with fewer than {MIN_RANK_VALUES} values are not ranked"
    return QueryAnswer(
        question="",
        plan=plan,
        answer=answer,
        calculation=calculation,
        table=table,
        chart="line" if ordered else "bar",
    )


def _in_group_order(grouped: pd.DataFrame, key: str) -> pd.DataFrame:
    """Groups in their own order: numbers by value, bins in order, other labels as they come."""
    if isinstance(grouped[key].dtype, pd.CategoricalDtype):
        return grouped.sort_values(key, kind="stable").reset_index(drop=True)
    return grouped.sort_values(
        key, key=lambda values: pd.to_numeric(values, errors="coerce"), kind="stable"
    ).reset_index(drop=True)


def _part_names(part: TimePart | None) -> dict[int, str]:
    """Names for the numbers of a date part: weekdays, months and quarters."""
    if part == "weekday":
        return dict(enumerate(calendar.day_name))
    if part == "month":
        return {number: calendar.month_name[number] for number in range(1, 13)}
    if part == "quarter":
        return QUARTER_NAMES
    return {}


def _grouping_name(plan: QueryPlan) -> str:
    """What a group is called in an answer: "Region", "month of the year", "hour"."""
    if plan.time_part in ("month", "quarter"):
        return f"{plan.time_part} of the year"
    if plan.bins:
        return f"{plan.dimension} group"
    return plan.time_part or str(plan.dimension)


def _group_difference_line(
    plan: QueryPlan, working: pd.DataFrame, roles: ColumnRoles, rankable: pd.DataFrame
) -> str:
    """One sentence on whether the groups of a ranked average really differ, from ``compare_groups``."""
    if plan.aggregation not in ("mean", "median") or plan.measure is None:
        return ""
    if not 2 <= len(rankable) <= GROUPING_LIMIT:
        return ""
    key = "__group"
    entity = entity_column(working, roles)
    entity = entity if entity not in (None, plan.dimension, plan.measure) else None
    columns = [plan.measure, *([entity] if entity else [])]
    frame = working[columns].assign(**{key: _group_values(working, plan)})
    if plan.time_part:
        frame = frame[working[plan.dimension].notna()]
    # Only the groups that were ranked are tested, so a group of two can't decide it.
    ranked = set(rankable.iloc[:, 0].map(str))
    codes = {str(number): str(name) for number, name in _part_names(plan.time_part).items()}
    frame = frame[frame[key].map(lambda value: codes.get(str(value), str(value))).isin(ranked)]
    comparison = compare_groups(frame, plan.measure, key, entity=entity)
    if comparison is None:
        return ""
    groups = f"{plan.dimension} groups" if plan.bins else f"{_grouping_name(plan)} groups"
    evidence = f"{comparison.test}, {format_p(comparison.decisive_p)}; {effect_text(comparison)}"
    if comparison.verdict == "none":
        return f" The {groups} do not clearly differ ({evidence}), so this ranking may be chance."
    if comparison.verdict == "negligible":
        return f" The {groups} differ only by a negligible amount ({evidence})."
    return f" The {groups} do differ ({evidence})."


def _skew_note(plan: QueryPlan, working: pd.DataFrame, leaders: list) -> str:
    """Say when a few large values carry the leading group's mean, with its median."""
    if plan.aggregation != "mean" or plan.measure is None or len(leaders) != 1:
        return ""
    groups = _group_values(working, plan)
    codes = {name: number for number, name in _part_names(plan.time_part).items()}
    leader = codes.get(leaders[0], leaders[0])
    values = working.loc[(groups == leader).to_numpy(), plan.measure].dropna()
    if values.empty or not _carried_by_few(float(values.mean()), float(values.median())):
        return ""
    median = format_number(float(values.median()), plan.measure, column_values=working[plan.measure].dropna())
    return (
        f" Its mean is more than twice its median ({median}), so a few large values carry it; "
        f'ask for "median {plan.measure} by {plan.dimension}" to rank typical values.'
    )


def _execute_correlation(
    plan: QueryPlan, working: pd.DataFrame, scope: str, applied: list[str]
) -> QueryAnswer:
    """Correlate the plan's two columns over the rows where both are recorded.

    Worded like the brief's relationship card, except that the pair was chosen by
    the reader, so nothing is said about picking it from several.
    """
    assert plan.measure is not None and plan.partner is not None
    paired = working[[plan.measure, plan.partner]].dropna()
    pairs = len(paired)
    if pairs < MIN_CORRELATION_PAIRS:
        return QueryAnswer(
            question="",
            plan=plan,
            answer=(
                f"Only {pairs:,} rows have both {plan.measure} and {plan.partner}{_phrase(applied)}; "
                f"a correlation needs at least {MIN_CORRELATION_PAIRS}."
            ),
            calculation=f"{pairs:,} complete pairs, fewer than {MIN_CORRELATION_PAIRS}{scope}",
        )
    correlation = measure_correlation(paired)
    if correlation is None:
        flat = next((column for column in paired.columns if paired[column].nunique() < 2), None)
        return QueryAnswer(
            question="",
            plan=plan,
            answer=(
                f"{flat} has a single value{_phrase(applied)}, so a correlation is undefined."
                if flat
                else f"A correlation is undefined for these values{_phrase(applied)}."
            ),
            calculation=f"Pearson r undefined over {pairs:,} complete pairs{scope}",
        )
    answer = describe_correlation(correlation, _phrase(applied))
    if abs(correlation.rank_divergence) <= RANK_DIVERGENCE:
        answer += f" The rank correlation is {correlation.spearman:.2f}."
    answer += " On its own this is not evidence of cause."
    calculation = (
        f"Pearson r over {pairs:,} paired non-missing values, 95% CI by Fisher transform, "
        f"checked against Spearman{scope}"
    )
    points = paired
    if pairs > SCATTER_LIMIT:
        points = paired.sample(SCATTER_LIMIT, random_state=0).sort_index()
        calculation += f"; the chart shows {SCATTER_LIMIT:,} of the {pairs:,} pairs, sampled"
    return QueryAnswer(
        question="",
        plan=plan,
        answer=answer,
        calculation=calculation,
        table=points.reset_index(drop=True),
        chart="scatter",
    )


# Words that ask whether groups differ, or which way: "does pH differ by site",
# "is troponin higher in positive than negative". A compare plan records the
# direction asked as its aggregation: "max" for higher, "min" for lower.
DIFFERENCE_WORDS = ("differ", "differs", "different", "difference", "differences", "vary", "varies")
DIRECTION_WORDS = {
    "higher": "max", "greater": "max", "larger": "max", "bigger": "max", "more": "max",
    "lower": "min", "smaller": "min", "less": "min",
}
# Words asking what explains a measure: "what predicts yield", "does dose affect
# growth", "the effect of temperature on growth", "which marker separates CLASS".
RESEARCH_WORDS = frozenset({
    "predict", "predicts", "predicted", "predicting", "predictor", "predictors",
    "drive", "drives", "driven", "driver", "drivers", "affect", "affects", "affected", "affecting",
    "influence", "influences", "influenced", "explain", "explains", "determine", "determines",
    "determined", "effect", "effects", "impact", "impacts", "impacted", "separate", "separates",
    "distinguish", "distinguishes",
})
# Nouns a research question uses for the columns it ranks or for what it asks
# about: "which factors affect growth", "does antibiotic use affect risk".
RESEARCH_NOUNS = frozenset({
    "factor", "factors", "feature", "features", "marker", "markers", "use", "usage", "exposure",
})
# Words that relate one column to another without a date: "is there a trend in
# crabs with distance", "does B change with concentration".
RELATION_PATTERN = re.compile(
    r"\b(?:trends?|changes?|changing|vary|varies|varying|rises?|falls?|increases?|decreases?)\b"
    r"(?:\s+\S+)*?\s+(?:with|along|across)\b"
)
# Phrasings that cause the outcome to be named first: "is growth affected by temperature".
PASSIVE_CAUSE = re.compile(
    r"\b(?:affected|influenced|driven|explained|predicted|determined|impacted)\s+by\b"
)
# A split of a continuous column at its median: "between high and low soil moisture".
MEDIAN_SPLIT = re.compile(
    r"\b(?:high|higher)\s+(?:and|vs|versus|or)\s+(?:low|lower)\b"
    r"|\b(?:low|lower)\s+(?:and|vs|versus|or)\s+(?:high|higher)\b"
)
# Words of comparison and research questions that ADA knows, so a refusal says
# what they lack rather than that nothing in the table is called that.
ANALYSIS_WORDS = frozenset({
    *DIFFERENCE_WORDS, *DIRECTION_WORDS, *RESEARCH_WORDS, *RESEARCH_NOUNS, "compare", "compared",
    "comparison", "significant", "significantly", "statistically", "associated", "linked",
})
# Grammar of comparison and "what correlates with" questions, beyond QUERY_GRAMMAR.
# The nouns name the rows being compared ("in positive patients") or the columns
# being ranked ("which variables").
COMPARISON_GRAMMAR = frozenset({
    *ANALYSIS_WORDS, "than", "among", "clear", "clearly", "really", "there", "group",
    "groups", "level", "levels", "strongly", "column", "columns", "variable",
    "variables", "measure", "measures", "patients", "participants", "subjects", "people",
    "individuals", "samples", "records", "cases", "animals", "ones", "high", "low",
    "along",
})
CORRELATES_PATTERN = re.compile(r"\b(?:correlat\w*|relat\w*|associated|linked)\b")


def _comparison_plan(question: str, dataframe: pd.DataFrame, roles: ColumnRoles) -> QueryPlan | None:
    """Plan a comparison, a relationship or a "what correlates with X" ranking; None for other questions.

    "does X differ by Y", "is X different between Y" and "compare X across Y"
    compare X across the groups of Y; "is X higher in A than B" compares groups
    A and B of one column, and "is X higher with Y" the rows with a 0/1 flag Y
    against those without. "what correlates with X" and "what predicts X" rank
    every numeric column against X; "does Y affect X", "the effect of Y on X"
    and "the trend in X with Y" relate the two (see ``_relation_plan``). A
    question with words outside this grammar is left to the other plans.
    """
    q = _norm(question)
    differ = any(re.search(rf"\b{word}\b", q) for word in DIFFERENCE_WORDS)
    compare = bool(re.search(r"\bcompare\b", q))
    direction = re.search(rf"\b({'|'.join(DIRECTION_WORDS)})\b.*\bthan\b", q)
    flag_direction = None if direction else re.search(
        rf"\b({'|'.join(DIRECTION_WORDS)})\b\s+(?:with|without|among|for|in|when)\b", q
    )
    correlates = bool(CORRELATES_PATTERN.search(q)) and bool(
        re.search(r"\b(?:what|which)\b|\bcorrelates of\b|\bmost\b", q)
    )
    research = bool(set(q.split()) & RESEARCH_WORDS)
    relation = RELATION_PATTERN.search(q)
    significant = bool(re.search(r"\bsignificant(?:ly)?\b", q))
    asked = (differ, compare, direction, flag_direction, correlates, research, relation, significant)
    if not any(asked):
        return None
    # The question is read as any other is, so every word must be a column, a
    # value, a number or grammar, here including the words of comparison.
    # Numbered groups ("condition 2 than condition 1") may repeat a column's name.
    names = _column_names(list(dataframe.columns))
    column_words = frozenset(word for forms in names.values() for form in forms for word in form.split())
    reading = _read_question(
        question, dataframe, roles, grammar=QUERY_GRAMMAR | COMPARISON_GRAMMAR | column_words, numbers_ok=True
    )
    if reading.problem or reading.outliers or reading.describe or reading.duplicates:
        return None
    if reading.across:
        return _columns_comparison(q, reading, dataframe, roles, names, differ or compare or significant)
    split = MEDIAN_SPLIT.search(q)
    if not split and {"high", "low"} & set(reading.remaining.split()):
        return None
    mentioned = _mentions(q, names)
    free = list(reading.mentioned)
    numeric_columns = [column for column in roles.numeric if column in dataframe.columns]
    scope = _scope(reading)
    filters = reading.value_filters

    if (correlates or research) and not (differ or direction or flag_direction or compare):
        if research and len(free) == 2:
            first, second = free
            outcome, cause = (first, second) if PASSIVE_CAUSE.search(q) else (second, first)
            return _relation_plan(outcome, cause, dataframe, roles, scope)
        numbers = [column for column in free if column in numeric_columns]
        if len(numbers) != 1 or len(free) != 1:
            return None
        if research:
            return _predictors_plan(numbers[0], scope)
        return QueryPlan(intent="correlates", measure=numbers[0], **scope)

    if relation and not (differ or direction or compare) and len(free) == 2:
        return _relation_plan(free[0], free[1], dataframe, roles, scope)

    if flag_direction:
        way = DIRECTION_WORDS[flag_direction.group(1)]
        return _flag_comparison(reading, free, numeric_columns, way, scope)

    conditions = scope["conditions"]
    if split:
        after = [column for column in free if mentioned[column] > split.start()]
        if not after or not _is_measurement(dataframe[after[0]]):
            return None
        grouping, compared, bins = after[0], (), 2
    else:
        grouping, compared = _compared_groups(q, dataframe, roles, names, mentioned, filters)
        bins = None
    if grouping is None:
        return None
    measures = [
        column for column in mentioned
        if column in numeric_columns and column != grouping and column not in reading.consumed
    ]
    if len(measures) > 1:
        return None
    measure = measures[0] if measures else roles.measure
    if measure is None or measure == grouping:
        return None
    kept = [item for item in filters if item.column != grouping]
    year, month = scope["year"], scope["month"]
    # "compare revenue across regions" asks for each region's total when revenue
    # adds up; only words of difference ask for the test.
    testing = differ or direction or significant
    if compare and not testing and metric_for(measure, roles).additive and not compared and not bins:
        return QueryPlan(
            intent="breakdown", aggregation=metric_for(measure, roles).aggregation, measure=measure,
            dimension=grouping, filters=tuple(kept), year=year, month=month, conditions=conditions,
        )
    if compared:
        kept.insert(0, ValueFilter(column=grouping, values=compared))
    return QueryPlan(
        intent="compare",
        aggregation=DIRECTION_WORDS[direction.group(1)] if direction and len(compared) == 2 else "mean",
        measure=measure,
        dimension=grouping,
        filters=tuple(kept),
        year=year,
        month=month,
        conditions=conditions,
        bins=bins,
    )


def _columns_comparison(
    q: str,
    reading: _Reading,
    dataframe: pd.DataFrame,
    roles: ColumnRoles,
    names: dict[str, tuple[str, ...]],
    testing: bool,
) -> QueryPlan | None:
    """Plan "which genes differ between AML and ALL": every column of a set compared across one grouping."""
    if not testing:
        return None
    mentioned = _mentions(q, names)
    grouping, compared = _compared_groups(q, dataframe, roles, names, mentioned, reading.value_filters)
    if grouping is None or any(column != grouping for column in reading.mentioned):
        return None
    filters = [item for item in reading.value_filters if item.column != grouping]
    if compared:
        filters.insert(0, ValueFilter(column=grouping, values=compared))
    return QueryPlan(
        intent="compare", aggregation="mean", dimension=grouping, columns=reading.across,
        **{**_scope(reading), "filters": tuple(filters)},
    )


def _predictors_plan(measure: str, scope: dict) -> QueryPlan:
    """Plan "what predicts X": the one place that decides how a question about predictors is answered.

    For now it ranks each numeric column by how strongly it relates to X, one
    at a time, as "what correlates with X" does; a joint model of the
    predictors is called from here once one is available.
    """
    note = (
        f'Read the question as which numeric columns relate most to {measure}, each judged on its own.'
    )
    return QueryPlan(intent="correlates", measure=measure, notes=(note,), **scope)


def _is_code(column: str, dataframe: pd.DataFrame, roles: ColumnRoles) -> bool:
    """Whether a column holds category codes to compare across rather than a quantity to correlate.

    Text, 0/1 flags, the numeric codes ADA declared (``roles.factors``) and
    whole numbers under an ID or code name are codes. A declared code whose name
    carries a unit ("Location_m", "Age (day)") is a quantity set at a few
    levels, and still correlates.
    """
    series = dataframe[column]
    if not _is_measurement(series) or _is_flag(series):
        return True
    if column in roles.factors:
        return not _unit_words(column)
    words = set(_norm(normalized_name(column)).split())
    values = series.dropna()
    return bool(words & {"id", "ids", "code", "codes"}) and bool((values % 1 == 0).all()) and (
        values.nunique() < len(values)
    )


def _relation_plan(
    outcome: str, cause: str, dataframe: pd.DataFrame, roles: ColumnRoles, scope: dict
) -> QueryPlan | None:
    """Relate two columns the way their kinds allow.

    Two quantities are correlated; a quantity and a set of codes are compared
    across the codes' groups; two sets of codes are cross-tabulated. The
    answer says when a correlation question became a comparison.
    """
    outcome_code, cause_code = (_is_code(column, dataframe, roles) for column in (outcome, cause))
    single = any(dataframe[column].nunique(dropna=True) < 2 for column in (outcome, cause))
    if single or not (outcome_code or cause_code):
        return QueryPlan(intent="correlation", measure=outcome, partner=cause, **scope)
    if outcome_code and cause_code:
        for column in (outcome, cause):
            if dataframe[column].nunique(dropna=True) > GROUPING_LIMIT:
                return None
        note = (
            f"{outcome} and {cause} both hold category codes, so they are cross-tabulated rather than "
            "correlated."
        )
        return QueryPlan(
            intent="crosstab", aggregation="count", dimension=outcome, second_dimension=cause,
            notes=(note,), **scope,
        )
    measure, grouping = (cause, outcome) if outcome_code else (outcome, cause)
    many = dataframe[grouping].nunique(dropna=True) > GROUPING_LIMIT
    if many and not _can_group_numeric(dataframe[grouping]):
        return None
    note = (
        f"{grouping} holds category codes, so {measure} is compared across its groups rather than "
        "correlated with it."
    )
    return QueryPlan(
        intent="compare", aggregation="mean", measure=measure, dimension=grouping, notes=(note,), **scope
    )


def _flag_comparison(
    reading: _Reading, free: list[str], numeric_columns: list[str], direction: Aggregation, scope: dict
) -> QueryPlan | None:
    """Plan "is X higher with Y" for a 0/1 flag Y: the rows where Y is set against those where it is not."""
    flags = [
        condition for condition in reading.conditions
        if condition.operator == "=" and condition.value in (0, 1)
    ]
    measures = [column for column in free if column in numeric_columns]
    if len(flags) != 1 or len(measures) != 1:
        return None
    flag = flags[0]
    asked = ("1", "0") if flag.value == 1 else ("0", "1")
    conditions = tuple(condition for condition in scope["conditions"] if condition != flag)
    filters = (ValueFilter(column=flag.column, values=asked), *scope["filters"])
    return QueryPlan(
        intent="compare", aggregation=direction, measure=measures[0], dimension=flag.column,
        **{**scope, "filters": filters, "conditions": conditions},
    )


def _compared_groups(
    q: str,
    dataframe: pd.DataFrame,
    roles: ColumnRoles,
    names: dict[str, tuple[str, ...]],
    mentioned: dict[str, int],
    filters: tuple[ValueFilter, ...],
) -> tuple[str | None, tuple[str, ...]]:
    """The column whose groups a comparison question names, and the two groups it names, in order.

    Groups are named as numbered groups ("condition 2 than condition 1") or by
    value ("positive and negative"). A column named after "by", "across" or
    "between" compares all of its groups, and values of other columns then
    only filter the rows; so does a column with a single named value.
    """
    for column in mentioned:
        if not is_numeric_dtype(dataframe[column]):
            continue
        found = sorted(
            (match.start(), float(match.group(1)))
            for form in names.get(column, ())
            for match in re.finditer(rf"\b{re.escape(form)}s?\s+(\d+(?:\.\d+)?)\b", q)
        )
        levels = dataframe[column].dropna().unique()
        named = list(dict.fromkeys(
            value for _, number in found for value in levels if float(value) == number
        ))
        if len(named) >= 2:
            return column, tuple(str(value) for value in named[:2])
    filtered = {item.column for item in filters}
    candidates = [
        column
        for column in mentioned
        if column not in filtered
        and _follows(q, mentioned[column], r"(?:by|per|across|between|among|each|in)")
        and (
            _can_group_text(dataframe, column, roles)
            or (
                is_numeric_dtype(dataframe[column])
                and 2 <= int(dataframe[column].nunique(dropna=True)) <= GROUPING_LIMIT
            )
        )
    ]
    # "the difference in risky_decision between fsi_high groups": "in" names the reading here.
    groupings = [
        column for column in candidates if not _follows(q, mentioned[column], r"in")
    ] or candidates
    if len(groupings) == 1:
        return groupings[0], ()
    if groupings or not filters:
        return None, ()
    chosen = max(filters, key=lambda item: len(item.values))
    ordered = sorted(chosen.values, key=lambda value: q.find(_norm(value)))
    return chosen.column, tuple(ordered[:2]) if len(ordered) >= 2 else ()


def _execute_compare(
    plan: QueryPlan, working: pd.DataFrame, roles: ColumnRoles, scope: str, applied: list[str]
) -> QueryAnswer:
    """Compare the measure across the plan's groups with the brief's test and effect size."""
    if plan.measure is None or plan.dimension is None or not is_numeric_dtype(working[plan.measure]):
        raise ValueError("A comparison needs a numeric measure and a grouping column")
    frame = working
    if plan.bins:
        frame = working.assign(**{plan.dimension: _binned(working[plan.dimension], plan.bins)})
    comparison = compare_groups(frame, plan.measure, plan.dimension, entity=entity_column(working, roles))
    if comparison is None:
        return QueryAnswer(
            question="",
            plan=plan,
            answer=(
                f"Fewer than two {plan.dimension} groups have two or more {plan.measure} values"
                f"{_phrase(applied)}, so there is nothing to compare."
            ),
            calculation=f"group comparison needs two groups of two or more values{scope}",
        )
    answer = describe_comparison(comparison, working[plan.measure].dropna())
    asked = next((item.values for item in plan.filters if item.column == plan.dimension), ())
    if plan.aggregation in ("max", "min") and len(asked) == 2 and comparison.two_groups:
        # "Is X higher in A than in B" gets its yes or no before the evidence.
        named_first = _same_value(comparison.higher.value, asked[0])
        if comparison.verdict != "clear":
            answer = f"Not clearly. {answer}"
        elif named_first == (plan.aggregation == "max"):
            answer = f"Yes. {answer}"
        else:
            answer = f"No, the other way round. {answer}"
    calculation = f"{comparison_calculation(comparison)}{scope}"
    if plan.bins:
        answer += f" {plan.dimension} is continuous, so its rows are split at the median into low and high."
        calculation += f"; {plan.dimension} split into {plan.bins} groups of about equal size"
    return QueryAnswer(
        question="",
        plan=plan,
        answer=answer,
        calculation=calculation,
        table=comparison_table(comparison),
        chart="bar",
    )


def _execute_crosstab(plan: QueryPlan, working: pd.DataFrame, scope: str, applied: list[str]) -> QueryAnswer:
    """Cross-tabulate two groupings: row counts with a chi-square test, or a measure per cell with its n."""
    assert plan.dimension is not None and plan.second_dimension is not None
    rows = _group_values(working, plan).rename(_group_key(plan, working))
    second = working[plan.second_dimension]
    columns = second.astype(object).where(second.notna(), unlabelled_label(second.dropna().unique()))
    # A cross-table of a column by itself still needs two keys.
    clash = plan.second_dimension == rows.name
    columns = columns.rename(f"{plan.second_dimension} " if clash else plan.second_dimension)
    names = _part_names(plan.time_part)
    if plan.time_part:
        keep = working[plan.dimension].notna()
        rows, columns, working = rows[keep], columns[keep], working[keep]
    row_name = rows.name
    if plan.aggregation == "count":
        counts = pd.crosstab(rows, columns)
        if names:
            counts.index = [names.get(value, value) for value in counts.index]
            counts.index.name = row_name
        answer = (
            f"Rows by {_grouping_name(plan)} and {plan.second_dimension}{_phrase(applied)}: a "
            f"{counts.shape[0]} × {counts.shape[1]} table of counts."
        )
        answer += _association_line(counts)
        table = counts.reset_index()
        table.columns = [str(column) for column in table.columns]
        return QueryAnswer(
            question="", plan=plan, answer=answer,
            calculation=(
                f"row count for each {row_name} and {plan.second_dimension}; chi-square test of independence "
                f"with Cramér's V{scope}"
            ),
            table=table,
        )
    assert plan.measure is not None
    values = working[plan.measure]
    grouped = values.groupby([rows, columns], observed=True)
    cells = grouped.agg(_group_function(plan)).rename("value").reset_index()
    cells["n"] = grouped.count().to_numpy()
    cells = cells[cells["n"] > 0]
    if cells.empty:
        return QueryAnswer(
            question="", plan=plan, answer=f"No cell has {plan.measure} values{_phrase(applied)}.",
            calculation=f"0 cells with values{scope}",
        )
    label = with_aggregate(_aggregation_label(plan), plan.measure)
    present = values.dropna()

    def cell_name(row: pd.Series) -> str:
        first = names.get(row[row_name], row[row_name]) if names else row[row_name]
        return f"{group_label(row_name, first)} and {group_label(columns.name, row[columns.name])}"

    top = cells.loc[cells["value"].idxmax()]
    bottom = cells.loc[cells["value"].idxmin()]

    def cell_value(row: pd.Series) -> str:
        shown = format_number(float(row["value"]), plan.measure, column_values=present)
        return f"{shown} (n = {int(row['n']):,})"

    answer = (
        f"{sentence_case(label)} by {_grouping_name(plan)} and {plan.second_dimension}{_phrase(applied)}: "
        f"highest for {cell_name(top)}, at {cell_value(top)}; lowest for {cell_name(bottom)}, at "
        f"{cell_value(bottom)}."
    )
    if int(cells["n"].min()) < MIN_RANK_VALUES:
        answer += f" Some cells have fewer than {MIN_RANK_VALUES} values; the table gives each cell's n."
    table = cells.pivot(index=row_name, columns=columns.name, values="value")
    counts = cells.pivot(index=row_name, columns=columns.name, values="n")
    table.columns = [f"{column}" for column in table.columns]
    counts.columns = [f"n {column}" for column in counts.columns]
    table = pd.concat([table, counts], axis=1)
    if names:
        table.index = [names.get(value, value) for value in table.index]
        table.index.name = row_name
    table = table.reset_index()
    return QueryAnswer(
        question="", plan=plan, answer=answer,
        calculation=(
            f"{_aggregation_formula(plan, plan.measure)} for each {row_name} and "
            f"{plan.second_dimension}{scope}"
        ),
        table=table.round(4),
    )


def _association_line(counts: pd.DataFrame) -> str:
    """One sentence on whether two groupings are associated: a chi-square test and Cramér's V."""
    from scipy.stats import chi2_contingency

    counts = counts.loc[counts.sum(axis=1) > 0, counts.sum(axis=0) > 0]
    if min(counts.shape) < 2:
        return ""
    chi2, p_value, dof, expected = chi2_contingency(counts.to_numpy())
    total = float(counts.to_numpy().sum())
    cramer = float(np.sqrt(chi2 / (total * (min(counts.shape) - 1)))) if total else 0.0
    evidence = f"chi-square = {chi2:.1f}, df = {dof}, {format_p(p_value)}; Cramér's V = {cramer:.2f}"
    rough = float((expected < 5).mean()) > 0.2
    caution = " Many cells expect fewer than 5 rows, so the test is only a rough guide." if rough else ""
    if p_value < 0.05:
        return f" The two are associated ({evidence}).{caution}"
    return f" There is no clear association ({evidence}).{caution}"


def _execute_shares(plan: QueryPlan, dataframe: pd.DataFrame, roles: ColumnRoles) -> QueryAnswer:
    """The share of each group's rows that meet the plan's conditions: "share of Result 1 by Gender".

    Each group's share is of its rows with the conditioned columns recorded,
    and a chi-square test says whether the shares differ.
    """
    assert plan.dimension is not None
    scoped = replace(plan, filters=(), conditions=())
    try:
        working, applied = _apply_filters(dataframe, scoped, roles)
    except TimeScopeUnavailable as unavailable:
        return QueryAnswer(
            question="", plan=plan,
            answer=f"This file has no date column, so I cannot narrow the answer to {unavailable}.",
            calculation=f"no date column available to scope to {unavailable}",
        )
    scope = _scoped(applied)
    columns = [item.column for item in plan.filters] + [
        condition.column for condition in plan.conditions if condition.operator != "missing"
    ]
    recorded = working.loc[working[list(dict.fromkeys(columns))].notna().all(axis=1)] if columns else working
    meets, _ = _apply_filters(recorded, replace(plan, year=None, month=None), roles)
    groups = _group_values(recorded, plan)
    totals = groups.groupby(groups, observed=True).size()
    hits = groups.loc[meets.index].groupby(groups.loc[meets.index], observed=True).size()
    hits = hits.reindex(totals.index, fill_value=0)
    described = " and ".join(
        [_describe_filter(item) for item in plan.filters] + _describe_conditions(plan.conditions)
    )
    names = _part_names(plan.time_part)
    key = _group_key(plan, dataframe)
    table = pd.DataFrame({
        key: [names.get(value, value) for value in totals.index] if names else list(totals.index),
        "Share %": (hits / totals * 100).round(1).to_numpy(),
        "Rows meeting": hits.to_numpy(),
        "Rows": totals.to_numpy(),
    }).sort_values("Share %", ascending=plan.ascending, kind="stable").reset_index(drop=True)
    if table.empty:
        return QueryAnswer(
            question="", plan=plan, answer="No rows match that scope.", calculation=f"0 rows{scope}"
        )
    shown = [
        f"{group_label(key, row[key]) if not names else row[key]} {row['Share %']:.1f}% "
        f"({int(row['Rows meeting']):,} of {int(row['Rows']):,})"
        for _, row in table.head(4).iterrows()
    ]
    more = f", and {len(table) - 4:,} more groups" if len(table) > 4 else ""
    answer = (
        f"Share of rows with {described} within each {_grouping_name(plan)}{_phrase(applied)}: "
        f"{'; '.join(shown)}{more}."
    )
    if len(table) >= 2:
        answer += _association_line(pd.DataFrame({"yes": hits, "no": totals - hits}))
    return QueryAnswer(
        question="", plan=plan, answer=answer,
        calculation=(
            f"rows with {described}, divided by the rows of each {key} with "
            f"{_and_listed(list(dict.fromkeys(columns))) if columns else 'any value'} recorded; "
            f"chi-square test of independence{scope}"
        ),
        table=table, chart="bar",
    )


def _execute_column_differences(
    plan: QueryPlan, working: pd.DataFrame, roles: ColumnRoles, scope: str, applied: list[str]
) -> QueryAnswer:
    """Compare each column of a set across the plan's groups, with p-values Holm-adjusted over every test."""
    assert plan.dimension is not None
    columns = [column for column in plan.columns if column in working.columns and column != plan.dimension]
    entity = entity_column(working, roles)
    scan = scan_differences(working, columns, [plan.dimension], entity=entity, limit=10)
    groups = next((item.values for item in plan.filters if item.column == plan.dimension), ())
    between = f"between {_and_listed(list(groups))}" if groups else f"across {plan.dimension}"
    if scan is None or scan.largest is None:
        return QueryAnswer(
            question="", plan=plan,
            answer=f"No column has enough values in two groups{_phrase(applied)} to compare.",
            calculation=f"group comparisons need two groups of two or more values{scope}",
        )
    tested = f"{len(scan.readings):,} columns"
    if scan.available > len(scan.readings):
        tested += f" (the {len(scan.readings):,} most variable of {scan.available:,})"
    largest = scan.largest
    if scan.named:
        shown = "; ".join(
            f"{item.measure} ({lead_clause(item)}, Holm-adjusted {format_p(item.decisive_p)})"
            for item in scan.named[:5]
        )
        answer = (
            f"{scan.clear:,} of the {tested} differ clearly {between}{_phrase(applied)} after adjusting for "
            f"{scan.tested:,} tests. The largest: {shown}."
        )
    else:
        answer = (
            f"None of the {tested} differs clearly {between}{_phrase(applied)} once the p-values are "
            f"adjusted for {scan.tested:,} tests. The largest gap is in {largest.measure} "
            f"({lead_clause(largest)}, "
            f"{format_p(largest.p_value)} before adjustment, Holm-adjusted {format_p(largest.decisive_p)})."
        )
    table = pd.DataFrame({
        "Column": [item.measure for item in (scan.named or (largest,))],
        "Higher group": [item.higher.label for item in (scan.named or (largest,))],
        "Effect": [effect_text(item) for item in (scan.named or (largest,))],
        "p": [item.p_value for item in (scan.named or (largest,))],
        "Holm-adjusted p": [item.decisive_p for item in (scan.named or (largest,))],
    })
    return QueryAnswer(
        question="", plan=plan, answer=answer,
        calculation=(
            f"each column compared {between} (Welch's t-test or ANOVA, or rank tests for skewed columns), "
            f"p-values Holm-adjusted over {scan.tested:,} tests{scope}"
        ),
        table=table,
    )


def lead_clause(comparison: GroupComparison) -> str:
    """The two centres of a comparison in a few words: "ALL 2.74 vs AML 2.34"."""
    word = "median" if comparison.uses_ranks else "mean"
    first, second = format_values_apart(
        comparison.centre(comparison.higher), comparison.centre(comparison.lower), column=comparison.measure
    )
    return f"{word} {first} in {comparison.higher.label} vs {second} in {comparison.lower.label}"


def _same_value(value: object, text: str) -> bool:
    """Whether a group's value is the one a filter names; numbers compare as numbers, so 1.0 is "1"."""
    if str(value) == text:
        return True
    try:
        return float(value) == float(text)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _execute_correlates(
    plan: QueryPlan, working: pd.DataFrame, roles: ColumnRoles, scope: str, applied: list[str]
) -> QueryAnswer:
    """Rank every numeric reading by how strongly it relates to the measure, as the brief does."""
    if plan.measure is None or not is_numeric_dtype(working[plan.measure]):
        raise ValueError("Ranking correlates needs a numeric measure")
    # In table order, which tells a total from the items before it.
    readings = {*numeric_readings(working, roles), plan.measure}
    columns = [column for column in working.columns if column in readings]
    scan = scan_relationships(working, columns, plan.measure, limit=BREAKDOWN_LIMIT, others=False)
    if scan is None or not scan.correlates:
        return QueryAnswer(
            question="",
            plan=plan,
            answer=(
                f"No other numeric column shares {MIN_CORRELATION_PAIRS} or more rows with "
                f"{plan.measure}{_phrase(applied)}, so nothing can be correlated with it."
            ),
            calculation=f"correlations need {MIN_CORRELATION_PAIRS} complete pairs{scope}",
        )
    top = scan.correlates[0]
    tested = scan.measure_tests
    if top.strength < NOTABLE_CORRELATION:
        answer = (
            f"Nothing relates to {plan.measure}{_phrase(applied)}: none of the {tested} numeric columns "
            f"tested reaches |r| = {NOTABLE_CORRELATION}. The closest is {top.second} (r = "
            f"{top.pearson:.2f}, Spearman {top.spearman:.2f}, n = {top.pairs:,})."
        )
    else:
        detail = measure_correlation(working[[plan.measure, top.second]].dropna())
        assert detail is not None
        answer = (
            f"{top.second} relates most to {plan.measure} of the {tested} numeric columns tested. "
            f"{describe_correlation(detail, _phrase(applied))}"
        )
        following = [pair for pair in scan.correlates[1:3] if pair.strength >= NOTABLE_CORRELATION]
        if following:
            answer += " Next: " + ", ".join(
                f"{pair.second} (r = {pair.pearson:.2f}, Spearman {pair.spearman:.2f})" for pair in following
            ) + "."
        answer += (
            " It was picked from several columns, so treat it as a lead to check, not as evidence of cause."
        )
    intervals = [correlation_interval(pair.pearson, pair.pairs) for pair in scan.correlates]
    table = pd.DataFrame(
        {
            "Column": [pair.second for pair in scan.correlates],
            "Pearson r": [round(pair.pearson, 3) for pair in scan.correlates],
            "95% CI low": [round(low, 3) for low, _ in intervals],
            "95% CI high": [round(high, 3) for _, high in intervals],
            "Spearman r": [round(pair.spearman, 3) for pair in scan.correlates],
            "n": [pair.pairs for pair in scan.correlates],
        }
    )
    calculation = (
        f"Pearson r with a 95% CI by Fisher transform, and Spearman r, of each of {tested} numeric "
        f"columns with {plan.measure} over their complete rows; ranked by the larger of |Spearman r| "
        "and |Pearson r| without the most influential record"
    )
    left_out = [item.describe() for item in scan.derived if plan.measure in (item.column, *item.parts)]
    left_out += [flag.describe() for flag in scan.flags if plan.measure in (flag.flag, flag.column)]
    left_out += [
        f"{first} and {second}, which move together almost exactly"
        for first, second, _ in scan.near_duplicates
        if plan.measure in (first, second)
    ]
    left_out += [
        f"{first} and {second}, named alike and moving together (r = {value:.2f})"
        for first, second, value in scan.shared_stems
        if plan.measure in (first, second)
    ]
    if left_out:
        calculation += "; left out: " + "; ".join(left_out)
        answer += (
            f" Left out as built from {plan.measure} or near-copies of it: "
            f"{_set_aside_columns(plan.measure, scan)}."
        )
    return QueryAnswer(
        question="",
        plan=plan,
        answer=answer,
        calculation=f"{calculation}{scope}",
        table=table,
        chart="bar",
    )


def _set_aside_columns(measure: str, scan) -> str:
    """The columns a correlates answer left out because they repeat or are built from the measure."""
    named: dict[str, str] = {}
    for first, second, value in (*scan.near_duplicates, *scan.shared_stems):
        if measure in (first, second):
            other = second if first == measure else first
            named.setdefault(other, f"{other} (r = {value:.2f})")
    for item in scan.derived:
        if measure in (item.column, *item.parts):
            for column in (item.column, *item.parts):
                if column != measure:
                    named.setdefault(column, f"{column} (derived)")
    for flag in scan.flags:
        if measure in (flag.flag, flag.column):
            other = flag.column if flag.flag == measure else flag.flag
            named.setdefault(other, f"{other} (marks its cap)")
    return _leaders(list(named.values()), limit=5)


def _rank_periods(
    plan: QueryPlan,
    working: pd.DataFrame,
    roles: ColumnRoles,
    scope: str,
    applied: list[str],
) -> QueryAnswer:
    """Name the period at the plan's grain with the highest (or lowest) value of the trend."""
    if not roles.date or plan.grain not in GRAIN_NAMES:
        raise ValueError("Ranking periods needs a date column and a grain")
    # The override of how the measure combines belongs to the detected measure only.
    aggregation = roles.aggregation if plan.measure == roles.measure else None
    scoped_roles = replace(roles, measure=plan.measure, dimension=None, aggregation=aggregation)
    series = build_trend(working, scoped_roles, frequency=plan.grain)
    # A first period the data starts part-way through is not compared with full ones.
    frame, leading = _without_leading_partial(series.frame, working[roles.date], plan.grain)
    # A rate leaves periods without rows empty rather than zero; they can't rank.
    trend = frame.dropna(subset=["Value"])
    grain_name = GRAIN_NAMES[plan.grain]
    if len(trend) < 2:
        return QueryAnswer(
            question="",
            plan=plan,
            answer=f"Fewer than two {grain_name}s have values in that scope, so there is nothing to compare.",
            calculation=f"ranking needs at least 2 {grain_name}s{scope}",
        )
    pick = trend.loc[trend["Value"].idxmin() if plan.ascending else trend["Value"].idxmax()]
    period = format_period(pd.Timestamp(pick["Period"]), plan.grain)
    if plan.grain == "W":
        period = f"The week of {period}"
    direction = "lowest" if plan.ascending else "highest"
    label = AGGREGATION_LABELS[plan.aggregation]
    described = sentence_case(with_aggregate(label, plan.measure)) if plan.measure else "row count"
    return QueryAnswer(
        question="",
        plan=plan,
        answer=(
            f"{period} had the {direction} {described} of any {grain_name}{_phrase(applied)}, at "
            f"{format_number(float(pick['Value']), plan.measure)}."
        ),
        calculation=_with_notes(
            f"{plan.aggregation}({plan.measure or 'rows'}) per {grain_name}, the {direction} of "
            f"{len(trend)} {grain_name}s{scope}" + (f"; {leading}" if leading else ""),
            series,
        ),
        table=frame,
        chart="line",
    )


def _execute_growth(
    plan: QueryPlan,
    working: pd.DataFrame,
    roles: ColumnRoles,
    scope: str,
    applied: list[str],
) -> QueryAnswer:
    assert roles.date is not None
    measure = plan.measure
    scoped_roles = _scoped_roles(roles, measure, plan.dimension)
    series = build_trend(working, scoped_roles, frequency=plan.grain)
    trend = series.frame
    if len(trend) < 2:
        return QueryAnswer(
            question="",
            plan=plan,
            answer="Not enough history in that scope to measure change.",
            calculation=f"growth needs at least 2 periods{scope}",
        )
    previous_period, current_period = trend.iloc[-2]["Period"], trend.iloc[-1]["Period"]

    if plan.dimension:
        grain = plan.grain or preferred_frequency(working[roles.date])
        frame = working[[roles.date, plan.dimension] + ([measure] if measure else [])].dropna(
            subset=[roles.date, plan.dimension]
        )
        frame = frame.assign(__period=frame[roles.date].dt.to_period(grain).dt.to_timestamp())
        frame = frame[frame["__period"].isin([previous_period, current_period])]
        relative = metric_for(measure, scoped_roles).additive
        if not measure:
            pivot = frame.groupby([plan.dimension, "__period"]).size().unstack(fill_value=0)
        elif relative:
            pivot = frame.groupby([plan.dimension, "__period"])[measure].sum().unstack(fill_value=0.0)
        else:
            # Average rates and levels per segment; a segment absent from either
            # period has no change to rank.
            pivot = frame.groupby([plan.dimension, "__period"])[measure].mean().unstack().dropna()
        if previous_period not in pivot.columns or current_period not in pivot.columns:
            return QueryAnswer(
                question="",
                plan=plan,
                answer="The latest two periods do not overlap across segments, so growth cannot be ranked.",
                calculation=f"per-{plan.dimension} growth unavailable{scope}",
            )
        result = pd.DataFrame(
            {
                plan.dimension: pivot.index,
                "Previous": pivot[previous_period].to_numpy(dtype=float),
                "Latest": pivot[current_period].to_numpy(dtype=float),
            }
        )
        if not relative:
            return _rank_level_changes(plan, result, working, scoped_roles, scope, applied)
        # Growth from zero is undefined, so those segments leave the ranking but
        # are still named in the answer.
        from_nothing = result[(result["Previous"] == 0) & (result["Latest"] != 0)]
        result = result[result["Previous"] != 0]
        if result.empty and not from_nothing.empty:
            newest = from_nothing.loc[from_nothing["Latest"].abs().idxmax()]
            return QueryAnswer(
                question="",
                plan=plan,
                answer=(
                    f"No {plan.dimension} has a growth rate this period, because every one "
                    f"of them started from zero. The largest new arrival is "
                    f"{group_label(plan.dimension, newest[plan.dimension])} at "
                    f"{format_number(float(newest['Latest']), measure)}."
                ),
                calculation=(
                    f"per-{plan.dimension} growth undefined from a zero base; "
                    f"{len(from_nothing)} segment(s) started at zero{scope}"
                ),
            )
        if result.empty:
            return QueryAnswer(
                question="",
                plan=plan,
                answer="Every segment starts from zero in the prior period, so growth rates are undefined.",
                calculation=f"per-{plan.dimension} growth undefined{scope}",
            )
        change = (result["Latest"] - result["Previous"]) / result["Previous"].abs() * 100
        result["Change %"] = change.round(1)
        result = result.sort_values("Change %", ascending=plan.ascending).reset_index(drop=True)
        if plan.top_n:
            result = result.head(plan.top_n)
        leader = result.iloc[0]
        start, end = format_values_apart(float(leader["Previous"]), float(leader["Latest"]), column=measure)
        answer = (
            f"{group_label(plan.dimension, leader[plan.dimension])} "
            f"{_ranked_move(leader['Change %'], plan.ascending)}{_phrase(applied)}: "
            f"{leader['Change %']:+.1f}% ({start} → {end}) in the latest period."
        )
        if not from_nothing.empty:
            newest = from_nothing.loc[from_nothing["Latest"].abs().idxmax()]
            names = ", ".join(group_label(plan.dimension, name) for name in from_nothing[plan.dimension])
            answer += (
                f" {len(from_nothing)} segment(s) are left out of the ranking because they "
                f"started from zero ({names}); the largest is "
                f"{group_label(plan.dimension, newest[plan.dimension])} at "
                f"{format_number(float(newest['Latest']), measure)}."
            )
        return QueryAnswer(
            question="",
            plan=plan,
            answer=answer,
            calculation=(
                f"per-{plan.dimension} {measure or 'row count'}: latest vs previous period, "
                f"ranked by % change{scope}"
            ),
            table=result.head(BREAKDOWN_LIMIT),
            chart="bar",
        )

    previous, current = float(trend.iloc[-2]["Value"]), float(trend.iloc[-1]["Value"])
    unit = _unit(measure, scoped_roles)
    column = working[measure].dropna() if measure else None
    size = format_change(previous, current, measure, unit, column)
    if size is None:
        return QueryAnswer(
            question="",
            plan=plan,
            answer="The previous period is zero or missing, so the change is undefined.",
            calculation=f"change undefined for a zero or missing base{scope}",
        )
    direction = "up" if current >= previous else "down"
    start, end = format_values_apart(previous, current, column=measure)
    answer = (
        f"{measure or 'Records'}{_phrase(applied)} is {direction} {size} versus the prior "
        f"period ({start} → {end})."
    )
    if unit in ("rate", "level"):
        formula = f"latest − previous period average{scope}"
    else:
        formula = f"(latest − previous) ÷ |previous| on period {'totals' if measure else 'counts'}{scope}"
    return QueryAnswer(
        question="",
        plan=plan,
        answer=answer,
        calculation=_with_notes(formula, series),
        table=trend,
        chart="line",
    )


def _ranked_move(change: float, ascending: bool) -> str:
    """Describe the top of a change ranking: "rose most", "fell least" and so on."""
    if ascending:
        return "fell most" if change < 0 else "rose least"
    return "rose most" if change >= 0 else "fell least"


def _rank_level_changes(
    plan: QueryPlan,
    result: pd.DataFrame,
    working: pd.DataFrame,
    roles: ColumnRoles,
    scope: str,
    applied: list[str],
) -> QueryAnswer:
    """Rank segments of a rate or level measure by the change in their average."""
    measure = plan.measure
    unit = _unit(measure, roles)
    column = working[measure].dropna()
    result["Change"] = result["Latest"] - result["Previous"]
    result = result.sort_values("Change", ascending=plan.ascending).reset_index(drop=True)
    if plan.top_n:
        result = result.head(plan.top_n)
    if result.empty:
        return QueryAnswer(
            question="",
            plan=plan,
            answer=f"No {plan.dimension} has values in both of the latest two periods.",
            calculation=f"per-{plan.dimension} change unavailable{scope}",
        )
    leader = result.iloc[0]
    previous, latest = float(leader["Previous"]), float(leader["Latest"])
    size = format_change(previous, latest, measure, unit, column, signed=True)
    start, end = format_values_apart(previous, latest, column=measure)
    answer = (
        f"{group_label(plan.dimension, leader[plan.dimension])} "
        f"{_ranked_move(latest - previous, plan.ascending)}{_phrase(applied)}: "
        f"{size} ({start} → {end}) in the latest period."
    )
    unit_label = "percentage points" if unit == "rate" else "difference"
    return QueryAnswer(
        question="",
        plan=plan,
        answer=answer,
        calculation=(
            f"per-{plan.dimension} average {measure}: latest − previous period, "
            f"ranked by {unit_label}{scope}"
        ),
        table=result.head(BREAKDOWN_LIMIT),
        chart="bar",
    )


def _with_notes(calculation: str, series: TrendSeries) -> str:
    """Append the trend's timeline notes (e.g. a dropped in-progress period) to the trace."""
    return "; ".join([calculation, *series.notes]) if series.notes else calculation


def _phrase(applied_filters: list[str]) -> str:
    return f" for {'; '.join(applied_filters)}" if applied_filters else ""


def answer_question(
    question: str,
    dataframe: pd.DataFrame,
    roles: ColumnRoles,
    *,
    rows_not_analyzed: int = 0,
    whole: WholeTable | None = None,
) -> QueryAnswer | None:
    """Parse and execute in one step; None means the rules could not read it.

    ``rows_not_analyzed`` is how many rows the row limit left out of
    ``dataframe``; the answer says so when it is not zero. ``whole`` holds a
    long table's every parameter when ``dataframe`` is the brief's focus on
    one: a question naming another parameter, or grouping by them, is answered
    from it, and every other answer says it covers the focus only.
    """
    if whole is not None and (wider := _whole_table_answer(question, whole, rows_not_analyzed)):
        return wider
    plan = parse_question(question, dataframe, roles)
    if plan is None:
        return None
    try:
        result = execute_plan(plan, dataframe, roles, rows_not_analyzed=rows_not_analyzed)
    except ValueError:
        return None
    answer = result.answer
    if whole is not None:
        answer += (
            f" This covers the {len(dataframe):,} rows with {whole.parameter} = {whole.focus}, which the "
            f"brief follows; name another {whole.parameter} to ask about it."
        )
    return QueryAnswer(
        question=question,
        plan=result.plan,
        answer=answer,
        calculation=result.calculation,
        table=result.table,
        chart=result.chart,
    )


def _scope_words(plan: QueryPlan) -> list[str]:
    """The plan's filters and conditions in words, as an answer's scope."""
    return [_describe_filter(item) for item in plan.filters] + _describe_conditions(plan.conditions)


def _whole_table_answer(question: str, whole: WholeTable, rows_not_analyzed: int) -> QueryAnswer | None:
    """Answer from every parameter of a long table when the question picks another one, or groups by them."""
    frame, roles = whole.frame, replace(whole.roles, focus=None)
    plan = parse_question(question, frame, roles)
    if plan is None:
        return None
    picked = next((item.values for item in plan.filters if item.column == whole.parameter), ())
    grouped = whole.parameter in (plan.dimension, plan.second_dimension)
    if not grouped and (not picked or tuple(picked) == (whole.focus,)):
        return None
    try:
        result = execute_plan(plan, frame, roles, rows_not_analyzed=rows_not_analyzed)
    except ValueError:
        return None
    answer = result.answer
    if grouped and plan.intent in ("rank", "breakdown") and result.table is not None:
        # Values of different parameters are in different units, so none of them "leads".
        groups = plan.dimension if plan.dimension == whole.parameter else plan.second_dimension
        label = "row count" if plan.aggregation == "count" else with_aggregate(
            _aggregation_label(plan), str(plan.measure)
        )
        answer = (
            f"The table gives the {label} for each {groups}{_phrase(_scope_words(plan))}. Each "
            f"{whole.parameter} is measured on its own scale, often in its own unit, so they are listed, "
            "not ranked against each other."
        )
    elif grouped:
        answer += (
            f" Each {whole.parameter} is measured on its own scale, often in its own unit, so compare each "
            "only with itself, not with the others."
        )
    return QueryAnswer(
        question=question, plan=result.plan, answer=answer, calculation=result.calculation,
        table=result.table, chart=result.chart,
    )


def _listed(columns: tuple[str, ...] | list[str]) -> str:
    """Join names as "a", "a or b", "a, b or c"."""
    return columns[0] if len(columns) == 1 else f"{', '.join(columns[:-1])} or {columns[-1]}"


def _listed_capped(columns: list[str], limit: int = 5) -> str:
    """Join up to ``limit`` names as alternatives, then say how many more: "a, b, c, d, e or 995 others"."""
    if len(columns) <= limit:
        return _listed(columns)
    return f"{', '.join(columns[:limit])} or {len(columns) - limit:,} others"


def _and_listed(columns: tuple[str, ...] | list[str]) -> str:
    """Join names as "a", "a and b", "a, b and c"."""
    return columns[0] if len(columns) == 1 else f"{', '.join(columns[:-1])} and {columns[-1]}"


def rephrase_hint(
    question: str,
    dataframe: pd.DataFrame,
    roles: ColumnRoles,
    *,
    rows_not_analyzed: int = 0,
    whole: WholeTable | None = None,
) -> str | None:
    """Say why a question was refused, or which column it may have meant; None without a lead.

    The reason comes from reading the question: a word that names nothing in
    the table, a part of a name several columns share ("Did you mean
    bill_length_mm or flipper_length_mm?"), a value of several columns, or a
    column, value, grouping or condition the plan would leave out. Otherwise
    it names a text column the question asks to correlate, or a numeric
    column it groups by that has too many values.
    """
    q = _norm(question)
    if not q:
        return None
    if whole is not None:
        dataframe, roles = whole.frame, replace(whole.roles, focus=None)
    reason = _plan_question(question, dataframe, roles, rows_not_analyzed)[1]
    if reason:
        return reason
    numeric_columns = [column for column in roles.numeric if column in dataframe.columns]
    names = _column_names(list(dataframe.columns))
    mentioned = _mentions(q, names)
    numbers = [column for column in mentioned if column in numeric_columns]
    unnamed = _without_columns(q, tuple(mentioned), names)
    if len(numbers) < 2 and any(re.search(rf"\b{word}\b", unnamed) for word in CORRELATION_WORDS):
        others = [column for column in mentioned if column not in numeric_columns]
        if others:
            return f"A correlation needs two numeric columns, and {_listed(others)} is not numeric."
    # The same column _choose_columns would group by: a ranked one first, then one after "by".
    grouping = next(
        (
            column
            for words in (RANKING_WORDS, GROUPING_WORDS)
            for column in numbers
            if _follows(q, mentioned[column], words)
        ),
        None,
    )
    if grouping is None or len(numbers) < 2 or _can_group_numeric(dataframe[grouping]):
        return None
    distinct = int(dataframe[grouping].nunique(dropna=True))
    return (
        f"{grouping} has {distinct:,} distinct values, too many to group by "
        f"(the limit is {GROUPING_LIMIT})."
    )


def matching_values(dataframe: pd.DataFrame, column: str, phrase: str) -> tuple[str, ...]:
    """The values of ``column`` a phrase names, matched the way questions are.

    The value itself (any case), or the one value that holds the phrase as
    whole words: "Tartrazine" names "Tartrazine, FD&C Yellow 5 (US), E102 (EU)".
    A phrase several values hold names none. A numeric column matches a number.
    """
    if column not in dataframe.columns:
        return ()
    series = dataframe[column]
    if is_numeric_dtype(series) and not is_bool_dtype(series):
        try:
            number = float(phrase)
        except ValueError:
            return ()
        return (_number_text(number),) if bool((series == number).any()) else ()
    run = tuple(_norm(phrase).split())
    if not run:
        return ()
    blank_roles = ColumnRoles(
        date=None, measure=None, dimension=None, identifier=None, numeric=(), dimensions=()
    )
    vocabulary = _vocabulary(dataframe[[column]], blank_roles)
    found, exact = _values_named(run, vocabulary, exact_only=False, column=column)
    chosen, values, _ = _pick_values(run, found, exact, phrase)
    return tuple(value.raw for value in values) if chosen is not None else ()


def suggested_questions(dataframe: pd.DataFrame, roles: ColumnRoles) -> list[str]:
    """Return up to four starter questions the rule-based parser can answer."""
    suggestions: list[str] = []
    if roles.measure:
        # "Average Order Value" already says how it combines; don't prefix it again.
        first_word = (roles.measure.split() or [""])[0].lower()
        if first_word in {"average", "avg", "mean", "median", "total"}:
            suggestions.append(roles.measure)
        else:
            suggestions.append(f"{metric_for(roles.measure, roles).combines_as.capitalize()} {roles.measure}")
    if roles.measure and roles.dimension:
        suggestions.append(f"Top 5 {roles.dimension} by {roles.measure}")
    if roles.measure and roles.date:
        suggestions.append(f"Monthly {roles.measure} trend")
    if roles.dimension and roles.date:
        suggestions.append(f"Which {roles.dimension} grew fastest?")
    if not suggestions:
        suggestions.append("How many rows are there?")
    return suggestions[:4]
