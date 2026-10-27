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
import os
import re
import unicodedata
import weakref
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from typing import Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from .aggregates import aggregate_rows
from .aggregation import TrendSeries, build_trend, preferred_frequency, unlabelled_label
from .business_insights import (
    MIN_CORRELATION_PAIRS,
    RANK_DIVERGENCE,
    describe_correlation,
    measure_correlation,
    numeric_readings,
    set_aside_note,
    tested_rows,
    within_group_note,
)
from .compare import (
    ADJUSTED,
    ADJUSTED_BY,
    MIN_NAMED_GROUP,
    MIN_SLOPE_UNITS,
    NOTABLE_CORRELATION,
    SCAN_ROWS,
    GroupComparison,
    RateComparison,
    compare_groups,
    compare_rates,
    comparison_calculation,
    comparison_table,
    correlation_interval,
    describe_comparison,
    effect_text,
    entity_column,
    format_p,
    grouping_rules,
    outcome_event,
    rate_calculation,
    repeated_correlation,
    repeated_unit,
    scan_differences,
    scan_relationships,
)
from .decisions import Decision, decide
from .formatting import (
    counted,
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
from .numeric import to_numbers
from .quality import MASKED_CHECKS, VALUE_CHECKS, QualityIssue, prefers_log_scale, show_value
from .quality import scan as scan_quality
from .schema import ColumnRoles, detect_roles, is_identifier_name, row_summary_columns, running_total_columns
from .sources import Source
from .timepoints import is_timepoint_name
from .timeseries import TREND_CONFIDENCE, period_positions, theil_sen, trend_test

Intent = Literal[
    "aggregate", "count", "rank", "breakdown", "trend", "growth", "correlation", "compare", "correlates",
    "describe", "columns", "crosstab",
]
#: Plans answered from the column profile or the data checks rather than by combining a measure.
ProfileIntent = Literal["missing", "duplicates", "outliers"]
Aggregation = Literal[
    "sum", "mean", "median", "min", "max", "count", "std", "var", "iqr", "quantile", "mode", "share", "range",
    "ratio",
]
Operator = Literal["=", "<", "<=", ">", ">=", "missing", "contains", "starts", "ends"]
TimePart = Literal["hour", "weekday", "month", "quarter", "year", "season"]

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
    "strongest": "max",
    "deepest": "max",
    "peak": "max",
    "min": "min",
    "minimum": "min",
    "lowest": "min",
    "smallest": "min",
    "shortest": "min",
    "weakest": "min",
    "shallowest": "min",
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
    "range": "range",
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
    "ratio": "Ratio of",
    "share": "Share of rows",
    "range": "Range of",
}
# Aggregations that describe how a measure spreads; they don't add up across groups.
SPREAD_AGGREGATIONS = ("std", "var", "iqr", "quantile", "range")

# Words asking for the bottom of a ranking. Each is also a superlative, so
# SUPERLATIVE_WORDS includes this tuple rather than repeating it.
ASCENDING_WORDS = (
    "worst", "bottom", "lowest", "smallest", "least", "fewest", "shortest", "quietest", "weakest",
    "shallowest",
)
# Superlatives of temperature, read as the highest or lowest value of the
# measure, and said so in the answer.
WARM_WORDS = ("warmest", "hottest")
COLD_WORDS = ("coldest", "coolest")
ASCENDING_WORDS = (*ASCENDING_WORDS, *COLD_WORDS)
SUPERLATIVE_WORDS = (
    # "most" ranks from the top, like "highest"; "least" is in ASCENDING_WORDS.
    "best", "top", "leading", "highest", "largest", "biggest", "most", "greatest", "longest",
    "tallest", "heaviest", "busiest", "strongest", "deepest", *WARM_WORDS,
    *ASCENDING_WORDS,
)
DECLINE_WORDS = ("slowest", "declined", "decreased", "dropped", "fell", "shrank")
ASCENDING_PATTERN = rf"\b({'|'.join((*ASCENDING_WORDS, *DECLINE_WORDS))})\b"
GROWTH_WORDS = (
    "grew", "grow", "growing", "growth", "changed", "change",
    "increase", "increased", "decrease", "decreased", "rose", "risen", "climbed",
    "decline", "declined", "dropped", "drop", "shrank", "fell", "fallen",
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
#: Words in a column's name that mark first names (nameFirst, forename) and last names (nameLast, surname).
FIRST_NAME_WORDS = frozenset({"first", "forename", "firstname"})
LAST_NAME_WORDS = frozenset({"last", "surname", "lastname", "family"})
#: A name column also needs one of these words, so "first_flight" is not a name.
NAME_WORDS = frozenset({"name", "forename", "surname", "firstname", "lastname"})
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
    "a", "about", "across", "all", "also", "an", "and", "any", "are", "as", "at", "be", "been",
    "between", "by", "can", "data", "did", "do", "does", "done", "each",
    "every", "for", "from", "get", "give", "got", "have", "had", "has", "having", "he", "how",
    "her", "his", "hows", "i", "if", "in", "is", "it", "its", "just", "last", "least", "make", "made",
    "many", "me", "measured", "most", "much", "my", "next", "of", "on", "only", "or", "our", "overall",
    "please", "re", "recorded", "s", "she", "show", "tell", "tested", "than", "that", "the", "their",
    "named", "called", "happened", "happen", "happens", "occurred", "occur", "occurs",
    "then", "there", "these", "this", "those", "to", "they", "us", "value", "values", "ve", "was", "we",
    "were", "what", "whats", "when", "where", "which", "who", "whom", "whose", "with", "without", "you",
    "your",
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
    (">", r"(?:greater|more|higher|larger|bigger) than|above|over|exceeds|exceeding|exceeded|exceed|topped|"
          r"surpassed"),
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
# Words that point at rows rather than name them; on their own they are a value only typed as the data
# spell it.
DETERMINERS = frozenset({
    "the", "a", "an", "other", "others", "another", "any", "all", "some", "this", "that", "these", "those",
    "same", "such", "each", "every",
})
VALUE_CUES = frozenset({
    "for", "in", "is", "at", "where", "when", "equals", "of", "from", "within", "among", "named", "called",
})
# Cue words that are grammar only when a filter or condition follows them.
FILTER_CUES = frozenset({"within", "among", "amongst"})
# Words before a 0/1 flag column that ask for the rows where it is set:
# "how many have hypertension". "without" asks for the rows where it is not.
# Words before a yes/no column that keep its yes rows; a verb of seeing says how it was noticed
# ("squirrels seen foraging").
FLAG_CUES = (
    r"have|has|having|had|with|without|are|were|"
    r"received|receive|receiving|given|got|took|taking|using|used|seen|spotted|observed|heard|"
    r"featured|featuring|features|feature|showed|shows|showing|show|included|includes|including|"
    r"depicted|depicting|appeared\s+in|appears\s+in"
)
# Grouping by a part of the date: "average PM2.5 by hour", "revenue by weekday",
# "which month of the year is wettest", which pools each month across the years.
TIME_PART_PATTERN = re.compile(
    r"\b(?:by|per|each|across|which|what)\s+(?:the\s+)?"
    r"(?:(?P<hour>hours?(?:\s+of\s+(?:the\s+)?day)?)|(?P<weekday>weekdays?|days?\s+of\s+(?:the\s+)?week)"
    r"|(?P<month>(?:calendar\s+)?months?\s+of\s+(?:the\s+)?year|calendar\s+months?)"
    r"|(?P<quarter>quarters?\s+of\s+(?:the\s+)?year)|(?P<season>seasons?))\b"
    # Spelled out in full, a part of the date may open the question: "month of year with the highest max".
    r"|^(?:(?P<hour_>hours?\s+of\s+(?:the\s+)?day)|(?P<weekday_>days?\s+of\s+(?:the\s+)?week)"
    r"|(?P<month_>months?\s+of\s+(?:the\s+)?year)|(?P<quarter_>quarters?\s+of\s+(?:the\s+)?year))\b"
)
# Names of the date parts a grouping can show, for answers and tables.
QUARTER_NAMES = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}
# Seasons as three-month spans of the calendar, named by their months so they read
# the same on either side of the equator.
SEASON_NAMES = {0: "Dec–Feb", 1: "Mar–May", 2: "Jun–Aug", 3: "Sep–Nov"}


@dataclass(frozen=True)
class ValueFilter:
    column: str
    values: tuple[str, ...]


@dataclass(frozen=True)
class Condition:
    """Rows whose ``column`` compares to ``value``: "pH below 5.5", "OBJECT is 0".

    ``"missing"`` keeps the rows where the column has no value and needs no
    ``value``; ``"contains"``, ``"starts"`` and ``"ends"`` match ``text`` within
    a text column, ignoring case: "sampleID containing Ivc".
    """

    column: str
    operator: Operator
    value: float | None = None
    #: "year" when ``value`` is a year compared with a date column: "since 1978"; "date" when it is a
    #: day, as days since 1970: "on April 27, 2011".
    part: str | None = None
    text: str | None = None


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
    #: For a share ("share of Result 1 by Gender"), what a row must meet to be counted;
    #: ``filters``, ``conditions`` and the time scope choose the rows the shares are of.
    counted: tuple[ValueFilter | Condition, ...] = ()
    #: Splits ``dimension`` into the rows before this value (a year, for a date) and those from it on.
    split: float | None = None
    #: Readings a comparison holds fixed: "does VB differ by feed after accounting for time".
    adjusted_for: tuple[str, ...] = ()
    #: The unit each value is first combined within before the groups are compared:
    #: "average quantity per transect by protection_status".
    per: tuple[str, ...] = ()
    #: Rows left out before the answer, by value: the World and region rows a decision leaves out.
    excluded: tuple[ValueFilter, ...] = ()
    #: What a ratio is given per: 100 for "claims per 100 holders".
    scale: float | None = None
    #: The column of counts a count adds up: encounter_count for "how many encounters".
    weight: str | None = None


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
class AnswerScope:
    """The rows behind an answer: how many were read, which the question kept, and what was left out.

    ``rows_used`` counts the rows the answer rests on (with the measure, and any
    second column, recorded). Each ``left_out`` entry is a reason and its count,
    such as ("rows with body_mass_g missing", 2). ``readings`` say how the
    question was read and ``cautions`` what may sway the answer.
    """

    rows_read: int
    rows_not_analyzed: int = 0
    rows_in_scope: int | None = None
    rows_used: int | None = None
    filters: tuple[str, ...] = ()
    left_out: tuple[tuple[str, int], ...] = ()
    readings: tuple[str, ...] = ()
    cautions: tuple[str, ...] = ()


@dataclass(frozen=True)
class QueryAnswer:
    question: str
    plan: QueryPlan
    answer: str
    calculation: str
    table: pd.DataFrame | None = None
    chart: Literal["bar", "line", "scatter"] | None = None
    #: The number the answer gives, at full precision, when it gives a single one.
    value: float | None = None
    #: The number for each group of a grouped answer, every group and not only those shown.
    values: tuple[tuple[str, float], ...] = ()
    #: Other numbers the answer states, by name: ("share_percent", 44.2), ("p_value", 0.003), ("n", 342).
    figures: tuple[tuple[str, float], ...] = ()
    #: The rows behind the answer, filled in by ``execute_plan``.
    scope: AnswerScope | None = None
    #: The file the answer was computed from, filled in by ``Analysis.ask``.
    source: Source | None = None
    #: The choices the answer rests on, each with its options and default (see ``ada.decisions``).
    decisions: tuple[Decision, ...] = ()

    def to_dict(self) -> dict:
        """The answer as JSON-safe data: the result, how the question was read, the calculation and its rows.

        ``result`` holds the full-precision numbers (``value`` for a single one,
        ``values`` per group, ``figures`` for the rest), ``reading`` the plan
        that ran, and ``evidence`` the source, the rows read, kept and used, and
        each reason rows were left out. ``assumptions`` say how the question was
        read; ``warnings`` say what may sway the answer. ``decisions`` list the
        choices it rests on, with the options a caller may pick instead.
        """
        scope = self.scope
        return {
            "schema_version": ANSWER_SCHEMA_VERSION,
            "status": "answered",
            "question": self.question,
            "answer": self.answer,
            "result": {
                "value": self.value,
                "values": [{"group": label, "value": value} for label, value in self.values],
                "figures": {name: value for name, value in self.figures if _as_number(value) is not None},
            },
            "reading": plan_reading(self.plan),
            "calculation": self.calculation,
            "evidence": {
                "source": None if self.source is None else self.source.to_dict(),
                "rows_read": None if scope is None else scope.rows_read,
                "rows_not_analyzed": 0 if scope is None else scope.rows_not_analyzed,
                "rows_in_scope": None if scope is None else scope.rows_in_scope,
                "rows_used": None if scope is None else scope.rows_used,
                "filters": [] if scope is None else list(scope.filters),
                "left_out": [] if scope is None else [
                    {"reason": reason, "count": count} for reason, count in scope.left_out
                ],
            },
            "assumptions": [] if scope is None else list(scope.readings),
            "warnings": [] if scope is None else list(scope.cautions),
            "decisions": [decision.to_dict() for decision in self.decisions],
        }


#: Version of the ``QueryAnswer.to_dict`` layout. Bumped when a key changes meaning.
ANSWER_SCHEMA_VERSION = 1


def plan_reading(plan: QueryPlan) -> dict:
    """What a plan computes, by name: the operation, the columns and the scope it was read with."""
    return {
        "intent": plan.intent,
        "operation": plan.aggregation,
        "measure": plan.measure,
        "second_column": plan.partner,
        "group_by": plan.dimension,
        "second_group_by": plan.second_dimension,
        "distinct_of": plan.count_column,
        "per": list(plan.per),
        "scale": plan.scale,
        "weight": plan.weight,
        "adjusted_for": list(plan.adjusted_for),
        "year": plan.year,
        "month": plan.month,
        "quantile": plan.quantile,
        "left_out_values": [{"column": item.column, "values": list(item.values)} for item in plan.excluded],
        "source": plan.source,
    }


def _norm(text: str) -> str:
    """Lowercase words without accents or punctuation; "orgánica" reads as "organica".

    A decimal point between digits stays, so "5.5" and "PM2.5" are one word each.
    """
    decomposed = unicodedata.normalize("NFKD", str(text))
    lowered = "".join(char for char in decomposed if not unicodedata.combining(char)).lower()
    lowered = re.sub(r"\b([a-z]+)['’](s|re|ve|ll|d|m|t)\b", r"\1", lowered)
    cleaned = re.sub(r"(?:(?!(?<=\d)\.(?=\d))[^0-9a-z])+", " ", lowered)
    return " ".join(cleaned.split())


# Adjectives that say which quantity a bound is on: "less than 1 km deep" is a depth.
DIMENSION_ADJECTIVES = {
    "deep": "depth", "long": "length", "tall": "height", "high": "height", "wide": "width",
    "heavy": "weight", "old": "age", "thick": "thickness",
}
# Their comparatives, with the way each compares: "deeper than 50 cm" is a depth more than 50 cm.
COMPARATIVE_ADJECTIVES = {
    "deeper": ("depth", "more"), "shallower": ("depth", "less"), "longer": ("length", "more"),
    "shorter": ("length", "less"), "taller": ("height", "more"), "wider": ("width", "more"),
    "narrower": ("width", "less"), "heavier": ("weight", "more"), "lighter": ("weight", "less"),
    "older": ("age", "more"), "younger": ("age", "less"), "thicker": ("thickness", "more"),
    "thinner": ("thickness", "less"),
}
_BOUND_NUMBER = r"(?P<number>\d[\d,]*(?:\.\d+)?)"
POSTPOSED_ADJECTIVE = re.compile(
    rf"\b(?P<bound>{'|'.join(words for _, words in COMPARISONS)}|at least|at most)\s+{_BOUND_NUMBER}"
    rf"(?P<unit>\s+[A-Za-z]+)?\s+(?P<adjective>{'|'.join(DIMENSION_ADJECTIVES)})\b",
    re.IGNORECASE,
)
COMPARATIVE_BOUND = re.compile(
    rf"\b(?P<adjective>{'|'.join(COMPARATIVE_ADJECTIVES)})\s+than\s+{_BOUND_NUMBER}", re.IGNORECASE
)


def _question_text(question: str) -> str:
    """The question with comparison symbols as words and a minus sign kept as "minus".

    A bound's adjective moves before it as the quantity it bounds: "less than 1 km deep" reads as "depth
    less than 1 km", and "deeper than 50 cm" as "depth more than 50 cm".
    """
    text = str(question)
    for symbol, words in SYMBOL_WORDS:
        text = text.replace(symbol, words)
    text = POSTPOSED_ADJECTIVE.sub(
        lambda match: f"{DIMENSION_ADJECTIVES[match.group('adjective').lower()]} {match.group('bound')} "
        f"{match.group('number')}{match.group('unit') or ''}", text)
    text = COMPARATIVE_BOUND.sub(
        lambda match: "{} {} than {}".format(
            *COMPARATIVE_ADJECTIVES[match.group("adjective").lower()], match.group("number")), text)
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




def _verb_nouns(word: str) -> set[str]:
    """The nouns a past participle may stand for as a value: "adopted" for "adoption", "returned" for
    "return", "transferred" for "transfer"."""
    if len(word) <= 5 or not word.endswith("ed"):
        return set()
    stems = {word[:-2], word[:-1]}
    if len(word) > 6 and word[-3] == word[-4]:
        stems.add(word[:-3])
    return stems | {f"{stem}ion" for stem in stems if stem.endswith("t")}


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
    """Each column's normalized name, then the shorter forms that name it alone.

    The shorter forms are its core name (without a trailing unit), its name
    without a prefix most columns share ("YS (MPa)" for "PROPERTY: YS (MPa)")
    and without LaTeX markup, and their cores. A form shared with another
    column, or found inside another column's name, is left out: with
    "Temp (°C)" and "Temp (°F)", "temp" names neither.
    """
    full = {column: _norm(column) for column in columns}
    candidates: dict[str, list[str]] = {}
    for column, spellings in _spellings(columns).items():
        forms = [_core_name(column)]
        for spelling in spellings[1:]:
            forms += [_norm(spelling), _core_name(spelling)]
        candidates[column] = [form for form in dict.fromkeys(forms) if form and form != full[column]]
    # Every run of whole words in every full name, with the columns it occurs in,
    # so finding a form inside another name is a lookup rather than a scan.
    owners: dict[str, set[str]] = defaultdict(set)
    for column, name in full.items():
        words = name.split()
        for start in range(len(words)):
            for end in range(start + 1, len(words) + 1):
                owners[" ".join(words[start:end])].add(column)
    for column, forms in candidates.items():
        for form in forms:
            owners[form].add(column)
    return {
        column: (full[column], *(form for form in candidates[column] if owners[form] == {column}))
        for column in columns
    }


# A prefix that most column names share before a colon, as in "PROPERTY: YS (MPa)".
NAME_PREFIX = re.compile(r"^\s*([^:]{1,40}):\s*(?=\S)")
# LaTeX markup in a name: "$\mu$m", "g/cm$^3$".
LATEX_MARKUP = re.compile(r"\$|\\[A-Za-z]+|[\^{}]")


def _spellings(columns: list[str]) -> dict[str, list[str]]:
    """Each column's name as written, without a shared prefix, without LaTeX markup and camelCase split."""
    prefixes = Counter(
        match.group(1).strip().casefold() for column in columns if (match := NAME_PREFIX.match(str(column)))
    )
    shared = {prefix for prefix, count in prefixes.items() if count >= 3 and 2 * count >= len(columns)}
    spellings: dict[str, list[str]] = {}
    for column in columns:
        forms = [str(column)]
        match = NAME_PREFIX.match(forms[0])
        if match and match.group(1).strip().casefold() in shared:
            forms.append(forms[0][match.end():])
        forms += [LATEX_MARKUP.sub("", form) for form in forms if LATEX_MARKUP.search(form)]
        # camelCase split, so "fastest lap speed" names fastestLapSpeed.
        forms += [split for form in forms if _norm(split := normalized_name(form)) != _norm(form)]
        spellings[column] = forms
    return spellings


@dataclass(frozen=True)
class _NameIndex:
    """A table's column names, indexed once so that each question is matched against them quickly.

    A table with thousands of columns would otherwise be scanned name by name,
    for every question and for every way of reading it.
    """

    names: dict[str, tuple[str, ...]]
    #: Every form of every name, longest first, as (form, column).
    forms: tuple[tuple[str, str], ...]
    #: Each spelling of a form (its plurals too) and the positions in ``forms`` it spells.
    spellings: dict[str, tuple[int, ...]]
    #: Most words in any spelling.
    longest: int
    #: Each column's words for partial matching (see ``_partial_words``).
    partial: dict[str, tuple[str, ...]]
    #: Each of those words and the columns whose names hold it.
    holding: dict[str, frozenset[str]]
    position: dict[str, int]
    #: Every word of every form, which a comparison may repeat ("condition 2 than condition 1").
    words: frozenset[str] = frozenset()
    #: Each column's unit words (see ``_unit_words``).
    units: dict[str, set[str]] = field(default_factory=dict)


_NAME_INDEXES: dict[tuple, _NameIndex] = {}


def _name_index(columns) -> _NameIndex:
    """The index of a table's column names, built once per set of names."""
    key = tuple(columns)
    cached = _NAME_INDEXES.get(key)
    if cached is not None:
        return cached
    names = _column_names(list(key))
    forms = tuple(sorted(
        ((form, column) for column, column_forms in names.items() for form in column_forms if form),
        key=lambda item: len(item[0]),
        reverse=True,
    ))
    spellings: dict[str, list[int]] = defaultdict(list)
    for rank, (form, _) in enumerate(forms):
        for variant in _word_variants(form):
            spellings[variant].append(rank)
    partial = _partial_words(list(key))
    holding: dict[str, set[str]] = defaultdict(set)
    for column, words in partial.items():
        for word in words:
            holding[word].add(column)
    index = _NameIndex(
        names=names,
        forms=forms,
        spellings={spelling: tuple(ranks) for spelling, ranks in spellings.items()},
        longest=max((len(spelling.split()) for spelling in spellings), default=1),
        partial=partial,
        holding={word: frozenset(found) for word, found in holding.items()},
        position={column: place for place, column in enumerate(key)},
        words=frozenset(word for form, _ in forms for word in form.split()),
        units={column: _unit_words(column) for column in key},
    )
    if len(_NAME_INDEXES) >= 16:
        _NAME_INDEXES.clear()
    _NAME_INDEXES[key] = index
    return index


def _follows(question: str, start: int, words: str = f"(?:{GROUPING_WORDS}|{RANKING_WORDS})") -> bool:
    """Whether the text at ``start`` comes right after one of ``words``, as in "by timepoint"."""
    return bool(re.search(rf"\b{words}\s+(?:the\s+)?$", question[:start]))


def _mention_spans(
    question: str, index: _NameIndex, *, worded: bool = False, protected: tuple[tuple[int, int], ...] = ()
) -> dict[str, tuple[int, int]]:
    """Map each column named in the question to the span that names it, in question order.

    Longer names claim their words first, so "revenue growth" is not also read
    as "revenue". A core name that is also a query word ("time" for "Time (h)")
    names its column only right after a grouping word: "by time", not "over time".
    With ``worded``, columns named by a number ("120") are left for later, and
    words inside a ``protected`` span (a value typed in full) name no column.
    """
    tokens = _tokens(question)
    words = [word for _, _, word in tokens]
    claimed = [any(start <= token[0] and token[1] <= end for start, end in protected) for token in tokens]
    hits: set[int] = set()
    for first in range(len(words)):
        for size in range(1, min(index.longest, len(words) - first) + 1):
            hits.update(index.spellings.get(" ".join(words[first:first + size]), ()))
    found: dict[str, tuple[int, int]] = {}
    # Every name as written first, then its plurals: "lake" names LAKE before it names LAKES.
    for rank, exact in [(rank, True) for rank in sorted(hits)] + [(rank, False) for rank in sorted(hits)]:
        form, column = index.forms[rank]
        if column in found or (worded and _is_number(index.names[column][0])):
            continue
        grouping_only = form in QUERY_GRAMMAR and form != index.names[column][0]
        variants = [form] if exact else sorted(_word_variants(form), key=lambda text: (-len(text), text))
        for variant in variants:
            wanted = variant.split()
            size = len(wanted)
            match = next(
                (
                    first
                    for first in range(len(words) - size + 1)
                    if words[first:first + size] == wanted
                    and not any(claimed[first:first + size])
                    and (not grouping_only or _follows(question, tokens[first][0]))
                ),
                None,
            )
            if match is not None:
                found[column] = (tokens[match][0], tokens[match + size - 1][1])
                claimed[match:match + size] = [True] * size
                break
    return dict(sorted(found.items(), key=lambda item: item[1][0]))


def _mentions(question: str, index: _NameIndex) -> dict[str, int]:
    """Map each column named in the question to where it is named, in question order."""
    return {column: start for column, (start, _) in _mention_spans(question, index).items()}


def _without_names(question: str, index: _NameIndex) -> str:
    """The normalized question with every column name in it blanked, each word kept in place.

    A word inside a name ("Impact" in "Charpy Impact Strength", a column called
    "max" or "SEP") belongs to the name and is never read as an instruction.
    """
    for start, end in _mention_spans(question, index).values():
        question = _blank(question, start, end)
    return question


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
    # "65 or older" and "5 or more" bound one number; "Texas or Ohio" is refused.
    r"or(?!\s+(?:older|over|more|above|younger|under|less|fewer|below)\b)"
    # "the last quarter of 2019" names its year.
    r")\b(?!\s+of\s+(?:19|20)\d{2})"
)


YEAR = r"(?:19|20)\d{2}"
# Ranges of years, read as conditions on the date: "since 1978", "before 1950",
# "between 1990 and 2000". Each gives the operator on the year, or "range".
YEAR_RANGES: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(rf"\bbetween\s+(?P<low>{YEAR})\s+and\s+(?P<high>{YEAR})\b"), "range"),
    (re.compile(rf"\bfrom\s+(?P<low>{YEAR})\s+(?:to|until|till|through)\s+(?P<high>{YEAR})\b"), "range"),
    (re.compile(
        rf"\b(?:since|from|starting(?:\s+in|\s+from)?|beginning(?:\s+in)?)\s+(?P<year>{YEAR})(?:\s+onwards?)?\b"
    ), ">="),
    (re.compile(rf"\b(?P<year>{YEAR})\s+(?:onwards?|and\s+later|and\s+after)\b"), ">="),
    (re.compile(rf"\bafter\s+(?P<year>{YEAR})\b"), ">"),
    (re.compile(rf"\bbefore\s+(?P<year>{YEAR})\b"), "<"),
    # "48 women had won by 2016": by a year is up to its end.
    (re.compile(rf"\b(?:until|till|up\s+to|through|by(?:\s+the\s+end\s+of)?)\s+(?P<year>{YEAR})\b"), "<="),
)
# A day written out: "April 27, 2011", "27 April 2011", "2011-04-27".
MONTH_WORD = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
DAY_PATTERN = re.compile(
    rf"\b(?:(?P<month>{MONTH_WORD})\.?\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?,?\s+(?P<year>(?:1[89]|20)\d{{2}})"
    rf"|(?P<day2>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<month2>{MONTH_WORD})\.?,?\s+(?P<year2>(?:1[89]|20)\d{{2}})"
    # A question's dashes are read as spaces, so "2011-04-27" is also "2011 04 27".
    r"|(?P<iso>(?:1[89]|20)\d{2}[-\s]\d{2}[-\s]\d{2}))\b",
    re.IGNORECASE,
)
EPOCH = pd.Timestamp("1970-01-01")
# Part of a year: "the first half of 2021", "the last quarter of 2019".
PART_OF_YEAR = re.compile(
    rf"\bthe\s+(?P<which>first|second|third|fourth|last|final|1st|2nd|3rd|4th)\s+(?P<part>half|quarter)\s+of\s+"
    rf"(?P<year>{YEAR})\b",
    re.IGNORECASE,
)
PART_ORDER = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4}


def day_named(match: re.Match) -> pd.Timestamp | None:
    """The day a ``DAY_PATTERN`` match names, or None when it is no real date (April 31)."""
    if match.group("iso"):
        text = re.sub(r"\s", "-", match.group("iso"))
    else:
        month = match.group("month") or match.group("month2")
        day = match.group("day") or match.group("day2")
        year = match.group("year") or match.group("year2")
        number = MONTH_NAMES.get(month.lower().rstrip("."), MONTH_NAMES.get(month.lower()[:3]))
        if number is None:
            return None
        text = f"{year}-{number:02d}-{int(day):02d}"
    try:
        return pd.Timestamp(text)
    except ValueError:
        return None


# Ages written out: "30- to 34-year-olds", "aged 50 and over", "people 65 and older", "20-year-olds".
_AGE_CUE = r"(?:aged|ages|age)\s+"
_YEAR_OLDS = r"\s*-?\s*years?\s*-?\s*olds?\b"
AGE_PHRASE = re.compile(
    rf"\b(?:{_AGE_CUE}(?P<low>\d{{1,3}})\s*-?\s*(?:to|-|–)\s*(?P<high>\d{{1,3}})(?:{_YEAR_OLDS})?"
    rf"|(?P<low2>\d{{1,3}})\s*-?\s*(?:to|-|–)\s*(?P<high2>\d{{1,3}}){_YEAR_OLDS}"
    rf"|(?:{_AGE_CUE})?(?P<older>\d{{1,3}})\s*(?:and|or)\s+older\b"
    rf"|{_AGE_CUE}(?P<over>\d{{1,3}})\s*(?:\+|\s+and\s+(?:over|up|above)\b|\s+or\s+(?:over|more|above)\b)"
    rf"|(?P<plus>\d{{1,3}})\+(?=\s|$|[,.;])"
    rf"|{_AGE_CUE}(?P<one>\d{{1,3}})\b(?!\s*(?:-|–|to\b|and\b|or\b|\+|%))"
    rf"|(?P<olds>\d{{1,3}}){_YEAR_OLDS})",
    re.IGNORECASE,
)
# An age bound after a linking verb, with no age column named: "was under 40", "were over 65".
AGE_BOUND = re.compile(
    r"\b(?:was|were|is|are|be|been|aged)\s+(?P<side>under|below|over|above|younger\s+than|older\s+than)\s+"
    r"(?P<age>\d{1,3})\b(?!\s*(?:%|per\s?cent|[.,]\d))",
    re.IGNORECASE,
)
# A band of ages as a table writes it: "30-34", "65+", ">50", "12".
BAND_VALUE = re.compile(r"^\s*(?:(?P<low>\d{1,3})\s*(?:-|–|to)\s*(?P<high>\d{1,3})|(?P<plus>\d{1,3})\s*\+|"
                        r">\s*(?P<above>\d{1,3})|(?P<one>\d{1,3}))\s*$")


def age_range(match: re.Match) -> tuple[float, float]:
    """The ages an ``AGE_PHRASE`` match names, inclusive; an open top is infinity."""
    found = {name: int(value) for name, value in match.groupdict().items() if value}
    if "low" in found or "low2" in found:
        return float(found.get("low", found.get("low2"))), float(found.get("high", found.get("high2")))
    start = next((found[name] for name in ("older", "over", "plus") if name in found), None)
    if start is not None:
        return float(start), float("inf")
    single = found.get("one", found.get("olds"))
    return float(single), float(single)


def _band(value: str) -> tuple[float, float] | None:
    """The ages a band value covers: "30-34" is 30 to 34, "65+" 65 and up, ">50" 51 and up."""
    found = BAND_VALUE.match(value)
    if not found:
        return None
    if found.group("low"):
        return float(found.group("low")), float(found.group("high"))
    if found.group("plus"):
        return float(found.group("plus")), float("inf")
    if found.group("above"):
        return float(found.group("above")) + 1, float("inf")
    return float(found.group("one")), float(found.group("one"))


def _age_column(
    dataframe: pd.DataFrame, roles: ColumnRoles
) -> tuple[str, dict[str, tuple[float, float]]] | None:
    """The column of ages, and its bands when it holds them as text ("30-34", "65+")."""
    named = [column for column in dataframe.columns
             if {"age", "ages", "agegroup"} & set(_norm(normalized_name(str(column))).split())]
    if len(named) != 1:
        return None
    column = named[0]
    series = dataframe[column]
    if _is_measurement(series):
        return column, {}
    values = [str(value) for value in series.dropna().unique()]
    bands = {value: band for value in values if (band := _band(value))}
    return (column, bands) if values and len(bands) == len(values) else None


def _claim_ages(reading: _Reading, dataframe: pd.DataFrame, roles: ColumnRoles) -> None:
    """Read ages written out as the rows of those ages.

    On a numeric age column, "aged 50 and over" is age >= 50. On a column of
    bands ("30-34", "65+"), the bands that make up the ages exactly are kept:
    "30- to 34-year-olds" is the band 30-34, "50 and over" is 50-64 and 65+.
    Ages that cut a band in two are refused, as no row says which side it is.
    """
    found = _age_column(dataframe, roles)
    if found is None:
        return
    column, bands = found
    for match in AGE_PHRASE.finditer(reading.question):
        low, high = age_range(match)
        said = _norm(_question_text(match.group()))
        start = reading.remaining.find(said)
        span = slice(start, start + len(said))
        if start < 0 or reading.remaining[span] != reading.original[span]:
            continue
        if not bands:
            if low == high:
                reading.conditions.append(Condition(column, "=", low))
            else:
                reading.conditions.append(Condition(column, ">=", low))
                if high != float("inf"):
                    reading.conditions.append(Condition(column, "<=", high))
        else:
            within = {value for value, band in bands.items() if band[0] >= low and band[1] <= high}
            inside = sorted((band, value) for value, band in bands.items() if value in within)
            cut = [value for value, band in bands.items()
                   if band[0] <= high and band[1] >= low and value not in within]
            covered = bool(inside) and inside[0][0][0] == low and inside[-1][0][1] == high and all(
                after[0][0] == before[0][1] + 1 for before, after in zip(inside, inside[1:], strict=False))
            if cut:
                reading.flag(f'"{match.group().strip()}" cuts the {column} band {cut[0]} in two, and no row '
                             "says which side its people are on.")
            elif not covered:
                continue
            else:
                picked = [_Value(column=column, raw=value, words=()) for _, value in inside]
                _add_filter(reading, picked, start)
                reading.notes.append(f'Read "{match.group().strip()}" as {column} in '
                                     f"{_listed([value for _, value in inside])}.")
        reading.consumed.add(column)
        reading.scoped.add(column)
        reading.claim(start, start + len(said))
    if bands:
        return
    for match in list(AGE_BOUND.finditer(reading.remaining)):
        # "was under 40" on a numeric age column: below 40, whatever the fraction of a year.
        side = match.group("side").lower()
        operator = "<" if side in ("under", "below") or side.startswith("younger") else ">"
        reading.conditions.append(Condition(column, operator, float(match.group("age"))))
        reading.notes.append(f'Read "{match.group("side")} {match.group("age")}" as {column} '
                             f'{OPERATOR_WORDS[operator]} {match.group("age")}.')
        reading.consumed.add(column)
        reading.scoped.add(column)
        reading.claim(match.start("side"), match.end())


# A decade: "the 2010s" is 2010 to 2019.
DECADE_PATTERN = re.compile(r"\b(?P<decade>(?:19|20)\d0)s\b")
# Two groups split at a year: "before and after 2000".
SPLIT_PATTERN = re.compile(
    rf"\b(?:before\s+(?:and|vs|versus)\s+after|pre\s+(?:and|vs|versus)\s+post)\s+(?P<year>{YEAR})\b"
)


def _asks_for_several_periods(q: str) -> bool:
    """Whether the question names two or more months or years, which a plan cannot hold.

    The two ends of a range of years ("between 1990 and 2000") are one condition.
    """
    for pattern, _ in YEAR_RANGES:
        q = pattern.sub(" ", q)
    months = {MONTH_NAMES[name] for name in MONTH_NAMES if re.search(rf"\b{name}\b", q)}
    years = set(re.findall(rf"\b{YEAR}\b", q))
    return len(months) >= 2 or len(years) >= 2


def unsupported_phrasing(question: str, columns=()) -> bool:
    """Whether the question uses a form no plan can hold, such as "!=", "or" or two separate years.

    ``columns`` are the table's column names, so a word in one ("SEP" and
    "OCT" as columns, "Sales or Returns") is read as the name it is.
    """
    q = _norm(question)
    if len(columns):
        q = _without_names(q, _name_index(columns))
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
    index: _NameIndex | None = None
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
    #: Plurals the values of several columns share, which name the rows ("characters").
    nouns: set[str] = field(default_factory=set)
    #: The number a rate is given per: 100 in "claims per 100 holders".
    scale: float | None = None
    #: The column of counts the rows' noun names: encounter_count for "encounters".
    weight: str | None = None
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
    #: Columns of filters and conditions introduced by "for", "where", "in" and the like,
    #: which narrow the rows rather than name what a share counts.
    scoped: set[str] = field(default_factory=set)
    #: Where in the question each filtered column's value was read, for the base of a share.
    starts: dict[str, int] = field(default_factory=dict)
    #: "rate of X", "X rate": the share of rows where X holds.
    rate: bool = False
    #: "before and after 2000": the column to split and where.
    split: tuple[str, float] | None = None
    #: Readings to hold fixed: "after accounting for time".
    adjusted: list[str] = field(default_factory=list)
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
    # Picking the text columns by dtype at once keeps a table of thousands of numeric columns quick.
    texts = dataframe.select_dtypes(exclude=["number", "bool", "datetime", "datetimetz"])
    for position in range(texts.shape[1]):
        column = texts.columns[position]
        if column == roles.date:
            continue
        uniques = texts.iloc[:, position].dropna().unique()
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


def _value_nouns(vocabulary: _Vocabulary) -> set[str]:
    """Plural words most values of two or more columns share: "characters" in "Male Characters".

    Such a word names the rows the values describe ("Good Characters", "Living
    Characters"), not a value of any one column.
    """
    columns: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for value in vocabulary.values:
        columns[value.column].append(value.words)
    sharing: dict[str, int] = defaultdict(int)
    for words in columns.values():
        if len(words) < 2:
            continue
        counts = Counter(word for value in words for word in set(value))
        for word, count in counts.items():
            if len(word) > 3 and word.endswith("s") and count * 2 >= len(words):
                sharing[word] += 1
    return {word for word, count in sharing.items() if count >= 2}


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
    if len(run) == 1 and len(run[0]) > 2:
        # "in" is not "Ines": a word of two letters has no plural to match.
        texts |= _word_variants(run[0])
    exact = [vocabulary.values[index] for text in texts for index in vocabulary.by_text.get(text, ())]
    if column is not None:
        exact = [value for value in exact if value.column == column]
    nouns = _verb_nouns(run[0]) if len(run) == 1 and not exact else set()
    named = [vocabulary.values[index] for text in nouns for index in vocabulary.by_text.get(text, ())
             if column is None or vocabulary.values[index].column == column]
    if named:
        # "were adopted" names "adoption"; when another column holds the noun inside a longer value
        # ("return to owner" beside "return"), the words could mean either.
        within = {index for noun in nouns for index in vocabulary.by_word.get(noun, ())}
        others = [vocabulary.values[index] for index in sorted(within)
                  if vocabulary.values[index].column not in {value.column for value in named}
                  and (column is None or vocabulary.values[index].column == column)]
        return [*named, *others], not others
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
    if not exact and dataframe is not None:
        # "TCD" inside a research body's name and inside a grant's title: a title is free text, and a
        # word within it does not pick a group the way a category's value does.
        categories = [column for column in columns if not _free_text(dataframe[column])]
        if not categories:
            return None, [], None
        found = [value for value in found if value.column in categories]
        columns = categories
    if not exact and re.sub(r"[^A-Za-z]", "", phrase).islower() and len({value.raw for value in found}) > 1:
        # "new TB cases": a word in lower case is not the capitalised word of several names ("New Zealand",
        # "Papua New Guinea"); one name it fits ("the southwest border") may still be meant.
        found = [value for value in found if not _capitalised_in(value.raw, run)]
        if not found:
            return None, [], None
        columns = list(dict.fromkeys(value.column for value in found))
    if len(columns) > 1 and dataframe is not None and _common_noun(phrase, question):
        # "dogs" is a kind in animal_type and one pet's name in animal_name: a word written in lower
        # case is a common noun, and a column of names does not hold kinds.
        kinds = [column for column in columns if not _names_column(dataframe[column])]
        if kinds and len(kinds) < len(columns):
            found = [value for value in found if value.column in kinds]
            columns = kinds
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


def _common_noun(phrase: str, question: str) -> bool:
    """Whether the question writes a phrase in lower case, as a common noun ("dogs") and not a name."""
    letters = re.sub(r"[^A-Za-z]", "", phrase)
    if not letters:
        return False
    if letters.islower():
        return True
    # "Dogs made up..." opens the sentence with a capital.
    return question.lstrip().startswith(phrase) and letters[1:].islower()


def _capitalised_in(raw: str, run: tuple[str, ...]) -> bool:
    """Whether the words of ``run`` start with a capital where ``raw`` holds them: "New" in "New Zealand"."""
    tokens = re.findall(r"[A-Za-z0-9]+", raw)
    lowered = [token.lower() for token in tokens]
    for index in range(len(tokens) - len(run) + 1):
        if tuple(lowered[index:index + len(run)]) == run or tuple(
            _norm(word) for word in lowered[index:index + len(run)]
        ) == run:
            return tokens[index][:1].isupper()
    return False


def _names_column(series: pd.Series) -> bool:
    """Whether a text column holds names: at least 50 different values, one for every ten rows or more."""
    values = series.dropna().astype(str)
    distinct = values.nunique()
    return distinct >= 50 and distinct >= 0.1 * len(values)


def _free_text(series: pd.Series) -> bool:
    """Whether a text column holds free text, such as titles: values mostly unique and five words or more."""
    values = series.dropna().astype(str)
    if values.empty or values.nunique() < 0.5 * len(values):
        return False
    return float(values.str.split().str.len().mean()) >= 5


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


def _typed_in_case(value: _Value, question: str) -> bool:
    """Whether a value with capitals is typed exactly as the data spells it: "ALL", not "all"."""
    return value.raw != value.raw.lower() and bool(
        re.search(rf"(?<![0-9A-Za-z]){re.escape(value.raw)}(?![0-9A-Za-z])", question)
    )


def _cued(reading: _Reading, start: int) -> bool:
    """Whether the word before ``start`` marks a filter: "for Top", "where ... is Open"."""
    before = reading.original[:start].split()
    return bool(before) and before[-1] in VALUE_CUES


# Words before a filter or condition that narrow the rows ("for West", "where pH is
# below 5") rather than name what a share counts ("share of Result 1", "with diabetes").
SCOPE_CUES = frozenset({"for", "in", "where", "when", "within", "among", "amongst", "only", "at", "from"})


def _scope_cued(reading: _Reading, start: int) -> bool:
    before = reading.original[:start].split()
    return bool(before) and before[-1] in SCOPE_CUES


def _add_filter(reading: _Reading, values: list[_Value], start: int | None = None) -> None:
    column = values[0].column
    if start is not None:
        reading.starts.setdefault(column, start)
    known = reading.filters.get(column, ())
    reading.filters[column] = tuple(dict.fromkeys((*known, *(value.raw for value in values))))


def _person_name_columns(dataframe: pd.DataFrame) -> tuple[str, str] | None:
    """The table's one column of first names and one of last names, when it has both."""

    def named(marks: frozenset[str]) -> list[str]:
        found = []
        for column in dataframe.columns:
            words = set(normalized_name(str(column)).split())
            if words & marks and words & NAME_WORDS and not is_numeric_dtype(dataframe[column]):
                found.append(column)
        return found

    firsts, lasts = named(FIRST_NAME_WORDS), named(LAST_NAME_WORDS)
    return (firsts[0], lasts[0]) if len(firsts) == 1 and len(lasts) == 1 else None


def _claim_person_names(reading: _Reading, dataframe: pd.DataFrame) -> None:
    """Filter on a person named by first and last name, each in a column of its own: "Barry Bonds".

    Two words in a row name a person when some row holds the first as its
    first name and the second as its last name. Name columns can hold more
    values than the value index keeps, so the pairs are looked up directly.
    """
    columns = _person_name_columns(dataframe)
    if columns is None:
        return
    first, last = columns
    pairs = dataframe[[first, last]].dropna().drop_duplicates()
    people: dict[tuple[str, str], tuple[str, str]] = {}
    for given, family in zip(pairs[first].astype(str), pairs[last].astype(str), strict=True):
        people.setdefault((_norm(given), _norm(family)), (given, family))
    tokens = list(_tokens(reading.remaining))
    for (start, end, word), (after, stop, following) in zip(tokens, tokens[1:], strict=False):
        found = people.get((word, following))
        if found is None or reading.original[end:after] != " ":
            continue
        reading.filters[first] = (found[0],)
        reading.filters[last] = (found[1],)
        reading.starts.setdefault(first, start)
        reading.starts.setdefault(last, start)
        reading.claim(start, stop)
        return


def _claim_values(
    reading: _Reading, vocabulary: _Vocabulary, dataframe: pd.DataFrame, *, exact_only: bool,
    row_noun: str | None = None, nouns: set[str] = frozenset(),
) -> None:
    """Filter on values of text columns found among the unclaimed words, longest first.

    A value made only of query words or nouns for rows ("Top", "Total",
    "Patients") needs a cue before it, so "top 2 contours" still ranks while
    "average Ca for Top" filters. The file's own noun for its rows ("members" in
    members.csv) names the rows: it is neither part of a longer value ("Himalaya
    Expeditions") nor a value at all unless typed as the data spell it ("Member").
    """
    row_nouns = (_word_variants(row_noun) if row_noun else set()) | set(nouns)
    for run in _runs(reading, content_only=False):
        index = 0
        while index < len(run):
            for size in range(min(MAX_VALUE_WORDS, len(run) - index), 0, -1):
                window = run[index:index + size]
                words = tuple(word for _, _, word in window)
                ordinary = [word for word in words if word not in QUERY_GRAMMAR and word not in ENTITY_NOUNS]
                cued = _cued(reading, window[0][0])
                if size == 1 and words[0] in DETERMINERS:
                    # "any other month", "39 of the languages": a determiner is a value ("Other", the
                    # language "The") only typed in the data's own capitals.
                    found, exact = _values_named(words, vocabulary, exact_only=True)
                    found = [value for value in found if _typed_in_case(value, reading.question)]
                    if not found:
                        continue
                elif size == 1 and words[0] in row_nouns:
                    # The file's own noun for its rows ("members" in members.csv) names the rows; it is
                    # a value only typed as the data spell it ("Member").
                    found, exact = _values_named(words, vocabulary, exact_only=True)
                    found = [value for value in found if _typed_in_case(value, reading.question)]
                    if not found:
                        continue
                elif not ordinary and not cued:
                    # A value spelled like a query word is still a value when typed in its own
                    # capitals: "between AML and ALL".
                    found, exact = _values_named(words, vocabulary, exact_only=True)
                    kinds = [value for value in found if size == 1 and words[0] in ENTITY_NOUNS
                             and not _names_column(dataframe[value.column])]
                    # "birds" in a table whose animal_type holds "bird" picks those rows.
                    found = kinds or [value for value in found if _typed_in_case(value, reading.question)]
                    if not found:
                        continue
                elif not exact_only and sum(len(word) for word in ordinary) < 3:
                    continue
                elif not exact_only and all(map(_is_number, words)):
                    # A number inside a longer value ("2020" in a race's web address) is left
                    # for the year, a condition or a count.
                    continue
                else:
                    found, exact = _values_named(words, vocabulary, exact_only=exact_only)
                    if all(word in QUERY_GRAMMAR for word in words):
                        # "is to", "for top": a query word after a cue names a value spelled just so,
                        # not "tos".
                        found = [value for value in found if value.words == words]
                found = [value for value in found if _plain_match(value, reading.question, cued)]
                adverb = not found and size == 1 and len(words[0]) > 4 and words[0].endswith("ly")
                if adverb:
                    # "highly satisfied": the adverb of a value, kept when the next words name its column.
                    found, exact = _values_named((words[0][:-2],), vocabulary, exact_only=True)
                if not found:
                    continue
                columns = list(dict.fromkeys(value.column for value in found))
                beside = _column_beside(reading, columns, window[0][0], window[-1][1]) if (
                    len(columns) > 1 or adverb or not exact) else None
                named = reading.original[beside[1][0]:beside[1][1]].split() if beside else []
                if not exact and not adverb and named and re.fullmatch(r"[a-z]+[^s]s", named[-1]):
                    # "London boroughs": a word only inside a value ("City of London"), before the plural of
                    # that value's own column, says which ones, not one of them. "the southwest border"
                    # still names Southwest Land Border.
                    continue
                if adverb and beside is None:
                    continue
                if beside is not None:
                    # "high satisfaction", "a high level of contact": the word beside the value names
                    # which of the columns holding it is meant.
                    found = [value for value in found if value.column == beside[0]]
                chosen, values, problem = _pick_values(words, found, exact, reading.question, dataframe)
                if not problem and chosen is None:
                    # Words found only inside free text pick nothing and stay unread.
                    continue
                if problem:
                    reading.flag(problem)
                elif chosen is not None:
                    _add_filter(reading, values, window[0][0])
                    if _scope_cued(reading, window[0][0]):
                        reading.scoped.add(chosen.column)
                    if beside is not None:
                        reading.consumed.add(chosen.column)
                        reading.claim(*beside[1])
                        said = reading.original[window[0][0]:beside[1][1]]
                        reading.notes.append(
                            f'Read "{_raw_words(reading.question, said.split())}" as {chosen.column} = '
                            f"{_listed([value.raw for value in values])}."
                        )
                reading.claim(window[0][0], window[-1][1])
                index += size
                break
            else:
                index += 1


# Words that may stand between a value and the column naming it: "a high level of contact".
BESIDE_FILLERS = frozenset({"level", "levels", "degree", "amount", "of", "in", "with"})


def _column_beside(
    reading: _Reading, columns: list[str], start: int, end: int
) -> tuple[str, tuple[int, int]] | None:
    """The one of ``columns`` named right after a value, and the words from the value's end to the name's.

    The name is a column already found there ("a high level of contact" with
    Cont), or the next unclaimed word, which a name word of three letters or
    more starts ("satisfaction" for Sat). Only filler words such as "level
    of" may come between.
    """
    after = reading.original[end:]
    gap = re.match(r"(?:\s+(?:" + "|".join(sorted(BESIDE_FILLERS)) + r"))*\s+", after)
    if gap is None:
        return None
    position = end + gap.end()
    spanned = [column for column in columns if reading.spans.get(column, (None,))[0] == position]
    if len(spanned) == 1:
        return spanned[0], (end, reading.spans[spanned[0]][1])
    word = re.match(r"[a-z]+", reading.remaining[position:])
    if word is None:
        return None
    names = _partial_words(columns)
    prefixed = [
        column for column in columns
        if any(len(name) >= 3 and word.group().startswith(name) for name in names[column])
    ]
    if len(prefixed) != 1:
        return None
    return prefixed[0], (end, position + word.end())


def _claim_codes(reading: _Reading, dataframe: pd.DataFrame, roles: ColumnRoles) -> None:
    """Filter on a code of letters and digits ("bondsba01") held by a text column too large to index.

    Columns with more than MAX_FILTER_CANDIDATES values are left out of the
    vocabulary; a word that mixes letters and digits is looked up in them
    exactly, and read as a value when one column holds it.
    """
    # Names are not codes: a person's name is read by the name rule (``_claim_person_names``).
    large = [
        column for column in dataframe.select_dtypes(exclude=["number", "bool", "datetime", "datetimetz"])
        if column != roles.date and not NAME_WORDS & set(normalized_name(str(column)).split())
        and dataframe[column].nunique() > MAX_FILTER_CANDIDATES
    ]
    if not large:
        return
    for start, end, word in _tokens(reading.remaining):
        if len(word) < 4 or not re.fullmatch(r"(?=.*[a-z])(?=.*\d)[a-z0-9_.-]+", word):
            continue
        held = []
        for column in large:
            values = dataframe[column].dropna().astype(str)
            matched = values[values.str.lower() == word]
            if len(matched):
                held.append(_Value(column=column, raw=str(matched.iloc[0]), words=(word,)))
        if len(held) == 1:
            _add_filter(reading, held, start)
            reading.claim(start, end)


# Words for the sexes, and the values a column of sexes spells them with: "men" is Male or M.
SEX_WORDS = {
    "men": "male", "man": "male", "males": "male", "boys": "male", "boy": "male",
    "women": "female", "woman": "female", "females": "female", "girls": "female", "girl": "female",
}
SEX_VALUES = {
    "male": frozenset({"male", "m", "man", "men"}), "female": frozenset({"female", "f", "woman", "women"}),
}
# Values a column of genders may hold besides the sexes: "Agender", "Non-binary", "Prefer not to say".
GENDER_VALUES = frozenset({
    "other", "unknown", "intersex", "not stated", "agender", "genderfluid", "gender fluid", "non binary",
    "nonbinary", "transgender", "bigender", "genderqueer", "prefer not to say", "not specified",
    "unspecified", "not known",
})


def _claim_sexes(reading: _Reading, vocabulary: _Vocabulary, nouns: set[str] = frozenset()) -> None:
    """Filter on a column of sexes for "men", "women", "boys" or "girls" ("Gender" with Male and Female).

    The column must hold both sexes and nothing but sexes, besides values
    such as "Other", "Unknown" or "Agender" (GENDER_VALUES), so "men" never
    picks a value of any other column. A plural every value shares ("Male
    Characters") is left out of the comparison.
    """
    columns: dict[str, list[_Value]] = defaultdict(list)
    for value in vocabulary.values:
        columns[value.column].append(value)

    def spelled(value: _Value) -> str:
        return " ".join(word for word in value.words if word not in nouns)

    allowed = SEX_VALUES["male"] | SEX_VALUES["female"] | GENDER_VALUES
    sexed = [
        column for column, values in columns.items()
        if {spelled(value) for value in values} <= allowed
        and all(any(spelled(value) in SEX_VALUES[side] for value in values) for side in SEX_VALUES)
    ]
    if len(sexed) != 1:
        return
    for start, end, word in _tokens(reading.remaining):
        side = SEX_WORDS.get(word)
        if side is None:
            continue
        values = [value for value in columns[sexed[0]] if spelled(value) in SEX_VALUES[side]]
        _add_filter(reading, values, start)
        reading.claim(start, end)
        reading.notes.append(
            f'Read "{_raw_words(reading.question, [word])}" as {sexed[0]} = '
            f"{_listed([value.raw for value in values])}."
        )


# The US states and their postal codes, which a column of states may hold in place of the names.
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "district of columbia": "dc", "florida": "fl",
    "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo",
    "montana": "mt", "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
}
# The fewest postal codes a column must hold, and nothing else, to be read as a column of states.
STATE_CODES_MIN = 10
# The abbreviations newspapers use for states, written with a full stop: "Calif." is California.
PRESS_STATES = {
    "ala": "alabama", "ariz": "arizona", "ark": "arkansas", "calif": "california", "colo": "colorado",
    "conn": "connecticut", "del": "delaware", "fla": "florida", "ga": "georgia", "ill": "illinois",
    "ind": "indiana", "kan": "kansas", "kans": "kansas", "ky": "kentucky", "la": "louisiana",
    "md": "maryland", "mass": "massachusetts", "mich": "michigan", "minn": "minnesota", "miss": "mississippi",
    "mo": "missouri", "mont": "montana", "neb": "nebraska", "nebr": "nebraska", "nev": "nevada",
    "okla": "oklahoma", "ore": "oregon", "pa": "pennsylvania", "tenn": "tennessee", "tex": "texas",
    "vt": "vermont", "va": "virginia", "wash": "washington", "wis": "wisconsin", "wyo": "wyoming",
}
# Postal codes that are also English words in capitals, never read as states.
WORD_CODES = frozenset({"in", "or", "me", "ok", "hi", "oh"})


# Words for the whole of the United States, and the values of a row that stands for it.
COUNTRY_WIDE = re.compile(
    r"\b(?:in|across|throughout)\s+(?:(?:the\s+)?(?:united\s+states(?:\s+of\s+america)?|usa)|"
    r"the\s+(?:u\.?s\.?(?:a\.?)?|country|nation))(?!\w)|\bnation(?:wide|ally)\b",
    re.IGNORECASE,
)
COUNTRY_WORDS = frozenset({
    "united states", "usa", "us", "united states of america", "total", "national", "all",
})


def _claim_country(reading: _Reading, vocabulary: _Vocabulary) -> None:
    """Read "in the United States" or "nationwide" as all the rows of a table of the states.

    Only a table with one column of state names or postal codes, and no row
    for the country as a whole, is read this way; the words then name no
    column ("States" is not the state column).
    """
    codes = set(US_STATES.values())
    columns: dict[str, list[str]] = defaultdict(list)
    for value in vocabulary.values:
        columns[value.column].append(" ".join(value.words))
    whole = [
        column for column, values in columns.items()
        if sum(value in US_STATES or value in codes for value in values) >= STATE_CODES_MIN
        and not any(value in COUNTRY_WORDS for value in values)
    ]
    if len(whole) != 1:
        return
    for match in list(COUNTRY_WIDE.finditer(reading.remaining)):
        reading.spans = {
            column: span for column, span in reading.spans.items()
            if span[1] <= match.start() or span[0] >= match.end()
        }
        reading.claim(match.start(), match.end())
        reading.notes.append(f'Read "{match.group().strip()}" as all the rows: {whole[0]} holds the states, '
                             "and no row is the country as a whole.")


def _claim_us_states(reading: _Reading, vocabulary: _Vocabulary) -> None:
    """Filter on a state's postal code for its name: "Texas" keeps state = tx.

    Only a column whose values are all postal codes, at least STATE_CODES_MIN
    of them, is read this way; a column that spells the names out matches
    them as values.
    """
    codes = set(US_STATES.values())
    columns: dict[str, list[_Value]] = defaultdict(list)
    for value in vocabulary.values:
        columns[value.column].append(value)
    named = [
        column for column, values in columns.items()
        if sum(" ".join(value.words) in US_STATES for value in values) >= STATE_CODES_MIN
    ]
    if len(named) == 1:
        _claim_state_codes(reading, columns[named[0]])
    held = [
        column for column, values in columns.items()
        if len(values) >= STATE_CODES_MIN and all(" ".join(value.words) in codes for value in values)
    ]
    if len(held) != 1:
        return
    names = sorted(US_STATES, key=len, reverse=True)
    for name in names:
        for match in re.finditer(rf"\b{name}\b", reading.remaining):
            values = [value for value in columns[held[0]] if " ".join(value.words) == US_STATES[name]]
            if not values:
                continue
            _add_filter(reading, values, match.start())
            reading.claim(match.start(), match.end())
            reading.notes.append(f'Read "{_raw_words(reading.question, name.split())}" as {held[0]} = '
                                 f"{values[0].raw}.")


# Words for dying and for surviving. A value spelled with one of them ("Died", "Deceased Characters")
# or a yes/no column named with one ("Survived") is what the others mean: "deaths" keeps result = Died.
OUTCOME_WORDS = {
    "died": frozenset({"died", "die", "dead", "death", "deaths", "deceased", "killed", "fatal", "fatality",
                       "fatalities"}),
    "survived": frozenset({"survived", "survive", "survivor", "survivors", "survival", "alive"}),
}
YES_NO_VALUES = {"yes": frozenset({"yes", "y", "true", "1"}), "no": frozenset({"no", "n", "false", "0"})}


def _outcome_columns(reading: _Reading, dataframe: pd.DataFrame) -> None:
    """Name a column of numbers called by a word for dying or surviving: "deaths" is fatal.

    Only when exactly one column of numbers has such a word for its whole name,
    and the question names no column with that word.
    """
    sides = {word: side for side, words in OUTCOME_WORDS.items() for word in words}
    for start, end, word in _tokens(reading.original):
        side = sides.get(word)
        if side is None or any(low <= start < high for low, high in reading.spans.values()):
            continue
        named = [
            column for column in dataframe.columns
            if is_numeric_dtype(dataframe[column]) and sides.get(_norm(normalized_name(str(column)))) == side
        ]
        if len(named) == 1 and named[0] not in reading.spans:
            reading.spans[named[0]] = (start, end)
            reading.notes.append(f'Read "{_raw_words(reading.question, [word])}" as {named[0]}.')


def _claim_outcomes(reading: _Reading, vocabulary: _Vocabulary) -> None:
    """Filter on dying or surviving for a word of the same meaning: "deaths", "killed", "survivors".

    The table must spell the outcome in one column only, as a value ("Died")
    or as a yes/no column named for it ("Survived" with Yes and No, where
    "deaths" keeps No). Otherwise nothing is read.
    """
    sides = {word: side for side, words in OUTCOME_WORDS.items() for word in words}
    columns: dict[str, list[_Value]] = defaultdict(list)
    for value in vocabulary.values:
        columns[value.column].append(value)
    for start, end, word in _tokens(reading.remaining):
        side = sides.get(word)
        if side is None:
            continue
        spelled = [
            value for value in vocabulary.values if any(sides.get(part) == side for part in value.words)
        ]
        if not spelled:
            named = [
                column for column, values in columns.items()
                if set(_norm(normalized_name(str(column))).split()) & set(sides)
                and {" ".join(value.words) for value in values} <= YES_NO_VALUES["yes"] | YES_NO_VALUES["no"]
            ]
            if len(named) != 1:
                continue
            name = _norm(normalized_name(str(named[0]))).split()
            same = sides[next(part for part in name if part in sides)]
            wanted = YES_NO_VALUES["yes" if same == side else "no"]
            spelled = [value for value in columns[named[0]] if " ".join(value.words) in wanted]
        if not spelled or len({value.column for value in spelled}) != 1:
            continue
        _add_filter(reading, spelled, start)
        reading.claim(start, end)
        reading.notes.append(
            f'Read "{_raw_words(reading.question, [word])}" as {spelled[0].column} = '
            f"{_listed([value.raw for value in spelled])}."
        )


def _claim_state_codes(reading: _Reading, values: list[_Value]) -> None:
    """Filter a column of state names on a postal code in capitals ("NM") or a newspaper's "Calif."."""
    by_code = {code: name for name, code in US_STATES.items()}
    raw = reading.question
    for start, end, word in _tokens(reading.remaining):
        spelled = _raw_words(raw, [word])
        state = None
        if len(word) == 2 and word in by_code and word not in WORD_CODES and spelled.isupper():
            state = by_code[word]
        elif word in PRESS_STATES and spelled[:1].isupper() and re.search(rf"\b{re.escape(spelled)}\.", raw):
            state = PRESS_STATES[word]
        found = [value for value in values if " ".join(value.words) == state] if state else []
        if not found:
            continue
        _add_filter(reading, found, start)
        reading.claim(start, end)
        reading.notes.append(f'Read "{spelled}" as {found[0].column} = {found[0].raw}.')


# Adjectives of nationality and the country names a column may spell them with: "Belgian" is Belgium.
DEMONYMS = {
    "american": ("united states", "usa", "us", "united states of america"), "argentine": ("argentina",),
    "argentinian": ("argentina",), "australian": ("australia",), "austrian": ("austria",),
    "belgian": ("belgium",), "brazilian": ("brazil",), "british": ("united kingdom", "uk", "great britain"),
    "canadian": ("canada",), "chilean": ("chile",), "chinese": ("china",), "colombian": ("colombia",),
    "croatian": ("croatia",), "czech": ("czech republic", "czechia"), "danish": ("denmark",),
    "dutch": ("netherlands", "the netherlands", "holland"), "egyptian": ("egypt",),
    "english": ("england",), "ethiopian": ("ethiopia",), "finnish": ("finland",), "french": ("france",),
    "german": ("germany",), "greek": ("greece",), "hungarian": ("hungary",), "indian": ("india",),
    "indonesian": ("indonesia",), "iranian": ("iran",), "irish": ("ireland",), "israeli": ("israel",),
    "italian": ("italy",), "jamaican": ("jamaica",), "japanese": ("japan",), "kenyan": ("kenya",),
    "korean": ("south korea", "korea"), "luxembourgish": ("luxembourg",), "luxembourger": ("luxembourg",),
    "mexican": ("mexico",), "moroccan": ("morocco",), "nigerian": ("nigeria",), "norwegian": ("norway",),
    "pakistani": ("pakistan",), "peruvian": ("peru",), "polish": ("poland",), "portuguese": ("portugal",),
    "romanian": ("romania",), "russian": ("russia",), "scottish": ("scotland",), "slovak": ("slovakia",),
    "slovenian": ("slovenia",), "spanish": ("spain",), "swedish": ("sweden",), "swiss": ("switzerland",),
    "turkish": ("turkey", "turkiye"), "ukrainian": ("ukraine",), "venezuelan": ("venezuela",),
    "vietnamese": ("vietnam", "viet nam"), "welsh": ("wales",),
}
# Name words of a column of nationalities, preferred for a nationality adjective over a country of birth.
NATIONALITY_WORDS = frozenset({"nationality", "citizenship", "nation"})


def _claim_demonyms(reading: _Reading, vocabulary: _Vocabulary) -> None:
    """Filter on a country for its adjective of nationality: "Belgian riders" keeps nationality Belgium.

    When several columns hold the country (a nationality and a country of
    birth), one named for nationality is used; otherwise nothing is read.
    """
    for start, end, word in _tokens(reading.remaining):
        # "Venezuelans" is the plural of an adjective of nationality.
        countries = DEMONYMS.get(word) or (DEMONYMS.get(word[:-1]) if word.endswith("s") else None)
        if countries is None:
            continue
        found = [value for value in vocabulary.values if " ".join(value.words) in countries]
        columns = list(dict.fromkeys(value.column for value in found))
        if len(columns) > 1:
            columns = [
                column for column in columns
                if NATIONALITY_WORDS & set(_norm(normalized_name(str(column))).split())
            ]
        if len(columns) != 1:
            continue
        values = [value for value in found if value.column == columns[0]]
        _add_filter(reading, values, start)
        reading.claim(start, end)
        # "Mexican nationals", "Belgian citizens": the noun says no more than the adjective.
        following = re.match(r"\s+(?:nationals|citizens)\b", reading.remaining[end:])
        if following:
            reading.claim(end, end + following.end())
        reading.notes.append(
            f'Read "{_raw_words(reading.question, [word])}" as {columns[0]} = '
            f"{_listed([value.raw for value in values])}."
        )


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
    names = reading.index or _name_index(dataframe.columns)
    words = {column: names.partial[column] for column in dataframe.columns if column not in reading.spans}
    found: list[str] = []
    for run in _runs(reading, content_only=True):
        index = 0
        while index < len(run):
            index += _claim_partial_at(reading, run, index, words, dataframe, roles, found)
    return found


def _holding(words: dict[str, tuple[str, ...]], phrase: tuple[str, ...], index: _NameIndex) -> list[str]:
    """The columns among ``words`` whose name holds ``phrase``, each word in singular or plural."""
    pool: set[str] = set()
    for variant in _word_variants(phrase[0]):
        pool |= index.holding.get(variant, frozenset())
    found = [column for column in pool if column in words and _holds_variants(words[column], phrase)]
    return sorted(found, key=index.position.__getitem__)


def _abbreviated_columns(phrase: tuple[str, ...], words: dict[str, tuple[str, ...]]) -> list[str]:
    """Columns whose name shortens the phrase word by word: "life exp" for "life expectancy".

    Each name word starts its word of the phrase, with three letters at least,
    and one of them is whole, so a run of short words is not read as initials.
    """
    size = len(phrase)

    def shortens(names: tuple[str, ...]) -> bool:
        return len(names) >= size and any(
            all(word.startswith(name) and (len(name) >= 3 or name == word) for name, word in pairs)
            and any(name == word for name, word in pairs)
            for start in range(len(names) - size + 1)
            if (pairs := list(zip(names[start:start + size], phrase, strict=True)))
        )

    return [column for column, names in words.items() if shortens(names)]


def _initialled_columns(phrase: tuple[str, ...], words: dict[str, tuple[str, ...]]) -> list[str]:
    """Columns named in capitals by the initials of the phrase: "home runs" for HR, "stolen bases" for SB."""
    if not all(word.isalpha() for word in phrase):
        return []
    initials = "".join(word[0] for word in phrase)
    return [
        column for column in words
        if str(column).isalpha() and str(column).isupper() and str(column).lower() == initials
    ]


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
        if size == 1 and phrase[0] == roles.row_noun:
            # The file's own noun for its rows, left for _claim_entity: "flights" in flights.csv.
            return 1
        candidates = _holding(words, phrase, reading.index or _name_index(dataframe.columns))
        if not candidates and size == 1:
            candidates = (_prefixed_columns(phrase[0], words) or _stemmed_columns(phrase[0], words)
                          or _skeleton_columns(phrase[0], words))
        if not candidates and 2 <= size <= 4:
            candidates = _abbreviated_columns(phrase, words) or _initialled_columns(phrase, words)
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


# Endings that turn a word into a column's name word: "fast" names fasting_ability.
STEM_ENDINGS = ("ing", "ed", "er", "ers", "ness", "ity", "ability", "ation", "ion", "ions", "ment", "al")


def _stemmed_columns(word: str, words: dict[str, tuple[str, ...]]) -> list[str]:
    """Columns with a name word that is ``word`` plus a common ending: "fast" for fasting_ability."""
    if not word.isalpha() or len(word) < 4:
        return []
    return [
        column
        for column, names in words.items()
        if any(name.startswith(word) and name[len(word):] in STEM_ENDINGS for name in names)
    ]


def _skeleton_columns(word: str, words: dict[str, tuple[str, ...]]) -> list[str]:
    """Columns with a name word that shortens ``word`` by dropping letters: "Dept" for "department".

    The name word is an abbreviation of three letters or more with at most one
    vowel, starts and ends as the word does, and keeps its letters in order;
    the word is at least three letters longer.
    """
    if not word.isalpha():
        return []

    def shortens(name: str) -> bool:
        if len(name) < 3 or len(word) < len(name) + 3 or sum(char in "aeiou" for char in name) > 1:
            return False
        if name[0] != word[0] or name[-1] != word[-1]:
            return False
        rest = iter(word)
        return all(char in rest for char in name)

    return [column for column, names in words.items() if any(shortens(name) for name in names)]


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
        value, length = _number(match.group("number")), match.end()
        unit = re.match(r"\s+([a-z]+)\b", after[length:])
        if unit and (converted := _in_column_unit(column, unit.group(1), value)) is not None:
            # "depth_km below 1 km", "depth_m above 50 cm": the bound's unit is the column's, or converts.
            value, length = converted, length + unit.end()
        return [Condition(column, operator, value)], length
    if (match := re.match(
        rf"\s+{LINKING_VERB}between\s+(?P<low>{NUMBER})\s+and\s+(?P<high>{NUMBER})(?![0-9a-z])", after
    )):
        low, high = sorted((_number(match.group("low")), _number(match.group("high"))))
        return [Condition(column, ">=", low), Condition(column, "<=", high)], match.end()
    if (match := re.match(rf"\s+{LINKING_VERB}(?P<sign>negative|positive|zero)\b", after)):
        return [_sign_condition(column, match.group("sign"))], match.end()
    if (match := re.match(rf"\s+(?:is|are|was|were|equals|equal\s+to)\s+{number}", after)):
        return [Condition(column, "=", _number(match.group("number")))], match.end()
    if (match := re.match(rf"\s+(?:of\s+)?{number}", after)):
        # "VMA of 0", "Pentanoic_acid 0", "Status 2": a value the column holds. A year is
        # left for the time scope when the table has a date.
        number = _number(match.group("number"))
        year = bool(roles.date) and bool(re.fullmatch(YEAR, match.group("number")))
        if not year and bool((series == number).any()):
            return [Condition(column, "=", number)], match.end()
    return None


def _in_column_unit(column: str, unit: str, value: float) -> float | None:
    """``value`` in ``unit`` as a number in the column's own unit, or None when the column's name gives
    no unit of that kind."""
    from .claims import QUANTITY_NAMES, UNITS, column_unit  # noqa: PLC0415

    if unit not in UNITS:
        return None
    own = column_unit(column)
    parts = [part.lower() for part in re.split(r"[^A-Za-z]+", str(column)) if part]
    if own is None and len(parts) >= 2 and parts[-1] == "m" and QUANTITY_NAMES["length"] & set(parts):
        # "depth_m": a length whose name ends in m is in metres.
        own = "metre"
    if own is None or UNITS[own][0] != UNITS[unit][0]:
        return None
    return value * UNITS[unit][1] / UNITS[own][1]


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
        after, before = _unclaimed_after(reading, end), _unclaimed_before(reading, start)
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
        elif is_datetime64_any_dtype(series) and (match := re.match(
            rf"\s+{LINKING_VERB}(?:{COMPARISON_PATTERN})\s+(?P<year>{YEAR})\b", after
        )):
            # "YEAR > 1977" on a date column compares its year.
            operator = next(
                operator for index, (operator, _) in enumerate(COMPARISONS) if match.group(f"op{index}")
            )
            reading.conditions.append(Condition(column, operator, float(match.group("year")), part="year"))
            claimed = (start, end + match.end())
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
        elif not numeric and not is_datetime64_any_dtype(series) and (match := TEXT_IN.search(before)):
            # "papers with COVID in the title": a word written before the text it is part of.
            chunks = _raw_chunks(reading.question, [match.group("word")])
            text = chunks[-1] if chunks else match.group("word")
            reading.conditions.append(Condition(column, "contains", text=text))
            claimed = (match.start(), end)
        elif not numeric and (found_text := _text_condition(reading, column, after)):
            condition, length = found_text
            reading.conditions.append(condition)
            claimed = (start, end + length)
        elif not numeric and (found_value := _attached_value(reading, column, end, vocabulary)):
            values, value_end = found_value
            _add_filter(reading, values, start)
            claimed = (start, value_end)
        if claimed:
            reading.consumed.add(column)
            if _scope_cued(reading, claimed[0]):
                reading.scoped.add(column)
            reading.claim(*claimed)


TEXT_MATCH = re.compile(
    r"\s+(?:(?:that|which)\s+)?(?P<how>contains?|containing|includes?|including|"
    r"(?:starts?|starting|begins?|beginning)\s+with|(?:ends?|ending)\s+with)\s+(?P<word>\S+)"
)


# A word before a text column it is part of: "had COVID in the title", 'with "war" in their name'.
TEXT_IN = re.compile(
    r"\b(?:with|had|has|have|having|mentions?|mentioned|mentioning|uses?|used|using)\s+"
    r"(?:the\s+(?:word|term)\s+)?[\"'“]?(?P<word>[^\s\"'”#]+)[\"'”]?\s+in\s+(?:the|its|their|his|her)\s+$"
)


def _text_condition(reading: _Reading, column: str, after: str) -> tuple[Condition, int] | None:
    """Part of a text written after its column: "sampleID containing Ivc", "Name starting with Ac"."""
    match = TEXT_MATCH.match(after)
    if match is None:
        return None
    how = match.group("how")
    operator = "starts" if how[0] in "sb" else "ends" if how[0] == "e" else "contains"
    chunks = _raw_chunks(reading.question, [match.group("word")])
    text = chunks[-1] if chunks else match.group("word")
    return Condition(column, operator, text=text), match.end()


def _unclaimed_after(reading: _Reading, end: int) -> str:
    """The unclaimed text right after ``end``, up to the next word something else claimed.

    A condition is read from it, so "compare Weight_kg between Probiotics 0
    and 1" is not read as Weight_kg between 0 and 1 across the name between.
    """
    rest, original = reading.remaining[end:], reading.original[end:]
    for offset, (left, right) in enumerate(zip(rest, original, strict=False)):
        if left != right:
            return rest[:offset]
    return rest


def _unclaimed_before(reading: _Reading, start: int) -> str:
    """The text before ``start`` with everything up to the last claimed word masked, positions kept."""
    rest, original = reading.remaining[:start], reading.original[:start]
    for offset in range(len(rest) - 1, -1, -1):
        if rest[offset] != original[offset]:
            return "#" * (offset + 1) + rest[offset + 1:]
    return rest


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
    index = reading.index or _name_index(dataframe.columns)
    units = {column: index.units[column] for column in roles.numeric if column in index.units}
    pattern = rf"\b(?:(?:at|after|on|for|of|with)\s+)?(?P<number>{NUMBER})\s+(?P<unit>[a-z]+)\b"
    for match in list(re.finditer(pattern, reading.remaining)):
        if reading.original[match.end("number"):match.start("unit")].strip():
            continue
        if match.group("unit") in COLUMN_WORDS:
            # "Row mean (146 columns)" counts the columns a summary is over, so "149 columns" sets nothing.
            continue
        variants = _word_variants(match.group("unit"))
        columns = [column for column, words in units.items() if words & variants]
        if len(columns) != 1:
            continue
        reading.conditions.append(Condition(columns[0], "=", _number(match.group("number"))))
        reading.scoped.add(columns[0])
        reading.claim(match.start(), match.end())


def _claim_time_points(reading: _Reading, dataframe: pd.DataFrame, roles: ColumnRoles) -> None:
    """Read "on day 21" as the one time-point column (ChickWeight's Time) at 21, when 21 is one of its values.

    It runs before the conditions, so "On day 21, chicks" does not read 21 as a chick.
    """
    _claim_lead_values(reading, dataframe, roles)
    points = [column for column in roles.numeric if column in dataframe.columns and is_timepoint_name(column)
              and column not in reading.spans]
    if len(points) != 1:
        return
    values = set(to_numbers(dataframe[points[0]]).dropna().tolist())
    pattern = rf"\b(?:(?:at|on|by|in|from)\s+)?(?:{'|'.join(POINT_WORDS)})\s+(?P<number>{NUMBER})\b"
    for match in list(re.finditer(pattern, reading.remaining)):
        number = _number(match.group("number"))
        if number in values:
            reading.conditions.append(Condition(points[0], "=", number))
            reading.scoped.add(points[0])
            reading.claim(match.start(), match.end())
            reading.notes.append(f'Read "{_raw_words(reading.question, match.group(0).split())}" as '
                                 f"{points[0]} = {format_number(number)}.")


ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
    "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12,
}


def _claim_ordinal_columns(reading: _Reading, dataframe: pd.DataFrame, roles: ColumnRoles) -> None:
    """Read "the fourth quarter" as a numeric column named quarter at 4, when 4 is one of its values.

    It runs before the conditions and the time parts, so the ordinal is not read as a rank.
    """
    names = {_norm(str(column)): column for column in roles.numeric if column in dataframe.columns}
    words = "|".join(ORDINAL_WORDS)
    pattern = rf"\b(?:the\s+)?(?P<ordinal>{words}|\d+(?:st|nd|rd|th))\s+(?P<name>[a-z]+)\b"
    for match in list(re.finditer(pattern, reading.original)):
        column = names.get(match.group("name"))
        if column is None or reading.spans.get(column, (match.start("name"),))[0] != match.start("name"):
            continue
        if not reading.remaining[match.start("ordinal"):match.end("ordinal")].strip():
            continue
        ordinal = match.group("ordinal")
        number = float(ORDINAL_WORDS.get(ordinal) or int(re.sub(r"\D", "", ordinal)))
        if number not in set(to_numbers(dataframe[column]).dropna().tolist()):
            continue
        reading.conditions.append(Condition(column, "=", number))
        reading.spans.pop(column, None)
        reading.consumed.add(column)
        reading.scoped.add(column)
        reading.claim(match.start(), match.end())
        reading.notes.append(f'Read "{_raw_words(reading.question, match.group(0).split())}" as {column} = '
                             f"{format_number(number)}.")


# Most values a column may hold for "At 14," to pick one of them.
LEAD_VALUES = 12


def _claim_lead_values(reading: _Reading, dataframe: pd.DataFrame, roles: ColumnRoles) -> None:
    """Read "at 14," as the one column of few values that holds 14 (Orthodont's age).

    Only a number after "at" that ends the clause, or is followed by years,
    months, weeks or days, is read, and only when exactly one numeric column
    of at most LEAD_VALUES values holds it.
    """
    pattern = rf"\bat\s+(?P<number>{NUMBER})(?:\s+(?:years?|months?|weeks?|days?)\b|(?=\s*$))"
    match = re.search(pattern, reading.original)
    comma = re.search(rf"\bat\s+({NUMBER})\s*,", reading.question, re.IGNORECASE)
    if not match and comma:
        match = re.search(rf"\bat\s+(?P<number>{re.escape(comma.group(1))})\b", reading.original)
    span = slice(match.start(), match.end()) if match else slice(0, 0)
    if not match or reading.remaining[span] != reading.original[span]:
        return
    number = _number(match.group("number"))
    holding = [
        column for column in roles.numeric
        if column in dataframe.columns and column not in reading.spans
        and (values := set(to_numbers(dataframe[column]).dropna().tolist()))
        and len(values) <= LEAD_VALUES and number in values
    ]
    if len(holding) != 1:
        return
    reading.conditions.append(Condition(holding[0], "=", number))
    reading.scoped.add(holding[0])
    reading.claim(match.start(), match.end())
    reading.notes.append(f'Read "{_raw_words(reading.question, match.group(0).split())}" as {holding[0]} = '
                         f"{format_number(number)}, the one column of few values that holds it.")


# Words before a number that name a time point: "day 21", "week 3", "visit 2".
POINT_WORDS = ("day", "week", "visit", "cycle", "session", "wave", "round")


# Words before a reading that ask to hold it fixed: "after accounting for time", "adjusted for age".
ADJUSTING = re.compile(
    r"\b(?:(?:after|when|while)\s+)?(?:accounting|adjusting|controlling|correcting)\s+for\s+(?:the\s+)?$"
    r"|\badjusted\s+for\s+(?:the\s+)?$"
)


def _claim_adjustment(reading: _Reading) -> None:
    """Read "after accounting for time", "adjusted for age and sex" or "holding dose fixed".

    Such a column is neither the measure nor the grouping; the comparison
    holds it fixed (see ``_execute_adjusted``).
    """
    previous: str | None = None
    for column, (start, end) in sorted(reading.spans.items(), key=lambda item: item[1][0]):
        before = reading.original[:start]
        lead = ADJUSTING.search(before)
        if lead is None and previous in reading.adjusted:
            lead = re.search(r"\b(?:and|or)\s+$", before)
        held = None
        if lead is None and (lead := re.search(r"\b(?:holding|with)\s+(?:the\s+)?$", before)):
            held = re.match(r"\s+(?:held\s+)?(?:constant|fixed)\b", reading.original[end:])
            lead = lead if held else None
        previous = column
        if lead is None:
            continue
        reading.adjusted.append(column)
        reading.consumed.add(column)
        reading.claim(lead.start(), lead.end())
        if held:
            reading.claim(end, end + held.end())


# Words before a number that make it a value rather than the column it names: "above 1", "top 5".
VALUE_LEAD = re.compile(
    rf"\b(?:{'|'.join(words for _, words in COMPARISONS)}|between|and|to|than|is|are|was|were|equals?|"
    r"equal\s+to|top|bottom|minus|at|from|since|after|before|until)\s+$"
)


def _claim_numbered_columns(reading: _Reading, roles: ColumnRoles) -> None:
    """Read a number that is a whole column name ("120") as that column, unless it stands where a value goes.

    It is read before the conditions, so "how many rows have 120 above 1" sets
    a condition on the column 120. A number after a comparison word, "top",
    "between" or a column's name ("Status 2", "VMA of 0") is a value, and a
    year is a year when the table has a date; ``_claim_numbers`` settles those.
    """
    numbered = {forms[0]: column for column, forms in reading.names.items() if _is_number(forms[0])}
    if not numbered:
        return
    for start, end, word in _tokens(reading.remaining):
        column = numbered.get(word)
        if column is None or column in reading.spans:
            continue
        if roles.date and re.fullmatch(r"(?:19|20)\d{2}", word):
            continue
        before = reading.original[:start]
        linked = re.search(r"\b(?:of|is|equals)\s+$", before)
        if VALUE_LEAD.search(before) or _after_name(reading, linked.start() if linked else start):
            continue
        reading.spans[column] = (start, end)
        reading.claim(start, end, keep_text=True)


def _typed_values(
    reading: _Reading, vocabulary: _Vocabulary, index: _NameIndex
) -> tuple[tuple[int, int], ...]:
    """Where the question types a value of two or more words in full, as the data spells it.

    Such a value is read whole before column names claim words inside it:
    "human_organs_tissues" is a value of source_table even in a table with a
    column called "organ". A value spelled like a column name is left to the name.
    """
    tokens = _tokens(reading.original)
    words = [word for _, _, word in tokens]
    spans: list[tuple[int, int]] = []
    first = 0
    while first < len(words):
        for size in range(min(MAX_VALUE_WORDS, len(words) - first), 1, -1):
            text = " ".join(words[first:first + size])
            found = vocabulary.by_text.get(text)
            if not found or text in index.spellings:
                continue
            typed = any(
                re.search(rf"(?<![0-9A-Za-z]){re.escape(vocabulary.values[position].raw)}(?![0-9A-Za-z])",
                          reading.question, re.IGNORECASE)
                for position in found
            )
            if typed:
                spans.append((tokens[first][0], tokens[first + size - 1][1]))
                first += size
                break
        else:
            first += 1
    return tuple(spans)


def _claim_rate_scale(reading: _Reading) -> None:
    """Claim the number a rate is given per, before a column: "claims per 100 holders"."""
    per = RATE_SCALE.search(reading.original)
    if per and reading.remaining[per.start("n"):per.end("n")].strip() and any(
        start == per.end() for start, _ in reading.spans.values()
    ):
        reading.scale = float(re.sub(r"[ ,]", "", per.group("n")))
        reading.claim(per.start("n"), per.end("n"), keep_text=True)


def _claim_numbers(reading: _Reading, dataframe: pd.DataFrame, roles: ColumnRoles) -> None:
    """Claim the number of a "top N", a year, and numbers that are whole column names ("1001")."""
    if (top := re.search(r"\b(?:top|bottom)\s+(?P<n>\d{1,3})\b", reading.remaining)):
        reading.top_n = top.group(0)
        reading.claim(top.start("n"), top.end("n"), keep_text=True)
    numbered = {forms[0]: column for column, forms in reading.names.items() if _is_number(forms[0])}
    for start, end, word in _tokens(reading.remaining):
        if not _is_number(word):
            continue
        is_year = _is_year(word, dataframe, roles)
        if word in numbered and not (is_year and roles.date):
            reading.spans[numbered[word]] = (start, end)
            reading.claim(start, end, keep_text=True)
        elif is_year and reading.year is None:
            reading.year = int(word)
            reading.claim(start, end, keep_text=True)


# The number a rate is given per, before the column it is per: "per 100 holders", "for every 1,000 people".
RATE_SCALE = re.compile(r"\b(?:per|for\s+every|for\s+each)\s+(?P<n>\d{1,3}(?:[ ,]\d{3})+|\d+)\s+(?=[a-z])")
RATE_WORDS = r"(?:per|for\s+every|for\s+each)(?:\s+\d{1,3}(?:[ ,]\d{3})+|\s+\d+)?"


def _ratio_plan(reading: _Reading, dataframe: pd.DataFrame, roles: ColumnRoles) -> QueryPlan | None:
    """Plan "Claims per Holders": the total of one column over the total of another.

    Both must be measurements, never negative and not rates or shares
    themselves (``metric_for``), so their totals mean something; the second
    must come right after "per", "for every" or "for each", with the number
    the rate is given per between them ("per 100 Holders"). None for other
    questions: a code ("per District"), or an average or total asked for
    ("average quantity per transect" averages within each unit).
    """
    mentioned = reading.mentioned
    after = [column for column, start in mentioned.items() if _follows(reading.text, start, RATE_WORDS)]
    if len(after) != 1:
        return None
    partner = after[0]
    others = [column for column in mentioned if column != partner]
    if len(others) != 1:
        return None
    measure = others[0]
    if any(re.search(rf"\b{word}\b", reading.unnamed) for word in AGGREGATION_WORDS):
        # "average quantity per transect" averages within each unit (_per_unit_plan).
        return None
    for column in (measure, partner):
        coded = column == roles.identifier or column in roles.dimensions or _is_code(column, dataframe, roles)
        if coded or not _is_measurement(dataframe[column]) or (
            metric_for(column, roles).unit == "rate" or (dataframe[column].dropna() < 0).any()
        ):
            return None
    return QueryPlan(
        intent="aggregate", aggregation="ratio", measure=measure, partner=partner, scale=reading.scale,
        **_scope(reading),
    )


def _execute_ratio(plan: QueryPlan, working: pd.DataFrame, scope: str, applied: list[str]) -> QueryAnswer:
    """The total of ``measure`` over the total of ``partner``, on the rows that hold both, times ``scale``."""
    assert plan.measure is not None and plan.partner is not None
    both = working[[plan.measure, plan.partner]].dropna()
    top, bottom = float(both[plan.measure].sum()), float(both[plan.partner].sum())
    if both.empty or bottom == 0:
        return QueryAnswer(
            question="",
            plan=plan,
            answer=f"{plan.partner} adds up to zero{_phrase(applied)}, so there is no rate to give.",
            calculation=f"sum({plan.partner}) is 0{scope}",
        )
    scale = plan.scale or 1.0
    value = top / bottom * scale
    per = f"per {scale:,g} {plan.partner}" if plan.scale else f"per {plan.partner}"
    formula = f"sum({plan.measure}) ÷ sum({plan.partner})" + (f" × {scale:,g}" if plan.scale else "")
    left = len(working) - len(both)
    answer = (
        f"{plan.measure} {per}{_phrase(applied)} is {value:,.6g}: {_plain_total(top)} {plan.measure} over "
        f"{_plain_total(bottom)} {plan.partner}, from {counted(len(both), 'row')}"
        + (f" ({counted(left, 'row')} missing one of them left out)" if left else "") + "."
    )
    return QueryAnswer(
        question="",
        plan=plan,
        answer=answer,
        calculation=f"{formula}{scope}",
        value=_as_number(value),
        figures=(("ratio", value), ("numerator", top), ("denominator", bottom), ("n", float(len(both)))),
    )


def _plain_total(value: float) -> str:
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.4g}"


def _is_year(word: str, dataframe: pd.DataFrame, roles: ColumnRoles) -> bool:
    """Whether a number in a question is a year: 1900 to 2099, or an earlier year the year column holds."""
    if re.fullmatch(r"(?:19|20)\d{2}", word):
        return True
    column = _year_column(dataframe, roles)
    if column is None or not re.fullmatch(r"1\d{3}", word):
        return False
    values = dataframe[column]
    years = values.dt.year if is_datetime64_any_dtype(values) else to_numbers(values)
    return bool((years == int(word)).any())


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
        # "which columns relate to VB" asks about the columns, not about the rows' labels.
        if word in QUERY_GRAMMAR or word in ENTITY_NOUNS or word in COLUMN_WORDS:
            continue
        if not word.isalpha() or len(word) < 3:
            continue
        if not _follows(reading.original, start, rf"(?:{RANKING_WORDS}|{GROUPING_WORDS})"):
            continue
        reading.spans[label] = (start, end)
        reading.claim(start, end, keep_text=True)
        reading.notes.append(f'Read "{_raw_words(reading.question, [word])}" as {label}, the row label.')
        return


def _claim_entity(
    reading: _Reading, dataframe: pd.DataFrame, roles: ColumnRoles, nouns: set[str] = frozenset()
) -> None:
    """Read a noun for the rows ("patients", "birds") as what is counted.

    The file's own noun for its rows (``roles.row_noun``, "penguins" in
    penguins.csv) is one wherever it stands: "average body mass of Gentoo penguins".
    So is a plural the values of several columns share (``nouns``, "characters"
    in "Male Characters" and "Good Characters").

    A plural that names nothing else, right before "with" or "where" and a
    condition, names the rows too: "average strength for mixes with admixture above 0".
    In a table of counts, the first plural that names nothing else names the cases
    the counts count: "female applicants" in UCBAdmissions.
    """
    counts = frequency_column(dataframe) is not None
    for start, end, word in _tokens(reading.remaining):
        if word in QUERY_GRAMMAR:
            continue
        counted = re.search(rf"{COUNT_PATTERN}\s+(?:of\s+)?(?:the\s+)?$", reading.original[:start])
        plural = word.endswith("s") and word.isalpha() and len(word) > 3
        described = plural and re.match(
            r"\s+(?:with|where|whose|having|that\s+have|which\s+have)\s+\S", reading.original[end:]
        )
        rows = word == roles.row_noun or word in nouns
        # In a table of counts the first plural noun left over names the cases counted: "female
        # applicants". A second one ("people who were cyclists") is a word the table may not hold.
        cases = plural and counts and reading.entity is None
        if word not in ENTITY_NOUNS and not rows and not (counted and plural) and not described and not cases:
            continue
        if not rows and _kind_value(word, dataframe):
            # "birds" in a shelter's table whose animal_type holds "bird" picks those rows (_claim_values).
            continue
        if not rows and (kin := _value_kin(word, dataframe)):
            # "senators" where chamber holds "senate" may mean those rows, not every row.
            reading.flag(f'"{word}" may mean the rows with {kin[0]} = {kin[1]}; name the value, as in '
                         f'"{kin[0]} {kin[1]}".')
            reading.claim(start, end)
            continue
        reading.claim(start, end)
        if reading.entity is None:
            reading.entity = word
            reading.entity_column = None if rows else _entity_column(word, dataframe, roles)
            reading.weight = None if reading.entity_column else _count_column(word, dataframe)


def _kind_value(word: str, dataframe: pd.DataFrame) -> bool:
    """Whether a noun for rows is also a kind a text column holds: "birds" where animal_type has "bird"."""
    forms = _word_variants(word)
    for column in dataframe.select_dtypes(exclude=["number", "bool", "datetime", "datetimetz"]).columns:
        series = dataframe[column]
        if _names_column(series):
            continue
        if forms & set(series.dropna().astype(str).str.lower().unique()):
            return True
    return False


def _value_kin(word: str, dataframe: pd.DataFrame) -> tuple[str, str] | None:
    """A one-word value a plural is formed from, as (column, value): "senators" from "senate", "democrats"
    from "democratic". The two share their first five letters or more and differ only in a short ending."""
    for column in dataframe.select_dtypes(exclude=["number", "bool", "datetime", "datetimetz"]).columns:
        series = dataframe[column]
        if _names_column(series) or series.nunique(dropna=True) > MAX_KIN_VALUES:
            continue
        for value in series.dropna().astype(str).unique():
            lowered = value.lower()
            shared = len(os.path.commonprefix([word, lowered]))
            if lowered.isalpha() and shared >= 5 and len(word) - shared <= 3 and len(lowered) - shared <= 2:
                return str(column), value
    return None


# Values a column may hold for a plural to be formed from one of them.
MAX_KIN_VALUES = 50


# Verbs whose noun names a column of counts: "how many homes were sold" adds up sales.
VERB_COUNTS = {
    "sold": "sales", "bought": "purchases", "purchased": "purchases", "listed": "listings",
    "booked": "bookings", "ordered": "orders", "delivered": "deliveries", "shipped": "shipments",
    "admitted": "admissions", "discharged": "discharges", "born": "births", "installed": "installs",
    "downloaded": "downloads", "registered": "registrations", "enrolled": "enrollments",
}


# Name words that say a column holds counts, so "encounter_count" counts encounters.
COUNT_NAME_WORDS = frozenset({"count", "counts", "number", "num", "n", "no", "total", "totals"})


# Words for money as a whole, and words in the name of a column of money.
MONEY_NOUNS = frozenset({
    "money", "funding", "funds", "spending", "revenue", "income", "investment", "budget",
})
AMOUNT_NAME_WORDS = frozenset({
    "amount", "value", "cost", "costs", "price", "commitment", "funding", "revenue", "spend", "spending",
    "sales", "income", "budget", "award", "payment", "paid", "eur", "usd", "gbp", "euro", "euros", "dollar",
    "dollars", "pounds",
})


def _amount_column(dataframe: pd.DataFrame) -> str | None:
    """The one column of money, by its name: current_total_commitment, sale_amount.

    Negative amounts (refunds, offsets) stay in, as they are part of the total.
    """
    found = [
        column for column in dataframe.columns
        if is_numeric_dtype(dataframe[column]) and not is_bool_dtype(dataframe[column])
        and set(_norm(normalized_name(str(column))).split()) & AMOUNT_NAME_WORDS
    ]
    return found[0] if len(found) == 1 else None


def _count_column(noun: str, dataframe: pd.DataFrame) -> str | None:
    """The one column of whole counts named for a plural: encounter_count for "encounters".

    Its name must be the plural itself ("deaths") or the noun with a word
    such as count or number ("encounter_count"), and its values whole numbers
    that are never negative. A column named with the singular alone ("Rat")
    numbers the things rather than counting them.
    """
    from .folder import singular  # noqa: PLC0415

    found = []
    for column in dataframe.columns:
        series = dataframe[column]
        if not is_numeric_dtype(series) or is_bool_dtype(series):
            continue
        named = _norm(normalized_name(str(column))).split()
        words = [word for word in named if word not in COUNT_NAME_WORDS]
        forms = {noun, singular(noun)} if len(words) < len(named) else {noun}
        values = series.dropna().to_numpy(dtype=float)
        whole = len(values) and (values >= 0).all() and (values % 1 == 0).all()
        if words and " ".join(words) in forms and whole:
            found.append(column)
    return found[0] if len(found) == 1 else None


def _claim_count_column(reading: _Reading, dataframe: pd.DataFrame, roles: ColumnRoles) -> None:
    """Read a plural naming a column of counts, after a word that counts, as what is counted.

    "how many encounters" and "share of encounters" add up encounter_count.
    The column's name must hold a word such as count, number or n; a column
    named with the plural alone ("Units") keeps its own reading. When an ID
    names each row once (a tag per cat), the rows are the things and are
    counted as they are, so n_cats is a column like any other.
    """
    ids = dataframe[roles.identifier] if roles.identifier in dataframe.columns else None
    for start, end, word in _tokens(reading.original):
        before = reading.original[:start]
        if word in MONEY_NOUNS and re.search(
            rf"\b(?:{'|'.join(SHARE_WORDS)})\s+of\s+(?:all\s+)?(?:the\s+)?$", before
        ):
            # "share of all the money" weighs each row by the one column of money.
            column = _amount_column(dataframe)
        elif ids is not None and ids.notna().all() and ids.is_unique:
            continue
        elif word.endswith("s") and re.search(rf"{COUNT_PATTERN}\s+(?:of\s+)?(?:all\s+)?(?:the\s+)?$",
                                              before):
            column = _count_column(word, dataframe)
            if column is not None and not set(_norm(normalized_name(str(column))).split()) & COUNT_NAME_WORDS:
                column = None
        elif word in VERB_COUNTS and (counted := re.search(
            rf"{COUNT_PATTERN}\s+(?:of\s+)?(?:the\s+)?(?P<noun>[a-z]+)\s+(?:were|was|are|is|have\s+been|had\s+been)"
            r"\s+$", before
        )):
            # "how many homes were sold" adds up sales, a column holding how many were sold on each row.
            column = _count_column(VERB_COUNTS[word], dataframe)
            if column is not None:
                reading.claim(*counted.span("noun"))
                reading.spans = {named: span for named, span in reading.spans.items() if span != (start, end)}
                reading.claim(start, end)
                reading.entity, reading.entity_column, reading.weight = counted.group("noun"), None, column
                return
        else:
            continue
        if column is None:
            continue
        reading.spans = {named: span for named, span in reading.spans.items() if span != (start, end)}
        reading.claim(start, end)
        reading.entity, reading.entity_column, reading.weight = word, None, column
        return


def _has_times(dates: pd.Series) -> bool:
    values = dates.dropna()
    return bool((values != values.dt.normalize()).any())


def _claim_time_part(reading: _Reading, dataframe: pd.DataFrame, roles: ColumnRoles) -> None:
    """Read "by hour", "by weekday" or "month of the year" as a grouping by that part of the date column."""
    match = TIME_PART_PATTERN.search(reading.remaining)
    if match is None:
        return
    group = next(name for name, value in match.groupdict().items() if value)
    part: TimePart = group.rstrip("_")  # type: ignore[assignment]
    reading.claim(*match.span(group))
    if not roles.date or roles.date not in dataframe.columns:
        reading.flag(f"This table has no date column, so there is no {part} to group by.")
    elif part == "hour" and not _has_times(dataframe[roles.date]):
        reading.flag(f"{roles.date} holds dates without times of day, so there is no hour to group by.")
    else:
        reading.time_part = part
        if part == "season":
            spans = _and_listed(list(SEASON_NAMES.values()))
            reading.notes.append(f"Seasons are the three-month spans {spans}.")


# Years of data from which "by month" asks for the month of the year rather than a timeline.
SEASONAL_YEARS = 2


def _claim_month_of_year(reading: _Reading, dataframe: pd.DataFrame, roles: ColumnRoles) -> None:
    """Read "by month" as the month of the year when the data spans several years.

    Over several years a breakdown by month asks for the seasonal profile, each
    month pooled across the years; the answer says so and how to ask for the
    timeline. "per month", "monthly" and any word of a trend keep the timeline.
    """
    if reading.time_part or not roles.date or roles.date not in dataframe.columns:
        return
    match = re.search(r"\b(?:by|each|across)\s+(?:the\s+)?(?P<month>months?)\b", reading.remaining)
    if match is None or any(phrase in reading.unnamed for phrase in (*TREND_WORDS, "monthly", "per month")):
        return
    dates = dataframe[roles.date].dropna()
    if dates.empty or not is_datetime64_any_dtype(dates):
        return
    years = (dates.max() - dates.min()).days / 365.25
    if years < SEASONAL_YEARS:
        return
    reading.claim(*match.span("month"))
    reading.time_part = "month"
    reading.notes.append(
        f'Read "by month" as the month of the year, each pooled across the {years:.0f} years of data; '
        'ask for the "monthly trend" for the timeline.'
    )


def _year_column(dataframe: pd.DataFrame, roles: ColumnRoles) -> str | None:
    """The column a year in the question refers to: the date column, or one numeric column of years."""
    if roles.date and roles.date in dataframe.columns:
        return roles.date
    named = [
        column for column in roles.numeric
        if column in dataframe.columns and {"year", "yr"} & set(_norm(normalized_name(column)).split())
    ]
    if named:
        return named[0] if len(named) == 1 else None
    # Chocolate ratings' review_date holds only years (2006 to 2021): a date of a year.
    dated = [
        column for column in roles.numeric
        if column in dataframe.columns and {"date", "season"} & set(_norm(normalized_name(column)).split())
        and _holds_years(dataframe[column])
    ]
    return dated[0] if len(dated) == 1 else None


def _holds_years(column: pd.Series) -> bool:
    """Whether every value of a column is a whole year from 1000 to 2100."""
    values = to_numbers(column).dropna()
    return not values.empty and bool(((values % 1 == 0) & values.between(1000, 2100)).all())


def _claim_part_of_year(reading: _Reading, roles: ColumnRoles) -> None:
    """Read part of a year ("the first half of 2021") as the days from its first to its last, before "quarter"
    can be read as a grain."""
    for match in list(PART_OF_YEAR.finditer(reading.remaining)):
        if any(start < match.end() and end > match.start() for start, end in reading.spans.values()):
            # A column called quarter holds the quarter: "the fourth quarter" is quarter = 4 there.
            continue
        parts = 2 if match.group("part").lower() == "half" else 4
        which = match.group("which").lower()
        number = parts if which in ("last", "final") else PART_ORDER[which]
        if number > parts:
            continue
        if not roles.date:
            reading.flag("This table has no date column to pick part of a year from.")
        else:
            months = 12 // parts
            first = pd.Timestamp(int(match.group("year")), (number - 1) * months + 1, 1)
            last = first + pd.DateOffset(months=months) - pd.Timedelta(days=1)
            reading.conditions.extend([
                Condition(roles.date, ">=", float((first - EPOCH).days), part="date"),
                Condition(roles.date, "<=", float((last - EPOCH).days), part="date"),
            ])
            reading.scoped.add(roles.date)
        reading.claim(*match.span())


def _claim_days(reading: _Reading, roles: ColumnRoles) -> None:
    """Read a day written out ("April 27, 2011", "2011-04-27") as the date column on that day."""
    for match in list(DAY_PATTERN.finditer(reading.remaining)):
        day = day_named(match)
        if day is None:
            continue
        if not roles.date:
            reading.flag("This table has no date column to pick a day from.")
        else:
            reading.conditions.append(Condition(roles.date, "=", float((day - EPOCH).days), part="date"))
            reading.scoped.add(roles.date)
        reading.claim(*match.span())


def _claim_year_ranges(reading: _Reading, dataframe: pd.DataFrame, roles: ColumnRoles) -> None:
    """Read ranges of years as conditions on the date ("since 1978") and "before and after 2000" as a split.

    Right after another numeric column's name, numbers within that column's
    own values are a range of it ("Elevation between 1900 and 2000"), which
    the conditions read instead.
    """
    column = _year_column(dataframe, roles)

    def own_range(match: re.Match) -> bool:
        named = next(
            (
                other for other, (_, end) in reading.spans.items()
                if end <= match.start() and not reading.original[end:match.start()].strip()
            ),
            None,
        )
        if named is None or named == column or not _is_measurement(dataframe[named]):
            return False
        values = dataframe[named].dropna()
        numbers = [float(number) for number in match.groupdict().values() if number]
        return not values.empty and all(values.min() <= number <= values.max() for number in numbers)

    if (match := SPLIT_PATTERN.search(reading.remaining)):
        if column is None:
            reading.flag("This table has no date or year column to split at a year.")
        else:
            reading.split = (column, float(match.group("year")))
        reading.claim(*match.span())
    for pattern, operator in YEAR_RANGES:
        for match in list(pattern.finditer(reading.remaining)):
            if own_range(match) or _items_of_a_year(reading, match, roles):
                continue
            if column is None:
                reading.flag("This table has no date or year column for a range of years.")
            elif operator == "range":
                low, high = sorted((float(match.group("low")), float(match.group("high"))))
                reading.conditions.extend(
                    [Condition(column, ">=", low, part="year"), Condition(column, "<=", high, part="year")]
                )
                reading.scoped.add(column)
            else:
                year = float(match.group("year"))
                reading.conditions.append(Condition(column, operator, year, part="year"))
                reading.scoped.add(column)
            reading.claim(*match.span())
    for match in list(DECADE_PATTERN.finditer(reading.remaining)):
        if column is None:
            reading.flag("This table has no date or year column for a decade.")
        else:
            low = float(match.group("decade"))
            reading.conditions.extend(
                [Condition(column, ">=", low, part="year"), Condition(column, "<=", low + 9, part="year")]
            )
            reading.scoped.add(column)
            reading.notes.append(f'Read "{match.group()}" as {low:.0f} to {low + 9:.0f}.')
        reading.claim(*match.span())


# Verbs that only date an event, right before a year: "sets released in 2020".
EVENT_VERBS = (
    "released", "published", "launched", "founded", "built", "produced", "issued", "introduced", "opened",
    "made", "created", "established", "registered", "reviewed", "held", "started", "sold", "decided",
    "filed", "heard", "ruled", "signed", "approved", "passed", "enacted", "recorded", "reported",
    "completed", "conducted", "observed", "collected", "surveyed", "caught", "born", "elected", "awarded",
)
EVENT_PATTERN = re.compile(
    rf"\b(?:{'|'.join(EVENT_VERBS)})\b(?=\s+(?:in|during|since|before|after|from|between|by)\s+{YEAR}\b)"
)
# Phrases that say where the rows are, not which: "on board", "in the data".
FILLER_PATTERN = re.compile(
    r"\b(?:on\s+board|aboard|in\s+the\s+(?:data|dataset|data\s+set|table|file|sample)|in\s+this\s+"
    r"(?:data|dataset|table|file|sample)|all\s+told|altogether|at\s+an?\s+(?:\S+\s+)?rate(?:\s+of)?|"
    r"surveyed|interviewed|polled|questioned|came\s+from|comes?\s+from)\b"
)
# A passive verb after what is counted says how the rows came about, not which: "medals were won".
PASSIVE_PATTERN = re.compile(
    r"\b(?:were|was|are|is|been|be|got|get)\s+(?:won|awarded|given|earned|handed\s+out|recorded|scored|"
    r"taken|reported|logged|filed|placed|played)\b"
)


def _claim_fillers(reading: _Reading) -> None:
    """Claim the words that date an event or say where the rows are, so they name nothing by mistake."""
    for pattern in (EVENT_PATTERN, FILLER_PATTERN, PASSIVE_PATTERN):
        for match in list(pattern.finditer(reading.remaining)):
            reading.claim(*match.span())



def _items_of_a_year(reading: _Reading, match: re.Match, roles: ColumnRoles) -> bool:
    """Whether "from 2019" right after the rows' own noun means that year: "sets from 2019" are 2019's sets.

    The year is then left for the plain year reading; "from 2019 on" and
    "since 2019" stay ranges.
    """
    text = match.group(0)
    if not text.startswith("from ") or re.search(r"\bonwards?$", text) or not roles.row_noun:
        return False
    before = reading.original[:match.start()].split()
    return bool(before) and before[-1] in _word_variants(roles.row_noun)


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
        reading.across = _ranked_readings(dataframe, roles)
        reading.claim(*match.span())
    _claim_matrix_columns(reading, dataframe, roles)


# The per-row mean ADA adds to a matrix table (see ``schema.row_summary_name``).
ROW_SUMMARY = re.compile(r"Row mean \([\d,]+ columns\)")


def _ranked_readings(dataframe: pd.DataFrame, roles: ColumnRoles) -> tuple[str, ...]:
    """The readings a "which column" question ranks, without the row mean ADA built from them."""
    readings = numeric_readings(dataframe, roles)
    return tuple(column for column in readings if not ROW_SUMMARY.fullmatch(str(column)))


def _claim_matrix_columns(reading: _Reading, dataframe: pd.DataFrame, roles: ColumnRoles) -> None:
    """Read "total reads per sample" in a matrix, whose columns are the samples, as a ranking of the columns.

    In a matrix of counts each column is one sample, so "per sample" asks for
    each column's total; the noun after the aggregation ("reads") names what
    the cells hold, and the answer says how it was read.
    """
    if roles.table_shape != "matrix" or reading.across:
        return
    match = re.search(
        r"\b(?:per|by|each|for\s+each|in\s+each|across)\s+(?:the\s+)?(?P<noun>samples?|columns?|librar(?:y|ies))\b",
        reading.remaining,
    )
    if match is None:
        return
    reading.across = _ranked_readings(dataframe, roles)
    reading.claim(*match.span())
    words = "|".join(word for word in AGGREGATION_WORDS if word not in SUPERLATIVE_WORDS)
    held = re.search(rf"\b(?:{words})\s+(?:number\s+of\s+)?(?P<noun>[a-z]{{3,}})\b", reading.remaining)
    if held and held.group("noun") not in QUERY_GRAMMAR:
        reading.claim(*held.span("noun"))
        noun = match.group("noun")
        unit = f"{noun[:-3]}y" if noun.endswith("ies") else noun.removesuffix("s")
        reading.notes.append(
            f'Read "{_raw_words(reading.question, [held.group("noun")])}" as the values in the '
            f"{len(reading.across):,} columns, one column per {unit}."
        )


def _claim_rate(reading: _Reading) -> None:
    """Read "rate" as asking for a share: "rate of target 1 by cp", "fasting_ability rate by sex".

    Only before "of" or right after a column's name, so "rate" can still name
    a column on its own ("average rate" for "Churn Rate").
    """
    for start, end, word in _tokens(reading.remaining):
        if word not in ("rate", "rates"):
            continue
        if re.match(r"\s+of\b", reading.original[end:]) or _after_name(reading, start):
            reading.rate = True
            reading.claim(start, end)


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

    It is when it follows "of", "in", a grouping or a ranking word, or a
    superlative ("average of max", "outliers in max", "top 10 genes by mean",
    "the highest max"); when it is typed exactly as the column is spelled
    and that spelling is not plain lower case ("MAX trend"); in a correlation
    ("correlation between mean and std"); when a condition follows ("std above
    2"); and when "trend" follows ("max trend"), since a timeline of the
    maximum is not something ADA builds.
    """
    superlatives = "|".join(SUPERLATIVE_WORDS)
    if _follows(reading.text, start, rf"(?:of|in|{GROUPING_WORDS}|{RANKING_WORDS}|{superlatives})"):
        return True
    name = str(column)
    if name != name.lower() and name in _raw_chunks(reading.question, reading.text[start:end].split()):
        return True
    if any(re.search(rf"\b{word}\b", reading.remaining) for word in CORRELATION_WORDS):
        return True
    after = reading.text[end:]
    condition = rf"\s+{LINKING_VERB}(?:{COMPARISON_PATTERN}|between|negative|positive|zero)\b"
    return bool(re.match(condition, after) or re.match(r"\s+(?:trend|over time|timeline)\b", after))


def _unit_word_names_rows(reading: _Reading, dataframe: pd.DataFrame, roles: ColumnRoles) -> None:
    """Read the word for the units ("rats" for a Rat column of codes) as the rows, not as a measure.

    Only when the question names another column too and the word does not
    follow a grouping word ("by Rat" still groups): "rats on diet 1 weighed",
    "the heaviest rat's weight". A plural that names a column labelling each
    row once ("the models" for Model) names the rows the same way. The word is
    then claimed, as "patients" is.
    """
    if len(reading.spans) < 2:
        return
    coded = [column for column in reading.spans if column in roles.factors or column == roles.identifier]
    unit = repeated_unit(dataframe, roles) if coded else None
    labels = [
        column for column, (start, end) in reading.spans.items()
        if reading.original[start:end].endswith("s") and not is_numeric_dtype(dataframe[column])
        and dataframe[column].is_unique
    ]
    named = unit if unit in coded else labels[0] if len(labels) == 1 else None
    if named is None or _follows(reading.original, reading.spans[named][0]):
        return
    start, end = reading.spans.pop(named)
    reading.entity, reading.entity_column = reading.original[start:end], named
    reading.claim(start, end)


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
    index = _name_index(dataframe.columns)
    reading = _Reading(question=question, original=q, text=q, remaining=q, names=index.names, index=index)
    vocabulary = _vocabulary(dataframe, roles)
    reading.spans = _mention_spans(q, index, worded=True, protected=_typed_values(reading, vocabulary, index))
    # "which month had the highest wind" names a period, not a column called month.
    period = re.search(r"\b(?:which|what)\s+(day|week|month|quarter|year)\b", q) if roles.date else None
    if period:
        reading.spans = {column: span for column, span in reading.spans.items() if span[0] != period.start(1)}
    # The file's own noun for its rows ("flights" in flights.csv) names the rows, not a
    # column spelled like its singular (flight, a flight number).
    if roles.row_noun:
        reading.spans = {
            column: span for column, span in reading.spans.items() if q[span[0]:span[1]] != roles.row_noun
        }
    _claim_country(reading, vocabulary)
    _claim_part_of_year(reading, roles)
    _unit_word_names_rows(reading, dataframe, roles)
    _claim_count_column(reading, dataframe, roles)
    _outcome_columns(reading, dataframe)
    for start, end in reading.spans.values():
        reading.claim(start, end, keep_text=True)
    _reread_aggregation_names(reading)
    _claim_numbered_columns(reading, roles)
    _claim_adjustment(reading)
    _claim_time_part(reading, dataframe, roles)
    _claim_month_of_year(reading, dataframe, roles)
    _claim_days(reading, roles)
    _claim_ages(reading, dataframe, roles)
    _claim_year_ranges(reading, dataframe, roles)
    _claim_fillers(reading)
    _claim_rate_scale(reading)
    _claim_time_points(reading, dataframe, roles)
    _claim_ordinal_columns(reading, dataframe, roles)
    _claim_conditions(reading, list(reading.spans), dataframe, roles, vocabulary)
    _claim_person_names(reading, dataframe)
    nouns = reading.nouns = _value_nouns(vocabulary)
    _claim_values(reading, vocabulary, dataframe, exact_only=True, row_noun=roles.row_noun, nouns=nouns)
    _claim_sexes(reading, vocabulary, nouns)
    _claim_outcomes(reading, vocabulary)
    _claim_demonyms(reading, vocabulary)
    _claim_us_states(reading, vocabulary)
    _claim_analysis_words(reading, dataframe, roles)
    _claim_rate(reading)
    partial = _claim_partial_names(reading, dataframe, roles)
    _claim_conditions(reading, partial, dataframe, roles, vocabulary)
    _claim_values(reading, vocabulary, dataframe, exact_only=False, row_noun=roles.row_noun, nouns=nouns)
    _claim_codes(reading, dataframe, roles)
    _claim_unit_conditions(reading, dataframe, roles)
    _claim_numbers(reading, dataframe, roles)
    _claim_row_label(reading, roles)
    _claim_entity(reading, dataframe, roles, nouns)
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


class _ComparisonRefused(Exception):
    """A comparison the question asks for that no plan can make as asked; the message says why."""


def _plan_question(
    question: str, dataframe: pd.DataFrame, roles: ColumnRoles, rows_not_analyzed: int = 0
) -> tuple[QueryPlan | None, str | None]:
    """The plan for a question, or None with the reason it was refused when there is one."""
    if not _norm(question):
        return None, None
    try:
        compared = _comparison_plan(question, dataframe, roles)
    except _ComparisonRefused as refused:
        return None, str(refused)
    if compared is None and unsupported_phrasing(question, dataframe.columns):
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
    weighted = ()
    if reading.weight and plan.aggregation == "count" and not plan.measure and not plan.count_column:
        # "how many encounters" adds up encounter_count: each row holds a count, not one encounter.
        weighted = (f'Read "{reading.entity}" as the sum of {reading.weight}, since each row holds a count.',)
        if plan.intent == "count" or plan.counted:
            plan = replace(plan, weight=reading.weight)
        else:
            # The trend and the change keep the column of counts named, so its totals read as counts.
            counts = reading.weight if plan.intent in ("trend", "growth") else None
            plan = replace(plan, measure=reading.weight, aggregation="sum", weight=counts)
    # In a table of counts each row stands for its count of cases, which the answer says itself.
    entity = () if frequency_column(dataframe) or weighted else _entity_note(reading, plan)
    notes = (*plan.notes, *reading.notes, *entity, *weighted, *_weather_note(reading, plan))
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

    if (ratio := _ratio_plan(reading, dataframe, roles)) is not None:
        return ratio
    if (per_unit := _per_unit_plan(reading, dataframe, roles)) is not None:
        return per_unit
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
    if reading.split is not None:
        if dimension:
            reading.refusal = _two_groupings([dimension, reading.split[0]])
            return None
        return _split_plan(reading, dataframe, roles, measure)
    time_part = reading.time_part
    by_year = re.search(r"\b(?:by|per|each|every|and)\s+(?:the\s+)?(?:years?|yearly)\b", unnamed)
    if dimension and roles.date and (time_part or by_year) and dimension != roles.date:
        # "count of crop by year", "average PM2.5 by station and month of the year": a cross-table.
        pair = (roles.date, dimension)
        return _crosstab_plan(reading, dataframe, roles, pair, time_part or "year", measure)
    if time_part:
        dimension = roles.date
    # "revenue by order date" asks for the trend; with a segment as well it asks for two groupings.
    by_date = roles.date in mentioned and _follows(q, mentioned[roles.date], GROUPING_WORDS)
    if by_date and dimension:
        if preferred_frequency(dataframe[roles.date]) == "Y":
            # A date of whole years ("count of crop by Year") groups like a year.
            return _crosstab_plan(reading, dataframe, roles, (roles.date, dimension), "year", measure)
        reading.refusal = _two_groupings([dimension, str(roles.date)])
        return None
    # A continuous grouping is split at its quartiles: "average growth by temperature".
    bins = (
        4 if dimension and not time_part and _is_measurement(dataframe[dimension])
        and dimension not in (*roles.dimensions, roles.identifier)
        and not _can_group_numeric(dataframe[dimension]) else None
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

    if (wants_count or reading.rate) and not wants_growth:
        within = reading.rate or bool(re.search(rf"\b(?:{'|'.join(SHARE_WORDS)})\b", intent_text))
        ranked = bool(top_match or superlative)
        over_time = bool((grain or by_date) and roles.date and not dimension)
        if within and ((dimension and (wants_breakdown or time_part or ranked)) or over_time):
            # "share of Result 1 by Gender" is the share within each Gender, not the split of Result 1 rows.
            share = _share_plan(
                reading, dataframe, dimension or str(roles.date), measure, base,
                intent="rank" if ranked and dimension else "breakdown", time_part=time_part,
                grain=grain if over_time else None, top_n=int(top_match.group(2)) if top_match else None,
                ascending=bool(re.search(ASCENDING_PATTERN, intent_text)),
            )
            if share is not None:
                return share
        if reading.rate:
            reading.refusal = (
                "A rate is the share of rows that meet a condition, and the question names none; ask as in "
                '"rate of <yes/no column> by <group>" or "rate of <column> above <number> by <group>".'
            )
            return None
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
        if year and not month and not grain and (column := _year_column(dataframe, roles)):
            # "rose in 2020" compares 2020 with the year before, not the last two periods inside 2020.
            years = (Condition(column, ">=", float(year - 1), part="year"),
                     Condition(column, "<=", float(year), part="year"))
            base = {**base, "year": None, "conditions": (*base["conditions"], *years)}
            reading.notes.append(f"Read a change in {year} as {year} against {year - 1}.")
        if wants_count and not measure:
            # "the number of earthquakes rose": with no measure named, the change is in the row count.
            return QueryPlan(
                intent="growth", aggregation="count", dimension=rank_dimension, ascending=ascending,
                top_n=top_n, **{**base, "measure": None},
            )
        return QueryPlan(
            intent="growth", aggregation=default_aggregation, dimension=rank_dimension,
            ascending=ascending, top_n=top_n, **base,
        )

    if period and superlative:
        # "which month had the most rows" and "which day had the most earthquakes" rank row counts: with
        # no measure named, "most" counts the rows rather than ranking a measure the question never names.
        counts = re.search(r"\b(?:rows|most|fewest|busiest|quietest)\b", intent_text)
        period_measure = None if not measure and counts else base["measure"]
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
        # "which boroughs have the most rescues": with no measure named, "most" counts each group's rows,
        # as it does for periods.
        # A verb of selling or spending ("which product sells the fewest") is about the measure.
        selling = re.search(r"\b(?:sells?|sold|earns?|earned|spends?|spent|costs?|pays?|paid)\b", intent_text)
        counted = not measure and not selling and bool(
            re.search(r"\b(?:most|fewest|busiest|quietest)\b", intent_text)
        )
        ranked = {**base, "measure": None} if counted else base
        usual = "count" if counted else default_aggregation
        grouped = _grouped_aggregation(aggregation, explicit_extreme, usual)
        if grouped == "mean" and (share := _flag_share(
            reading, dataframe, dimension, base, intent="rank", top_n=top_n, ascending=ascending,
            time_part=time_part,
        )):
            return share
        return QueryPlan(
            intent="rank",
            aggregation=grouped,
            dimension=dimension,
            top_n=top_n,
            ascending=ascending,
            time_part=time_part,
            **ranked,
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
        grouped = _grouped_aggregation(aggregation, explicit_extreme, default_aggregation)
        if grouped == "mean" and (share := _flag_share(
            reading, dataframe, dimension, base, intent="breakdown", time_part=time_part,
        )):
            return share
        return QueryPlan(
            intent="breakdown",
            aggregation=grouped,
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


def _per_unit_plan(reading: _Reading, dataframe: pd.DataFrame, roles: ColumnRoles) -> QueryPlan | None:
    """Plan "average quantity per transect by protection_status": each unit's total, compared across groups.

    The column after "per" is the unit each value is first combined within;
    the column after "by" groups those units. None for other questions.
    """
    q, mentioned = reading.text, reading.mentioned
    unit = next((column for column, start in mentioned.items() if _follows(q, start, "per")), None)
    if unit is None:
        unit = _compound_unit(reading, dataframe, roles)
        return _unit_average_plan(reading, dataframe, roles, unit) if unit else None
    groupings = [
        column for column, start in mentioned.items()
        if column != unit and _follows(q, start, r"(?:by|across|among|between)")
    ]
    if not groupings:
        return _unit_average_plan(reading, dataframe, roles, unit)
    if len(groupings) != 1 or not _groups_rows(groupings[0], dataframe, roles):
        return None
    grouping = groupings[0]
    measures = [column for column in mentioned if column not in (unit, grouping)]
    if len(measures) > 1 or (measures and not _is_measurement(dataframe[measures[0]])):
        return None
    measure = measures[0] if measures else roles.measure
    if measure is None or measure in (unit, grouping):
        return None
    if _unit_keys(dataframe, unit, grouping, roles) is None:
        reading.refusal = (
            f"{_two_groupings([unit, grouping]).removesuffix('.')}: {grouping} changes within a {unit}, "
            f"and no other column holds it fixed for each {unit}, so the {unit}s can't be grouped by it."
        )
        return None
    aggregation = next(
        (AGGREGATION_WORDS[word] for word in AGGREGATION_WORDS if re.search(rf"\b{word}\b", reading.unnamed)),
        "mean",
    )
    return QueryPlan(
        intent="breakdown", aggregation=_grouped_aggregation(aggregation, False, "mean"), measure=measure,
        dimension=grouping, per=(unit,), **_scope(reading),
    )


# A key after "per" with more values than a grouping may have is a unit to combine within ("per
# order", 99,440 orders), not a grouping to list ("per site").
MIN_PER_UNITS = GROUPING_LIMIT + 1
# Words asking for one figure over the units: "average order value", not each order's.
UNIT_AVERAGE_WORDS = ("average", "mean", "median", "typical")


def _compound_unit(reading: _Reading, dataframe: pd.DataFrame, roles: ColumnRoles) -> str | None:
    """The key named right before a measure, as in "average order value": order_id, then payment_value.

    Only a key (the identifier, or a column named like one) with MIN_PER_UNITS
    or more values counts; its name and the measure's must be next to each other.
    """
    spans = reading.spans
    for column, (_, end) in spans.items():
        if column != roles.identifier and not is_identifier_name(str(column)):
            continue
        following = [
            other for other, (start, _) in spans.items()
            if other != column and start == end + 1 and _is_measurement(dataframe[other])
        ]
        word = re.match(r"\s([a-z]+)\b", reading.original[end:])
        generic = word is not None and word.group(1) in MEASURE_WORDS and roles.measure is not None
        if following or generic:
            return column
    return None


# Words that stand for the table's measure after a key: "order value", "order amount".
MEASURE_WORDS = frozenset({"value", "values", "amount", "amounts", "total", "totals", "size", "spend"})


def _unit_average_plan(
    reading: _Reading, dataframe: pd.DataFrame, roles: ColumnRoles, unit: str
) -> QueryPlan | None:
    """Plan "average payment_value per order_id": each order's total, then the average over orders.

    Only for a key (the identifier, or a column named like one) with more values
    than a grouping may have (MIN_PER_UNITS) that repeats across rows, so listing
    it would make no sense, and an aggregation word that asks for one figure.
    None otherwise, and the question is read as a breakdown.
    """
    values = dataframe[unit].dropna()
    keyed = unit == roles.identifier or is_identifier_name(str(unit))
    if not keyed or values.nunique() < MIN_PER_UNITS or not values.duplicated().any():
        return None
    asked = next(
        (word for word in UNIT_AVERAGE_WORDS if re.search(rf"\b{word}\b", reading.unnamed)), None
    )
    if asked is None:
        return None
    measures = [column for column in reading.mentioned if column != unit]
    if len(measures) > 1 or (measures and not _is_measurement(dataframe[measures[0]])):
        return None
    measure = measures[0] if measures else roles.measure
    if measure is None or measure == unit:
        return None
    aggregation = "median" if asked == "median" else "mean"
    return QueryPlan(
        intent="aggregate", aggregation=aggregation, measure=measure, per=(unit,), **_scope(reading)
    )


def _execute_unit_average(plan: QueryPlan, working: pd.DataFrame, roles: ColumnRoles, scope: str,
                          applied: list[str]) -> QueryAnswer:
    """Combine the measure within each unit (a total, or a mean for a level), then average the units."""
    assert plan.measure is not None
    unit = plan.per[0]
    data = working.dropna(subset=[unit, plan.measure])
    additive = metric_for(plan.measure, roles).additive
    grouped = data.groupby(unit, observed=True)[plan.measure]
    per_unit = grouped.sum() if additive else grouped.mean()
    if per_unit.empty:
        return QueryAnswer(
            question="", plan=plan, answer=f"No {unit} has {plan.measure} recorded{_phrase(applied)}.",
            calculation=f"0 {unit} values with {plan.measure}{scope}",
        )
    value = float(per_unit.median() if plan.aggregation == "median" else per_unit.mean())
    measured = data[plan.measure]
    per_row = float(measured.median() if plan.aggregation == "median" else measured.mean())
    combined, combine = ("total", "sum") if additive else ("average", "mean")
    label = "median" if plan.aggregation == "median" else "average"
    shown = format_number(value, plan.measure, column_values=data[plan.measure])
    row_shown = format_number(per_row, plan.measure, column_values=data[plan.measure])
    answer = (
        f"The {label} {plan.measure} per {unit}{_phrase(applied)} is {shown}: each {unit}'s {combined} "
        f"{plan.measure} ({len(per_unit):,} {unit} values, from {len(data):,} rows), then the {label} of "
        "those. "
        f"Taken row by row instead, the {label} is {row_shown}, because one {unit} can have several rows."
    )
    calculation = (
        f"{combine}({plan.measure}) per {unit}, then {plan.aggregation} over the {len(per_unit):,} {unit} "
        f"values{scope}"
    )
    return QueryAnswer(
        question="", plan=plan, answer=answer, calculation=calculation, value=_as_number(value),
        figures=(("units", float(len(per_unit))), ("n", float(len(data))), ("per_row", per_row)),
    )


def _unit_keys(frame: pd.DataFrame, unit: str, grouping: str, roles: ColumnRoles) -> list[str] | None:
    """The columns that pick out one unit: the unit alone, or with the column its numbers restart within.

    Transects numbered 1 to 6 on every reef are 88 transects, not 6: the unit
    is the reef and the transect together. That column is the one that keeps
    the grouping fixed within each unit and splits the rows into the most
    units, among those with repeated rows; a name beats a numeric code on a tie.
    None when no column keeps the grouping fixed within each unit.
    """
    present = frame.dropna(subset=[unit, grouping])
    if bool((present.groupby(unit, observed=True)[grouping].nunique() <= 1).all()):
        return [unit]
    best: tuple[int, bool, str] | None = None
    for column in present.columns:
        if column in (unit, grouping, roles.date):
            continue
        series = present[column]
        numeric = _is_measurement(series)
        if is_datetime64_any_dtype(series) or (numeric and not bool((series.dropna() % 1 == 0).all())):
            continue
        if not 2 <= series.nunique() <= len(present) / 2:
            continue
        pairs = present[[column, unit]].dropna().drop_duplicates()
        if len(pairs) > len(present) / 2:
            continue
        if int(present.groupby([column, unit], observed=True)[grouping].nunique().max()) > 1:
            continue
        candidate = (len(pairs), not numeric, column)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return [best[2], unit] if best else None


def _execute_per_unit(
    plan: QueryPlan, working: pd.DataFrame, roles: ColumnRoles, scope: str, applied: list[str]
) -> QueryAnswer:
    """Combine the measure within each unit, then compare the units across the groups."""
    assert plan.measure is not None and plan.dimension is not None
    unit = plan.per[0]
    keys = _unit_keys(working, unit, plan.dimension, roles)
    if keys is None:
        return QueryAnswer(
            question="", plan=plan,
            answer=(
                f"{plan.dimension} changes within a {unit}, and no other column holds it fixed for each "
                f"{unit}, so the {unit}s can't be grouped by {plan.dimension}."
            ),
            calculation=f"{plan.dimension} must be one value within each {unit}{scope}",
        )
    data = working.dropna(subset=[*keys, plan.dimension])
    additive = metric_for(plan.measure, roles).additive
    grouped = data.groupby(keys, observed=True)
    values = grouped[plan.measure].sum(min_count=1) if additive else grouped[plan.measure].mean()
    units = pd.DataFrame({plan.dimension: grouped[plan.dimension].first(), plan.measure: values})
    units = units.reset_index(drop=True)
    result = _execute_grouping(
        replace(plan, per=(), filters=(), conditions=(), year=None, month=None), units,
        replace(roles, identifier=None, date=None), "", applied,
    )
    combined = "total" if additive else "average"
    named = " and ".join(keys)
    # A suggestion to ask for the median keeps the unit the answer is per.
    asked = result.answer.replace(
        f'"median {plan.measure} by {plan.dimension}"',
        f'"median {plan.measure} per {unit} by {plan.dimension}"',
    )
    answer = (
        f"{asked} Each value is the {combined} {plan.measure} of one {unit} ({named} together, "
        f"{len(units):,} in all), so the {plan.dimension} groups are compared {unit} by {unit}, not "
        "record by record."
    )
    combine = "sum" if additive else "mean"
    calculation = f"{combine}({plan.measure}) per {named}, then {result.calculation}{scope}"
    return replace(result, plan=plan, answer=answer, calculation=calculation)


def _split_plan(
    reading: _Reading, dataframe: pd.DataFrame, roles: ColumnRoles, measure: str | None
) -> QueryPlan | None:
    """Plan "average DO before and after 2000": the rows before the year against those from it on."""
    assert reading.split is not None
    column, at = reading.split
    measure = measure or roles.measure
    if measure is None or not _is_measurement(dataframe[measure]):
        reading.refusal = (
            'A split at a year compares a numeric measure, as in "average <measure> before and after 2000".'
        )
        return None
    scope = _scope(reading)
    if _is_flag(dataframe[measure]) and outcome_event(dataframe[measure]) is not None:
        return QueryPlan(
            intent="compare", aggregation="mean", dimension=column, split=at,
            counted=(_event(measure, dataframe[measure]),), notes=(_rate_note(measure),), **scope,
        )
    return QueryPlan(
        intent="compare", aggregation="mean", measure=measure, dimension=column, split=at, **scope
    )


def _split_labels(series: pd.Series, at: float) -> pd.Series:
    """Each row's side of a split: "before 2000" or "2000 onwards" for a date, by value otherwise."""
    dated = is_datetime64_any_dtype(series)
    values = series.dt.year.astype(float) if dated else to_numbers(series)
    shown = _number_text(at)
    if dated:
        before, after = f"the years before {shown}", f"{shown} and later"
    else:
        before, after = f"rows below {shown}", f"rows at {shown} or above"
    labels = pd.Series(np.where(values < at, before, after), index=series.index, dtype=object)
    return labels.where(values.notna()).astype(pd.CategoricalDtype([before, after], ordered=True))


def _event(column: str, series: pd.Series) -> ValueFilter | Condition:
    """What a yes/no column counts: the rows where it is 1, True or "yes"."""
    event = outcome_event(series)
    if is_numeric_dtype(series) or is_bool_dtype(series):
        return Condition(column, "=", float(event))
    return ValueFilter(column=column, values=(str(event),))


def _share_plan(
    reading: _Reading, dataframe: pd.DataFrame, dimension: str, measure: str | None, scope: dict, **fields
) -> QueryPlan | None:
    """Plan the share of each group's rows that meet a condition: "share of Result 1 by Gender".

    What is counted is the condition or value the question names without a cue
    such as "for" or "where" ("share of Result 1", "rows with pH below 5");
    those with a cue narrow the rows the shares are of. With none named, the
    yes/no column the question names is counted where it is yes ("share of
    fasting_ability by ethnic_group"). Returns None when nothing says what to count.
    """
    filters = [item for item in reading.value_filters if item.column != dimension]
    stated = [*filters, *reading.conditions]
    counted = [item for item in stated if item.column not in reading.scoped] or stated
    if not counted:
        candidates = [measure] if measure else [column for column in reading.mentioned if column != dimension]
        outcomes = [
            column for column in candidates
            if column and column != dimension and outcome_event(dataframe[column]) is not None
        ]
        if len(outcomes) != 1:
            return None
        counted = [_event(outcomes[0], dataframe[outcomes[0]])]
    return QueryPlan(
        aggregation="share", dimension=dimension, counted=tuple(counted),
        filters=tuple(item for item in filters if item not in counted),
        conditions=tuple(item for item in reading.conditions if item not in counted),
        year=scope["year"], month=scope["month"], **fields,
    )


def _flag_share(
    reading: _Reading, dataframe: pd.DataFrame, dimension: str, base: dict, **fields
) -> QueryPlan | None:
    """The share plan for the average of a 0/1 column by a grouping, which is the share of rows where it is 1.

    A yes/no outcome is answered as a rate per group with a chi-squared or
    Fisher test, as the brief's rate card does, not as means with a t-test.
    """
    measure = base["measure"]
    if not measure or base["bins"] or not _is_flag(dataframe[measure]):
        return None
    if outcome_event(dataframe[measure]) is None:
        return None
    return QueryPlan(
        aggregation="share", dimension=dimension, counted=(_event(measure, dataframe[measure]),),
        filters=base["filters"], conditions=base["conditions"], year=base["year"], month=base["month"],
        notes=(_rate_note(measure),), **fields,
    )


def _rate_note(measure: str) -> str:
    return (
        f"{measure} holds only 0 and 1, so the groups are compared by the share of rows where it is 1 "
        "(its average) with a test for rates."
    )


def _count_plan(
    reading: _Reading, dataframe: pd.DataFrame, roles: ColumnRoles, ungrouped: dict
) -> QueryPlan | None:
    """Plan "how many": distinct values of a named column, the rows meeting the conditions, or a total.

    "how many cities" counts distinct cities and "how many stations" a numbered
    station column. A quantity counts by its total ("how many units"); a
    measurement such as a temperature has no count, so it is refused.
    """
    counting = {**ungrouped, "measure": None}
    picked = {item.column for item in counting["filters"]}
    if counting["year"]:
        picked.add(_year_column(dataframe, roles))
    for column, start in reading.mentioned.items():
        if column in picked and not reading.distinct:
            # "what share have blue eyes", "how many reviews in 2019": the column picks the rows, and
            # its distinct values among them are the one picked.
            continue
        series = dataframe[column]
        if _is_flag(series) and not reading.distinct and column != roles.identifier:
            # "share of Antibiotics" counts the rows where the 0/1 flag is set.
            flagged = Condition(column, "=", 1.0)
            return QueryPlan(
                intent="count", aggregation="count",
                **{**counting, "conditions": (*counting["conditions"], flagged)},
            )
        event = None if is_numeric_dtype(series) else outcome_event(series)
        if event is not None and not reading.distinct and column != roles.identifier:
            # "what share survived" counts the rows where Survived is "Yes", as a 0/1 flag counts its 1s;
            # "what share of women survived" counts it among the women.
            answered = ValueFilter(column, (str(event),))
            if _asks_share(reading) and (counting["filters"] or counting["conditions"]):
                return QueryPlan(intent="count", aggregation="count", counted=(answered,), **counting)
            return QueryPlan(
                intent="count", aggregation="count",
                **{**counting, "filters": (*counting["filters"], answered)},
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
    unit = repeated_unit(dataframe, roles) if reading.entity and not reading.entity_column else None
    if unit and reading.entity not in ROW_NOUNS and reading.entity != roles.row_noun and (
        reading.entity not in reading.nouns and not reading.weight
    ):
        # Loblolly's 84 rows are 14 trees measured at 6 ages; Portal's rows are captures at 24 plots. The
        # data can't say which the noun means.
        each = dataframe[unit].value_counts()
        rows = f"{each.min():,}" if each.min() == each.max() else f"{each.min():,} to {each.max():,}"
        reading.refusal = (
            f"The {reading.entity} may be the rows or the {unit} values, which repeat ({rows} rows each), "
            f'and no column names the {reading.entity}. Ask "how many rows" or "how many distinct {unit}".'
        )
        return None
    based = _share_base(reading, counting) if not reading.entity_column else None
    if based:
        # "what share of Adelie are female": the Adelie are the base, female is counted.
        base, counted = based
        return QueryPlan(intent="count", aggregation="count", counted=counted, **{**counting, **base})
    return QueryPlan(intent="count", aggregation="count", count_column=reading.entity_column, **counting)


SHARE_ANCHOR = re.compile(rf"\b(?:{'|'.join(SHARE_WORDS)}|rate)\s+of\b")
# The verb that ends what a share is of: "share of Adelie | are female", "of passengers | who survived".
LINK_VERB = re.compile(r"\b(?:are|is|were|was|have|has|had|got|did|do|does|who|that|which|whose)\b")
# The words that open a clause on the base: "victims who died", "Adelie that are female".
RELATIVE_WORDS = ("who", "that", "which", "whose")
# Past tenses that do not end in "ed": "won", "lost".
PAST_FORMS = frozenset({
    "won", "lost", "left", "fell", "rose", "grew", "ran", "met", "got", "went", "came", "made",
})


def _asks_share(reading: _Reading) -> bool:
    return bool(SHARE_ANCHOR.search(reading.original))


def _share_base(reading: _Reading, counting: dict) -> tuple[dict, tuple] | None:
    """Split a share's filters into the rows it is of and what it counts, at a verb after "share of".

    Values and conditions before the verb make the base; those after are
    counted. The verb is the sentence's own one when a clause on the base
    comes first: in "share of victims who died were men" the base is the
    victims who died, while in "share of victims were men who died" the men
    who died are counted. Otherwise it is the first verb, as in "share of
    Adelie that are female". A verb after "and" or "or" joins what is counted
    and never splits. None when the question asks no share, or one side is empty.
    """
    anchor = SHARE_ANCHOR.search(reading.original)
    if anchor is None:
        return None
    text = reading.original
    verbs = [
        verb for verb in LINK_VERB.finditer(text, anchor.end())
        if not re.search(r"\b(?:and|or)\s*$", text[:verb.start()])
    ]
    main = [
        verb for verb in verbs
        if verb.group() not in RELATIVE_WORDS and not re.search(rf"\b(?:{'|'.join(RELATIVE_WORDS)})\s+$",
                                                                text[:verb.start()])
    ]
    items = [*counting["filters"], *counting["conditions"]]

    def place(item: ValueFilter | Condition) -> int | None:
        if isinstance(item, ValueFilter):
            return reading.starts.get(item.column)
        span = reading.spans.get(item.column)
        return span[0] if span else None

    places = {id(item): place(item) for item in items}
    starts = [verb.start() for verb in [*main[:1], *verbs[:1]]]
    if not verbs:
        # "share of male victims died": with no verb to split at, a value written as a past tense
        # ("died", "won") after the others is what is counted.
        for item in sorted(items, key=lambda item: places[id(item)] or -1)[1:]:
            word = re.match(r"[a-z]+", text[places[id(item)]:]) if places[id(item)] is not None else None
            if word and (word.group().endswith("ed") or word.group() in PAST_FORMS):
                starts.append(places[id(item)])
                break
    for start in starts:
        counted = tuple(item for item in items if (places[id(item)] or -1) >= start)
        if counted and len(counted) < len(items):
            base = {
                "filters": tuple(item for item in counting["filters"] if item not in counted),
                "conditions": tuple(item for item in counting["conditions"] if item not in counted),
            }
            return base, counted
    return None


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
    elif plan.year and (column := _year_column(dataframe, roles)):
        used.add(column)
    return used


# Aggregations that treat a column as a quantity, which a column numbering units is not.
UNIT_AGGREGATIONS = frozenset({"mean", "sum", "median", "max", "min", "std", "var"})


def _unused_part(
    plan: QueryPlan, reading: _Reading, dataframe: pd.DataFrame, roles: ColumnRoles, rows_not_analyzed: int
) -> str | None:
    """Why the plan answers less than the question asks, or None when it uses every part."""
    summarised = plan.intent in ("aggregate", "rank", "compare") and plan.aggregation in UNIT_AGGREGATIONS
    if plan.measure and summarised and plan.measure == (unit := repeated_unit(dataframe, roles)):
        # "the heaviest rat" with a Rat column of 16 codes: the codes number the rats and measure nothing.
        return (
            f"{unit} numbers the {int(dataframe[unit].nunique()):,} units the rows belong to, so it is "
            "not a quantity to summarise; name the measure."
        )
    used = _plan_columns(plan, dataframe, roles)
    for column in reading.spans:
        if column not in used:
            return _unused_column_reason(column, reading, dataframe, rows_not_analyzed)
    planned = {item.column for item in (*plan.filters, *plan.counted) if isinstance(item, ValueFilter)}
    for item in reading.value_filters:
        if item.column not in planned:
            return (
                f"{_listed(list(item.values))} is a value of {item.column}, which the answer "
                "groups by, so it can't also filter the rows."
            )
    grouped = {plan.dimension, plan.second_dimension, *plan.per}
    for item in plan.filters:
        span = reading.spans.get(item.column)
        if span and item.column not in grouped and _follows(reading.original, span[0]):
            # "by Soil_Type_Gravel", once one-hot columns are read as one Soil_Type, names a value.
            return (
                f"{_listed(list(item.values))} is a value of {item.column}, so the answer would keep only "
                f'those rows rather than group them; ask "by {item.column}" to compare every {item.column}.'
            )
    if set(reading.conditions) - set(plan.conditions) - set(plan.counted):
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
    if reading.split and plan.split is None:
        return f"The answer would leave out the split at {_number_text(reading.split[1])}."
    if reading.adjusted and plan.adjusted_for != tuple(reading.adjusted):
        return (
            f"The answer would leave out holding {_and_listed(reading.adjusted)} fixed; ask as in "
            f'"does <measure> differ by <group> after accounting for {reading.adjusted[0]}".'
        )
    if reading.time_part and plan.time_part != reading.time_part:
        return f"The answer would leave out the {reading.time_part} grouping."
    entity_at = reading.original.find(reading.entity) if reading.entity else -1
    if entity_at >= 0 and not plan.dimension and _follows(reading.original, entity_at):
        # "average growth per patient" asks for a grouping no column provides.
        return f'Grouping by "{reading.entity}" needs a column that names each one, and ADA found none.'
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
        column = _year_column(working, roles)
        if column is None or (column != roles.date and plan.month):
            # Report the scope as unavailable rather than answer with the all-time figure.
            raise TimeScopeUnavailable(_when_label(plan))
        if column != roles.date:
            # A column of years, such as OWID's year from 1750 on, scopes a year as a date would.
            working = working.loc[to_numbers(working[column]) == plan.year]
            applied.append(f"{column} = {plan.year}")
        else:
            dates = working[roles.date]
            if plan.year:
                working = working.loc[dates.dt.year == plan.year]
                dates = working[roles.date]
            if plan.month:
                working = working.loc[dates.dt.month == plan.month]
            applied.append(f"{roles.date} in {_when_label(plan)}")
    for excluded in plan.excluded:
        if excluded.column in working.columns:
            working = working.loc[~_value_mask(working[excluded.column], excluded.values)]
    return working, applied


def _value_mask(series: pd.Series, values: tuple[str, ...]) -> pd.Series:
    """Rows whose value is one of ``values``; numbers compare as numbers, so "2" matches 2.0."""
    if is_numeric_dtype(series) and not is_bool_dtype(series):
        wanted = to_numbers(pd.Series(list(values), dtype=object)).dropna()
        if len(wanted) == len(values):
            return series.isin(wanted.tolist())
    return series.astype(str).isin(values)


def _condition_mask(series: pd.Series, condition: Condition) -> pd.Series:
    if condition.operator == "missing":
        return series.isna()
    if condition.operator in TEXT_OPERATORS:
        text = series.astype("string").str.casefold()
        wanted = str(condition.text).casefold()
        found = {
            "contains": text.str.contains(wanted, regex=False), "starts": text.str.startswith(wanted),
            "ends": text.str.endswith(wanted),
        }[condition.operator]
        return found.fillna(False).astype(bool)
    if is_datetime64_any_dtype(series) and condition.part == "date":
        # "on April 27, 2011": the day, as days since 1970, whatever the time of day.
        days = series.dt.tz_localize(None) if series.dt.tz is not None else series
        values = (days.dt.normalize() - EPOCH).dt.days.astype(float)
    elif is_datetime64_any_dtype(series):
        # A condition on a date compares its year: "since 1978".
        values = series.dt.year.astype(float)
    else:
        values = to_numbers(series)
    compare = {"=": values.eq, "<": values.lt, "<=": values.le, ">": values.gt, ">=": values.ge}
    return compare[condition.operator](condition.value)


def _number_text(value: float | None) -> str:
    number = float(value or 0.0)
    return f"{int(number)}" if number.is_integer() else f"{number:g}"


def _describe_filter(value_filter: ValueFilter) -> str:
    if len(value_filter.values) == 1:
        return f"{value_filter.column} = {value_filter.values[0]}"
    return f"{value_filter.column} in ({', '.join(value_filter.values)})"


# How a condition on part of a text reads: 'sampleID containing "Ivc"'.
TEXT_OPERATORS = {"contains": "containing", "starts": "starting with", "ends": "ending with"}
# How a condition on a year reads: "YEAR from 1978", "YEAR before 1950".
YEAR_OPERATOR_WORDS = {"=": "in", "<": "before", "<=": "up to", ">": "after", ">=": "from"}


def _condition_value(condition: Condition) -> str:
    """A condition's value as written: a day for a condition on the day ("2021-06-30"), else the number."""
    if condition.part == "date":
        return (EPOCH + pd.Timedelta(days=float(condition.value or 0))).date().isoformat()
    return _number_text(condition.value)


def _describe_condition(condition: Condition) -> str:
    if condition.operator == "missing":
        return f"{condition.column} missing"
    if condition.operator in TEXT_OPERATORS:
        return f'{condition.column} {TEXT_OPERATORS[condition.operator]} "{condition.text}"'
    if condition.part == "year":
        return f"{condition.column} {YEAR_OPERATOR_WORDS[condition.operator]} {_number_text(condition.value)}"
    return f"{condition.column} {OPERATOR_WORDS[condition.operator]} {_condition_value(condition)}"


def _describe_conditions(conditions: tuple[Condition, ...]) -> list[str]:
    """Each condition in words, with a range on one column as "pH between 5 and 6"."""
    described: list[str] = []
    index = 0
    while index < len(conditions):
        condition, following = conditions[index], conditions[index + 1:index + 2]
        if following and condition.operator == ">=" and following[0].operator == "<=" and (
            following[0].column == condition.column
        ):
            low, high = _condition_value(condition), _condition_value(following[0])
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


def _as_number(value) -> float | None:
    """``value`` as a float when it is a finite number, else None (a mode that is a label, a NaN)."""
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | float | np.integer | np.floating):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _labelled_values(labels: pd.Series, numbers: pd.Series) -> tuple[tuple[str, float], ...]:
    """(label, number) for every group with a finite number, in order."""
    pairs = []
    for label, number in zip(labels, numbers, strict=True):
        value = _as_number(number)
        if value is not None:
            pairs.append((str(label), value))
    return tuple(pairs)


def _aggregate_series(series: pd.Series, aggregation: Aggregation, quantile: float | None = None) -> float:
    """Combine a column's values; spreads are sample statistics (n − 1), a tied mode its smallest value."""
    values = series.dropna()
    if aggregation == "iqr":
        return float(values.quantile(0.75) - values.quantile(0.25))
    if aggregation == "range":
        return float(values.max() - values.min()) if len(values) else float("nan")
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
    if plan.aggregation == "range":
        return f"max − min of {column}"
    return f"{plan.aggregation}({column})"


def _group_key(plan: QueryPlan, dataframe: pd.DataFrame) -> str:
    """The column a grouped answer's table is keyed by: the grouping, or "Hour" or "Weekday"."""
    assert plan.dimension is not None
    if not plan.time_part:
        return plan.dimension
    name = plan.time_part.capitalize()
    if name.lower() == str(plan.dimension).lower():
        # A date column called "Year" grouped by its year is still "Year".
        return plan.dimension
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
    if part == "season":
        # Three-month seasons: 0 for December to February, then March to May and so on.
        return stamps.dt.month % 12 // 3
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
    dataframe: pd.DataFrame, plan: QueryPlan, value_label: str, groups: pd.Series | None = None,
    weight: str | None = None,
) -> pd.DataFrame:
    """The plan's value per group, with how many values each group has (``n``) when a measure is combined.

    ``groups`` is each row's group when the caller has it already (see ``_group_values``).
    With ``weight``, the column of counts in a table of counts, a row count sums it.
    """
    assert plan.dimension is not None
    key = _group_key(plan, dataframe)
    groups = (_group_values(dataframe, plan) if groups is None else groups).rename(key)
    if plan.time_part:
        groups = groups[dataframe[plan.dimension].notna()]
    working = dataframe.loc[groups.index]
    if plan.aggregation == "count" or not plan.measure:
        if plan.count_column:
            grouped = working.groupby(groups, observed=True)[plan.count_column].nunique()
        elif weight:
            grouped = working.groupby(groups, observed=True)[weight].sum()
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
    if plan.aggregation in ("iqr", "quantile", "mode", "range"):
        return lambda values: _aggregate_series(values, plan.aggregation, plan.quantile)
    return plan.aggregation


def execute_plan(
    plan: QueryPlan, dataframe: pd.DataFrame, roles: ColumnRoles, *, rows_not_analyzed: int = 0
) -> QueryAnswer:
    """Run a plan locally and return the answer with its calculation trace.

    The answer ends with how the question was read, how many rows it used and,
    when the row limit left ``rows_not_analyzed`` rows out, that it covers only
    the rows analysed. Raises ``ValueError`` for an unknown column or intent.
    The returned ``question`` is empty; ``answer_question`` fills it in. A
    comparison, correlation or grouped average runs on the rows the brief's
    tests use: the distinct rows, when many rows exactly repeat others.
    """
    basis, repeats = _distinct_basis(dataframe, roles) if _tested(plan) else (dataframe, 0)
    note = (
        f"As in the brief, this uses the {len(basis):,} distinct rows, leaving out {repeats:,} exact repeats "
        "of other rows."
        if repeats else ""
    )
    return _with_scope(_run_plan(plan, basis, roles), basis, roles, rows_not_analyzed, note, repeats)


def _tested(plan: QueryPlan) -> bool:
    """Whether a plan's answer rests on a test or a correlation, which copies of rows would inflate."""
    if plan.intent in LEFT_OUT_INTENTS:
        return True
    return plan.intent in ("rank", "breakdown", "crosstab") and bool(plan.measure or plan.counted)


# The distinct rows of the last few tables asked about, with the repeats they leave out.
_BASES: dict[int, tuple[weakref.ref, pd.DataFrame, int]] = {}


def _distinct_basis(dataframe: pd.DataFrame, roles: ColumnRoles) -> tuple[pd.DataFrame, int]:
    """The rows the brief tests on (``business_insights.tested_rows``), found once per table."""
    cached = _BASES.get(id(dataframe))
    if cached and cached[0]() is dataframe:
        return cached[1], cached[2]
    tested, repeats = tested_rows(dataframe)
    if repeats:
        # The checks flag cells by row label, so the table's own scan serves its distinct rows too.
        _remember_checks(tested, roles, _data_checks(dataframe, roles))
    if len(_BASES) >= 8:
        _BASES.clear()
    _BASES[id(dataframe)] = (weakref.ref(dataframe), tested, repeats)
    return tested, repeats


def _with_scope(
    result: QueryAnswer,
    dataframe: pd.DataFrame,
    roles: ColumnRoles,
    rows_not_analyzed: int,
    basis: str = "",
    repeats: int = 0,
) -> QueryAnswer:
    """Add the plan's readings, the rows the answer used and any rows the row limit left out.

    The rows used are those with the measure recorded, so an answer about a
    column filled on 110 of 120 rows says it rests on those 110. ``basis``
    says when the answer runs on the distinct rows, as the brief's tests do,
    leaving out ``repeats`` exact copies. The same facts go into the answer's
    ``scope``, for a program to read.
    """
    plan = result.plan
    readings = [*plan.notes, *([basis] if basis else [])]
    sentences = [result.answer, *readings]
    cautions: list[str] = []
    try:
        working, applied = _apply_filters(dataframe, plan, roles)
    except TimeScopeUnavailable:
        working, applied = None, []
    if working is not None and not working.empty:
        cautions.extend(_flag_cautions(plan, dataframe, working, roles))
        cautions.extend(_condition_cautions(plan, dataframe, working, roles))
    cautions.extend(_blanked_notes(plan, dataframe, roles))
    if result.scope is not None:
        cautions[:0] = result.scope.cautions
    sentences.extend(cautions)
    left_out = [*(result.scope.left_out if result.scope else ())]
    if repeats:
        left_out.insert(0, ("exact repeats of other rows", repeats))
    if plan.excluded and working is not None:
        kept, _ = _apply_filters(dataframe, replace(plan, excluded=()), roles)
        if (aggregates := len(kept) - len(working)):
            named = [value for item in plan.excluded for value in item.values
                     if value in set(kept[item.column].astype(str))]
            left_out.append((f"rows that add up other rows ({list_names(named)})", aggregates))
            sentences.append(
                f"Left out {counted(aggregates, 'row')} that add up other rows ({list_names(named)})."
            )
    if rows_not_analyzed:
        left_out.append(("rows beyond the row limit", rows_not_analyzed))
    used = None
    # A count says its own rows, and an unfiltered aggregate says how many rows it used.
    if working is not None and not working.empty and plan.intent not in ("count", "missing"):
        # A comparison tests the groups only; a breakdown shows rows with no group as their own.
        grouping = (plan.dimension,) if plan.intent == "compare" else ()
        needed = [
            column for column in (plan.measure, plan.partner, *grouping)
            if column and column in working.columns and plan.intent != "duplicates"
        ]
        used = int(working[needed].notna().all(axis=1).sum()) if needed else len(working)
        if used < len(working):
            left_out.append((f"rows with {_listed(needed)} missing", len(working) - used))
        # An unfiltered aggregate already says it was calculated from those rows.
        whole = plan.intent == "aggregate" and len(working) == len(dataframe) and not rows_not_analyzed
        if used < len(working) and not whole:
            scoped = f", of the {len(working):,} in that scope" if len(working) < len(dataframe) else (
                f", of {_analysed_rows(dataframe, roles, rows_not_analyzed)}"
            )
            sentences.append(f"Based on the {used:,} rows with {_and_listed(needed)} recorded{scoped}.")
        elif len(working) < len(dataframe):
            rows = _analysed_rows(dataframe, roles, rows_not_analyzed)
            sentences.append(f"Based on {len(working):,} of {rows}.")
        elif (plan.intent != "aggregate" and not basis) or rows_not_analyzed:
            sentences.append(f"Based on {_analysed_rows(dataframe, roles, rows_not_analyzed, every=True)}.")
    elif rows_not_analyzed:
        sentences.append(f"ADA analysed {_analysed_rows(dataframe, roles, rows_not_analyzed, every=True)}.")
    if working is not None and plan.intent in ("count", "missing"):
        used = len(working)
    if result.scope is not None and result.scope.rows_used is not None:
        # The rows left once the flagged values were set aside, counted where they were.
        used = result.scope.rows_used
    scope = AnswerScope(
        rows_read=len(dataframe) + repeats,
        rows_not_analyzed=rows_not_analyzed,
        rows_in_scope=None if working is None else len(working),
        rows_used=used,
        filters=tuple(applied),
        left_out=tuple(left_out),
        readings=tuple(readings),
        cautions=tuple(cautions),
    )
    return replace(result, answer=" ".join(sentences), scope=scope)


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
    if plan.intent in LEFT_OUT_INTENTS:
        # Those values were left out of the answer (see _left_out), so they need no caution.
        issues = tuple(issue for issue in issues if issue.check not in MASKED_CHECKS)
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


def _condition_cautions(
    plan: QueryPlan, dataframe: pd.DataFrame, working: pd.DataFrame, roles: ColumnRoles
) -> list[str]:
    """A caution when the rows a condition picks include values the data checks flag.

    "How many patients have Na above 150" counts rows by their Na, so when
    some of those Na values are flagged as impossible the count rests on them.
    """
    checked = [
        item for item in (*plan.conditions, *plan.counted)
        if isinstance(item, Condition) and item.operator not in ("missing", *TEXT_OPERATORS)
        and item.column != plan.measure and item.column in working.columns
    ]
    if not checked:
        return []
    issues = _data_checks(dataframe, roles)
    cautions = []
    for column in dict.fromkeys(item.column for item in checked):
        rows = working.index
        if plan.counted:
            rows = working.index[_counted_mask(working, plan.counted).to_numpy()]
        flagged = _value_issues(issues, column, rows)
        if not flagged:
            continue
        cells = pd.Index([])
        for issue in flagged:
            cells = cells.union(rows.intersection(issue.cells[column]))
        summaries = "; ".join(issue.summary for issue in flagged)
        noun = "row has" if len(cells) == 1 else "rows have"
        cautions.append(
            f"Caution: {len(cells):,} of these {noun} {column} values the data checks flag ({summaries})."
        )
    return cautions


def _blanked_notes(plan: QueryPlan, dataframe: pd.DataFrame, roles: ColumnRoles) -> list[str]:
    """Say when a condition would have matched a code that cleaning set to missing.

    "How many rows have pm2_5 below 0" finds none once the -1 codes are
    missing values, and the answer says that is why.
    """
    conditions = [
        item for item in (*plan.conditions, *plan.counted)
        if isinstance(item, Condition) and item.operator in ("=", "<", "<=", ">", ">=") and item.part is None
    ]
    if not conditions:
        return []
    notes = []
    for issue in _data_checks(dataframe, roles):
        # Codes cleaning set to missing are listed by the checks with no cells left to flag.
        if issue.check != "sentinel" or issue.cells or not isinstance(issue.group, int | float):
            continue
        for condition in conditions:
            if condition.column in issue.columns and bool(
                _condition_mask(pd.Series([float(issue.group)]), condition).iloc[0]
            ):
                code = show_value(float(issue.group))
                notes.append(
                    f"{code} in {condition.column} was read as a missing-value code and set to missing "
                    f"during cleaning, so those cells are not counted ({issue.summary})."
                )
                break
    return notes


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


# Answers that compare or relate columns, which the brief computes without the cells
# the data checks find can't be readings as they stand.
LEFT_OUT_INTENTS = ("correlation", "correlates", "compare")


def _related_columns(plan: QueryPlan, working: pd.DataFrame, roles: ColumnRoles) -> list[str]:
    """The numeric columns a comparison or correlation reads."""
    if plan.intent == "correlates":
        readings = numeric_readings(working, roles)
        return list(dict.fromkeys([*readings, *([plan.measure] if plan.measure else [])]))
    return [column for column in (plan.measure, plan.partner, *plan.columns) if column]


def _run_related(
    plan: QueryPlan, working: pd.DataFrame, roles: ColumnRoles, scope: str, applied: list[str]
) -> QueryAnswer:
    if plan.intent == "correlation":
        return _execute_correlation(plan, working, roles, scope, applied)
    if plan.intent == "correlates":
        return _execute_correlates(plan, working, roles, scope, applied)
    if plan.columns:
        return _execute_column_differences(plan, working, roles, scope, applied)
    if plan.adjusted_for:
        return _execute_adjusted(plan, working, scope, applied)
    return _execute_compare(plan, working, roles, scope, applied)


def _left_out(
    working: pd.DataFrame, issues: tuple[QualityIssue, ...], columns: list[str]
) -> tuple[pd.DataFrame, str, int]:
    """``working`` without the values the brief leaves out of comparisons and correlations, with a note.

    Those are the cells the data checks find can't be readings as they stand
    (``quality.MASKED_CHECKS``): impossible values, codes, fill values, swapped
    or stuck readings. The note names the checks, so a figure is never left to
    rest on them unexplained; the count is of the values left out.
    """
    present = set(working.columns)
    hits: dict[str, pd.Index] = {}
    reasons: list[str] = []
    for issue in issues:
        if issue.check not in MASKED_CHECKS:
            continue
        found = False
        for column in columns:
            labels = issue.cells.get(column)
            if labels is None or column not in present:
                continue
            rows = working.index.intersection(labels)
            rows = rows[working.loc[rows, column].notna().to_numpy()]
            if len(rows):
                hits[column] = hits[column].union(rows) if column in hits else rows
                found = True
        if found:
            reasons.append(issue.summary)
    if not hits:
        return working, "", 0
    masked = working.copy()
    for column, rows in hits.items():
        masked.loc[rows, column] = np.nan
    cells = sum(len(rows) for rows in hits.values())
    named = "; ".join(reasons[:3]) + ("; and more" if len(reasons) > 3 else "")
    noun, verb = ("value", "is") if cells == 1 else ("values", "are")
    return masked, f"As in the brief, {cells:,} {noun} the data checks flag {verb} left out ({named}).", cells


def _run_plan(plan: QueryPlan, dataframe: pd.DataFrame, roles: ColumnRoles) -> QueryAnswer:
    """Compute the answer to a plan, without the scope sentences ``_with_scope`` adds."""
    columns = (plan.measure, plan.dimension, plan.count_column, plan.partner)
    for column in (*columns, *(item.column for item in (*plan.conditions, *plan.counted))):
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

    if plan.aggregation == "share" or (plan.counted and plan.dimension):
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

    if plan.intent == "aggregate" and plan.per:
        return _execute_unit_average(plan, working, roles, scope, applied)

    if plan.intent == "aggregate" and plan.aggregation == "ratio":
        return _execute_ratio(plan, working, scope, applied)

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
            f"calculated from {counted(rows, 'row')}."
        )
        if plan.aggregation == "mode":
            answer += _mode_note(working[plan.measure])
        return QueryAnswer(
            question="",
            plan=plan,
            answer=answer,
            calculation=f"{_aggregation_formula(plan, plan.measure)}{scope}",
            value=_as_number(value),
            figures=(("n", float(rows)),),
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

    if plan.intent in LEFT_OUT_INTENTS:
        # Comparisons and correlations leave out the cells the brief leaves out of its scans.
        issues = _data_checks(dataframe, roles)
        working, left_out, cells = _left_out(working, issues, _related_columns(plan, working, roles))
        result = _run_related(plan, working, roles, scope, applied)
        if not left_out:
            return result
        grouping = (plan.dimension,) if plan.intent == "compare" else ()
        columns = (plan.measure, plan.partner, *grouping)
        needed = [column for column in columns if column and column in working.columns]
        flagged = AnswerScope(
            rows_read=len(dataframe),
            rows_used=int(working[needed].notna().all(axis=1).sum()) if needed else None,
            left_out=(("values the data checks flag", cells),),
            cautions=result.scope.cautions if result.scope is not None else (),
        )
        return replace(result, answer=f"{result.answer} {left_out}", scope=flagged)

    if plan.intent == "rank" and plan.dimension is None and plan.grain:
        return _rank_periods(plan, working, roles, scope, applied)

    if plan.per:
        return _execute_per_unit(plan, working, roles, scope, applied)

    if plan.intent in ("rank", "breakdown"):
        weight = plan.weight or frequency_column(dataframe)
        return _execute_grouping(plan, working, roles, scope, applied, weight)

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
    # A column of counts added up ("how many homes were sold": sales) reads as counts, not as money.
    shown = None if plan.weight and plan.weight == plan.measure else plan.measure
    unit = _unit(shown, scoped_roles) if shown else None
    # Periods from the first to the last with a value: empty ones before a
    # reading starts are not part of its record.
    spanned = (
        pd.Period(pd.Timestamp(present.iloc[-1]["Period"]), grain)
        - pd.Period(pd.Timestamp(present.iloc[0]["Period"]), grain)
    ).n + 1
    counted = f"Distinct {plan.count_column}" if plan.count_column else "Records"
    named = plan.measure or counted
    first_period = _period_words(present.iloc[0]["Period"], grain)
    last_period = _period_words(present.iloc[-1]["Period"], grain)
    yearly = _complete_years(present, grain)
    if len(yearly) >= SLOPE_YEARS:
        # A seasonal series compares whole years, not a summer with an autumn.
        first, last = float(yearly.iloc[0]), float(yearly.iloc[-1])
        first_period, last_period = str(yearly.index[0]), str(yearly.index[-1])
        spanned, grain_name = len(yearly), "complete year"
    size = format_change(first, last, shown, unit, column, signed=True)
    across = f"{spanned} {grain_name}s"
    if len(yearly) < SLOPE_YEARS and len(present) < spanned:
        # Periods with no value are not counted as part of the change.
        across += f", {len(present)} of them with values"
    movement = f"{size} across {across}" if size else f"across {across}, from a starting period of zero"
    start, end = format_values_apart(first, last, column=shown)
    per = GRAIN_NAMES.get(grain, "period")
    shown_as = ", averaged over each complete year," if len(yearly) >= SLOPE_YEARS else f" per {per}"
    answer = (
        f"{named}{shown_as}{_phrase(applied)} moved from {start} to {end} ({movement}), "
        f"from {first_period} to {last_period}."
    )
    if plan.measure:
        # Say when the reading's own record starts after the table's.
        dates = working[roles.date].dropna()
        recorded = working.loc[working[plan.measure].notna(), roles.date].dropna()
        if not recorded.empty and pd.Period(recorded.min(), grain) > pd.Period(dates.min(), grain):
            answer += f" {named} is recorded only from {_period_words(present.iloc[0]['Period'], grain)}."
    if present["Value"].nunique() == 1:
        # A series that never moves has no slope, peak or low worth naming.
        answer = (
            f"{named} per {per}{_phrase(applied)} is "
            f"{format_number(first, shown, column_values=column)} in every {per} with a value, "
            f"from {_period_words(present.iloc[0]['Period'], grain)} to "
            f"{_period_words(present.iloc[-1]['Period'], grain)}."
        )
        slope, slope_calculation = "", ""
    else:
        slope, slope_calculation = _trend_slope(present, grain, plan.measure, column)
    answer += slope
    if len(present) >= 4 and slope:
        peak, low = present.loc[present["Value"].idxmax()], present.loc[present["Value"].idxmin()]
        answer += (
            f" It peaked at {format_number(float(peak['Value']), shown, column_values=column)} in "
            f"{_period_words(peak['Period'], grain)} and was lowest, at "
            f"{format_number(float(low['Value']), shown, column_values=column)}, in "
            f"{_period_words(low['Period'], grain)}."
        )
    calculation = f"{plan.aggregation}({plan.measure or 'rows'}) grouped per {per}{scope}"
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
        values=_labelled_values(present["Period"].astype(str), present["Value"]),
        figures=(("first", first), ("last", last)),
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


def _complete_years(present: pd.DataFrame, grain: str) -> pd.Series:
    """The average of each year with nine in ten of its periods recorded, over three or more years.

    Empty for a yearly grain, a grain with no periods per year, or a span
    shorter than SLOPE_YEARS.
    """
    periods = pd.DatetimeIndex(present["Period"])
    per_year = PERIODS_PER_YEAR.get(grain)
    if not per_year or grain == "Y" or (periods[-1] - periods[0]).days / 365.25 < SLOPE_YEARS:
        return pd.Series(dtype=float)
    frame = pd.DataFrame({"year": periods.year, "value": present["Value"].to_numpy(dtype=float)})
    counts = frame.groupby("year")["value"].count()
    full = counts[counts >= 0.9 * per_year].index
    return frame[frame["year"].isin(full)].groupby("year")["value"].mean()


def _trend_slope(
    present: pd.DataFrame, grain: str, measure: str | None, column: pd.Series | None
) -> tuple[str, str]:
    """The Theil-Sen slope of a trend as a sentence and its calculation, per year where that makes sense."""
    periods = pd.DatetimeIndex(present["Period"])
    values = present["Value"].to_numpy(dtype=float)
    yearly = _complete_years(present, grain)
    if len(yearly) >= SLOPE_YEARS:
        positions = yearly.index.to_numpy(dtype=float)
        slope, _ = theil_sen(positions, yearly.to_numpy())
        text = _per_year_text(slope, len(yearly), measure, column)
        verdict, tested = _slope_verdict(positions, yearly.to_numpy(), measure, column)
        return (
            f" The Theil-Sen trend is {text}, fitted on the yearly averages of {len(yearly)} "
            f"complete years{verdict}",
            f"Theil-Sen slope over the yearly averages of {len(yearly)} complete years{tested}",
        )
    # Positions at the answer's own grain: re-inferred from the periods that
    # hold values, sparse months can read as quarters.
    positions = period_positions(periods, grain)
    slope, _ = theil_sen(positions, values)
    if grain == "Y":
        text = _per_year_text(slope, len(values), measure, column)
    else:
        text = f"{_signed(slope, measure, column)} per {GRAIN_NAMES.get(grain, 'period')}"
    verdict, tested = _slope_verdict(positions, values, measure, column)
    return f" The Theil-Sen trend is {text}{verdict}", f"Theil-Sen slope over {len(values)} periods{tested}"


def _slope_verdict(
    positions: np.ndarray, values: np.ndarray, measure: str | None, column: pd.Series | None
) -> tuple[str, str]:
    """The slope's interval and Mann-Kendall test, ending the sentence.

    An interval that spans zero is "no clear trend", whatever the slope's size.
    """
    test = trend_test(positions, values)
    if test is None:
        return ".", ""
    low, high = _signed(test.low, measure, column), _signed(test.high, measure, column)
    interval = f"{TREND_CONFIDENCE:.0%} CI {low} to {high}"
    tested = (
        f"; {TREND_CONFIDENCE:.0%} CI by scipy.stats.theilslopes and Mann-Kendall test by "
        "scipy.stats.kendalltau, both allowing for lag-1 autocorrelation"
    )
    if not test.clear:
        return f" ({interval}), so there is no clear trend: the interval spans zero.", tested
    return f" ({interval}; Mann-Kendall {format_p(test.p_value)}).", tested


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


# Names of the column that says how many cases each row of a table of counts stands for.
FREQUENCY_NAMES = frozenset({"freq", "frequency", "count", "counts", "n", "cases"})


def frequency_column(dataframe: pd.DataFrame) -> str | None:
    """The column of counts in a table whose rows are the cells of a cross-tabulation, or None.

    Titanic's 32 rows are the combinations of Class, Sex, Age and Survived,
    each with Freq, the number of people in it. The counts must be whole and
    not negative, every other column a category (text or a few whole numbers,
    besides a label on every row), and no combination may repeat.
    """
    named = [column for column in dataframe.columns if _norm(str(column)) in FREQUENCY_NAMES]
    if len(named) != 1 or len(dataframe) < 2:
        return None
    counts = dataframe[named[0]]
    if not is_numeric_dtype(counts) or is_bool_dtype(counts) or counts.isna().any():
        return None
    values = counts.to_numpy(dtype=float)
    if (values < 0).any() or (values % 1 != 0).any():
        return None
    others = [
        column for column in dataframe.columns
        if column != named[0] and dataframe[column].nunique(dropna=False) < len(dataframe)
    ]
    for column in others:
        series = dataframe[column].dropna()
        if series.nunique() > GROUPING_LIMIT or (is_numeric_dtype(series) and bool((series % 1 != 0).any())):
            return None
    if not others or dataframe.duplicated(others).any():
        return None
    return named[0]


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
        figures: tuple = ()
        share_text = ""
        if plan.filters or plan.conditions:
            # "half of the squirrels were foraging": the distinct values picked, of those with the
            # filtered columns recorded.
            base, _ = _apply_filters(dataframe, replace(plan, filters=(), conditions=()), roles)
            filtered = [item.column for item in plan.filters] + [
                item.column for item in plan.conditions if item.operator != "missing"
            ]
            for column in dict.fromkeys(filtered):
                base = base.loc[base[column].notna()]
            total = int(base[plan.count_column].nunique(dropna=True))
            if total:
                share = distinct / total * 100
                figures = (("share_percent", share), ("denominator", float(total)))
                share_text = f", {_share_text(share)} of the {total:,} in all"
        return QueryAnswer(
            question="",
            plan=plan,
            answer=(f"There are {distinct:,} distinct {plan.count_column} values{_phrase(applied)}"
                    f"{share_text}."),
            calculation=f"count distinct {plan.count_column}{scope}"
            + (", divided by the count of distinct values over all rows" if figures else ""),
            value=float(distinct),
            figures=figures,
        )
    weight = plan.weight or frequency_column(dataframe)
    each = (f"; each row counts {weight} cases, the number its combination of values holds"
            if weight and not plan.weight else f"; each row counts its {weight}" if weight else "")
    if plan.counted:
        return _counted_share(plan, working, weight, each, scope)
    if not (plan.filters or plan.conditions):
        if weight:
            total = int(working[weight].sum())
            return QueryAnswer(
                question="",
                plan=plan,
                answer=f"{total:,} cases in {len(working):,} rows{_phrase(applied)}{each}.",
                calculation=f"sum({weight}){scope}",
                value=float(total),
            )
        return QueryAnswer(
            question="",
            plan=plan,
            answer=f"{len(working):,} rows match{_phrase(applied)}.",
            calculation=f"row count{scope}",
            value=float(len(working)),
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
    if weight:
        count, total = int(working[weight].sum()), int(base[weight].sum())
        share = count / total * 100 if total else 0.0
        recorded_text = f" with {_and_listed(recorded)} recorded" if recorded else ""
        return QueryAnswer(
            question="",
            plan=plan,
            answer=(
                f"{count:,} of the {total:,} cases{when}{recorded_text} ({_share_text(share)}) have "
                f"{described}{each}."
            ),
            calculation=(
                f"sum({weight}) over the {len(working):,} rows with {described}, divided by "
                f"sum({weight}) over the {len(base):,} rows{recorded_text}{scope}"
            ),
            value=float(count),
            figures=(("share_percent", share), ("denominator", float(total))),
        )
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
        value=float(len(working)),
        figures=(("share_percent", share), ("denominator", float(len(base)))),
    )


def _counted_share(
    plan: QueryPlan, working: pd.DataFrame, weight: str | None, each: str, scope: str
) -> QueryAnswer:
    """The rows meeting what the plan counts, as a share of the rows its filters keep.

    "what share of women survived" is of the women; the base keeps the rows
    where the counted columns are recorded, and says so when that leaves some out.
    """
    columns = list(dict.fromkeys(
        item.column for item in plan.counted if getattr(item, "operator", None) != "missing"
    ))
    recorded = [column for column in columns if working[column].isna().any()]
    base = working.loc[working[columns].notna().all(axis=1)] if columns else working
    hits = _counted_mask(base, plan.counted)
    described = " and ".join(_describe_counted(item) for item in plan.counted)
    among = " and ".join(
        [_describe_filter(item) for item in plan.filters] + _describe_conditions(plan.conditions)
    )
    when = f" in {_when_label(plan)}" if plan.year or plan.month else ""
    kept = f", {_and_listed(recorded)} recorded" if recorded else ""
    if weight:
        count, total = int(base.loc[hits, weight].sum()), int(base[weight].sum())
        unit = "cases"
        calculation = (
            f"sum({weight}) over the rows with {among} and {described}, divided by sum({weight}) over the "
            f"{len(base):,} rows with {among}{kept}{scope}"
        )
    else:
        count, total = int(hits.sum()), len(base)
        unit = "rows"
        calculation = f"rows with {described} among the {total:,} rows with {among}{kept}{scope}"
    share = count / total * 100 if total else 0.0
    return QueryAnswer(
        question="",
        plan=plan,
        answer=(
            f"{count:,} of the {total:,} {unit} with {among}{when}{kept} ({_share_text(share)}) have "
            f"{described}{each}."
        ),
        calculation=calculation,
        value=float(count),
        figures=(("share_percent", share), ("denominator", float(total))),
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
            value=float(missing),
            figures=(("share_percent", missing / rows * 100), ("denominator", float(rows))),
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
        values=_labelled_values(gaps["Column"], gaps["Missing"]),
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
    return QueryAnswer(
        question="", plan=plan, answer=answer, calculation=calculation, table=table,
        figures=tuple(
            (str(name).lower(), number) for name, value in stats.items()
            if (number := _as_number(value)) is not None
        ),
    )


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
    _remember_checks(dataframe, roles, issues)
    return issues


def _remember_checks(dataframe: pd.DataFrame, roles: ColumnRoles, issues: tuple[QualityIssue, ...]) -> None:
    if len(_CHECKS_CACHE) >= 8 and id(dataframe) not in _CHECKS_CACHE:
        _CHECKS_CACHE.clear()
    _CHECKS_CACHE[id(dataframe)] = (weakref.ref(dataframe), roles, issues)


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
            verb = "sits" if len(index) == 1 else "sit"
            answer = (
                f"{len(index):,} of the {len(present):,} {column} values{_phrase(applied)} {verb} outside "
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
    plan: QueryPlan, working: pd.DataFrame, roles: ColumnRoles, scope: str, applied: list[str],
    weight: str | None = None,
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
    In a table of counts, ``weight`` names the column of counts, and a row
    count is the sum of its cases.
    """
    assert plan.dimension is not None
    counted = plan.aggregation == "count" or not plan.measure
    weight = weight if counted and not plan.count_column and weight != plan.dimension else None
    if counted:
        value_label = f"Distinct {plan.count_column}" if plan.count_column else "Cases" if weight else "Rows"
    else:
        value_label = with_aggregate(_aggregation_label(plan), plan.measure)
    groups = _group_values(working, plan)
    grouped = _grouped_frame(working, plan, value_label, groups, weight)
    key = grouped.columns[0]
    single_rows = not counted and not (plan.time_part or plan.bins) and 1 < len(grouped) == len(working)
    plain = not (counted or single_rows or plan.time_part or plan.bins)
    if plan.intent == "rank" and plain and len(grouped) > 2:
        if float((grouped["n"] == 1).mean()) >= SINGLE_ROW_GROUPS:
            return _rank_rows(plan, working, scope, applied, grouped)
    if single_rows:
        assert plan.measure is not None
        grouped = grouped.drop(columns="n").rename(columns={value_label: plan.measure})
        value_label = plan.measure
    numbered = bool(plan.time_part or plan.bins) or is_numeric_dtype(working[plan.dimension])
    # A time axis (a gait cycle in 99 steps, days since infection) is a profile
    # however many points it has, so it is shown in order, not by its extremes.
    profile = numbered and not (plan.time_part or plan.bins) and is_timepoint_name(plan.dimension)
    ordered = numbered and plan.intent == "breakdown" and (len(grouped) <= GROUPING_LIMIT or profile)
    rankable, small = grouped, grouped.iloc[0:0]
    fragile = False
    if "n" in grouped.columns and plan.aggregation != "sum":
        enough = grouped["n"] >= MIN_RANK_VALUES
        if enough.any() and not enough.all():
            rankable, small = grouped[enough], grouped[~enough]
        fragile = not enough.any()
    if not (plan.time_part or plan.bins) and len(rankable) > 1:
        # Rows with no value for the grouping, or a placeholder such as "-" or "unknown", are
        # counted and listed, but they are not a group to name as the highest or the lowest.
        # Only the leading groups can be named, so a long list is checked at its head.
        head = rankable.head(PLACEHOLDER_CHECKS)
        placeholder = _placeholders(head[key], working[plan.dimension])
        if placeholder.any() and len(rankable) > int(placeholder.sum()):
            rankable = rankable.drop(head.index[placeholder.to_numpy()])
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
            most = f"{most} {f'distinct {plan.count_column}' if plan.count_column else value_label.lower()}"
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
    if ordered and profile and not counted and len(rankable) > 1:
        low = rankable.loc[rankable[value_label].idxmin()]
        column = working[plan.measure].dropna()
        lowest = format_number(float(low[value_label]), plan.measure, column_values=column)
        answer += (
            f" It is lowest at {group_label(key, low[key])}, at {lowest}; the table follows {plan.dimension} "
            "in order."
        )
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
        answer += _group_difference_line(plan, working, roles, rankable, groups)
        answer += _skew_note(plan, working, leaders, groups)
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
        what = f"sum({weight})" if weight else what
    else:
        what = _aggregation_formula(plan, plan.measure)
    if weight:
        answer += f" Each row counts {weight} cases, the number its combination of values holds."
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
        values=_labelled_values(grouped[key], grouped[value_label]),
    )


# Labels that stand for no value rather than name a group, once normalized: "-", "N/A", "unknown".
PLACEHOLDER_LABELS = frozenset({
    "", "na", "n a", "nan", "none", "null", "unknown", "missing", "not recorded", "not available",
    "not applicable",
})
# Leading groups of a ranking checked for placeholder labels.
PLACEHOLDER_CHECKS = 200
# Share of single-row groups at which a ranking of groups ranks the rows instead.
SINGLE_ROW_GROUPS = 0.5


def _placeholders(labels: pd.Series, column: pd.Series) -> pd.Series:
    """Which group labels stand for no value: the label for rows without one, or a placeholder text."""
    shown = labels.astype(str)
    found = shown == unlabelled_label(column.dropna().unique())
    if not is_numeric_dtype(column) or is_bool_dtype(column):
        # Normalized at once for the whole column, as _norm would: lower case, words only.
        words = shown.str.lower().str.replace(r"[^0-9a-z]+", " ", regex=True).str.strip()
        found |= words.isin(PLACEHOLDER_LABELS)
    return found.astype(bool)


def _rank_rows(
    plan: QueryPlan, working: pd.DataFrame, scope: str, applied: list[str], grouped: pd.DataFrame
) -> QueryAnswer:
    """Rank the rows themselves when most groups are a single row: "top 5 Name by intensity".

    An average over one row is that row, and a group with one row is never
    ranked against groups of five, so ranking groups would leave out nearly
    every label. Placeholder labels ("-", "unknown") and rows without a label
    are left out.
    """
    assert plan.measure is not None and plan.dimension is not None
    rows = working[[plan.dimension, plan.measure]].dropna()
    placeholder = _placeholders(rows[plan.dimension], working[plan.dimension])
    rows = rows[~placeholder.to_numpy()]
    if rows.empty:
        return QueryAnswer(
            question="", plan=plan,
            answer=f"Every row with {plan.measure}{_phrase(applied)} has a placeholder {plan.dimension}.",
            calculation=f"rows with a {plan.dimension} label{scope}",
        )
    shown = plan.top_n or BREAKDOWN_LIMIT
    table = rows.sort_values(plan.measure, ascending=plan.ascending, kind="stable").head(shown)
    table = table.reset_index(drop=True)
    values = working[plan.measure].dropna()
    top = table.iloc[0]
    high = "lowest" if plan.ascending else "highest"
    label = group_label(plan.dimension, top[plan.dimension])
    answer = (
        f"{label} has the {high} {plan.measure}{_phrase(applied)}, "
        f"at {format_number(float(top[plan.measure]), plan.measure, column_values=values)}."
    )
    if len(table) > 1:
        answer += f" The table lists the {len(table)} rows with the {high} values."
    single = int((grouped["n"] == 1).sum())
    answer += (
        f" {single:,} of the {len(grouped):,} {plan.dimension} values label a single row, so the rows are "
        "ranked by their own value rather than the groups by an average."
    )
    if placeholder.any():
        left = working.loc[placeholder[placeholder].index, plan.dimension].astype(str)
        labels = list_names([f'"{label}"' for label in pd.unique(left)][:3])
        answer += f" {int(placeholder.sum()):,} rows labelled {labels} are left out."
    return QueryAnswer(
        question="", plan=plan, answer=answer,
        calculation=(
            f"{plan.measure} of each row labelled by {plan.dimension}, "
            f"{'ascending' if plan.ascending else 'descending'}, showing {len(table)}{scope}; "
            "placeholder labels left out"
        ),
        table=table, chart="bar",
    )


def _in_group_order(grouped: pd.DataFrame, key: str) -> pd.DataFrame:
    """Groups in their own order: numbers by value, bins in order, other labels as they come."""
    if isinstance(grouped[key].dtype, pd.CategoricalDtype):
        return grouped.sort_values(key, kind="stable").reset_index(drop=True)
    return grouped.sort_values(
        key, key=lambda values: to_numbers(values), kind="stable"
    ).reset_index(drop=True)


def _part_names(part: TimePart | None) -> dict[int, str]:
    """Names for the numbers of a date part: weekdays, months and quarters."""
    if part == "weekday":
        return dict(enumerate(calendar.day_name))
    if part == "month":
        return {number: calendar.month_name[number] for number in range(1, 13)}
    if part == "quarter":
        return QUARTER_NAMES
    if part == "season":
        return SEASON_NAMES
    return {}


def _grouping_name(plan: QueryPlan) -> str:
    """What a group is called in an answer: "Region", "month of the year", "hour"."""
    if plan.time_part in ("month", "quarter"):
        return f"{plan.time_part} of the year"
    if plan.time_part == "season":
        return "season"
    if plan.bins:
        return f"{plan.dimension} group"
    return plan.time_part or str(plan.dimension)


def _group_difference_line(
    plan: QueryPlan, working: pd.DataFrame, roles: ColumnRoles, rankable: pd.DataFrame, groups: pd.Series
) -> str:
    """One sentence on whether the groups of a ranked average really differ, from ``compare_groups``."""
    if plan.aggregation not in ("mean", "median") or plan.measure is None:
        return ""
    if not 2 <= len(rankable) <= GROUPING_LIMIT:
        return ""
    if not (plan.time_part or plan.bins) and (rule := _set_by(working, plan.measure, plan.dimension, roles)):
        made = f"so its groups differ in {plan.measure} by how they are made; no test is run"
        return f" {sentence_case(rule)}, {made}."
    key = "__group"
    entity = entity_column(working, roles)
    entity = entity if entity not in (None, plan.dimension, plan.measure) else None
    columns = [plan.measure, *([entity] if entity else [])]
    frame = working[columns].assign(**{key: groups})
    if plan.time_part:
        frame = frame[working[plan.dimension].notna()]
    sampled = ""
    if entity is None and len(frame) > SCAN_ROWS:
        # A long table is tested on a fixed random sample, as the brief's scan is.
        frame = frame.sample(SCAN_ROWS, random_state=0)
        sampled = f", on a random sample of {SCAN_ROWS:,} rows"
    # Only the groups that were ranked are tested, so a group of two can't decide it.
    ranked = set(rankable.iloc[:, 0].map(str))
    codes = {str(number): str(name) for number, name in _part_names(plan.time_part).items()}
    # A date with gaps gives its parts as floats (3.0 for March), named by their whole number.
    kept = [
        value for value in frame[key].dropna().unique()
        if codes.get(_number_text(value) if isinstance(value, float) else str(value), str(value)) in ranked
    ]
    frame = frame[frame[key].isin(kept)]
    comparison = compare_groups(frame, plan.measure, key, entity=entity)
    if comparison is None:
        return ""
    groups = f"{plan.dimension} groups" if plan.bins else f"{_grouping_name(plan)} groups"
    evidence = f"{comparison.test}, {format_p(comparison.decisive_p)}; {effect_text(comparison)}{sampled}"
    if comparison.verdict == "none":
        return f" The {groups} do not clearly differ ({evidence}), so this ranking may be chance."
    if comparison.verdict == "negligible":
        return f" The {groups} differ only by a negligible amount ({evidence})."
    return f" The {groups} do differ ({evidence})."


def _set_by(working: pd.DataFrame, measure: str, grouping: str, roles: ColumnRoles) -> str | None:
    """How ``grouping`` is cut from ``measure``, when it is (``compare.grouping_rules``), or None.

    "categori is set by max: BAIK up to 50, ..." or "critical names the
    largest of pm25, pm10 and o3": comparing the reading across such groups
    only restates how they were made, so the brief tests none of them.
    """
    if grouping not in working.columns or measure not in working.columns:
        return None
    keys = {re.sub(r"[^a-z0-9]", "", str(level).lower()) for level in working[grouping].dropna().unique()}
    named = [
        column for column in roles.numeric
        if column in working.columns and re.sub(r"[^a-z0-9]", "", str(column).lower()) in keys
    ]
    readings = list(dict.fromkeys([measure, *named]))
    rules = grouping_rules(working, readings, [grouping], date=roles.date)
    return next((rule.describe() for rule in rules if measure in rule.readings), None)


def _skew_note(plan: QueryPlan, working: pd.DataFrame, leaders: list, groups: pd.Series) -> str:
    """Say when a few large values carry the leading group's mean, with its median."""
    if plan.aggregation != "mean" or plan.measure is None or len(leaders) != 1:
        return ""
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
    plan: QueryPlan, working: pd.DataFrame, roles: ColumnRoles, scope: str, applied: list[str]
) -> QueryAnswer:
    """Correlate the plan's two columns over the rows where both are recorded.

    Worded like the brief's relationship card, except that the pair was chosen by
    the reader, so nothing is said about picking it from several. When several
    rows come from each unit (``compare.entity_column``), the units are the
    evidence instead (``_execute_repeated_correlation``). A pooled r is set
    against the r within the groups of a grouping column, and the answer says
    when the two disagree (Simpson's paradox).
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
    unit = entity_column(working, roles)
    if unit is not None and unit not in (plan.measure, plan.partner):
        first, second = (
            (plan.measure, plan.partner)
            if is_timepoint_name(plan.measure) and not is_timepoint_name(plan.partner)
            else (plan.partner, plan.measure)
        )
        repeated = repeated_correlation(working, first, second, unit)
        # Within units, a few units leave the pooled r and the within-group note to say what can be said.
        if repeated is not None and (not repeated.within or repeated.sloped >= MIN_SLOPE_UNITS):
            return _execute_repeated_correlation(plan, repeated, scope, applied)
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
        f"checked against Spearman; least-squares slope of each column on the other "
        f"(scipy.stats.linregress){scope}"
    )
    slopes = _slopes(paired, plan.measure, plan.partner)
    points = paired
    if pairs > SCATTER_LIMIT:
        points = paired.sample(SCATTER_LIMIT, random_state=0).sort_index()
        calculation += f"; the chart shows {SCATTER_LIMIT:,} of the {pairs:,} pairs, sampled"
    # The brief's own check: r inside the groups of a grouping, when it differs from the pooled r.
    caution = within_group_note(working, roles, correlation).strip()
    return QueryAnswer(
        question="",
        plan=plan,
        answer=answer,
        calculation=calculation,
        table=points.reset_index(drop=True),
        chart="scatter",
        value=_as_number(correlation.pearson),
        figures=(
            ("ci_low", correlation.low), ("ci_high", correlation.high),
            ("spearman", correlation.spearman), ("n", float(pairs)), *slopes,
        ),
        scope=AnswerScope(rows_read=len(working), cautions=(caution,)) if caution else None,
    )


def _slopes(paired: pd.DataFrame, first: str, second: str) -> tuple[tuple[str, float], ...]:
    """The least-squares slope of each column on the other, named "slope of <y> on <x>"."""
    from scipy.stats import linregress  # noqa: PLC0415

    found = []
    for y, x in ((first, second), (second, first)):
        if paired[x].nunique() > 1:
            fit = linregress(paired[x].to_numpy(dtype=float), paired[y].to_numpy(dtype=float))
            found.append((f"slope of {y} on {x}", float(fit.slope)))
    return tuple(found)


# Plural time words that name a column of elapsed time, said in the singular after "per".
_TIME_PLURALS = frozenset({"seconds", "minutes", "hours", "days", "weeks", "months", "years", "visits"})


def _owner(name: str) -> str:
    """The possessive of a name: "Subject's", "species'"."""
    return f"{name}'" if name.endswith("s") else f"{name}'s"


def _one(name: str) -> str:
    """"Day" for a column named "Days", so a slope reads "per Day"; any other name as it is."""
    return name[:-1] if name.lower() in _TIME_PLURALS else name


def _execute_repeated_correlation(plan: QueryPlan, found, scope: str, applied: list[str]) -> QueryAnswer:
    """Word a correlation computed with the units, not the rows, as the evidence."""
    unit, first, second = found.unit, found.first, found.second
    each = f"{found.rows / found.units:.0f}" if found.rows % found.units == 0 else "several"
    if found.within:
        if found.slope is None or not np.isfinite(found.p):
            answer = (
                f"Only {found.sloped:,} {unit} values have three or more rows with {first} changing, too "
                f"few to test the slope across {unit}s. The correlation within {unit}s, once each "
                f"{_owner(unit)} own means are taken away, is r = {found.r:.2f}{_phrase(applied)}."
            )
        else:
            direction = "rises with" if found.slope > 0 else "falls as"
            if found.p >= 0.05:
                lead = f"Within {unit}s, {second} shows no clear trend with {first}"
            else:
                lead = f"Within each {unit}, {second} {direction} {first}"
            answer = (
                f"{lead}{_phrase(applied)}: the {found.sloped:,} {unit}s' own slopes average "
                f"{format_number(found.slope, second)} {second} per {_one(first)} (95% CI "
                f"{format_number(found.low, second)} to {format_number(found.high, second)}; t-test across "
                f"the {found.sloped:,} slopes, {format_p(found.p)}), and {found.positive:,} of them are "
                f"positive. Once each {_owner(unit)} own means are taken away, the correlation is "
                f"r = {found.r:.2f}."
            )
        answer += (
            f" The {found.rows:,} rows come {each} to a {unit}, so they are not counted as independent: "
            f"each {unit} counts once."
        )
        both = f" (and of {first} on {second})" if found.reverse is not None else ""
        calculation = (
            f"least-squares slope of {second} on {first}{both} within each {unit} with 3 or more rows, then "
            f"their mean with a one-sample t-test; repeated-measures r: Pearson r of the values centred on "
            f"each {_owner(unit)} means{scope}"
        )
        figures = [("r", found.r), ("units", float(found.units)), ("n", float(found.rows))]
        if found.slope is not None:
            figures += [("slope", found.slope), (f"slope of {second} on {first}", found.slope),
                        ("ci_low", found.low), ("ci_high", found.high),
                        ("p_value", found.p), ("positive_slopes", float(found.positive))]
        if found.reverse is not None:
            # A claim may state the change the other way round: "rises 0.66 mm per year of age".
            figures.append((f"slope of {first} on {second}", found.reverse))
        return QueryAnswer(
            question="", plan=plan, answer=answer, calculation=calculation, value=_as_number(found.r),
            figures=tuple((name, value) for name, value in figures if _as_number(value) is not None),
        )
    if found.units < MIN_CORRELATION_PAIRS:
        return QueryAnswer(
            question="", plan=plan,
            answer=(
                f"At least one of {first} and {second} stays fixed within a {unit}, so the evidence is one "
                f"value per {unit}, and only {found.units:,} {unit}s have both{_phrase(applied)}; a "
                f"correlation needs at least {MIN_CORRELATION_PAIRS}."
            ),
            calculation=f"{found.units:,} {unit} means, fewer than {MIN_CORRELATION_PAIRS}{scope}",
        )
    answer = (
        f"Across {unit}s{_phrase(applied)}, Pearson r = {found.r:.2f} (95% CI {found.low:.2f} to "
        f"{found.high:.2f}, {format_p(found.p)}, n = {found.units:,} {unit}s). At least one of {first} and "
        f"{second} stays fixed within a {unit}, so each {unit} counts once, with the mean of its "
        f"{found.rows:,} rows' values, rather than its rows counting as separate evidence."
    )
    calculation = (
        f"Pearson r over the {found.units:,} {unit} means of {first} and {second}, 95% CI by Fisher "
        f"transform{scope}"
    )
    return QueryAnswer(
        question="", plan=plan, answer=answer, calculation=calculation, value=_as_number(found.r),
        figures=(("ci_low", found.low), ("ci_high", found.high), ("p_value", found.p),
                 ("units", float(found.units)), ("n", float(found.rows))),
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
    index = _name_index(dataframe.columns)
    names = index.names
    # The words of comparison are read with the column names blanked: "Impact" in
    # "Charpy Impact Strength" is part of a name, not a question about an effect.
    asking = _without_names(q, index)
    differ = any(re.search(rf"\b{word}\b", asking) for word in DIFFERENCE_WORDS)
    compare = bool(re.search(r"\bcompare\b", asking))
    # "less than 1" is a bound on a column, not "less in A than in B".
    direction = re.search(
        rf"\b({'|'.join(DIRECTION_WORDS)})\b(?!\s+than\s+(?:minus\s+)?\d).*\bthan\b", asking
    )
    flag_direction = None if direction else re.search(
        rf"\b({'|'.join(DIRECTION_WORDS)})\b\s+(?:with|without|among|for|in|when)\b", asking
    )
    correlates = bool(CORRELATES_PATTERN.search(asking)) and bool(
        re.search(r"\b(?:what|which)\b|\bcorrelates of\b|\bmost\b", asking)
    )
    research = bool(set(asking.split()) & RESEARCH_WORDS)
    relation = RELATION_PATTERN.search(asking)
    significant = bool(re.search(r"\bsignificant(?:ly)?\b", asking))
    asked = (differ, compare, direction, flag_direction, correlates, research, relation, significant)
    if not any(asked):
        return None
    # The question is read as any other is, so every word must be a column, a
    # value, a number or grammar, here including the words of comparison.
    # Numbered groups ("condition 2 than condition 1") may repeat a column's name.
    reading = _read_question(
        question, dataframe, roles, grammar=QUERY_GRAMMAR | COMPARISON_GRAMMAR | index.words, numbers_ok=True
    )
    if reading.problem or reading.outliers or reading.describe or reading.duplicates:
        return None
    if reading.across:
        return _columns_comparison(q, reading, dataframe, roles, index, differ or compare or significant)
    split = MEDIAN_SPLIT.search(q)
    if not split and {"high", "low"} & set(reading.remaining.split()):
        return None
    mentioned = _mentions(q, index)
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
    at = None
    if reading.split:
        (grouping, at), compared, bins = reading.split, (), None
        # "does pH differ by site before and after 2000" asks for a second grouping beside the split.
        named = [
            column for column in free
            if column not in (grouping, *numeric_columns) and _follows(q, mentioned[column], GROUPING_WORDS)
        ]
        if named:
            raise _ComparisonRefused(_two_groupings([named[0], grouping]))
    elif split:
        after = [column for column in free if mentioned[column] > split.start()]
        if not after or not _is_measurement(dataframe[after[0]]):
            return None
        grouping, compared, bins = after[0], (), 2
    else:
        grouping, compared = _compared_groups(q, dataframe, roles, names, mentioned, filters)
        bins = None
    if grouping is None:
        return None
    if compared:
        # The comparison is settled first: "condition 1 and condition 2" name the groups compared,
        # so the parser's reading of "condition 1" as a condition does not also filter the rows.
        conditions = tuple(condition for condition in conditions if condition.column != grouping)
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
    if not bins and _is_flag(dataframe[measure]) and outcome_event(dataframe[measure]) is not None:
        # A yes/no outcome is compared by its rate in each group, with a chi-squared or Fisher test.
        if reading.adjusted:
            raise _ComparisonRefused(
                f"{measure} holds only 0 and 1, so it is compared as a rate in each group, and Ask holds "
                f"{_and_listed(reading.adjusted)} fixed only for a numeric measure; ask "
                f'"does {measure} differ by {grouping}" for the rates as they are.'
            )
        if compared:
            kept.insert(0, ValueFilter(column=grouping, values=compared))
        return QueryPlan(
            intent="compare",
            aggregation=DIRECTION_WORDS[direction.group(1)] if direction and len(compared) == 2 else "mean",
            dimension=grouping, counted=(_event(measure, dataframe[measure]),), filters=tuple(kept),
            year=year, month=month, conditions=conditions, split=at,
            notes=(_rate_note(measure),),
        )
    # "compare revenue across regions" asks for each region's total when revenue
    # adds up; only words of difference ask for the test.
    testing = differ or direction or significant
    plain = not (compared or bins or reading.split or reading.adjusted)
    if compare and not testing and metric_for(measure, roles).additive and plain:
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
        split=at,
        adjusted_for=tuple(reading.adjusted),
    )


def _columns_comparison(
    q: str,
    reading: _Reading,
    dataframe: pd.DataFrame,
    roles: ColumnRoles,
    index: _NameIndex,
    testing: bool,
) -> QueryPlan | None:
    """Plan "which genes differ between AML and ALL": every column of a set compared across one grouping."""
    if not testing:
        return None
    mentioned = _mentions(q, index)
    grouping, compared = _compared_groups(q, dataframe, roles, index.names, mentioned, reading.value_filters)
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


# Name words of a quantity, which a column set at a few levels still measures.
QUANTITY_WORDS = frozenset({"age", "dose", "dosage", "time", "temperature", "temp", "concentration", "conc"})


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
    words = set(_norm(normalized_name(column)).split())
    if column in roles.factors:
        # Ages 8, 10, 12 and 14, or doses of 0.5, 1 and 2: a quantity set at a few levels correlates.
        return not (_unit_words(column) or words & QUANTITY_WORDS)
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
    if _is_flag(dataframe[measure]) and outcome_event(dataframe[measure]) is not None:
        return QueryPlan(
            intent="compare", aggregation="mean", dimension=grouping,
            counted=(_event(measure, dataframe[measure]),), notes=(_rate_note(measure),), **scope,
        )
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
        levels = dataframe[column].dropna().unique()
        found = sorted(
            (match.start(group), float(match.group(group)))
            for form in names.get(column, ())
            for match in re.finditer(
                rf"\b{re.escape(form)}s?\s+(\d+(?:\.\d+)?)"
                rf"(?:\s+(?:and|vs|versus|or|to|than)\s+(?:(?:in\s+)?{re.escape(form)}s?\s+)?(\d+(?:\.\d+)?))?\b",
                q,
            )
            for group in (1, 2)
            if match.group(group) is not None
        )
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
    if plan.split is not None:
        frame = working.assign(**{plan.dimension: _split_labels(working[plan.dimension], plan.split)})
    if not (plan.bins or plan.split is not None) and (
        rule := _set_by(working, plan.measure, plan.dimension, roles)
    ):
        return _defined_difference(plan, working, rule, scope, applied)
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
    if plan.split is not None:
        calculation += f"; {plan.dimension} split at {_number_text(plan.split)}"
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
        values=tuple(
            (str(group.label), value) for group in comparison.groups
            if (value := _as_number(comparison.centre(group))) is not None
        ),
        figures=tuple(
            (name, number) for name, number in (
                ("p_value", comparison.p_value), ("p_adjusted", comparison.adjusted_p),
                ("difference", comparison.difference), ("ci_low", comparison.low),
                ("ci_high", comparison.high), ("hedges_g", comparison.hedges_g),
                ("explained", comparison.explained),
            )
            if _as_number(number) is not None
        ),
    )


def _execute_adjusted(plan: QueryPlan, working: pd.DataFrame, scope: str, applied: list[str]) -> QueryAnswer:
    """Compare the measure across the groups with other readings held fixed, by ordinary least squares.

    Each group but the first enters as a 0/1 column and each held reading as
    it is (``statsmodels`` OLS). For two groups the answer gives the adjusted
    gap with its 95% interval; for more, an F test that every adjusted gap is
    zero. The gap without adjusting is given beside it, since holding a
    reading fixed can shrink a difference, or reverse it.
    """
    assert plan.measure is not None and plan.dimension is not None
    from statsmodels.regression.linear_model import OLS

    measure, dimension, held = plan.measure, plan.dimension, list(plan.adjusted_for)
    data = working[[measure, dimension, *held]].copy()
    if plan.bins:
        data[dimension] = _binned(working[dimension], plan.bins)
    if plan.split is not None:
        data[dimension] = _split_labels(working[dimension], plan.split)
    data[[measure, *held]] = data[[measure, *held]].apply(pd.to_numeric, errors="coerce")
    data = data.dropna()
    levels = list(data[dimension].dropna().unique())
    if is_numeric_dtype(data[dimension]) or isinstance(data[dimension].dtype, pd.CategoricalDtype):
        levels = sorted(levels)
    terms = len(levels) - 1 + len(held)
    if len(levels) < 2 or len(data) < max(3 * (terms + 1), 10):
        return QueryAnswer(
            question="", plan=plan,
            answer=(
                f"Only {len(data):,} rows have {measure}, {dimension} and {_and_listed(held)} all recorded"
                f"{_phrase(applied)}, too few to hold {_and_listed(held)} fixed."
            ),
            calculation=f"least squares needs about three rows per term{scope}",
        )
    indicators = np.column_stack([(data[dimension] == level).to_numpy(dtype=float) for level in levels[1:]])
    design = np.column_stack([np.ones(len(data)), indicators, data[held].to_numpy(dtype=float)])
    target = data[measure].to_numpy(dtype=float)
    fit = OLS(target, design).fit()
    means = data.groupby(dimension, observed=True)[measure].mean()
    values = working[measure].dropna()

    def shown(value: float) -> str:
        return format_number(float(value), measure, column_values=values)

    names = [group_label(dimension, level) for level in levels]
    fixed = _and_listed(held)
    if len(levels) == 2:
        gap = float(fit.params[1])
        low, high = (float(bound) for bound in fit.conf_int(0.05)[1])
        p_value = float(fit.pvalues[1])
        raw = float(means[levels[1]] - means[levels[0]])
        bounds = f"{_signed(low, measure, values)} to {_signed(high, measure, values)}"
        interval = f"95% CI {bounds}; {format_p(p_value)}"
        if p_value < 0.05:
            way = "higher" if gap > 0 else "lower"
            answer = (
                f"With {fixed} held fixed, {measure} is {shown(abs(gap))} {way} in {names[1]} than in "
                f"{names[0]}{_phrase(applied)} ({interval})."
            )
        else:
            answer = (
                f"With {fixed} held fixed, {measure} does not clearly differ between {names[1]} and "
                f"{names[0]}{_phrase(applied)}: the adjusted gap is {_signed(gap, measure, values)} "
                f"({interval})."
            )
        answer += (
            f" Without adjusting, the gap is {_signed(raw, measure, values)} (mean {shown(means[levels[1]])} "
            f"vs {shown(means[levels[0]])})."
        )
        test = "the t test of the group term"
    else:
        restriction = np.zeros((len(levels) - 1, design.shape[1]))
        restriction[:, 1:len(levels)] = np.eye(len(levels) - 1)
        joint = fit.f_test(restriction)
        p_value = float(joint.pvalue)
        verdict = "do differ" if p_value < 0.05 else "do not clearly differ"
        answer = (
            f"With {fixed} held fixed, the {dimension} groups {verdict} in {measure}{_phrase(applied)} "
            f"(F test of the {len(levels) - 1} group terms, {format_p(p_value)})."
        )
        test = "an F test that every group term is zero"
    answer += (
        f" This holds {fixed} fixed only as a straight line would, over the {len(data):,} rows with all of "
        "them recorded, and is not evidence of cause."
    )
    gaps = [0.0, *(float(value) for value in fit.params[1:len(levels)])]
    table = pd.DataFrame({
        dimension: names,
        f"Mean {measure}": [float(means[level]) for level in levels],
        f"Gap from {names[0]}, {fixed} held fixed": gaps,
        "n": [int((data[dimension] == level).sum()) for level in levels],
    })
    calculation = (
        f"ordinary least squares of {measure} on 0/1 columns for the {dimension} groups (against "
        f"{names[0]}) and on {fixed}, over {len(data):,} complete rows (statsmodels OLS); {test}{scope}"
    )
    return QueryAnswer(
        question="", plan=plan, answer=answer, calculation=calculation, table=table.round(6), chart="bar"
    )


def _defined_difference(
    plan: QueryPlan, working: pd.DataFrame, rule: str, scope: str, applied: list[str]
) -> QueryAnswer:
    """Answer a comparison across a grouping cut from the measure itself: the means, and how it was cut."""
    assert plan.measure is not None and plan.dimension is not None
    means = working.groupby(plan.dimension, observed=True)[plan.measure].agg(["mean", "count"])
    means = means[means["count"] > 0].sort_values("mean", ascending=False)
    values = working[plan.measure].dropna()

    def shown(name: object) -> str:
        row = means.loc[name]
        number = format_number(float(row["mean"]), plan.measure, column_values=values)
        return f"{number} in {group_label(plan.dimension, name)} (n = {int(row['count']):,})"

    answer = (
        f"{sentence_case(rule)}, so {plan.measure} differs across {plan.dimension} by how the groups are "
        "made, "
        f"not as a finding, and no test is run. Means{_phrase(applied)} run from {shown(means.index[0])} "
        f"down to {shown(means.index[-1])}."
    )
    table = means.rename(columns={"mean": f"Mean {plan.measure}", "count": "n"}).reset_index()
    return QueryAnswer(
        question="", plan=plan, answer=answer,
        calculation=(
            f"mean {plan.measure} per {plan.dimension}; the rule from compare.grouping_rules{scope}"
        ),
        table=table.round(6), chart="bar",
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
        # In a table of counts each row stands for its frequency, as a plain count adds it up.
        weight = plan.weight or frequency_column(working)
        if weight:
            counts = pd.crosstab(rows, columns, values=working[weight], aggfunc="sum").fillna(0)
            if bool((counts % 1 == 0).all().all()):
                counts = counts.astype(int)
        else:
            counts = pd.crosstab(rows, columns)
        if names:
            counts.index = [names.get(value, value) for value in counts.index]
            counts.index.name = row_name
        what, counted = (f"Counts of {weight}", f"sum({weight})") if weight else ("Rows", "row count")
        answer = (
            f"{what} by {_grouping_name(plan)} and {plan.second_dimension}{_phrase(applied)}: a "
            f"{counts.shape[0]} × {counts.shape[1]} table of counts."
        )
        shares = _yes_shares(counts)
        if shares is not None:
            table, line = shares
            answer += line
        else:
            table = counts.reset_index()
            table.columns = [str(column) for column in table.columns]
        answer += _association_line(counts, "cases" if weight else "rows")
        return QueryAnswer(
            question="", plan=plan, answer=answer,
            calculation=(
                f"{counted} for each {row_name} and {plan.second_dimension}"
                + (", and the share of each group with the yes value" if shares is not None else "")
                + f"; chi-square test of independence with Cramér's V{scope}"
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


def _yes_level(levels: list[object]) -> object | None:
    """The yes of two levels that say yes and no: "B" of "B" and "NoB", "Yes" of "Yes" and "No"."""
    if len(levels) != 2:
        return None
    first, second = (str(level).strip() for level in levels)
    for yes, no in ((first, second), (second, first)):
        if re.fullmatch(rf"(?:no|non|not)[\s_-]?{re.escape(yes)}", no, re.IGNORECASE) or (
            yes.lower() in YES_NO_VALUES["yes"] and no.lower() in YES_NO_VALUES["no"]
        ):
            return levels[0] if yes == first else levels[1]
    return None


def _yes_shares(counts: pd.DataFrame) -> tuple[pd.DataFrame, str] | None:
    """The share with the yes value in each group of the other column, when one column says yes and no.

    "how does breathlessness change with age" over Breathlessness (B, NoB) and
    Age gives the share with B in each age band, one row per band.
    """
    for grid, side in ((counts, "index"), (counts.T, "columns")):
        yes = _yes_level(list(grid.index))
        if yes is None:
            continue
        name = counts.index.name if side == "index" else counts.columns.name
        other = counts.columns.name if side == "index" else counts.index.name
        totals = grid.sum(axis=0)
        kept = totals[totals > 0].index
        share = grid.loc[yes, kept] / totals[kept]
        table = pd.DataFrame({str(other): [str(level) for level in kept]})
        for level in grid.index:
            table[str(level)] = grid.loc[level, kept].to_numpy()
        table["n"] = totals[kept].to_numpy()
        table[f"share {yes}"] = share.round(4).to_numpy()
        low, high = share.idxmin(), share.idxmax()
        line = (
            f" The share with {name} = {yes} is lowest for {group_label(str(other), low)}, at "
            f"{share[low]:.1%}, and highest for {group_label(str(other), high)}, at {share[high]:.1%}."
        )
        return table, line
    return None


def _association_line(counts: pd.DataFrame, unit: str = "rows") -> str:
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
    caution = f" Many cells expect fewer than 5 {unit}, so the test is only a rough guide." if rough else ""
    if p_value < 0.05:
        return f" The two are associated ({evidence}).{caution}"
    return f" There is no clear association ({evidence}).{caution}"


def _execute_shares(plan: QueryPlan, dataframe: pd.DataFrame, roles: ColumnRoles) -> QueryAnswer:
    """The share of each group's rows that meet what the plan counts: "share of Result 1 by Gender".

    The filters, conditions and time scope choose the rows; each group's share
    is of its rows with the counted columns recorded. The shares are compared
    as the brief compares rates (``compare.compare_rates``): chi-squared, or
    Fisher's exact test for two small groups, with Cramér's V and the
    difference in points between the highest and lowest share. Rows with no
    group are listed but not compared.
    """
    assert plan.dimension is not None
    if not plan.counted:
        plan = replace(plan, counted=(*plan.filters, *plan.conditions), filters=(), conditions=())
    try:
        working, applied = _apply_filters(dataframe, replace(plan, counted=()), roles)
    except TimeScopeUnavailable as unavailable:
        return QueryAnswer(
            question="", plan=plan,
            answer=f"This file has no date column, so I cannot narrow the answer to {unavailable}.",
            calculation=f"no date column available to scope to {unavailable}",
        )
    scope = _scoped(applied)
    # The groups a comparison names are its groups, not a narrowing of the rows to state.
    picked = {_describe_filter(item) for item in plan.filters if item.column == plan.dimension}
    applied = [text for text in applied if text not in picked]
    columns = list(dict.fromkeys(
        item.column for item in plan.counted if getattr(item, "operator", None) != "missing"
    ))
    recorded = working.loc[working[columns].notna().all(axis=1)] if columns else working
    meets = _counted_mask(recorded, plan.counted)
    key = _group_key(plan, dataframe)
    groups, ordered = _share_groups(recorded, plan)
    names = _share_names(plan)
    if names:
        groups = groups.map(names, na_action="ignore")
    unlabelled = groups.isna()
    missing_label = unlabelled_label(groups.dropna().unique())
    shown = groups.astype(object).where(~unlabelled, missing_label)
    # A table of counts (Titanic's Class, Sex, Age, Survived and Freq) counts its cases, not its rows.
    weight = plan.weight or frequency_column(dataframe)
    cases = recorded[weight].astype(float) if weight else pd.Series(1.0, index=recorded.index)
    unit = "cases" if weight else "rows"
    totals = cases.groupby(shown, sort=False).sum().round().astype(int)
    hits = (cases * meets).groupby(shown, sort=False).sum().reindex(totals.index, fill_value=0).round()
    hits = hits.astype(int)
    described = " and ".join(_describe_counted(item) for item in plan.counted)
    if totals.empty:
        return QueryAnswer(
            question="", plan=plan, answer="No rows match that scope.", calculation=f"0 rows{scope}"
        )
    table = pd.DataFrame({
        key: list(totals.index),
        "Share %": (hits / totals * 100).round(1).to_numpy(),
        "Rows meeting": hits.to_numpy(),
        "Rows": totals.to_numpy(),
    })
    if ordered:
        order = list(dict.fromkeys(groups.dropna().iloc[np.argsort(_order_keys(recorded, plan, groups))]))
        rest = [value for value in table[key] if value not in order]
        table = table.set_index(key).loc[[*order, *rest]].reset_index()
    else:
        table = table.sort_values("Share %", ascending=plan.ascending, kind="stable").reset_index(drop=True)
    # Rows with no group are listed last: they are not a group to name first.
    last = table[key].astype(str) == missing_label
    table = pd.concat([table[~last], table[last]]).reset_index(drop=True)
    frame = pd.DataFrame({key: groups[~unlabelled], "__counted": meets[~unlabelled].astype(int),
                          "__cases": cases[~unlabelled]})
    comparison = compare_rates(frame, "__counted", key, weight="__cases")
    grouping = _grouping_name(plan)

    def label(value: object) -> str:
        return str(value) if names else group_label(key, value)

    def stated(row: pd.Series) -> str:
        return f"{_share_text(row['Share %'])} ({int(row['Rows meeting']):,} of {int(row['Rows']):,})"

    named = table[table[key].astype(str) != missing_label]
    if plan.intent == "rank" and len(named):
        large = named[named["Rows"] >= MIN_NAMED_GROUP]
        candidates = large if len(large) else named
        ranked = candidates.sort_values("Share %", ascending=plan.ascending, kind="stable")
        top = ranked.iloc[0]
        side = "lowest" if plan.ascending else "highest"
        answer = (
            f"{label(top[key])} has the {side} share of {unit} with {described} of any {grouping}"
            f"{_phrase(applied)}, at {stated(top)}."
        )
        if plan.top_n and plan.top_n > 1:
            table = ranked.head(plan.top_n).reset_index(drop=True)
        small = len(named) - len(large)
        if len(large) and small:
            answer += (
                f" {small:,} {grouping} group{'s' if small > 1 else ''} with fewer than {MIN_NAMED_GROUP} "
                f"{unit} {'are' if small > 1 else 'is'} not ranked."
            )
    else:
        listed = [f"{label(row[key])} {stated(row)}" for _, row in table.head(4).iterrows()]
        rest = len(table) - 4
        more = f", and {rest:,} more group{'s' if rest > 1 else ''}" if rest > 0 else ""
        answer = (
            f"Share of {unit} with {described} within each {grouping}{_phrase(applied)}: "
            f"{'; '.join(listed)}{more}."
        )
    asked = next((item.values for item in plan.filters if item.column == plan.dimension), ())
    if comparison is not None:
        if plan.intent == "compare" and plan.aggregation in ("max", "min") and len(asked) == 2 and (
            len(comparison.groups) == 2
        ):
            # "Is the death rate higher in grade 3 than in grade 1" gets its yes or no first.
            first = next(
                (group for group in comparison.groups if _same_value(group.value, asked[0])),
                comparison.groups[0],
            )
            other = next(group for group in comparison.groups if group is not first)
            if comparison.verdict != "clear":
                answer = f"Not clearly. {answer}"
            elif (first.rate > other.rate) == (plan.aggregation == "max"):
                answer = f"Yes. {answer}"
            else:
                answer = f"No, the other way round. {answer}"
        answer += _share_difference(comparison)
    if weight:
        answer += f" Each row counts {weight} cases, the number its combination of values holds."
    kept = f" with {_and_listed(columns)} recorded" if columns else ""
    calculation = f"rows with {described}, divided by the rows of each {key}{kept}"
    if weight:
        calculation = (
            f"sum({weight}) over the rows with {described}, divided by sum({weight}) over the rows of "
            f"each {key}{kept}"
        )
    if comparison is not None:
        calculation += f"; {rate_calculation(comparison).split('; ', 1)[1]}"
    exact = hits / totals * 100
    if weight:
        table = table.rename(columns={"Rows meeting": "Cases meeting", "Rows": "Cases"})
    return QueryAnswer(
        question="", plan=plan, answer=answer, calculation=f"{calculation}{scope}",
        table=table, chart="line" if plan.grain else "bar",
        values=_labelled_values(table[key], table[key].map(exact)),
    )


def _counted_mask(frame: pd.DataFrame, counted: tuple[ValueFilter | Condition, ...]) -> pd.Series:
    """Rows that meet every value and condition a share counts."""
    mask = pd.Series(True, index=frame.index)
    for item in counted:
        if isinstance(item, ValueFilter):
            mask &= _value_mask(frame[item.column], item.values)
        else:
            mask &= _condition_mask(frame[item.column], item).fillna(False).astype(bool)
    return mask


def _describe_counted(item: ValueFilter | Condition) -> str:
    return _describe_filter(item) if isinstance(item, ValueFilter) else _describe_condition(item)


def _share_groups(frame: pd.DataFrame, plan: QueryPlan) -> tuple[pd.Series, bool]:
    """Each row's group for a share (missing when it has none), and whether the groups have an order.

    Periods of the date, parts of the date and bins keep their order; other
    groups are listed by their share.
    """
    assert plan.dimension is not None
    column = frame[plan.dimension]
    if plan.split is not None:
        return _split_labels(column, plan.split).astype(object), True
    if plan.grain:
        return column.dt.to_period(plan.grain).dt.start_time, True
    if plan.time_part:
        return _time_part_values(column, plan.time_part), True
    if plan.bins:
        return _binned(column, plan.bins).astype(object), True
    return column, False


def _order_keys(frame: pd.DataFrame, plan: QueryPlan, groups: pd.Series) -> np.ndarray:
    """A sort key for each grouped row that keeps periods, date parts and bins in their order."""
    assert plan.dimension is not None
    present = groups.notna()
    column = frame.loc[present, plan.dimension]
    if plan.split is not None:
        return _split_labels(column, plan.split).cat.codes.to_numpy()
    if plan.grain:
        return column.to_numpy(dtype="datetime64[ns]").astype("int64")
    if plan.time_part:
        return _time_part_values(column, plan.time_part).to_numpy(dtype=float)
    return _binned(frame[plan.dimension], int(plan.bins or 2))[present].cat.codes.to_numpy()


def _share_names(plan: QueryPlan):
    """How a share's groups are named, when not by their own values: periods, months, weekdays."""
    if plan.grain:
        return lambda value: _period_words(value, str(plan.grain))
    names = _part_names(plan.time_part)
    if names:
        return lambda value: names.get(int(value), value)
    return None


def _share_difference(comparison: RateComparison) -> str:
    """One sentence on whether the shares differ, judged as the brief's rate card judges them."""
    c = comparison
    high, low = c.higher, c.lower
    evidence = f"{c.test}, {format_p(c.p_value)}; Cramér's V = {c.cramers_v:.2f}"
    pair = "" if len(c.groups) == 2 else f" between {high.label} and {low.label}"
    gap = (
        f"a difference of {c.difference * 100:.1f} percentage points{pair} "
        f"(95% CI {c.low * 100:.1f} to {c.high * 100:.1f})"
    )
    verdict = c.verdict
    if verdict == "clear":
        return f" The shares differ: {gap}; {evidence}."
    if verdict == "negligible":
        return f" The shares differ only by a negligible amount: {gap}; {evidence}."
    if verdict == "thin":
        small = high if high.n <= low.n else low
        return f" {small.n:,} rows in {small.label} are too few to call the gap clear: {gap}; {evidence}."
    return f" The shares do not clearly differ: {gap}; {evidence}."


def _execute_column_differences(
    plan: QueryPlan, working: pd.DataFrame, roles: ColumnRoles, scope: str, applied: list[str]
) -> QueryAnswer:
    """Compare each column of a set across the plan's groups, with p-values adjusted over every test.

    The adjustment is the scan's own (Holm, or the false-discovery rate for a
    screen of very many columns), and is named as the brief names it.
    """
    assert plan.dimension is not None
    columns = [column for column in plan.columns if column in working.columns and column != plan.dimension]
    entity = entity_column(working, roles)
    scan = scan_differences(working, columns, [plan.dimension], entity=entity, limit=10)
    groups = next((item.values for item in plan.filters if item.column == plan.dimension), ())
    between = f"between {_and_listed(list(groups))}" if groups else f"across {plan.dimension}"
    # The groups named are the comparison, not a narrowing of the rows to state again.
    picked = {_describe_filter(item) for item in plan.filters if item.column == plan.dimension}
    applied = [text for text in applied if text not in picked]
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
    adjusted = ADJUSTED[scan.correction]
    if scan.named:
        shown = "; ".join(
            f"{item.measure} ({lead_clause(item)}, {adjusted} {format_p(item.decisive_p)})"
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
            f"{format_p(largest.p_value)} before adjustment, {adjusted} {format_p(largest.decisive_p)})."
        )
    table = pd.DataFrame({
        "Column": [item.measure for item in (scan.named or (largest,))],
        "Higher group": [item.higher.label for item in (scan.named or (largest,))],
        "Effect": [effect_text(item) for item in (scan.named or (largest,))],
        "p": [item.p_value for item in (scan.named or (largest,))],
        f"{adjusted[0].upper()}{adjusted[1:]} p": [item.decisive_p for item in (scan.named or (largest,))],
    })
    return QueryAnswer(
        question="", plan=plan, answer=answer,
        calculation=(
            f"each column compared {between} (Welch's t-test or ANOVA, or rank tests for skewed columns), "
            f"p-values {adjusted} over {scan.tested:,} tests ({ADJUSTED_BY[scan.correction]}){scope}"
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
    # The row mean ADA built from a matrix's columns holds each of them, so the brief never
    # relates them to it, and neither does Ask.
    summaries = [
        column for column in columns
        if ROW_SUMMARY.fullmatch(str(column)) and column != plan.measure
        and plan.measure in row_summary_columns(roles, working)
    ]
    columns = [column for column in columns if column not in summaries]
    scan = scan_relationships(working, columns, plan.measure, limit=BREAKDOWN_LIMIT, others=False)
    if scan is None or not scan.correlates:
        if scan is not None and scan.set_aside:
            answer = (
                f"Every other numeric column that shares rows with {plan.measure}{_phrase(applied)} "
                f"repeats it or is built from it, so nothing is left to rank.{set_aside_note(scan)}"
            )
        else:
            answer = (
                f"Nothing can be correlated with {plan.measure}{_phrase(applied)}: "
                f"{_uncorrelated_reason(plan.measure, working, roles, columns)}."
            )
        return QueryAnswer(
            question="",
            plan=plan,
            answer=answer,
            calculation=f"correlations need {MIN_CORRELATION_PAIRS} complete pairs{scope}",
        )
    top = scan.correlates[0]
    # Columns set aside as copies of the measure, or built from it, were never candidates.
    tested = scan.measure_tests - len(scan.set_aside)
    if top.strength < NOTABLE_CORRELATION:
        figures = f"r = {top.pearson:.2f}, Spearman {top.spearman:.2f}, n = {top.pairs:,}"
        if tested == 1:
            answer = (
                f"Nothing relates to {plan.measure}{_phrase(applied)}: {top.second}, the only other "
                f"numeric column tested, falls short of |r| = {NOTABLE_CORRELATION} ({figures})."
            )
        else:
            answer = (
                f"Nothing relates to {plan.measure}{_phrase(applied)}: none of the {tested} numeric columns "
                f"tested reaches |r| = {NOTABLE_CORRELATION}. The closest is {top.second} ({figures})."
            )
    elif tested == 1:
        detail = measure_correlation(working[[plan.measure, top.second]].dropna())
        assert detail is not None
        answer = (
            f"{top.second} is the only other numeric column tested against {plan.measure}. "
            f"{describe_correlation(detail, _phrase(applied))}"
        )
    else:
        detail = measure_correlation(working[[plan.measure, top.second]].dropna())
        assert detail is not None
        answer = (
            f"{top.second} relates most to {plan.measure} of the {tested} numeric columns tested. "
            f"{describe_correlation(detail, _phrase(applied))}"
        )
        following = [pair for pair in scan.correlates[1:3] if pair.strength >= NOTABLE_CORRELATION]
        # The ranking uses each pair's strength: the larger of |Spearman r| and |r| without its most
        # influential record. When r and Spearman alone would order the pairs differently, the
        # strengths are stated, so the order shown is the order of the figures given.
        shown = [max(abs(pair.pearson), abs(pair.spearman)) for pair in (top, *following)]
        ranked_by = any(later > earlier + 0.005 for earlier, later in zip(shown, shown[1:], strict=False))
        if ranked_by:
            answer += (
                f" It ranks first at {top.strength:.2f}, the larger of its rank correlation and its r "
                "without the most influential record."
            )
        if following:
            answer += " Next: " + ", ".join(
                f"{pair.second} ({f'ranked at {pair.strength:.2f}; ' if ranked_by else ''}"
                f"r = {pair.pearson:.2f}, Spearman {pair.spearman:.2f})"
                for pair in following
            ) + "."
        answer += (
            " It was picked from several columns, so treat it as a lead to check, not as evidence of cause."
        )
    answer += set_aside_note(scan)
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
    left_out = [f"{column}, the row mean built from {plan.measure} and the others" for column in summaries]
    left_out += [item.describe() for item in scan.derived if plan.measure in (item.column, *item.parts)]
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
    return QueryAnswer(
        question="",
        plan=plan,
        answer=answer,
        calculation=f"{calculation}{scope}",
        table=table,
        chart="bar",
    )


def _uncorrelated_reason(
    measure: str, working: pd.DataFrame, roles: ColumnRoles, readings: list[str]
) -> str:
    """Why no column could be correlated with ``measure``, column by column, from the real reasons."""
    reasons: dict[str, list[str]] = defaultdict(list)
    for column in roles.numeric:
        if column == measure or column not in working.columns:
            continue
        values = working[column]
        paired = int(working[[measure, column]].dropna().shape[0])
        if values.nunique(dropna=True) < 2:
            reasons["never vary"].append(column)
        elif _is_flag(values):
            reasons["hold only 0 and 1, so they are compared as groups rather than correlated"].append(column)
        elif column not in readings:
            reasons["are groupings or labels rather than readings"].append(column)
        elif paired < MIN_CORRELATION_PAIRS:
            reasons[f"share fewer than {MIN_CORRELATION_PAIRS} rows with it"].append(column)
    if not reasons:
        return "the table has no other numeric column"
    return "; ".join(
        f"{_leaders(columns, 4)} {why if len(columns) > 1 else _singular(why)}"
        for why, columns in reasons.items()
    )


def _singular(phrase: str) -> str:
    """The verb of a reason for one column: "never vary" becomes "never varies"."""
    words = {"vary": "varies", "hold": "holds", "are": "is", "share": "shares", "groupings": "a grouping",
             "labels": "a label", "readings": "a reading", "they": "it"}
    return " ".join(words.get(word, word) for word in phrase.split())


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
    # Two years named ("from 2023 to 2024") are compared as whole years, for each group too.
    named = _named_years(plan)
    grain = plan.grain or ("Y" if named else None)
    series = build_trend(working, scoped_roles, frequency=grain)
    trend = series.frame
    if len(trend) < 2:
        return QueryAnswer(
            question="",
            plan=plan,
            answer="Not enough history in that scope to measure change.",
            calculation=f"growth needs at least 2 periods{scope}",
        )
    previous_period, current_period = trend.iloc[-2]["Period"], trend.iloc[-1]["Period"]
    spanned = bool(named) and (trend.iloc[0]["Period"].year, trend.iloc[-1]["Period"].year) == named
    if spanned:
        previous_period = trend.iloc[0]["Period"]

    if plan.dimension:
        grain = grain or preferred_frequency(working[roles.date])
        when = f"from {named[0]} to {named[1]}" if spanned else "in the latest period"
        versus = f"{named[1]} vs {named[0]}" if spanned else "latest vs previous period"
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
            return _rank_level_changes(plan, result, working, scoped_roles, scope, applied, when)
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
            f"{leader['Change %']:+.1f}% ({start} → {end}) {when}."
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
                f"per-{plan.dimension} {measure or 'row count'}: {versus}, ranked by % change{scope}"
            ),
            table=result.head(BREAKDOWN_LIMIT),
            chart="bar",
        )

    previous, current = float(trend.iloc[-2]["Value"]), float(trend.iloc[-1]["Value"])
    prior, latest = "previous period", "latest"
    if spanned:
        # "between 1982 and 1988" compares the two years named, not the last two in the range.
        previous, current = float(trend.iloc[0]["Value"]), float(trend.iloc[-1]["Value"])
        prior, latest = str(named[0]), str(named[1])
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
    versus = "the prior period" if prior == "previous period" else prior
    answer = (
        f"{measure or 'Records'}{_phrase(applied)} is {direction} {size} versus {versus} "
        f"({start} → {end})."
    )
    # format_change gave a size, so the base is not zero.
    relative = ("change_percent", (current - previous) / abs(previous) * 100)
    if unit in ("rate", "level"):
        formula = f"{latest} − {prior} average{scope}"
        moved = ("change", current - previous)
        # A level that rises by a percent ("deaths fell by 19%") is a relative change; a rate's is in points.
        extra = (relative,) if unit == "level" else ()
    else:
        formula = f"({latest} − {prior}) ÷ |{prior}| on period {'totals' if measure else 'counts'}{scope}"
        moved = relative
        extra = ()
    return QueryAnswer(
        question="",
        plan=plan,
        answer=answer,
        calculation=_with_notes(formula, series),
        table=trend,
        chart="line",
        value=_as_number(moved[1]),
        figures=(moved, *extra, ("previous", previous), ("latest", current)),
    )


def _named_years(plan: QueryPlan) -> tuple[int, int] | None:
    """The first and last year of a range the question names: (1982, 1988) for "between 1982 and 1988"."""
    years = [item for item in plan.conditions if item.part == "year"]
    low = next((item.value for item in years if item.operator == ">="), None)
    high = next((item.value for item in years if item.operator == "<="), None)
    if low is None or high is None or not float(low).is_integer() or not float(high).is_integer():
        return None
    return int(low), int(high)


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
    when: str = "in the latest period",
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
        f"{size} ({start} → {end}) {when}."
    )
    unit_label = "percentage points" if unit == "rate" else "difference"
    return QueryAnswer(
        question="",
        plan=plan,
        answer=answer,
        calculation=(
            f"per-{plan.dimension} average {measure}: "
            f"{'latest − previous period' if when == 'in the latest period' else when}, "
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
    checks: tuple[QualityIssue, ...] | None = None,
    choices: Mapping[str, str] | None = None,
) -> QueryAnswer | None:
    """Parse and execute in one step; None means the rules could not read it.

    ``rows_not_analyzed`` is how many rows the row limit left out of
    ``dataframe``; the answer says so when it is not zero. ``whole`` holds a
    long table's every parameter when ``dataframe`` is the brief's focus on
    one: a question naming another parameter, or grouping by them, is answered
    from it, and every other answer says it covers the focus only. ``checks``
    is the data checks' scan of ``dataframe`` when the caller has it, so the
    answers use the very issues the brief reports instead of scanning again.
    ``choices`` pick options for the decisions the answer takes (see
    ``ada.decisions``); the answer's ``decisions`` list each one taken.
    """
    if checks is not None:
        _remember_checks(dataframe, roles, checks)
    if whole is not None and (wider := _whole_table_answer(question, whole, rows_not_analyzed)):
        return wider
    if whole is not None and (elsewhere := _part_answer(question, dataframe, whole, rows_not_analyzed)):
        return elsewhere
    plan = parse_question(question, dataframe, roles)
    if plan is None:
        return None
    plan, decisions = _decide_aggregates(plan, dataframe, roles, choices or {})
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
    return replace(result, question=question, answer=answer, decisions=decisions)


def _decide_aggregates(
    plan: QueryPlan, dataframe: pd.DataFrame, roles: ColumnRoles, choices: Mapping[str, str]
) -> tuple[QueryPlan, tuple[Decision, ...]]:
    """Leave out the rows that add up others (a World row, regions), or keep them, as ``choices`` say.

    Nothing is decided when the table has no such rows (``ada.aggregates``),
    when the question picks rows of their column itself ("co2 of World"), or
    when none is in its scope. By default they are left out, and a total over
    the whole scope is read from the table's one total row (World, Total) in
    it, with what the other rows add up to beside it.
    """
    found = aggregate_rows(dataframe)
    if found is None or plan.intent == "duplicates":
        return plan, ()
    if any(item.column == found.column for item in (*plan.filters, *plan.conditions, *plan.counted)):
        return plan, ()
    try:
        working, _ = _apply_filters(dataframe, plan, roles)
    except (TimeScopeUnavailable, ValueError):
        return plan, ()
    aggregates = found.mask(working)
    seen = set(working.loc[aggregates, found.column].astype(str))
    present = [label for label in found.labels if label in seen]
    if not present:
        return plan, ()
    rows = int(aggregates.sum())
    leave = decide(
        "aggregate_rows", choices,
        f"Leave out the {counted(rows, 'row')} that add up other rows ({found.named(present)})?",
        f"They add up other values of {found.column}, so counting them with those values counts some twice.",
    )
    if leave.chosen == "keep":
        return plan, (leave,)
    totals = [label for label in present if label in found.totals]
    labels = (*found.totals, *(label for label in found.labels if label not in found.totals))
    left_out = replace(plan, excluded=(*plan.excluded, ValueFilter(found.column, labels)))
    if not (plan.intent == "aggregate" and plan.aggregation == "sum" and plan.measure and not plan.per
            and len(totals) == 1):
        return left_out, (leave,)
    total = totals[0]
    source = decide(
        "total_from", choices,
        f"Read the total from the {total} row, or add up the other rows?",
        f"The table gives {total} as the total of the other rows, which can hold what no other row does.",
    )
    if source.chosen == "sum_of_parts":
        return left_out, (leave, source)
    parts = float(to_numbers(working.loc[~aggregates, plan.measure]).sum())
    note = (
        f"Read the total from the {total} row; the other {found.column} values add up to "
        f"{format_number(parts, plan.measure, compact=False)}."
    )
    return replace(
        plan, filters=(*plan.filters, ValueFilter(found.column, (total,))), notes=(*plan.notes, note)
    ), (leave, source)


def _named_columns(question: str, frame: pd.DataFrame) -> list[str]:
    return list(_mentions(_norm(_question_text(question)), _name_index(frame.columns)))


def _parts_recording(whole: WholeTable, columns: list[str], by: str | None = None) -> list[str]:
    """The values of ``by`` (the parameter) whose rows record every one of ``columns``, in table order.

    Rows with no value of ``by`` are no part anyone can name, so they are left out.
    """
    frame = whole.frame
    by = by or whole.parameter
    present = frame[frame[by].notna()]
    recorded = present[columns].notna().groupby(present[by].astype(str), sort=False).any()
    parts = [str(part) for part, row in recorded.iterrows() if bool(row.all())]
    return [part for part in parts if by != whole.parameter or part != whole.focus]


def _stacking_column(whole: WholeTable, columns: list[str]) -> str:
    """The column whose values mark the part of the file that records ``columns``: the parameter first.

    A file stacking several tables can mark them in a column of its own
    ("source_table") while the brief follows a long table inside one of them.
    That column is used when the parameter has no part recording them and
    every row recording all of them holds one of its values. The column whose
    part records them on the largest share of its rows wins, then the first.
    """
    if _parts_recording(whole, columns):
        return whole.parameter
    frame = whole.frame
    recording = frame[columns].notna().all(axis=1)
    if not recording.any():
        return whole.parameter
    best: tuple[float, str] | None = None
    for column in frame.columns:
        if column == whole.parameter or is_numeric_dtype(frame[column]):
            continue
        held = frame.loc[recording, column]
        if held.isna().any() or held.nunique() != 1 or not 2 <= frame[column].nunique() <= GROUPING_LIMIT:
            continue
        share = float(recording.sum()) / float((frame[column] == held.iloc[0]).sum())
        if best is None or share > best[0]:
            best = (share, str(column))
    return best[1] if best else whole.parameter


def _unit_reading(parameter: str, columns: pd.Index) -> str | None:
    """The one column a unit column describes, by name: "abundance unit" describes abundance.

    A column called just "unit" describes every reading beside it, so None.
    """
    words = _norm(parameter).split()
    if len(words) < 2 or words[-1] not in ("unit", "units", "uom"):
        return None
    named = [column for column in columns if _norm(column) == " ".join(words[:-1])]
    return named[0] if len(named) == 1 else None


def _part_frame(whole: WholeTable, part: str, by: str | None = None) -> tuple[pd.DataFrame, ColumnRoles]:
    """The rows of one part of the file, with roles regrouped for it as the brief would."""
    # Imported here because the pipeline itself builds on this module.
    from .pipeline import apply_focus

    if by and by != whole.parameter:
        return whole.frame[whole.frame[by].astype(str) == part], whole.roles
    roles = replace(whole.roles, dimension=whole.parameter, focus=None)
    return apply_focus(whole.frame, roles, part)


def _answer_in_part(
    question: str, whole: WholeTable, part: str, rows_not_analyzed: int, by: str | None = None
) -> tuple[QueryAnswer | None, pd.DataFrame, ColumnRoles, WholeTable | None]:
    """Answer within one part of a stacked file, with the roles detected on that part's own columns.

    Another part has its own columns, which the roles detected for the part
    the brief follows know nothing of. A part that is itself a long table (a
    unit beside each value) follows one of its values, as the brief would.
    ``by`` names the column that marks the part, the parameter by default.
    Returns the answer with the frame, roles and long table it came from.
    """
    # Imported here because the pipeline itself builds on this module.
    from .pipeline import apply_focus

    frame, _ = _part_frame(whole, part, by)
    frame = frame.loc[:, frame.notna().any().to_numpy()]
    roles = detect_roles(frame)
    inner = None
    plan = parse_question(question, frame, replace(roles, focus=None)) if roles.focus else None
    # Only values in different units need the one-unit view; counting rows does not, nor does a
    # reading other than the one a unit column is named for ("abundance unit" is abundance's).
    described = _unit_reading(roles.dimension, frame.columns) if roles.dimension else None
    readings = (plan.measure, plan.partner) if plan else ()
    if roles.focus and roles.dimension and (plan is None or any(readings)) and (
        described is None or plan is None or described in readings
    ):
        inner = WholeTable(frame, replace(roles, focus=None), roles.dimension, roles.focus)
        frame, roles = apply_focus(frame, roles, roles.focus)
    roles = replace(roles, focus=None)
    found = answer_question(question, frame, roles, rows_not_analyzed=rows_not_analyzed, whole=inner)
    return found, frame, roles, inner


def _named_parts(question: str, whole: WholeTable) -> list[str]:
    """The parts of the file other than the focus that the question names by their value."""
    q = _norm(_question_text(question))
    parts = whole.frame[whole.parameter].dropna().astype(str).unique()
    return [
        part for part in parts
        if part != whole.focus and (text := _norm(part)) and re.search(_word_regex(text), q)
    ]


def _part_answer(
    question: str, dataframe: pd.DataFrame, whole: WholeTable, rows_not_analyzed: int
) -> QueryAnswer | None:
    """Answer within the one other part of the file that records the columns a question names.

    A file stacking several tables keeps each table's columns empty in the
    others. When the question names a column with no values in the part the
    brief follows, and exactly one other part records every column it names,
    the answer comes from that part and says so.
    """
    named = [column for column in _named_columns(question, whole.frame) if column != whole.parameter]
    empty = [column for column in named if column in dataframe.columns and dataframe[column].isna().all()]
    if not empty:
        return None
    by = _stacking_column(whole, named)
    parts = _parts_recording(whole, named, by)
    if len(parts) != 1:
        return None
    found, frame, roles, _ = _answer_in_part(question, whole, parts[0], rows_not_analyzed, by)
    if found is None:
        return None
    # A name inside a value the question types ("organ" in "human_organs_tissues") is no column it uses.
    empty = [column for column in empty if column in _plan_columns(found.plan, frame, roles)] or empty
    verb = "is" if len(empty) == 1 else "are"
    rows = int((whole.frame[by].astype(str) == parts[0]).sum())
    note = (
        f" {_and_listed(empty)} {verb} only recorded in the rows with {by} = {parts[0]} "
        f"({rows:,} rows), so this answer covers that part of the file, not the "
        f"{whole.parameter} = {whole.focus} part the brief follows."
    )
    return replace(found, answer=found.answer + note)


def _part_reason(question: str, dataframe: pd.DataFrame, whole: WholeTable) -> str | None:
    """Why a question can't be answered in the part of the file the brief follows, when that is the reason.

    A column it names may have no values there, recorded only in other parts,
    or a single value there although the file holds several.
    """
    named = [column for column in _named_columns(question, whole.frame) if column != whole.parameter]
    focus = f"the {whole.parameter} = {whole.focus} part the brief follows"
    if len(parts := _named_parts(question, whole)) == 1:
        # The question names its part, so the reason comes from that part's own rows.
        _, frame, roles, inner = _answer_in_part(question, whole, parts[0], 0)
        return rephrase_hint(question, frame, roles, whole=inner)
    for column in named:
        if column not in dataframe.columns:
            continue
        values = dataframe[column].dropna()
        if values.empty:
            if whole.frame[column].isna().all():
                return f"{column} has no values anywhere in the file."
            by = _stacking_column(whole, [column])
            parts = _parts_recording(whole, [column], by)
            if not parts:
                return f"{column} has no values in {focus}; it is recorded only in rows with no {by}."
            others = [other for other in named if other != column]
            together = _parts_recording(whole, named, _stacking_column(whole, named)) if others else parts
            where = _listed_capped(parts, 4)
            if others and not together:
                return (
                    f"{column} has no values in {focus}; it is recorded only where {by} is "
                    f"{where}, and no part of the file records {_and_listed(named)} together."
                )
            return (
                f"{column} has no values in {focus}; it is recorded where {by} is {where}. "
                f'Name one, as in "{question.strip().rstrip("?")} for {by} {parts[0]}".'
            )
        everywhere = whole.frame[column].dropna()
        if values.nunique() == 1 and everywhere.nunique() > 1:
            parts = whole.frame[whole.parameter].dropna().astype(str).unique()
            others = [part for part in parts if part != whole.focus]
            return (
                f"{column} is {values.iloc[0]} on every row of {focus}, so it has nothing to group or "
                f"compare there; the whole file has {everywhere.nunique():,} values of it. Name another "
                f"{whole.parameter} ({_listed_capped(others, 4)}) to ask within it."
            )
    return None


def _scope_words(plan: QueryPlan) -> list[str]:
    """The plan's filters and conditions in words, as an answer's scope."""
    return [_describe_filter(item) for item in plan.filters] + _describe_conditions(plan.conditions)


def _names_parameter(question: str, whole: WholeTable) -> bool:
    """Whether a question names a long table's parameter column or one of its other values.

    A quick screen, so a question about the focus is not read twice against every row.
    """
    q = _norm(_question_text(question))
    words = set(q.split())
    named = set(_partial_words([whole.parameter])[whole.parameter]) - QUERY_GRAMMAR
    if any(_word_variants(word) & words for word in named if len(word) >= 4):
        return True
    others = [value for value in whole.frame[whole.parameter].dropna().unique() if str(value) != whole.focus]
    return any((text := _norm(value)) and re.search(_word_regex(text), q) for value in others)


def _whole_table_answer(question: str, whole: WholeTable, rows_not_analyzed: int) -> QueryAnswer | None:
    """Answer from every parameter of a long table when the question picks another one, or groups by them."""
    if not _names_parameter(question, whole):
        return None
    frame, roles = whole.frame, replace(whole.roles, focus=None)
    plan = parse_question(question, frame, roles)
    if plan is None:
        # A question naming one other part of a stacked file is answered within it.
        parts = _named_parts(question, whole)
        if len(parts) != 1:
            return None
        found, *_ = _answer_in_part(question, whole, parts[0], rows_not_analyzed)
        return found
    picked = next((item.values for item in plan.filters if item.column == whole.parameter), ())
    grouped = whole.parameter in (plan.dimension, plan.second_dimension)
    if not grouped and (not picked or tuple(picked) == (whole.focus,)):
        return None
    try:
        result = execute_plan(plan, frame, roles, rows_not_analyzed=rows_not_analyzed)
    except ValueError:
        return None
    answer = result.answer
    if grouped and plan.intent in ("rank", "breakdown") and result.table is not None and (
        plan.aggregation != "count" and plan.measure
    ):
        # Values of different parameters are in different units, so none of them "leads";
        # a count of rows is the same unit in every part, and is ranked as usual.
        groups = plan.dimension if plan.dimension == whole.parameter else plan.second_dimension
        label = "row count" if plan.aggregation == "count" else with_aggregate(
            _aggregation_label(plan), str(plan.measure)
        )
        answer = (
            f"The table gives the {label} for each {groups}{_phrase(_scope_words(plan))}. Each "
            f"{whole.parameter} is measured on its own scale, often in its own unit, so they are listed, "
            "not ranked against each other."
        )
    elif grouped and plan.aggregation != "count" and plan.measure:
        answer += (
            f" Each {whole.parameter} is measured on its own scale, often in its own unit, so compare each "
            "only with itself, not with the others."
        )
    return replace(result, question=question, answer=answer)


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
        if (part := _part_reason(question, dataframe, whole)):
            return part
        dataframe, roles = whole.frame, replace(whole.roles, focus=None)
    reason = _plan_question(question, dataframe, roles, rows_not_analyzed)[1]
    if reason:
        return reason
    numeric_columns = [column for column in roles.numeric if column in dataframe.columns]
    index = _name_index(dataframe.columns)
    names = index.names
    mentioned = _mentions(q, index)
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
    # Money and counts are ranked and grown; a reading is compared and related, as a researcher would.
    additive = bool(roles.measure) and metric_for(roles.measure, roles).additive
    if roles.measure and roles.dimension:
        suggestions.append(
            f"Top 5 {roles.dimension} by {roles.measure}"
            if additive
            else f"Does {roles.measure} differ by {roles.dimension}?"
        )
    if roles.measure and roles.date:
        suggestions.append(f"Monthly {roles.measure} trend")
    if additive and roles.dimension and roles.date:
        suggestions.append(f"Which {roles.dimension} grew fastest?")
    elif roles.measure and len(roles.numeric) > 1:
        suggestions.append(f"What relates to {roles.measure}?")
    if roles.measure and len(suggestions) < 4:
        suggestions.append(f"Outliers in {roles.measure}")
    if not suggestions:
        suggestions.append("How many rows are there?")
    return suggestions[:4]
