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
from collections import defaultdict
from dataclasses import dataclass, field, fields, replace
from typing import Literal

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from .aggregation import TrendSeries, build_trend, preferred_frequency, unlabelled_label
from .business_insights import (
    MIN_CORRELATION_PAIRS,
    RANK_DIVERGENCE,
    describe_correlation,
    measure_correlation,
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
from .schema import ColumnRoles

Intent = Literal["aggregate", "count", "rank", "breakdown", "trend", "growth", "correlation"]
#: Plans answered from the column profile rather than by combining a measure.
ProfileIntent = Literal["missing"]
Aggregation = Literal["sum", "mean", "median", "min", "max", "count"]
Operator = Literal["=", "<", "<=", ">", ">=", "missing"]
TimePart = Literal["hour", "weekday"]

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

AGGREGATION_LABELS: dict[Aggregation, str] = {
    "sum": "Total",
    "mean": "Average",
    "median": "Median",
    "max": "Maximum",
    "min": "Minimum",
    "count": "Count of",
}

# Words asking for the bottom of a ranking. Each is also a superlative, so
# SUPERLATIVE_WORDS includes this tuple rather than repeating it.
ASCENDING_WORDS = ("worst", "bottom", "lowest", "smallest", "least", "fewest", "shortest")
SUPERLATIVE_WORDS = (
    # "most" ranks from the top, like "highest"; "least" is in ASCENDING_WORDS.
    "best", "top", "leading", "highest", "largest", "biggest", "most", "greatest", "longest",
    "tallest", "heaviest",
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
GRAIN_NAMES = {"D": "day", "W": "week", "M": "month", "Q": "quarter", "Y": "year"}
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
    "between", "by", "can", "column", "columns", "data", "did", "do", "does", "done", "each",
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
VALUE_CUES = frozenset({"for", "in", "is", "at", "where", "when", "equals", "of", "from"})
# Words before a 0/1 flag column that ask for the rows where it is set:
# "how many have hypertension". "without" asks for the rows where it is not.
FLAG_CUES = r"have|has|having|had|with|without|are|were"
# Grouping by a part of the date: "average PM2.5 by hour", "revenue by weekday".
TIME_PART_PATTERN = re.compile(
    r"\b(?:by|per|each|across|which|what)\s+(?:the\s+)?"
    r"(?:(?P<hour>hours?(?:\s+of\s+(?:the\s+)?day)?)|(?P<weekday>weekdays?|days?\s+of\s+(?:the\s+)?week))\b"
)


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


def _runs(reading: _Reading, *, content_only: bool) -> list[list[tuple[int, int, str]]]:
    """Unclaimed words in unbroken runs; a claimed word between two words breaks the run.

    With ``content_only`` the runs hold only words that are not query grammar.
    """
    runs: list[list[tuple[int, int, str]]] = [[]]
    for token in _tokens(reading.remaining):
        previous = runs[-1][-1] if runs[-1] else None
        if content_only and token[2] in QUERY_GRAMMAR:
            runs.append([])
        elif previous is None or reading.original[previous[1]:token[0]].strip() == "":
            runs[-1].append(token)
        else:
            runs.append([token])
    return [run for run in runs if run]


def _raw_words(question: str, words: list[str] | tuple[str, ...]) -> str:
    """The question's own spelling of some normalized words: "pH" for "ph"."""
    wanted = set(words)
    chunks = [
        chunk.strip("?.,!;:'\"")
        for chunk in _question_text(question).split()
        if set(_norm(chunk).split()) & wanted
    ]
    return " ".join(chunks) or " ".join(words)


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
        if len(candidates) > 1:
            role = _role_of_position(reading.text, start)
            suited = [column for column in candidates if _suits(column, role, dataframe, roles)]
            if len(suited) != 1:
                if size == 1 and (phrase[0] in ENTITY_NOUNS or role == "count"):
                    # Left for _claim_entity: "how many samples" counts rows.
                    return 1
                reading.flag(f"Did you mean {_listed(candidates)}?")
                reading.claim(start, end, keep_text=True)
                return size
            candidates = suited
        column = candidates[0]
        del words[column]
        reading.spans[column] = (start, end)
        reading.claim(start, end, keep_text=True)
        reading.notes.append(f'Read "{_raw_words(reading.question, phrase)}" as {column}.')
        found.append(column)
        return size
    return 1


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
    """A condition written after a numeric column: "pH below 5.5", "OBJECT is 0", "Status 2"."""
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
        elif numeric and (match := re.search(r"\b(?P<sign>negative|positive)\s+$", before)):
            reading.conditions.append(_sign_condition(column, match.group("sign")))
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
    """Read "by hour" or "by weekday" as a grouping by that part of the date column."""
    match = TIME_PART_PATTERN.search(reading.remaining)
    if match is None:
        return
    part: TimePart = "hour" if match.group("hour") else "weekday"
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
    read. A column called just "MAX" is the word "max" unless the question has
    another aggregation word or names it after "of" or a grouping word.
    """
    by_form = {form: column for column, forms in reading.names.items() for form in forms}
    outside = reading.remaining
    elsewhere = any(re.search(rf"\b{word}\b", outside) for word in AGGREGATION_WORDS)
    for column, (start, end) in list(reading.spans.items()):
        words = reading.text[start:end].split()
        if words[0] not in AGGREGATION_WORDS or elsewhere:
            continue
        if len(words) == 1:
            if not _follows(reading.text, start, rf"(?:of|{GROUPING_WORDS}|{RANKING_WORDS})"):
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


def _read_question(
    question: str, dataframe: pd.DataFrame, roles: ColumnRoles, rows_not_analyzed: int = 0
) -> _Reading:
    """Claim every word of the question for a column, a value, a condition or the grammar."""
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
    partial = _claim_partial_names(reading, dataframe, roles)
    _claim_conditions(reading, partial, dataframe, roles, vocabulary)
    _claim_values(reading, vocabulary, dataframe, exact_only=False)
    _claim_unit_conditions(reading, dataframe, roles)
    _claim_numbers(reading, dataframe, roles)
    _claim_entity(reading, dataframe, roles)
    for start, end, word in _tokens(reading.remaining):
        if word in MISSING_WORDS:
            reading.asks_missing = True
            reading.claim(start, end)
    _, reading.month = _detect_time_filter(reading.text)
    unknown = [word for _, _, word in _tokens(reading.remaining) if word not in QUERY_GRAMMAR]
    if unknown:
        reading.flag(_unknown_reason(question, unknown, len(dataframe), rows_not_analyzed))
    return reading


def _unknown_reason(question: str, unknown: list[str], rows: int, rows_not_analyzed: int) -> str:
    phrase = _raw_words(question, unknown)
    reason = f'ADA found no column or value called "{phrase}" in this table'
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
    if is_numeric_dtype(series) and not is_bool_dtype(series):
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
        if _can_group_numeric(dataframe[grouping]):
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
    """
    return _plan_question(question, dataframe, roles)[0]


def _plan_question(
    question: str, dataframe: pd.DataFrame, roles: ColumnRoles, rows_not_analyzed: int = 0
) -> tuple[QueryPlan | None, str | None]:
    """The plan for a question, or None with the reason it was refused when there is one."""
    if not _norm(question) or unsupported_phrasing(question):
        return None, None
    reading = _read_question(question, dataframe, roles, rows_not_analyzed)
    if reading.problem:
        return None, reading.problem
    if reading.asks_missing:
        plan = _missing_plan(reading)
    else:
        plan = _plan_from_reading(reading, dataframe, roles)
    if plan is None:
        return None, reading.refusal
    problem = _unused_part(plan, reading, dataframe, roles, rows_not_analyzed)
    if problem:
        return None, problem
    notes = (*plan.notes, *reading.notes, *_entity_note(reading, plan))
    return replace(plan, notes=notes), None


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


def _plan_from_reading(reading: _Reading, dataframe: pd.DataFrame, roles: ColumnRoles) -> QueryPlan | None:
    """Choose the intent, measure and grouping for a question whose words are all read."""
    q = reading.text
    numeric_columns = [column for column in roles.numeric if column in dataframe.columns]
    names = reading.names
    mentioned = reading.mentioned
    # "which month had the highest wind" ranks the periods of the date column;
    # the grain word names the period, not a column that happens to be called "month".
    period = re.search(r"\b(?:which|what)\s+(day|week|month|quarter|year)\b", q) if roles.date else None
    year, month = reading.year, reading.month
    grain = _detect_grain(q)
    filters_including_dimension = reading.value_filters
    conditions = tuple(reading.conditions)

    unnamed = _without_columns(q, tuple(mentioned), names)
    if any(re.search(rf"\b{word}\b", unnamed) for word in CORRELATION_WORDS):
        correlation = _correlation_plan(
            q, numeric_columns, mentioned, filters_including_dimension, year, month
        )
        return replace(correlation, conditions=conditions) if correlation else None

    chosen = _choose_columns(q, dataframe, roles, names, mentioned)
    groupings = [column for column, start in mentioned.items() if _follows(q, start)]
    if chosen is None:
        if len(groupings) > 1:
            reading.refusal = _two_groupings(groupings)
        return None
    measure, dimension = chosen
    time_part = reading.time_part
    if time_part:
        if dimension:
            reading.refusal = _two_groupings([dimension, f"the {time_part}"])
            return None
        dimension = roles.date
    # "revenue by order date" asks for the trend; with a segment as well it asks for two groupings.
    by_date = roles.date in mentioned and _follows(q, mentioned[roles.date], GROUPING_WORDS)
    if by_date and dimension:
        reading.refusal = _two_groupings([dimension, str(roles.date)])
        return None
    # A grouped answer groups by the matched dimension, so its values are not
    # filters. Ungrouped answers keep them: "revenue for Region West" filters
    # to West even though "Region" also matched as the dimension.
    filters = tuple(item for item in filters_including_dimension if item.column != dimension)

    # Read intent words with the matched column names removed (see _without_columns).
    intent_text = _without_columns(q, (measure, dimension), names)
    aggregation: Aggregation | None = next(
        (
            AGGREGATION_WORDS[word]
            for word in AGGREGATION_WORDS
            if re.search(rf"\b{word}\b", intent_text)
        ),
        None,
    )
    top_match = re.search(r"\b(top|bottom)\s+(\d{1,3})\b", q)
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
    }

    ungrouped = {**base, "filters": filters_including_dimension}
    # Same default as the dashboard: rates and levels average, amounts sum. An
    # explicit "total" or "sum" still sums. It is taken from the final measure, so
    # every intent combines a given measure the same way.
    default_aggregation = metric_for(base["measure"], roles).aggregation

    if wants_count and not wants_growth:
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
        if not is_numeric_dtype(series) or is_bool_dtype(series) or column == roles.identifier:
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
    grain = _detect_grain(reading.text)
    periods = plan.intent in ("trend", "growth") or (plan.intent == "rank" and not plan.dimension)
    if grain and not periods:
        if not roles.date:
            return f"This table has no date column, so there is no {GRAIN_NAMES[grain]} view."
        return (
            f"The answer would leave out the {GRAIN_NAMES[grain]} grouping; "
            "ask for one grouping at a time."
        )
    if reading.time_part and plan.time_part != reading.time_part:
        return f"The answer would leave out the {reading.time_part} grouping."
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
    if aggregation in ("mean", "median", "sum"):
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


def _aggregate_series(series: pd.Series, aggregation: Aggregation) -> float:
    return float(getattr(series.dropna(), aggregation)())


def _group_key(plan: QueryPlan, dataframe: pd.DataFrame) -> str:
    """The column a grouped answer's table is keyed by: the grouping, or "Hour" or "Weekday"."""
    assert plan.dimension is not None
    if not plan.time_part:
        return plan.dimension
    name = plan.time_part.capitalize()
    return f"{name} of {plan.dimension}" if name in dataframe.columns else name


def _grouped_frame(
    dataframe: pd.DataFrame, plan: QueryPlan, value_label: str
) -> pd.DataFrame:
    assert plan.dimension is not None
    key = _group_key(plan, dataframe)
    needed = list(dict.fromkeys(
        column for column in (plan.dimension, plan.measure, plan.count_column) if column
    ))
    if plan.time_part:
        labelled = dataframe.loc[dataframe[plan.dimension].notna(), needed].copy()
        stamps = labelled[plan.dimension]
        labelled[key] = stamps.dt.hour if plan.time_part == "hour" else stamps.dt.dayofweek
    else:
        # Keep unlabelled rows as their own group so shares add up to the full total.
        labelled = dataframe[needed].copy()
        labelled[key] = labelled[key].astype(object).where(
            labelled[key].notna(),
            unlabelled_label(labelled[key].dropna().unique()),
        )
    working = labelled
    if plan.aggregation == "count" or not plan.measure:
        if plan.count_column:
            grouped = working.groupby(key, as_index=False)[plan.count_column].nunique()
        else:
            grouped = working.groupby(key, as_index=False).size()
        grouped.columns = [key, value_label]
    else:
        grouped = working.groupby(key, as_index=False)[plan.measure].agg(plan.aggregation)
        grouped.columns = [key, value_label]
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
    try:
        working, _ = _apply_filters(dataframe, plan, roles)
    except TimeScopeUnavailable:
        working = None
    # A count says its own rows, and an unfiltered aggregate says how many rows it used.
    if working is not None and not working.empty and plan.intent not in ("count", "missing"):
        if len(working) < len(dataframe):
            sentences.append(f"Based on {len(working):,} of {len(dataframe):,} rows.")
        elif plan.intent != "aggregate":
            sentences.append(f"Based on all {len(dataframe):,} rows.")
    if rows_not_analyzed:
        # A dated table keeps its newest rows; one without a date is sampled at random.
        if roles.date:
            sentences.append(
                f"The row limit left out {rows_not_analyzed:,} older rows, so this answer covers "
                f"only the {len(dataframe):,} rows ADA analysed."
            )
        else:
            sentences.append(
                f"The row limit left out {rows_not_analyzed:,} rows, so this answer covers only "
                f"a random sample of {len(dataframe):,}."
            )
    return replace(result, answer=" ".join(sentences))


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
        value = _aggregate_series(working[plan.measure], plan.aggregation)
        label = AGGREGATION_LABELS[plan.aggregation]
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
        return QueryAnswer(
            question="",
            plan=plan,
            answer=(
                f"{with_aggregate(label, plan.measure)}{_phrase(applied)} is "
                f"{format_number(value, plan.measure, column_values=working[plan.measure].dropna())}, "
                f"calculated from {rows:,} rows."
            ),
            calculation=f"{plan.aggregation}({plan.measure}){scope}",
        )

    if plan.intent == "correlation":
        return _execute_correlation(plan, working, scope, applied)

    if plan.intent == "rank" and plan.dimension is None and plan.grain:
        return _rank_periods(plan, working, roles, scope, applied)

    if plan.intent in ("rank", "breakdown"):
        return _execute_grouping(plan, working, scope, applied)

    if plan.intent == "trend":
        assert roles.date is not None
        scoped_roles = _scoped_roles(roles, plan.measure, None)
        grain = plan.grain or preferred_frequency(working[roles.date])
        series = build_trend(working, scoped_roles, frequency=grain, count_column=plan.count_column)
        trend = series.frame
        if len(trend) < 2:
            return QueryAnswer(
                question="",
                plan=plan,
                answer="Not enough periods in that scope to draw a trend.",
                calculation=f"trend needs at least 2 periods{scope}",
            )
        first, last = float(trend.iloc[0]["Value"]), float(trend.iloc[-1]["Value"])
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
        answer = (
            f"{plan.measure or counted} per {grain_name}{_phrase(applied)} moved from "
            f"{start} to {end} ({movement})."
        )
        return QueryAnswer(
            question="",
            plan=plan,
            answer=answer,
            calculation=_with_notes(
                f"{plan.aggregation}({plan.measure or 'rows'}) grouped per {grain_name}{scope}", series
            ),
            table=trend,
            chart="line",
        )

    if plan.intent == "growth":
        return _execute_growth(plan, working, roles, scope, applied)

    raise ValueError(f"Unsupported intent: {plan.intent}")


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


def _leaders(names: list[str], limit: int = 3) -> str:
    """Name up to ``limit`` items, then how many more: "A, B, C and 2 more"."""
    shown = [str(name) for name in names[:limit]]
    if len(names) > limit:
        return f"{', '.join(shown)} and {len(names) - limit} more"
    return _and_listed(shown)


def _execute_grouping(plan: QueryPlan, working: pd.DataFrame, scope: str, applied: list[str]) -> QueryAnswer:
    """Rank or break down the measure by a grouping column, or by the hour or weekday of the date.

    Numbered groups (timepoint 0, 1, 2...) are listed in their own order, up to
    GROUPING_LIMIT of them; a ranking still sorts them by value. A breakdown
    with more than twice EXTREME_GROUPS groups shows the highest and the lowest
    and says how many there are. When every group is one row (a label on each
    row), the answer names the row and its own value. Ties at the top are named.
    """
    assert plan.dimension is not None
    counted = plan.aggregation == "count" or not plan.measure
    if counted:
        value_label = f"Distinct {plan.count_column}" if plan.count_column else "Rows"
    else:
        value_label = with_aggregate(AGGREGATION_LABELS[plan.aggregation], plan.measure)
    grouped = _grouped_frame(working, plan, value_label)
    key = grouped.columns[0]
    single_rows = not counted and not plan.time_part and 1 < len(grouped) == len(working)
    if single_rows:
        assert plan.measure is not None
        grouped = grouped.rename(columns={value_label: plan.measure})
        value_label = plan.measure
    numbered = bool(plan.time_part) or is_numeric_dtype(working[plan.dimension])
    ordered = numbered and plan.intent == "breakdown" and len(grouped) <= GROUPING_LIMIT
    extremes = False
    if ordered:
        table = grouped.sort_values(
            key, key=lambda values: pd.to_numeric(values, errors="coerce"), kind="stable"
        ).reset_index(drop=True)
        order = f"in {plan.time_part or key} order"
    elif plan.intent == "rank":
        table = grouped.head(plan.top_n or BREAKDOWN_LIMIT)
        order = "ascending" if plan.ascending else "descending"
    else:
        extremes = len(grouped) > 2 * EXTREME_GROUPS
        if extremes:
            table = pd.concat([grouped.head(EXTREME_GROUPS), grouped.tail(EXTREME_GROUPS)])
        else:
            table = grouped
        table = table.reset_index(drop=True)
        order = "ascending" if plan.ascending else "descending"
    if plan.time_part == "weekday":
        names = dict(enumerate(calendar.day_name))
        table = table.assign(**{key: table[key].map(names)})
        grouped = grouped.assign(**{key: grouped[key].map(names)})

    best = grouped[value_label].iloc[0]
    leaders = grouped.loc[grouped[value_label] == best, key].tolist()
    tie = len(leaders) > 1
    named = list_names([group_label(key, value) if numbered else str(value) for value in leaders])
    share_note = ""
    if "Share %" in grouped.columns and not tie:
        share_note = f" ({grouped['Share %'].iloc[0]:.1f}% of the total)"
    # Format a row count as a plain number, not in the measure's unit, and pass
    # the column's values so codes in a money-named column don't print as money.
    if counted:
        value_text = format_number(float(best))
    else:
        value_text = format_number(float(best), plan.measure, column_values=working[plan.measure].dropna())
    high = "lowest" if plan.ascending else "highest"
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
        grouping = plan.time_part or plan.dimension
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
    if extremes:
        showing = f"the {EXTREME_GROUPS} highest and {EXTREME_GROUPS} lowest of {len(grouped):,}"
    else:
        cut = len(table) < len(grouped) and ordered
        showing = f"{len(table)} of {len(grouped)}" if cut else str(len(table))
    by = f"{plan.time_part} of {plan.dimension}" if plan.time_part else plan.dimension
    if counted:
        # Name what ran: row counts use .size(), which differs from
        # count(<measure>) when the measure has missing values.
        what = f"count distinct {plan.count_column}" if plan.count_column else "row count"
    else:
        what = f"{plan.aggregation}({plan.measure})"
    return QueryAnswer(
        question="",
        plan=plan,
        answer=answer,
        calculation=f"{what} by {by}, {order}, showing {showing}{scope}",
        table=table,
        chart="line" if ordered else "bar",
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
    # A rate leaves periods without rows empty rather than zero; they can't rank.
    trend = series.frame.dropna(subset=["Value"])
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
            f"{len(trend)} {grain_name}s{scope}",
            series,
        ),
        table=series.frame,
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
    question: str, dataframe: pd.DataFrame, roles: ColumnRoles, *, rows_not_analyzed: int = 0
) -> QueryAnswer | None:
    """Parse and execute in one step; None means the rules could not read it.

    ``rows_not_analyzed`` is how many rows the row limit left out of
    ``dataframe``; the answer says so when it is not zero.
    """
    plan = parse_question(question, dataframe, roles)
    if plan is None:
        return None
    try:
        result = execute_plan(plan, dataframe, roles, rows_not_analyzed=rows_not_analyzed)
    except ValueError:
        return None
    return QueryAnswer(
        question=question,
        plan=result.plan,
        answer=result.answer,
        calculation=result.calculation,
        table=result.table,
        chart=result.chart,
    )


def _listed(columns: tuple[str, ...] | list[str]) -> str:
    """Join names as "a", "a or b", "a, b or c"."""
    return columns[0] if len(columns) == 1 else f"{', '.join(columns[:-1])} or {columns[-1]}"


def _and_listed(columns: tuple[str, ...] | list[str]) -> str:
    """Join names as "a", "a and b", "a, b and c"."""
    return columns[0] if len(columns) == 1 else f"{', '.join(columns[:-1])} and {columns[-1]}"


def rephrase_hint(
    question: str, dataframe: pd.DataFrame, roles: ColumnRoles, *, rows_not_analyzed: int = 0
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
