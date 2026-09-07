"""Deterministic natural-language queries over the analyzed dataset.

Every question becomes an explicit, auditable ``QueryPlan``. The plan is
executed locally with pandas and returns the answer together with the exact
calculation that produced it. No network call or model is required; an
optional AI planner may emit the same ``QueryPlan`` shape for questions the
rules cannot parse, and it goes through the same local executor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from aggregation import TrendSeries, build_trend, measure_aggregation, preferred_frequency, unlabelled_label
from formatting import format_number, format_period, rate_scale
from metrics import resolve_metric
from schema import IDENTIFIER_NAME_WORDS, ColumnRoles, is_identifier_name

Intent = Literal["aggregate", "count", "rank", "breakdown", "trend", "growth"]
Aggregation = Literal["sum", "mean", "median", "min", "max", "count"]

AGGREGATION_WORDS: dict[str, Aggregation] = {
    "total": "sum",
    "sum": "sum",
    "overall": "sum",
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
    "peak": "max",
    "min": "min",
    "minimum": "min",
    "lowest": "min",
    "smallest": "min",
}

AGGREGATION_LABELS: dict[Aggregation, str] = {
    "sum": "Total",
    "mean": "Average",
    "median": "Median",
    "max": "Maximum",
    "min": "Minimum",
    "count": "Count of",
}

# Words asking for the bottom of a ranking. Every one of them is also a
# superlative, so the two tuples are composed rather than typed out twice --
# a word listed in only one of them reads as a request for the opposite end.
ASCENDING_WORDS = ("worst", "bottom", "lowest", "smallest", "least", "fewest")
SUPERLATIVE_WORDS = (
    "best", "top", "leading", "highest", "largest", "biggest",
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

MAX_FILTER_CANDIDATES = 200
BREAKDOWN_LIMIT = 12

QUERY_STOPWORDS = {
    "a", "about", "across", "all", "and", "are", "by", "can", "do", "does", "each",
    "for", "from", "have", "he", "how", "i", "in", "is", "it", "me", "many", "my",
    "of", "on", "or", "our", "please", "re", "s", "she", "show", "tell", "than", "that",
    "the", "there", "this", "to", "they", "ve", "we", "what", "which", "with", "you", "your",
}


@dataclass(frozen=True)
class ValueFilter:
    column: str
    values: tuple[str, ...]


@dataclass(frozen=True)
class QueryPlan:
    intent: Intent
    aggregation: Aggregation = "sum"
    measure: str | None = None
    dimension: str | None = None
    count_column: str | None = None
    top_n: int | None = None
    ascending: bool = False
    filters: tuple[ValueFilter, ...] = ()
    year: int | None = None
    month: int | None = None
    day: int | None = None
    quarter: int | None = None
    grain: str | None = None
    source: str = "rules"


@dataclass(frozen=True)
class QueryAnswer:
    question: str
    plan: QueryPlan
    answer: str
    calculation: str
    table: pd.DataFrame | None = None
    chart: Literal["bar", "line"] | None = None


def _norm(text: str) -> str:
    lowered = str(text).lower().replace("’", "'")
    lowered = re.sub(r"\b([a-z]+)'(s|re|ve|ll|d|m|t)\b", r"\1", lowered)
    lowered = re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", lowered)
    cleaned = re.sub(r"[^0-9a-z]+", " ", lowered)
    return " ".join(cleaned.split())


def _word_variants(normalized_name: str) -> set[str]:
    variants = {normalized_name, f"{normalized_name}s", f"{normalized_name}es"}
    if normalized_name.endswith("s"):
        variants.add(normalized_name[:-1])
    if normalized_name.endswith("y"):
        # "how many countries" asks about the Country column.
        variants.add(f"{normalized_name[:-1]}ies")
    return variants


def _mentioned(normalized_name: str, question: str) -> bool:
    return any(
        re.search(rf"\b{re.escape(variant)}\b", question)
        for variant in _word_variants(normalized_name)
        if variant
    )


class _Spans:
    """The stretches of a normalized question that something has claimed.

    A plan is accepted only once every meaningful span is claimed: by a
    column, a value, a date, a number the plan uses, or a word of the
    grammar. What is left over is what the plan would have silently
    ignored, and a question with an ignored part is refused rather than
    answered as a different, easier question.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.taken: list[tuple[int, int]] = []

    def _free(self, start: int, end: int) -> bool:
        return all(end <= taken_start or start >= taken_end for taken_start, taken_end in self.taken)

    def take(self, pattern: str, group: int = 0) -> re.Match[str] | None:
        """Claim the first free match of the pattern (or of one group), or None."""
        for match in re.finditer(pattern, self.text):
            start, end = match.span(group)
            if end > start and self._free(start, end):
                self.taken.append((start, end))
                return match
        return None

    def take_all(self, pattern: str) -> list[re.Match[str]]:
        found: list[re.Match[str]] = []
        while (match := self.take(pattern)) is not None:
            found.append(match)
        return found

    def release(self, start: int, end: int) -> None:
        self.taken = [span for span in self.taken if span != (start, end)]

    def blanked(self) -> str:
        """The text with every claimed span blanked out, positions kept.

        Grammar is read from this: a dimension the file calls "Change (pp)"
        has claimed its word, so "revenue by change" is a breakdown and not a
        request for growth.
        """
        characters = list(self.text)
        for start, end in self.taken:
            characters[start:end] = " " * (end - start)
        return "".join(characters)

    def leftover(self) -> str:
        return " ".join(self.blanked().split())


BREAKDOWN_PATTERN = r"\b(by|per|across|breakdown|split|each)\b"
COUNT_PATTERN = r"\b(how many|count|number of)\b"
TOP_PATTERN = r"\b(top|bottom)\s+(\d{1,3})\b"
# A one- or two-character value ("UK", "A") is only read as a filter when a
# preposition grounds it; on its own it is as likely to be a stray word.
SHORT_VALUE = 3
GROUNDING_WORDS = ("for", "in", "of", "at", "from", "to", "on", "within", "is", "equals")
YEAR_PATTERN = r"(1[89]\d{2}|2[0-2]\d{2})"
MONTH_PATTERN = "|".join(re.escape(name) for name in MONTH_NAMES)
ORDINAL_QUARTERS = {"first": 1, "second": 2, "third": 3, "fourth": 4}

# Words the grammar itself supplies. Anything else left unclaimed in a
# question is a part of it the plan cannot carry.
GRAMMAR_WORDS = frozenset(
    set(QUERY_STOPWORDS)
    | set(AGGREGATION_WORDS)
    | set(ASCENDING_WORDS)
    | set(SUPERLATIVE_WORDS)
    | set(DECLINE_WORDS)
    | set(GROWTH_WORDS)
    | set(GRAIN_WORDS)
    | set(MONTH_NAMES)
    | {word for phrase in TREND_WORDS for word in phrase.split()}
    | {
        "amount", "any", "as", "been", "breakdown", "count", "did", "different", "distinct",
        "down", "fastest", "few", "figure", "figures", "find", "get", "give", "group", "grouped",
        "groups", "had", "has", "list", "most", "much", "number", "only", "order", "ordered", "over",
        "per", "rank", "ranked", "ranking", "record", "records", "row", "rows", "sells", "so",
        "sold", "sort", "sorted", "split", "those", "these", "them", "their", "its", "time",
        "top", "trajectory", "trend", "unique", "up", "value", "values", "was", "were",
    }
)


@dataclass(frozen=True)
class TimeScope:
    year: int | None = None
    month: int | None = None
    day: int | None = None
    quarter: int | None = None


@dataclass(frozen=True)
class Mentions:
    """Everything in the question that names a part of the dataset."""

    measure: str | None
    dimension: str | None
    countable: str | None
    # An entity named without "how many": the customers in "top 5 customers
    # by revenue", ranked one per identifier.
    entity: str | None
    filters: tuple[ValueFilter, ...]
    when: TimeScope
    date_mentioned: bool
    wants_count: bool


def _take_time_scope(spans: _Spans) -> TimeScope | None:
    """Claim the dates in the question, or None when they cannot be one scope."""
    year = month = day = quarter = None
    iso = spans.take(rf"\b{YEAR_PATTERN} (\d{{1,2}}) (\d{{1,2}})\b")
    if iso:
        year, month, day = int(iso.group(1)), int(iso.group(2)), int(iso.group(3))
    named = spans.take(rf"\b({MONTH_PATTERN}) (\d{{1,2}})\b") or spans.take(
        rf"\b(\d{{1,2}}) (?:of )?({MONTH_PATTERN})\b"
    )
    if named:
        if month is not None:
            return None
        first, second = named.groups()
        name, number = (first, second) if first.isalpha() else (second, first)
        month, day = MONTH_NAMES[name], int(number)
    months = spans.take_all(rf"\b({MONTH_PATTERN})\b")
    if months:
        if month is not None or len(months) > 1:
            return None
        month = MONTH_NAMES[months[0].group(1)]
    for pattern in (r"\bq([1-4])\b", r"\bquarter ([1-4])\b", r"\b([1-4]) quarter\b"):
        found = spans.take(pattern)
        if found:
            if quarter is not None:
                return None
            quarter = int(found.group(1))
    ordinal = spans.take(r"\b(first|second|third|fourth) quarter\b")
    if ordinal:
        if quarter is not None:
            return None
        quarter = ORDINAL_QUARTERS[ordinal.group(1)]
    years = spans.take_all(rf"\b{YEAR_PATTERN}\b")
    if years:
        if year is not None or len(years) > 1:
            return None
        year = int(years[0].group(1))
    if month is not None and not 1 <= month <= 12:
        return None
    if day is not None and (month is None or not 1 <= day <= 31):
        return None
    if quarter is not None and month is not None:
        return None
    return TimeScope(year=year, month=month, day=day, quarter=quarter)


def _take_countable(spans: _Spans, roles: ColumnRoles, dataframe: pd.DataFrame) -> str | None:
    """Claim 'customers' in "how many customers" for a Customer ID column.

    A Customer ID that repeats is not unique enough to be the identifier
    role, but "how many customers" still means distinct customers.
    """
    keyed = [column for column in dataframe.columns if is_identifier_name(str(column))]
    candidates = [column for column in (roles.identifier, *keyed, *roles.dimensions) if column]
    for column in candidates:
        normalized = _norm(column)
        first_word = normalized.split(" ")[0]
        for token in (normalized, first_word):
            if not token or token in IDENTIFIER_NAME_WORDS:
                continue
            for variant in sorted(_word_variants(token), key=len, reverse=True):
                if spans.take(rf"\b{re.escape(variant)}\b"):
                    return column
    return None


def _take_mentions(
    spans: _Spans, dataframe: pd.DataFrame, roles: ColumnRoles
) -> tuple[list[tuple[str, int, int]], tuple[ValueFilter, ...] | None]:
    """Claim every column name and dimension value, longest span first.

    "New York" is claimed before "York" can be, and "West Region" before
    "Region", so a shorter name never steals part of a longer one. Returns
    the columns with where they were found, and the value filters -- or
    None for the filters when one value belongs to two columns and the
    question does not say which.
    """
    candidates: list[tuple[str, str, str, str | None]] = []
    for column in dataframe.columns:
        name = _norm(column)
        if name:
            for variant in _word_variants(name):
                candidates.append((variant, "column", str(column), None))
    for column in roles.dimensions:
        if column not in dataframe.columns:
            continue
        uniques = dataframe[column].dropna().unique()
        if len(uniques) > MAX_FILTER_CANDIDATES:
            continue
        for value in uniques:
            normalized = _norm(value)
            if normalized and normalized not in GRAMMAR_WORDS:
                candidates.append((normalized, "value", column, str(value)))
    # Columns named after a grammar word ("Total", "Count", "Year") are matched
    # last, so they only claim the word when nothing else in the question does.
    candidates.sort(key=lambda item: (item[1] == "column" and item[0] in GRAMMAR_WORDS, -len(item[0])))

    columns: list[tuple[str, int, int]] = []
    seen: set[str] = set()
    values: dict[str, list[str]] = {}
    claimed_values: dict[str, str] = {}
    for normalized, kind, column, value in candidates:
        pattern = rf"\b{re.escape(normalized)}\b"
        if kind == "value" and len(normalized) < SHORT_VALUE:
            pattern = rf"\b(?:{'|'.join(GROUNDING_WORDS)}) {re.escape(normalized)}\b"
        if kind == "value" and normalized in claimed_values and claimed_values[normalized] != column:
            # The same label lives in two columns; whichever this question
            # means, filtering the other is wrong, and both is wrong too.
            if re.search(pattern, spans.text):
                return columns, None
            continue
        if kind == "column" and normalized in GRAMMAR_WORDS:
            # "how many rows by Rows": the column called Rows is the one
            # after "by"; the other "rows" is the grammar's. A grammar-named
            # column claims one occurrence, the grounded one first.
            lead_in = "|".join((*GROUNDING_WORDS, "by", "per", "across", "each"))
            grounded = rf"\b(?:{lead_in}) ({re.escape(normalized)})\b"
            found = spans.take(grounded, group=1)
            spans_found = [found.span(1)] if found else []
            if not found and (found := spans.take(pattern)):
                spans_found = [found.span()]
        else:
            spans_found = [match.span() for match in spans.take_all(pattern)]
        if not spans_found:
            continue
        if kind == "column":
            if column not in seen:
                seen.add(column)
                columns.append((column, *spans_found[0]))
        else:
            claimed_values[normalized] = column
            values.setdefault(column, []).append(value)  # type: ignore[arg-type]
    filters = tuple(ValueFilter(column=column, values=tuple(found)) for column, found in values.items())
    return columns, filters


def _resolve_mentions(
    spans: _Spans, dataframe: pd.DataFrame, roles: ColumnRoles
) -> Mentions | None:
    """Decide what the question's columns, values and dates refer to.

    None means the question names more than the plan can hold -- two
    metrics, two grouping columns, two months -- and must not be answered
    as though it named one.
    """
    columns, filters = _take_mentions(spans, dataframe, roles)
    if filters is None:
        return None
    when = _take_time_scope(spans)
    if when is None:
        return None

    numeric_columns = set(roles.numeric)
    dimension_columns = set(roles.dimensions)
    measures = [column for column, _, _ in columns if column in numeric_columns]
    if len(measures) > 1:
        # "total Revenue" in a file with a Total column: the grammar word
        # wins, and gets its span back so the grammar can read it.
        for column, start, end in columns:
            if column in measures and _norm(column) in GRAMMAR_WORDS:
                spans.release(start, end)
        measures = [column for column in measures if _norm(column) not in GRAMMAR_WORDS]
    if len(measures) > 1:
        return None
    measure = measures[0] if measures else None
    date_mentioned = any(column == roles.date for column, _, _ in columns)

    grammar = spans.blanked()
    breakdown = re.search(BREAKDOWN_PATTERN, grammar)
    grouped = [
        column for column, start, _ in columns
        if column in dimension_columns and breakdown is not None and start > breakdown.start()
    ]
    leading = [
        column for column, start, _ in columns
        if column in dimension_columns and (breakdown is None or start < breakdown.start())
    ]
    wants_count = bool(re.search(COUNT_PATTERN, grammar))
    if not wants_count and leading and grouped and measure is None:
        # "customers by region" with no metric is a count of one by the other.
        wants_count = True
    countable = entity = None
    if wants_count:
        if leading:
            countable, leading = leading[0], leading[1:]
        else:
            countable = _take_countable(spans, roles, dataframe)
    else:
        entity = _take_countable(spans, roles, dataframe)
        if entity is not None and measure is None and (breakdown is not None or not (grouped or leading)):
            # "customers by region" is a count of one by the other.
            wants_count, countable, entity = True, entity, None
    dimensions = grouped + leading
    if len(dimensions) > 1:
        return None
    return Mentions(
        measure=measure,
        dimension=dimensions[0] if dimensions else None,
        countable=countable,
        entity=entity if entity not in dimensions else None,
        filters=filters,
        when=when,
        date_mentioned=date_mentioned,
        wants_count=wants_count,
    )


def _detect_grain(question: str) -> str | None:
    for word, grain in GRAIN_WORDS.items():
        if re.search(rf"\b{word}(ly)?\b", question):
            return grain
    return None


# Phrasings the grammar cannot represent. Answering them with the nearest
# supported question -- "revenue > 200" as the total of everything, "January
# and February" as January, "Revenue / Profit" as Revenue -- is a precise
# answer to a question nobody asked. Refusing hands them to the optional
# planner, or to the suggestion chips. Operators are checked on the raw text,
# because _norm strips them.
UNSUPPORTED_OPERATORS = re.compile(r"[<>=≥≤≠+*×÷]|\s-\s|(?<=[A-Za-z)])\s*/\s*(?=[A-Za-z(])")
UNSUPPORTED_PHRASES = re.compile(
    r"\b(?:"
    r"greater than|less than|more than \d+|fewer than|at least \d+|at most \d+|"
    r"above \d+|below \d+|over \d+|under \d+|between \d+|exceeds?|exceeding|"
    r"this (?:year|month|quarter|week)|last (?:year|month|quarter|week|\d+ (?:days|weeks|months|years))|"
    r"next (?:year|month|quarter|week)|previous (?:year|month|quarter|week)|"
    r"year to date|ytd|today|yesterday|past \d+|"
    r"minus|plus|divided by|multiplied by|times|ratio of|difference between|"
    r"versus|vs|compared (?:to|with)|comparison|"
    r"or"
    r")\b"
)


def _asks_for_several_periods(q: str) -> bool:
    """Two months or two years in one question is a set the plan cannot hold."""
    months = {MONTH_NAMES[name] for name in MONTH_NAMES if re.search(rf"\b{name}\b", q)}
    years = set(re.findall(rf"\b{YEAR_PATTERN}\b", q))
    return len(months) >= 2 or len(years) >= 2


def unsupported_phrasing(question: str) -> bool:
    q = _norm(question)
    return bool(
        UNSUPPORTED_OPERATORS.search(str(question))
        or UNSUPPORTED_PHRASES.search(q)
        or _asks_for_several_periods(q)
    )


def question_is_representable(question: str, dataframe: pd.DataFrame, roles: ColumnRoles) -> bool:
    """Whether any QueryPlan could hold this question at all.

    The optional planner is asked only questions the plan shape can carry.
    Otherwise it picks one of two metrics, or one of two months, and the
    approval sentence describes a different question from the one asked.
    """
    q = _norm(question)
    if not q or unsupported_phrasing(question):
        return False
    return _resolve_mentions(_Spans(q), dataframe, roles) is not None


def plan_accounts_for_numbers(question: str, plan: QueryPlan) -> bool:
    """Every number in the question has to show up somewhere in the plan.

    A plan that reads "Revenue 2024-01-31" as January 2024 has dropped the
    31, and with it the question.
    """
    q = _norm(question)
    numbers = {int(token) for token in re.findall(r"\b\d+\b", q)}
    quarters = {int(token) for token in re.findall(r"\bq([1-4])\b", q)}
    known = {plan.year, plan.month, plan.day, plan.quarter, plan.top_n}
    for value_filter in plan.filters:
        for value in value_filter.values:
            known.update(int(token) for token in re.findall(r"\b\d+\b", _norm(value)))
    return numbers <= {number for number in known if number is not None} and (
        not quarters or plan.quarter in quarters
    )


def parse_question(question: str, dataframe: pd.DataFrame, roles: ColumnRoles) -> QueryPlan | None:
    """Turn a plain-English question into an explicit plan, or None if unsupported."""
    q = _norm(question)
    if not q or unsupported_phrasing(question):
        return None

    spans = _Spans(q)
    mentions = _resolve_mentions(spans, dataframe, roles)
    if mentions is None:
        return None
    top_match = spans.take(TOP_PATTERN)
    if top_match and int(top_match.group(2)) < 1:
        # "top 0" is not a ranking; it was showing the fallback twelve.
        return None
    # Whatever is left must be grammar. A number nobody claimed, or a word
    # that is neither a column, a value nor part of the grammar, is a part
    # of the question the plan would have ignored.
    for token in spans.leftover().split():
        if not token.isalpha() or token not in GRAMMAR_WORDS:
            return None

    measure, dimension, when = mentions.measure, mentions.dimension, mentions.when
    if dimension is None and top_match and mentions.entity:
        # "top 5 customers by revenue" ranks one row per customer.
        dimension = mentions.entity
    # The grammar is read from what no column, value or date has claimed:
    # "first quarter" is a period to look inside, not a grain, and a
    # dimension called "Change (pp)" is not a request for growth.
    grammar = spans.blanked()
    grain = _detect_grain(grammar)
    aggregation: Aggregation | None = next(
        (AGGREGATION_WORDS[word] for word in AGGREGATION_WORDS if re.search(rf"\b{word}\b", grammar)),
        None,
    )
    # "highest" and "largest" rank; only the literal words ask for an extreme
    # per group. Both live in AGGREGATION_WORDS as max, so they are told apart
    # here rather than by silently downgrading every min/max to a sum.
    explicit_extreme = bool(re.search(r"\b(min|minimum|max|maximum)\b", grammar))
    superlative = any(re.search(rf"\b{word}\b", grammar) for word in SUPERLATIVE_WORDS)
    wants_breakdown = bool(re.search(BREAKDOWN_PATTERN, grammar))
    wants_growth = any(re.search(rf"\b{word}\b", grammar) for word in GROWTH_WORDS)
    wants_trend = (
        grain is not None
        or any(phrase in grammar for phrase in TREND_WORDS)
        or (mentions.date_mentioned and wants_breakdown and dimension is None)
    )
    if dimension is not None and dimension == measure:
        return None

    base = {
        "measure": measure or roles.measure,
        "filters": mentions.filters,
        "year": when.year,
        "month": when.month,
        "day": when.day,
        "quarter": when.quarter,
        "grain": grain,
    }
    # Chat combines a measure the way the dashboard does: rates average,
    # amounts add. Saying "total" or "sum" out loud still gets a sum.
    default_aggregation = measure_aggregation(base["measure"])

    if (wants_growth or wants_trend) and not roles.date and not mentions.wants_count:
        # Not a total in disguise: the executor explains that there is no
        # timeline to measure this over.
        return QueryPlan(intent="growth" if wants_growth else "trend", **base)

    if mentions.wants_count and not wants_growth:
        countable = mentions.countable
        counted = {**base, "measure": None}
        if wants_trend and roles.date:
            return QueryPlan(intent="trend", aggregation="count", count_column=countable, **counted)
        if dimension and top_match:
            return QueryPlan(
                intent="rank", aggregation="count", dimension=dimension,
                top_n=int(top_match.group(2)), ascending=top_match.group(1) == "bottom",
                count_column=countable if countable != dimension else None, **counted,
            )
        if dimension and (wants_breakdown or superlative):
            return QueryPlan(
                intent="breakdown", aggregation="count", dimension=dimension,
                count_column=countable if countable != dimension else None, **counted,
            )
        return QueryPlan(intent="count", aggregation="count", count_column=countable, **counted)

    if wants_growth and roles.date:
        if aggregation not in (None, default_aggregation):
            # Growth is measured on periods combined the way the measure
            # combines. "Average revenue growth" would be growth of sums.
            return None
        wants_ranked_growth = superlative or any(word in grammar for word in ("which", "fastest", "slowest"))
        rank_dimension = dimension or (roles.dimension if wants_ranked_growth else None)
        ascending = bool(re.search(ASCENDING_PATTERN, grammar))
        top_n = int(top_match.group(2)) if top_match else None
        return QueryPlan(
            intent="growth", aggregation=default_aggregation, dimension=rank_dimension,
            ascending=ascending, top_n=top_n, **base,
        )

    if top_match and not dimension:
        # "top 2 Revenue" ranks nothing: rows or groups, the question does
        # not say, and the all-time total is neither.
        return None

    if (top_match or superlative) and dimension:
        if top_match:
            top_n = int(top_match.group(2))
            ascending = top_match.group(1) == "bottom"
        else:
            top_n = 1
            ascending = bool(re.search(ASCENDING_PATTERN, grammar))
        return QueryPlan(
            intent="rank",
            aggregation=_grouped_aggregation(aggregation, explicit_extreme, default_aggregation),
            dimension=dimension,
            top_n=top_n,
            ascending=ascending,
            **base,
        )

    if wants_trend and roles.date:
        if aggregation not in (None, default_aggregation):
            # The trend executor combines each period the way the measure
            # combines. "Average revenue monthly" would be answered with sums,
            # so it is not answered here; "average conversion rate monthly"
            # is exactly what the executor does, so it is.
            return None
        return QueryPlan(intent="trend", aggregation=default_aggregation, **base)

    if dimension and (wants_breakdown or not measure):
        return QueryPlan(
            intent="breakdown",
            aggregation=_grouped_aggregation(aggregation, explicit_extreme, default_aggregation),
            dimension=dimension,
            **base,
        )

    if base["measure"] and (aggregation or measure):
        return QueryPlan(intent="aggregate", aggregation=aggregation or default_aggregation, **base)

    return None


def _grouped_aggregation(
    aggregation: Aggregation | None, explicit_extreme: bool, default: Aggregation
) -> Aggregation:
    """What a rank or breakdown computes per group."""
    if aggregation in ("sum", "mean", "median"):
        # Said out loud, so honoured -- including a sum of a rate, which the
        # ungrouped path already allows; the two must agree.
        return aggregation
    if aggregation in ("min", "max") and explicit_extreme:
        return aggregation
    return default


class TimeScopeUnavailable(Exception):
    """A year or month was asked for in a file that has no date column."""


# Full names by number. Deriving these by searching MONTH_NAMES backwards used
# to drop May, whose only spelling is three letters long, leaving the scope
# label empty in the answer sentence.
MONTH_LABELS = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December",
}


def _when_label(plan: QueryPlan) -> str:
    if plan.quarter:
        within = f"Q{plan.quarter}"
    elif plan.month:
        within = MONTH_LABELS[plan.month] + (f" {plan.day}" if plan.day else "")
    else:
        within = None
    year = str(plan.year) if plan.year else None
    return " ".join(part for part in (within, year) if part)


def _has_time_scope(plan: QueryPlan) -> bool:
    return any(part is not None for part in (plan.year, plan.month, plan.day, plan.quarter))


def _apply_filters(
    dataframe: pd.DataFrame, plan: QueryPlan, roles: ColumnRoles
) -> tuple[pd.DataFrame, list[str]]:
    working = dataframe
    applied: list[str] = []
    for value_filter in plan.filters:
        if value_filter.column not in working.columns:
            raise ValueError(f"Unknown filter column: {value_filter.column}")
        mask = working[value_filter.column].astype(str).isin(value_filter.values)
        working = working.loc[mask]
        applied.append(f"{value_filter.column} in ({', '.join(value_filter.values)})")
    if _has_time_scope(plan):
        if not roles.date or roles.date not in working.columns:
            # Returning the all-time figure under a sentence that names a year
            # is the worst available answer, so the scope is reported as
            # impossible instead of dropped.
            raise TimeScopeUnavailable(_when_label(plan))
        dates = working[roles.date]
        if plan.year:
            working = working.loc[dates.dt.year == plan.year]
            dates = working[roles.date]
        if plan.quarter:
            working = working.loc[dates.dt.quarter == plan.quarter]
            dates = working[roles.date]
        if plan.month:
            working = working.loc[dates.dt.month == plan.month]
            dates = working[roles.date]
        if plan.day:
            working = working.loc[dates.dt.day == plan.day]
        applied.append(f"{roles.date} in {_when_label(plan)}")
    return working, applied


def _scoped(applied_filters: list[str]) -> str:
    return f" · filtered to {', '.join(applied_filters)}" if applied_filters else ""


def _aggregate_series(series: pd.Series, aggregation: Aggregation) -> float:
    return float(getattr(series.dropna(), aggregation)())


def distinct_label(wanted: str, taken) -> str:
    """A display column name that is not already one of the user's columns."""
    present = {str(name) for name in taken}
    label, suffix = wanted, 2
    while label in present:
        label = f"{wanted} ({suffix})"
        suffix += 1
    return label


def _grouped_frame(
    dataframe: pd.DataFrame, plan: QueryPlan, value_label: str
) -> tuple[pd.DataFrame, str, str | None]:
    """Group, and return the frame with the display names it actually used.

    Everything is computed on private column names, so a dimension the file
    calls "Share %" or "Rows" cannot be overwritten by the columns built here.
    The display names are then chosen not to collide with the dimension's own.
    """
    assert plan.dimension is not None
    dimension, measure = plan.dimension, plan.measure
    columns = [dimension]
    if measure and measure != dimension:
        columns.append(measure)
    if plan.count_column and plan.count_column not in columns:
        columns.append(plan.count_column)
    working = dataframe[columns].copy()
    private = {name: f"__{index}" for index, name in enumerate(columns)}
    working.columns = [private[name] for name in columns]
    segment = private[dimension]
    # Rows with no label are a group, not a rounding error. Dropping them and
    # then taking percentages against what is left reports a share of a total
    # the reader never saw.
    working[segment] = working[segment].astype(object).where(
        working[segment].notna(), unlabelled_label(working[segment].dropna().unique())
    )
    if plan.aggregation == "count" and plan.count_column:
        # "How many customers by region" counts customers, not rows.
        grouped = working.groupby(segment, as_index=False)[private[plan.count_column]].nunique()
    elif plan.aggregation == "count" or not measure:
        grouped = working.groupby(segment, as_index=False).size()
    else:
        grouped = working.groupby(segment, as_index=False)[private[measure]].agg(plan.aggregation)
    grouped.columns = [segment, "__value"]
    grouped = grouped.sort_values("__value", ascending=plan.ascending)
    share_name = None
    if plan.aggregation in ("sum", "count"):
        values = grouped["__value"].to_numpy(dtype=float)
        total = float(values.sum())
        # A share is only a share when every part carries the same sign as the
        # whole. Mixed signs give 500% and -250% from arithmetic that is
        # working exactly as written.
        if total and ((values >= 0).all() or (values <= 0).all()):
            grouped["__share"] = (grouped["__value"] / total * 100).round(1)
    value_name = distinct_label(value_label, {dimension})
    names = {segment: dimension, "__value": value_name}
    if "__share" in grouped.columns:
        share_name = distinct_label("Share %", {dimension, value_name})
        names["__share"] = share_name
    return grouped.rename(columns=names).reset_index(drop=True), value_name, share_name


def _shown(value: float, measure: str | None, frame: pd.DataFrame) -> str:
    """Format a chat figure against the whole column it came from.

    A rate column of [0.5, 2.0] is in percentage points; formatting the scalar
    0.5 on its own reads it as a fraction and prints 50.0% where the brief
    beside it prints 0.5%. The column settles the scale, once.
    """
    values = frame[measure].dropna() if measure and measure in frame.columns else None
    return format_number(value, measure, column_values=values)


def execute_plan(plan: QueryPlan, dataframe: pd.DataFrame, roles: ColumnRoles) -> QueryAnswer:
    """Run a validated plan locally and package the auditable answer."""
    for column in (plan.measure, plan.dimension, plan.count_column):
        if column is not None and column not in dataframe.columns:
            raise ValueError(f"Unknown column in plan: {column}")

    if plan.intent in ("trend", "growth") and not roles.date:
        subject = plan.measure or "the row count"
        return QueryAnswer(
            question="",
            plan=plan,
            answer=(
                f"This file has no date column, so there is no timeline to measure {subject} "
                "over. If one of the columns holds dates, pick it as the date in the sidebar "
                "and ask again."
            ),
            calculation="no date column; a trend or growth rate needs one",
        )

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
    if working.empty:
        return QueryAnswer(
            question="",
            plan=plan,
            answer="No rows match that scope, so there is nothing to calculate.",
            calculation=f"0 rows after filters{scope}",
        )

    if plan.intent == "count":
        if plan.count_column:
            distinct = int(working[plan.count_column].nunique(dropna=True))
            return QueryAnswer(
                question="",
                plan=plan,
                answer=f"There are {distinct:,} distinct {plan.count_column} values{_phrase(applied)}.",
                calculation=f"count distinct {plan.count_column}{scope}",
            )
        return QueryAnswer(
            question="",
            plan=plan,
            answer=f"{len(working):,} rows match{_phrase(applied)}.",
            calculation=f"row count{scope}",
        )

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
                f"{label} {plan.measure}{_phrase(applied)} is "
                f"{_shown(value, plan.measure, dataframe)}, calculated from {rows:,} rows."
            ),
            calculation=f"{plan.aggregation}({plan.measure}){scope}",
        )

    if plan.intent in ("rank", "breakdown"):
        assert plan.dimension is not None
        label = AGGREGATION_LABELS[plan.aggregation]
        counted = plan.aggregation == "count" or not plan.measure
        if counted and plan.count_column:
            value_label = f"Distinct {plan.count_column}"
            counted_as = f"count distinct {plan.count_column}"
        elif counted:
            value_label = "Rows"
            counted_as = "row count"
        else:
            value_label = f"{label} {plan.measure}"
            counted_as = f"{plan.aggregation}({plan.measure})"
        grouped, value_name, share_name = _grouped_frame(working, plan, value_label)
        limit = plan.top_n if plan.intent == "rank" else BREAKDOWN_LIMIT
        table = grouped.head(limit or BREAKDOWN_LIMIT)
        leader = table.iloc[0]
        leader_value = float(leader[value_name])
        direction = "lowest" if plan.ascending else "leading"
        share_note = f" ({leader[share_name]:.1f}% of the total)" if share_name else ""
        # A row count is not money, whatever the measure column is called.
        counted = value_label == "Rows"
        answer = (
            f"{leader[plan.dimension]} is the {direction} {plan.dimension} by {value_label.lower()}"
            f"{_phrase(applied)} at "
            f"{_shown(leader_value, None if counted else plan.measure, dataframe)}{share_note}."
        )
        order = "ascending" if plan.ascending else "descending"
        return QueryAnswer(
            question="",
            plan=plan,
            answer=answer,
            calculation=(
                # value_label already records whether rows or the measure were
                # aggregated; naming count(<measure>) when .size() ran flipped
                # which group won.
                f"{counted_as} by {plan.dimension}, {order}, showing {len(table)}{scope}"
            ),
            table=table,
            chart="bar",
        )

    if plan.intent == "trend":
        assert roles.date is not None
        counted = plan.aggregation == "count"
        scoped_roles = ColumnRoles(
            date=roles.date,
            measure=None if counted else plan.measure,
            dimension=None,
            identifier=roles.identifier,
            numeric=roles.numeric,
            dimensions=roles.dimensions,
        )
        grain = plan.grain or preferred_frequency(working[roles.date])
        series = build_trend(
            working, scoped_roles, frequency=grain, count_column=plan.count_column if counted else None
        )
        trend = series.frame
        if len(trend) < 2:
            return QueryAnswer(
                question="",
                plan=plan,
                answer="Not enough periods in that scope to draw a trend.",
                calculation=f"trend needs at least 2 periods{scope}",
            )
        first, last = float(trend.iloc[0]["Value"]), float(trend.iloc[-1]["Value"])
        grain_name = {"D": "day", "W": "week", "M": "month", "Q": "quarter", "Y": "year"}.get(grain, "period")
        # Growing from nothing has no percentage. Printing +0.0% beside two
        # different numbers contradicts the rest of the sentence.
        movement = (
            f"{(last - first) / abs(first) * 100:+.1f}% across {len(trend)} {grain_name}s"
            if first
            else f"across {len(trend)} {grain_name}s, from a starting period of zero"
        )
        if counted and plan.count_column:
            subject, shown_measure = f"Distinct {plan.count_column}", None
            counted_as = f"count distinct {plan.count_column}"
        elif counted or not plan.measure:
            subject, shown_measure, counted_as = "Records", None, "row count"
        else:
            subject, shown_measure = plan.measure, plan.measure
            counted_as = f"{plan.aggregation}({plan.measure})"
        answer = (
            f"{subject} per {grain_name}{_phrase(applied)} moved from "
            f"{_shown(first, shown_measure, dataframe)} to {_shown(last, shown_measure, dataframe)} "
            f"({movement})."
        )
        return QueryAnswer(
            question="",
            plan=plan,
            answer=answer,
            calculation=_with_notes(f"{counted_as} grouped per {grain_name}{scope}", series),
            table=trend,
            chart="line",
        )

    if plan.intent == "growth":
        return _execute_growth(plan, working, roles, scope, applied)

    raise ValueError(f"Unsupported intent: {plan.intent}")


def _execute_growth(
    plan: QueryPlan,
    working: pd.DataFrame,
    roles: ColumnRoles,
    scope: str,
    applied: list[str],
) -> QueryAnswer:
    assert roles.date is not None
    measure = plan.measure
    scoped_roles = ColumnRoles(
        date=roles.date,
        measure=measure,
        dimension=plan.dimension,
        identifier=roles.identifier,
        numeric=roles.numeric,
        dimensions=roles.dimensions,
    )
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

    metric = resolve_metric(measure)
    is_rate = bool(measure) and metric.unit == "rate"
    # A rate moves in percentage points: 10% to 20% is ten points, and the
    # "100% growth" the relative formula gives is what nobody means by it.
    scale = 100.0 if is_rate and rate_scale(working[measure].dropna()) == "fraction" else 1.0

    if plan.dimension:
        grain = plan.grain or preferred_frequency(working[roles.date])
        picked = [roles.date, plan.dimension] + ([measure] if measure and measure != plan.dimension else [])
        frame = working[picked].dropna(subset=[roles.date, plan.dimension]).copy()
        # Private names from here on: a measure or dimension the file calls
        # "__period" was being overwritten by the buckets built next.
        frame.columns = ["__date", "__segment"] + (["__measure"] if len(picked) == 3 else [])
        frame = frame.assign(__period=frame["__date"].dt.to_period(grain).dt.to_timestamp())
        frame = frame[frame["__period"].isin([previous_period, current_period])]
        if measure:
            pivot = frame.groupby(["__segment", "__period"])["__measure"].agg(metric.aggregation).unstack()
        else:
            pivot = frame.groupby(["__segment", "__period"]).size().unstack()
        if current_period not in pivot.columns:
            return QueryAnswer(
                question="",
                plan=plan,
                answer="The latest period has no rows with a segment label, so growth cannot be ranked.",
                calculation=f"per-{plan.dimension} growth unavailable{scope}",
            )
        # A previous period with no rows at all is still a period: the
        # timeline showed it, so the ranking is measured against it too.
        pivot = pivot.reindex(columns=[previous_period, current_period])
        if metric.additive:
            # No rows in a period is a period of nothing sold: zero.
            pivot = pivot.fillna(0.0)
        else:
            # No rows in a period is no average for that period. A segment
            # has to be present on both sides to have moved at all.
            pivot = pivot.dropna(subset=[previous_period, current_period])
        # The result's own column names are chosen against the dimension's:
        # a dimension the file calls "Latest" was being overwritten by the
        # latest-period column, and "100.0 moved fastest" replaced Alpha.
        taken = {plan.dimension}
        previous_name = distinct_label("Previous", taken)
        taken.add(previous_name)
        latest_name = distinct_label("Latest", taken)
        taken.add(latest_name)
        change_name = distinct_label("Change (pp)" if is_rate else "Change %", taken)
        result = pd.DataFrame(
            {
                plan.dimension: pivot.index,
                previous_name: pivot[previous_period].to_numpy(dtype=float),
                latest_name: pivot[current_period].to_numpy(dtype=float),
            }
        )
        if is_rate:
            from_nothing = result.iloc[0:0]
        else:
            # A percentage from zero is undefined, but a segment that went from
            # nothing to something is usually the most newsworthy row in the file.
            # It leaves the ranking and keeps its sentence.
            from_nothing = result[(result[previous_name] == 0) & (result[latest_name] != 0)]
            result = result[result[previous_name] != 0]
        if result.empty and not from_nothing.empty:
            newest = from_nothing.loc[from_nothing[latest_name].abs().idxmax()]
            return QueryAnswer(
                question="",
                plan=plan,
                answer=(
                    f"No {plan.dimension} has a growth rate this period, because every one "
                    f"of them started from zero. The largest new arrival is "
                    f"{newest[plan.dimension]} at "
                    f"{_shown(float(newest[latest_name]), measure, working)}."
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
                answer=(
                    "No segment is present in both of the latest two periods, so there is "
                    "nothing to compare."
                    if is_rate
                    else "Every segment starts from zero in the prior period, so growth rates are undefined."
                ),
                calculation=f"per-{plan.dimension} growth undefined{scope}",
            )
        if is_rate:
            change = (result[latest_name] - result[previous_name]) * scale
        else:
            change = (result[latest_name] - result[previous_name]) / result[previous_name].abs() * 100
        result[change_name] = change.round(1)
        result = result.sort_values(change_name, ascending=plan.ascending).reset_index(drop=True)
        if plan.top_n:
            result = result.head(plan.top_n)
        leader = result.iloc[0]
        direction = "slowest" if plan.ascending else "fastest"
        moved = (
            f"{leader[change_name]:+.1f} percentage points" if is_rate else f"{leader[change_name]:+.1f}%"
        )
        answer = (
            f"{leader[plan.dimension]} moved {direction}{_phrase(applied)}: {moved} "
            f"({_shown(float(leader[previous_name]), measure, working)} → "
            f"{_shown(float(leader[latest_name]), measure, working)}) in the latest period."
        )
        if not from_nothing.empty:
            newest = from_nothing.loc[from_nothing[latest_name].abs().idxmax()]
            names = ", ".join(str(name) for name in from_nothing[plan.dimension])
            answer += (
                f" {len(from_nothing)} segment(s) are left out of the ranking because they "
                f"started from zero ({names}); the largest is {newest[plan.dimension]} at "
                f"{_shown(float(newest[latest_name]), measure, working)}."
            )
        combined = metric.combines_as if measure else "count"
        ranked_by = "change in percentage points" if is_rate else "% change"
        return QueryAnswer(
            question="",
            plan=plan,
            answer=answer,
            calculation=(
                f"per-{plan.dimension} {measure or 'row count'} ({combined} per period): latest vs "
                f"previous period, ranked by {ranked_by}{scope}"
            ),
            table=result.head(BREAKDOWN_LIMIT),
            chart="bar",
        )

    previous, current = float(trend.iloc[-2]["Value"]), float(trend.iloc[-1]["Value"])
    previous_period = trend.iloc[-2]["Period"]
    if not (np.isfinite(previous) and np.isfinite(current)):
        return QueryAnswer(
            question="",
            plan=plan,
            answer="One of the latest two periods has no measured value, so a change cannot be calculated.",
            calculation=f"growth undefined: a period is missing its value{scope}",
        )
    previous_label = format_period(previous_period, series.frequency)
    if previous == 0 and not is_rate and (
        working[roles.date].dt.to_period(series.frequency).dt.to_timestamp() == previous_period
    ).sum() == 0:
        # The zero is a period with no rows, which the timeline filled in
        # as zero for an additive metric. Growth from it is undefined, and
        # the sentence should say why rather than call it a measured zero.
        return QueryAnswer(
            question="",
            plan=plan,
            answer=(
                f"{previous_label} has no rows{_phrase(applied)}, so there is no previous value to "
                f"measure the latest period against."
            ),
            calculation=f"growth undefined: {previous_label} is an empty period{scope}",
        )
    if is_rate:
        change = (current - previous) * scale
        direction = "up" if change >= 0 else "down"
        answer = (
            f"{measure}{_phrase(applied)} is {direction} {abs(change):.1f} percentage points versus "
            f"the prior period ({_shown(previous, measure, working)} → {_shown(current, measure, working)})."
        )
        calculation = "latest period average − previous period average, in percentage points"
    else:
        if previous == 0:
            return QueryAnswer(
                question="",
                plan=plan,
                answer="The previous period is zero, so a growth rate is undefined.",
                calculation=f"growth undefined for zero base{scope}",
            )
        change = (current - previous) / abs(previous) * 100
        direction = "up" if change >= 0 else "down"
        answer = (
            f"{measure or 'Records'}{_phrase(applied)} is {direction} {abs(change):.1f}% versus the prior "
            f"period ({_shown(previous, measure, working)} → {_shown(current, measure, working)})."
        )
        combined = "period counts" if not measure else f"period {metric.combines_as}s"
        calculation = f"(latest − previous) ÷ |previous| on {combined}"
    return QueryAnswer(
        question="",
        plan=plan,
        answer=answer,
        calculation=_with_notes(f"{calculation}{scope}", series),
        table=trend,
        chart="line",
    )


def _with_notes(calculation: str, series: TrendSeries) -> str:
    """Carry the timeline adjustments into the answer's calculation trace.

    A chat answer that quietly drops an in-progress period owes the reader
    the same disclosure the dashboard gives.
    """
    return "; ".join([calculation, *series.notes]) if series.notes else calculation


def _phrase(applied_filters: list[str]) -> str:
    return f" for {'; '.join(applied_filters)}" if applied_filters else ""


def answer_question(question: str, dataframe: pd.DataFrame, roles: ColumnRoles) -> QueryAnswer | None:
    """Parse and execute in one step; None means the rules could not read it."""
    plan = parse_question(question, dataframe, roles)
    if plan is None:
        return None
    try:
        result = execute_plan(plan, dataframe, roles)
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


def suggested_questions(dataframe: pd.DataFrame, roles: ColumnRoles) -> list[str]:
    """Offer starter questions that the deterministic engine can definitely answer."""
    suggestions: list[str] = []
    if roles.measure:
        suggestions.append(f"Total {roles.measure}")
    if roles.measure and roles.dimension:
        suggestions.append(f"Top 5 {roles.dimension} by {roles.measure}")
    if roles.measure and roles.date:
        suggestions.append(f"Monthly {roles.measure} trend")
    if roles.dimension and roles.date:
        suggestions.append(f"Which {roles.dimension} grew fastest?")
    if not suggestions:
        suggestions.append("How many rows are there?")
    return suggestions[:4]
