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

import pandas as pd

from aggregation import TrendSeries, build_trend, measure_aggregation, preferred_frequency, unlabelled_label
from formatting import format_number
from schema import ColumnRoles

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
KEY_WORDS = frozenset({
    "id", "ids", "code", "codes", "zip", "postal", "phone", "sku", "number", "no", "key", "uuid",
})

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
    cleaned = re.sub(r"[^0-9a-z]+", " ", lowered)
    return " ".join(cleaned.split())


def _word_variants(normalized_name: str) -> set[str]:
    variants = {normalized_name, f"{normalized_name}s", f"{normalized_name}es"}
    if normalized_name.endswith("s"):
        variants.add(normalized_name[:-1])
    return variants


def _mentioned(normalized_name: str, question: str) -> bool:
    return any(
        re.search(rf"\b{re.escape(variant)}\b", question)
        for variant in _word_variants(normalized_name)
        if variant
    )


def _match_column(question: str, columns: list[str]) -> str | None:
    """Return the column with the longest name mentioned in the question."""
    matches = [column for column in columns if _mentioned(_norm(column), question)]
    return max(matches, key=lambda column: len(_norm(column))) if matches else None


def _named_like_a_key(name: str) -> bool:
    words = _norm(name).split()
    return bool(words) and (words[-1] in KEY_WORDS or words == ["id"])


def _match_countable(question: str, roles: ColumnRoles, dataframe: pd.DataFrame | None = None) -> str | None:
    """Match 'how many <entities>' to an identifier, key-named or dimension column.

    A Customer ID that repeats is not unique enough to be the identifier
    role, but "how many customers" still means distinct customers.
    """
    keyed = [
        column for column in (dataframe.columns if dataframe is not None else ())
        if _named_like_a_key(str(column))
    ]
    candidates = [column for column in (roles.identifier, *keyed, *roles.dimensions) if column]
    for column in candidates:
        tokens = {_norm(column), _norm(column).split(" ")[0]}
        if any(_mentioned(token, question) for token in tokens if token):
            return column
    return None


def _detect_value_filters(
    question: str,
    dataframe: pd.DataFrame,
    roles: ColumnRoles,
    *,
    exclude: tuple[str | None, ...] = (),
) -> tuple[ValueFilter, ...]:
    filters: list[ValueFilter] = []
    for column in roles.dimensions:
        if column in exclude or column not in dataframe.columns:
            continue
        uniques = dataframe[column].dropna().unique()
        if len(uniques) > MAX_FILTER_CANDIDATES:
            continue
        matched = [
            str(value)
            for value in uniques
            if len(_norm(value)) >= 3 and _mentioned(_norm(value), question)
        ]
        if matched:
            filters.append(ValueFilter(column=column, values=tuple(matched)))
    return tuple(filters)


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


def _unrecognized_query_tokens(
    question: str, dataframe: pd.DataFrame, roles: ColumnRoles
) -> set[str]:
    """Return content words that cannot refer to this dataset or query grammar."""
    allowed = set(QUERY_STOPWORDS)
    allowed.update(AGGREGATION_WORDS)
    allowed.update(ASCENDING_WORDS)
    allowed.update(SUPERLATIVE_WORDS)
    allowed.update(DECLINE_WORDS)
    allowed.update(GROWTH_WORDS)
    allowed.update(GRAIN_WORDS)
    allowed.update(MONTH_NAMES)
    allowed.update(
        {
            "bottom",
            "count",
            "breakdown",
            "fastest",
            "history",
            "number",
            "per",
            "rows",
            "sells",
            "split",
            "sold",
            "top",
            "trajectory",
            "trend",
            "over",
            "time",
            "timeline",
        }
    )

    for column in dataframe.columns:
        normalized = _norm(column)
        allowed.update(normalized.split())
        allowed.update(_word_variants(normalized))
        first_word = normalized.split(" ", 1)[0]
        allowed.update(_word_variants(first_word))

    for column in roles.dimensions:
        if column not in dataframe.columns:
            continue
        unique_values = dataframe[column].dropna().astype(str).unique()
        if len(unique_values) <= MAX_FILTER_CANDIDATES:
            for value in unique_values:
                allowed.update(_norm(value).split())

    return {token for token in question.split() if token.isalpha() and token not in allowed}


# Phrasings the grammar cannot represent. Answering them with the nearest
# supported question -- "revenue > 200" as the total of everything, "January
# and February" as January -- is a precise answer to a question nobody asked.
# Refusing hands them to the optional planner, or to the suggestion chips.
# Comparison operators are checked on the raw text, because _norm strips them.
UNSUPPORTED_OPERATORS = re.compile(r"[<>=\u2265\u2264\u2260]")
UNSUPPORTED_PHRASES = re.compile(
    r"\b(?:"
    r"greater than|less than|more than \d+|fewer than|at least \d+|at most \d+|"
    r"above \d+|below \d+|over \d+|under \d+|between \d+|exceeds?|exceeding|"
    r"this (?:year|month|quarter|week)|last (?:year|month|quarter|week|\d+ (?:days|weeks|months|years))|"
    r"next (?:year|month|quarter|week)|previous (?:year|month|quarter|week)|"
    r"year to date|ytd|today|yesterday|past \d+|"
    r"or"
    r")\b"
)


def _asks_for_several_periods(q: str) -> bool:
    """Two months or two years in one question is a set the plan cannot hold."""
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


def parse_question(question: str, dataframe: pd.DataFrame, roles: ColumnRoles) -> QueryPlan | None:
    """Turn a plain-English question into an explicit plan, or None if unsupported."""
    q = _norm(question)
    if not q or unsupported_phrasing(question):
        return None

    numeric_columns = [column for column in roles.numeric if column in dataframe.columns]
    dimension_columns = [column for column in roles.dimensions if column in dataframe.columns]

    measure = _match_column(q, numeric_columns)
    dimension = _match_column(q, dimension_columns)
    if _unrecognized_query_tokens(q, dataframe, roles):
        return None
    year, month = _detect_time_filter(q)
    grain = _detect_grain(q)
    # For a grouped answer the matched dimension is the thing being grouped,
    # so its values are not read as filters. For a plain total they are:
    # "revenue for Region West" is a filter, and naming the column must not
    # steal the value from it -- that answered with the all-region total.
    filters = _detect_value_filters(q, dataframe, roles, exclude=(dimension,))
    filters_including_dimension = _detect_value_filters(q, dataframe, roles)

    aggregation: Aggregation | None = next(
        (AGGREGATION_WORDS[word] for word in AGGREGATION_WORDS if re.search(rf"\b{word}\b", q)),
        None,
    )
    top_match = re.search(r"\b(top|bottom)\s+(\d{1,3})\b", q)
    if top_match and int(top_match.group(2)) < 1:
        # "top 0" is not a ranking; it was showing the fallback twelve.
        return None
    # "highest" and "largest" rank; only the literal words ask for an extreme
    # per group. Both live in AGGREGATION_WORDS as max, so they are told apart
    # here rather than by silently downgrading every min/max to a sum.
    explicit_extreme = bool(re.search(r"\b(min|minimum|max|maximum)\b", q))
    superlative = any(re.search(rf"\b{word}\b", q) for word in SUPERLATIVE_WORDS)
    wants_breakdown = bool(re.search(r"\b(by|per|across|breakdown|split|each)\b", q))
    wants_count = bool(re.search(r"\b(how many|count|number of)\b", q))
    wants_growth = any(re.search(rf"\b{word}\b", q) for word in GROWTH_WORDS)
    wants_trend = grain is not None or any(phrase in q for phrase in TREND_WORDS)

    base = {
        "measure": measure or roles.measure,
        "filters": filters,
        "year": year,
        "month": month,
        "grain": grain,
    }

    ungrouped = {**base, "filters": filters_including_dimension}
    # Chat combines a measure the way the dashboard does: rates average,
    # amounts add. Saying "total" or "sum" out loud still gets a sum.
    default_aggregation = measure_aggregation(base["measure"])

    if wants_count and not wants_growth:
        countable = _match_countable(q, roles, dataframe)
        if dimension and wants_breakdown:
            return QueryPlan(
                intent="breakdown", aggregation="count", dimension=dimension,
                count_column=countable if countable != dimension else None, **base,
            )
        return QueryPlan(intent="count", aggregation="count", count_column=countable, **ungrouped)

    if wants_growth and roles.date:
        wants_ranked_growth = superlative or any(word in q for word in ("which", "fastest", "slowest"))
        rank_dimension = dimension or (roles.dimension if wants_ranked_growth else None)
        ascending = bool(re.search(ASCENDING_PATTERN, q))
        top_n = int(top_match.group(2)) if top_match else None
        return QueryPlan(
            intent="growth", dimension=rank_dimension, ascending=ascending, top_n=top_n, **base
        )

    if (top_match or superlative) and dimension:
        if top_match:
            top_n = int(top_match.group(2))
            ascending = top_match.group(1) == "bottom"
        else:
            top_n = 1
            ascending = bool(re.search(ASCENDING_PATTERN, q))
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
        return QueryPlan(intent="trend", aggregation=default_aggregation, **ungrouped)

    if dimension and (wants_breakdown or not measure):
        return QueryPlan(
            intent="breakdown",
            aggregation=_grouped_aggregation(aggregation, explicit_extreme, default_aggregation),
            dimension=dimension,
            **base,
        )

    if base["measure"] and (aggregation or measure):
        return QueryPlan(intent="aggregate", aggregation=aggregation or default_aggregation, **ungrouped)

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
        mask = working[value_filter.column].astype(str).isin(value_filter.values)
        working = working.loc[mask]
        applied.append(f"{value_filter.column} in ({', '.join(value_filter.values)})")
    if plan.year or plan.month:
        if not roles.date or roles.date not in working.columns:
            # Returning the all-time figure under a sentence that names a year
            # is the worst available answer, so the scope is reported as
            # impossible instead of dropped.
            raise TimeScopeUnavailable(_when_label(plan))
        dates = working[roles.date]
        if plan.year:
            working = working.loc[dates.dt.year == plan.year]
            dates = working[roles.date]
        if plan.month:
            working = working.loc[dates.dt.month == plan.month]
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
        elif counted:
            value_label = "Rows"
        else:
            value_label = f"{label} {plan.measure}"
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
                f"{'row count' if value_label == 'Rows' else f'{plan.aggregation}({plan.measure})'}"
                f" by {plan.dimension}, {order}, showing {len(table)}{scope}"
            ),
            table=table,
            chart="bar",
        )

    if plan.intent == "trend":
        assert roles.date is not None
        scoped_roles = ColumnRoles(
            date=roles.date,
            measure=plan.measure,
            dimension=None,
            identifier=roles.identifier,
            numeric=roles.numeric,
            dimensions=roles.dimensions,
        )
        grain = plan.grain or preferred_frequency(working[roles.date])
        series = build_trend(working, scoped_roles, frequency=grain)
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
        answer = (
            f"{plan.measure or 'Records'} per {grain_name}{_phrase(applied)} moved from "
            f"{_shown(first, plan.measure, dataframe)} to {_shown(last, plan.measure, dataframe)} "
            f"({movement})."
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
            pivot = (
                frame.groupby(["__segment", "__period"])["__measure"]
                .agg(measure_aggregation(measure))
                .unstack(fill_value=0.0)
            )
        else:
            pivot = frame.groupby(["__segment", "__period"]).size().unstack(fill_value=0)
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
        # A percentage from zero is undefined, but a segment that went from
        # nothing to something is usually the most newsworthy row in the file.
        # It leaves the ranking and keeps its sentence.
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
                    f"{newest[plan.dimension]} at "
                    f"{_shown(float(newest['Latest']), measure, working)}."
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
        direction = "slowest" if plan.ascending else "fastest"
        answer = (
            f"{leader[plan.dimension]} moved {direction}{_phrase(applied)}: {leader['Change %']:+.1f}% "
            f"({_shown(float(leader['Previous']), measure, working)} → "
            f"{_shown(float(leader['Latest']), measure, working)}) in the latest period."
        )
        if not from_nothing.empty:
            newest = from_nothing.loc[from_nothing["Latest"].abs().idxmax()]
            names = ", ".join(str(name) for name in from_nothing[plan.dimension])
            answer += (
                f" {len(from_nothing)} segment(s) are left out of the ranking because they "
                f"started from zero ({names}); the largest is {newest[plan.dimension]} at "
                f"{_shown(float(newest['Latest']), measure, working)}."
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
    return QueryAnswer(
        question="",
        plan=plan,
        answer=answer,
        calculation=_with_notes(f"(latest − previous) ÷ |previous| on period sums{scope}", series),
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
