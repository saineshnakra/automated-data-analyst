"""Deterministic natural-language queries over the analyzed dataset.

Each question becomes an explicit ``QueryPlan`` that pandas executes locally,
returning the answer with the calculation behind it. An optional AI planner may
emit the same plan shape for questions the rules cannot parse.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from .aggregation import TrendSeries, build_trend, measure_aggregation, preferred_frequency, unlabelled_label
from .formatting import format_change, format_number, format_values_apart, sentence_case
from .metrics import resolve_metric
from .schema import ColumnRoles

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

# Words asking for the bottom of a ranking. Each is also a superlative, so
# SUPERLATIVE_WORDS includes this tuple rather than repeating it.
ASCENDING_WORDS = ("worst", "bottom", "lowest", "smallest", "least", "fewest")
SUPERLATIVE_WORDS = (
    # "most" ranks from the top, like "highest"; "least" is in ASCENDING_WORDS.
    "best", "top", "leading", "highest", "largest", "biggest", "most",
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
    "a", "about", "across", "all", "also", "and", "any", "are", "been", "by", "can",
    "did", "do", "does", "done", "each", "for", "from", "get", "give", "got", "have",
    "has", "he", "how", "hows", "i", "in", "is", "it", "just", "last", "least", "make", "made",
    "many", "me", "most", "much", "my", "next", "of", "on", "only", "or", "our",
    "please", "re", "s", "she", "show", "tell", "than", "that", "the", "then", "there",
    "this", "to", "they", "us", "ve", "was", "we", "were", "what", "whats", "which",
    "with", "you", "your",
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
    """The name with simple English plurals added or removed ("category", "categories")."""
    variants = {normalized_name, f"{normalized_name}s", f"{normalized_name}es"}
    if normalized_name.endswith("s"):
        variants.add(normalized_name[:-1])
    if re.search(r"[^aeiou]y$", normalized_name):
        variants.add(f"{normalized_name[:-1]}ies")
    if normalized_name.endswith("ies"):
        variants.add(f"{normalized_name[:-3]}y")
    return variants


def _mentioned(normalized_name: str, question: str) -> bool:
    return any(
        re.search(rf"\b{re.escape(variant)}\b", question)
        for variant in _word_variants(normalized_name)
        if variant
    )


def _without_columns(question: str, columns: tuple[str | None, ...]) -> str:
    """Return the question with the matched column names blanked out.

    Column names contain words the intent scan would otherwise read: "Cell
    Count" would make "total cell count" a row count, and "Total Sales" would
    outrank an explicit "average".
    """
    blanked = question
    for column in columns:
        if not column:
            continue
        name = _norm(column)
        if name:
            blanked = re.sub(rf"\b{re.escape(name)}\b", " ", blanked)
    return " ".join(blanked.split())


def _unit(measure: str | None) -> str:
    """The measure's unit from the metric contract: amount, rate, level or count."""
    return resolve_metric(measure).unit if measure else "count"


def _match_column(question: str, columns: list[str]) -> str | None:
    """Return the column with the longest name mentioned in the question."""
    matches = [column for column in columns if _mentioned(_norm(column), question)]
    return max(matches, key=lambda column: len(_norm(column))) if matches else None


def _match_countable(question: str, roles: ColumnRoles) -> str | None:
    """Match 'how many <entities>' to an identifier or dimension column."""
    candidates = [column for column in (roles.identifier, *roles.dimensions) if column]
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


# Phrasings a plan cannot represent, such as "revenue > 200" or "last 30 days".
# They are refused rather than approximated, leaving them to the optional
# planner or the suggestion chips. Operators are matched on the raw text
# because _norm strips them.
UNSUPPORTED_OPERATORS = re.compile(r"[<>=\u2265\u2264\u2260]")
UNSUPPORTED_PHRASES = re.compile(
    r"\b(?:"
    r"greater than|less than|more than \d+|fewer than|at least \d+|at most \d+|"
    r"above \d+|below \d+|over \d+|under \d+|between \d+|exceeds?|exceeding|"
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
    # A grouped answer groups by the matched dimension, so its values are not
    # filters. Ungrouped answers keep them: "revenue for Region West" filters
    # to West even though "Region" also matched as the dimension.
    filters = _detect_value_filters(q, dataframe, roles, exclude=(dimension,))
    filters_including_dimension = _detect_value_filters(q, dataframe, roles)

    # Read intent words with the matched column names removed (see _without_columns).
    intent_text = _without_columns(q, (measure, dimension))
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
    wants_count = bool(re.search(r"\b(how many|count|number of)\b", intent_text))
    wants_growth = any(re.search(rf"\b{word}\b", intent_text) for word in GROWTH_WORDS)
    wants_trend = grain is not None or any(phrase in intent_text for phrase in TREND_WORDS)

    base = {
        "measure": measure or roles.measure,
        "filters": filters,
        "year": year,
        "month": month,
        "grain": grain,
    }

    ungrouped = {**base, "filters": filters_including_dimension}
    # Same default as the dashboard: rates average, amounts sum. An explicit
    # "total" or "sum" still sums.
    default_aggregation = measure_aggregation(base["measure"])

    if wants_count and not wants_growth:
        if dimension and wants_breakdown:
            return QueryPlan(intent="breakdown", aggregation="count", dimension=dimension, **base)
        countable = _match_countable(q, roles)
        return QueryPlan(intent="count", aggregation="count", count_column=countable, **ungrouped)

    if wants_growth and roles.date:
        wants_ranked_growth = superlative or bool(re.search(r"\b(which|fastest|slowest)\b", intent_text))
        rank_dimension = dimension or (roles.dimension if wants_ranked_growth else None)
        ascending = bool(re.search(ASCENDING_PATTERN, intent_text))
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
            ascending = bool(re.search(ASCENDING_PATTERN, intent_text))
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
            # The trend executor always uses the measure's default aggregation,
            # so "average revenue monthly" is refused (it would get sums) while
            # "average conversion rate monthly" is answered.
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
    """Choose the aggregation a rank or breakdown computes per group."""
    if aggregation in ("mean", "median"):
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
        mask = working[value_filter.column].astype(str).isin(value_filter.values)
        working = working.loc[mask]
        applied.append(f"{value_filter.column} in ({', '.join(value_filter.values)})")
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


def _scoped(applied_filters: list[str]) -> str:
    return f" · filtered to {', '.join(applied_filters)}" if applied_filters else ""


def _aggregate_series(series: pd.Series, aggregation: Aggregation) -> float:
    return float(getattr(series.dropna(), aggregation)())


def _grouped_frame(
    dataframe: pd.DataFrame, plan: QueryPlan, value_label: str
) -> pd.DataFrame:
    assert plan.dimension is not None
    # Keep unlabelled rows as their own group so shares add up to the full total.
    labelled = dataframe.copy()
    labelled[plan.dimension] = labelled[plan.dimension].astype(object).where(
        labelled[plan.dimension].notna(),
        unlabelled_label(labelled[plan.dimension].dropna().unique()),
    )
    working = labelled
    if plan.aggregation == "count" or not plan.measure:
        grouped = working.groupby(plan.dimension, as_index=False).size()
        grouped.columns = [plan.dimension, value_label]
    else:
        grouped = working.groupby(plan.dimension, as_index=False)[plan.measure].agg(plan.aggregation)
        grouped.columns = [plan.dimension, value_label]
    grouped = grouped.sort_values(value_label, ascending=plan.ascending)
    if plan.aggregation in ("sum", "count"):
        values = grouped[value_label].to_numpy(dtype=float)
        total = float(values.sum())
        # Shares need every group to have the same sign; mixed signs give values
        # like 500% and -250%.
        if total and ((values >= 0).all() or (values <= 0).all()):
            grouped["Share %"] = (grouped[value_label] / total * 100).round(1)
    return grouped.reset_index(drop=True)


def execute_plan(plan: QueryPlan, dataframe: pd.DataFrame, roles: ColumnRoles) -> QueryAnswer:
    """Run a plan locally and return the answer with its calculation trace.

    Raises ``ValueError`` for an unknown column or intent. The returned
    ``question`` is empty; ``answer_question`` fills it in.
    """
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
                f"{format_number(value, plan.measure)}, calculated from {rows:,} rows."
            ),
            calculation=f"{plan.aggregation}({plan.measure}){scope}",
        )

    if plan.intent in ("rank", "breakdown"):
        assert plan.dimension is not None
        label = AGGREGATION_LABELS[plan.aggregation]
        value_label = f"{label} {plan.measure}" if plan.measure and plan.aggregation != "count" else "Rows"
        grouped = _grouped_frame(working, plan, value_label)
        limit = plan.top_n if plan.intent == "rank" else BREAKDOWN_LIMIT
        table = grouped.head(limit or BREAKDOWN_LIMIT)
        leader = table.iloc[0]
        leader_value = float(leader[value_label])
        direction = "lowest" if plan.ascending else "leading"
        share_note = f" ({leader['Share %']:.1f}% of the total)" if "Share %" in table.columns else ""
        # Format a row count as a plain number, not in the measure's unit.
        counted = value_label == "Rows"
        answer = (
            f"{leader[plan.dimension]} is the {direction} {plan.dimension} by {sentence_case(value_label)}"
            f"{_phrase(applied)} at "
            f"{format_number(leader_value, None if counted else plan.measure)}{share_note}."
        )
        order = "ascending" if plan.ascending else "descending"
        return QueryAnswer(
            question="",
            plan=plan,
            answer=answer,
            calculation=(
                # Name what ran: row counts use .size(), which differs from
                # count(<measure>) when the measure has missing values.
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
        # Rates move in points and levels by a plain difference (see format_change).
        column = working[plan.measure].dropna() if plan.measure else None
        size = format_change(first, last, plan.measure, _unit(plan.measure), column, signed=True)
        movement = (
            f"{size} across {len(trend)} {grain_name}s"
            if size
            else f"across {len(trend)} {grain_name}s, from a starting period of zero"
        )
        start, end = format_values_apart(first, last, column=plan.measure)
        answer = (
            f"{plan.measure or 'Records'} per {grain_name}{_phrase(applied)} moved from "
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
        frame = working[[roles.date, plan.dimension] + ([measure] if measure else [])].dropna(
            subset=[roles.date, plan.dimension]
        )
        frame = frame.assign(__period=frame[roles.date].dt.to_period(grain).dt.to_timestamp())
        frame = frame[frame["__period"].isin([previous_period, current_period])]
        relative = not measure or resolve_metric(measure).additive
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
            return _rank_level_changes(plan, result, working, scope, applied)
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
                    f"{newest[plan.dimension]} at "
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
            f"{leader[plan.dimension]} {_ranked_move(leader['Change %'], plan.ascending)}"
            f"{_phrase(applied)}: {leader['Change %']:+.1f}% ({start} → {end}) in the latest period."
        )
        if not from_nothing.empty:
            newest = from_nothing.loc[from_nothing["Latest"].abs().idxmax()]
            names = ", ".join(str(name) for name in from_nothing[plan.dimension])
            answer += (
                f" {len(from_nothing)} segment(s) are left out of the ranking because they "
                f"started from zero ({names}); the largest is {newest[plan.dimension]} at "
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
    unit = _unit(measure)
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
    scope: str,
    applied: list[str],
) -> QueryAnswer:
    """Rank segments of a rate or level measure by the change in their average."""
    measure = plan.measure
    unit = _unit(measure)
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
        f"{leader[plan.dimension]} {_ranked_move(latest - previous, plan.ascending)}"
        f"{_phrase(applied)}: {size} ({start} → {end}) in the latest period."
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
    """Return up to four starter questions the rule-based parser can answer."""
    suggestions: list[str] = []
    if roles.measure:
        # "Average Order Value" already says how it combines; don't prefix it again.
        first_word = (roles.measure.split() or [""])[0].lower()
        if first_word in {"average", "avg", "mean", "median", "total"}:
            suggestions.append(roles.measure)
        else:
            suggestions.append(f"{resolve_metric(roles.measure).combines_as.capitalize()} {roles.measure}")
    if roles.measure and roles.dimension:
        suggestions.append(f"Top 5 {roles.dimension} by {roles.measure}")
    if roles.measure and roles.date:
        suggestions.append(f"Monthly {roles.measure} trend")
    if roles.dimension and roles.date:
        suggestions.append(f"Which {roles.dimension} grew fastest?")
    if not suggestions:
        suggestions.append("How many rows are there?")
    return suggestions[:4]
