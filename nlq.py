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

from business_insights import ColumnRoles, format_number, preferred_frequency, trend_frame

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

SUPERLATIVE_WORDS = (
    "best", "worst", "top", "bottom", "leading",
    "highest", "lowest", "largest", "biggest", "smallest",
)
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
    cleaned = re.sub(r"[^0-9a-z]+", " ", str(text).lower())
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


def parse_question(question: str, dataframe: pd.DataFrame, roles: ColumnRoles) -> QueryPlan | None:
    """Turn a plain-English question into an explicit plan, or None if unsupported."""
    q = _norm(question)
    if not q:
        return None

    numeric_columns = [column for column in roles.numeric if column in dataframe.columns]
    dimension_columns = [column for column in roles.dimensions if column in dataframe.columns]

    measure = _match_column(q, numeric_columns)
    dimension = _match_column(q, dimension_columns)
    year, month = _detect_time_filter(q)
    grain = _detect_grain(q)
    filters = _detect_value_filters(q, dataframe, roles, exclude=(dimension,))

    aggregation: Aggregation | None = next(
        (AGGREGATION_WORDS[word] for word in AGGREGATION_WORDS if re.search(rf"\b{word}\b", q)),
        None,
    )
    top_match = re.search(r"\b(top|bottom)\s+(\d{1,3})\b", q)
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

    if wants_count and not wants_growth:
        if dimension and wants_breakdown:
            return QueryPlan(intent="breakdown", aggregation="count", dimension=dimension, **base)
        countable = _match_countable(q, roles)
        return QueryPlan(intent="count", aggregation="count", count_column=countable, **base)

    if wants_growth and roles.date:
        wants_ranked_growth = superlative or any(word in q for word in ("which", "fastest", "slowest"))
        rank_dimension = dimension or (roles.dimension if wants_ranked_growth else None)
        ascending = bool(re.search(r"\b(slowest|least|declined|decreased|dropped|fell|shrank|worst)\b", q))
        return QueryPlan(intent="growth", dimension=rank_dimension, ascending=ascending, **base)

    if (top_match or superlative) and dimension:
        if top_match:
            top_n = int(top_match.group(2))
            ascending = top_match.group(1) == "bottom"
        else:
            top_n = 1
            ascending = bool(re.search(r"\b(worst|lowest|smallest|bottom)\b", q))
        return QueryPlan(
            intent="rank",
            aggregation=aggregation if aggregation in ("mean", "median") else "sum",
            dimension=dimension,
            top_n=top_n,
            ascending=ascending,
            **base,
        )

    if wants_trend and roles.date:
        return QueryPlan(intent="trend", aggregation="sum", **base)

    if dimension and (wants_breakdown or not measure):
        return QueryPlan(
            intent="breakdown",
            aggregation=aggregation if aggregation in ("mean", "median") else "sum",
            dimension=dimension,
            **base,
        )

    if base["measure"] and (aggregation or measure):
        return QueryPlan(intent="aggregate", aggregation=aggregation or "sum", **base)

    return None


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
    if roles.date and roles.date in working.columns and (plan.year or plan.month):
        dates = working[roles.date]
        if plan.year:
            working = working.loc[dates.dt.year == plan.year]
            dates = working[roles.date]
        if plan.month:
            working = working.loc[dates.dt.month == plan.month]
        month_name = next(
            (name.title() for name, number in MONTH_NAMES.items() if number == plan.month and len(name) > 3),
            None,
        )
        when = " ".join(part for part in (month_name, str(plan.year) if plan.year else None) if part)
        applied.append(f"{roles.date} in {when}")
    return working, applied


def _scoped(applied_filters: list[str]) -> str:
    return f" · filtered to {', '.join(applied_filters)}" if applied_filters else ""


def _aggregate_series(series: pd.Series, aggregation: Aggregation) -> float:
    return float(getattr(series.dropna(), aggregation)())


def _grouped_frame(
    dataframe: pd.DataFrame, plan: QueryPlan, value_label: str
) -> pd.DataFrame:
    assert plan.dimension is not None
    working = dataframe.dropna(subset=[plan.dimension])
    if plan.aggregation == "count" or not plan.measure:
        grouped = working.groupby(plan.dimension, as_index=False).size()
        grouped.columns = [plan.dimension, value_label]
    else:
        grouped = working.groupby(plan.dimension, as_index=False)[plan.measure].agg(plan.aggregation)
        grouped.columns = [plan.dimension, value_label]
    grouped = grouped.sort_values(value_label, ascending=plan.ascending)
    if plan.aggregation in ("sum", "count"):
        total = float(grouped[value_label].sum())
        if total:
            grouped["Share %"] = (grouped[value_label] / total * 100).round(1)
    return grouped.reset_index(drop=True)


def execute_plan(plan: QueryPlan, dataframe: pd.DataFrame, roles: ColumnRoles) -> QueryAnswer:
    """Run a validated plan locally and package the auditable answer."""
    for column in (plan.measure, plan.dimension, plan.count_column):
        if column is not None and column not in dataframe.columns:
            raise ValueError(f"Unknown column in plan: {column}")

    working, applied = _apply_filters(dataframe, plan, roles)
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
        answer = (
            f"{leader[plan.dimension]} is the {direction} {plan.dimension} by {value_label.lower()}"
            f"{_phrase(applied)} at {format_number(leader_value, plan.measure)}{share_note}."
        )
        order = "ascending" if plan.ascending else "descending"
        return QueryAnswer(
            question="",
            plan=plan,
            answer=answer,
            calculation=(
                f"{plan.aggregation}({plan.measure or 'rows'}) by {plan.dimension}, "
                f"{order}, showing {len(table)}{scope}"
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
        trend = trend_frame(working, scoped_roles, frequency=grain)
        if len(trend) < 2:
            return QueryAnswer(
                question="",
                plan=plan,
                answer="Not enough periods in that scope to draw a trend.",
                calculation=f"trend needs at least 2 periods{scope}",
            )
        first, last = float(trend.iloc[0]["Value"]), float(trend.iloc[-1]["Value"])
        change = (last - first) / abs(first) * 100 if first else 0.0
        grain_name = {"D": "day", "W": "week", "M": "month", "Q": "quarter", "Y": "year"}.get(grain, "period")
        answer = (
            f"{plan.measure or 'Records'} per {grain_name}{_phrase(applied)} moved from "
            f"{format_number(first, plan.measure)} to {format_number(last, plan.measure)} "
            f"({change:+.1f}% across {len(trend)} {grain_name}s)."
        )
        return QueryAnswer(
            question="",
            plan=plan,
            answer=answer,
            calculation=f"sum({plan.measure or 'rows'}) grouped per {grain_name}{scope}",
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
    trend = trend_frame(working, scoped_roles, frequency=plan.grain)
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
        frame = frame.assign(Period=frame[roles.date].dt.to_period(grain).dt.to_timestamp())
        frame = frame[frame["Period"].isin([previous_period, current_period])]
        if measure:
            pivot = frame.groupby([plan.dimension, "Period"])[measure].sum().unstack(fill_value=0.0)
        else:
            pivot = frame.groupby([plan.dimension, "Period"]).size().unstack(fill_value=0)
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
        result = result[result["Previous"] != 0]
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
        leader = result.iloc[0]
        direction = "slowest" if plan.ascending else "fastest"
        answer = (
            f"{leader[plan.dimension]} moved {direction}{_phrase(applied)}: {leader['Change %']:+.1f}% "
            f"({format_number(float(leader['Previous']), measure)} → "
            f"{format_number(float(leader['Latest']), measure)}) in the latest period."
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
        f"period ({format_number(previous, measure)} → {format_number(current, measure)})."
    )
    return QueryAnswer(
        question="",
        plan=plan,
        answer=answer,
        calculation=f"(latest − previous) ÷ |previous| on period sums{scope}",
        table=trend,
        chart="line",
    )


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
