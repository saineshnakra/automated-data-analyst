"""Explainable business intelligence built from deterministic calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


@dataclass(frozen=True)
class ColumnRoles:
    date: str | None
    measure: str | None
    dimension: str | None
    identifier: str | None
    numeric: tuple[str, ...]
    dimensions: tuple[str, ...]


@dataclass(frozen=True)
class KPI:
    label: str
    value: str
    context: str
    tone: str = "neutral"


@dataclass(frozen=True)
class Evidence:
    kind: str
    title: str
    value: str
    statement: str
    calculation: str
    tone: str = "neutral"
    subject: str | None = None


@dataclass(frozen=True)
class Recommendation:
    priority: str
    title: str
    action: str
    rationale: str


@dataclass(frozen=True)
class BusinessBrief:
    headline: str
    summary: str
    roles: ColumnRoles
    kpis: tuple[KPI, ...]
    evidence: tuple[Evidence, ...]
    recommendations: tuple[Recommendation, ...]


MEASURE_KEYWORDS = {
    "revenue": 14,
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
    "quantity": 6,
    "units": 6,
    "orders": 6,
    "volume": 6,
    "balance": 6,
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
    "status": 6,
    "type": 5,
}

CURRENCY_TOKENS = (
    "revenue",
    "sales",
    "gmv",
    "profit",
    "amount",
    "income",
    "spend",
    "cost",
    "expense",
    "price",
    "balance",
)

IDENTIFIER_TOKENS = ("id", "uuid", "key", "code", "number", "invoice", "order")
TIME_PART_TOKENS = ("year", "month", "week", "day", "hour", "minute", "quarter")


def _normalized(name: str) -> str:
    return " ".join(name.lower().replace("_", " ").replace("-", " ").split())


def _keyword_score(name: str, keywords: dict[str, int]) -> int:
    normalized = _normalized(name)
    return max((score for token, score in keywords.items() if token in normalized), default=0)


def _looks_like_identifier(name: str, series: pd.Series) -> bool:
    normalized = _normalized(name)
    token_match = any(token in normalized.split() for token in IDENTIFIER_TOKENS)
    unique_ratio = series.nunique(dropna=True) / max(int(series.notna().sum()), 1)
    return token_match and unique_ratio >= 0.8


def detect_roles(dataframe: pd.DataFrame) -> ColumnRoles:
    """Infer likely business roles from names, types, and cardinality."""
    numeric = dataframe.select_dtypes(include=np.number).columns.tolist()
    date_columns = [
        column for column in dataframe.columns if is_datetime64_any_dtype(dataframe[column])
    ]

    date = max(
        date_columns,
        key=lambda column: (
            1 if any(token in _normalized(column) for token in ("date", "time", "created")) else 0,
            int(dataframe[column].notna().sum()),
        ),
        default=None,
    )

    measure_candidates: list[tuple[int, float, str]] = []
    for column in numeric:
        series = dataframe[column]
        name = _normalized(column)
        score = _keyword_score(column, MEASURE_KEYWORDS)
        if _looks_like_identifier(column, series):
            score -= 20
        if any(token == name or name.endswith(f" {token}") for token in TIME_PART_TOKENS):
            score -= 15
        non_null_ratio = float(series.notna().mean())
        measure_candidates.append((score, non_null_ratio, column))

    measure = None
    if measure_candidates:
        measure = max(measure_candidates, key=lambda candidate: (candidate[0], candidate[1]))[2]

    dimensions: list[str] = []
    dimension_candidates: list[tuple[int, int, str]] = []
    for column in dataframe.columns:
        if column == date or column in numeric:
            continue
        series = dataframe[column]
        unique = int(series.nunique(dropna=True))
        non_null = int(series.notna().sum())
        if unique < 2 or unique > 100 or unique / max(non_null, 1) > 0.65:
            continue
        dimensions.append(column)
        score = _keyword_score(column, DIMENSION_KEYWORDS)
        preferred_size = -abs(unique - 10)
        dimension_candidates.append((score, preferred_size, column))

    dimension = (
        max(dimension_candidates, key=lambda candidate: (candidate[0], candidate[1]))[2]
        if dimension_candidates
        else None
    )

    identifier_candidates = [
        column
        for column in dataframe.columns
        if _looks_like_identifier(column, dataframe[column])
    ]
    identifier = identifier_candidates[0] if identifier_candidates else None

    return ColumnRoles(
        date=date,
        measure=measure,
        dimension=dimension,
        identifier=identifier,
        numeric=tuple(numeric),
        dimensions=tuple(dimensions),
    )


def override_roles(
    roles: ColumnRoles,
    *,
    date: str | None = None,
    measure: str | None = None,
    dimension: str | None = None,
) -> ColumnRoles:
    return ColumnRoles(
        date=date if date is not None else roles.date,
        measure=measure if measure is not None else roles.measure,
        dimension=dimension if dimension is not None else roles.dimension,
        identifier=roles.identifier,
        numeric=roles.numeric,
        dimensions=roles.dimensions,
    )


def _is_currency(column: str | None) -> bool:
    return bool(column and any(token in _normalized(column) for token in CURRENCY_TOKENS))


def format_number(value: float, column: str | None = None, *, compact: bool = True) -> str:
    """Format a metric according to likely business meaning."""
    if not np.isfinite(value):
        return "—"

    absolute = abs(value)
    prefix = "$" if _is_currency(column) else ""
    suffix = ""
    scaled = value
    if compact and absolute >= 1_000_000_000:
        scaled, suffix = value / 1_000_000_000, "B"
    elif compact and absolute >= 1_000_000:
        scaled, suffix = value / 1_000_000, "M"
    elif compact and absolute >= 1_000:
        scaled, suffix = value / 1_000, "K"

    if suffix:
        return f"{prefix}{scaled:,.1f}{suffix}"
    if float(value).is_integer() and not _is_currency(column):
        return f"{int(value):,}"
    return f"{prefix}{value:,.2f}"


def _period_frequency(date_series: pd.Series) -> tuple[str, str]:
    span_days = max((date_series.max() - date_series.min()).days, 0)
    if span_days <= 120:
        return "W", "week"
    if span_days <= 900:
        return "M", "month"
    return "Q", "quarter"


def trend_frame(dataframe: pd.DataFrame, roles: ColumnRoles) -> pd.DataFrame:
    """Aggregate the selected measure over a human-sized time grain."""
    if not roles.date:
        return pd.DataFrame(columns=["Period", "Value"])

    columns = [roles.date] + ([roles.measure] if roles.measure else [])
    working = dataframe[columns].dropna(subset=[roles.date]).copy()
    if working.empty:
        return pd.DataFrame(columns=["Period", "Value"])

    frequency, _ = _period_frequency(working[roles.date])
    working["Period"] = working[roles.date].dt.to_period(frequency).dt.to_timestamp()
    if roles.measure:
        result = working.groupby("Period", as_index=False)[roles.measure].sum()
        result = result.rename(columns={roles.measure: "Value"})
    else:
        result = working.groupby("Period", as_index=False).size().rename(columns={"size": "Value"})
    return result.sort_values("Period").reset_index(drop=True)


def segment_frame(dataframe: pd.DataFrame, roles: ColumnRoles, limit: int = 12) -> pd.DataFrame:
    """Rank the selected business segment by the selected measure or record count."""
    if not roles.dimension:
        return pd.DataFrame(columns=["Segment", "Value"])

    working = dataframe.dropna(subset=[roles.dimension]).copy()
    if working.empty:
        return pd.DataFrame(columns=["Segment", "Value"])

    if roles.measure:
        result = working.groupby(roles.dimension, as_index=False)[roles.measure].sum()
        result = result.rename(columns={roles.dimension: "Segment", roles.measure: "Value"})
    else:
        result = (
            working.groupby(roles.dimension, as_index=False)
            .size()
            .rename(columns={roles.dimension: "Segment", "size": "Value"})
        )
    return result.sort_values("Value", ascending=False).head(limit).reset_index(drop=True)


def _growth_evidence(dataframe: pd.DataFrame, roles: ColumnRoles) -> Evidence | None:
    trend = trend_frame(dataframe, roles)
    if len(trend) < 2:
        return None

    previous = float(trend.iloc[-2]["Value"])
    current = float(trend.iloc[-1]["Value"])
    if previous == 0:
        return None
    change = (current - previous) / abs(previous) * 100
    period = trend.iloc[-1]["Period"].strftime("%b %Y")
    measure = roles.measure or "Records"
    direction = "increased" if change >= 0 else "decreased"
    return Evidence(
        kind="trend",
        title=f"Latest {measure.lower()} movement",
        value=f"{change:+.1f}%",
        statement=(
            f"{measure} {direction} {abs(change):.1f}% in the latest observed period "
            f"({period}), from {format_number(previous, roles.measure)} to "
            f"{format_number(current, roles.measure)}."
        ),
        calculation="(Latest period − previous period) ÷ |previous period|",
        tone="positive" if change >= 0 else "negative",
    )


def _change_driver_evidence(dataframe: pd.DataFrame, roles: ColumnRoles) -> Evidence | None:
    """Identify the segment contributing most to the latest net movement."""
    if not roles.date or not roles.measure or not roles.dimension:
        return None

    trend = trend_frame(dataframe, roles)
    if len(trend) < 2:
        return None
    previous_period = trend.iloc[-2]["Period"]
    current_period = trend.iloc[-1]["Period"]
    frequency, _ = _period_frequency(dataframe[roles.date].dropna())
    working = dataframe[[roles.date, roles.measure, roles.dimension]].dropna().copy()
    working["Period"] = working[roles.date].dt.to_period(frequency).dt.to_timestamp()
    comparison = working[working["Period"].isin([previous_period, current_period])]
    grouped = comparison.groupby([roles.dimension, "Period"])[roles.measure].sum().unstack(fill_value=0)
    if previous_period not in grouped or current_period not in grouped:
        return None

    grouped["Change"] = grouped[current_period] - grouped[previous_period]
    net_change = float(grouped["Change"].sum())
    if net_change == 0 or grouped["Change"].abs().max() == 0:
        return None

    driver_name = (
        str(grouped["Change"].idxmax()) if net_change > 0 else str(grouped["Change"].idxmin())
    )
    previous_value = float(grouped.loc[driver_name, previous_period])
    current_value = float(grouped.loc[driver_name, current_period])
    driver_change = float(grouped.loc[driver_name, "Change"])
    share = abs(driver_change / net_change) * 100
    direction = "increased" if driver_change > 0 else "decreased"
    sign = "+" if driver_change > 0 else "−"
    return Evidence(
        kind="driver",
        title="Largest change driver",
        value=f"{sign}{format_number(abs(driver_change), roles.measure)}",
        statement=(
            f"{driver_name} was the largest {roles.dimension.lower()} driver: "
            f"{roles.measure} {direction} by {format_number(abs(driver_change), roles.measure)}, "
            f"from {format_number(previous_value, roles.measure)} to "
            f"{format_number(current_value, roles.measure)}. That is equivalent to "
            f"{share:.1f}% of the net movement."
        ),
        calculation=(
            f"Latest {driver_name} {roles.measure} − previous {driver_name} {roles.measure}; "
            "ranked across segments"
        ),
        tone="positive" if driver_change > 0 else "negative",
        subject=driver_name,
    )


def _segment_evidence(dataframe: pd.DataFrame, roles: ColumnRoles) -> tuple[Evidence, Evidence] | tuple[()]:
    segments = segment_frame(dataframe, roles, limit=100)
    if segments.empty:
        return ()

    total = float(segments["Value"].sum())
    if total == 0:
        return ()
    leader = segments.iloc[0]
    leader_share = float(leader["Value"] / total * 100)
    top_three_share = float(segments.head(3)["Value"].sum() / total * 100)
    measure = roles.measure or "records"
    dimension = roles.dimension or "segment"
    return (
        Evidence(
            kind="leader",
            title=f"Leading {dimension.lower()}",
            value=f"{leader_share:.1f}%",
            statement=(
                f"{leader['Segment']} is the largest {dimension.lower()}, contributing "
                f"{leader_share:.1f}% of {measure.lower()} "
                f"({format_number(float(leader['Value']), roles.measure)})."
            ),
            calculation=f"{leader['Segment']} {measure} ÷ total {measure}",
            tone="positive",
        ),
        Evidence(
            kind="concentration",
            title="Top-three concentration",
            value=f"{top_three_share:.1f}%",
            statement=(
                f"The top three {dimension.lower()} values account for {top_three_share:.1f}% "
                f"of measured {measure.lower()}."
            ),
            calculation=f"Top three {dimension} {measure} ÷ total {measure}",
            tone="warning" if top_three_share >= 70 else "neutral",
        ),
    )


def _relationship_evidence(dataframe: pd.DataFrame, roles: ColumnRoles) -> Evidence | None:
    usable = [
        column
        for column in roles.numeric
        if not _looks_like_identifier(column, dataframe[column])
        and not any(token == _normalized(column) for token in TIME_PART_TOKENS)
    ]
    if len(usable) < 2:
        return None
    correlations = dataframe[usable].corr()
    upper = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(bool))
    stacked = upper.stack().dropna()
    if stacked.empty:
        return None
    pair = stacked.abs().idxmax()
    value = float(correlations.loc[pair[0], pair[1]])
    if abs(value) < 0.45:
        return None
    relationship = "move together" if value > 0 else "move in opposite directions"
    return Evidence(
        kind="relationship",
        title="Strongest measurable relationship",
        value=f"r = {value:.2f}",
        statement=(
            f"{pair[0]} and {pair[1]} {relationship}; their Pearson correlation is {value:.2f}. "
            "This is an association, not proof of causation."
        ),
        calculation="Pearson correlation across non-missing paired values",
    )


def _outlier_evidence(dataframe: pd.DataFrame, roles: ColumnRoles) -> Evidence | None:
    if not roles.measure:
        return None
    series = dataframe[roles.measure].dropna()
    if len(series) < 12 or series.nunique() < 4:
        return None
    first_quartile, third_quartile = series.quantile([0.25, 0.75])
    iqr = third_quartile - first_quartile
    if iqr == 0:
        return None
    outlier_count = int(
        ((series < first_quartile - 1.5 * iqr) | (series > third_quartile + 1.5 * iqr)).sum()
    )
    if outlier_count == 0:
        return None
    rate = outlier_count / len(series) * 100
    return Evidence(
        kind="outliers",
        title="Exceptional records",
        value=f"{outlier_count:,}",
        statement=(
            f"{outlier_count:,} {roles.measure} values ({rate:.1f}% of non-missing records) "
            "sit outside the standard 1.5×IQR range."
        ),
        calculation="Values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR",
        tone="warning",
    )


def _quality_evidence(dataframe: pd.DataFrame) -> Evidence | None:
    total_cells = dataframe.shape[0] * dataframe.shape[1]
    missing_cells = int(dataframe.isna().sum().sum())
    if total_cells == 0 or missing_cells == 0:
        return None
    rate = missing_cells / total_cells * 100
    return Evidence(
        kind="quality",
        title="Data completeness",
        value=f"{100 - rate:.1f}%",
        statement=(
            f"{missing_cells:,} cells are missing, leaving the analyzed dataset "
            f"{100 - rate:.1f}% complete."
        ),
        calculation="Non-missing cells ÷ all cells",
        tone="warning" if rate >= 5 else "neutral",
    )


def _recommendations(evidence: list[Evidence], roles: ColumnRoles) -> tuple[Recommendation, ...]:
    recommendations: list[Recommendation] = []
    by_kind = {item.kind: item for item in evidence}

    trend = by_kind.get("trend")
    driver = by_kind.get("driver")
    if trend and trend.tone == "negative":
        recommendations.append(
            Recommendation(
                "Now",
                (
                    f"Start with {driver.subject}"
                    if driver and driver.subject
                    else "Find where the decline started"
                ),
                (
                    f"Reconcile the {driver.subject} decline by customer, channel, and transaction; "
                    "separate lost volume from pricing or mix."
                    if driver and driver.subject
                    else "Break the latest period down by "
                    f"{roles.dimension or 'your main operating segment'} and compare each segment "
                    "with its prior period."
                ),
                f"{trend.statement} {driver.statement}" if driver else trend.statement,
            )
        )
    elif trend:
        recommendations.append(
            Recommendation(
                "Now",
                (
                    f"Make {driver.subject}'s growth repeatable"
                    if driver and driver.subject
                    else "Protect the growth driver"
                ),
                (
                    f"Break {driver.subject}'s lift into volume, pricing, and mix; preserve the "
                    "repeatable driver and test it in the next-best segment."
                    if driver and driver.subject
                    else f"Identify which {roles.dimension or 'operating segment'} created the "
                    "latest increase, then test whether that lift is repeatable rather than one-off."
                ),
                f"{trend.statement} {driver.statement}" if driver else trend.statement,
            )
        )

    concentration = by_kind.get("concentration")
    leader = by_kind.get("leader")
    if concentration and concentration.tone == "warning":
        recommendations.append(
            Recommendation(
                "Next",
                "Reduce concentration risk",
                (
                    f"Stress-test the business if the leading {roles.dimension or 'segment'} "
                    "falls 10–20%, and build a growth plan for the next two segments."
                ),
                concentration.statement,
            )
        )
    elif leader:
        recommendations.append(
            Recommendation(
                "Next",
                "Replicate the leader's playbook",
                (
                    f"Compare the leading {roles.dimension or 'segment'} with the median on "
                    "pricing, volume, and mix; scale the difference that is operationally controllable."
                ),
                leader.statement,
            )
        )

    relationship = by_kind.get("relationship")
    if relationship:
        recommendations.append(
            Recommendation(
                "Test",
                "Validate a potential operating lever",
                (
                    "Run a segmented or time-controlled analysis before acting. If the relationship "
                    "survives, test a small intervention and measure the outcome."
                ),
                relationship.statement,
            )
        )

    outliers = by_kind.get("outliers")
    if outliers:
        recommendations.append(
            Recommendation(
                "Watch",
                "Audit exceptional records",
                (
                    "Review the largest exceptions for data-entry errors, refunds, enterprise deals, "
                    "or operational incidents before they distort planning."
                ),
                outliers.statement,
            )
        )

    quality = by_kind.get("quality")
    if quality and quality.tone == "warning":
        recommendations.append(
            Recommendation(
                "Fix",
                "Close the measurement gap",
                (
                    "Prioritize missing fields used in revenue, margin, customer, or time analysis; "
                    "decisions based on incomplete segments can be systematically biased."
                ),
                quality.statement,
            )
        )

    if not recommendations:
        recommendations.append(
            Recommendation(
                "Next",
                "Add decision context",
                (
                    "Include a date, a business outcome such as revenue or cost, and a segment such "
                    "as product or region to unlock trend and driver analysis."
                ),
                "The current schema does not expose enough business structure for a specific recommendation.",
            )
        )
    return tuple(recommendations[:4])


def analyze_business(dataframe: pd.DataFrame, roles: ColumnRoles | None = None) -> BusinessBrief:
    """Create an executive brief from explainable calculations and rule-based interpretation."""
    roles = roles or detect_roles(dataframe)
    evidence: list[Evidence] = []

    growth = _growth_evidence(dataframe, roles)
    if growth:
        evidence.append(growth)
    change_driver = _change_driver_evidence(dataframe, roles)
    if change_driver:
        evidence.append(change_driver)
    evidence.extend(_segment_evidence(dataframe, roles))
    for optional_evidence in (
        _relationship_evidence(dataframe, roles),
        _outlier_evidence(dataframe, roles),
        _quality_evidence(dataframe),
    ):
        if optional_evidence:
            evidence.append(optional_evidence)

    kpis: list[KPI] = []
    if roles.measure:
        measure_values = dataframe[roles.measure].dropna()
        total = float(measure_values.sum())
        average = float(measure_values.mean())
        kpis.extend(
            [
                KPI(
                    f"Total {roles.measure}",
                    format_number(total, roles.measure),
                    "Across all analyzed records",
                ),
                KPI(
                    f"Average {roles.measure}",
                    format_number(average, roles.measure),
                    "Per non-missing record",
                ),
            ]
        )

    if growth:
        kpis.append(
            KPI(
                "Latest movement",
                growth.value,
                "Versus the previous observed period",
                growth.tone,
            )
        )
    if evidence_by_kind := {item.kind: item for item in evidence}:
        if leader := evidence_by_kind.get("leader"):
            kpis.append(KPI(f"Top {roles.dimension}", leader.value, "Share contributed by the leader"))

    while len(kpis) < 4:
        if roles.identifier and not any(item.label == f"Distinct {roles.identifier}" for item in kpis):
            kpis.append(
                KPI(
                    f"Distinct {roles.identifier}",
                    f"{dataframe[roles.identifier].nunique(dropna=True):,}",
                    "Unique entities in the dataset",
                )
            )
        elif not any(item.label == "Records analyzed" for item in kpis):
            kpis.append(KPI("Records analyzed", f"{len(dataframe):,}", "After conservative cleaning"))
        else:
            completeness = 1 - dataframe.isna().sum().sum() / max(dataframe.size, 1)
            kpis.append(KPI("Data completeness", f"{completeness * 100:.1f}%", "Share of populated cells"))
    kpis = kpis[:4]

    recommendations = _recommendations(evidence, roles)
    growth_text = growth.statement if growth else None
    leader = next((item for item in evidence if item.kind == "leader"), None)
    if growth and leader:
        headline = (
            f"{growth.value} latest movement, with {leader.value} coming from the leading "
            f"{roles.dimension.lower()}."
        )
    elif growth:
        headline = growth.statement.split(".")[0] + "."
    elif leader:
        headline = leader.statement
    elif roles.measure:
        total_measure = format_number(float(dataframe[roles.measure].sum()), roles.measure)
        headline = f"{roles.measure} totals {total_measure} across the analyzed data."
    else:
        headline = f"{len(dataframe):,} records are ready for operational review."

    summary_parts = []
    if growth_text:
        summary_parts.append(growth_text)
    if leader:
        summary_parts.append(leader.statement)
    if not summary_parts:
        summary_parts.append(
            "ADA found the clearest available signals in the uploaded schema and separated "
            "calculations from recommendations."
        )

    return BusinessBrief(
        headline=headline,
        summary=" ".join(summary_parts),
        roles=roles,
        kpis=tuple(kpis),
        evidence=tuple(evidence[:6]),
        recommendations=recommendations,
    )


def build_business_report(
    dataframe: pd.DataFrame,
    brief: BusinessBrief,
    *,
    source_name: str,
    context: str = "",
) -> str:
    """Create a portable, evidence-linked executive report."""
    lines = [
        "# ADA Executive Brief",
        "",
        f"**Source:** {source_name}",
        f"**Rows analyzed:** {len(dataframe):,}",
    ]
    if context.strip():
        lines.append(f"**Business context:** {context.strip()}")
    lines.extend(["", "## Executive read", "", brief.headline, "", brief.summary, ""])
    lines.extend(["## What the data says", ""])
    if brief.evidence:
        for item in brief.evidence:
            lines.extend(
                [
                    f"### {item.title} — {item.value}",
                    "",
                    item.statement,
                    "",
                    f"_Calculation: {item.calculation}_",
                    "",
                ]
            )
    else:
        lines.extend(["No sufficiently strong business signals were detected.", ""])

    lines.extend(["## What ADA recommends", ""])
    for item in brief.recommendations:
        lines.extend(
            [
                f"### {item.priority}: {item.title}",
                "",
                item.action,
                "",
                f"_Why: {item.rationale}_",
                "",
            ]
        )
    lines.extend(
        [
            "---",
            "Recommendations are deterministic interpretations of the calculations above, not causal proof.",
            "Uploaded data was not sent to an external AI service.",
        ]
    )
    return "\n".join(lines)
