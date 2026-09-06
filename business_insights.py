"""Explainable business intelligence built from deterministic calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aggregation import (
    build_trend,
    preferred_frequency,
    segment_frame,
    segment_period_change,
    trend_frame,
)
from anomalies import detect_anomalies
from formatting import (
    format_number,
    format_percentage,
    format_period,
    is_percentage,
    normalized_name,
    percentage_outranks_currency,
)
from schema import TIME_PART_TOKENS, ColumnRoles, detect_roles, looks_like_identifier
from timeseries import robust_scale


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


MIN_PERIODS_FOR_VOLATILITY = 5


def _period_changes(values: np.ndarray) -> np.ndarray:
    """Period-over-period percentage changes, skipping divisions by zero."""
    previous, current = values[:-1], values[1:]
    usable = previous != 0
    return (current[usable] - previous[usable]) / np.abs(previous[usable]) * 100


def _movement_in_context(values: np.ndarray, change: float) -> str:
    """Say whether the latest movement is unusual for this particular series.

    A metric that routinely swings 30% has not told you anything by swinging
    30% again. Without that context every movement reads as a development.
    """
    prior = _period_changes(values[:-1])
    if len(prior) < MIN_PERIODS_FOR_VOLATILITY:
        return ""

    spread = robust_scale(prior)
    typical = float(np.median(np.abs(prior)))
    if spread == 0:
        return ""

    distance = abs(change - float(np.median(prior))) / spread
    # The measurement is distance from the norm, not size. A flat month in a
    # series that always moves 5% is unusual, but calling 0.0% "an unusually
    # large swing" describes the opposite of what happened.
    moved_more = abs(change) > typical
    if distance < 1.0:
        verdict = "This is within normal period-to-period variation"
    elif distance < 2.0:
        verdict = "This is a larger swing than usual" if moved_more else "This is quieter than usual"
    else:
        verdict = (
            "This is an unusually large swing"
            if moved_more
            else "This is an unusually quiet period for this series"
        )
    return f" {verdict} — this series typically moves about {format_percentage(typical)} per period."


def _growth_evidence(dataframe: pd.DataFrame, roles: ColumnRoles) -> Evidence | None:
    series = build_trend(dataframe, roles)
    trend = series.frame
    if len(trend) < 2:
        return None

    values = trend["Value"].to_numpy(dtype=float)
    previous = float(values[-2])
    current = float(values[-1])
    if previous == 0 or not np.isfinite(previous) or not np.isfinite(current):
        return None
    change = (current - previous) / abs(previous) * 100
    # A quarter is "Q4 2023", not "Oct 2023" -- which the anomaly card and the
    # chart axis already knew, so the brief was contradicting its own page.
    period = format_period(trend.iloc[-1]["Period"], series.frequency)
    measure = roles.measure or "Records"
    measure_values = dataframe[roles.measure].dropna() if roles.measure else None
    direction = "increased" if change >= 0 else "decreased"
    context = _movement_in_context(values, change)
    return Evidence(
        kind="trend",
        title=f"Latest {measure.lower()} movement",
        value=format_percentage(change, signed=True),
        statement=(
            f"{measure} {direction} {format_percentage(abs(change))} in the latest complete period "
            f"({period}), from {format_number(previous, roles.measure, column_values=measure_values)} to "
            f"{format_number(current, roles.measure, column_values=measure_values)}.{context}"
        ),
        calculation=(
            "(Latest period − previous period) ÷ |previous period|, set against the spread "
            "of past period-over-period changes"
        ),
        tone="positive" if change >= 0 else "negative",
    )


def _change_driver_evidence(dataframe: pd.DataFrame, roles: ColumnRoles) -> Evidence | None:
    """Identify the segment contributing most to the latest net movement."""
    result = segment_period_change(dataframe, roles)
    if result is None:
        return None
    grouped, previous_period, current_period = result
    measure_values = dataframe[roles.measure].dropna() if roles.measure else None
    changes = grouped["Change"]
    net_change = float(changes.sum())
    gross_change = float(changes.abs().sum())
    if gross_change == 0:
        return None

    # Look the row up by its real index value; a boolean segment column has an
    # index of True/False, and str() turned that into a KeyError.
    driver_key = changes.abs().idxmax()
    driver_name = str(driver_key)
    previous_value = float(grouped.loc[driver_key, previous_period])
    current_value = float(grouped.loc[driver_key, current_period])
    driver_change = float(grouped.loc[driver_key, "Change"])
    direction = "increased" if driver_change > 0 else "decreased"
    sign = "+" if driver_change > 0 else "−"

    movement = (
        f"{driver_name} moved the most of any {roles.dimension.lower()}: "
        f"{roles.measure} {direction} by "
        f"{format_number(abs(driver_change), roles.measure, column_values=measure_values)}, "
        f"from {format_number(previous_value, roles.measure, column_values=measure_values)} to "
        f"{format_number(current_value, roles.measure, column_values=measure_values)}."
    )

    # Segments that cancel out leave a tiny net change, and a share of that
    # net reads as an absurd multiple. Measure against total movement instead
    # and say plainly that the offsetting is what is really going on.
    if abs(net_change) < OFFSETTING_NET_RATIO * gross_change:
        share = abs(driver_change) / gross_change * 100
        statement = (
            f"{movement} Segments largely offset each other this period — "
            f"{format_number(gross_change, roles.measure, column_values=measure_values)} "
            f"of movement nets to just "
            f"{format_number(abs(net_change), roles.measure, column_values=measure_values)} — so this is "
            f"{format_percentage(share)} of all movement rather than of the net."
        )
        calculation = (
            f"Latest {driver_name} {roles.measure} − previous; ranked by absolute change; "
            "share taken against total absolute movement because the net is near zero"
        )
    else:
        share = abs(driver_change / net_change) * 100
        statement = f"{movement} That is equivalent to {format_percentage(share)} of the net movement."
        calculation = (
            f"Latest {driver_name} {roles.measure} − previous {driver_name} {roles.measure}; "
            "ranked across segments"
        )

    return Evidence(
        kind="driver",
        title="Largest change driver",
        value=f"{sign}{format_number(abs(driver_change), roles.measure, column_values=measure_values)}",
        statement=statement,
        calculation=calculation,
        tone="positive" if driver_change > 0 else "negative",
        subject=driver_name,
    )


def _anomaly_evidence(dataframe: pd.DataFrame, roles: ColumnRoles) -> Evidence | None:
    if not roles.date:
        return None
    anomalies = detect_anomalies(trend_frame(dataframe, roles))
    if not anomalies:
        return None
    grain = preferred_frequency(dataframe[roles.date])
    worst = anomalies[0]
    measure = roles.measure or "Records"
    measure_values = dataframe[roles.measure].dropna() if roles.measure else None
    label = format_period(worst.period, grain)
    plural = "periods sit" if len(anomalies) > 1 else "period sits"
    return Evidence(
        kind="anomaly",
        title="Anomalous periods",
        value=f"{len(anomalies)}",
        statement=(
            f"{label} is the sharpest anomaly: {measure.lower()} reached "
            f"{format_number(worst.value, roles.measure, column_values=measure_values)}, "
            f"{worst.direction} the expected "
            f"{format_number(worst.expected_low, roles.measure, column_values=measure_values)}–"
            f"{format_number(worst.expected_high, roles.measure, column_values=measure_values)} range. "
            f"{len(anomalies)} {plural} outside the trendline band."
        ),
        calculation=(
            "Period totals vs Theil–Sen trendline ± a band calibrated so that only "
            "1 stable series in 20 raises a flag"
        ),
        tone="warning",
    )


CONCENTRATED_EFFECTIVE_SEGMENTS = 3.0


def _effective_segments(values: np.ndarray, total: float) -> float | None:
    """How many equally sized segments the business really rests on.

    The reciprocal of the Herfindahl index. Ten segments where one holds
    ninety percent of the total behave like barely more than one, which the
    top-three share cannot express -- and with only four segments in the
    file, "the top three hold 80%" is nearly a tautology.

    Undefined when any segment is negative, since a share of a total that
    parts of it subtract from means nothing.
    """
    if total <= 0 or (values < 0).any():
        return None
    shares = values / total
    herfindahl = float(np.sum(shares**2))
    return 1.0 / herfindahl if herfindahl > 0 else None


def _shares_are_meaningful(values: np.ndarray) -> bool:
    """A share of a total that parts of it subtract from is not a share.

    All-positive is the ordinary case. All-negative is a cost or loss column,
    where -6k of -20k is honestly 30%. Mixed signs are the case with no answer:
    the denominator is a net figure the parts do not sum into, which is how a
    segment ends up "contributing 2,000,000% of profit".
    """
    if not values.size:
        return False
    return bool((values >= 0).all() or (values <= 0).all())


def _segment_evidence(dataframe: pd.DataFrame, roles: ColumnRoles) -> tuple[Evidence, ...]:
    segments = segment_frame(dataframe, roles, limit=100)
    if segments.empty:
        return ()

    total = float(segments["Value"].sum())
    if total == 0 or not np.isfinite(total):
        return ()
    values = segments["Value"].to_numpy(dtype=float)
    shares_hold = _shares_are_meaningful(values)
    if not shares_hold:
        # Without a usable denominator the only honest ordering is by size of
        # the number itself, and no percentage may be quoted from it.
        segments = segments.reindex(
            segments["Value"].abs().sort_values(ascending=False).index
        ).reset_index(drop=True)
    elif total < 0:
        # A cost column: the biggest contributor is the most negative one, not
        # the one nearest zero that a descending sort puts on top.
        segments = segments.sort_values("Value").reset_index(drop=True)
    leader = segments.iloc[0]
    leader_share = float(leader["Value"] / total * 100) if shares_hold else float("nan")
    top_three_share = (
        float(segments.head(3)["Value"].sum() / total * 100) if shares_hold else float("nan")
    )
    effective = _effective_segments(segments["Value"].to_numpy(dtype=float), total)
    measure = roles.measure or "records"
    measure_values = dataframe[roles.measure].dropna() if roles.measure else None
    dimension = roles.dimension or "segment"
    leader_amount = format_number(
        float(leader["Value"]), roles.measure, column_values=measure_values
    )
    return (
        Evidence(
            kind="leader",
            title=f"Leading {dimension.lower()}",
            value=format_percentage(leader_share) if shares_hold else leader_amount,
            statement=(
                (
                    f"{leader['Segment']} is the largest {dimension.lower()}, contributing "
                    f"{format_percentage(leader_share)} of {measure.lower()} ({leader_amount})."
                )
                if shares_hold
                else (
                    f"{leader['Segment']} is the largest {dimension.lower()} by size at "
                    f"{leader_amount}. {measure} runs both positive and negative here, so no "
                    f"share of the total can be quoted -- the parts do not add up to it."
                )
            ),
            calculation=(
                f"{leader['Segment']} {measure} ÷ total {measure}"
                if shares_hold
                else f"largest |{measure}| by {dimension}; share undefined on mixed signs"
            ),
            tone="positive" if shares_hold else "warning",
        ),
        # Concentration is a statement about shares. Without shares there is
        # nothing to say, so the card is left out rather than printed as nan%.
        *(
            (
                _concentration_evidence(
                    dimension=dimension,
                    measure=measure,
                    segment_count=len(segments),
                    top_three_share=top_three_share,
                    effective=effective,
                ),
            )
            if shares_hold
            else ()
        ),
    )


def _concentration_evidence(
    *,
    dimension: str,
    measure: str,
    segment_count: int,
    top_three_share: float,
    effective: float | None,
) -> Evidence:
    headline = (
        f"The top three {dimension.lower()} values account for {format_percentage(top_three_share)} "
        f"of measured {measure.lower()}."
    )
    if effective is None:
        return Evidence(
            kind="concentration",
            title="Top-three concentration",
            value=format_percentage(top_three_share),
            statement=headline,
            calculation=f"Top three {dimension} {measure} ÷ total {measure}",
            tone="warning" if top_three_share >= 70 else "neutral",
        )

    return Evidence(
        kind="concentration",
        title="Effective segment count",
        value=f"{effective:.1f} of {segment_count}",
        statement=(
            f"{headline} Weighting every {dimension.lower()} by its share, the {segment_count} "
            f"of them carry as much risk as {effective:.1f} equally sized ones."
        ),
        calculation=(
            f"1 ÷ Herfindahl index (sum of squared {dimension} shares of {measure}), "
            "the number of equal segments that would concentrate risk the same way"
        ),
        tone="warning" if effective < CONCENTRATED_EFFECTIVE_SEGMENTS else "neutral",
    )


OFFSETTING_NET_RATIO = 0.25
MIN_CORRELATION_PAIRS = 12
MATERIAL_CORRELATION = 0.45
RANK_DIVERGENCE = 0.25
NORMAL_95 = 1.959963985


def _correlation_interval(value: float, pairs: int) -> tuple[float, float]:
    """95% confidence interval for a correlation, via the Fisher transform."""
    if pairs <= 3 or abs(value) >= 1.0:
        return value, value
    centre = np.arctanh(value)
    margin = NORMAL_95 / np.sqrt(pairs - 3)
    return float(np.tanh(centre - margin)), float(np.tanh(centre + margin))


def _relationship_evidence(dataframe: pd.DataFrame, roles: ColumnRoles) -> Evidence | None:
    usable = [
        column
        for column in roles.numeric
        if not looks_like_identifier(column, dataframe[column])
        and not any(token == normalized_name(column) for token in TIME_PART_TOKENS)
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
    if abs(value) < MATERIAL_CORRELATION:
        return None

    paired = dataframe[[pair[0], pair[1]]].dropna()
    pairs = len(paired)
    if pairs < MIN_CORRELATION_PAIRS:
        return None

    low, high = _correlation_interval(value, pairs)
    relationship = "move together" if value > 0 else "move in opposite directions"
    statement = (
        f"{pair[0]} and {pair[1]} {relationship}; their Pearson correlation is {value:.2f} "
        f"(95% CI {low:.2f} to {high:.2f}, n = {pairs:,})."
    )

    # An interval spanning zero means the sample cannot rule out no relationship.
    if low <= 0.0 <= high:
        statement += (
            f" With only {pairs:,} paired values this cannot be told apart from no "
            "relationship at all."
        )

    # Pearson measures straight lines and a handful of extreme records can
    # create one. A rank correlation that disagrees says exactly that.
    ranked = float(paired.corr(method="spearman").iloc[0, 1])
    if abs(value - ranked) > RANK_DIVERGENCE:
        statement += (
            f" The rank correlation is {ranked:.2f}, so the linear figure is being carried "
            "by a few extreme records rather than the bulk of the data."
        )

    statement += " This is an association, not proof of causation."
    return Evidence(
        kind="relationship",
        title="Strongest measurable relationship",
        value=f"r = {value:.2f}",
        statement=statement,
        calculation=(
            "Pearson correlation across non-missing paired values, with a Fisher-transform "
            "confidence interval and a Spearman rank check"
        ),
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
            f"{outlier_count:,} {roles.measure} values ({format_percentage(rate)} of non-missing records) "
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
        value=format_percentage(100 - rate),
        statement=(
            f"{missing_cells:,} cells are missing, leaving the analyzed dataset "
            f"{format_percentage(100 - rate)} complete."
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

    anomaly = by_kind.get("anomaly")
    if anomaly:
        recommendations.append(
            Recommendation(
                "Now",
                "Explain the anomalous periods",
                (
                    "Tie each flagged period to a concrete event—promotion, outage, pricing "
                    "change, or data error—before it distorts forecasts and targets."
                ),
                anomaly.statement,
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
        _anomaly_evidence(dataframe, roles),
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
        # Adding percentages up produces a number with no meaning: eleven
        # months of margin do not total 1,189% of anything.
        rate_like = is_percentage(roles.measure) and percentage_outranks_currency(roles.measure)
        if not rate_like:
            kpis.append(
                KPI(
                    f"Total {roles.measure}",
                    format_number(total, roles.measure, column_values=measure_values),
                    "Across all analyzed records",
                )
            )
        kpis.append(
            KPI(
                f"Average {roles.measure}",
                format_number(average, roles.measure, column_values=measure_values),
                "Per non-missing record",
            )
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

    # Back-fill towards four tiles from a finite list of fallbacks. The old
    # loop ran until it had four and its last branch had no already-added
    # guard, so a thin file rendered "Data completeness" three times and read
    # as a broken page. Three real tiles beats four with two copies.
    completeness = 1 - dataframe.isna().sum().sum() / max(dataframe.size, 1)
    fallbacks = []
    if roles.identifier:
        fallbacks.append(
            KPI(
                f"Distinct {roles.identifier}",
                f"{dataframe[roles.identifier].nunique(dropna=True):,}",
                "Unique entities in the dataset",
            )
        )
    fallbacks.append(KPI("Records analyzed", f"{len(dataframe):,}", "After conservative cleaning"))
    fallbacks.append(
        KPI("Data completeness", format_percentage(completeness * 100), "Share of populated cells")
    )
    for candidate in fallbacks:
        if len(kpis) >= 4:
            break
        if not any(item.label == candidate.label for item in kpis):
            kpis.append(candidate)
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
        # Cut at a sentence end, not at the decimal point inside "18.5%".
        headline = growth.statement.split(". ")[0].rstrip(".") + "."
    elif leader:
        headline = leader.statement
    elif roles.measure:
        measure_values = dataframe[roles.measure].dropna()
        total_measure = format_number(
            float(measure_values.sum()),
            roles.measure,
            column_values=measure_values,
        )
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
            "Every figure above was computed locally with pandas.",
        ]
    )
    return "\n".join(lines)
