"""Turn a dataset into an executive brief: KPIs, evidence cards and recommendations.

Every figure comes from a deterministic pandas calculation that is shown beside it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .aggregation import (
    build_trend,
    measure_aggregation,
    preferred_frequency,
    segment_frame,
    segment_period_change,
    trend_frame,
)
from .anomalies import detect_anomalies
from .formatting import (
    format_change,
    format_number,
    format_percentage,
    format_period,
    normalized_name,
    sentence_case,
)
from .metrics import increase_is_welcome, resolve_metric
from .schema import (
    TIME_PART_TOKENS,
    ColumnRoles,
    detect_roles,
    is_design_name,
    is_entity_name,
    looks_like_identifier,
)
from .timeseries import robust_scale


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


def _period_changes(values: np.ndarray, *, relative: bool = True) -> np.ndarray:
    """Period-over-period changes: percentages (skipping zero bases) or plain differences."""
    previous, current = values[:-1], values[1:]
    if not relative:
        return current - previous
    usable = previous != 0
    return (current[usable] - previous[usable]) / np.abs(previous[usable]) * 100


def _movement_in_context(values: np.ndarray, change: float, describe=format_percentage) -> str:
    """Describe the latest change relative to this series' usual volatility.

    ``change`` is a percentage when ``describe`` is ``format_percentage`` and a
    plain difference otherwise. Returns a sentence with a leading space, or ""
    when there are too few prior changes or they have no spread.
    """
    relative = describe is format_percentage
    prior = _period_changes(values[:-1], relative=relative)
    if len(prior) < MIN_PERIODS_FOR_VOLATILITY:
        return ""

    spread = robust_scale(prior)
    typical = float(np.median(np.abs(prior)))
    if spread == 0:
        return ""

    distance = abs(change - float(np.median(prior))) / spread
    # Distance from the median decides how unusual; the size of the move picks
    # the wording, so a flat period in a volatile series reads as "quiet".
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
    return f" {verdict}; this series typically moves about {describe(typical)} per period."


def _growth_evidence(dataframe: pd.DataFrame, roles: ColumnRoles) -> Evidence | None:
    series = build_trend(dataframe, roles)
    trend = series.frame
    if len(trend) < 2:
        return None

    values = trend["Value"].to_numpy(dtype=float)
    previous = float(values[-2])
    current = float(values[-1])
    unit = resolve_metric(roles.measure).unit
    measure_values = dataframe[roles.measure].dropna() if roles.measure else None
    moved = format_change(previous, current, roles.measure, unit, measure_values)
    if moved is None:
        return None
    delta = current - previous
    sign = "-" if delta < 0 else "+"
    if unit in ("rate", "level"):
        # Rates move in points and levels by a plain difference: a percentage of
        # a temperature in °C means nothing, and 3% to 4% is one point.
        value = f"{sign}{moved.split()[0]} pts" if unit == "rate" else f"{sign}{moved}"
        moved = f"by {moved}"
        context = _movement_in_context(
            values, delta, lambda size: format_change(0, size, roles.measure, unit, measure_values)
        )
        calculation = "Latest period average − previous period average"
        if unit == "rate":
            calculation += ", in percentage points"
    else:
        change = delta / abs(previous) * 100
        value = format_percentage(change, signed=True)
        context = _movement_in_context(values, change)
        calculation = "(Latest period − previous period) ÷ |previous period|"
    # Format at the series frequency so a quarter reads "Q4 2023", matching the chart axis.
    period = format_period(trend.iloc[-1]["Period"], series.frequency)
    # Without enough periods to check completeness, the last one may be partial.
    completeness = "complete period" if series.completeness_checked else "period so far"
    measure = roles.measure or "Records"
    direction = "increased" if delta >= 0 else "decreased"
    # Tone follows meaning (a rising cost is negative); recommendations follow tone.
    welcome = increase_is_welcome(roles.measure)
    if series.short_coverage:
        context += (
            f" The last period only reaches {series.short_coverage}, so part of this may be "
            "missing data rather than a real move."
        )
    return Evidence(
        kind="trend",
        title=f"Latest {sentence_case(measure)} movement",
        value=value,
        statement=(
            f"{measure} {direction} {moved} in the latest {completeness} "
            f"({period}), from {format_number(previous, roles.measure, column_values=measure_values)} to "
            f"{format_number(current, roles.measure, column_values=measure_values)}.{context}"
        ),
        calculation=f"{calculation}, compared with past period-to-period changes",
        tone="positive" if (delta >= 0) == welcome else "negative",
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

    # Index by the original key, not its str(): a boolean segment column has True/False keys.
    driver_key = changes.abs().idxmax()
    driver_name = str(driver_key)
    previous_value = float(grouped.loc[driver_key, previous_period])
    current_value = float(grouped.loc[driver_key, current_period])
    driver_change = float(grouped.loc[driver_key, "Change"])
    direction = "increased" if driver_change > 0 else "decreased"
    sign = "+" if driver_change > 0 else "-"
    unit = resolve_metric(roles.measure).unit
    if unit in ("rate", "level"):
        size = format_change(previous_value, current_value, roles.measure, unit, measure_values)
        short = f"{size.split()[0]} pts" if unit == "rate" else size
    else:
        size = short = format_number(abs(driver_change), roles.measure, column_values=measure_values)

    movement = (
        f"{driver_name} moved the most of any {sentence_case(roles.dimension)}: "
        f"{roles.measure} {direction} by {size}, "
        f"from {format_number(previous_value, roles.measure, column_values=measure_values)} to "
        f"{format_number(current_value, roles.measure, column_values=measure_values)}."
    )

    if not resolve_metric(roles.measure).additive:
        # Changes in segment averages don't add up to the change in the overall average.
        statement = movement
        calculation = (
            f"Latest {driver_name} average {roles.measure} − previous; ranked across segments"
        )
    # When segments largely cancel, a share of the small net would be a large
    # multiple, so measure against gross movement and report the offsetting.
    elif abs(net_change) < OFFSETTING_NET_RATIO * gross_change:
        share = abs(driver_change) / gross_change * 100
        statement = (
            f"{movement} Segments largely offset each other this period: "
            f"{format_number(gross_change, roles.measure, column_values=measure_values)} "
            f"of movement nets to "
            f"{format_number(abs(net_change), roles.measure, column_values=measure_values)}, so this is "
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
        value=f"{sign}{short}",
        statement=statement,
        calculation=calculation,
        tone="positive" if (driver_change > 0) == increase_is_welcome(roles.measure) else "negative",
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
            f"{label} is the sharpest anomaly: {sentence_case(measure)} reached "
            f"{format_number(worst.value, roles.measure, column_values=measure_values)}, "
            f"{worst.direction} the expected "
            f"{format_number(worst.expected_low, roles.measure, column_values=measure_values)}–"
            f"{format_number(worst.expected_high, roles.measure, column_values=measure_values)} range. "
            f"{len(anomalies)} {plural} outside the trendline band."
        ),
        calculation=(
            "Period values vs Theil–Sen trendline ± a band calibrated so that only "
            "1 stable series in 20 raises a flag"
        ),
        tone="warning",
    )


CONCENTRATED_EFFECTIVE_SEGMENTS = 3.0


def _effective_segments(values: np.ndarray, total: float) -> float | None:
    """Effective number of equally sized segments (1 / Herfindahl index).

    Ten segments where one holds 90% score just above 1. Returns None when the
    total is not positive or any segment is negative.
    """
    if total <= 0 or (values < 0).any():
        return None
    shares = values / total
    herfindahl = float(np.sum(shares**2))
    return 1.0 / herfindahl if herfindahl > 0 else None


def _shares_are_meaningful(values: np.ndarray) -> bool:
    """True when all values share a sign, so each one's share of the total is meaningful.

    All-negative counts (a cost column: -6k of -20k is 30%). With mixed signs the
    total is a net figure and one segment's share can run to thousands of percent.
    Empty input returns False.
    """
    if not values.size:
        return False
    return bool((values >= 0).all() or (values <= 0).all())


def _segment_evidence(dataframe: pd.DataFrame, roles: ColumnRoles) -> tuple[Evidence, ...]:
    # All segments, so shares and concentration are measured against the full total.
    segments = segment_frame(dataframe, roles, limit=None)
    if segments.empty:
        return ()

    # Shares also need an additive measure: segment churn rates of 10% and 20%
    # don't make one segment two thirds of churn.
    additive = resolve_metric(roles.measure).additive if roles.measure else True
    if not additive:
        segments = segments.dropna(subset=["Value"])
        if segments.empty:
            return ()
    total = float(segments["Value"].sum())
    if additive and (total == 0 or not np.isfinite(total)):
        return ()
    values = segments["Value"].to_numpy(dtype=float)
    shares_hold = additive and _shares_are_meaningful(values)
    if not additive:
        # Averages rank by value, so -20 °C is the lowest, not the largest.
        segments = segments.sort_values("Value", ascending=False).reset_index(drop=True)
    elif not shares_hold:
        # Mixed signs: no usable denominator, so rank by size and quote no percentages.
        segments = segments.reindex(
            segments["Value"].abs().sort_values(ascending=False).index
        ).reset_index(drop=True)
    elif total < 0:
        # All-negative (cost) column: the largest contributor is the most negative value.
        segments = segments.sort_values("Value").reset_index(drop=True)
    leader = segments.iloc[0]
    leader_share = float(leader["Value"] / total * 100) if shares_hold else float("nan")
    top_three_share = (
        float(segments.head(3)["Value"].sum() / total * 100) if shares_hold else float("nan")
    )
    effective = _effective_segments(segments["Value"].to_numpy(dtype=float), total)
    measure = roles.measure or "records"
    measure_values = dataframe[roles.measure].dropna() if roles.measure else None
    dimension = sentence_case(roles.dimension or "segment")

    def amount(value: float) -> str:
        return format_number(float(value), roles.measure, column_values=measure_values)

    leader_amount = amount(leader["Value"])
    if shares_hold:
        statement = (
            f"{leader['Segment']} is the largest {dimension}, with "
            f"{format_percentage(leader_share)} of {sentence_case(measure)} ({leader_amount})."
        )
        calculation = f"{leader['Segment']} {measure} ÷ total {measure}"
    elif not additive:
        statement = f"{leader['Segment']} has the highest average {measure}, at {leader_amount}."
        if len(segments) > 1:
            lowest = segments.loc[segments["Value"].idxmin()]
            statement += f" {lowest['Segment']} has the lowest, at {amount(lowest['Value'])}."
        calculation = f"mean {measure} per {roles.dimension}; averages are not split into shares"
    else:
        statement = (
            f"{leader['Segment']} is the largest {dimension} by size, at {leader_amount}. "
            f"{measure} has both positive and negative values, so shares of the total are not shown."
        )
        calculation = f"largest |{measure}| by {roles.dimension}; shares are undefined with mixed signs"
    return (
        Evidence(
            kind="leader",
            title=f"Leading {dimension}",
            value=format_percentage(leader_share) if shares_hold else leader_amount,
            statement=statement,
            calculation=calculation,
            tone="positive" if shares_hold else ("neutral" if not additive else "warning"),
            subject=str(leader["Segment"]),
        ),
        # Concentration needs shares; omit the card rather than print nan%.
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
        f"The top three {dimension} values account for {format_percentage(top_three_share)} "
        f"of {sentence_case(measure)}."
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
            f"{headline} Weighted by share, the {segment_count} {dimension} values are as "
            f"concentrated as {effective:.1f} equally sized ones would be."
        ),
        calculation=f"1 ÷ Herfindahl index (sum of squared {dimension} shares of {measure})",
        tone="warning" if effective < CONCENTRATED_EFFECTIVE_SEGMENTS else "neutral",
    )


OFFSETTING_NET_RATIO = 0.25
MIN_CORRELATION_PAIRS = 12
MATERIAL_CORRELATION = 0.45
# Pairs this close are one quantity twice (a reading and its adjusted copy, a size
# in two units), so reporting them as a relationship tells the reader nothing.
NEAR_DUPLICATE_CORRELATION = 0.99
NEAR_DUPLICATES_NAMED = 3
RANK_DIVERGENCE = 0.25
NORMAL_95 = 1.959963985


def _correlation_interval(value: float, pairs: int) -> tuple[float, float]:
    """95% confidence interval for a correlation, via the Fisher transform."""
    if pairs <= 3 or abs(value) >= 1.0:
        return value, value
    centre = np.arctanh(value)
    margin = NORMAL_95 / np.sqrt(pairs - 3)
    return float(np.tanh(centre - margin)), float(np.tanh(centre + margin))


def _rank_correlation(dataframe: pd.DataFrame, first: str, second: str) -> float:
    return float(dataframe[[first, second]].dropna().corr(method="spearman").iloc[0, 1])


def _is_near_duplicate(dataframe: pd.DataFrame, first: str, second: str, value: float) -> bool:
    """True when two columns are one quantity twice: r near ±1, with ranks that agree.

    Two extreme records can push r this high on their own; then the ranks
    disagree and the pair stays in, to be reported with that caveat.
    """
    if abs(value) < NEAR_DUPLICATE_CORRELATION:
        return False
    return abs(value) - abs(_rank_correlation(dataframe, first, second)) <= RANK_DIVERGENCE


def _near_duplicate_note(duplicates: pd.Series) -> str:
    """One sentence naming the pairs left out for moving together almost exactly."""
    ordered = duplicates.reindex(duplicates.abs().sort_values(ascending=False, kind="stable").index)
    named = [
        f"{first} and {second} (r = {value:.2f})"
        for (first, second), value in ordered.head(NEAR_DUPLICATES_NAMED).items()
    ]
    if len(ordered) > NEAR_DUPLICATES_NAMED:
        named.append(f"and {len(ordered) - NEAR_DUPLICATES_NAMED} more pairs")
    if (ordered > 0).all():
        motion = "together"
    elif (ordered < 0).all():
        motion = "in opposite directions"
    else:
        motion = "together or in opposite directions"
    return (
        f" {', '.join(named)} were left out: they move {motion} almost exactly, so they "
        "probably measure the same thing or one is derived from another."
    )


def _relationship_evidence(dataframe: pd.DataFrame, roles: ColumnRoles) -> Evidence | None:
    usable = [
        column
        for column in roles.numeric
        if not looks_like_identifier(column, dataframe[column])
        and not is_design_name(column)
        and not is_entity_name(column)
        and not any(token == normalized_name(column) for token in TIME_PART_TOKENS)
    ]
    if len(usable) < 2:
        return None
    # A pair with too few complete values gets no coefficient, so it can neither
    # win nor be called a near-duplicate on a handful of rows.
    correlations = dataframe[usable].corr(min_periods=MIN_CORRELATION_PAIRS)
    upper = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(bool))
    stacked = upper.stack().dropna()
    near_duplicate = pd.Series(
        [_is_near_duplicate(dataframe, first, second, r) for (first, second), r in stacked.items()],
        index=stacked.index,
        dtype=bool,
    )
    candidates = stacked[~near_duplicate]
    if candidates.empty:
        return None
    pair = candidates.abs().idxmax()
    value = float(correlations.loc[pair[0], pair[1]])
    if abs(value) < MATERIAL_CORRELATION:
        return None

    paired = dataframe[[pair[0], pair[1]]].dropna()
    pairs = len(paired)

    low, high = _correlation_interval(value, pairs)
    relationship = "move together" if value > 0 else "move in opposite directions"
    statement = (
        f"{pair[0]} and {pair[1]} {relationship}: Pearson r = {value:.2f} "
        f"(95% CI {low:.2f} to {high:.2f}, n = {pairs:,})."
    )

    # An interval spanning zero means the sample cannot rule out no relationship.
    if low <= 0.0 <= high:
        statement += (
            f" With only {pairs:,} paired values this cannot be told apart from no "
            "relationship at all."
        )

    # Spearman well above Pearson: monotonic but curved. Well below: a few
    # extreme records are producing the linear figure.
    ranked = float(paired.corr(method="spearman").iloc[0, 1])
    if abs(ranked) - abs(value) > RANK_DIVERGENCE:
        statement += (
            f" The rank correlation is stronger ({ranked:.2f}), so the relationship is "
            "consistent but not a straight line."
        )
    elif abs(value) - abs(ranked) > RANK_DIVERGENCE:
        statement += (
            f" The rank correlation is only {ranked:.2f}, so a few extreme records are "
            "driving the linear figure."
        )

    statement += (
        " It is the strongest of several pairs tested, so treat it as a lead to check, "
        "not as evidence of cause."
    )
    if near_duplicate.any():
        statement += _near_duplicate_note(stacked[near_duplicate])
    return Evidence(
        kind="relationship",
        title="Strongest measurable relationship",
        value=f"r = {value:.2f}",
        statement=statement,
        calculation=(
            "Pearson r over paired non-missing values, 95% CI by Fisher transform, checked "
            f"against Spearman. Strongest of all numeric pairs with {MIN_CORRELATION_PAIRS} or "
            f"more paired values, leaving out near-duplicates (|r| ≥ {NEAR_DUPLICATE_CORRELATION}); "
            "the interval is not adjusted for that selection"
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


def _shown_evidence(evidence: list[Evidence], limit: int = 6) -> list[Evidence]:
    """Return the first ``limit`` cards, swapping in a data-quality card if it was cut.

    A quality card past the limit replaces the last shown card, since it
    qualifies every other figure.
    """
    shown = list(evidence[:limit])
    quality = next((item for item in evidence[limit:] if item.kind == "quality"), None)
    if quality is not None:
        shown = shown[: limit - 1] + [quality]
    return shown


def _recommendations(evidence: list[Evidence], roles: ColumnRoles) -> tuple[Recommendation, ...]:
    recommendations: list[Recommendation] = []
    by_kind = {item.kind: item for item in evidence}
    group = sentence_case(roles.dimension) if roles.dimension else "group"

    trend = by_kind.get("trend")
    driver = by_kind.get("driver")
    # Name the driver only when it moved with the total; a segment that grew
    # while the total fell didn't cause the fall.
    if driver and trend and driver.tone != trend.tone:
        driver = None
    adverse = "decline" if increase_is_welcome(roles.measure) else "increase"
    if trend and trend.tone == "negative":
        recommendations.append(
            Recommendation(
                "Now",
                (
                    f"Start with {driver.subject}"
                    if driver and driver.subject
                    else f"Find where the {adverse} started"
                ),
                (
                    f"Compare {driver.subject}'s latest period with the one before, record by record, "
                    "to see what changed."
                    if driver and driver.subject
                    else f"Break the latest period down by {group} and compare each with its previous period."
                ),
                f"{trend.statement} {driver.statement}" if driver else trend.statement,
            )
        )
    elif trend:
        recommendations.append(
            Recommendation(
                "Now",
                (
                    f"Check whether {driver.subject}'s change holds"
                    if driver and driver.subject
                    else "Check whether the change holds"
                ),
                (
                    f"Compare {driver.subject} with earlier periods to see whether the change persists "
                    "or is a one-off."
                    if driver and driver.subject
                    else f"Find which {group} produced the latest change and whether it persists next period."
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
                "Check how much rests on the leader",
                f"Recompute the overall figures without the leading {group} to see how much they change.",
                concentration.statement,
            )
        )
    elif leader and not resolve_metric(roles.measure).additive:
        recommendations.append(
            Recommendation(
                "Next",
                f"Compare the highest and lowest {group}",
                (
                    f"Check whether the gap between {group} averages is larger than the spread "
                    f"within each {group} before reading it as a real difference."
                ),
                leader.statement,
            )
        )
    elif leader:
        recommendations.append(
            Recommendation(
                "Next",
                f"Compare the leading {group} with the rest",
                f"Look at what differs between the leading {group} and a typical one.",
                leader.statement,
            )
        )

    anomaly = by_kind.get("anomaly")
    if anomaly:
        recommendations.append(
            Recommendation(
                "Now",
                "Explain the flagged periods",
                (
                    "Match each flagged period to a known event (a change in method, an outage, "
                    "a data-entry error) before using the series for forecasts."
                ),
                anomaly.statement,
            )
        )

    relationship = by_kind.get("relationship")
    if relationship:
        recommendations.append(
            Recommendation(
                "Test",
                "Test the relationship",
                (
                    f"Check whether it holds within each {group} and time window before acting on it; "
                    "a correlation alone doesn't show cause."
                ),
                relationship.statement,
            )
        )

    outliers = by_kind.get("outliers")
    if outliers:
        recommendations.append(
            Recommendation(
                "Watch",
                "Review the extreme records",
                "Check the most extreme values for entry or measurement errors before they skew averages.",
                outliers.statement,
            )
        )

    quality = by_kind.get("quality")
    if quality and quality.tone == "warning":
        recommendations.append(
            Recommendation(
                "Fix",
                "Look into the missing values",
                (
                    "Check whether values are missing at random; gaps concentrated in one group "
                    "or period bias averages and comparisons."
                ),
                quality.statement,
            )
        )

    if not recommendations:
        recommendations.append(
            Recommendation(
                "Next",
                "Add a date or a grouping column",
                (
                    "A date column enables trends, anomalies and forecasts; a grouping column "
                    "(treatment, site, region) enables comparisons."
                ),
                "The table has no date or grouping column ADA could use.",
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
        # Only additive measures get a total; summing monthly margin % is meaningless.
        if resolve_metric(roles.measure).additive:
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
            if "÷ total" in leader.calculation:
                context = "Share contributed by the leader"
            elif leader.calculation.startswith("mean "):
                context = "Highest average of any segment"
            else:
                context = "Largest in size of any segment"
            kpis.append(KPI(f"Top {roles.dimension}", leader.value, context))

    # Pad towards four tiles from a fixed list, skipping labels already shown;
    # a thin file gets fewer tiles rather than duplicates.
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
        # "coming from" implies a share; when the leader card quotes a level
        # (non-additive or mixed-sign measure), call it the highest instead.
        quotes_a_share = "÷ total" in leader.calculation
        headline = (
            f"{growth.value} latest movement, with {leader.value} coming from the leading "
            f"{sentence_case(roles.dimension)}."
            if quotes_a_share
            else (
                f"{growth.value} latest movement; {leader.subject} has the highest average, "
                f"at {leader.value}."
                if not resolve_metric(roles.measure).additive
                else f"{growth.value} latest movement; {leader.subject} is the largest by size, "
                f"at {leader.value}."
            )
        )
    elif growth:
        # Cut at a sentence end, not at the decimal point inside "18.5%".
        headline = growth.statement.split(". ")[0].rstrip(".") + "."
    elif leader:
        headline = leader.statement
    elif roles.measure:
        measure_values = dataframe[roles.measure].dropna()
        combined = (
            measure_values.mean() if measure_aggregation(roles.measure) == "mean" else measure_values.sum()
        )
        total_measure = format_number(
            float(combined),
            roles.measure,
            column_values=measure_values,
        )
        verb = "averages" if measure_aggregation(roles.measure) == "mean" else "totals"
        headline = f"{roles.measure} {verb} {total_measure} across the analyzed data."
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
        evidence=tuple(_shown_evidence(evidence)),
        recommendations=recommendations,
    )


def build_business_report(
    dataframe: pd.DataFrame,
    brief: BusinessBrief,
    *,
    source_name: str,
    context: str = "",
) -> str:
    """Render the brief as a Markdown report with each figure's calculation beside it."""
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
