"""Turn a dataset into an executive brief: KPIs, evidence cards and recommendations.

Every figure comes from a deterministic pandas calculation that is shown beside it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .aggregation import (
    MatchedChange,
    SeasonalContext,
    TrendSeries,
    build_trend,
    group_coverage,
    matched_period_change,
    preferred_frequency,
    seasonal_context,
    segment_frame,
    segment_period_change,
    unlabelled_label,
)
from .anomalies import detect_anomalies
from .compare import (
    INFLUENCE_CHANGE,
    MAX_SCAN_LEVELS,
    NEAR_DUPLICATE_CORRELATION,
    NOTABLE_CORRELATION,
    RANK_DIVERGENCE,
    SCAN_ROWS,
    SHARED_STEM_CORRELATION,
    GroupComparison,
    Pair,
    RelationshipScan,
    combined_groupings,
    compare_groups,
    comparison_calculation,
    correlation_interval,
    describe_comparison,
    effect_clause,
    effect_text,
    entity_column,
    lead_sentence,
    most_influential,
    scan_differences,
    scan_relationships,
    variance_explained,
    within_group_correlation,
    without_extremes,
)
from .compare import (
    MIN_CORRELATION_PAIRS as MIN_CORRELATION_PAIRS,
)
from .formatting import (
    format_change,
    format_moment,
    format_number,
    format_percentage,
    format_period,
    format_values_apart,
    group_label,
    is_money,
    list_names,
    normalized_name,
    sentence_case,
    with_aggregate,
)
from .metrics import MetricSpec, metric_for
from .schema import (
    TIME_PART_TOKENS,
    ColumnRoles,
    detect_roles,
    is_design_name,
    is_entity_name,
    is_month_name,
    is_year_name,
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


def _tone(rose: bool, metric: MetricSpec) -> str:
    """Positive or negative by what the measure means: a rising cost is negative.

    Neutral for a level whose name doesn't say which way is good, so a rise in
    CO2 or rainfall is reported without being called good or bad news.
    """
    if not metric.direction_known:
        return "neutral"
    return "positive" if rose == metric.increase_is_welcome else "negative"


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


# A figure resting on fewer rows than this is quoted with its row count and a caveat.
SMALL_SIDE_ROWS = 30


def _rows(count: int) -> str:
    return f"{count:,} row{'' if count == 1 else 's'}"


def _segment_labels(dataframe: pd.DataFrame, dimension: str) -> pd.Series:
    """The grouping column with unlabelled rows under their own label, as the aggregations group it."""
    labels = dataframe[dimension]
    return labels.astype(object).where(labels.notna(), unlabelled_label(labels.dropna().unique()))


def _rows_per_period(
    dataframe: pd.DataFrame,
    roles: ColumnRoles,
    frequency: str,
    periods: tuple[pd.Timestamp, ...],
    segment: object = None,
) -> tuple[int, ...]:
    """Rows with a recorded measure in each period, within one segment when one is given."""
    kept = dataframe[roles.date].notna()
    if roles.measure:
        kept &= dataframe[roles.measure].notna()
    if segment is not None and roles.dimension:
        kept &= _segment_labels(dataframe, roles.dimension) == segment
    buckets = dataframe.loc[kept, roles.date].dt.to_period(frequency).dt.to_timestamp()
    counts = buckets.value_counts()
    return tuple(int(counts.get(period, 0)) for period in periods)


def _thin_sides_note(previous_rows: int, latest_rows: int) -> str:
    """Name the rows behind each side of a change in an average when either side has few, else ""."""
    if min(previous_rows, latest_rows) >= SMALL_SIDE_ROWS:
        return ""
    if previous_rows == latest_rows:
        sides = f" Each figure is the average of {_rows(latest_rows)}"
    else:
        sides = (
            f" The previous figure is the average of {_rows(previous_rows)} and the latest "
            f"of {_rows(latest_rows)}"
        )
    return f"{sides}, so a few records can swing it."


def _growth_evidence(
    dataframe: pd.DataFrame, roles: ColumnRoles, frequency: str | None = None
) -> Evidence | None:
    series = build_trend(dataframe, roles, frequency)
    trend = series.frame
    if len(trend) < 2:
        return None

    values = trend["Value"].to_numpy(dtype=float)
    previous = float(values[-2])
    current = float(values[-1])
    # Groups that start or stop reporting would move a pooled figure on their own,
    # so the change, and the history it is judged against, use the groups in both.
    matched = matched_period_change(dataframe, roles, series)
    pooled = (previous, current)
    history = series
    if matched is not None and matched.groups:
        previous, current = matched.previous, matched.current
        kept = dataframe[dataframe[roles.dimension].isin(matched.kept)]
        history = build_trend(kept, roles, series.frequency)
        values = history.frame["Value"].to_numpy(dtype=float)
    metric = metric_for(roles.measure, roles)
    unit = metric.unit
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
        change = delta
    else:
        change = delta / abs(previous) * 100
        value = format_percentage(change, signed=True)
        context = _movement_in_context(values, change)
        calculation = "(Latest period − previous period) ÷ |previous period|"
    compared = "compared with past period-to-period changes"
    seasonal = seasonal_context(history, relative=unit not in ("rate", "level"))
    if seasonal is not None:
        context = _seasonal_note(seasonal, history, change, current, roles, unit, measure_values)
        compared = "compared with the same move in earlier years and the same period a year earlier"
    # Format at the series frequency so a quarter reads "Q4 2023", matching the chart axis.
    period = format_period(trend.iloc[-1]["Period"], series.frequency)
    # Without enough periods to check completeness, the last one may be partial,
    # and one that falls short is not called complete.
    completeness = "complete period" if series.completeness_checked else "period so far"
    if series.short_coverage:
        completeness = "period"
    measure = roles.measure or "Records"
    direction = "increased" if delta >= 0 else "decreased"
    start, end = format_values_apart(
        previous, current, column=roles.measure, column_values=measure_values
    )
    if roles.measure and not metric.additive:
        # A total is everything recorded, but an average of a few rows is noisy.
        periods = (trend.iloc[-2]["Period"], trend.iloc[-1]["Period"])
        context = _thin_sides_note(*_rows_per_period(dataframe, roles, series.frequency, periods)) + context
    if series.short_coverage:
        context += (
            f" The last period only reaches {series.short_coverage}, so part of this may be "
            "missing data rather than a real move."
        )
    if matched is not None:
        context = _matched_note(matched, roles, pooled, series, measure_values) + context
        if matched.groups:
            calculation = f"On the {roles.dimension} values present in both periods: {calculation}"
    return Evidence(
        kind="trend",
        title=f"Latest {sentence_case(measure)} movement",
        value=value,
        statement=(
            f"{measure} {direction} {moved} in the latest {completeness} "
            f"({period}), from {start} to {end}.{context}"
        ),
        calculation=f"{calculation}, {compared}",
        # Recommendations follow this tone.
        tone=_tone(delta >= 0, metric),
    )


GROUPS_NAMED = 3


def _listed(groups: tuple, column: str | None) -> str:
    """Name up to GROUPS_NAMED groups ("East and West"), then count the rest."""
    names = [group_label(column, group) for group in groups[:GROUPS_NAMED]]
    if len(groups) > GROUPS_NAMED:
        names.append(f"{len(groups) - GROUPS_NAMED:,} more")
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _matched_note(
    matched: MatchedChange,
    roles: ColumnRoles,
    pooled: tuple[float, float],
    series: TrendSeries,
    measure_values: pd.Series | None,
) -> str:
    """Say which groups the latest change was measured on, and what all of them showed."""
    dimension = sentence_case(roles.dimension or "group")
    # "in Q3 1945", "in the week of 20 Jan 2025", but "at 15:55" for a clock time.
    when = {"s": "at", "min": "at", "h": "at", "W": "in the week of"}.get(series.frequency, "in")
    previous, latest = (
        format_period(series.frame.iloc[position]["Period"], series.frequency) for position in (-2, -1)
    )
    changes = []
    if matched.stopped:
        stopped = _listed(matched.stopped, roles.dimension)
        changes.append(f"{stopped} reported {when} {previous} but not {when} {latest}")
    if matched.started:
        started = _listed(matched.started, roles.dimension)
        changes.append(f"{started} reported {when} {latest} but not {when} {previous}")
    if not matched.groups:
        return (
            f" No {dimension} value reported in both periods ({'; '.join(changes)}), so this "
            "compares different groups."
        )
    start, end = format_values_apart(*pooled, column=roles.measure, column_values=measure_values)
    plural = "values" if matched.groups > 1 else "value"
    return (
        f" This compares the {matched.groups:,} {dimension} {plural} that reported in both periods; "
        f"{'; '.join(changes)}. With every {dimension} included it went from {start} to {end}."
    )


def _season_name(period: pd.Timestamp, grain: str) -> str:
    return f"Q{period.quarter}" if grain == "Q" else period.strftime("%b")


def _seasonal_note(
    seasonal: SeasonalContext,
    series: TrendSeries,
    change: float,
    current: float,
    roles: ColumnRoles,
    unit: str,
    measure_values: pd.Series | None,
) -> str:
    """Set the latest move beside the same move in earlier years and the same period a year ago.

    Replaces the comparison with all past period-to-period changes, which a
    yearly cycle makes look unusual every spring.
    """
    relative = unit not in ("rate", "level")

    def describe(size: float) -> str:
        if relative:
            return format_percentage(size, signed=True)
        return format_change(0, size, roles.measure, unit, measure_values, signed=True) or "—"

    latest = pd.Timestamp(series.frame.iloc[-1]["Period"])
    season = _season_name(latest, series.frequency)
    note = ""
    usual = np.asarray(seasonal.usual_changes, dtype=float)
    if usual.size == 1:
        note = f" In the one earlier year on record, the move into {season} was {describe(usual[0])}."
    elif usual.size:
        low, high = float(usual.min()), float(usual.max())
        within = "within" if low <= change <= high else "outside"
        note = (
            f" In the {usual.size} earlier years on record, the move into {season} ranged from "
            f"{describe(low)} to {describe(high)} (median {describe(float(np.median(usual)))}), "
            f"so this one is {within} the range usual for the time of year."
        )
    if seasonal.year_ago is not None:
        then = format_period(seasonal.year_ago, series.frequency)
        before = seasonal.year_ago_value
        size = format_change(before, current, roles.measure, unit, measure_values)
        shown = format_values_apart(before, current, column=roles.measure, column_values=measure_values)[0]
        if size is not None:
            verb = "up" if current > before else "down" if current < before else "level"
            note += f" Against {then} ({shown}), it is {verb}" + (f" {size}." if verb != "level" else ".")
    return note


def _change_driver_evidence(
    dataframe: pd.DataFrame, roles: ColumnRoles, frequency: str | None = None
) -> Evidence | None:
    """Identify the segment contributing most to the latest net movement."""
    result = segment_period_change(dataframe, roles, frequency)
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
    driver_name = group_label(roles.dimension, driver_key)
    previous_value = float(grouped.loc[driver_key, previous_period])
    current_value = float(grouped.loc[driver_key, current_period])
    driver_change = float(grouped.loc[driver_key, "Change"])
    direction = "increased" if driver_change > 0 else "decreased"
    sign = "+" if driver_change > 0 else "-"
    metric = metric_for(roles.measure, roles)
    unit = metric.unit
    if unit in ("rate", "level"):
        size = format_change(previous_value, current_value, roles.measure, unit, measure_values)
        short = f"{size.split()[0]} pts" if unit == "rate" else size
    else:
        size = short = format_number(abs(driver_change), roles.measure, column_values=measure_values)

    start, end = format_values_apart(
        previous_value, current_value, column=roles.measure, column_values=measure_values
    )
    thin = ""
    if not metric.additive:
        grain = frequency or preferred_frequency(dataframe[roles.date])
        thin = _thin_sides_note(
            *_rows_per_period(dataframe, roles, grain, (previous_period, current_period), segment=driver_key)
        )
    movement = (
        f"{driver_name} moved the most of any {sentence_case(roles.dimension)}: "
        f"{roles.measure} {direction} by {size}, from {start} to {end}."
    )

    if len(changes) == 1:
        # One segment is all of its own movement; a share would say nothing.
        dimension = sentence_case(roles.dimension)
        statement = f"{movement} It is the only {dimension} with figures in both periods."
        calculation = f"Latest {driver_name} {roles.measure} − previous"
    elif not metric.additive:
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
    left_out = grouped.attrs.get("left_out", ())
    if left_out:
        # Their change is reporting starting or stopping, not a move in the measure.
        noun, verb = ("value", "is") if len(left_out) == 1 else ("values", "are")
        statement += (
            f" {len(left_out):,} {sentence_case(roles.dimension)} {noun} with figures in only one "
            f"of the two periods ({_listed(left_out, roles.dimension)}) {verb} left out."
        )
        calculation += "; only segments with figures in both periods"

    return Evidence(
        kind="driver",
        title=f"Largest change by {sentence_case(roles.dimension)}",
        value=f"{sign}{short}",
        statement=statement + thin,
        calculation=calculation,
        tone=_tone(driver_change > 0, metric),
        subject=driver_name,
    )


def _anomaly_evidence(
    dataframe: pd.DataFrame, roles: ColumnRoles, frequency: str | None = None
) -> Evidence | None:
    if not roles.date:
        return None
    series = build_trend(dataframe, roles, frequency)
    metric = metric_for(roles.measure, roles)
    anomalies = detect_anomalies(series.frame, floor_at_zero=metric.floors_at_zero)
    if not anomalies:
        return None
    # Label at the series' own grain, which a fixed frequency may have set.
    grain = series.frequency
    worst = anomalies[0]
    measure = roles.measure or "Records"
    measure_values = dataframe[roles.measure].dropna() if roles.measure else None
    label = format_period(worst.period, grain)
    plural = "periods sit" if len(anomalies) > 1 else "period sits"
    # The value sits just outside one end of the range, so all three are kept apart.
    reached, low, high = format_values_apart(
        worst.value, worst.expected_low, worst.expected_high,
        column=roles.measure, column_values=measure_values,
    )
    band = "trendline band"
    expected = "Theil–Sen trendline"
    if worst.seasonal:
        band = "band for its time of year"
        expected += " plus the usual offset of the same month or quarter in other years"
    return Evidence(
        kind="anomaly",
        title="Anomalous periods",
        value=f"{len(anomalies)}",
        statement=(
            f"{label} is the sharpest anomaly: {sentence_case(measure)} reached {reached}, "
            f"{worst.direction} the expected {low}–{high} range. "
            f"{len(anomalies)} {plural} outside the {band}."
        ),
        calculation=(
            f"Period values vs {expected} ± a band calibrated so that only "
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


# Group figures this close, relative to their size, are one value computed two
# ways, so the groups tie.
TIE_TOLERANCE = 1e-9


def _counts_money_or_things(
    roles: ColumnRoles, metric: MetricSpec, measure_values: pd.Series | None
) -> bool:
    """Whether the measure is money or a count of things, the quantities concentration describes.

    How concentrated revenue or orders are across products is a real question.
    How concentrated rainfall is across weather stations, or a reading across
    sensors, is not, even when the readings can be summed.
    """
    if not roles.measure or metric.unit != "amount" or measure_values is None:
        return False
    values = measure_values.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    whole = bool(values.size) and bool((values == np.floor(values)).all())
    return whole or is_money(roles.measure, measure_values)


def _tied_with(segments: pd.DataFrame, value: float) -> pd.DataFrame:
    """The segments whose value equals ``value``, allowing for rounding in how each was computed."""
    return segments[np.isclose(segments["Value"].astype(float), value, rtol=TIE_TOLERANCE, atol=0.0)]


def _group_names(segments: pd.DataFrame, dimension: str | None) -> list[str]:
    return [group_label(dimension, value) for value in segments["Segment"]]


def _who(names: list[str]) -> str:
    """The subject and verb of a ranking: "A has" for one group, "A and B tie for" for several."""
    return f"{list_names(names)} {'has' if len(names) == 1 else 'tie for'}"


def _subject(names: list[str]) -> str:
    """One group's name for a KPI or a recommendation, saying when others tie with it."""
    others = len(names) - 1
    return names[0] if not others else f"{names[0]} (tied with {others} other{'s' if others != 1 else ''})"


def _rows_per_group(dataframe: pd.DataFrame, roles: ColumnRoles) -> pd.Series:
    """Rows with a recorded measure in each group, indexed like ``segment_frame``'s segments."""
    labels = _segment_labels(dataframe, roles.dimension)
    if roles.measure:
        return dataframe[roles.measure].notna().groupby(labels, sort=False).sum()
    return labels.value_counts(sort=False)


def _rows_label(counts: list[int]) -> str:
    """" (4 rows)" for one group, " (4 rows each)" for several alike, " (4 to 9 rows)" otherwise."""
    if len(counts) == 1:
        return f" ({_rows(counts[0])})"
    if min(counts) == max(counts):
        return f" ({_rows(counts[0])} each)"
    return f" ({min(counts):,} to {max(counts):,} rows)"


def _highest_and_lowest(
    segments: pd.DataFrame,
    roles: ColumnRoles,
    rows: pd.Series,
    measure_values: pd.Series | None,
) -> tuple[str, str, str]:
    """The statement, value and subject naming the groups with the highest and lowest average.

    Ties are named. The two figures are kept apart, so different averages
    never print alike, and when a named group rests on few rows every named
    group is quoted with its row count.
    """
    measure, dimension = roles.measure, sentence_case(roles.dimension)
    values = segments["Value"].astype(float)
    highest, lowest = float(values.iloc[0]), float(values.min())
    top, bottom = _tied_with(segments, highest), _tied_with(segments, lowest)
    high, low = format_values_apart(highest, lowest, column=measure, column_values=measure_values)
    top_names = _group_names(top, roles.dimension)
    if len(segments) > 1 and len(top) == len(segments):
        statement = f"All {len(segments)} {dimension} groups have the same average {measure}, {high}."
        return statement, high, _subject(top_names)

    def counts(groups: pd.DataFrame) -> list[int]:
        return [int(rows.get(segment, 0)) for segment in groups["Segment"]]

    thin = min(counts(top) + counts(bottom)) < SMALL_SIDE_ROWS
    top_rows = _rows_label(counts(top)) if thin else ""
    statement = f"{_who(top_names)} the highest average {measure}, at {high}{top_rows}."
    if len(segments) > 1:
        bottom_rows = _rows_label(counts(bottom)) if thin else ""
        statement += f" {_who(_group_names(bottom, roles.dimension))} the lowest, at {low}{bottom_rows}."
    if thin:
        statement += (
            f" With fewer than {SMALL_SIDE_ROWS} rows behind a group's average, a few records "
            "can change the order."
        )
    return statement, high, _subject(top_names)


def _segment_evidence(dataframe: pd.DataFrame, roles: ColumnRoles) -> tuple[Evidence, ...]:
    # All segments, so shares and concentration are measured against the full total.
    segments = segment_frame(dataframe, roles, limit=None)
    if segments.empty:
        return ()

    # Shares also need an additive measure: segment churn rates of 10% and 20%
    # don't make one segment two thirds of churn.
    metric = metric_for(roles.measure, roles)
    additive = metric.additive
    if not additive:
        compared = _group_difference_evidence(dataframe, roles)
        if compared is not None:
            return (compared,)
    # A segment with no values for the measure has no figure, not a figure of zero.
    no_data = tuple(segments.loc[segments["Value"].isna(), "Segment"])
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
    measure = roles.measure or "rows"
    measure_values = dataframe[roles.measure].dropna() if roles.measure else None
    dimension = sentence_case(roles.dimension or "segment")
    leader_name = group_label(roles.dimension, leader["Segment"])

    def amount(value: float) -> str:
        return format_number(float(value), roles.measure, column_values=measure_values)

    leader_amount = amount(leader["Value"])
    value = format_percentage(leader_share) if shares_hold else leader_amount
    subject = leader_name
    if shares_hold:
        names = _group_names(_tied_with(segments, float(leader["Value"])), roles.dimension)
        subject = _subject(names)
        share = format_percentage(leader_share)
        each = " each" if len(names) > 1 else ""
        title = f"Largest share by {dimension}" if roles.measure else f"Most rows by {dimension}"
        if len(segments) > 1 and len(names) == len(segments):
            statement = (
                f"All {len(segments)} {dimension} groups have the same {sentence_case(measure)}: "
                f"{share} each."
            )
        elif roles.measure:
            statement = (
                f"{_who(names)} the largest share of {sentence_case(measure)}: {share}{each} "
                f"({leader_amount}{each})."
            )
        else:
            statement = (
                f"{_who(names)} the most rows: {int(leader['Value']):,}{each} of {int(total):,} "
                f"({share}{each})."
            )
        calculation = f"{leader_name} {measure} ÷ total {measure}"
    elif not additive:
        title = f"Highest average by {dimension}"
        statement, value, subject = _highest_and_lowest(
            segments, roles, _rows_per_group(dataframe, roles), measure_values
        )
        ranked = [leader["Segment"]]
        if len(segments) > 1:
            ranked.append(segments.loc[segments["Value"].idxmin(), "Segment"])
            statement += (
                " Too few groups have two or more values to test whether the gap is larger "
                "than the noise."
            )
        statement += _coverage_note(dataframe, roles, ranked)
        calculation = f"mean {measure} per {roles.dimension}; averages are not split into shares"
    else:
        title = f"Largest {dimension} by size"
        statement = (
            f"{leader_name} is the largest {dimension} by size, at {leader_amount}. "
            f"{measure} has both positive and negative values, so shares of the total are not shown."
        )
        calculation = f"largest |{measure}| by {roles.dimension}; shares are undefined with mixed signs"
    if additive:
        statement += _coverage_note(dataframe, roles, [leader["Segment"]])
    if no_data:
        verb, pronoun = ("has", "it is") if len(no_data) == 1 else ("have", "they are")
        statement += (
            f" {_listed(no_data, roles.dimension)} {verb} no {measure} values, so {pronoun} "
            "left out rather than counted as zero."
        )
    return (
        Evidence(
            kind="leader",
            title=title,
            value=value,
            statement=statement,
            calculation=calculation,
            tone="positive" if shares_hold else ("neutral" if not additive else "warning"),
            subject=subject,
        ),
        # Concentration needs shares, and describes only money and counts of
        # things; omit the card rather than print nan%.
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
            if shares_hold and _counts_money_or_things(roles, metric, measure_values)
            else ()
        ),
    )


# A ranked group whose figures span less than this share of the table's dates is
# named with its own dates, since its average covers a different stretch of time.
PARTIAL_COVERAGE_SHARE = 0.75


def _coverage_note(dataframe: pd.DataFrame, roles: ColumnRoles, groups: list) -> str:
    """Name the ranked groups whose figures cover only part of the table's dates."""
    coverage = group_coverage(dataframe, roles)
    if coverage.empty:
        return ""
    start, end = coverage["From"].min(), coverage["To"].max()
    span = end - start
    if span <= pd.Timedelta(0):
        return ""
    notes = []
    for group in groups:
        if group not in coverage.index:
            continue
        first, last = coverage.loc[group, "From"], coverage.loc[group, "To"]
        if (last - first) < PARTIAL_COVERAGE_SHARE * span:
            dates = (
                f"on {format_moment(first)}"
                if format_moment(first) == format_moment(last)
                else f"from {format_moment(first)} to {format_moment(last)}"
            )
            notes.append(f"{group_label(roles.dimension, group)} only has figures {dates}")
    if not notes:
        return ""
    covers = "its figure covers" if len(notes) == 1 else "their figures cover"
    return (
        f" {'; '.join(notes)}, while the table runs from {format_moment(start)} to "
        f"{format_moment(end)}, so {covers} a different stretch of time."
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
        title=f"Concentration by {dimension}",
        value=f"{effective:.1f} of {segment_count}",
        statement=(
            f"{headline} Weighted by share, the {segment_count} {dimension} values are as "
            f"concentrated as {effective:.1f} equally sized ones would be."
        ),
        calculation=f"1 ÷ Herfindahl index (sum of squared {dimension} shares of {measure})",
        tone="warning" if effective < CONCENTRATED_EFFECTIVE_SEGMENTS else "neutral",
    )


def _leading_groups(comparison: GroupComparison) -> list[str]:
    """The highest group and any tied with it, highest first."""
    top = comparison.centre(comparison.higher)
    return [group.label for group in comparison.groups if np.isclose(comparison.centre(group), top)]


def _group_difference_evidence(dataframe: pd.DataFrame, roles: ColumnRoles) -> Evidence | None:
    """The averaged measure compared across the grouping, with its test and effect size.

    A gap the test can't tell from noise is reported as no clear difference,
    so the card never ranks groups that don't differ. None when no two groups
    have two or more values.
    """
    if not roles.measure or not roles.dimension:
        return None
    comparison = compare_groups(
        dataframe, roles.measure, roles.dimension, entity=entity_column(dataframe, roles)
    )
    if comparison is None:
        return None
    values = dataframe[roles.measure].dropna()
    dimension = sentence_case(roles.dimension)
    clear = comparison.verdict == "clear"
    calculation = comparison_calculation(comparison)
    centre = "median" if calculation.startswith("median ") else "average"
    return Evidence(
        kind="leader",
        title=f"Highest {centre} by {dimension}" if clear else f"No clear difference by {dimension}",
        value=(
            format_number(comparison.centre(comparison.higher), roles.measure, column_values=values)
            if clear
            else effect_text(comparison).replace("Hedges' ", "")
        ),
        # A group that reported for only part of the dates is said to have done so.
        statement=describe_comparison(comparison, values)
        + _coverage_note(dataframe, roles, [comparison.higher.value, comparison.lower.value]),
        calculation=calculation,
        subject=_subject(_leading_groups(comparison)) if clear else None,
    )


DIFFERENCES_NAMED = 3
# A scan with nothing clear is still reported once it covers this many comparisons.
MIN_NULL_COMPARISONS = 3


def _same_partition(dataframe: pd.DataFrame, first: str, second: str) -> bool:
    """Whether two columns split the rows into the same groups, as a code and its label do."""
    data = dataframe[[first, second]].dropna()
    return bool(
        not data.empty
        and data.groupby(first, observed=True)[second].nunique().max() == 1
        and data.groupby(second, observed=True)[first].nunique().max() == 1
    )


def _comparison_groupings(dataframe: pd.DataFrame, roles: ColumnRoles, entity: str | None) -> list[str]:
    """The grouping column, then other groupings with 2 to MAX_SCAN_LEVELS levels, each split once.

    The column naming each row's subject is left out: readings differ between
    subjects as a matter of course.
    """
    candidates = [roles.dimension] if roles.dimension else []
    candidates += [
        column
        for column in (*roles.dimensions, *roles.factors)
        if column in dataframe.columns and 2 <= dataframe[column].nunique() <= MAX_SCAN_LEVELS
    ]
    chosen: list[str] = []
    for column in dict.fromkeys(candidates):
        if column in (entity, roles.identifier, roles.measure) or column not in dataframe.columns:
            continue
        if dataframe[column].nunique() < 2:
            continue
        if any(_same_partition(dataframe, column, other) for other in chosen):
            continue
        chosen.append(column)
    return chosen


def _count(number: int, noun: str) -> str:
    """A count with its noun, singular for one: "1 pair", "465 pairs"."""
    return f"{number:,} {noun}{'' if number == 1 else 's'}"


def _variance_shares(measure: str, shares: tuple[tuple[str, float], ...], combined: dict) -> str:
    """One sentence on how much of the measure's variance each grouping explains on its own."""
    plain = [(grouping, share) for grouping, share in shares if grouping not in combined]
    text = f"Of the variance in {measure}, " + ", ".join(
        f"{grouping} explains {share:.0%}" if index == 0 else f"{grouping} {share:.0%}"
        for index, (grouping, share) in enumerate(plain)
    )
    for grouping, share in shares:
        if grouping in combined:
            first, second = combined[grouping]
            text += f"; {grouping}, which combines {first} and {second}, explains {share:.0%}"
    return text + "."


def _differences_evidence(
    dataframe: pd.DataFrame, roles: ColumnRoles, readings: list[str] | None = None
) -> Evidence | None:
    """Which readings differ between groups, largest standardised difference first.

    Every numeric reading is compared across the grouping column and the other
    groupings with few levels, with p-values adjusted for the number of tests.
    A table with several groupings also gets the share of the measure's
    variance each one explains.
    """
    entity = entity_column(dataframe, roles)
    groupings = _comparison_groupings(dataframe, roles, entity)
    if not groupings:
        return None
    readings = numeric_readings(dataframe, roles) if readings is None else readings
    if not readings:
        return None
    # A cell code made of two other groupings explains as much as both together,
    # which says nothing about which of them matters.
    combined = {
        grouping: parts
        for grouping, parts in combined_groupings(dataframe, groupings).items()
        if all(part in groupings for part in parts)
    }
    scanned = [grouping for grouping in groupings if grouping not in combined]
    # The leading-group card already compares an averaged measure across the grouping.
    covered = (roles.measure, roles.dimension) if not metric_for(roles.measure, roles).additive else None
    scan = scan_differences(dataframe, readings, scanned, entity=entity, skip=(covered,) if covered else ())
    # A flag that only marks where the measure hit its cap explains it by definition.
    flags = {grouping for reading, grouping in (scan.flagged if scan else ()) if reading == roles.measure}
    sharing = [grouping for grouping in groupings if grouping not in flags]
    shares = (
        variance_explained(dataframe, roles.measure, sharing)
        if roles.measure in readings and len(sharing) > 1
        else ()
    )
    if scan is None or not (scan.named or shares or scan.tested >= MIN_NULL_COMPARISONS):
        return None

    where = list_names(list(scan.groupings), limit=len(scan.groupings))
    comparisons = _count(scan.tested, "comparison")
    parts = []
    if scan.named:
        lead = scan.named[0]
        parts.append(lead_sentence(lead, dataframe[lead.measure].dropna()))
        if scan.tested == 1:
            parts.append("It is the only comparison made.")
        elif scan.clear == 1:
            parts.append(
                f"It is the only clear difference among {comparisons} across {where}, "
                f"after a Holm adjustment for all {scan.tested}."
            )
        else:
            parts.append(
                f"It is the largest of {scan.clear} clear differences among {comparisons} "
                f"across {where}, after a Holm adjustment for all {scan.tested}."
            )
        if lead.shared_entities:
            parts.append(
                f"The same {lead.entity} appears in more than one {lead.grouping} group, so those "
                "groups are not independent and the p-value is only a guide."
            )
        if len(scan.named) > 1:
            parts.append(
                "Next: "
                + "; ".join(
                    f"{item.measure} by {item.grouping} ({effect_clause(item)})" for item in scan.named[1:]
                )
                + "."
            )
    elif scan.largest is not None:
        largest = scan.largest
        # A passing test on a trivial gap would otherwise read as a contradiction.
        trivial = ": detectable, but too small to matter" if largest.verdict == "negligible" else ""
        none_of = "The one comparison" if scan.tested == 1 else f"None of the {comparisons}"
        parts.append(
            f"{none_of} across {where} shows a clear difference once all tests are allowed for; "
            f"the largest is {largest.measure} by {largest.grouping} ({effect_clause(largest)}{trivial})."
        )
    if shares:
        parts.append(_variance_shares(roles.measure, shares, combined))
    if scan.entity:
        parts.append(f"Each {scan.entity} counts once per group.")

    calculation = (
        f"each numeric reading compared across {where}: Welch's t-test or ANOVA on means, or "
        "Mann-Whitney or Kruskal-Wallis on ranks for a skewed or capped reading; p-values "
        f"Holm-adjusted for {_count(scan.tested, 'test')}; ranked by share of variance explained "
        "(η², or ε² on ranks)"
    )
    if scan.entity:
        calculation += f"; one value per {scan.entity} in each group before testing"
    if scan.sampled:
        calculation += f"; on a fixed random sample of {SCAN_ROWS:,} of the {scan.rows:,} rows"
    if len(scan.readings) < scan.available:
        calculation += f"; the {len(scan.readings)} most variable of {scan.available:,} readings"
    if covered and covered[0] in readings and covered[1] in scan.groupings:
        calculation += f"; {covered[0]} by {covered[1]} is compared on its own card"
    for reading, grouping in scan.flagged:
        calculation += (
            f"; {reading} by {grouping} left out: {grouping} only marks where {reading} hits its cap"
        )
    if scan.recoded:
        named = ", ".join(
            f"{reading} by {grouping}" for reading, grouping in scan.recoded[:DIFFERENCES_NAMED]
        )
        extra = len(scan.recoded) - DIFFERENCES_NAMED
        more = f" and {extra} more" if extra > 0 else ""
        calculation += f"; left out as codes for the grouping, with one value per group: {named}{more}"
    for grouping, (first, second) in combined.items():
        calculation += f"; {grouping} combines {first} and {second}, so those two are compared instead"
    if shares:
        calculation += f"; variance shares are η² of {roles.measure} for each grouping alone"
    return Evidence(
        kind="differences",
        title="What differs between groups",
        value=f"{scan.clear} of {scan.tested}",
        statement=" ".join(parts),
        calculation=calculation,
        subject=scan.named[0].measure if scan.named else None,
    )


OFFSETTING_NET_RATIO = 0.25
MATERIAL_CORRELATION = 0.45
NEAR_DUPLICATES_NAMED = 3
# Correlates of the measure named in the relationship card, and derived columns listed in its calculation.
CORRELATES_NAMED = 3
DERIVED_NAMED = 3
# A table whose strongest correlation is below this, over many rows, has columns
# that ignore one another, which measured quantities rarely do.
UNRELATED_CORRELATION = 0.05
UNRELATED_MIN_ROWS = 1_000
UNRELATED_MIN_PAIRS = 10
# A within-group r this far from the pooled one changes what the pooled figure means.
WITHIN_GROUP_GAP = 0.3


def _near_duplicate_note(duplicates: list[tuple[str, str, float]]) -> str:
    """One sentence naming the pairs left out for moving together almost exactly."""
    named = [
        f"{first} and {second} (r = {value:.2f})"
        for first, second, value in duplicates[:NEAR_DUPLICATES_NAMED]
    ]
    if len(duplicates) > NEAR_DUPLICATES_NAMED:
        named.append(f"and {len(duplicates) - NEAR_DUPLICATES_NAMED} more pairs")
    signs = {value > 0 for _, _, value in duplicates}
    motion = (
        "together or in opposite directions"
        if len(signs) > 1
        else ("together" if True in signs else "in opposite directions")
    )
    return (
        f" {', '.join(named)} were left out: they move {motion} almost exactly, so they "
        "probably measure the same thing or one is derived from another."
    )


@dataclass(frozen=True)
class Correlation:
    """Pearson r over the complete pairs of two columns, with its interval, Spearman's r and checks.

    The checks say whether one record, or a few extreme ones, decide the
    figure: r without the record whose removal moves it most (and that
    record's two values), and r without records far outside either column.
    """

    first: str
    second: str
    pearson: float
    low: float
    high: float
    pairs: int
    spearman: float
    without_influential: float | None = None
    influential: tuple[float, float] | None = None
    without_extremes: float | None = None
    extremes: int = 0

    @property
    def rank_divergence(self) -> float:
        """How far the rank correlation is from the linear one, in absolute strength."""
        return abs(self.spearman) - abs(self.pearson)

    @property
    def rests_on_one_record(self) -> bool:
        return (
            self.without_influential is not None
            and abs(self.pearson - self.without_influential) > INFLUENCE_CHANGE
        )


def measure_correlation(paired: pd.DataFrame) -> Correlation | None:
    """Correlate the two columns of ``paired``, which holds only complete rows.

    Returns None when r is undefined, as it is when either column does not vary.
    """
    value = float(paired.corr().iloc[0, 1])
    if not np.isfinite(value):
        return None
    low, high = correlation_interval(value, len(paired))
    ranked = float(paired.corr(method="spearman").iloc[0, 1])
    first, second = (str(column) for column in paired.columns[:2])
    x = paired.iloc[:, 0].to_numpy(dtype=float)
    y = paired.iloc[:, 1].to_numpy(dtype=float)
    influence = most_influential(x, y)
    trimmed = without_extremes(x, y)
    return Correlation(
        first,
        second,
        value,
        low,
        high,
        len(paired),
        ranked,
        without_influential=influence[1] if influence else None,
        influential=(float(x[influence[0]]), float(y[influence[0]])) if influence else None,
        without_extremes=trimmed[0] if trimmed else None,
        extremes=trimmed[1] if trimmed else 0,
    )


def describe_correlation(correlation: Correlation, scope: str = "") -> str:
    """State r with its interval, adding why the sample, one record or a few extremes may undercut it.

    The brief and Ask both use this, so a pair reads the same in either. ``scope``
    follows the column names, as in " for Site in (North)".
    """
    value, pairs = correlation.pearson, correlation.pairs
    if round(value, 2) == 0:
        relationship = "show no linear relationship"
    else:
        relationship = "move together" if value > 0 else "move in opposite directions"
        if abs(value) < MATERIAL_CORRELATION:
            relationship += " only weakly"
    statement = (
        f"{correlation.first} and {correlation.second} {relationship}{scope}: "
        f"Pearson r = {value:.2f} (95% CI {correlation.low:.2f} to {correlation.high:.2f}, "
        f"n = {pairs:,})."
    )

    # An interval spanning zero means the sample cannot rule out no relationship.
    # A strong r that still spans zero is a thin sample; a weak one is just weak.
    if correlation.low <= 0.0 <= correlation.high:
        if abs(value) >= MATERIAL_CORRELATION:
            statement += (
                f" With only {pairs:,} paired values this cannot be told apart from no "
                "relationship at all."
            )
        else:
            statement += (
                " The interval includes zero, so this cannot be told apart from no "
                "relationship at all."
            )

    # One record that moves r this far decides the figure on its own, whichever
    # way it moves it; that says more than any comparison with the ranks.
    trimmed = correlation.without_extremes
    if correlation.rests_on_one_record and correlation.influential is not None:
        first_value, second_value = correlation.influential
        statement += (
            f" It rests on one record: without the one where {correlation.first} is "
            f"{format_number(first_value, correlation.first, compact=False)} and {correlation.second} is "
            f"{format_number(second_value, correlation.second, compact=False)}, r is "
            f"{correlation.without_influential:.2f}."
        )
    # Spearman well above Pearson: extreme records hiding a straight line, or a
    # curve. Well below: extreme records producing the linear figure.
    elif correlation.rank_divergence > RANK_DIVERGENCE:
        if trimmed is not None and abs(trimmed) - abs(value) > RANK_DIVERGENCE:
            statement += (
                f" The rank correlation is stronger ({correlation.spearman:.2f}) because "
                f"{correlation.extremes} extreme records hide the relationship: without them "
                f"r is {trimmed:.2f}."
            )
        else:
            statement += (
                f" The rank correlation is stronger ({correlation.spearman:.2f}), so the relationship is "
                "consistent but not a straight line."
            )
    elif correlation.rank_divergence < -RANK_DIVERGENCE:
        statement += (
            f" The rank correlation is only {correlation.spearman:.2f}, so a few extreme records are "
            "driving the linear figure"
        )
        if trimmed is not None and abs(value) - abs(trimmed) > RANK_DIVERGENCE:
            statement += f": without the {correlation.extremes} most extreme, r is {trimmed:.2f}"
        statement += "."
    return statement


def _codes_a_grouping(dataframe: pd.DataFrame, column: str, groupings: list[str]) -> bool:
    """Whether a numeric column numbers the groups of a grouping column, as "Group" 1-12 does "Gp"."""
    values = dataframe[column]
    for grouping in groupings:
        levels = dataframe[grouping].nunique()
        # Most columns have more values in their first rows than the grouping has levels.
        if values.iloc[:2_000].nunique() <= levels and values.nunique() == levels:
            if _same_partition(dataframe, column, grouping):
                return True
    return False


def _is_flag(values: pd.Series) -> bool:
    """Whether a numeric column holds two values, a yes/no flag that groups rows rather than measures them."""
    return values.iloc[:2_000].nunique() <= 2 and values.nunique() <= 2


def numeric_readings(dataframe: pd.DataFrame, roles: ColumnRoles) -> list[str]:
    """Numeric columns that hold readings, not labels: no ids, codes, flags, design labels or time parts."""
    groupings = {*roles.dimensions, *roles.factors, roles.dimension}
    text_groupings = [
        column for column in roles.dimensions if column in dataframe.columns and column not in roles.factors
    ]
    return [
        column
        for column in roles.numeric
        if column in dataframe.columns
        and column not in groupings
        and not looks_like_identifier(column, dataframe[column])
        and not is_design_name(column)
        and not is_entity_name(column)
        and not is_year_name(column)
        and not is_month_name(column)
        and not any(token == normalized_name(column) for token in TIME_PART_TOKENS)
        and not _is_flag(dataframe[column])
        and not _codes_a_grouping(dataframe, column, text_groupings)
    ]


def _within_group_note(dataframe: pd.DataFrame, roles: ColumnRoles, correlation: Correlation) -> str:
    """A sentence when r inside the groups differs in sign or clearly in size from the pooled r."""
    groupings = [roles.dimension] if roles.dimension else []
    groupings += [
        column
        for column in roles.dimensions
        if column != roles.dimension and 2 <= dataframe[column].nunique() <= MAX_SCAN_LEVELS
    ]
    pooled = correlation.pearson
    worst = None
    for grouping in dict.fromkeys(groupings):
        if grouping not in dataframe.columns:
            continue
        within = within_group_correlation(dataframe, correlation.first, correlation.second, grouping)
        if within is None:
            continue
        reversed_sign = pooled * within.within < 0 and min(abs(pooled), abs(within.within)) >= 0.1
        if reversed_sign or abs(pooled - within.within) >= WITHIN_GROUP_GAP:
            if worst is None or abs(pooled - within.within) > abs(pooled - worst.within):
                worst = within
    if worst is None:
        return ""
    group = sentence_case(worst.grouping)
    spread = (
        f"r = {worst.within:.2f} once each {group}'s own means are removed (from {worst.low:.2f} "
        f"to {worst.high:.2f} across {worst.groups} groups)"
    )
    if pooled * worst.within < 0:
        return (
            f" Within each {group} the relationship runs the other way: {spread}, so the pooled "
            f"figure reflects differences between the {group} groups."
        )
    if abs(worst.within) > abs(pooled):
        return (
            f" Within each {group} it is stronger: {spread}; pooling groups at different levels "
            "weakens it."
        )
    return (
        f" Within each {group} it is weaker: {spread}, so much of the pooled figure comes from "
        f"differences between the {group} groups."
    )


def _scanned_pair(dataframe: pd.DataFrame, roles: ColumnRoles, pair: Pair) -> tuple[Correlation, str]:
    """The full correlation of a pair the scan picked, with its within-group sentence."""
    correlation = measure_correlation(dataframe[[pair.first, pair.second]].dropna())
    assert correlation is not None
    return correlation, _within_group_note(dataframe, roles, correlation)


def _chance_note(pair: Pair, tests: int) -> str:
    """Say so when a picked correlation could be chance once every test run is allowed for."""
    if tests > 1 and min(1.0, pair.p_value * tests) >= 0.05:
        return f" With {tests:,} correlations tried, one this size could arise by chance."
    return ""


def _left_out(scan: RelationshipScan) -> list[str]:
    """The calculation's list of pairs set aside as related by construction."""
    notes = []
    if scan.derived:
        named = [item.describe() for item in scan.derived[:DERIVED_NAMED]]
        if len(scan.derived) > DERIVED_NAMED:
            named.append(f"{len(scan.derived) - DERIVED_NAMED} more")
        notes.append("pairs of a column and the columns it is built from (" + "; ".join(named) + ")")
    notes.extend(f"{flag.describe()}" for flag in scan.flags)
    if scan.shared_stems:
        named = [f"{first} and {second}" for first, second, _ in scan.shared_stems[:NEAR_DUPLICATES_NAMED]]
        notes.append(f"same-stem pairs above |r| = {SHARED_STEM_CORRELATION} ({', '.join(named)})")
    if scan.near_duplicates:
        notes.append(f"near-duplicates (|r| ≥ {NEAR_DUPLICATE_CORRELATION})")
    return notes


def _relationship_evidence(
    dataframe: pd.DataFrame, roles: ColumnRoles, columns: list[str] | None = None
) -> Evidence | None:
    """What relates to the measure, then the strongest other pair, or a plain null result.

    Every pair is screened at once; pairs related by construction are set
    aside and listed in the calculation, and the ranking discounts a
    correlation that one record makes or hides.
    """
    columns = numeric_readings(dataframe, roles) if columns is None else columns
    if len(columns) < 2:
        return None
    measure = roles.measure if roles.measure in columns else None
    scan = scan_relationships(dataframe, columns, measure)
    if scan is None or not (scan.correlates or scan.others):
        return None

    parts: list[str] = []
    value = None
    reported = 0
    if scan.measure:
        notable = [pair for pair in scan.correlates if pair.strength >= NOTABLE_CORRELATION]
        if notable:
            lead, within = _scanned_pair(dataframe, roles, notable[0])
            value = f"r = {lead.pearson:.2f}"
            reported += 1
            parts.append(f"{describe_correlation(lead)}{within}")
            if scan.measure_tests > 1:
                parts.append(
                    f"Of the {scan.measure_tests:,} columns tested against {measure}, "
                    f"{notable[0].second} relates most.{_chance_note(notable[0], scan.measure_tests)}"
                )
            if len(notable) > 1:
                parts.append(
                    "Next: "
                    + ", ".join(
                        f"{pair.second} (r = {pair.pearson:.2f}, Spearman {pair.spearman:.2f}, "
                        f"n = {pair.pairs:,})"
                        for pair in notable[1:CORRELATES_NAMED]
                    )
                    + "."
                )
        elif scan.correlates:
            closest = scan.correlates[0]
            figures = f"r = {closest.pearson:.2f}, Spearman {closest.spearman:.2f}, n = {closest.pairs:,}"
            parts.append(
                f"{closest.second} does not relate to {measure}: {figures}."
                if scan.measure_tests == 1
                else f"None of the {scan.measure_tests:,} columns tested against {measure} reaches "
                f"|r| = {NOTABLE_CORRELATION}; the closest is {closest.second} ({figures})."
            )
    others = [pair for pair in scan.others if pair.strength >= NOTABLE_CORRELATION]
    if others:
        pair, within = _scanned_pair(dataframe, roles, others[0])
        # The card's figure answers its title, which names the measure when there is one.
        if not scan.measure:
            value = f"r = {pair.pearson:.2f}"
        reported += 1
        lead_in = "Among the other columns, the strongest pair: " if scan.measure else ""
        chance = _chance_note(others[0], scan.pair_tests)
        parts.append(f"{lead_in}{describe_correlation(pair)}{within}{chance}")
    elif scan.others:
        strongest = scan.others[0]
        parts.append(
            f"No {'other ' if scan.measure else ''}pair reaches |r| = {NOTABLE_CORRELATION}; the "
            f"strongest is {strongest.first} and {strongest.second} (r = {strongest.pearson:.2f}, "
            f"n = {strongest.pairs:,})."
        )
    if value is None:
        value = f"|r| < {NOTABLE_CORRELATION}"
        largest = max(pair.strength for pair in (*scan.correlates, *scan.others))
        if (
            largest < UNRELATED_CORRELATION
            and len(dataframe) >= UNRELATED_MIN_ROWS
            and scan.pair_tests >= UNRELATED_MIN_PAIRS
        ):
            parts.append(
                "Measured quantities usually relate to one another, so a table in which none do is "
                "worth a check of how its values were produced."
            )
    elif scan.pair_tests == 1:
        parts.append("Treat it as a lead to check, not as evidence of cause.")
    elif reported == 1:
        parts.append(
            f"It was picked from {scan.pair_tests:,} pairs, so treat it as a lead to check, "
            "not as evidence of cause."
        )
    else:
        parts.append(
            f"These were picked from {scan.pair_tests:,} pairs, so treat them as leads to check, "
            "not as evidence of cause."
        )
    if scan.near_duplicates:
        parts[-1] += _near_duplicate_note(list(scan.near_duplicates))

    calculation = (
        "Pearson r over each pair's complete rows, 95% CI by Fisher transform, and Spearman r; "
        + (f"{_count(scan.measure_tests, 'correlation')} with {measure} and " if scan.measure else "")
        + f"{_count(scan.pair_tests, 'pair')} scanned in all, ranked by the larger of |Spearman r| "
        "and |Pearson r| without the pair's most influential record; intervals not adjusted for "
        "that selection"
    )
    if len(scan.columns) < scan.available:
        calculation += (
            f"; scan limited to {len(scan.columns)} of the {scan.available:,} columns: the measure "
            "and the most variable others"
        )
    if "once each" in " ".join(parts):
        calculation += "; within-group r after removing each group's means"
    left_out = _left_out(scan)
    if left_out:
        calculation += "; left out: " + "; ".join(left_out)
    return Evidence(
        kind="relationship",
        title=f"What relates to {measure}" if scan.measure else "Strongest measurable relationship",
        value=value,
        statement=" ".join(parts),
        calculation=calculation,
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

    metric = metric_for(roles.measure, roles)
    trend = by_kind.get("trend")
    driver = by_kind.get("driver")
    # Name the driver only when it moved with the total; a segment that grew
    # while the total fell didn't cause the fall. Both cards lead with the sign.
    if driver and trend and driver.value.startswith("-") != trend.value.startswith("-"):
        driver = None
    adverse = "decline" if metric.increase_is_welcome else "increase"
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
    largest = leader.subject if leader and leader.subject else f"the largest {group}"
    if concentration and concentration.tone == "warning":
        recommendations.append(
            Recommendation(
                "Next",
                f"Check how much rests on {largest}",
                f"Recompute the overall figures without {largest} to see how much they change.",
                concentration.statement,
            )
        )
    elif leader and not metric.additive and leader.subject is None:
        recommendations.append(
            Recommendation(
                "Next",
                f"Don't rank the {group} groups on {roles.measure}",
                (
                    f"The {group} groups can't be told apart on {roles.measure}; compare them on a "
                    "reading that does differ, or collect more records per group."
                ),
                leader.statement,
            )
        )
    elif leader and not metric.additive:
        recommendations.append(
            Recommendation(
                "Next",
                f"Find out what sets {leader.subject} apart",
                (
                    f"Check what else differs between {leader.subject} and the lowest {group} "
                    "(method, timing, sample makeup) before reading the gap as an effect of the group."
                ),
                leader.statement,
            )
        )
    elif leader:
        recommendations.append(
            Recommendation(
                "Next",
                f"Compare {largest} with the other {group} values",
                f"Look at what differs between {largest} and a typical {group}.",
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
        # Name only the splits the table has: a file with no date has no time window.
        splits = [f"each {group}" if roles.dimension else "", "each time window" if roles.date else ""]
        within = " and ".join(split for split in splits if split) or "different subsets of the rows"
        recommendations.append(
            Recommendation(
                "Test",
                "Test the relationship",
                (
                    f"Check whether it holds within {within} before acting on it; "
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
                (
                    "Check whether the most extreme values are real or entry and measurement "
                    "errors; either way they pull averages."
                ),
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


def _row_key(dataframe: pd.DataFrame, roles: ColumnRoles) -> str | None:
    """The identifier column when it names each row once, so one row is one of it."""
    if not roles.identifier or dataframe.empty:
        return None
    unique = dataframe[roles.identifier].nunique(dropna=True) == len(dataframe)
    return roles.identifier if unique else None


def _measure_alternatives(roles: ColumnRoles) -> tuple[str, ...]:
    """Columns that came close to being picked as the measure, other than the one in use."""
    # Older ColumnRoles have no such field.
    close = getattr(roles, "measure_alternatives", ()) or ()
    return tuple(column for column in close if column != roles.measure)


def _measure_choice_note(roles: ColumnRoles) -> str:
    """Say when the measure was a close call, and how to pick another, or ""."""
    alternatives = _measure_alternatives(roles)
    if not roles.measure or not alternatives:
        return ""
    return (
        f"Measure picked: {roles.measure} (also plausible: {list_names(list(alternatives))}). "
        f'Change it with measure="{alternatives[0]}" or in the app.'
    )


def _table_description(dataframe: pd.DataFrame, roles: ColumnRoles) -> str:
    """What the table seems to be: its rows and what one row is, the dates, the measure and the groups."""
    opening = _rows(len(dataframe))
    if row_key := _row_key(dataframe, roles):
        opening += f", one per {row_key}"
    if roles.date:
        dates = dataframe[roles.date].dropna()
        if not dates.empty:
            first, last = f"{dates.min():%d %b %Y}", f"{dates.max():%d %b %Y}"
            opening += f", dated {first}" if first == last else f", dated {first} to {last}"
    sentences = [f"{opening}."]
    if not roles.measure:
        sentences.append("No numeric measure was found, so rows are counted.")
    elif not _measure_alternatives(roles):
        combined = "averaged" if metric_for(roles.measure, roles).aggregation == "mean" else "summed"
        sentences.append(f"Measure: {roles.measure}, {combined} across rows.")
    if roles.dimension:
        groups = dataframe[roles.dimension].nunique(dropna=True)
        sentences.append(f"Groups: the {groups:,} values of {roles.dimension}.")
    missing = [name for name, column in (("date", roles.date), ("grouping", roles.dimension)) if not column]
    if missing:
        sentences.append(f"No {' or '.join(missing)} column was found.")
    return " ".join(sentences)


def _first_sentence(statement: str) -> str:
    # Cut where a capital starts the next sentence, not at the decimal point
    # inside "18.5%" or the "No. 40" inside a group's name.
    return re.split(r"\.\s+(?=[A-Z])", statement, maxsplit=1)[0].rstrip(".") + "."


def _averaged_leader_phrase(leader: Evidence, roles: ColumnRoles) -> str:
    """The leading-group clause of a headline for an averaged measure."""
    if leader.subject is None:
        return f"no clear difference in {roles.measure} between {sentence_case(roles.dimension)} groups."
    centre = "median" if leader.calculation.startswith("median ") else "average"
    return f"{leader.subject} has the highest {centre}, at {leader.value}."


def analyze_business(
    dataframe: pd.DataFrame,
    roles: ColumnRoles | None = None,
    frequency: str | None = None,
) -> BusinessBrief:
    """Create an executive brief from explainable calculations and rule-based interpretation.

    ``frequency`` fixes the period grain (``s min h D W M Q Y``) for the trend, driver and
    anomaly findings, so they match a trend or forecast built at that grain.
    Without it the grain follows how often the data was recorded.
    """
    roles = roles or detect_roles(dataframe)
    metric = metric_for(roles.measure, roles)
    evidence: list[Evidence] = []

    growth = _growth_evidence(dataframe, roles, frequency)
    if growth:
        evidence.append(growth)
    change_driver = _change_driver_evidence(dataframe, roles, frequency)
    if change_driver:
        evidence.append(change_driver)
    evidence.extend(_segment_evidence(dataframe, roles))
    # Both scans work on the same numeric readings, which are slow to pick out of a wide table.
    readings = numeric_readings(dataframe, roles)
    differences = _differences_evidence(dataframe, roles, readings)
    found_difference = differences is not None and differences.subject is not None
    if found_difference:
        evidence.append(differences)
    for optional_evidence in (
        _anomaly_evidence(dataframe, roles, frequency),
        _relationship_evidence(dataframe, roles, readings),
        _outlier_evidence(dataframe, roles),
        # A scan that found nothing ranks below the findings.
        None if found_difference else differences,
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
        if metric.additive:
            kpis.append(
                KPI(
                    with_aggregate("Total", roles.measure),
                    format_number(total, roles.measure, column_values=measure_values),
                    "Summed across all analyzed rows",
                )
            )
        kpis.append(
            KPI(
                with_aggregate("Average", roles.measure),
                format_number(average, roles.measure, column_values=measure_values),
                "Per row with a value",
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
    leader = next((item for item in evidence if item.kind == "leader"), None)
    # A comparison that found no clear difference has no top group to show.
    if leader and leader.subject is not None:
        # The card's title says what was ranked; the tile names the group.
        kpis.append(KPI(leader.title, leader.value, leader.subject))

    # Pad towards four tiles from a fixed list, skipping labels already shown;
    # a thin file gets fewer tiles rather than duplicates.
    completeness = 1 - dataframe.isna().sum().sum() / max(dataframe.size, 1)
    fallbacks = []
    row_key = _row_key(dataframe, roles)
    if roles.identifier and not row_key:
        fallbacks.append(
            KPI(
                f"Distinct {roles.identifier}",
                f"{dataframe[roles.identifier].nunique(dropna=True):,}",
                "Different values in the column",
            )
        )
    fallbacks.append(
        KPI(
            "Rows analyzed",
            f"{len(dataframe):,}",
            f"One per {row_key}" if row_key else "After conservative cleaning",
        )
    )
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
    if growth and leader:
        # A share is quoted only when the leader card divides by a total; a
        # level (non-additive or mixed-sign measure) is the highest or largest.
        if "÷ total" in leader.calculation:
            ranked = (
                f"has the largest share of {sentence_case(roles.measure)}, at {leader.value}"
                if roles.measure
                else f"has the most rows, {leader.value} of them"
            )
            headline = f"{growth.value} latest movement; {leader.subject} {ranked}."
        elif not metric.additive:
            headline = f"{growth.value} latest movement; {_averaged_leader_phrase(leader, roles)}"
        else:
            headline = (
                f"{growth.value} latest movement; {leader.subject} is the largest by size, "
                f"at {leader.value}."
            )
    elif growth:
        headline = _first_sentence(growth.statement)
    elif leader:
        # A comparison that found nothing gives way to a difference found elsewhere.
        found_elsewhere = leader.subject is None and differences is not None and differences.subject
        headline = _first_sentence(differences.statement if found_elsewhere else leader.statement)
    elif differences is not None and differences.subject:
        headline = _first_sentence(differences.statement)
    elif roles.measure:
        measure_values = dataframe[roles.measure].dropna()
        combined = measure_values.mean() if metric.aggregation == "mean" else measure_values.sum()
        total_measure = format_number(
            float(combined),
            roles.measure,
            column_values=measure_values,
        )
        verb = "averages" if metric.aggregation == "mean" else "totals"
        headline = f"{roles.measure} {verb} {total_measure} across all rows."
    else:
        headline = f"{len(dataframe):,} rows, with no date, measure or grouping column ADA could use."

    # Without a trend the headline already gives the one finding there is, so
    # the summary says what the table seems to be instead of repeating it.
    if growth:
        summary_parts = [growth.statement] + ([leader.statement] if leader else [])
    else:
        summary_parts = [_table_description(dataframe, roles)]
    if differences is not None and differences.subject and (leader is None or leader.subject is None):
        summary_parts.append(_first_sentence(differences.statement))
    if note := _measure_choice_note(roles):
        summary_parts.append(note)

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
        lines.append(f"**Context:** {context.strip()}")
    lines.extend(["", "## Summary", "", brief.headline, "", brief.summary, ""])
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
        lines.extend(["No finding was strong enough to report.", ""])

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
