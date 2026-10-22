"""Turn a dataset into an executive brief: KPIs, evidence cards and recommendations.

Every figure comes from a deterministic pandas calculation that is shown beside it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy import stats

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
from .anomalies import detect_anomalies, finer_trend
from .compare import (
    ADJUSTED_BY,
    ALPHA,
    CAP_MIN_DISTINCT,
    CAP_MIN_ROWS,
    CAP_SHARE,
    FALSE_DISCOVERY,
    INFLUENCE_CHANGE,
    MAX_MODEL_TERMS,
    MAX_SCAN_LEVELS,
    MAX_VIF,
    MIN_NAMED_GROUP,
    NEAR_DUPLICATE_CORRELATION,
    NOTABLE_CORRELATION,
    RANK_DIVERGENCE,
    SCAN_ROWS,
    SHARED_STEM_CORRELATION,
    STRATUM_EXPLAINED,
    Construction,
    Derivation,
    GroupComparison,
    GroupingRule,
    Pair,
    RelationshipScan,
    WithinGroups,
    centre_word,
    combined_groupings,
    compare_groups,
    compare_rates,
    comparison_calculation,
    constant_within,
    construction,
    correlation_interval,
    date_part,
    describe_comparison,
    describe_rates,
    describe_stratified,
    effect_clause,
    effect_text,
    entity_column,
    explained_share,
    far_out,
    fit_joint_model,
    format_p,
    gain_within,
    grouping_rules,
    holm_adjust,
    lead_sentence,
    leads_with_trend,
    longest_stretch,
    most_influential,
    nested_within,
    one_row_per,
    outcome_event,
    rate_calculation,
    same_stem,
    scan_differences,
    scan_relationships,
    stratified_comparison,
    stratified_lead,
    stratified_rates,
    variance_explained,
    within_axis,
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
from .quality import (
    INFO_CHECKS,
    TABLE_CHECKS,
    QualityIssue,
    counts_units,
    flagged_index,
    is_location_name,
    own_columns,
    prefers_log_scale,
    show_value,
    statistics_frame,
)
from .quality import scan as scan_quality
from .schema import (
    INPUT_WORDS,
    TIME_PART_TOKENS,
    TIME_PART_WORDS,
    ColumnRoles,
    detect_roles,
    is_design_name,
    is_entity_name,
    is_month_name,
    is_year_name,
    looks_like_identifier,
    row_summary_columns,
)
from .timeseries import MIN_SHIFT_EFFECT, level_shift, observed_periods, robust_scale


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
    #: One sentence for the headline when this card holds the brief's strongest finding, or "".
    lead: str = ""
    #: The card's change is within the usual variation, so it is no clear change.
    quiet: bool = False


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


def _movement_in_context(
    values: np.ndarray, change: float, describe=format_percentage
) -> tuple[str, bool | None]:
    """Describe the latest change relative to this series' usual volatility.

    ``change`` is a percentage when ``describe`` is ``format_percentage`` and a
    plain difference otherwise. Returns a sentence with a leading space, or ""
    when there are too few prior changes or they have no spread, and whether
    the change is within the usual variation (None when it can't be judged).
    """
    relative = describe is format_percentage
    prior = _period_changes(values[:-1], relative=relative)
    if len(prior) < MIN_PERIODS_FOR_VOLATILITY:
        return "", None

    spread = robust_scale(prior)
    typical = float(np.median(np.abs(prior)))
    if spread == 0:
        return "", None

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
    quiet = distance < 1.0 or not moved_more
    return f" {verdict}; this series typically moves about {describe(typical)} per period.", quiet


def _rows_in_period(
    dataframe: pd.DataFrame, roles: ColumnRoles, frequency: str, period: pd.Timestamp
) -> np.ndarray:
    """The measure's recorded values in one period."""
    stamps = dataframe[roles.date].dt.to_period(frequency).dt.to_timestamp()
    return dataframe.loc[stamps == period, roles.measure].dropna().to_numpy(dtype=float)


def _noise_note(previous: np.ndarray, current: np.ndarray) -> tuple[str, bool | None]:
    """Whether two periods' averages differ by more than the noise in their rows, by Welch's t-test.

    Used when the series is too short to judge a change against its own
    history. Returns a sentence with a leading space and whether the change is
    within the noise.
    """
    if min(len(previous), len(current)) < 2 or not (previous.std() > 0 or current.std() > 0):
        return "", None
    p = float(stats.ttest_ind(current, previous, equal_var=False).pvalue)
    if not np.isfinite(p):
        return "", None
    if p >= ALPHA:
        return (
            f" That is within the noise: the two periods' rows do not differ clearly (Welch's t-test, "
            f"{format_p(p)}).",
            True,
        )
    return f" The two periods' rows differ clearly (Welch's t-test, {format_p(p)}).", False


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
    # Without a grouping column, the unit each row belongs to (a station) is the group.
    grouped = roles
    if not roles.dimension and (unit := entity_column(dataframe, roles)):
        grouped = replace(roles, dimension=unit)
    matched = matched_period_change(dataframe, grouped, series)
    pooled = (previous, current)
    history = series
    compared = dataframe
    if matched is not None and matched.groups:
        previous, current = matched.previous, matched.current
        compared = dataframe[dataframe[grouped.dimension].isin(matched.kept)]
        history = build_trend(compared, roles, series.frequency)
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
        context, quiet = _movement_in_context(
            values, delta, lambda size: format_change(0, size, roles.measure, unit, measure_values)
        )
        calculation = "Latest period average − previous period average"
        if unit == "rate":
            calculation += ", in percentage points"
        change = delta
    else:
        change = delta / abs(previous) * 100
        value = format_percentage(change, signed=True)
        context, quiet = _movement_in_context(values, change)
        calculation = "(Latest period − previous period) ÷ |previous period|"
    against = "compared with past period-to-period changes"
    seasonal = seasonal_context(history, relative=unit not in ("rate", "level"))
    if seasonal is not None:
        context, quiet = _seasonal_note(seasonal, history, change, current, roles, unit, measure_values)
        against = "compared with the same move in earlier years and the same period a year earlier"
    if quiet is None and roles.measure and not metric.additive:
        # Too short a history to judge the move against, so it is judged against
        # the spread of the rows in the two periods instead.
        periods = (trend.iloc[-2]["Period"], trend.iloc[-1]["Period"])
        sides = [_rows_in_period(compared, roles, series.frequency, period) for period in periods]
        note, quiet = _noise_note(*sides)
        if note:
            context += note
            against = "compared with the spread of the two periods' rows by Welch's t-test (scipy.stats)"
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
        context = _matched_note(matched, grouped, pooled, series, measure_values) + context
        if matched.groups:
            calculation = f"On the {grouped.dimension} values present in both periods: {calculation}"
    return Evidence(
        kind="trend",
        title=f"Latest {sentence_case(measure)} movement",
        value=value,
        statement=(
            f"{measure} {direction} {moved} in the latest {completeness} "
            f"({period}), from {start} to {end}.{context}"
        ),
        calculation=f"{calculation}, {against}",
        # Recommendations follow this tone.
        tone=_tone(delta >= 0, metric),
        subject=period,
        quiet=bool(quiet),
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
) -> tuple[str, bool | None]:
    """Set the latest move beside the same move in earlier years and the same period a year ago.

    Replaces the comparison with all past period-to-period changes, which a
    yearly cycle makes look unusual every spring. Also returns whether the move
    is within the range of earlier years (None with fewer than two of them).
    """
    relative = unit not in ("rate", "level")

    def describe(size: float) -> str:
        if relative:
            return format_percentage(size, signed=True)
        return format_change(0, size, roles.measure, unit, measure_values, signed=True) or "—"

    latest = pd.Timestamp(series.frame.iloc[-1]["Period"])
    season = _season_name(latest, series.frequency)
    note = ""
    quiet = None
    usual = np.asarray(seasonal.usual_changes, dtype=float)
    if usual.size == 1:
        note = f" In the one earlier year on record, the move into {season} was {describe(usual[0])}."
    elif usual.size:
        low, high = float(usual.min()), float(usual.max())
        quiet = bool(low <= change <= high)
        within = "within" if quiet else "outside"
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
    return note, quiet


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
    # Too few periods at the chosen grain to calibrate a band, so a finer one is checked.
    finer = finer_trend(dataframe, roles, series)
    checked = finer or series
    metric = metric_for(roles.measure, roles)
    anomalies = detect_anomalies(checked.frame, floor_at_zero=metric.floors_at_zero)
    if not anomalies:
        return None
    # Label at the series' own grain, which a fixed frequency may have set.
    grain = checked.frequency
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
    finer_note = ""
    if finer is not None:
        finer_note = (
            f" Periods are checked by {GRAIN_NAMES.get(grain, grain)}, since by "
            f"{GRAIN_NAMES.get(series.frequency, series.frequency)} there are only "
            f"{len(observed_periods(series.frame))}, too few to set the band."
        )
    return Evidence(
        kind="anomaly",
        title="Anomalous periods",
        value=f"{len(anomalies)}",
        statement=(
            f"{label} is the sharpest anomaly: {sentence_case(measure)} reached {reached}, "
            f"{worst.direction} the expected {low}–{high} range. "
            f"{len(anomalies)} {plural} outside the {band}.{finer_note}"
        ),
        calculation=(
            f"Period values vs {expected} ± a band calibrated so that only "
            "1 stable series in 20 raises a flag"
        ),
        tone="warning",
        lead=f"{label} is an anomalous period: {sentence_case(measure)} reached {reached}, "
        f"{worst.direction} the expected {low}–{high} range.",
    )


GRAIN_NAMES = {
    "s": "second", "min": "minute", "h": "hour", "D": "day", "W": "week", "M": "month", "Q": "quarter",
    "Y": "year",
}


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


def _segment_evidence(
    dataframe: pd.DataFrame,
    roles: ColumnRoles,
    tested: pd.DataFrame | None = None,
    note: str = "",
    found: _Groupings | None = None,
    *,
    rules: tuple[GroupingRule, ...] = (),
    built: Construction | None = None,
) -> tuple[Evidence, ...]:
    """The leading group: a share of the total for money and counts, a group comparison otherwise.

    ``tested`` holds the rows the comparison runs on (the distinct rows when
    many repeat exactly), and ``note`` says so in its calculation. A grouping
    cut from the measure (``rules``, with ``built`` for the columns built from
    it) ranks its groups by definition, so the comparisons card states the
    rule instead.
    """
    if roles.measure and roles.dimension and _rule_for(rules, built, roles.measure, roles.dimension):
        return ()
    # All segments, so shares and concentration are measured against the full total.
    segments = segment_frame(dataframe, roles, limit=None)
    if segments.empty:
        return ()

    # Shares also need an additive measure: segment churn rates of 10% and 20%
    # don't make one segment two thirds of churn. A summed per-row score (seconds
    # of looking time per infant) is compared too: its shares mostly count rows.
    metric = metric_for(roles.measure, roles)
    additive = metric.additive
    measure_values = dataframe[roles.measure].dropna() if roles.measure else None
    if not additive or (roles.measure and not _counts_money_or_things(roles, metric, measure_values)):
        compared = _group_difference_evidence(dataframe if tested is None else tested, roles, note, found)
        if compared is not None:
            if additive:
                shares = _share_note(dataframe, roles, segments)
                compared = replace(compared, statement=compared.statement + shares)
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
            # An average ranked without a test names no group in the headline.
            lead=_first_sentence(statement) if additive else "",
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


def _share_note(dataframe: pd.DataFrame, roles: ColumnRoles, segments: pd.DataFrame) -> str:
    """The largest group's share of a summed per-row score, beside its share of the rows."""
    present = segments.dropna(subset=["Value"])
    values = present["Value"].to_numpy(dtype=float)
    total = float(values.sum()) if values.size else 0.0
    if not total > 0 or not _shares_are_meaningful(values):
        return ""
    leader = present.loc[present["Value"].astype(float).idxmax()]
    rows = _rows_per_group(dataframe, roles)
    count, everything = int(rows.get(leader["Segment"], 0)), int(rows.sum())
    if not everything:
        return ""
    share, row_share = float(leader["Value"]) / total * 100, count / everything * 100
    mostly_rows = ""
    if abs(share - row_share) <= 10:
        mostly_rows = ", so its share mostly reflects how many rows it has"
    return (
        f" Summed, {group_label(roles.dimension, leader['Segment'])} holds {format_percentage(share)} of "
        f"all {roles.measure} and {count:,} of the {everything:,} rows ({format_percentage(row_share)})"
        f"{mostly_rows}."
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


def _group_difference_evidence(
    dataframe: pd.DataFrame, roles: ColumnRoles, note: str = "", found: _Groupings | None = None
) -> Evidence | None:
    """The measure compared across the grouping, with its test and effect size.

    A gap the test can't tell from noise is reported as no clear difference,
    so the card never ranks groups that don't differ. When the grouping is the
    combination of two others (a cell code for depth and contour), each of
    those factors is tested on its own as well, since the cells say nothing
    about which factor matters. ``note`` is added to the calculation. None
    when no two groups have two or more values.
    """
    if not roles.measure or not roles.dimension:
        return None
    found = found or _find_groupings(dataframe, roles)
    entity = found.entity
    comparison = compare_groups(dataframe, roles.measure, roles.dimension, entity=entity, axis=found.axis)
    if comparison is None:
        return None
    values = dataframe[roles.measure].dropna()
    dimension = sentence_case(roles.dimension)
    clear = comparison.verdict == "clear"
    calculation = comparison_calculation(comparison)
    centre = "median" if comparison.uses_ranks and not comparison.zero_heavy else "average"
    if comparison.entity:
        # The tile shows a figure per entity, which a plain average of rows would not match.
        centre = f"per-{comparison.entity} {centre}"
    statement = describe_comparison(comparison, values)
    lead = lead_sentence(comparison, values) if clear or leads_with_trend(comparison) else ""
    stratified = stratified_comparison(
        dataframe, roles.measure, roles.dimension, _stratum_candidates(dataframe, roles, found),
        entity=entity,
    )
    if stratified is not None:
        statement += describe_stratified(stratified, comparison, values)
        calculation += (
            f"; compared again within each {stratified.stratum} (tied to {roles.dimension} with Cramér's V = "
            f"{stratified.association:.2f}, scipy.stats.contingency.association)"
        ) + _sample_note(stratified.sampled)
        if stratified.overturns(comparison):
            lead = stratified_lead(stratified, comparison, values)
    factors = _factor_comparisons(dataframe, roles, found)
    if factors:
        first, second = factors
        statement += (
            f" {roles.dimension} combines {first.grouping} and {second.grouping}; each on its own: "
            f"{_factor_clause(first, values)}; {_factor_clause(second, values)}."
        )
        calculation += f"; {first.grouping} and {second.grouping} also compared on their own"
        differing = [factor for factor in factors if factor.verdict == "clear"]
        if differing and not clear:
            lead = lead_sentence(max(differing, key=lambda factor: factor.explained), values)
    imbalance = _input_imbalance(dataframe, roles, entity)
    if imbalance:
        statement += imbalance
        calculation += "; inputs compared across the same groups, Holm-adjusted"
    if note:
        calculation += note
    return Evidence(
        kind="leader",
        title=(
            f"Highest {centre} by {dimension}"
            if clear
            else f"Too few values to rank by {dimension}"
            if comparison.verdict == "thin"
            else f"No clear difference by {dimension}"
        ),
        value=(
            format_number(comparison.centre(comparison.higher), roles.measure, column_values=values)
            if clear
            else effect_text(comparison).replace("Hedges' ", "")
        ),
        # A group that reported for only part of the dates is said to have done so.
        statement=statement
        + _coverage_note(dataframe, roles, [comparison.higher.value, comparison.lower.value]),
        calculation=calculation,
        subject=_subject(_leading_groups(comparison)) if clear else None,
        lead=lead,
    )


def _sample_note(rows: int) -> str:
    """Words for a figure computed on the scans' fixed sample of a long table, or ""."""
    return f", on a fixed random sample of {SCAN_ROWS:,} of the {rows:,} rows" if rows else ""


def _stratum_candidates(dataframe: pd.DataFrame, roles: ColumnRoles, found: _Groupings) -> list[str]:
    """Groupings that could make or hide a comparison: every grouping with a few levels, scanned or not."""
    return [
        column
        for column in dict.fromkeys([*found.chosen, *found.capped, *roles.dimensions, *roles.factors])
        if column in dataframe.columns and column not in (found.entity, roles.measure)
    ]


def _input_imbalance(dataframe: pd.DataFrame, roles: ColumnRoles, entity: str | None) -> str:
    """Say when the inputs of a mix or dose differ across the groups, since the comparison then mixes them in.

    In a table of concrete mixes tested at different ages, a strength gap
    between ages is partly a gap between mixes when the mixes differ too.
    Inputs are readings named as what goes into a mix or a dose (cement,
    water, dose).
    """
    inputs = [
        column
        for column in numeric_readings(dataframe, roles)
        if column != roles.measure and set(normalized_name(column).split()) & INPUT_WORDS
    ]
    if not inputs:
        return ""
    scan = scan_differences(dataframe, inputs, [roles.dimension], entity=entity, limit=DIFFERENCES_NAMED)
    if scan is None or not scan.named:
        return ""
    named = list_names([f"{item.measure} ({effect_text(item)})" for item in scan.named])
    return (
        f" The inputs differ across {roles.dimension} too: {named}. The groups were not given the same "
        f"inputs, so the comparison of {roles.measure} is partly a comparison of inputs."
    )


def _factor_comparisons(
    dataframe: pd.DataFrame, roles: ColumnRoles, found: _Groupings
) -> tuple[GroupComparison, GroupComparison] | None:
    """The measure compared across each of the two factors the grouping combines, or None."""
    parts = found.combined.get(roles.dimension)
    if parts is None:
        return None
    compared = [
        compare_groups(dataframe, roles.measure, part, entity=found.entity, axis=found.axis) for part in parts
    ]
    if any(comparison is None for comparison in compared):
        return None
    return compared[0], compared[1]


def _factor_clause(comparison: GroupComparison, values: pd.Series) -> str:
    """One factor's comparison in a clause: its extremes and effect, or that it shows no clear difference."""
    c = comparison
    if c.verdict != "clear":
        return f"{c.grouping} shows no clear difference ({effect_clause(c)})"
    first, second = format_values_apart(
        c.centre(c.higher), c.centre(c.lower), column=c.measure, column_values=values
    )
    return (
        f"{c.grouping} differs, {centre_word(c)} {first} ({c.higher.label}, n = {c.higher.n:,}) against "
        f"{second} ({c.lower.label}, n = {c.lower.n:,}), {effect_clause(c)}"
    )


DIFFERENCES_NAMED = 3
# A scan with nothing clear is still reported once it covers this many comparisons.
MIN_NULL_COMPARISONS = 3
# The scan compares readings across at most this many groupings, design factors first,
# so dozens of codes can't bury the effect of the design in a sea of tests; a table
# with this many or fewer has every one scanned. A scan of more than SCAN_GROUPING_CELLS
# values across all its groupings (rows scanned times readings times groupings) keeps
# fewer, but never fewer than MIN_SCAN_GROUPINGS, so a long, wide table stays fast.
MAX_SCAN_GROUPINGS = 12
MIN_SCAN_GROUPINGS = 6
SCAN_GROUPING_CELLS = 2_500_000


def _same_partition(dataframe: pd.DataFrame, first: str, second: str) -> bool:
    """Whether two columns split the rows into the same groups, as a code and its label do.

    They do when each value of one goes with exactly one value of the other,
    which integer codes show without a groupby: a table with many groupings
    checks every pair.
    """
    return _codes_match(_partition_codes(dataframe[first]), _partition_codes(dataframe[second]))


def _partition_codes(values: pd.Series) -> np.ndarray:
    return pd.factorize(values)[0].astype(np.int64)


def _codes_match(one: np.ndarray, other: np.ndarray) -> bool:
    kept = (one >= 0) & (other >= 0)
    if not kept.any():
        return False
    one, other = one[kept], other[kept]
    pairs = np.unique(one * (int(other.max()) + 1) + other).size
    return bool(pairs == np.unique(one).size == np.unique(other).size)


@dataclass(frozen=True)
class _Groupings:
    """The groupings the comparisons use, found once for the brief, and the candidates left out and why."""

    chosen: tuple[str, ...]
    #: The column naming each row's subject, counted once per group.
    entity: str | None = None
    #: Groupings that combine two others, with the two.
    combined: dict = field(default_factory=dict)
    #: (column, part): a numeric column that repeats a part of the date column.
    date_parts: tuple[tuple[str, str], ...] = ()
    #: (finer, coarser): a grouping whose levels each occur with only some levels of another.
    nested: tuple[tuple[str, str], ...] = ()
    #: Groupings past the cap on how many are scanned.
    capped: tuple[str, ...] = ()
    #: A numeric column that climbs within each entity, such as a time point (``within_axis``).
    axis: str | None = None
    #: The measure is the same on every row of an entity: the entity is the sample it was read on.
    copied: bool = False
    #: Numeric settings with a few values, each fixed within an entity (a test case's feed rate),
    #: compared as groupings rather than as readings.
    settings: tuple[str, ...] = ()


def _comparison_groupings(
    dataframe: pd.DataFrame, roles: ColumnRoles, entity: str | None, settings: tuple[str, ...] = ()
) -> _Groupings:
    """The grouping column, then other groupings with 2 to MAX_SCAN_LEVELS levels, each split once.

    Design-like groupings come first: text labels, then two-level codes, then
    other codes by how few levels they have, so a rating item read as a code
    comes last, after the ``settings`` fixed within each entity; at most
    MAX_SCAN_GROUPINGS are kept, and when there are more candidates, the codes
    that explain most of the measure come first. The column naming each
    row's subject is left out, since readings differ between subjects as a
    matter of course; so is a month or year taken from the date column, or
    named as one beside it, which only shows the season or the trend, and a
    grouping nested within one already kept, whose differences repeat that one's.
    """
    frame = dataframe.sample(SCAN_ROWS, random_state=0) if len(dataframe) > SCAN_ROWS else dataframe

    def levels(column: str) -> int:
        return int(frame[column].nunique())

    def eligible(column: str) -> bool:
        return column in frame.columns and 2 <= levels(column) <= MAX_SCAN_LEVELS

    labels = [column for column in roles.dimensions if column not in roles.factors and eligible(column)]
    codes = sorted((column for column in roles.factors if eligible(column)), key=levels)
    per_grouping = len(frame) * max(len(roles.numeric), 1)
    cap = min(MAX_SCAN_GROUPINGS, max(MIN_SCAN_GROUPINGS, SCAN_GROUPING_CELLS // per_grouping))
    codes = [*(column for column in settings if eligible(column) and column not in codes), *codes]
    candidates = [roles.dimension] if roles.dimension else []
    if len(labels) + len(codes) + len(candidates) > cap and roles.measure in frame.columns:
        # Too many to scan: the codes that go with the measure most, not the first few, are kept.
        codes.sort(key=lambda column: -explained_share(frame[roles.measure], frame[column]))
    chosen: list[str] = []
    # Each column coded once, since every pair is checked.
    partitions: dict[str, np.ndarray] = {}
    capped: list[str] = []
    date_parts: list[tuple[str, str]] = []
    nested: list[tuple[str, str]] = []
    for column in dict.fromkeys([*candidates, *labels, *codes]):
        if column in (entity, roles.identifier, roles.measure) or column not in frame.columns:
            continue
        if frame[column].nunique() < 2:
            continue
        if len(chosen) >= cap:
            capped.append(column)
            continue
        partitions[column] = _partition_codes(frame[column])
        if any(_codes_match(partitions[column], partitions[other]) for other in chosen):
            continue
        part = date_part(dataframe[column], dataframe[roles.date]) if roles.date else None
        if part is not None:
            date_parts.append((column, part))
            continue
        # A year or month column beside the dates holds a cycle that belongs to the timeline.
        if roles.date and _last_word(column) in TIME_PART_WORDS:
            continue
        coarser = next((other for other in chosen if nested_within(frame, column, other)), None)
        if coarser is not None and column != roles.dimension:
            nested.append((column, coarser))
            continue
        chosen.append(column)
    return _Groupings(
        chosen=tuple(chosen),
        date_parts=tuple(date_parts),
        nested=tuple(nested),
        capped=tuple(capped),
    )


def _find_groupings(dataframe: pd.DataFrame, roles: ColumnRoles) -> _Groupings:
    """The brief's groupings, with the subject column and the groupings that combine two others.

    The subject column brings its clock (a time point climbing within each
    subject) and the settings fixed within each subject, which group the
    subjects (a test case's feed rate and material).
    """
    entity = entity_column(dataframe, roles)
    axis = within_axis(dataframe, entity, exclude=(roles.measure,)) if entity else None
    copied = bool(entity and roles.measure and constant_within(dataframe, entity, roles.measure))
    settings = _unit_settings(dataframe, roles, entity) if entity and not copied else ()
    found = _comparison_groupings(dataframe, roles, entity, settings)
    chosen = list(found.chosen)
    return replace(
        found,
        entity=entity,
        combined=combined_groupings(dataframe, chosen) if chosen else {},
        axis=axis,
        copied=copied,
        settings=tuple(column for column in settings if column in chosen),
    )


# A setting fixed within each entity holds at most this many values, each on at least two entities.
MAX_SETTING_LEVELS = 4


def _unit_settings(dataframe: pd.DataFrame, roles: ColumnRoles, entity: str) -> tuple[str, ...]:
    """Numeric readings with 2 to MAX_SETTING_LEVELS values, each fixed within every entity.

    A depth of cut of 0.75 or 1.5 set once per test case is a factor of the
    design: it groups the cases, and averaging it as a reading says nothing.
    """
    found = []
    known = {roles.measure, roles.dimension, entity, *roles.dimensions, *roles.factors}
    for column in roles.numeric:
        if column in known or column not in dataframe.columns:
            continue
        values = dataframe[column]
        if not 2 <= values.nunique() <= MAX_SETTING_LEVELS or not constant_within(dataframe, entity, column):
            continue
        units = dataframe[[entity, column]].dropna().drop_duplicates(subset=[entity])[column].value_counts()
        if len(units) >= 2 and int(units.min()) >= 2:
            found.append(column)
    return tuple(found)


def _rule_groupings(roles: ColumnRoles, groupings: _Groupings) -> list[str]:
    """The groupings the brief compares on, which a reading may have set: the main one and the scanned."""
    return list(dict.fromkeys([*([roles.dimension] if roles.dimension else []), *groupings.chosen]))


def _rule_for(
    rules: tuple[GroupingRule, ...], built: Construction | None, reading: str, grouping: str
) -> GroupingRule | None:
    """The rule by which ``grouping`` is set from ``reading``, or from a column built from or into it."""
    for rule in rules:
        if rule.grouping != grouping:
            continue
        if reading in rule.readings:
            return rule
        if built is not None and reading in built.position and any(
            other in built.position and built.relation(reading, other) for other in rule.readings
        ):
            return rule
    return None


def _built_from_grouping(built: Construction | None, reading: str, grouping: str) -> bool:
    """Whether a score is built from a grouping column, as a scale is from its rated items."""
    return (
        built is not None
        and reading in built.position
        and grouping in built.position
        and (built.built_from(reading, grouping) or built.built_from(grouping, reading))
    )


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


def _pairs_text(pairs: list[tuple[str, str]], limit: int = DIFFERENCES_NAMED) -> str:
    """ "a by b, c by d and 2 more" for a calculation."""
    named = ", ".join(f"{reading} by {grouping}" for reading, grouping in pairs[:limit])
    extra = len(pairs) - limit
    return named + (f" and {extra} more" if extra > 0 else "")


def _adjustment(scan) -> str:
    """How the scan's p-values were adjusted, for a sentence."""
    if scan.correction == FALSE_DISCOVERY:
        return f"with a false discovery rate of 5% (Benjamini-Hochberg) over all {scan.tested:,} tests"
    return f"after a Holm adjustment for all {scan.tested:,} tests"


def _differences_evidence(
    dataframe: pd.DataFrame,
    roles: ColumnRoles,
    readings: list[str] | None = None,
    note: str = "",
    found: _Groupings | None = None,
    *,
    built: Construction | None = None,
    rules: tuple[GroupingRule, ...] | None = None,
) -> Evidence | None:
    """Which readings differ between groups, largest standardised difference first.

    Every numeric reading is compared across the grouping column and the other
    groupings with few levels, with p-values adjusted for the number of tests.
    Pairs related by construction are left out and named: a grouping cut from
    the reading (stated as a rule), a score and the items it is built from, a
    reading fixed within each group, and copies of one reading in other
    units, which are compared once. A table with several groupings also gets
    the share of the measure's variance each one explains. ``built``,
    ``rules`` and ``found`` are the brief's screens of the table, made here
    when not given, and ``note`` is added to the calculation.
    """
    found = found if found is not None else _find_groupings(dataframe, roles)
    entity, groupings = found.entity, list(found.chosen)
    if not groupings:
        return None
    readings = numeric_readings(dataframe, roles) if readings is None else readings
    # The subject's code and its clock are not readings: how long a subject was
    # followed differs between groups by design, not as a finding.
    readings = [
        reading for reading in readings if reading not in (entity, found.axis, *found.settings)
    ]
    if not readings:
        return None
    frame = built.masked(dataframe) if built is not None else dataframe
    # One quantity in two units (°C and °F) is compared once, as the measure where it is one.
    copies = []
    if built is not None:
        prefer = (roles.measure,) if roles.measure else ()
        for reading in readings:
            kept = built.representative(reading, prefer)
            if kept != reading and kept in readings:
                copies.append((reading, kept))
    readings = [reading for reading in readings if reading not in {copy for copy, _ in copies}]
    if rules is None:
        rules = grouping_rules(frame, readings, _rule_groupings(roles, found), date=roles.date)
    # A cell code made of two other groupings explains as much as both together,
    # which says nothing about which of them matters.
    combined = {
        grouping: parts
        for grouping, parts in found.combined.items()
        if all(part in groupings for part in parts)
    }
    scanned = [grouping for grouping in groupings if grouping not in combined]
    set_by = [
        (reading, grouping)
        for grouping in scanned
        for reading in readings
        if _rule_for(rules, built, reading, grouping)
    ]
    items = [
        (reading, grouping)
        for grouping in scanned
        for reading in readings
        if (reading, grouping) not in set_by and _built_from_grouping(built, reading, grouping)
    ]
    # The leading-group card already compares the measure across the grouping,
    # unless the measure is money or a count, whose card gives shares instead.
    metric = metric_for(roles.measure, roles)
    measure_values = dataframe[roles.measure].dropna() if roles.measure in dataframe.columns else None
    compared_on_card = not metric.additive or not _counts_money_or_things(roles, metric, measure_values)
    covered = (roles.measure, roles.dimension) if roles.measure and compared_on_card else None
    skip = tuple(set_by + items + ([covered] if covered else []))
    scan = scan_differences(frame, readings, scanned, entity=entity, axis=found.axis, skip=skip)
    used_rules = [rule for rule in rules if rule.grouping in scanned or rule.grouping == roles.dimension]
    # A flag that only marks where the measure hit its cap explains it by definition,
    # as does a grouping cut from the measure or one of its items.
    flags = {grouping for reading, grouping in (scan.flagged if scan else ()) if reading == roles.measure}
    circular = {grouping for reading, grouping in [*set_by, *items] if reading == roles.measure}
    sharing = [grouping for grouping in groupings if grouping not in flags | circular]
    shares = (
        variance_explained(frame, roles.measure, sharing, entity=entity, axis=found.axis)
        if roles.measure in readings and len(sharing) > 1
        else ()
    )
    rule_text = " ".join(
        f"{rule.describe()}, so it is not compared with {_rule_subject(rule)}." for rule in used_rules
    )
    if scan is None or not (scan.named or shares or scan.compared >= MIN_NULL_COMPARISONS):
        if not used_rules:
            return None
        return Evidence(
            kind="differences",
            title="Groupings set by the readings",
            value=_count(len(used_rules), "rule"),
            statement=rule_text,
            calculation=_rules_calculation(used_rules) + note,
        )

    where = list_names(list(scan.groupings), limit=len(scan.groupings))
    comparisons = _count(scan.compared, "comparison")
    parts = []
    lead = ""
    if scan.named:
        first = scan.named[0]
        sentence = lead_sentence(first, frame[first.measure].dropna())
        lead = f"Largest difference found: {sentence}"
        parts.append(sentence)
        if scan.compared == 1:
            parts.append("It is the only comparison made.")
        elif scan.clear == 1:
            parts.append(
                f"It is the only clear finding among {comparisons} across {where}, {_adjustment(scan)}."
            )
        else:
            parts.append(
                f"It is the largest of {scan.clear} clear findings among {comparisons} across {where}, "
                f"{_adjustment(scan)}."
            )
        if first.paired:
            parts.append(
                f"Every {first.entity} appears in every {first.grouping} group, so each is compared with "
                f"itself ({first.test})."
            )
        elif first.shared_entities:
            parts.append(
                f"The same {first.entity} appears in more than one {first.grouping} group, so those "
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
        trivial = {
            "negligible": ": detectable, but too small to matter",
            "thin": f": it rests on a group of fewer than {MIN_NAMED_GROUP} values",
            "disputed": f": {largest.test} and {largest.check} disagree",
        }.get(largest.verdict, "")
        if scan.compared == 1:
            parts.append(
                f"The one comparison across {where} shows no clear difference: {largest.measure} by "
                f"{largest.grouping} ({effect_clause(largest)}{trivial})."
            )
        else:
            parts.append(
                f"None of the {comparisons} across {where} shows a clear difference once all tests are "
                f"allowed for; the largest is {largest.measure} by {largest.grouping} "
                f"({effect_clause(largest)}{trivial})."
            )
    if scan.correction == FALSE_DISCOVERY:
        expected = ALPHA * scan.tested
        parts.append(
            f"{scan.passing:,} of the {scan.tested:,} tests have an unadjusted p below 0.05, against about "
            f"{expected:,.0f} that chance alone would give."
        )
    if shares:
        unit = f", one value per {entity} in each group" if entity else ""
        parts.append(_variance_shares(roles.measure, shares, combined).rstrip(".") + f"{unit}.")
    if scan.entity:
        parts.append(f"Each {scan.entity} counts once per group.")
    if rule_text:
        parts.append(rule_text)
    if found.nested:
        within: dict[str, list[str]] = {}
        for finer, coarser in found.nested:
            within.setdefault(coarser, []).append(finer)
        nested = _joined_names(
            [f"{_joined_names(finer)} within {coarser}" for coarser, finer in within.items()]
        )
        parts.append(
            "Groupings nested in another, whose levels each occur with only some of its values, are not "
            f"compared on their own, since their differences would repeat it: {nested}."
        )

    method = "Holm-adjusted"
    if scan.correction == FALSE_DISCOVERY:
        method = "Benjamini-Hochberg false-discovery-rate adjusted"
    calculation = (
        f"each numeric reading compared across {where}: Welch's t-test or ANOVA on means (pooled variances "
        "when a group has fewer than 3 values), or Mann-Whitney or Kruskal-Wallis on ranks for a skewed "
        "or capped reading, each checked with the other; p-values from scipy.stats distributions, "
        f"{method} for {_count(scan.tested, 'test')} ({ADJUSTED_BY[scan.correction]}); a finding "
        "needs both tests to agree and groups of 5 or more values; ranked by share of variance explained "
        "(η², or ε² on ranks), those on under 20% of rows last"
    )
    if scan.trends:
        calculation += (
            f"; {_count(scan.trends, 'trend test')} (Spearman's ρ against the levels of each ordered "
            "numeric grouping) among them"
        )
    if scan.paired:
        calculation += (
            f"; across {list_names(list(scan.paired))}, where every {scan.entity} appears in every group, "
            "a repeated-measures ANOVA or Friedman test instead (paired t-test or Wilcoxon signed-rank "
            "test for two groups)"
        )
    if scan.entity:
        calculation += f"; one value per {scan.entity} in each group before testing"
    if found.axis and scan.entity:
        calculation += (
            f"; groups followed for different lengths of {found.axis} compared over the stretch they share"
        )
    if found.settings:
        calculation += (
            f"; {list_names(list(found.settings))} {'is' if len(found.settings) == 1 else 'are'} fixed "
            f"within each {entity}, so compared as groupings"
        )
    if scan.sampled:
        calculation += f"; on a fixed random sample of {SCAN_ROWS:,} of the {scan.rows:,} rows"
    if len(scan.readings) < scan.available:
        calculation += f"; the {len(scan.readings):,} most variable of {scan.available:,} readings"
    if covered and covered[0] in readings and covered[1] in scan.groupings and covered not in set_by + items:
        calculation += f"; {covered[0]} by {covered[1]} is compared on its own card"
    if set_by:
        calculation += f"; left out as set by the reading: {_pairs_text(set_by)}"
    if items:
        calculation += f"; left out as a score compared with its own items: {_pairs_text(items)}"
    for copy, kept in copies:
        calculation += f"; {copy} is {kept} recorded again, so only {kept} is compared"
    if scan.attributes:
        calculation += (
            "; left out as fixed within each group (an attribute of the group, not a difference "
            f"between its members): {_pairs_text(list(scan.attributes))}"
        )
    if scan.placed:
        calculation += (
            f"; {list_names(list(scan.placed))} not compared: coordinates place a row and compass "
            "directions have no linear mean"
        )
    for reading, grouping in scan.flagged:
        calculation += (
            f"; {reading} by {grouping} left out: {grouping} only marks where {reading} hits its cap"
        )
    if scan.recoded:
        calculation += (
            "; left out as codes for the grouping, with one value per group: "
            + _pairs_text(list(scan.recoded))
        )
    for grouping, (first_part, second_part) in combined.items():
        calculation += (
            f"; {grouping} combines {first_part} and {second_part}, so those two are compared instead"
        )
    for column, part in found.date_parts:
        calculation += f"; {column} is the {part} of {roles.date}, so it is not used as a grouping"
    if found.capped:
        calculation += (
            f"; {len(found.chosen)} groupings only"
            + (f", the codes that go with {roles.measure} most kept" if roles.measure else "")
            + f", so {list_names(list(found.capped))} {'is' if len(found.capped) == 1 else 'are'} not scanned"
        )
    if shares:
        calculation += f"; variance shares are η² of {roles.measure} for each grouping alone"
    calculation += note
    return Evidence(
        kind="differences",
        title="What differs between groups",
        value=f"{scan.clear:,} of {scan.compared:,}",
        statement=" ".join(parts),
        calculation=calculation,
        subject=scan.named[0].measure if scan.named else None,
        lead=lead,
    )


def _rule_subject(rule: GroupingRule) -> str:
    """What a grouping set by a rule is not compared with: the reading, or the columns it picks between."""
    if rule.kind == "largest":
        return f"{_joined_names(list(rule.readings))} or anything built from them"
    return f"{rule.readings[0]} or anything built from it"


def _joined_names(names: list[str]) -> str:
    return list_names(names, limit=len(names))


def _rules_calculation(rules: list[GroupingRule]) -> str:
    """The calculation behind the rules: bands of a reading, or the largest of several, row by row."""
    steps = []
    for rule in rules:
        if rule.kind == "largest":
            steps.append(
                f"{rule.grouping} checked against the largest of {_joined_names(list(rule.readings))} "
                f"on each of {rule.rows:,} rows"
            )
        else:
            within = f" within each {rule.by}" if rule.by else ""
            steps.append(
                f"cut points of {rule.readings[0]}{within} that place the most of {rule.rows:,} rows "
                f"in their {rule.grouping} group"
            )
    return "; ".join(steps)


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
# A within-group r this far from the pooled one, and beyond its sampling noise,
# changes what the pooled figure means, whichever way it moves.
WITHIN_GROUP_GAP = 0.1
# With this many groups or fewer, each group's own r is listed.
WITHIN_GROUPS_LISTED = 3
# Records far out in a column, mostly in one unbroken stretch of at least this
# many, are a phase of the run rather than a few stray values.
STRETCH_MIN_RECORDS = 10
STRETCH_SHARE = 0.8


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

    The checks say whether one record, a few extreme ones, a long stretch of
    them or a pile-up at a floor decide the figure: r without the record whose
    removal moves it most (and that record's two values), r without records
    far outside either column, how many records lie that far out and how many
    of them run unbroken, and the largest pile-up at a column's minimum or
    maximum.
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
    #: Records more than EXTREME_Z robust SDs out in either column, and the longest unbroken run of them.
    far_out: int = 0
    stretch: int = 0
    #: (column, "minimum" or "maximum", value, records) for the largest pile-up at a column's edge.
    pile: tuple[str, str, float, int] | None = None

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


def _pile(paired: pd.DataFrame) -> tuple[str, str, float, int] | None:
    """The largest pile-up of equal values at a column's minimum or maximum: a floor or a ceiling."""
    found = None
    for column in paired.columns[:2]:
        values = paired[column]
        if values.nunique() < CAP_MIN_DISTINCT:
            continue
        for edge, value in (("minimum", values.min()), ("maximum", values.max())):
            count = int((values == value).sum())
            if count >= max(CAP_MIN_ROWS, CAP_SHARE * len(values)) and (found is None or count > found[3]):
                found = (str(column), edge, float(value), count)
    return found


def measure_correlation(paired: pd.DataFrame) -> Correlation | None:
    """Correlate the two columns of ``paired``, which holds only complete rows, in row order.

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
    outside = far_out(x, y)
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
        far_out=int(outside.sum()) if outside is not None else 0,
        stretch=longest_stretch(outside) if outside is not None else 0,
        pile=_pile(paired),
    )


def _relationship_words(value: float) -> str:
    """ "move together", "move in opposite directions only weakly", or "show no linear relationship"."""
    if round(value, 2) == 0:
        return "show no linear relationship"
    words = "move together" if value > 0 else "move in opposite directions"
    return words + (" only weakly" if abs(value) < MATERIAL_CORRELATION else "")


def _interval_note(value: float, low: float, high: float, pairs: int) -> str:
    """A sentence when the interval spans zero: a thin sample for a strong r, plain weakness otherwise."""
    if not low <= 0.0 <= high:
        return ""
    if abs(value) >= MATERIAL_CORRELATION:
        return f" With only {pairs:,} paired values this cannot be told apart from no relationship at all."
    return " The interval includes zero, so this cannot be told apart from no relationship at all."


def _rank_gap_cause(correlation: Correlation) -> str:
    """Why the linear figure is well above the rank one: a floor, a long stretch of records, a few extremes.

    Ties at a floor or a ceiling lower the rank figure; a phase of the run in
    which both columns are far higher raises the linear one. Only when the
    extreme records are few and scattered are they "a few extreme records".
    """
    trimmed = correlation.without_extremes
    few = trimmed is not None and abs(correlation.pearson) - abs(trimmed) > INFLUENCE_CHANGE
    causes = []
    if few:
        causes.append(
            f"a few extreme records drive the linear figure (without the {correlation.extremes} most "
            f"extreme, r is {trimmed:.2f})"
        )
    if correlation.stretch >= max(STRETCH_MIN_RECORDS, STRETCH_SHARE * correlation.far_out):
        causes.append(
            f"one unbroken stretch of {correlation.stretch:,} records lies far beyond the rest and sets "
            "the linear figure"
        )
    if correlation.pile is not None:
        column, edge, value, count = correlation.pile
        causes.append(
            f"{column} sits at its {edge} of {format_number(value, column, compact=False)} in "
            f"{count:,} records, which tie in rank"
        )
    if few and len(causes) == 1:
        return (
            f" A few extreme records are driving the linear figure: without the {correlation.extremes} "
            f"most extreme, r is {trimmed:.2f}."
        )
    if causes:
        return f" The two figures differ because {' and '.join(causes)}."
    if correlation.far_out <= max(3, 0.05 * correlation.pairs):
        return " A few extreme records are driving the linear figure."
    return (
        f" {correlation.far_out:,} records lie far beyond the rest in one column or both and drive the "
        "linear figure."
    )


def describe_correlation(correlation: Correlation, scope: str = "") -> str:
    """State r with its interval, leading with the figure that one record or a few extremes don't decide.

    The brief and Ask both use this, so a pair reads the same in either. ``scope``
    follows the column names, as in " for Site in (North)". When one record
    or a few extreme ones move r far, the sentence leads with r without them
    and says whether they hide the relationship or make it.
    """
    value, pairs = correlation.pearson, correlation.pairs
    pair = f"{correlation.first} and {correlation.second}"
    trimmed = correlation.without_extremes

    # One record that moves r this far decides the figure on its own, whichever
    # way it moves it; that says more than any comparison with the ranks.
    if correlation.rests_on_one_record and correlation.influential is not None:
        robust = float(correlation.without_influential)
        low, high = correlation_interval(robust, pairs - 1)
        first_value, second_value = correlation.influential
        record = (
            f"the one where {correlation.first} is "
            f"{format_number(first_value, correlation.first, compact=False)} and {correlation.second} is "
            f"{format_number(second_value, correlation.second, compact=False)}"
        )
        statement = (
            f"{pair} {_relationship_words(robust)}{scope} once one record is set aside: Pearson r = "
            f"{robust:.2f} without it (95% CI {low:.2f} to {high:.2f}, n = {pairs - 1:,})."
            + _interval_note(robust, low, high, pairs - 1)
        )
        if abs(robust) > abs(value):
            hidden = "a strong relationship" if abs(robust) >= MATERIAL_CORRELATION else "the relationship"
            return statement + f" One record hides {hidden}: with {record}, r is only {value:.2f}."
        return statement + f" It rests on one record: with {record}, r is {value:.2f}."

    # Spearman well above Pearson because extreme records hide a straight line.
    if (
        correlation.rank_divergence > RANK_DIVERGENCE
        and trimmed is not None
        and abs(trimmed) - abs(value) > RANK_DIVERGENCE
    ):
        kept = pairs - correlation.extremes
        low, high = correlation_interval(trimmed, kept)
        return (
            f"{pair} {_relationship_words(trimmed)}{scope} once {correlation.extremes} extreme records "
            f"are set aside: Pearson r = {trimmed:.2f} without them (95% CI {low:.2f} to {high:.2f}, "
            f"n = {kept:,}), and the rank correlation is {correlation.spearman:.2f} on all {pairs:,}. "
            f"The extreme records hide the relationship: with them, Pearson r is only {value:.2f}."
        )
    figures = (
        f"Pearson r = {value:.2f} (95% CI {correlation.low:.2f} to {correlation.high:.2f}, n = {pairs:,})"
    )
    ranked = _relationship_words(correlation.spearman)
    # Spearman well above Pearson otherwise: a consistent relationship that is not a straight line.
    if correlation.rank_divergence > RANK_DIVERGENCE:
        return (
            f"{pair} {ranked}{scope} in rank: the rank correlation is {correlation.spearman:.2f}, while "
            f"{figures}. The relationship is consistent but not a straight line."
        )
    # Pearson well above Spearman: the linear figure comes from part of the records.
    if correlation.rank_divergence < -RANK_DIVERGENCE:
        return (
            f"{pair} {ranked}{scope} by rank: the rank correlation is only {correlation.spearman:.2f}, "
            f"while {figures}." + _rank_gap_cause(correlation)
        )
    return (
        f"{pair} {_relationship_words(value)}{scope}: {figures}."
        + _interval_note(value, correlation.low, correlation.high, pairs)
    )


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
        and column not in roles.non_readings
        and not looks_like_identifier(column, dataframe[column])
        and not is_design_name(column)
        and not is_entity_name(column)
        and not is_year_name(column, dataframe[column])
        and not is_month_name(column, dataframe[column])
        and not any(token == normalized_name(column) for token in TIME_PART_TOKENS)
        and not _is_flag(dataframe[column])
        and not _codes_a_grouping(dataframe, column, text_groupings)
    ]


def _differs_within(pooled: float, within: float, pairs: int) -> bool:
    """Whether r within the groups differs from the pooled r in sign, or by WITHIN_GROUP_GAP beyond noise."""
    if pooled * within < 0 and min(abs(pooled), abs(within)) >= 0.1:
        return True
    if abs(pooled - within) < WITHIN_GROUP_GAP or pairs <= 3:
        return False
    clip = 1 - 1e-9
    gap = abs(np.arctanh(np.clip(within, -clip, clip)) - np.arctanh(np.clip(pooled, -clip, clip)))
    return bool(gap > 2 / np.sqrt(pairs - 3))


# A pooled r of at least NOTABLE_CORRELATION that falls below this within every group came from mixing them.
VANISHED_CORRELATION = 0.1


def _vanishes(pooled: float, within: float) -> bool:
    return abs(pooled) >= NOTABLE_CORRELATION and abs(within) < VANISHED_CORRELATION


def _within_sentence(within: WithinGroups, pooled: float) -> str:
    group = sentence_case(within.grouping)
    removed = f"{within.within:.2f} once each {group}'s own means are removed"
    if len(within.per_group) <= WITHIN_GROUPS_LISTED:
        listed = _joined_names(
            [f"{r:.2f} in {group_label(within.grouping, level)}" for level, r, _ in within.per_group]
        )
        spread = f"r = {listed} ({removed})"
    else:
        spread = f"r = {removed} (from {within.low:.2f} to {within.high:.2f} across {within.groups} groups)"
    if _vanishes(pooled, within.within):
        return (
            f" Within each {group} it all but vanishes: {spread}, so the pooled figure comes from the "
            f"difference between the {group} groups, not from how the two move together within one."
        )
    if pooled * within.within < 0:
        return (
            f" Within each {group} the relationship runs the other way: {spread}, so the pooled "
            f"figure reflects differences between the {group} groups."
        )
    if abs(within.within) > abs(pooled):
        return (
            f" Within each {group} it is stronger: {spread}; pooling groups at different levels weakens it."
        )
    return (
        f" Within each {group} it is weaker: {spread}, so part of the pooled figure comes from "
        f"differences between the {group} groups."
    )


def _labels(dataframe: pd.DataFrame, roles: ColumnRoles) -> list[str]:
    """Groupings with a few levels: the grouping column, the text labels and the codes."""
    return [
        column
        for column in dict.fromkeys(
            [*([roles.dimension] if roles.dimension else []), *roles.dimensions, *roles.factors]
        )
        if column in dataframe.columns and 2 <= dataframe[column].nunique() <= MAX_SCAN_LEVELS
    ]


def _strongest_label(dataframe: pd.DataFrame, roles: ColumnRoles, column: str) -> str | None:
    """The grouping that explains most of ``column``, when it explains at least STRATUM_EXPLAINED."""
    shares = [
        (explained_share(dataframe[column], dataframe[label]), label)
        for label in _labels(dataframe, roles)
        if label != column
    ]
    best = max(shares, default=(0.0, None))
    return best[1] if best[0] >= STRATUM_EXPLAINED else None


def _within_group_note(
    dataframe: pd.DataFrame, roles: ColumnRoles, correlation: Correlation, skip: frozenset[str] = frozenset()
) -> str:
    """Sentences when r inside the groups differs from the pooled r, in sign or clearly in size, either way.

    The grouping column is always checked and reported when it differs;
    another grouping is added when it differs more. The text labels are
    checked, and so is any code that explains STRATUM_EXPLAINED of either
    column (a 0/1 label the readings differ by), since mixing groups at
    different levels makes a correlation of its own. ``skip`` holds groupings
    set by either column or built into it: cutting a column into bands
    narrows its range within each, which weakens r there by itself, so such
    a grouping is reported only when r all but vanishes within it.
    """
    first, second = correlation.first, correlation.second
    screen = dataframe.sample(SCAN_ROWS, random_state=0) if len(dataframe) > SCAN_ROWS else dataframe
    groupings = [roles.dimension] if roles.dimension else []
    groupings += [
        column
        for column in _labels(dataframe, roles)
        if column in roles.dimensions and column not in roles.factors
        or max(
            explained_share(screen[first], screen[column]),
            explained_share(screen[second], screen[column]),
        )
        >= STRATUM_EXPLAINED
    ]
    pooled = correlation.pearson
    differing = []
    for grouping in dict.fromkeys(groupings):
        if grouping not in dataframe.columns or grouping in (first, second):
            continue
        within = within_group_correlation(dataframe, first, second, grouping)
        if within is None:
            continue
        if grouping in skip:
            if _vanishes(pooled, within.within):
                differing.append(within)
        elif _differs_within(pooled, within.within, correlation.pairs):
            differing.append(within)
    if not differing:
        return ""
    main = next((within for within in differing if within.grouping == roles.dimension), None)
    largest = max(differing, key=lambda within: abs(pooled - within.within))
    chosen = [main] if main is not None else []
    if largest is not main:
        chosen.append(largest)
    return "".join(_within_sentence(within, pooled) for within in chosen)


def _scanned_pair(
    dataframe: pd.DataFrame,
    roles: ColumnRoles,
    pair: Pair,
    rules: tuple[GroupingRule, ...] = (),
    built: Construction | None = None,
) -> tuple[Correlation, str]:
    """The full correlation of a pair the scan picked, over every row, with its within-group sentence."""
    correlation = measure_correlation(dataframe[[pair.first, pair.second]].dropna())
    assert correlation is not None
    skip = frozenset(
        grouping
        for grouping in (roles.dimension, *roles.dimensions)
        if grouping
        and any(
            _rule_for(rules, built, column, grouping) or _built_from_grouping(built, column, grouping)
            for column in (pair.first, pair.second)
        )
    )
    return correlation, _within_group_note(dataframe, roles, correlation, skip)


def _pair_numbers(dataframe: pd.DataFrame, pair: Pair, sampled: bool) -> tuple[float, float, int]:
    """Pearson r, Spearman r and n for a pair, over every row when the scan used a sample."""
    if not sampled:
        return pair.pearson, pair.spearman, pair.pairs
    paired = dataframe[[pair.first, pair.second]].dropna()
    return float(paired.corr().iloc[0, 1]), float(paired.corr(method="spearman").iloc[0, 1]), len(paired)


def _figures(dataframe: pd.DataFrame, pair: Pair, sampled: bool) -> str:
    """ "r = 0.83, Spearman 0.74, n = 2,435" for a pair."""
    pearson, spearman, pairs = _pair_numbers(dataframe, pair, sampled)
    return f"r = {pearson:.2f}, Spearman {spearman:.2f}, n = {pairs:,}"


# Why a column paired with the measure was set aside, by the scan's reason.
SET_ASIDE_REASONS = {
    "copy": "the same measurement under another name or unit",
    "built": "built from it, or it from them",
    "flag": "marking where it hits its cap",
    "total": "adding up to a fixed total with it",
    "place": "map coordinates like it",
}


def set_aside_note(scan: RelationshipScan) -> str:
    """A sentence naming the columns left out of the measure's ranking, and why, or ""."""
    if not scan.measure or not scan.set_aside:
        return ""
    by_reason: dict[str, list[str]] = {}
    for column, reason in scan.set_aside:
        by_reason.setdefault(reason, []).append(column)
    clauses = [
        f"{list_names(columns)} ({SET_ASIDE_REASONS.get(reason, reason)})"
        for reason, columns in by_reason.items()
    ]
    return f" Left out of the ranking for {scan.measure}: {'; '.join(clauses)}."


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
        notes.append("pairs of columns built from one another (" + "; ".join(named) + ")")
    notes.extend(f"{flag.describe()}" for flag in scan.flags)
    if scan.shared_stems:
        named = [f"{first} and {second}" for first, second, _ in scan.shared_stems[:NEAR_DUPLICATES_NAMED]]
        notes.append(f"same-stem pairs above |r| = {SHARED_STEM_CORRELATION} ({', '.join(named)})")
    if scan.near_duplicates:
        notes.append(f"near-duplicates (|r| ≥ {NEAR_DUPLICATE_CORRELATION})")
    if scan.coordinates:
        named = [f"{first} and {second}" for first, second in scan.coordinates[:NEAR_DUPLICATES_NAMED]]
        notes.append(f"pairs of map coordinates, which describe where rows were taken ({', '.join(named)})")
    if scan.angles:
        notes.append(
            f"{_joined_names(list(scan.angles))}, compass directions, whose 0 and 360 are the same place"
        )
    return notes


def _construction(dataframe: pd.DataFrame, roles: ColumnRoles, readings: list[str]) -> Construction | None:
    """How the readings are built from one another and from numeric codes, screened once for both scans.

    Rating items read as codes may be what a score is the mean of, so they are
    tried as parts. A row mean added to a matrix table is built from its columns.
    """
    items = [column for column in roles.factors if column in dataframe.columns and column not in readings]
    block = row_summary_columns(roles, dataframe)
    known = (
        (Derivation(roles.measure, tuple(block), (1 / len(block),) * len(block), 0.0, "mean"),)
        if block and roles.measure
        else ()
    )
    measure = roles.measure if roles.measure in readings else None
    return construction(dataframe, readings, measure, items=items, known=known)


def _relationship_scan(
    dataframe: pd.DataFrame,
    roles: ColumnRoles,
    columns: list[str] | None = None,
    built: Construction | None = None,
) -> RelationshipScan | None:
    """Every pair of numeric readings screened, with the measure's correlates first."""
    columns = numeric_readings(dataframe, roles) if columns is None else columns
    if len(columns) < 2:
        return None
    measure = roles.measure if roles.measure in columns else None
    if built is None:
        built = _construction(dataframe, roles, columns)
    return scan_relationships(dataframe, columns, measure, built=built)


def _relationship_evidence(
    dataframe: pd.DataFrame,
    roles: ColumnRoles,
    columns: list[str] | None = None,
    scan: RelationshipScan | None = None,
    note: str = "",
    *,
    built: Construction | None = None,
    rules: tuple[GroupingRule, ...] = (),
) -> Evidence | None:
    """What relates to the measure, then the strongest other pair, or a plain null result.

    Every pair is screened at once, on a fixed sample of a very long table;
    pairs related by construction are set aside and listed in the
    calculation, and the ranking discounts a correlation that one record
    makes or hides. The pairs reported are computed on every row. ``scan`` is
    the screen when it has been run already, with ``built`` the screen of how
    the readings are built from one another, and ``note`` is added to the
    calculation.
    """
    if scan is None:
        columns = numeric_readings(dataframe, roles) if columns is None else columns
        if built is None and len(columns) >= 2:
            built = _construction(dataframe, roles, columns)
        scan = _relationship_scan(dataframe, roles, columns, built)
    if scan is None or not (scan.correlates or scan.others):
        return None
    measure = scan.measure
    frame = built.masked(dataframe) if built is not None else dataframe

    parts: list[str] = []
    value = None
    reported = 0
    headline = ""
    # A measure made of every other column (a row mean) has nothing left to relate to.
    whole = (
        scan.measure is not None
        and not scan.correlates
        and scan.measure_tests > 0
        and len(scan.set_aside) == scan.measure_tests
    )
    if whole:
        derivation = built.derivation(measure) if built is not None else None
        made = f" ({derivation.describe()})" if derivation is not None else ""
        parts.append(
            f"Every column tested against {measure} is part of it{made}, so none is said to relate to it."
        )
    elif scan.measure:
        notable = [pair for pair in scan.correlates if pair.strength >= NOTABLE_CORRELATION]
        if notable:
            lead, within = _scanned_pair(frame, roles, notable[0], rules, built)
            value = f"r = {lead.pearson:.2f}"
            reported += 1
            parts.append(f"{describe_correlation(lead)}{within}")
            if not lead.low <= 0 <= lead.high:
                headline = _first_sentence(describe_correlation(lead))
            if scan.measure_tests > 1:
                parts.append(
                    f"Of the {scan.measure_tests:,} columns tested against {measure}, "
                    f"{notable[0].second} relates most.{_chance_note(notable[0], scan.measure_tests)}"
                    + set_aside_note(scan)
                )
            if len(notable) > 1:
                parts.append(
                    "Next: "
                    + ", ".join(
                        f"{pair.second} ({_figures(frame, pair, scan.sampled)})"
                        for pair in notable[1:CORRELATES_NAMED]
                    )
                    + "."
                )
        elif scan.correlates:
            closest = scan.correlates[0]
            figures = _figures(frame, closest, scan.sampled)
            parts.append(
                f"{closest.second} does not relate to {measure}: {figures}."
                if scan.measure_tests == 1
                else f"None of the {scan.measure_tests:,} columns tested against {measure} reaches "
                f"|r| = {NOTABLE_CORRELATION}; the closest is {closest.second} ({figures})."
            )
    others = [pair for pair in scan.others if pair.strength >= NOTABLE_CORRELATION]
    if others:
        pair, within = _scanned_pair(frame, roles, others[0], rules, built)
        # The card's figure answers its title, which names the measure when there is one.
        if not scan.measure or whole:
            value = f"r = {pair.pearson:.2f}"
        reported += 1
        lead_in = "Among the other columns, the strongest pair: " if scan.measure else ""
        chance = _chance_note(others[0], scan.pair_tests)
        parts.append(f"{lead_in}{describe_correlation(pair)}{within}{chance}")
        if not scan.measure and not chance and not pair.low <= 0 <= pair.high:
            headline = _first_sentence(describe_correlation(pair))
    elif scan.others:
        strongest = scan.others[0]
        pearson, _, pairs = _pair_numbers(frame, strongest, scan.sampled)
        parts.append(
            f"No {'other ' if scan.measure else ''}pair reaches |r| = {NOTABLE_CORRELATION}; the "
            f"strongest is {strongest.first} and {strongest.second} (r = {pearson:.2f}, n = {pairs:,})."
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
    # Rows that break a block of shares, and placeholder zeros, are named once, here.
    for item in scan.derived:
        parts.extend(item.exceptions())

    calculation = (
        "Pearson r over each pair's complete rows, 95% CI by Fisher transform, and Spearman r; "
        + (f"{_count(scan.measure_tests, 'correlation')} with {measure} and " if scan.measure else "")
        + f"{_count(scan.pair_tests, 'pair')} scanned in all, ranked by the larger of |Spearman r| "
        "and |Pearson r| without the pair's most influential record; intervals not adjusted for "
        "that selection"
    )
    if scan.sampled:
        calculation += (
            f"; pairs screened on a fixed random sample of {SCAN_ROWS:,} of the {scan.rows:,} rows, and "
            "the pairs named computed on every row"
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
        title=(
            f"What relates to {measure}"
            if scan.measure and not whole
            else "Strongest measurable relationship"
        ),
        value=value,
        statement=" ".join(parts),
        calculation=calculation + note,
        lead=headline,
    )


# Last words of yes/no columns that describe who or what was studied, how it
# was assigned, or whether the procedure worked, rather than an outcome: the
# share of women among smokers is the sample's makeup, not a finding about smoking.
NOT_OUTCOME_WORDS = frozenset({
    "sex", "gender", "race", "ethnicity", "treatment", "arm", "cohort", "condition", "group",
    "block", "batch", "phase", "site", "visit", "check",
})


def _yes_no_columns(dataframe: pd.DataFrame, roles: ColumnRoles) -> list[str]:
    """Grouping and flag columns that hold a yes/no outcome, in table order."""
    candidates = {roles.dimension, *roles.dimensions, *roles.factors}
    candidates |= {
        column for column in roles.numeric if column in dataframe.columns and _is_flag(dataframe[column])
    }
    return [
        column
        for column in dataframe.columns
        if column in candidates
        and _last_word(column) not in NOT_OUTCOME_WORDS
        and outcome_event(dataframe[column]) is not None
    ]


def _last_word(name: str) -> str:
    """The last word of a column name, which says what the column is."""
    words = re.findall(r"[a-z]+", normalized_name(name))
    return words[-1] if words else ""


def _rate_evidence(
    dataframe: pd.DataFrame, roles: ColumnRoles, note: str = "", found: _Groupings | None = None
) -> Evidence | None:
    """How often a yes/no outcome occurs in each group of another column, when it differs clearly.

    The outcome is the grouping column when that is yes/no, and otherwise the
    last yes/no column in the table, where outcomes usually sit. It is
    compared across every other grouping, with p-values Holm-adjusted for the
    number of groupings. A grouping in which every group is all yes or all no,
    or one named for the same thing, is another code for the outcome and is
    left out. When the outcome is fixed for each subject (a cell line's
    methylation flag on each of its drug rows), each subject counts once. A
    grouping tied to the rate's grouping that goes with the outcome too (kidney
    disease behind both hypertension and anaemia) gets the rate compared within
    each of its levels, which leads when the gap does not hold there.
    """
    flags = _yes_no_columns(dataframe, roles)
    if not flags:
        return None
    outcome = roles.dimension if roles.dimension in flags else flags[-1]
    found = found or _find_groupings(dataframe, roles)
    # A grouping named for the same thing as the outcome is the outcome recorded another way.
    # Counting rates is cheap, so groupings past the scan's cap are tested too.
    groupings = [
        grouping
        for grouping in [*found.chosen, *found.capped]
        if grouping != outcome
        and not same_stem(grouping, outcome)
        and not _same_partition(dataframe, grouping, outcome)
    ]
    tested = [
        comparison
        for comparison in (
            compare_rates(dataframe, outcome, grouping, entity=found.entity) for grouping in groupings
        )
        if comparison is not None and not all(group.events in (0, group.n) for group in comparison.groups)
    ]
    if not tested:
        return None
    adjusted = holm_adjust(np.array([comparison.p_value for comparison in tested]))
    tested = [
        replace(comparison, adjusted_p=float(p), tests=len(tested)) if len(tested) > 1 else comparison
        for comparison, p in zip(tested, adjusted, strict=True)
    ]
    clear = sorted(
        (comparison for comparison in tested if comparison.verdict == "clear"),
        key=lambda comparison: comparison.cramers_v,
        reverse=True,
    )
    if not clear:
        return None
    first = clear[0]
    statement = describe_rates(first)
    lead = _first_sentence(statement)
    calculation = rate_calculation(first)
    candidates = [
        column for column in _stratum_candidates(dataframe, roles, found) if column != outcome
    ]
    stratified = stratified_rates(
        dataframe, outcome, first.grouping, candidates, entity=found.entity, prefer=roles.dimension
    )
    if stratified is not None:
        statement += describe_stratified(stratified, first)
        calculation += f"; compared again within each {stratified.stratum}" + _sample_note(stratified.sampled)
        if stratified.overturns(first):
            lead = stratified_lead(stratified, first)
    if len(tested) > 1:
        statement += (
            f" It is the clearest of {len(clear)} clear differences in the rate across "
            f"{_count(len(tested), 'grouping')}."
            if len(clear) > 1
            else f" Of the {len(tested)} groupings tested, it is the only one where the rate differs clearly."
        )
    if len(clear) > 1:
        statement += " Next: " + "; ".join(
            f"by {comparison.grouping} ({comparison.higher.rate:.1%} in {comparison.higher.label} vs "
            f"{comparison.lower.rate:.1%} in {comparison.lower.label}, "
            f"Cramér's V = {comparison.cramers_v:.2f})"
            for comparison in clear[1:DIFFERENCES_NAMED]
        ) + "."
    return Evidence(
        kind="rates",
        title=f"Rate of {first.event_label} by {sentence_case(first.grouping)}",
        value=f"{first.higher.rate:.0%} vs {first.lower.rate:.0%}",
        statement=statement,
        calculation=calculation + note,
        lead=lead,
    )


# A joint model is reported when it explains this much more of the measure than its best reading alone.
MODEL_GAIN = 0.1


def _coefficient(value: float) -> str:
    return f"{value:+.3g}"


def _joint_model_evidence(
    dataframe: pd.DataFrame, roles: ColumnRoles, scan: RelationshipScan | None, note: str = ""
) -> Evidence | None:
    """The measure fitted on the readings that relate to it, when the fit says more than any one of them.

    The predictors are the measure's correlates with p below 0.05, at most
    MAX_MODEL_TERMS. The card appears when the readings together explain
    clearly more than the best one alone, or when a reading's coefficient
    points the other way from its own correlation. It is left out when a label
    the measure differs by (a 0/1 outcome, a site type) takes away what the
    readings explain: once its levels are allowed for, they add less than
    MODEL_GAIN, so the fit only restates the gap between its groups.
    """
    if scan is None or not scan.measure:
        return None
    predictors = [pair.second for pair in scan.correlates if pair.p_value < ALPHA][:MAX_MODEL_TERMS]
    model = fit_joint_model(dataframe, roles.measure, predictors)
    if model is None:
        return None
    best = model.best_single
    gain = model.r2 - best.pairwise**2
    if gain < MODEL_GAIN and not model.reversed:
        return None
    label = _strongest_label(dataframe, roles, roles.measure)
    if label is not None:
        alone, together = gain_within(dataframe, roles.measure, list(model.predictors), label)
        if together - alone < MODEL_GAIN <= model.r2 - (together - alone):
            return None
    names = list_names(list(model.predictors), limit=len(model.predictors))
    equation = f"{roles.measure} = {model.intercept:.3g}" + "".join(
        f" {'+' if term.coefficient >= 0 else '−'} {abs(term.coefficient):.3g} × {term.name}"
        for term in model.terms
    )
    statement = (
        f"Together, {names} explain {model.r2:.0%} of the variance in {roles.measure} (adjusted R² = "
        f"{model.adjusted_r2:.2f}, n = {model.n:,}), against {best.pairwise**2:.0%} for {best.name} alone: "
        f"{equation}. With the others held fixed: "
        + "; ".join(
            f"{term.name} {_coefficient(term.coefficient)} per unit (95% CI {term.low:.3g} to "
            f"{term.high:.3g}, {format_p(term.p_value)})"
            for term in model.terms
        )
        + "."
    )
    for term in model.reversed:
        direction = "higher" if term.coefficient > 0 else "lower"
        statement += (
            f" With the others held fixed, more {term.name} goes with {direction} {roles.measure}, though on "
            f"its own it correlates the other way (r = {term.pairwise:.2f}): that correlation partly stands "
            "for the readings that travel with it."
        )
    if model.collinear:
        verb = "is" if len(model.collinear) == 1 else "are"
        statement += (
            f" {list_names(list(model.collinear))} {verb} largely a mix of the other predictors (variance "
            f"inflation factor above {MAX_VIF:.0f}), so the separate coefficients are unstable."
        )
    statement += " A coefficient is an association with the others held fixed, not evidence of cause."
    calculation = (
        f"ordinary least squares of {roles.measure} on the readings that relate to it with p < 0.05 (at "
        f"most {MAX_MODEL_TERMS}), over the {model.n:,} rows with all of them recorded (statsmodels OLS); "
        "95% CIs from the t distribution; variance inflation factors from statsmodels "
        "variance_inflation_factor"
    )
    if model.rows > model.n:
        calculation += f"; on a fixed random sample of {model.n:,} of the {model.rows:,} rows"
    lead = (
        f"Together, {names} explain {model.r2:.0%} of the variance in {roles.measure} (n = {model.n:,}), "
        f"against {best.pairwise**2:.0%} for {best.name} alone."
    )
    if model.reversed and gain < MODEL_GAIN:
        term = model.reversed[0]
        lead = (
            f"With the other readings held fixed, {term.name}'s link to {roles.measure} reverses: "
            f"r = {term.pairwise:.2f} on its own, coefficient {_coefficient(term.coefficient)} in a joint "
            "model."
        )
    return Evidence(
        kind="model",
        title=f"Joint model of {roles.measure}",
        value=f"R² = {model.r2:.2f}",
        statement=statement,
        calculation=calculation + note,
        lead=lead,
    )


def _level_shift_evidence(
    dataframe: pd.DataFrame, roles: ColumnRoles, frequency: str | None = None
) -> Evidence | None:
    """The largest lasting change in the level of the measure's period series, when there is one."""
    if not roles.date or not roles.measure:
        return None
    series = build_trend(dataframe, roles, frequency)
    trend = observed_periods(series.frame)
    found = level_shift(trend["Value"].to_numpy(dtype=float))
    if found is None:
        return None
    periods = pd.DatetimeIndex(pd.to_datetime(trend["Period"]))
    grain = series.frequency
    measure_values = dataframe[roles.measure].dropna()
    before, after = format_values_apart(
        found.before, found.after, column=roles.measure, column_values=measure_values
    )
    start = format_period(periods[found.position], grain)
    last = format_period(periods[found.position - 1], grain)
    first, end = format_period(periods[0], grain), format_period(periods[-1], grain)
    verb = "rose" if found.after > found.before else "fell"
    size = format_number(abs(found.after - found.before), roles.measure, column_values=measure_values)
    per = f"per {GRAIN_NAMES.get(grain, 'period')}"
    statement = (
        f"{roles.measure} {verb} to a new level from {start}: it averaged {before} {per} from {first} "
        f"to {last} and {after} from {start} to {end}, a shift of {size}, {found.effect:.1f} times the "
        "spread within the two stretches. A lasting shift like this usually follows a change in method, "
        "instrument or conditions."
    )
    statement += _shift_groups_note(dataframe, roles, grain, periods[found.position])
    return Evidence(
        kind="shift",
        title="Change in level",
        value=start,
        statement=statement,
        calculation=(
            "binary segmentation of the period values into two stretches with the least squared error "
            "(ruptures Binseg, l2 cost), each at least 15% of the series; kept when its t statistic, "
            "discounted for autocorrelation, passes the threshold at which only 1 stable series in 20 "
            "raises a flag, the two-level fit beats a straight line by a BIC-like margin, and the shift is "
            f"at least {MIN_SHIFT_EFFECT:g} within-stretch standard deviations"
        ),
        tone="warning",
        lead=(
            f"{roles.measure} {verb} to a new level from {start}: from {before} to {after} {per} on average."
        ),
    )


def _shift_groups_note(dataframe: pd.DataFrame, roles: ColumnRoles, grain: str, start: pd.Timestamp) -> str:
    """Say when the groups reporting before and after a shift differ, since that alone can move the level."""
    if not roles.dimension:
        return ""
    stamps = dataframe[roles.date].dt.to_period(grain).dt.to_timestamp()
    recorded = dataframe[roles.measure].notna()
    before = set(dataframe.loc[recorded & (stamps < start), roles.dimension].dropna())
    after = set(dataframe.loc[recorded & (stamps >= start), roles.dimension].dropna())
    stopped, started = before - after, after - before
    if not stopped and not started:
        return ""
    changes = []
    if stopped:
        changes.append(f"{_listed(tuple(sorted(stopped, key=str)), roles.dimension)} stopped reporting")
    if started:
        changes.append(f"{_listed(tuple(sorted(started, key=str)), roles.dimension)} started")
    return (
        f" The {sentence_case(roles.dimension)} groups changed at the same time ({'; '.join(changes)}), "
        "which may explain part of it."
    )


#: Share of zeros at which a measure counts as heavy-tailed and no outliers are listed.
HEAVY_TAIL_ZEROS = 0.5
#: Values a group needs for its own quartiles; smaller groups use the pooled ones.
MIN_GROUP_FOR_FENCES = 12
#: Share of the flagged values that one unbroken run in time must hold to read as a shift.
SHIFT_RUN_SHARE = 0.5
MIN_SHIFT_RUN = 5
#: Share of values a normal distribution puts outside Q1 − 1.5×IQR and Q3 + 1.5×IQR
#: (each fence lies 4 × 0.6745 standard deviations out, so about 0.70%).
NORMAL_OUTLIER_SHARE = float(2 * stats.norm.sf(4 * stats.norm.ppf(0.75)))
#: A reading's count is reported when a normal sample of its size would reach
#: it less often than this (binomial upper tail).
OUTLIER_CHANCE = 0.01
#: Share of a reading's values outside the fences above which the count is
#: described as the shape of its tails rather than as exceptional records.
HEAVY_TAIL_SHARE = 0.05
#: Readings counted at most, the measure first, and named beside the leading one.
OUTLIER_READINGS = 40
OUTLIERS_NAMED = 3
#: Rows the quartiles and the choice of grouping come from in a large table;
#: values are counted on every row.
FENCE_SAMPLE_ROWS = 50_000
#: Share of a reading's variance (epsilon squared) a grouping must explain to
#: be judged within, and the most levels it may have.
FENCE_GROUPING_SHARE = 0.01
FENCE_GROUPING_LEVELS = 500
#: Most levels a numeric setting (a swept frequency, a dose series) may have.
FENCE_SETTING_LEVELS = 60
#: Share of a reading one value may hold before its IQR measures that pile
#: rather than the spread, and outlier fences say nothing.
FENCE_PILE_SHARE = 0.25


@dataclass(frozen=True, eq=False)
class _Outliers:
    """One reading's values outside 1.5×IQR and how that compares with chance.

    ``flagged`` and ``present`` are over every row of the table, by position.
    """

    column: str
    flagged: np.ndarray
    present: np.ndarray
    log: bool
    grouping: str | None

    @property
    def count(self) -> int:
        return int(self.flagged.sum())

    @property
    def values(self) -> int:
        return int(self.present.sum())

    @property
    def expected(self) -> float:
        return NORMAL_OUTLIER_SHARE * self.values

    @property
    def beyond_chance(self) -> bool:
        return bool(stats.binom.sf(self.count - 1, self.values, NORMAL_OUTLIER_SHARE) < OUTLIER_CHANCE)


def _is_setting(values: np.ndarray) -> bool:
    """Whether a numeric column is a swept setting: a few evenly spaced levels, each about equally common.

    A frequency stepped from 100 to 590 GHz in 50 equal steps, every step
    measured as often, is a design setting a reading can be judged within; a
    reading with few distinct values (temperatures in whole degrees) is not
    balanced like that.
    """
    values = values[np.isfinite(values)]
    if values.size == 0:
        return False
    levels, counts = np.unique(values, return_counts=True)
    if not 2 <= levels.size <= FENCE_SETTING_LEVELS or counts.max() > 2 * counts.min():
        return False
    steps = np.diff(levels)
    return bool(np.allclose(steps, steps[0], rtol=1e-6))


def _fence_candidates(
    dataframe: pd.DataFrame, roles: ColumnRoles, sample: np.ndarray
) -> dict[str, np.ndarray]:
    """Columns a reading may be judged within, coded on every row.

    The groupings, the codes, the subject column, and swept numeric settings
    (``_is_setting``); each must leave its typical group in the ``sample`` rows
    MIN_GROUP_FOR_FENCES values for quartiles of its own.
    """
    entity = entity_column(dataframe, roles)
    named = [roles.dimension, *roles.dimensions, *roles.factors, entity]
    # A matrix table's columns are its readings, never settings.
    numeric = roles.numeric if roles.table_shape != "matrix" else ()
    settings = [
        column for column in numeric
        if column not in named and column != roles.measure and column in dataframe.columns
        and pd.api.types.is_numeric_dtype(dataframe[column])
        and _is_setting(dataframe[column].to_numpy(dtype="float64", na_value=np.nan)[sample])
    ]
    candidates = {}
    for column in dict.fromkeys(column for column in [*named, *settings] if column):
        if column not in dataframe.columns or column == roles.date:
            continue
        codes, levels = pd.factorize(dataframe[column])
        if not 2 <= len(levels) <= FENCE_GROUPING_LEVELS:
            continue
        sampled = codes[sample]
        sizes = np.bincount(sampled[sampled >= 0])
        if sizes.size == 0 or np.median(sizes[sizes > 0]) < MIN_GROUP_FOR_FENCES:
            continue
        candidates[column] = codes
    return candidates


def _explaining_grouping(
    values: np.ndarray, column: str, candidates: dict[str, np.ndarray], sample: np.ndarray
) -> str | None:
    """The candidate that explains most of a reading's variance (epsilon squared), if any explains enough.

    ``values`` are the reading's (scaled) values on the ``sample`` rows.
    """
    present = np.isfinite(values)
    best, best_share = None, FENCE_GROUPING_SHARE
    for grouping, all_codes in candidates.items():
        if grouping == column:
            continue
        codes = all_codes[sample]
        kept = present & (codes >= 0)
        if kept.sum() < 2 * MIN_GROUP_FOR_FENCES:
            continue
        x, group = values[kept], codes[kept]
        counts = np.bincount(group)
        used = counts > 0
        means = np.bincount(group, weights=x)[used] / counts[used]
        total = float(((x - x.mean()) ** 2).sum())
        between = float((counts[used] * (means - x.mean()) ** 2).sum())
        levels, n = int(used.sum()), x.size
        if total <= 0 or n <= levels:
            continue
        share = (between - (levels - 1) * (total - between) / (n - levels)) / total
        if share > best_share:
            best, best_share = grouping, share
    return best


def _reading_outliers(
    dataframe: pd.DataFrame, column: str, candidates: dict[str, np.ndarray]
) -> _Outliers | None:
    """Values of one reading outside 1.5×IQR, judged on its shape and within the grouping that explains it.

    The scale, the grouping and the pooled quartiles come from at most
    FENCE_SAMPLE_ROWS of the rows that hold the reading, chosen at random;
    each group's quartiles come from all its rows, and every row is counted.
    The grouping is named only when some group had quartiles of its own.
    """
    if counts_units(column) or is_location_name(column):
        return None
    if not pd.api.types.is_numeric_dtype(dataframe[column]):
        return None
    values = dataframe[column].to_numpy(dtype="float64", na_value=np.nan)
    present = np.isfinite(values)
    if present.sum() < 12:
        return None
    if float((values[present] == 0).mean()) >= HEAVY_TAIL_ZEROS:
        return None
    sample = np.flatnonzero(present)
    if sample.size > FENCE_SAMPLE_ROWS:
        sample = np.sort(np.random.default_rng(0).choice(sample, FENCE_SAMPLE_ROWS, replace=False))
    sampled = values[sample]
    levels, counts = np.unique(sampled, return_counts=True)
    if levels.size < 4 or counts.max() >= FENCE_PILE_SHARE * sampled.size:
        return None
    log = prefers_log_scale(np.sort(sampled), column)
    if log:
        with np.errstate(divide="ignore", invalid="ignore"):
            scaled = np.where(values > 0, np.log10(values), np.nan)
    else:
        scaled = values
    grouping = _explaining_grouping(scaled[sample], column, candidates, sample)
    low, high, own = _fences(scaled, candidates.get(grouping) if grouping else None, sample)
    with np.errstate(invalid="ignore"):
        flagged = present & ((scaled < low) | (scaled > high))
    return _Outliers(column, flagged, present, log, grouping if own else None)


def _outlier_evidence(
    dataframe: pd.DataFrame,
    roles: ColumnRoles,
    readings: list[str] | None = None,
    issues: tuple[QualityIssue, ...] = (),
    rules: tuple[GroupingRule, ...] = (),
) -> Evidence | None:
    """Count values outside 1.5×IQR in every reading, and report those beyond chance.

    Each reading is judged the way it is shaped: a right-skewed positive
    reading on a log scale, and against the quartiles of the grouping that
    explains most of its variance (a station, a frequency, a sample), so a
    long tail or one group's level is not called exceptional; a grouping cut
    from the reading itself (``rules``: health bands of an index) is never
    used. A count of subjects, a coordinate, a reading a quarter of whose
    values are one value (a floor, whose IQR measures the pile), or one the
    data checks find piled in a sliver of its range is skipped, and the card
    says so. A normal sample puts about 0.70% of its values outside the
    fences, so a count is reported only when chance would rarely reach it,
    and a count above HEAVY_TAIL_SHARE is described as the column's shape. A
    measure that is mostly zeros gets no list at all. When one reading's
    flagged values form one run in time, the card reports a shift starting
    then rather than scattered records. ``readings`` are the other columns to
    count; a matrix table's columns are not counted one by one.
    """
    if roles.measure and roles.measure in dataframe.columns:
        measure_values = dataframe[roles.measure].dropna().to_numpy(dtype=float)
        zeros = float((measure_values == 0).mean()) if len(measure_values) else 0.0
        if len(measure_values) >= 12 and zeros >= HEAVY_TAIL_ZEROS:
            return _heavy_tail_evidence(roles.measure, measure_values, zeros)
    if roles.table_shape == "matrix" or readings is None:
        readings = []
    columns = list(dict.fromkeys(column for column in [roles.measure, *readings] if column))
    columns = [column for column in columns if column in dataframe.columns][:OUTLIER_READINGS]
    piles = {issue.columns[0]: issue for issue in issues if issue.check == "pile"}
    piled = [column for column in columns if column in piles]
    columns = [column for column in columns if column not in piles]
    if not columns:
        return None
    sample = np.arange(len(dataframe))
    if len(dataframe) > FENCE_SAMPLE_ROWS:
        sample = np.sort(np.random.default_rng(0).choice(len(dataframe), FENCE_SAMPLE_ROWS, replace=False))
    candidates = _fence_candidates(dataframe, roles, sample)
    derived = {(rule.grouping, reading) for rule in rules for reading in rule.readings}
    found = [
        result for column in columns
        if (result := _reading_outliers(
            dataframe, column,
            {grouping: codes for grouping, codes in candidates.items() if (grouping, column) not in derived},
        ))
    ]
    orders: dict = {}
    for result in found:
        shift = _shift(dataframe, roles, result, orders)
        if shift is not None:
            return shift
    beyond = [result for result in found if result.count and result.beyond_chance]
    if not beyond:
        return None
    measure = next((result for result in found if result.column == roles.measure), None)
    lead = measure if measure in beyond else max(beyond, key=lambda result: result.count - result.expected)
    others = sorted(
        (result for result in beyond if result is not lead), key=lambda result: result.expected - result.count
    )
    parts = [_outlier_sentence(lead)]
    if note := _pile_note(lead.column, issues):
        parts.append(note)
    if others:
        named = [
            f"{result.column} {result.count:,} ({_share_of(result)}"
            + (f", within its {result.grouping})" if result.grouping not in (None, lead.grouping) else ")")
            for result in others[:OUTLIERS_NAMED]
        ]
        more = len(others) - OUTLIERS_NAMED
        parts.append(f"Also beyond chance: {list_names(named + ([f'{more} more'] if more > 0 else []))}.")
    if measure is not None and measure not in beyond:
        held = _values_of(measure.count) if measure.count else "none"
        parts.append(f"{measure.column}, the measure, has {held}, about what chance gives.")
    if piled:
        examples = [
            f"{column} ({format_percentage(piles[column].share)} of its values in a sliver of its range)"
            for column in piled
        ]
        since = "its IQR measures that pile" if len(piled) == 1 else "their IQRs measure those piles"
        parts.append(f"Not counted: {list_names(examples)}, since {since} rather than the spread.")
    grouped = [result for result in found if result.grouping]
    if not grouped:
        within = "with pooled quartiles"
    elif len(grouped) == len(found):
        within = (
            "with quartiles taken within the grouping that explains most of its variance (epsilon squared of "
            "1% or more)"
        )
    else:
        within = (
            "with quartiles taken within the grouping that explains most of its variance (epsilon squared of "
            "1% or more) where there is one, pooled otherwise"
        )
    calculation = (
        f"Values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR in each numeric reading, {within}, on log10 for "
        "right-skewed readings; a count is reported when a normal sample of the same size would reach it "
        f"with under {OUTLIER_CHANCE:.0%} chance (binomial, {NORMAL_OUTLIER_SHARE:.2%} expected outside)"
    )
    if any(result.values > FENCE_SAMPLE_ROWS for result in found):
        calculation += (
            f"; pooled quartiles and groupings from a random sample of {FENCE_SAMPLE_ROWS:,} values of each "
            "reading, each group's quartiles and the counts from every row"
        )
    return Evidence(
        kind="outliers",
        title="Long tails" if lead.count > HEAVY_TAIL_SHARE * lead.values else "Exceptional records",
        value=f"{lead.count:,}",
        statement=" ".join(parts),
        calculation=calculation,
        tone="warning",
    )


def _values_of(count: int) -> str:
    return f"{count:,} {'value' if count == 1 else 'values'}"


def _share_of(result: _Outliers) -> str:
    """The flagged share of a reading's values, "under 0.1%" rather than "0.0%"."""
    share = result.count / result.values * 100
    return "under 0.1%" if 0 < share < 0.1 else format_percentage(share)


def _outlier_sentence(result: _Outliers) -> str:
    """How many of a reading's values sit outside 1.5×IQR, against how many chance gives."""
    within = f" of their own {sentence_case(result.grouping)}" if result.grouping else ""
    on_log = f" on a log scale, since {result.column} is right-skewed" if result.log else ""
    verb = "sits" if result.count == 1 else "sit"
    chance = f"about {result.expected:,.0f}" if result.expected >= 1 else "fewer than 1"
    sentence = (
        f"{_values_of(result.count)} of {result.column} ({_share_of(result)} of its non-missing values) "
        f"{verb} outside 1.5×IQR{within}{on_log}, against {chance} that chance gives in normal data of "
        "this size."
    )
    if result.count > HEAVY_TAIL_SHARE * result.values:
        sentence += (
            f" At that share they are part of the shape of {result.column}, long tails or a second cluster "
            "of values, rather than a handful of exceptional records."
        )
    return sentence


def _pile_note(column: str, issues: tuple[QualityIssue, ...]) -> str:
    """A sentence when a cap or floor in the column narrows its IQR, so the count partly reflects the pile."""
    for issue in issues:
        if (
            issue.check == "cap" and column in issue.columns and issue.share >= 10
            and issue.group[0] in ("maximum", "minimum")
        ):
            side, value = issue.group
            kind = "cap" if side == "maximum" else "floor"
            return (
                f"Its {kind} at {show_value(value)} holds {format_percentage(issue.share)} of its values and "
                "narrows the IQR, so part of this count reflects the pile-up rather than the tails."
            )
    return ""


def _heavy_tail_evidence(measure: str, values: np.ndarray, zeros: float) -> Evidence:
    """Describe a mostly-zero measure's tail instead of listing its large values as outliers."""
    top = float(np.percentile(values, 99))
    return Evidence(
        kind="outliers",
        title="Heavy-tailed measure",
        value=f"{format_percentage(zeros * 100)} zeros",
        statement=(
            f"{measure} is heavy-tailed: {format_percentage(zeros * 100)} of its values are 0 and the "
            f"top 1% reach {show_value(top)} or more. An outlier rule would flag ordinary values here, "
            "so none are listed; medians describe it better than means."
        ),
        calculation="Share of values equal to 0, and the 99th percentile; outlier rules are skipped "
        f"when {format_percentage(HEAVY_TAIL_ZEROS * 100)} or more of the values are 0",
        tone="neutral",
    )


def _fences(
    scaled: np.ndarray, codes: np.ndarray | None, sample: np.ndarray
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Q1 − 1.5×IQR and Q3 + 1.5×IQR for each row, from its group's quartiles where it has enough,
    and whether any group had quartiles of its own.

    The pooled quartiles come from the ``sample`` rows. A group's come from
    all its rows, since a sample spread over a few hundred groups leaves each
    too few values for steady quartiles; a group with fewer than
    MIN_GROUP_FOR_FENCES values, or no spread, uses the pooled ones.
    """
    values = scaled[sample]
    finite = np.isfinite(values)
    first, third = np.percentile(values[finite], [25, 75]) if finite.any() else (np.nan, np.nan)
    low = np.full(scaled.shape, first - 1.5 * (third - first))
    high = np.full(scaled.shape, third + 1.5 * (third - first))
    if codes is None or codes.max() < 0:
        return low, high, False
    finite = np.isfinite(scaled)
    kept = finite & (codes >= 0)
    table = pd.Series(scaled[kept]).groupby(codes[kept])
    quartiles = table.quantile([0.25, 0.75]).unstack()
    sizes = table.size()
    spread = quartiles[0.75] - quartiles[0.25]
    usable = (sizes >= MIN_GROUP_FOR_FENCES) & (spread > 0)
    own_low = np.full(int(codes.max()) + 1, np.nan)
    own_high = np.full(int(codes.max()) + 1, np.nan)
    groups = usable.index[usable.to_numpy()].to_numpy(dtype=int)
    own_low[groups] = (quartiles[0.25] - 1.5 * spread)[usable].to_numpy()
    own_high[groups] = (quartiles[0.75] + 1.5 * spread)[usable].to_numpy()
    labelled = codes >= 0
    row_low = np.where(labelled, own_low[np.where(labelled, codes, 0)], np.nan)
    row_high = np.where(labelled, own_high[np.where(labelled, codes, 0)], np.nan)
    pooled = np.isnan(row_low)
    return np.where(pooled, low, row_low), np.where(np.isnan(row_high), high, row_high), bool(groups.size)


def _time_order(
    dataframe: pd.DataFrame, roles: ColumnRoles, grouping: str | None
) -> tuple[np.ndarray, np.ndarray]:
    """Row positions with a time, in time order within each group, and each one's group code."""
    times = dataframe[roles.date]
    keys = pd.factorize(dataframe[grouping])[0] if grouping else np.zeros(len(dataframe), dtype=int)
    order = np.lexsort((times.to_numpy(), keys))
    order = order[times.notna().to_numpy()[order]]
    return order, keys[order]


def _shift(
    dataframe: pd.DataFrame,
    roles: ColumnRoles,
    result: _Outliers,
    orders: dict | None = None,
) -> Evidence | None:
    """Report a reading's flagged values that form one unbroken run in time as a shift starting then.

    Rows are put in time order within each group of the grouping the reading
    was judged within. A run counts when it holds SHIFT_RUN_SHARE of the
    flagged values and at least MIN_SHIFT_RUN of them, at as many distinct
    times. ``orders`` caches the time order by grouping across readings.
    """
    if not roles.date or roles.date not in dataframe.columns or result.count < MIN_SHIFT_RUN:
        return None
    column, grouping = result.column, result.grouping
    orders = {} if orders is None else orders
    if grouping not in orders:
        orders[grouping] = _time_order(dataframe, roles, grouping)
    order, keys = orders[grouping]
    marked = result.flagged
    kept = result.present[order]
    sequence, keys = order[kept], keys[kept]
    flags = marked[sequence]
    if not flags.any():
        return None
    same_group = np.r_[False, keys[1:] == keys[:-1]]
    starts = flags & ~(np.r_[False, flags[:-1]] & same_group)
    runs = np.cumsum(starts) * flags
    lengths = np.bincount(runs)[1:]
    longest = int(np.argmax(lengths)) + 1
    length = int(lengths[longest - 1])
    total = int(flags.sum())
    if length < max(MIN_SHIFT_RUN, SHIFT_RUN_SHARE * total):
        return None
    run = np.flatnonzero(runs == longest)
    times = dataframe[roles.date]
    # Rows sharing one time are one record repeated (a visit's chemistry on each species row), not a shift.
    if times.iloc[sequence[run]].nunique() < MIN_SHIFT_RUN:
        return None
    first, last = times.iloc[sequence[run[0]]], times.iloc[sequence[run[-1]]]
    group_end = run[-1] == len(sequence) - 1 or keys[run[-1] + 1] != keys[run[-1]]
    begun, ended = _moment(first), _moment(last)
    where = ""
    if grouping:
        value = dataframe[grouping].iloc[sequence[run[0]]]
        label = group_label(grouping, value)
        where = f" for {label if label != str(value) else f'{grouping} {label}'}"
    until = "to the end of the record" if group_end else f"until {ended}"
    return Evidence(
        kind="outliers",
        title="Level shift",
        value=begun,
        statement=(
            f"{column} left its usual range{where} at {begun} and stayed out for {length:,} "
            f"readings in a row, {until}: {length:,} of the {total:,} values outside 1.5×IQR form this "
            "one run, so this is a shift starting then rather than scattered entry errors."
        ),
        calculation=(
            "Values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR, put in time order; the longest unbroken "
            f"run holds at least {format_percentage(SHIFT_RUN_SHARE * 100)} of them"
        ),
        tone="warning",
    )


def _moment(value: pd.Timestamp) -> str:
    """A timestamp at the precision it was recorded: a date, or a date and time."""
    stamp = pd.Timestamp(value)
    if stamp == stamp.normalize():
        return f"{stamp:%d %b %Y}"
    return f"{stamp:%d %b %Y %H:%M:%S}"


#: Data-check issues named on the brief's card, about values and about gaps;
#: ``Analysis.quality()`` lists them all.
QUALITY_ISSUES_SHOWN = 3
GAP_ISSUES_SHOWN = 2
#: Issues named in the brief's summary as the ones to check first.
CHECK_FIRST_SHOWN = 2
#: Share of the table, in percent, at which an issue about the whole table
#: (copied rows, simulated-looking data) is named first whatever the headline,
#: and the share of a column at which any other issue about values is.
CHECK_FIRST_SHARE = 10.0
CHECK_FIRST_BROAD = 25.0


def _leading_issues(issues: tuple[QualityIssue, ...], limit: int) -> list[QualityIssue]:
    """The issues to name first: those covering the most rows or cells, descriptions of the table last."""
    return sorted(issues, key=lambda issue: (issue.check in INFO_CHECKS, -issue.rows))[:limit]


def _quality_evidence(
    dataframe: pd.DataFrame,
    issues: tuple[QualityIssue, ...] = (),
    measure: str | None = None,
    roles: ColumnRoles | None = None,
) -> Evidence | None:
    """The data checks: the leading issues by name and count, and how complete the table is.

    One part of a stacked file is judged on its own columns (``own_columns``).
    """
    own = own_columns(dataframe, roles)
    total_cells = own.shape[0] * own.shape[1]
    missing_cells = int(own.isna().sum().sum())
    if total_cells == 0 or (missing_cells == 0 and not issues):
        return None
    rate = missing_cells / total_cells * 100
    completeness = (
        f"{missing_cells:,} cells are missing, leaving the analyzed dataset "
        f"{format_percentage(100 - rate)} complete."
    )
    if own.shape[1] < dataframe.shape[1]:
        others = dataframe.shape[1] - own.shape[1]
        completeness = completeness[:-1] + (
            f" across its own {own.shape[1]:,} columns; the {others:,} columns of the other stacked tables, "
            "empty throughout this one, are left out."
        )
    if not issues:
        return Evidence(
            kind="quality",
            title="Data completeness",
            value=format_percentage(100 - rate),
            statement=completeness,
            calculation="Non-missing cells ÷ all cells",
            tone="warning" if rate >= 5 else "neutral",
        )
    values = tuple(issue for issue in issues if not issue.check.startswith("missing"))
    gaps = tuple(issue for issue in issues if issue.check.startswith("missing"))
    shown = _leading_issues(values, QUALITY_ISSUES_SHOWN)
    shown_gaps = _leading_issues(gaps, GAP_ISSUES_SHOWN)
    parts = []
    if shown:
        parts.append("; ".join(issue.summary for issue in shown) + ".")
    parts.append(completeness if missing_cells else "No cells are missing.")
    if shown_gaps:
        parts.append("Gaps: " + "; ".join(issue.summary for issue in shown_gaps) + ".")
    more = len(issues) - len(shown) - len(shown_gaps)
    if more:
        parts.append(f"{more} more {'is' if more == 1 else 'are'} in the full list of checks.")
    noun = "issue" if len(issues) == 1 else "issues"
    return Evidence(
        kind="quality",
        title="Data checks",
        value=f"{len(issues)} {noun}",
        statement=f"The data checks found {len(issues)} {noun} to look at before trusting these figures. "
        + " ".join(parts),
        calculation=(
            "Every numeric column checked for values outside what its name or a physical limit allows, "
            "missing-value codes, caps, repeated fill values, values more than 10 robust standard "
            "deviations out, second clusters, swapped and copied columns, max below min, stuck readings, "
            "a group off by a unit factor, and gaps that follow a group or another column; the table for "
            "copied rows and for flat, unrelated columns that suggest simulated data; issues ranked by the "
            "rows they cover; Analysis.quality() lists every issue with its rule"
        ),
        tone="warning",
    )


def _issues_on_headline(
    issues: tuple[QualityIssue, ...], roles: ColumnRoles, leader: Evidence | None
) -> list[QualityIssue]:
    """Issues that touch the measure or the group the headline names."""
    def concerns_leader(issue: QualityIssue) -> bool:
        if leader is None or leader.subject is None or issue.group is None:
            return False
        named = issue.group if isinstance(issue.group, tuple) else (issue.group,)
        return any(
            leader.subject in (str(group), group_label(roles.dimension, group)) for group in named
        )

    return [
        issue
        for issue in issues
        if (roles.measure in issue.columns and issue.check != "missing") or concerns_leader(issue)
    ]


def _check_first(
    issues: tuple[QualityIssue, ...],
    roles: ColumnRoles,
    leader: Evidence | None,
    resolved: tuple[QualityIssue, ...] = (),
) -> str:
    """One line naming the issues behind the headline, or "" when none touch it.

    Issues about the whole table (copied rows, data that look simulated)
    count when they cover CHECK_FIRST_SHARE of it, and any other issue about
    values when it covers CHECK_FIRST_BROAD of its column (fill values in 61% of
    rows), since they qualify figures well beyond the headline. They are
    ranked as the data checks card ranks them, and those still in the data
    come before ``resolved`` ones, codes cleaning already set to missing: the
    card's leading open issues take a resolved code's place, since the code no
    longer touches any figure. Descriptions of the table (a constant column)
    are never named here.
    """
    open_values = [
        issue for issue in issues
        if not issue.check.startswith("missing") and issue.check not in INFO_CHECKS and issue not in resolved
    ]
    broad = [
        issue for issue in issues
        if not issue.check.startswith("missing") and issue.check not in INFO_CHECKS
        and issue.share >= (CHECK_FIRST_SHARE if issue.check in TABLE_CHECKS else CHECK_FIRST_BROAD)
    ]
    behind = [
        issue for issue in _issues_on_headline(issues, roles, leader)
        if issue not in broad and issue.check not in INFO_CHECKS
    ]
    relevant = broad + behind
    if not relevant:
        return ""
    current = _leading_issues(tuple(issue for issue in relevant if issue not in resolved), CHECK_FIRST_SHOWN)
    handled = [issue for issue in relevant if issue in resolved]
    leading = [issue for issue in _leading_issues(tuple(open_values), CHECK_FIRST_SHOWN)
               if issue not in current]
    chosen = (current + (leading if handled else []) + handled)[:CHECK_FIRST_SHOWN]
    chosen.sort(key=lambda issue: (issue in resolved, issue.check in INFO_CHECKS, -issue.rows))
    return "Check these first: " + "; ".join(issue.summary for issue in chosen) + "."


def _flag_driven_leader(
    evidence: list[Evidence], dataframe: pd.DataFrame, roles: ColumnRoles, issues: tuple[QualityIssue, ...]
) -> list[Evidence]:
    """Say when the leading group's average rests on flagged values, with its median beside it.

    Applies to an averaged measure. The leader's average is recomputed without
    the values the data checks flagged; when that changes it by more than 5%, or
    hands the lead to another group, the card says so.
    """
    position = next((i for i, item in enumerate(evidence) if item.kind == "leader"), None)
    if position is None or not roles.measure or not roles.dimension:
        return evidence
    if metric_for(roles.measure, roles).additive:
        return evidence
    flagged = flagged_index(issues, roles.measure)
    if flagged.empty:
        return evidence
    measure, groups = dataframe[roles.measure], dataframe[roles.dimension]
    means = measure.groupby(groups).mean().dropna()
    if means.empty:
        return evidence
    top = means.idxmax()
    leader = evidence[position]
    if leader.subject not in (str(top), group_label(roles.dimension, top)):
        return evidence
    in_top = (groups == top) & measure.notna()
    hits = flagged.intersection(dataframe.index[in_top])
    if hits.empty:
        return evidence
    kept = ~dataframe.index.isin(flagged)
    clean = measure[kept].groupby(groups[kept]).mean().dropna()
    without = float(clean.get(top, np.nan))
    successor = clean.idxmax() if not clean.empty else top
    moved = not np.isfinite(without) or abs(means[top] - without) > 0.05 * abs(means[top])
    if not moved and successor == top:
        return evidence
    column_values = measure.dropna()
    median = format_number(float(measure[in_top].median()), roles.measure, column_values=column_values)
    shown = ", ".join(show_value(value) for value in measure.loc[hits].head(3))
    noun = "value" if len(hits) == 1 else "values"
    note = (
        f" That average rests on {len(hits):,} flagged {roles.measure} {noun} ({shown}); "
        f"{group_label(roles.dimension, top)}'s median is {median}"
    )
    if successor != top:
        note += f", and without the flagged values {group_label(roles.dimension, successor)} would lead"
    evidence[position] = replace(leader, statement=leader.statement + note + ".", tone="warning")
    return evidence


def _shown_evidence(evidence: list[Evidence], limit: int = 8) -> list[Evidence]:
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
    if trend and trend.tone == "negative" and not trend.quiet:
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
    elif trend and not trend.quiet:
        # A change within the usual variation is no change to follow up.
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

    shift = by_kind.get("shift")
    if shift:
        recommendations.append(
            Recommendation(
                "Now",
                f"Find what changed in {shift.value}",
                (
                    "Match the start of the new level to an event (a change in method, instrument, site "
                    "or conditions) before comparing periods on either side of it."
                ),
                shift.statement,
            )
        )

    concentration = by_kind.get("concentration")
    leader = by_kind.get("leader")
    largest = leader.subject if leader and leader.subject else f"the largest {group}"
    # A leader card that compares groups, rather than giving shares or sizes.
    compares = (
        leader is not None
        and "÷ total" not in leader.calculation
        and not leader.calculation.startswith("largest |")
    )
    if concentration and concentration.tone == "warning":
        recommendations.append(
            Recommendation(
                "Next",
                f"Check how much rests on {largest}",
                f"Recompute the overall figures without {largest} to see how much they change.",
                concentration.statement,
            )
        )
    elif compares and leader.title.startswith("Too few values"):
        recommendations.append(
            Recommendation(
                "Next",
                f"Collect more records per {group}",
                (
                    f"The {group} groups hold fewer than {MIN_NAMED_GROUP} values each, too few to rank; "
                    "compare a coarser grouping, or collect more records per group."
                ),
                leader.statement,
            )
        )
    elif compares and leader.subject is None:
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
    elif compares:
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
    # A scan that found no relationship leaves nothing to test.
    if relationship and not relationship.value.startswith("|r|"):
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
    if outliers and outliers.title == "Level shift":
        recommendations.append(
            Recommendation(
                "Now",
                "Find what changed when the run began",
                (
                    "Match the start of the run to an event (a change in the process, the instrument "
                    "or the conditions) before treating these values as errors."
                ),
                outliers.statement,
            )
        )
    elif outliers and outliers.title == "Heavy-tailed measure":
        recommendations.append(
            Recommendation(
                "Watch",
                "Compare medians, not means",
                "Compare groups by their medians or on a log scale; a few large values pull means up.",
                outliers.statement,
            )
        )
    elif outliers:
        recommendations.append(
            Recommendation(
                "Watch",
                "Review the extreme records",
                (
                    "Look at what sets the most extreme records apart before averaging over them: "
                    "values beyond 1.5×IQR are often real but rare, and either way they pull averages."
                ),
                outliers.statement,
            )
        )

    quality = by_kind.get("quality")
    if quality and quality.title == "Data checks":
        # Flagged values qualify every other figure, so they are checked first.
        recommendations.insert(
            0,
            Recommendation(
                "Fix",
                "Check the flagged values first",
                (
                    "Look at the values the data checks flagged, and decide whether to correct or "
                    "exclude them, before relying on averages and correlations."
                ),
                quality.statement,
            ),
        )
    elif quality and quality.tone == "warning":
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

    if not recommendations and (roles.date or roles.dimension) and evidence:
        recommendations.append(
            Recommendation(
                "Next",
                "Collect more before acting",
                (
                    "Nothing here stands out from the noise yet; more periods, or more records per "
                    "group, would show smaller effects."
                ),
                evidence[0].statement,
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
    """The leading-group clause of a headline for a measure whose groups are compared."""
    if leader.lead:
        return leader.lead
    return f"no clear difference in {roles.measure} between {sentence_case(roles.dimension)} groups."


def _leader_phrase(leader: Evidence, roles: ColumnRoles) -> str:
    """The leading-group clause of a headline after the latest movement."""
    if "÷ total" in leader.calculation:
        if roles.measure:
            measure = sentence_case(roles.measure)
            return f"{leader.subject} has the largest share of {measure}, at {leader.value}."
        return f"{leader.subject} has the most rows, {leader.value} of them."
    if leader.calculation.startswith("largest |"):
        return f"{leader.subject} is the largest by size, at {leader.value}."
    return _averaged_leader_phrase(leader, roles)


# Exact repeats of whole rows beyond this share make every test over-confident.
REPEATED_SHARE = 0.1
# A numeric column with fractional values and at least this many distinct ones is a continuous reading.
CONTINUOUS_VALUES = 20


def _distinct_rows(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """The rows the tests run on: the distinct ones when many rows exactly repeat a continuous reading.

    Two records of a continuous measurement can't agree on every column by
    chance, so beyond REPEATED_SHARE such repeats are copies, and counting them
    would make every test over-confident. Counts and labels can coincide
    legitimately (two quadrats with no crabs), so a table with no continuous
    reading is left whole. Returns the rows and the number of repeats left out.
    """
    continuous = None
    for column in dataframe.select_dtypes("number").columns:
        values = dataframe[column].dropna()
        if values.nunique() >= CONTINUOUS_VALUES and bool((values % 1 != 0).any()):
            continuous = column
            break
    if continuous is None:
        return dataframe, 0
    repeated = dataframe.duplicated()
    count = int(repeated.sum())
    if count <= REPEATED_SHARE * len(dataframe):
        return dataframe, 0
    return dataframe.loc[~repeated], count


def _headline(
    evidence: list[Evidence], growth: Evidence | None, dataframe: pd.DataFrame, roles: ColumnRoles
) -> str:
    """The brief's first line, from its most informative finding.

    A total (money, counts) leads with its latest movement, as a sales report
    does, and so does any measure whose latest movement is unusual. A latest
    movement within the usual variation is "no clear change" and gives way to
    a finding about the measure: its own group comparison, a change in its
    level, a rate, the largest difference found. Without a date, a joint model
    or a relationship can lead too; over time a quiet movement says more than
    a correlation between two readings. A bare overall figure leads only when
    nothing else was found, and a group is never named on fewer than five
    values or without a clear test.
    """
    metric = metric_for(roles.measure, roles)
    by_kind: dict[str, Evidence] = {}
    for item in evidence:
        by_kind.setdefault(item.kind, item)
    leader = by_kind.get("leader")

    def leads(*kinds: str) -> list[str]:
        return [by_kind[kind].lead for kind in kinds if kind in by_kind and by_kind[kind].lead]

    findings = leads("leader", "shift", "rates", "differences")
    if growth and (metric.additive or not growth.quiet):
        movement = f"{growth.value} latest movement" + (", within the usual range" if growth.quiet else "")
        if leader:
            return f"{movement}; {_leader_phrase(leader, roles)}"
        if growth.quiet and findings:
            return findings[0]
        return _first_sentence(growth.statement)
    if growth:
        if findings:
            return findings[0]
        when = growth.subject or "the latest period"
        quiet = f"No clear change in {roles.measure} in {when} ({growth.value}, within the usual range)"
        if leader and leader.subject is None and roles.dimension:
            return f"{quiet}; no clear difference between {sentence_case(roles.dimension)} groups."
        return f"{quiet}."
    findings += leads("model", "relationship")
    if findings:
        return findings[0]
    if leader:
        return _first_sentence(leader.statement)
    if roles.measure:
        measure_values = dataframe[roles.measure].dropna()
        combined = measure_values.mean() if metric.aggregation == "mean" else measure_values.sum()
        total_measure = format_number(float(combined), roles.measure, column_values=measure_values)
        verb = "averages" if metric.aggregation == "mean" else "totals"
        return f"{roles.measure} {verb} {total_measure} across all rows."
    return f"{len(dataframe):,} rows, with no date, measure or grouping column ADA could use."


def _subjects_note(evidence: Evidence | None, frame: pd.DataFrame, entity: str | None) -> Evidence | None:
    """The relationship card with a word on its rows when several come from each subject.

    Readings of one subject over time are not independent, so the interval of a
    correlation over all of them is narrower than the evidence behind it.
    """
    if evidence is None or entity is None or entity not in frame.columns:
        return evidence
    subjects = frame[entity].nunique()
    if not subjects or subjects >= frame[entity].notna().sum():
        return evidence
    return replace(
        evidence,
        statement=(
            f"{evidence.statement} The rows come from {subjects:,} {entity} values, several from each, so "
            "they are not independent and the intervals are narrower than the evidence."
        ),
    )


def _noting(evidence: Evidence | None, left_out: str) -> Evidence | None:
    """The card with the cells the data checks left out added to its calculation."""
    if evidence is None or not left_out or evidence.title == "Heavy-tailed measure":
        return evidence
    return replace(evidence, calculation=f"{evidence.calculation}; {left_out}")


def analyze_business(
    dataframe: pd.DataFrame,
    roles: ColumnRoles | None = None,
    frequency: str | None = None,
    blanked: tuple[QualityIssue, ...] = (),
) -> BusinessBrief:
    """Create an executive brief from explainable calculations and rule-based interpretation.

    ``frequency`` fixes the period grain (``s min h D W M Q Y``) for the trend, driver and
    anomaly findings, so they match a trend or forecast built at that grain.
    Without it the grain follows how often the data was recorded.
    ``blanked`` holds the missing-value codes cleaning set to missing
    (``CleaningReport.blanked_codes``), so the data checks card lists them too.
    """
    roles = roles or detect_roles(dataframe)
    metric = metric_for(roles.measure, roles)
    issues = scan_quality(dataframe, roles, blanked=blanked)
    # Cells the checks found can't be readings (codes, impossible values, fill
    # values, swapped or stuck readings) are blanked for the scans that compare
    # and relate columns, and each of those cards says how many were left out.
    statistics, left_out = statistics_frame(dataframe, issues)
    # Exact copies of rows would count as independent evidence in every test.
    tested, repeats = _distinct_rows(dataframe)
    note = f"; on the {len(tested):,} distinct rows, leaving out {repeats:,} exact repeats" if repeats else ""
    tested_statistics = statistics.loc[tested.index] if repeats else statistics
    evidence: list[Evidence] = []

    growth = _growth_evidence(dataframe, roles, frequency)
    if growth:
        evidence.append(growth)
    change_driver = _change_driver_evidence(dataframe, roles, frequency)
    if change_driver:
        evidence.append(change_driver)
    # Both scans work on the same numeric readings, which are slow to pick out of a
    # wide table, and on one screen of how they are built from one another. The
    # column naming each row's subject is a code, not a reading.
    found = _find_groupings(tested, roles)
    readings = [column for column in numeric_readings(dataframe, roles) if column != found.entity]
    built = _construction(tested, roles, readings)
    masked = built.masked(tested) if built is not None else tested
    rules = grouping_rules(masked, readings, _rule_groupings(roles, found), date=roles.date)
    evidence.extend(_segment_evidence(dataframe, roles, tested, note, found, rules=rules, built=built))
    evidence = _flag_driven_leader(evidence, dataframe, roles, issues)
    shift = _level_shift_evidence(dataframe, roles, frequency)
    if shift:
        evidence.append(shift)
    differences = _noting(
        _differences_evidence(tested_statistics, roles, readings, note, found, built=built, rules=rules),
        left_out,
    )
    found_difference = differences is not None and differences.subject is not None
    if found_difference:
        evidence.append(differences)
    # A reading copied onto every row of a sample is correlated once per sample.
    # When most readings are copied like it, the table is one of samples with
    # rows repeated, and correlations count each sample once; readings that vary
    # within a sample (a sensor through a test run) keep their rows.
    related, related_readings, related_note, related_built = tested_statistics, readings, note, built
    collapsed = False
    if found.copied:
        unit, varying = one_row_per(tested_statistics, found.entity)
        kept = [column for column in readings if column not in varying]
        if len(kept) >= len(readings) / 2:
            collapsed = True
            related, related_readings = unit, kept
            # The screen holds correlations over its own rows, so the cut table gets its own.
            related_built = _construction(related, roles, related_readings)
            related_note += (
                f"; one row per {found.entity} ({len(related):,} of the {len(tested):,} rows), since "
                f"{roles.measure} is the same on every row of a {found.entity}"
            )
    relationships = _relationship_scan(related, roles, related_readings, related_built)
    for optional_evidence in (
        _rate_evidence(tested, roles, note, found),
        _anomaly_evidence(dataframe, roles, frequency),
        _noting(
            _subjects_note(
                _relationship_evidence(
                    related, roles, related_readings, relationships, related_note, built=related_built,
                    rules=rules,
                ),
                related,
                None if collapsed else found.entity,
            ),
            left_out,
        ),
        _noting(_joint_model_evidence(related, roles, relationships, related_note), left_out),
        _noting(_outlier_evidence(statistics, roles, readings, issues, rules), left_out),
        # A scan that found nothing ranks below the findings.
        None if found_difference else differences,
        _quality_evidence(dataframe, issues, roles.measure, roles),
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
                "Versus the previous observed period" + (", within the usual range" if growth.quiet else ""),
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
    own = own_columns(dataframe, roles)
    completeness = 1 - own.isna().sum().sum() / max(own.size, 1)
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
    headline = _headline(evidence, growth, dataframe, roles)

    # Without a trend the headline already gives the one finding there is, so
    # the summary says what the table seems to be instead of repeating it.
    if growth:
        summary_parts = [growth.statement] + ([leader.statement] if leader else [])
    else:
        summary_parts = [_table_description(dataframe, roles)]
    if repeats:
        share = format_percentage(repeats / len(dataframe) * 100)
        summary_parts.append(
            f"{repeats:,} of the {len(dataframe):,} rows ({share}) repeat another row exactly, so the "
            f"comparisons and correlations use the {len(tested):,} distinct rows."
        )
    if found.copied:
        samples = tested[found.entity].nunique()
        summary_parts.append(
            f"{roles.measure} is the same on every row of a {found.entity} ({samples:,} of them on "
            f"{len(tested):,} rows), so each {found.entity} counts once in the comparisons"
            f"{' and correlations' if collapsed else ''}."
        )
    extras = [shift]
    if found_difference and (leader is None or leader.subject is None):
        extras.append(differences)
    for item in extras:
        if item is not None and item.lead and item.lead != headline:
            summary_parts.append(item.lead)
    if check_first := _check_first(issues, roles, leader, blanked):
        summary_parts.append(check_first)
    if note_text := _measure_choice_note(roles):
        summary_parts.append(note_text)

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
