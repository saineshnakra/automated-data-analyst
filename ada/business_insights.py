"""Turn a dataset into an executive brief: KPIs, evidence cards and recommendations.

Every figure comes from a deterministic pandas calculation that is shown beside it.
"""

from __future__ import annotations

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
    # "in Q3 1945", but "at 15:55" for a clock time.
    when = "at" if series.frequency in ("s", "min", "h") else "in"
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
            notes.append(
                f"{group_label(roles.dimension, group)} only has figures from "
                f"{format_moment(first)} to {format_moment(last)}"
            )
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


@dataclass(frozen=True)
class Correlation:
    """Pearson r over the complete pairs of two columns, with its interval and Spearman's r."""

    first: str
    second: str
    pearson: float
    low: float
    high: float
    pairs: int
    spearman: float

    @property
    def rank_divergence(self) -> float:
        """How far the rank correlation is from the linear one, in absolute strength."""
        return abs(self.spearman) - abs(self.pearson)


def measure_correlation(paired: pd.DataFrame) -> Correlation | None:
    """Correlate the two columns of ``paired``, which holds only complete rows.

    Returns None when r is undefined, as it is when either column does not vary.
    """
    value = float(paired.corr().iloc[0, 1])
    if not np.isfinite(value):
        return None
    low, high = _correlation_interval(value, len(paired))
    ranked = float(paired.corr(method="spearman").iloc[0, 1])
    first, second = (str(column) for column in paired.columns[:2])
    return Correlation(first, second, value, low, high, len(paired), ranked)


def describe_correlation(correlation: Correlation, scope: str = "") -> str:
    """State r with its interval, adding why the sample or the rank correlation may undercut it.

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

    # Spearman well above Pearson: monotonic but curved. Well below: a few
    # extreme records are producing the linear figure.
    if correlation.rank_divergence > RANK_DIVERGENCE:
        statement += (
            f" The rank correlation is stronger ({correlation.spearman:.2f}), so the relationship is "
            "consistent but not a straight line."
        )
    elif correlation.rank_divergence < -RANK_DIVERGENCE:
        statement += (
            f" The rank correlation is only {correlation.spearman:.2f}, so a few extreme records are "
            "driving the linear figure."
        )
    return statement


def _relationship_evidence(dataframe: pd.DataFrame, roles: ColumnRoles) -> Evidence | None:
    factors = set(roles.factors)
    usable = [
        column
        for column in roles.numeric
        if column not in factors
        and not looks_like_identifier(column, dataframe[column])
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
    correlation = measure_correlation(paired)
    if correlation is None:
        return None

    statement = describe_correlation(correlation) + (
        " It is the strongest of several pairs tested, so treat it as a lead to check, "
        "not as evidence of cause."
    )
    if near_duplicate.any():
        statement += _near_duplicate_note(stacked[near_duplicate])
    return Evidence(
        kind="relationship",
        title="Strongest measurable relationship",
        value=f"r = {correlation.pearson:.2f}",
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
    elif leader and not metric.additive:
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
    for optional_evidence in (
        _anomaly_evidence(dataframe, roles, frequency),
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
    if leader:
        # The card's title says what was ranked; the tile names the group.
        kpis.append(KPI(leader.title, leader.value, leader.subject or ""))

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
        elif not metric.additive:
            ranked = f"has the highest average, at {leader.value}"
        else:
            ranked = f"is the largest by size, at {leader.value}"
        headline = f"{growth.value} latest movement; {leader.subject} {ranked}."
    elif growth:
        # Cut at a sentence end, not at the decimal point inside "18.5%".
        headline = growth.statement.split(". ")[0].rstrip(".") + "."
    elif leader:
        headline = leader.statement
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
