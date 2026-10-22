"""Aggregate rows into the period and segment frames the other modules read.

Period grain, empty-period filling and trailing partial periods are decided once
here and recorded on ``TrendSeries``, so the trend, driver, anomaly and forecast
paths agree on what a period is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .compare import entity_column
from .formatting import format_period, sentence_case
from .metrics import metric_for
from .schema import ColumnRoles, is_entity_name
from .timeseries import (
    PERIODS_PER_YEAR,
    fit_trendline,
    observed_periods,
    season_numbers,
    yearly_pattern,
)

GRAIN_ORDER = ("s", "min", "h", "D", "W", "M", "Q", "Y")
DAY_SECONDS = 86_400
# The longest span (in seconds) each finer grain is used for, so a timeline has
# at most a few hundred periods: ten minutes by the second, eight hours by the
# minute, ten days by the hour, a month by the day.
SUB_DAY_SPANS = (("s", 600), ("min", 8 * 3_600), ("h", 10 * DAY_SECONDS), ("D", 31 * DAY_SECONDS))
# The widest recording gap (in seconds) each finer grain can still fill.
SUB_DAY_CADENCES = (("s", 1), ("min", 60), ("h", 3_600), ("D", DAY_SECONDS))


def _carries_time_of_day(dates: pd.Series) -> bool:
    """Whether most distinct timestamps fall at a time other than midnight."""
    unique = dates.drop_duplicates()
    return bool((unique != unique.dt.normalize()).mean() > 0.5)


def _grain_for_span(span: pd.Timedelta, timed: bool) -> str:
    if timed:
        seconds = span.total_seconds()
        for grain, longest in SUB_DAY_SPANS:
            if seconds <= longest:
                return grain
    span_days = max(span.days, 0)
    if span_days <= 120:
        return "W"
    if span_days <= 900:
        return "M"
    return "Q"


def _grain_for_cadence(dates: pd.Series, timed: bool) -> str:
    """Finest grain the recording cadence is dense enough to fill."""
    unique = dates.drop_duplicates()
    if len(unique) < 3:
        return GRAIN_ORDER[0] if timed else "W"
    spacing = np.diff(unique.sort_values().to_numpy()).astype("timedelta64[s]").astype(float)
    typical = float(np.median(spacing)) if spacing.size else 0.0
    if timed:
        for grain, widest in SUB_DAY_CADENCES:
            if typical <= widest:
                return grain
    typical /= DAY_SECONDS
    if typical <= 10:
        return "W"
    if typical <= 45:
        return "M"
    if typical <= 140:
        return "Q"
    # Annual readings stay yearly; quarters would add three zero-filled periods per year.
    return "Y"


def _period_frequency(date_series: pd.Series) -> str:
    """Pick a human-sized grain the data can populate.

    Takes the coarser of the span-based and cadence-based grains: five monthly
    readings over four months stay monthly instead of becoming seventeen weeks.
    Grains finer than a week are used only for timestamps with a time of day,
    so a fifty-minute log of readings a second apart is charted by the minute.
    """
    dates = date_series.dropna()
    if dates.empty:
        return "M"

    timed = _carries_time_of_day(dates)
    span = max(dates.max() - dates.min(), pd.Timedelta(0))
    return max(_grain_for_span(span, timed), _grain_for_cadence(dates, timed), key=GRAIN_ORDER.index)


def preferred_frequency(date_series: pd.Series) -> str:
    """Pick a human-sized period grain for the observed dates."""
    return _period_frequency(date_series)


EMPTY_TREND = pd.DataFrame({"Period": pd.Series(dtype="datetime64[ns]"), "Value": pd.Series(dtype=float)})
PARTIAL_COVERAGE_MARGIN = 0.2
# Coverage this far below typical, but within PARTIAL_COVERAGE_MARGIN, is reported, not excluded.
SHORT_COVERAGE_MARGIN = 0.05
MIN_PERIODS_FOR_PARTIAL_CHECK = 4


@dataclass(frozen=True)
class TrendSeries:
    """Period totals plus the adjustments made to line them up.

    Filled empty periods and an excluded trailing partial period are recorded
    here so captions can report them.
    """

    frame: pd.DataFrame
    frequency: str
    filled_periods: int = 0
    #: True when empty periods were filled with zero (an additive measure),
    #: False when they were left as NaN (an averaged measure has no observation to average).
    filled_as_zero: bool = True
    partial_period: pd.Timestamp | None = None
    partial_coverage: str = ""
    # Set when the last period ends short of typical but not enough to exclude;
    # a truncated extract and a few quiet days look the same here.
    short_coverage: str = ""
    # False when there are too few periods to judge completeness, so downstream
    # code must not call the last period complete.
    completeness_checked: bool = True

    @property
    def notes(self) -> tuple[str, ...]:
        notes: list[str] = []
        if self.short_coverage:
            notes.append(
                f"The last period only reaches {self.short_coverage}, so part of its "
                "shortfall may be missing data rather than a real fall."
            )
        if self.partial_period is not None:
            notes.append(
                f"{format_period(self.partial_period, self.frequency)} is still in progress "
                f"({self.partial_coverage}) and is excluded, so a half-finished period cannot "
                "read as a collapse."
            )
        if self.filled_periods:
            plural = "periods" if self.filled_periods > 1 else "period"
            # An averaged measure's gaps are NaN, not zero, so the caption must
            # match the break in the line.
            settled = (
                "counted as zero" if self.filled_as_zero else "left empty, since an average "
                "needs an observation"
            )
            notes.append(
                f"{self.filled_periods} {plural} with no rows {settled}, keeping the "
                "timeline evenly spaced."
            )
        return tuple(notes)


def _period_bounds(periods: pd.DatetimeIndex, frequency: str) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    index = pd.PeriodIndex(periods, freq=frequency)
    return index.start_time, index.end_time


def _trailing_partial_period(
    dates: pd.Series, periods: pd.Series, frequency: str
) -> tuple[pd.Timestamp | None, str, str, bool]:
    """Detect a final period the data stops part-way through.

    Compares how far the data reaches into the last period with the typical
    reach of earlier ones: an extract cut on the 12th covers a third of its month.
    Returns ``(partial_period, coverage, short_coverage, checked)``.
    """
    last_seen = dates.groupby(periods).max().sort_index()
    if len(last_seen) < MIN_PERIODS_FOR_PARTIAL_CHECK:
        # Too few periods to establish typical coverage.
        return None, "", "", False
    if frequency == "s":
        # A second has no part to cover.
        return None, "", "", True

    starts, ends = _period_bounds(pd.DatetimeIndex(last_seen.index), frequency)
    # Nanoseconds, so a minute's span (59.999999999 s) still counts as 60 seconds.
    spans = (ends - starts).to_numpy().astype("timedelta64[ns]").astype(float) / 1e9
    reached = (last_seen.to_numpy() - starts.to_numpy()).astype("timedelta64[ns]").astype(float) / 1e9
    coverage = np.divide(reached, spans, out=np.zeros_like(reached), where=spans > 0)
    covered, total, reach = _reach(reached[-1], spans[-1], frequency)

    typical = float(np.median(coverage[:-1]))
    if coverage[-1] >= typical - PARTIAL_COVERAGE_MARGIN:
        # Too close to typical to exclude (a cut on the 26th looks like a quiet
        # last week), so only report the shortfall, and never a period it fills.
        shortfall = coverage[-1] < typical - SHORT_COVERAGE_MARGIN and covered < total
        return None, "", reach if shortfall else "", True

    return pd.Timestamp(last_seen.index[-1]), reach, "", True


# What the reach into a period is counted in, by grain: days for a week or
# longer, hours for a day, minutes for an hour, seconds for a minute.
REACH_UNITS = {"D": ("hours", 3_600), "h": ("minutes", 60), "min": ("seconds", 1)}


def _reach(reached: float, span: float, frequency: str) -> tuple[int, int, str]:
    """How far the data gets into a period, in whole units: (covered, total, "6 of 7 days").

    The unit the data stops in counts as covered, so data ending on Saturday
    covers 6 of a week's 7 days.
    """
    unit, size = REACH_UNITS.get(frequency, ("days", DAY_SECONDS))
    total = max(int(round(span / size)), 1)
    covered = min(max(int(reached // size) + 1, 1), total)
    return covered, total, f"{covered} of {total} {unit}"


def build_trend(
    dataframe: pd.DataFrame,
    roles: ColumnRoles,
    frequency: str | None = None,
    count_column: str | None = None,
) -> TrendSeries:
    """Aggregate the measure over a human-sized grain, on an even timeline.

    With ``count_column`` and no measure, each period holds the number of
    distinct values in that column: customers per month, not rows per month.
    """
    if not roles.date:
        return TrendSeries(frame=EMPTY_TREND.copy(), frequency=frequency or "M")

    counted = count_column if count_column and not roles.measure and count_column != roles.date else None
    columns = [roles.date] + ([roles.measure] if roles.measure else []) + ([counted] if counted else [])
    working = dataframe[columns].dropna(subset=[roles.date]).copy()
    if working.empty:
        return TrendSeries(frame=EMPTY_TREND.copy(), frequency=frequency or "M")
    # Switch to internal column names so a user column called "Period" can't
    # collide with the period buckets built below.
    working.columns = ["__date"] + (["__measure"] if roles.measure else []) + (["__count"] if counted else [])

    frequency = frequency or _period_frequency(working["__date"])
    working["Period"] = working["__date"].dt.to_period(frequency).dt.to_timestamp()

    partial_period, partial_coverage, short_coverage, completeness_checked = (
        _trailing_partial_period(working["__date"], working["Period"], frequency)
    )
    if partial_period is not None:
        remaining = working[working["Period"] < partial_period]
        if remaining["Period"].nunique() >= 2:
            working = remaining
        else:
            partial_period, partial_coverage = None, ""

    metric = metric_for(roles.measure, roles)
    if counted:
        result = working.groupby("Period")["__count"].nunique().rename("Value").reset_index()
    elif roles.measure:
        grouped = working.groupby("Period")["__measure"]
        # min_count=1 keeps an all-missing period NaN instead of summing it to zero.
        totals = grouped.sum(min_count=1) if metric.aggregation == "sum" else grouped.mean()
        result = totals.rename("Value").reset_index()
    else:
        result = working.groupby("Period", as_index=False).size().rename(columns={"size": "Value"})
    result = result.sort_values("Period").reset_index(drop=True)

    # Empty periods are zero for additive measures and NaN for averaged ones (no observation).
    # Fewer than three distinct dates give no cadence to fill against, so skip filling.
    fill_value = 0.0 if metric.additive else float("nan")
    if working["__date"].nunique() >= 3:
        result, filled = _fill_empty_periods(result, frequency, fill_value)
    else:
        filled = 0
    return TrendSeries(
        frame=result,
        frequency=frequency,
        filled_periods=filled,
        filled_as_zero=metric.additive,
        partial_period=partial_period,
        partial_coverage=partial_coverage,
        short_coverage=short_coverage,
        completeness_checked=completeness_checked,
    )


def _fill_empty_periods(
    trend: pd.DataFrame, frequency: str, fill_value: float = 0.0
) -> tuple[pd.DataFrame, int]:
    """Insert periods with no rows as ``fill_value`` so the timeline is evenly spaced.

    Returns the frame and the number of periods inserted.
    """
    if len(trend) < 2:
        return trend, 0

    complete = pd.period_range(
        pd.Period(trend["Period"].iloc[0], freq=frequency),
        pd.Period(trend["Period"].iloc[-1], freq=frequency),
        freq=frequency,
    ).to_timestamp()
    missing = len(complete) - len(trend)
    if missing <= 0:
        return trend, 0

    filled = (
        trend.set_index("Period")
        .reindex(complete, fill_value=fill_value)
        .rename_axis("Period")
        .reset_index()
    )
    return filled, missing


def trend_frame(
    dataframe: pd.DataFrame,
    roles: ColumnRoles,
    frequency: str | None = None,
) -> pd.DataFrame:
    """Period totals only, for callers that do not need the adjustments."""
    return build_trend(dataframe, roles, frequency).frame


def segment_frame(dataframe: pd.DataFrame, roles: ColumnRoles, limit: int | None = 12) -> pd.DataFrame:
    """Rank the grouping column's values by the measure, or by record count; ``limit=None`` keeps all."""
    if not roles.dimension:
        return pd.DataFrame(columns=["Segment", "Value"])

    working = dataframe[[roles.dimension] + ([roles.measure] if roles.measure else [])].copy()
    working.columns = ["__segment"] + (["__measure"] if roles.measure else [])
    # Keep unlabelled rows under their own label so shares add up to the full total.
    label = unlabelled_label(working["__segment"].dropna().unique())
    working["__segment"] = working["__segment"].astype(object).where(working["__segment"].notna(), label)
    if working.empty:
        return pd.DataFrame(columns=["Segment", "Value"])

    if roles.measure:
        grouped = working.groupby("__segment")["__measure"]
        # min_count=1: a group with no values (a station with no rain gauge) has
        # no total, not a total of zero.
        totals = (
            grouped.sum(min_count=1)
            if metric_for(roles.measure, roles).aggregation == "sum"
            else grouped.mean()
        )
        result = totals.rename("Value").rename_axis("Segment").reset_index()
    else:
        result = (
            working.groupby("__segment", as_index=False)
            .size()
            .rename(columns={"__segment": "Segment", "size": "Value"})
        )
    result = result.sort_values("Value", ascending=False)
    return (result if limit is None else result.head(limit)).reset_index(drop=True)


UNLABELLED_SEGMENT = "(not recorded)"


def unlabelled_label(existing) -> str:
    """Return a label for rows with no segment that doesn't clash with a real value.

    Some exports already contain "(not recorded)"; the label then steps to
    "(not recorded, 2)", "(not recorded, 3)" and so on until it is unused.
    """
    present = {str(value) for value in existing if value is not None}
    label = UNLABELLED_SEGMENT
    suffix = 2
    while label in present:
        label = f"{UNLABELLED_SEGMENT[:-1]}, {suffix})"
        suffix += 1
    return label


def segment_period_change(
    dataframe: pd.DataFrame, roles: ColumnRoles, frequency: str | None = None
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp] | None:
    """Per-segment totals for the latest two periods, or None when unavailable.

    Only segments with values in both periods are compared; the ones left out
    are in the frame's ``attrs["left_out"]``. ``frequency`` fixes the grain;
    without it the trend's own grain is used.
    """
    if not roles.date or not roles.measure or not roles.dimension:
        return None

    series = build_trend(dataframe, roles, frequency)
    trend = series.frame
    if len(trend) < 2:
        return None
    previous_period = trend.iloc[-2]["Period"]
    current_period = trend.iloc[-1]["Period"]
    # Reuse the trend's grain; re-deriving it from these rows can pick a different
    # one, and the compared periods would not line up.
    frequency = series.frequency
    working = dataframe[[roles.date, roles.measure, roles.dimension]].copy()
    working.columns = ["__date", "__measure", "__segment"]
    working = working.dropna(subset=["__date", "__measure"])
    # Keep unlabelled rows so segment changes add up to the trend's change.
    working["__segment"] = working["__segment"].astype(object).where(
        working["__segment"].notna(), unlabelled_label(working["__segment"].dropna().unique())
    )
    working["Period"] = working["__date"].dt.to_period(frequency).dt.to_timestamp()
    comparison = working[working["Period"].isin([previous_period, current_period])]
    metric = metric_for(roles.measure, roles)
    grouped = (
        comparison.groupby(["__segment", "Period"])["__measure"]
        .agg(metric.aggregation)
        .unstack()
    )
    grouped.index.name = roles.dimension
    if previous_period not in grouped or current_period not in grouped:
        return None
    # Only segments with values in both periods are compared. One that starts or
    # stops reporting would otherwise read as rising from, or falling to, zero
    # ("100% of the movement"), and an averaged one has no value to compare.
    # The ones left out are recorded on the frame for the caption.
    unmatched = grouped[[previous_period, current_period]].isna().any(axis=1)
    left_out = tuple(grouped.index[unmatched])
    grouped = grouped.loc[~unmatched].copy()
    if grouped.empty:
        return None
    grouped.attrs["left_out"] = left_out

    grouped["Change"] = grouped[current_period] - grouped[previous_period]
    return grouped, previous_period, current_period


def driver_frame(dataframe: pd.DataFrame, roles: ColumnRoles, limit: int = 9) -> pd.DataFrame:
    """Waterfall-ready per-segment change between the latest two periods."""
    result = segment_period_change(dataframe, roles)
    if result is None:
        return pd.DataFrame(columns=["Segment", "Change"])
    grouped, _, _ = result
    changes = grouped["Change"].sort_values(key=lambda values: values.abs(), ascending=False)
    top = changes.head(limit)
    frame = pd.DataFrame({"Segment": top.index.astype(str), "Change": top.to_numpy(dtype=float)})
    remainder = float(changes.iloc[limit:].sum())
    # Changes in segment averages don't sum, so an averaged measure gets no "Other segments" bar.
    if len(changes) > limit and remainder and metric_for(roles.measure, roles).additive:
        other = pd.DataFrame({"Segment": ["Other segments"], "Change": [remainder]})
        frame = pd.concat([frame, other], ignore_index=True)
    return frame


def heatmap_frame(
    dataframe: pd.DataFrame,
    roles: ColumnRoles,
    limit: int = 8,
    frequency: str | None = None,
) -> pd.DataFrame:
    """Segment × period matrix of the measure (or row counts) for the top segments.

    Pass the trend's ``frequency`` so both charts share a time axis; re-deriving
    it from this filtered subset can pick a different grain.
    """
    if not roles.date or not roles.dimension:
        return pd.DataFrame()

    top_segments = segment_frame(dataframe, roles, limit=limit)["Segment"]
    columns = [roles.date, roles.dimension] + ([roles.measure] if roles.measure else [])
    working = dataframe[columns].dropna(subset=[roles.date]).copy()
    working.columns = ["__date", "__segment"] + (["__measure"] if roles.measure else [])
    working["__segment"] = working["__segment"].astype(object).where(
        working["__segment"].notna(), unlabelled_label(working["__segment"].dropna().unique())
    )
    working = working[working["__segment"].isin(top_segments)]
    if working.empty:
        return pd.DataFrame()

    frequency = frequency or _period_frequency(working["__date"])
    working["Period"] = working["__date"].dt.to_period(frequency).dt.to_timestamp()
    if roles.measure:
        metric = metric_for(roles.measure, roles)
        cells = working.groupby(["__segment", "Period"])["__measure"]
        grouped = cells.sum(min_count=1) if metric.aggregation == "sum" else cells.mean()
        # A cell with no rows is zero for a total and blank for an average; a
        # cell whose rows hold no values is blank either way.
        pivot = grouped.unstack(fill_value=0) if metric.additive else grouped.unstack()
        order = pivot.sum(axis=1) if metric.additive else pivot.mean(axis=1)
    else:
        pivot = working.groupby(["__segment", "Period"]).size().unstack(fill_value=0)
        order = pivot.sum(axis=1)
    pivot.index.name = roles.dimension
    return pivot.loc[order.sort_values(ascending=False).index]


@dataclass(frozen=True)
class MatchedChange:
    """The latest change measured on the groups present in both periods.

    Set when groups start or stop reporting between the two periods, so the
    pooled change partly reflects which groups were there.
    """

    previous: float
    current: float
    groups: int
    #: Groups with values in only the previous or only the latest period.
    stopped: tuple
    started: tuple
    #: Groups with values in both periods.
    kept: tuple = ()


# Places and outlets that keep reporting over time, beside the entities the
# schema names (station, site, plot, subject): their presence in a period is
# reporting, while a per-row condition (a weather code, a status, the pollutant
# that peaked) appears when it happens, so its mix is part of the outcome.
REPORTING_UNIT_WORDS = frozenset({
    "region", "city", "town", "country", "state", "county", "district", "province", "municipality",
    "store", "shop", "branch", "outlet", "location", "area", "zone", "sensor", "gauge", "device",
    "facility", "plant", "hospital", "clinic", "school", "farm", "field", "lake", "river", "basin",
    "catchment", "beach", "reef", "transect", "quadrat", "well", "borehole", "buoy", "monitor",
})
# Share of a date's rows a unit may repeat on and still read as one report each,
# and share of dates that must hold several units for them to read as side by side.
SIDE_BY_SIDE_DATES = 0.5
ONE_REPORT_SHARE = 0.95


def reporting_unit(dataframe: pd.DataFrame, roles: ColumnRoles, column: str) -> bool:
    """Whether a grouping names units that report over time (stations, stores), not a per-row condition.

    Only for such units may a change be measured on the ones present in both
    periods. A unit is named like one, or reports side by side with the
    others: on most dates several of them have a row, each at most one.
    Other category codes and yes/no flags are conditions.
    """
    if not roles.date:
        return False
    words = set(re.findall(r"[a-z]+", str(column).casefold()))
    if is_entity_name(column) or words & REPORTING_UNIT_WORDS:
        return True
    if column in roles.factors or column in roles.flags:
        return False
    working = dataframe[[roles.date, column]].dropna()
    if working.empty:
        return False
    per_date = working.groupby(roles.date)[column].agg(["size", "nunique"])
    shared = per_date[per_date["size"] > 1]
    if len(shared) < SIDE_BY_SIDE_DATES * len(per_date):
        return False
    return bool((shared["nunique"] == shared["size"]).mean() >= ONE_REPORT_SHARE)


def _period_groups(
    working: pd.DataFrame, previous_period: pd.Timestamp, current_period: pd.Timestamp
) -> tuple[set, set]:
    """The groups with a value (or, with no measure, a row) in each of the two periods."""
    valued = working.dropna(subset=["__measure"]) if "__measure" in working else working
    return (
        set(valued.loc[valued["Period"] == previous_period, "__segment"]),
        set(valued.loc[valued["Period"] == current_period, "__segment"]),
    )


def matched_period_change(
    dataframe: pd.DataFrame, roles: ColumnRoles, series: TrendSeries
) -> MatchedChange | None:
    """Recompute the trend's latest change on the groups present in both periods.

    Returns None without a grouping column, with fewer than two periods, or
    when the same groups report in both periods. With no group in both, the
    values are NaN and ``groups`` is 0.
    """
    trend = series.frame
    if not roles.dimension or not roles.date or len(trend) < 2:
        return None
    previous_period, current_period = trend.iloc[-2]["Period"], trend.iloc[-1]["Period"]
    columns = [roles.date, roles.dimension] + ([roles.measure] if roles.measure else [])
    working = dataframe[columns].dropna(subset=[roles.date, roles.dimension])
    working.columns = ["__date", "__segment"] + (["__measure"] if roles.measure else [])
    working = working.assign(Period=working["__date"].dt.to_period(series.frequency).dt.to_timestamp())
    working = working[working["Period"].isin([previous_period, current_period])]
    before, after = _period_groups(working, previous_period, current_period)
    both = before & after
    if before == after:
        return None
    matched = working[working["__segment"].isin(both)]
    if roles.measure:
        grouped = matched.groupby("Period")["__measure"]
        metric = metric_for(roles.measure, roles)
        values = grouped.sum(min_count=1) if metric.aggregation == "sum" else grouped.mean()
    else:
        values = matched.groupby("Period").size().astype(float)
    return MatchedChange(
        previous=float(values.get(previous_period, np.nan)),
        current=float(values.get(current_period, np.nan)),
        groups=len(both),
        stopped=tuple(sorted(before - after, key=str)),
        started=tuple(sorted(after - before, key=str)),
        kept=tuple(sorted(both, key=str)),
    )


def group_coverage(dataframe: pd.DataFrame, roles: ColumnRoles) -> pd.DataFrame:
    """First and last date with a value for each group, as columns From and To."""
    if not roles.date or not roles.dimension:
        return pd.DataFrame(columns=["From", "To"])
    columns = [roles.date, roles.dimension] + ([roles.measure] if roles.measure else [])
    working = dataframe[columns].dropna()
    dates = working.groupby(roles.dimension)[roles.date]
    return pd.DataFrame({"From": dates.min(), "To": dates.max()})


@dataclass(frozen=True)
class SeasonalContext:
    """The latest period set against the same time of year in earlier years."""

    #: The same period a year earlier and its value, when the trend has one.
    year_ago: pd.Timestamp | None
    year_ago_value: float
    #: Changes into the latest period's month or quarter in earlier years,
    #: from the period before each.
    usual_changes: tuple[float, ...]
    #: Each earlier period's change from the same period a year before it, in
    #: the same terms, so the latest level can be judged against last year's.
    year_over_year: tuple[float, ...] = ()


def seasonal_context(
    series: TrendSeries, *, relative: bool, tested: bool = True
) -> SeasonalContext | None:
    """Compare the latest period with the same time of year, when the trend has a yearly pattern.

    A winter-to-spring rise is not news in a series that rises every spring, so
    the latest change is set beside the same change in earlier years, and the
    latest value beside the same period a year earlier. ``relative`` gives the
    changes as percentages (for a total) rather than differences. Returns None
    for a grain with no yearly cycle or, while ``tested``, a series without a
    clear yearly pattern; pass False to look at the same time of year anyway
    before calling a move unusual.
    """
    trend = observed_periods(series.frame)
    if series.frequency not in PERIODS_PER_YEAR or len(trend) < 3:
        return None
    periods = pd.DatetimeIndex(pd.to_datetime(trend["Period"]))
    values = trend["Value"].to_numpy(dtype=float)
    line = fit_trendline(periods, values, series.frequency)
    if tested and yearly_pattern(periods, line.residuals(values), grain=series.frequency) is None:
        return None
    by_period = pd.Series(values, index=periods)
    latest = periods[-1]
    year_ago = latest - pd.DateOffset(years=1)
    step = pd.DateOffset(months=12 // PERIODS_PER_YEAR[series.frequency])
    changes = []
    for period in periods[:-1]:
        before = period - step
        if season_numbers(pd.DatetimeIndex([period]), series.frequency)[0] != season_numbers(
            pd.DatetimeIndex([latest]), series.frequency
        )[0] or before not in by_period.index:
            continue
        start, end = by_period[before], by_period[period]
        if relative:
            if start == 0:
                continue
            changes.append((end - start) / abs(start) * 100)
        else:
            changes.append(end - start)
    has_year_ago = year_ago in by_period.index
    earlier = by_period.reindex(periods[:-1] - pd.DateOffset(years=1)).to_numpy(dtype=float)
    later = values[:-1]
    paired = np.isfinite(earlier) & ((earlier != 0) if relative else np.ones(len(earlier), dtype=bool))
    yearly = (later - earlier)[paired]
    if relative:
        yearly = yearly / np.abs(earlier[paired]) * 100
    return SeasonalContext(
        year_ago=year_ago if has_year_ago else None,
        year_ago_value=float(by_period[year_ago]) if has_year_ago else float("nan"),
        usual_changes=tuple(float(change) for change in changes),
        year_over_year=tuple(float(change) for change in yearly),
    )


# Share of rows that may repeat a unit within a period, or lack a unit, in a
# fixed design: a stray double entry does not undo the design.
DESIGN_SLACK = 0.02
# Periods a design needs before its row counts read as fixed rather than chance.
MIN_DESIGN_PERIODS = 3
# The fewest units in a period, as a share of the most, for the count to be steady by design.
STEADY_DESIGN_SHARE = 0.8


@dataclass(frozen=True)
class FixedDesign:
    """Rows per period set by a design: each unit observed at most once a period, period after period."""

    unit: str
    frequency: str
    periods: int
    units: int
    fewest: int
    most: int

    @property
    def sentence(self) -> str:
        grain = DESIGN_GRAINS.get(self.frequency, "period")
        observed = f"{self.most:,}" if self.fewest == self.most else f"{self.fewest:,} to {self.most:,}"
        return (
            f"Each {sentence_case(self.unit)} appears at most once a {grain}: {observed} of the "
            f"{self.units:,} in each of the {self.periods:,} {grain}s. Rows per {grain} follow that design, "
            "so they are not trended or forecast."
        )


DESIGN_GRAINS = {"D": "day", "W": "week", "M": "month", "Q": "quarter", "Y": "year"}


def fixed_design(
    dataframe: pd.DataFrame, roles: ColumnRoles, frequency: str | None = None
) -> FixedDesign | None:
    """The design behind a row count, when each unit appears at most once a period and keeps recurring.

    With no measure, the brief would trend the number of rows per period; when
    the same plots or subjects are observed once every year, that number is
    set in advance and says nothing. Returns None with a measure, without a
    unit column (``compare.entity_column``), when units repeat within a period
    or do not recur across most periods, or when their number moves from
    period to period, as active customers do.
    """
    if roles.measure or not roles.date:
        return None
    unit = entity_column(dataframe, roles)
    if unit is None:
        return None
    working = dataframe[[roles.date, unit]].dropna(subset=[roles.date])
    if working.empty or working[unit].isna().mean() > DESIGN_SLACK:
        return None
    frequency = frequency or _period_frequency(working[roles.date])
    periods = working[roles.date].dt.to_period(frequency)
    pairs = pd.DataFrame({"period": periods, "unit": working[unit]}).dropna()
    if pairs.duplicated().mean() > DESIGN_SLACK:
        return None
    count = periods.nunique()
    seen = pairs.drop_duplicates().groupby("unit")["period"].nunique()
    if count < MIN_DESIGN_PERIODS or float(seen.median()) < max(MIN_DESIGN_PERIODS, 0.5 * count):
        return None
    per_period = pairs.drop_duplicates().groupby("period")["unit"].nunique()
    if per_period.min() < STEADY_DESIGN_SHARE * per_period.max():
        # A count that moves (customers lost and won each month) is worth following.
        return None
    return FixedDesign(
        unit=unit,
        frequency=frequency,
        periods=int(count),
        units=int(seen.size),
        fewest=int(per_period.min()),
        most=int(per_period.max()),
    )


# A gap between sampling dates of at least this many days, and this many times
# the usual gap, ends a sampling trip; a few trips with most periods between
# them empty are campaigns, not a timeline.
CAMPAIGN_GAP_DAYS = 14
CAMPAIGN_GAP_RATIO = 5
MIN_CAMPAIGNS = 3
CAMPAIGN_EMPTY_SHARE = 0.5


@dataclass(frozen=True)
class SamplingCampaigns:
    """Dates that cluster into a few sampling trips with long empty stretches between them."""

    #: The trip each row belongs to, numbered from 0 in time order (NaN for a row with no date).
    labels: pd.Series
    starts: tuple[pd.Timestamp, ...]
    ends: tuple[pd.Timestamp, ...]
    sizes: tuple[int, ...]
    frequency: str
    periods: int
    empty_periods: int

    def label(self, trip: int) -> str:
        """A trip's dates for a sentence: "16–19 Jun 2003", "26 Apr – 03 May 2004", "09 Sep 2003"."""
        start, end = self.starts[trip], self.ends[trip]
        if start.normalize() == end.normalize():
            return f"{start:%d %b %Y}"
        if (start.year, start.month) == (end.year, end.month):
            return f"{start:%d}–{end:%d %b %Y}"
        return f"{start:%d %b} – {end:%d %b %Y}"


def sampling_campaigns(
    dataframe: pd.DataFrame, roles: ColumnRoles, frequency: str | None = None
) -> SamplingCampaigns | None:
    """The sampling trips behind a table's dates, when a calendar grid would be mostly empty.

    Distinct dates are split into trips at gaps of at least CAMPAIGN_GAP_DAYS
    and CAMPAIGN_GAP_RATIO times the median gap. Ten field trips of a few
    days each over three years, at a monthly grain, leave two months in three
    empty, so a trend or forecast over months would run over gaps. Returns
    None with fewer than MIN_CAMPAIGNS trips or when most periods hold data.
    """
    if not roles.date:
        return None
    dates = dataframe[roles.date]
    distinct = dates.dropna().drop_duplicates().sort_values()
    if len(distinct) < 2 * MIN_CAMPAIGNS:
        return None
    gaps = distinct.diff().dt.total_seconds() / DAY_SECONDS
    breaks = (gaps > CAMPAIGN_GAP_DAYS) & (gaps > CAMPAIGN_GAP_RATIO * float(gaps.median()))
    numbers = breaks.cumsum().to_numpy()
    if numbers[-1] + 1 < MIN_CAMPAIGNS:
        return None
    frequency = frequency or _period_frequency(distinct)
    stamps = distinct.dt.to_period(frequency)
    periods = len(pd.period_range(stamps.min(), stamps.max(), freq=frequency))
    empty = periods - stamps.nunique()
    if empty < CAMPAIGN_EMPTY_SHARE * periods:
        return None
    labels = dates.map(pd.Series(numbers, index=distinct.to_numpy()))
    trips = pd.DataFrame({"date": distinct.to_numpy(), "trip": numbers}).groupby("trip")["date"]
    return SamplingCampaigns(
        labels=labels,
        starts=tuple(pd.Timestamp(stamp) for stamp in trips.min()),
        ends=tuple(pd.Timestamp(stamp) for stamp in trips.max()),
        sizes=tuple(int(size) for size in labels.value_counts().sort_index()),
        frequency=frequency,
        periods=periods,
        empty_periods=int(empty),
    )


# Periods in a year at each grain, for judging whether a year is complete.
PERIODS_IN_YEAR = {"D": 365, "W": 52, "M": 12, "Q": 4, "Y": 1}
# Share of its periods with a value a year needs to count as complete: a year
# missing its summer would read as a cold year.
COMPLETE_YEAR_SHARE = 0.9


def yearly_values(series: TrendSeries, *, additive: bool) -> pd.Series:
    """The measure per complete year, indexed by year, from the trend's periods.

    A year counts when COMPLETE_YEAR_SHARE of its periods hold a value. Its
    value is the total of its periods for an additive measure and their mean
    otherwise, so each season weighs the same however many rows it holds.
    Grains finer than a day give an empty series.
    """
    per_year = PERIODS_IN_YEAR.get(series.frequency)
    trend = observed_periods(series.frame)
    if per_year is None or trend.empty:
        return pd.Series(dtype=float)
    years = pd.DatetimeIndex(pd.to_datetime(trend["Period"])).year
    values = pd.Series(trend["Value"].to_numpy(dtype=float), index=years)
    counts = values.groupby(level=0).count()
    complete = counts.index[counts >= COMPLETE_YEAR_SHARE * per_year]
    grouped = values[values.index.isin(complete)].groupby(level=0)
    return (grouped.sum() if additive else grouped.mean()).sort_index()


# Month names a column may carry in a table with one column per month.
MONTH_HEADERS = tuple(
    (name[:3], name) for name in (
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    )
)


def month_columns(dataframe: pd.DataFrame) -> tuple[str, ...]:
    """The twelve numeric columns named after the months, in calendar order, or ().

    A table with one row per year and one column per month (JAN to DEC, or
    January to December) holds one monthly series laid out wide.
    """
    found: dict[int, str] = {}
    for column in dataframe.columns:
        name = str(column).strip().casefold().rstrip(".")
        for month, (short, full) in enumerate(MONTH_HEADERS):
            if name in (short, full) or (month == 8 and name == "sept"):
                if month not in found and pd.api.types.is_numeric_dtype(dataframe[column]):
                    found[month] = column
                break
    return tuple(found[month] for month in range(12)) if len(found) == 12 else ()


def months_recorded(
    dataframe: pd.DataFrame, roles: ColumnRoles, period: pd.Timestamp
) -> tuple[int, int] | None:
    """How many of a year's month columns hold a value in its row: (recorded, 12), or None.

    None unless the table has a column per month and one row for that year.
    """
    months = month_columns(dataframe)
    if not months or not roles.date:
        return None
    rows = dataframe.loc[dataframe[roles.date].dt.year == pd.Timestamp(period).year, list(months)]
    if len(rows) != 1:
        return None
    return int(rows.iloc[0].notna().sum()), len(months)


def months_long(dataframe: pd.DataFrame, date: str) -> pd.DataFrame:
    """A table with a column per month as one row per month: Date (the 1st) and Value.

    Empty when the table has no column per month. Rows without a year, and
    months with no value, are left out.
    """
    months = month_columns(dataframe)
    if not months:
        return pd.DataFrame({"Date": pd.Series(dtype="datetime64[ns]"), "Value": pd.Series(dtype=float)})
    working = dataframe[[date, *months]].dropna(subset=[date])
    years = working[date].dt.year.to_numpy()
    values = working[list(months)].to_numpy(dtype=float)
    stamps = pd.to_datetime(
        {"year": np.repeat(years, 12), "month": np.tile(np.arange(1, 13), len(years)), "day": 1}
    )
    long = pd.DataFrame({"Date": stamps, "Value": values.ravel()})
    return long.dropna(subset=["Value"]).sort_values("Date").reset_index(drop=True)
