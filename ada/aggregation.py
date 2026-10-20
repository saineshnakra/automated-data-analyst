"""Aggregate rows into the period and segment frames the other modules read.

Period grain, empty-period filling and trailing partial periods are decided once
here and recorded on ``TrendSeries``, so the trend, driver, anomaly and forecast
paths agree on what a period is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .formatting import format_period
from .metrics import metric_for
from .schema import ColumnRoles
from .timeseries import (
    PERIODS_PER_YEAR,
    fit_trendline,
    observed_periods,
    season_numbers,
    yearly_pattern,
)

GRAIN_ORDER = ("s", "min", "h", "D", "W", "M", "Q", "Y")
# Grains finer than a day, open only to timestamps that carry a time of day.
SUB_DAY_GRAINS = ("s", "min", "h", "D")
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


def seasonal_context(series: TrendSeries, *, relative: bool) -> SeasonalContext | None:
    """Compare the latest period with the same time of year, when the trend has a yearly pattern.

    A winter-to-spring rise is not news in a series that rises every spring, so
    the latest change is set beside the same change in earlier years, and the
    latest value beside the same period a year earlier. ``relative`` gives the
    changes as percentages (for a total) rather than differences. Returns None
    for a grain with no yearly cycle or a series without a clear yearly pattern.
    """
    trend = observed_periods(series.frame)
    if series.frequency not in PERIODS_PER_YEAR or len(trend) < 3:
        return None
    periods = pd.DatetimeIndex(pd.to_datetime(trend["Period"]))
    values = trend["Value"].to_numpy(dtype=float)
    if yearly_pattern(periods, fit_trendline(periods, values).residuals(values)) is None:
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
    return SeasonalContext(
        year_ago=year_ago if has_year_ago else None,
        year_ago_value=float(by_period[year_ago]) if has_year_ago else float("nan"),
        usual_changes=tuple(float(change) for change in changes),
    )
