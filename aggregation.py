"""Turning rows into the frames every other module reads.

This is the only place that decides what a period is. Getting that wrong is
quiet and expensive: too fine a grain invents periods nobody measured, a
missing period shortens the timeline, and a trailing period the data stops
part-way through reads as a collapse. Those decisions are made once here,
recorded on the TrendSeries, and reused by the trend, driver, anomaly, and
forecast paths so they cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from formatting import format_period
from metrics import resolve_metric
from schema import ColumnRoles

GRAIN_ORDER = ("W", "M", "Q", "Y")


def _grain_for_span(span_days: int) -> str:
    if span_days <= 120:
        return "W"
    if span_days <= 900:
        return "M"
    return "Q"


def _grain_for_cadence(dates: pd.Series) -> str:
    """The finest grain the data is actually dense enough to fill."""
    unique = dates.drop_duplicates()
    if len(unique) < 3:
        return "W"
    spacing = np.diff(unique.sort_values().to_numpy()).astype("timedelta64[D]").astype(int)
    typical = float(np.median(spacing)) if spacing.size else 0.0
    if typical <= 10:
        return "W"
    if typical <= 45:
        return "M"
    if typical <= 140:
        return "Q"
    # Four annual figures are four years, not sixteen quarters twelve of
    # which nobody measured and which were being filled with zero.
    return "Y"


def measure_aggregation(measure: str | None) -> str:
    """How a measure combines across rows: rates average, amounts add."""
    return resolve_metric(measure).aggregation


def _period_frequency(date_series: pd.Series) -> str:
    """Pick a human-sized grain the data can actually populate.

    The observed span suggests a grain, but so does how often the data is
    recorded. Five monthly readings spanning four months must not be charted
    as seventeen weeks, twelve of which nobody measured. The coarser of the
    two answers wins.
    """
    dates = date_series.dropna()
    if dates.empty:
        return "M"

    span_days = max((dates.max() - dates.min()).days, 0)
    return max(_grain_for_span(span_days), _grain_for_cadence(dates), key=GRAIN_ORDER.index)


def preferred_frequency(date_series: pd.Series) -> str:
    """Pick a human-sized period grain for the observed dates."""
    return _period_frequency(date_series)


EMPTY_TREND = pd.DataFrame({"Period": pd.Series(dtype="datetime64[ns]"), "Value": pd.Series(dtype=float)})
PARTIAL_COVERAGE_MARGIN = 0.2
# Below typical by this much, but not enough to exclude: worth saying out loud.
SHORT_COVERAGE_MARGIN = 0.05
MIN_PERIODS_FOR_PARTIAL_CHECK = 4


@dataclass(frozen=True)
class TrendSeries:
    """Period totals, plus what had to be assumed to line them up.

    Two adjustments happen before any trend, anomaly, or forecast maths sees
    the numbers, and both are recorded here rather than applied silently.
    """

    frame: pd.DataFrame
    frequency: str
    filled_periods: int = 0
    partial_period: pd.Timestamp | None = None
    partial_coverage: str = ""
    # A period the data stops part-way through, but not far enough through for
    # exclusion to be safe -- "no orders for three days" looks the same from
    # here. Saying so beats guessing either way.
    short_coverage: str = ""
    # False when there were too few periods to judge completeness at all, so
    # nothing downstream may call the last one complete.
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
            notes.append(
                f"{self.filled_periods} {plural} with no rows counted as zero, keeping the "
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

    Compares how much of each period the data actually reaches with how much
    it typically reaches. An extract cut on the 12th covers a third of its
    month while every earlier month covers essentially all of one; a quiet
    final week is nowhere near that different, and is left alone.
    """
    last_seen = dates.groupby(periods).max().sort_index()
    if len(last_seen) < MIN_PERIODS_FOR_PARTIAL_CHECK:
        # Too few periods to know what "typical coverage" looks like here.
        return None, "", "", False

    starts, ends = _period_bounds(pd.DatetimeIndex(last_seen.index), frequency)
    spans = (ends - starts).to_numpy().astype("timedelta64[s]").astype(float)
    reached = (last_seen.to_numpy() - starts.to_numpy()).astype("timedelta64[s]").astype(float)
    coverage = np.divide(reached, spans, out=np.zeros_like(reached), where=spans > 0)

    period_days = max(int(round(spans[-1] / 86_400)), 1)
    covered_days = max(int(round(reached[-1] / 86_400)) + 1, 1)
    reach = f"{covered_days} of {period_days} days"

    typical = float(np.median(coverage[:-1]))
    if coverage[-1] >= typical - PARTIAL_COVERAGE_MARGIN:
        # Not short enough to exclude safely: an extract cut on the 26th and a
        # quiet last week look identical from here. Report the shortfall rather
        # than silently presenting the drop as a real one.
        short = reach if coverage[-1] < typical - SHORT_COVERAGE_MARGIN else ""
        return None, "", short, True

    return pd.Timestamp(last_seen.index[-1]), reach, "", True


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
    # Internal names from here on. A measure the file calls "Period" was being
    # overwritten by the period buckets built below, then summed as datetimes.
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

    metric = resolve_metric(roles.measure)
    if counted:
        result = working.groupby("Period")["__count"].nunique().rename("Value").reset_index()
    elif roles.measure:
        grouped = working.groupby("Period")["__measure"]
        # A period whose every value is missing is a missing period, not a
        # period that measured zero: sum(min_count=1) keeps it NaN.
        totals = grouped.sum(min_count=1) if metric.aggregation == "sum" else grouped.mean()
        result = totals.rename("Value").reset_index()
    else:
        result = working.groupby("Period", as_index=False).size().rename(columns={"size": "Value"})
    result = result.sort_values("Period").reset_index(drop=True)

    # A month with no rows is a month of zero sales, but it is not a month
    # of zero conversion rate -- there is no observation, so the gap stays
    # empty. And with fewer than three distinct dates there is no cadence to
    # fill against at all; two month-end readings were being spread into
    # three invented zero weeks.
    fill_value = 0.0 if metric.additive else float("nan")
    if working["__date"].nunique() >= 3:
        result, filled = _fill_empty_periods(result, frequency, fill_value)
    else:
        filled = 0
    return TrendSeries(
        frame=result,
        frequency=frequency,
        filled_periods=filled,
        partial_period=partial_period,
        partial_coverage=partial_coverage,
        short_coverage=short_coverage,
        completeness_checked=completeness_checked,
    )


def _fill_empty_periods(
    trend: pd.DataFrame, frequency: str, fill_value: float = 0.0
) -> tuple[pd.DataFrame, int]:
    """Materialise periods with no rows as zero, so gaps stop bending the fit.

    A month in which nothing was sold is a month of zero sales, not a month
    that never happened. Dropping it shortens the timeline and flattens every
    slope fitted through it.
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


def segment_frame(dataframe: pd.DataFrame, roles: ColumnRoles, limit: int = 12) -> pd.DataFrame:
    """Rank the selected business segment by the selected measure or record count."""
    if not roles.dimension:
        return pd.DataFrame(columns=["Segment", "Value"])

    working = dataframe[[roles.dimension] + ([roles.measure] if roles.measure else [])].copy()
    working.columns = ["__segment"] + (["__measure"] if roles.measure else [])
    # Rows with no label still hold the measure. Dropping them made every
    # share on the page a share of the labelled rows only, and a file that was
    # mostly unlabelled produced no segment evidence at all.
    label = unlabelled_label(working["__segment"].dropna().unique())
    working["__segment"] = working["__segment"].astype(object).where(working["__segment"].notna(), label)
    if working.empty:
        return pd.DataFrame(columns=["Segment", "Value"])

    if roles.measure:
        result = working.groupby("__segment", as_index=False)["__measure"].agg(
            measure_aggregation(roles.measure)
        )
        result = result.rename(columns={"__segment": "Segment", "__measure": "Value"})
    else:
        result = (
            working.groupby("__segment", as_index=False)
            .size()
            .rename(columns={"__segment": "Segment", "size": "Value"})
        )
    return result.sort_values("Value", ascending=False).head(limit).reset_index(drop=True)


UNLABELLED_SEGMENT = "(not recorded)"


def unlabelled_label(existing) -> str:
    """The name shown for rows with no segment, chosen not to be a real one.

    The file can already contain the literal "(not recorded)" -- some exports
    write exactly that -- and folding genuinely blank rows into it merged two
    different populations into one number. Step the label until it is unused.
    """
    present = {str(value) for value in existing if value is not None}
    label = UNLABELLED_SEGMENT
    suffix = 2
    while label in present:
        label = f"{UNLABELLED_SEGMENT[:-1]}, {suffix})"
        suffix += 1
    return label


def segment_period_change(
    dataframe: pd.DataFrame, roles: ColumnRoles
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp] | None:
    """Per-segment totals for the latest two periods, or None when unavailable."""
    if not roles.date or not roles.measure or not roles.dimension:
        return None

    series = build_trend(dataframe, roles)
    trend = series.frame
    if len(trend) < 2:
        return None
    previous_period = trend.iloc[-2]["Period"]
    current_period = trend.iloc[-1]["Period"]
    # The grain has to come from the trend itself: deriving it again from a
    # different subset of rows can land on another grain, and then the two
    # periods being compared exist in one view and not the other.
    frequency = series.frequency
    working = dataframe[[roles.date, roles.measure, roles.dimension]].copy()
    working.columns = ["__date", "__measure", "__segment"]
    working = working.dropna(subset=["__date", "__measure"])
    # A row with no segment label still carries measure value. Dropping it
    # here and keeping it in the trend meant the drivers were shares of a
    # movement they did not add up to.
    working["__segment"] = working["__segment"].astype(object).where(
        working["__segment"].notna(), unlabelled_label(working["__segment"].dropna().unique())
    )
    working["Period"] = working["__date"].dt.to_period(frequency).dt.to_timestamp()
    comparison = working[working["Period"].isin([previous_period, current_period])]
    metric = resolve_metric(roles.measure)
    grouped = (
        comparison.groupby(["__segment", "Period"])["__measure"]
        .agg(metric.aggregation)
        .unstack()
    )
    grouped.index.name = roles.dimension
    if previous_period not in grouped or current_period not in grouped:
        return None
    if metric.additive:
        # A segment with no rows in a period sold nothing in it: zero.
        grouped = grouped.fillna(0.0)
    else:
        # A segment with no rows in a period has no average in it. Filling
        # with zero made a region that first appears this month "rise from
        # 0.0% to 31.0%", so a segment has to be present on both sides to
        # have a change at all.
        grouped = grouped.dropna(subset=[previous_period, current_period])
        if grouped.empty:
            return None

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
    # Changes in segment averages do not sum to anything, so there is no
    # "other" bar to add up for a rate -- only the segments shown.
    if len(changes) > limit and remainder and resolve_metric(roles.measure).additive:
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

    The grain is taken from the caller so the heatmap shares a time axis with
    the trend beside it. Re-deriving it from this filtered subset landed on a
    different grain, and the two charts on one page then disagreed about what
    a column meant.
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
        pivot = (
            working.groupby(["__segment", "Period"])["__measure"]
            .agg(measure_aggregation(roles.measure))
            .unstack(fill_value=0)
        )
    else:
        pivot = working.groupby(["__segment", "Period"]).size().unstack(fill_value=0)
    pivot.index.name = roles.dimension
    return pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
