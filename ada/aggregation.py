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
from .metrics import resolve_metric
from .schema import ColumnRoles

GRAIN_ORDER = ("W", "M", "Q", "Y")


def _grain_for_span(span_days: int) -> str:
    if span_days <= 120:
        return "W"
    if span_days <= 900:
        return "M"
    return "Q"


def _grain_for_cadence(dates: pd.Series) -> str:
    """Finest grain the recording cadence is dense enough to fill."""
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
    # Annual readings stay yearly; quarters would add three zero-filled periods per year.
    return "Y"


def measure_aggregation(measure: str | None) -> str:
    """How a measure combines across rows: rates average, amounts add."""
    return resolve_metric(measure).aggregation


def _period_frequency(date_series: pd.Series) -> str:
    """Pick a human-sized grain the data can populate.

    Takes the coarser of the span-based and cadence-based grains: five monthly
    readings over four months stay monthly instead of becoming seventeen weeks.
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
    #: False when they were left as NaN (a rate has no observation to average).
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
            # A rate's gaps are NaN, not zero, so the caption must match the break in the line.
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

    starts, ends = _period_bounds(pd.DatetimeIndex(last_seen.index), frequency)
    spans = (ends - starts).to_numpy().astype("timedelta64[s]").astype(float)
    reached = (last_seen.to_numpy() - starts.to_numpy()).astype("timedelta64[s]").astype(float)
    coverage = np.divide(reached, spans, out=np.zeros_like(reached), where=spans > 0)

    period_days = max(int(round(spans[-1] / 86_400)), 1)
    covered_days = max(int(round(reached[-1] / 86_400)) + 1, 1)
    reach = f"{covered_days} of {period_days} days"

    typical = float(np.median(coverage[:-1]))
    if coverage[-1] >= typical - PARTIAL_COVERAGE_MARGIN:
        # Too close to typical to exclude (a cut on the 26th looks like a quiet
        # last week), so only report the shortfall.
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

    metric = resolve_metric(roles.measure)
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

    # Empty periods are zero for additive measures and NaN for rates (no observation).
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


def segment_frame(dataframe: pd.DataFrame, roles: ColumnRoles, limit: int = 12) -> pd.DataFrame:
    """Rank the selected business segment by the selected measure or record count."""
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
        # No rows in a period means zero for an additive measure.
        grouped = grouped.fillna(0.0)
    else:
        # A rate has no value in a period without rows, so keep only segments in
        # both; zero-filling would show a new region rising "from 0.0% to 31.0%".
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
    # Changes in segment averages don't sum, so a rate gets no "Other segments" bar.
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
        pivot = (
            working.groupby(["__segment", "Period"])["__measure"]
            .agg(measure_aggregation(roles.measure))
            .unstack(fill_value=0)
        )
    else:
        pivot = working.groupby(["__segment", "Period"]).size().unstack(fill_value=0)
    pivot.index.name = roles.dimension
    return pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
