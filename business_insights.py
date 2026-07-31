"""Explainable business intelligence built from deterministic calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from anomalies import detect_anomalies
from formatting import format_number, format_period, normalized_name
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


GRAIN_ORDER = ("W", "M", "Q")


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
    return "Q"


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
OFFSETTING_NET_RATIO = 0.25
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

    @property
    def notes(self) -> tuple[str, ...]:
        notes: list[str] = []
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
) -> tuple[pd.Timestamp | None, str]:
    """Detect a final period the data stops part-way through.

    Compares how much of each period the data actually reaches with how much
    it typically reaches. An extract cut on the 12th covers a third of its
    month while every earlier month covers essentially all of one; a quiet
    final week is nowhere near that different, and is left alone.
    """
    last_seen = dates.groupby(periods).max().sort_index()
    if len(last_seen) < MIN_PERIODS_FOR_PARTIAL_CHECK:
        return None, ""

    starts, ends = _period_bounds(pd.DatetimeIndex(last_seen.index), frequency)
    spans = (ends - starts).to_numpy().astype("timedelta64[s]").astype(float)
    reached = (last_seen.to_numpy() - starts.to_numpy()).astype("timedelta64[s]").astype(float)
    coverage = np.divide(reached, spans, out=np.zeros_like(reached), where=spans > 0)

    typical = float(np.median(coverage[:-1]))
    if coverage[-1] >= typical - PARTIAL_COVERAGE_MARGIN:
        return None, ""

    period_days = max(int(round(spans[-1] / 86_400)), 1)
    covered_days = max(int(round(reached[-1] / 86_400)) + 1, 1)
    return pd.Timestamp(last_seen.index[-1]), f"{covered_days} of {period_days} days"


def build_trend(
    dataframe: pd.DataFrame,
    roles: ColumnRoles,
    frequency: str | None = None,
) -> TrendSeries:
    """Aggregate the measure over a human-sized grain, on an even timeline."""
    if not roles.date:
        return TrendSeries(frame=EMPTY_TREND.copy(), frequency=frequency or "M")

    columns = [roles.date] + ([roles.measure] if roles.measure else [])
    working = dataframe[columns].dropna(subset=[roles.date]).copy()
    if working.empty:
        return TrendSeries(frame=EMPTY_TREND.copy(), frequency=frequency or "M")

    frequency = frequency or _period_frequency(working[roles.date])
    working["Period"] = working[roles.date].dt.to_period(frequency).dt.to_timestamp()

    partial_period, partial_coverage = _trailing_partial_period(
        working[roles.date], working["Period"], frequency
    )
    if partial_period is not None:
        remaining = working[working["Period"] < partial_period]
        if remaining["Period"].nunique() >= 2:
            working = remaining
        else:
            partial_period, partial_coverage = None, ""

    if roles.measure:
        result = working.groupby("Period", as_index=False)[roles.measure].sum()
        result = result.rename(columns={roles.measure: "Value"})
    else:
        result = working.groupby("Period", as_index=False).size().rename(columns={"size": "Value"})
    result = result.sort_values("Period").reset_index(drop=True)

    result, filled = _fill_empty_periods(result, frequency)
    return TrendSeries(
        frame=result,
        frequency=frequency,
        filled_periods=filled,
        partial_period=partial_period,
        partial_coverage=partial_coverage,
    )


def _fill_empty_periods(trend: pd.DataFrame, frequency: str) -> tuple[pd.DataFrame, int]:
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
        .reindex(complete, fill_value=0.0)
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
    if distance < 1.0:
        verdict = "This is within normal period-to-period variation"
    elif distance < 2.0:
        verdict = "This is a larger swing than usual"
    else:
        verdict = "This is an unusually large swing"
    return f" {verdict} — this series typically moves about {typical:.1f}% per period."


def _growth_evidence(dataframe: pd.DataFrame, roles: ColumnRoles) -> Evidence | None:
    trend = trend_frame(dataframe, roles)
    if len(trend) < 2:
        return None

    values = trend["Value"].to_numpy(dtype=float)
    previous = float(values[-2])
    current = float(values[-1])
    if previous == 0:
        return None
    change = (current - previous) / abs(previous) * 100
    period = trend.iloc[-1]["Period"].strftime("%b %Y")
    measure = roles.measure or "Records"
    direction = "increased" if change >= 0 else "decreased"
    context = _movement_in_context(values, change)
    return Evidence(
        kind="trend",
        title=f"Latest {measure.lower()} movement",
        value=f"{change:+.1f}%",
        statement=(
            f"{measure} {direction} {abs(change):.1f}% in the latest complete period "
            f"({period}), from {format_number(previous, roles.measure)} to "
            f"{format_number(current, roles.measure)}.{context}"
        ),
        calculation=(
            "(Latest period − previous period) ÷ |previous period|, set against the spread "
            "of past period-over-period changes"
        ),
        tone="positive" if change >= 0 else "negative",
    )


def _segment_period_change(
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
    working = dataframe[[roles.date, roles.measure, roles.dimension]].dropna().copy()
    working["Period"] = working[roles.date].dt.to_period(frequency).dt.to_timestamp()
    comparison = working[working["Period"].isin([previous_period, current_period])]
    grouped = comparison.groupby([roles.dimension, "Period"])[roles.measure].sum().unstack(fill_value=0)
    if previous_period not in grouped or current_period not in grouped:
        return None

    grouped["Change"] = grouped[current_period] - grouped[previous_period]
    return grouped, previous_period, current_period


def driver_frame(dataframe: pd.DataFrame, roles: ColumnRoles, limit: int = 9) -> pd.DataFrame:
    """Waterfall-ready per-segment change between the latest two periods."""
    result = _segment_period_change(dataframe, roles)
    if result is None:
        return pd.DataFrame(columns=["Segment", "Change"])
    grouped, _, _ = result
    changes = grouped["Change"].sort_values(key=lambda values: values.abs(), ascending=False)
    top = changes.head(limit)
    frame = pd.DataFrame({"Segment": top.index.astype(str), "Change": top.to_numpy(dtype=float)})
    remainder = float(changes.iloc[limit:].sum())
    if len(changes) > limit and remainder:
        other = pd.DataFrame({"Segment": ["Other segments"], "Change": [remainder]})
        frame = pd.concat([frame, other], ignore_index=True)
    return frame


def heatmap_frame(dataframe: pd.DataFrame, roles: ColumnRoles, limit: int = 8) -> pd.DataFrame:
    """Segment × period matrix of the measure (or row counts) for the top segments."""
    if not roles.date or not roles.dimension:
        return pd.DataFrame()

    top_segments = segment_frame(dataframe, roles, limit=limit)["Segment"]
    columns = [roles.date, roles.dimension] + ([roles.measure] if roles.measure else [])
    working = dataframe[columns].dropna(subset=[roles.date, roles.dimension]).copy()
    working = working[working[roles.dimension].isin(top_segments)]
    if working.empty:
        return pd.DataFrame()

    frequency = _period_frequency(working[roles.date])
    working["Period"] = working[roles.date].dt.to_period(frequency).dt.to_timestamp()
    if roles.measure:
        pivot = working.groupby([roles.dimension, "Period"])[roles.measure].sum().unstack(fill_value=0)
    else:
        pivot = working.groupby([roles.dimension, "Period"]).size().unstack(fill_value=0)
    return pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]


def _change_driver_evidence(dataframe: pd.DataFrame, roles: ColumnRoles) -> Evidence | None:
    """Identify the segment contributing most to the latest net movement."""
    result = _segment_period_change(dataframe, roles)
    if result is None:
        return None
    grouped, previous_period, current_period = result
    changes = grouped["Change"]
    net_change = float(changes.sum())
    gross_change = float(changes.abs().sum())
    if gross_change == 0:
        return None

    driver_name = str(changes.abs().idxmax())
    previous_value = float(grouped.loc[driver_name, previous_period])
    current_value = float(grouped.loc[driver_name, current_period])
    driver_change = float(grouped.loc[driver_name, "Change"])
    direction = "increased" if driver_change > 0 else "decreased"
    sign = "+" if driver_change > 0 else "−"

    movement = (
        f"{driver_name} moved the most of any {roles.dimension.lower()}: "
        f"{roles.measure} {direction} by {format_number(abs(driver_change), roles.measure)}, "
        f"from {format_number(previous_value, roles.measure)} to "
        f"{format_number(current_value, roles.measure)}."
    )

    # Segments that cancel out leave a tiny net change, and a share of that
    # net reads as an absurd multiple. Measure against total movement instead
    # and say plainly that the offsetting is what is really going on.
    if abs(net_change) < OFFSETTING_NET_RATIO * gross_change:
        share = abs(driver_change) / gross_change * 100
        statement = (
            f"{movement} Segments largely offset each other this period — "
            f"{format_number(gross_change, roles.measure)} of movement nets to just "
            f"{format_number(abs(net_change), roles.measure)} — so this is "
            f"{share:.1f}% of all movement rather than of the net."
        )
        calculation = (
            f"Latest {driver_name} {roles.measure} − previous; ranked by absolute change; "
            "share taken against total absolute movement because the net is near zero"
        )
    else:
        share = abs(driver_change / net_change) * 100
        statement = f"{movement} That is equivalent to {share:.1f}% of the net movement."
        calculation = (
            f"Latest {driver_name} {roles.measure} − previous {driver_name} {roles.measure}; "
            "ranked across segments"
        )

    return Evidence(
        kind="driver",
        title="Largest change driver",
        value=f"{sign}{format_number(abs(driver_change), roles.measure)}",
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
    label = format_period(worst.period, grain)
    plural = "periods sit" if len(anomalies) > 1 else "period sits"
    return Evidence(
        kind="anomaly",
        title="Anomalous periods",
        value=f"{len(anomalies)}",
        statement=(
            f"{label} is the sharpest anomaly: {measure.lower()} reached "
            f"{format_number(worst.value, roles.measure)}, {worst.direction} the expected "
            f"{format_number(worst.expected_low, roles.measure)}–"
            f"{format_number(worst.expected_high, roles.measure)} range. "
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
    effective = _effective_segments(segments["Value"].to_numpy(dtype=float), total)
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
        _concentration_evidence(
            dimension=dimension,
            measure=measure,
            segment_count=len(segments),
            top_three_share=top_three_share,
            effective=effective,
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
        f"The top three {dimension.lower()} values account for {top_three_share:.1f}% "
        f"of measured {measure.lower()}."
    )
    if effective is None:
        return Evidence(
            kind="concentration",
            title="Top-three concentration",
            value=f"{top_three_share:.1f}%",
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
