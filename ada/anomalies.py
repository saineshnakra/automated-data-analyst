"""Flag periods whose residual from the shared Theil-Sen trendline is unusually large.

The threshold is a multiple of the robust scale, calibrated per history length
so only ``FALSE_ALARM_RATE`` of stable series raise any flag. The calibration
assumes roughly normal noise; heavy-tailed measures (spiky ad spend, a few large
deals) are flagged more often. A series with a clear yearly pattern is judged
against the usual level for each time of year, so winter is not an anomaly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .aggregation import GRAIN_ORDER, TrendSeries, build_trend
from .schema import ColumnRoles
from .timeseries import fit_trendline, observed_periods, period_grain, robust_scale, yearly_pattern

MIN_PERIODS = 8
FALSE_ALARM_RATE = 0.05
# Years of history needed before the time of year sets the expected level: with
# three, each period is compared with the median of at least two others.
MIN_SEASONAL_CYCLES = 3

# (history length, residual multiplier) from tools/calibrate_anomalies.py. A fixed
# 3.0 would flag about a quarter of stable series, since every period is tested.
CRITICAL_VALUES: tuple[tuple[int, float], ...] = (
    (8, 7.01),
    (10, 5.82),
    (12, 5.21),
    (16, 4.63),
    (20, 4.33),
    (26, 4.07),
    (34, 3.94),
    (45, 3.88),
    (60, 3.79),
    (80, 3.79),
    (110, 3.77),
    (150, 3.79),
    (220, 3.88),
    (320, 3.88),
)


def critical_value(periods: int) -> float:
    """Residual multiplier that holds false alarms near ``FALSE_ALARM_RATE``.

    Interpolated across history length on a log scale, and held flat beyond
    the ends of the calibrated range.
    """
    lengths = np.array([length for length, _ in CRITICAL_VALUES], dtype=float)
    multipliers = np.array([value for _, value in CRITICAL_VALUES], dtype=float)
    return float(np.interp(np.log(max(periods, 1)), np.log(lengths), multipliers))


@dataclass(frozen=True)
class Anomaly:
    period: pd.Timestamp
    value: float
    expected: float
    expected_low: float
    expected_high: float
    direction: str  # "above" or "below"
    severity: float  # |residual| in robust-scale units
    #: True when the expected value allows for the time of year.
    seasonal: bool = False


def detect_anomalies(
    trend: pd.DataFrame,
    *,
    min_periods: int = MIN_PERIODS,
    threshold: float | None = None,
    limit: int = 5,
    floor_at_zero: bool = True,
    grain: str | None = None,
) -> tuple[Anomaly, ...]:
    """Flag periods whose value falls outside the trendline's expected range.

    ``threshold`` defaults to the calibrated multiplier for this history length;
    an explicit value bypasses the ``FALSE_ALARM_RATE`` calibration. With a
    yearly pattern over ``MIN_SEASONAL_CYCLES`` years or more, the expected
    value adds the usual offset for the period's month or quarter. With
    ``floor_at_zero`` and a history that never goes below zero, the expected
    range stops at zero, as for a forecast (``MetricSpec.floors_at_zero``).
    ``grain`` is the grain the trend was built at; without it, it is read
    from the spacing of the periods. Returns at most ``limit`` anomalies,
    most severe first.
    """
    if trend.empty or not {"Period", "Value"}.issubset(trend.columns):
        return ()
    # Read the grain before empty periods are dropped, which can make months look like quarters.
    grain = grain or period_grain(pd.DatetimeIndex(pd.to_datetime(trend["Period"])))
    # Drop unobserved periods: one NaN makes the fit and scale NaN, which flags nothing.
    trend = observed_periods(trend)
    if len(trend) < min_periods:
        return ()

    periods = pd.DatetimeIndex(pd.to_datetime(trend["Period"]))
    values = trend["Value"].to_numpy(dtype=float)

    line = fit_trendline(periods, values, grain)
    expected = line.fitted()
    residuals = values - expected
    seasonal = yearly_pattern(periods, residuals, min_cycles=MIN_SEASONAL_CYCLES, grain=grain)
    if seasonal is not None:
        expected = expected + seasonal
        residuals = values - expected

    scale = robust_scale(residuals)
    if scale == 0:
        return ()

    band = (critical_value(len(values)) if threshold is None else threshold) * scale
    low = expected - band
    if floor_at_zero and float(values.min()) >= 0:
        low = np.maximum(low, 0.0)
    anomalies = [
        Anomaly(
            period=pd.Timestamp(periods[position]),
            value=float(values[position]),
            expected=float(expected[position]),
            expected_low=float(low[position]),
            expected_high=float(expected[position] + band),
            direction="above" if residuals[position] > 0 else "below",
            severity=round(abs(float(residuals[position])) / scale, 2),
            seasonal=seasonal is not None,
        )
        for position in range(len(values))
        if abs(float(residuals[position])) > band
    ]
    anomalies.sort(key=lambda anomaly: anomaly.severity, reverse=True)
    return tuple(anomalies[:limit])


# A finer grain with more than this share of its periods empty is finer than the data was recorded at.
MAX_EMPTY_SHARE = 0.2


def finer_trend(dataframe: pd.DataFrame, roles: ColumnRoles, series: TrendSeries) -> TrendSeries | None:
    """The measure's trend at a finer grain with enough periods, when ``series`` has too few to check.

    A table of 147 daily readings rolls up to 5 months, too few to calibrate a
    band, but gives 21 weeks. The coarsest finer grain with ``MIN_PERIODS``
    recorded periods is used. A grain finer than the data was recorded at only
    adds empty periods, so one with over MAX_EMPTY_SHARE of them empty is
    skipped. Returns None when ``series`` has enough periods or no finer grain does.
    """
    if len(observed_periods(series.frame)) >= MIN_PERIODS or series.frequency not in GRAIN_ORDER:
        return None
    for grain in reversed(GRAIN_ORDER[: GRAIN_ORDER.index(series.frequency)]):
        candidate = build_trend(dataframe, roles, grain)
        periods, empty = len(candidate.frame), candidate.filled_periods
        if periods - empty >= MIN_PERIODS and empty <= MAX_EMPTY_SHARE * periods:
            return candidate
    return None
