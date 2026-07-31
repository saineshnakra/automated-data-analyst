"""Robust, explainable anomaly detection over period aggregates.

Periods are compared against the shared Theil-Sen trendline from
``timeseries``: the slope is the median of every pairwise slope and periods
are placed on the calendar, so neither a single wild period nor a missing
one can bend the baseline. A period is flagged when its residual exceeds a
calibrated multiple of the robust scale, and the expected value and range
are reported alongside the observed one.

The multiplier is measured, not assumed. A fixed three deviations sounds
strict but flags at least one period in roughly a quarter of perfectly
stable series, because the question being asked is "is *any* of these n
periods unusual" and because the robust scale is itself unstable on short
histories. ``CRITICAL_VALUES`` instead holds the multiplier at which only
``FALSE_ALARM_RATE`` of stable series would raise a flag at all, simulated
per history length by ``tools/calibrate_anomalies.py``.

That guarantee assumes roughly normal period-to-period noise. A measure with
genuinely heavy tails -- spiky marketing spend, a handful of enterprise
deals dominating a month -- will exceed the band more often than one series
in twenty, because for such a measure those periods are ordinary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from timeseries import fit_trendline, robust_scale

MIN_PERIODS = 8
FALSE_ALARM_RATE = 0.05

# (history length, residual multiplier) from tools/calibrate_anomalies.py.
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
    severity: float  # residual in scaled-MAD units


def detect_anomalies(
    trend: pd.DataFrame,
    *,
    min_periods: int = MIN_PERIODS,
    threshold: float | None = None,
    limit: int = 5,
) -> tuple[Anomaly, ...]:
    """Flag periods whose value escapes the trendline's expected range.

    ``threshold`` defaults to the calibrated multiplier for this history
    length. Passing an explicit value trades the false-alarm guarantee for
    sensitivity, which is occasionally the right call but should be a
    deliberate one.
    """
    if trend.empty or not {"Period", "Value"}.issubset(trend.columns) or len(trend) < min_periods:
        return ()

    periods = pd.DatetimeIndex(pd.to_datetime(trend["Period"]))
    values = trend["Value"].to_numpy(dtype=float)

    line = fit_trendline(periods, values)
    expected = line.fitted()
    residuals = values - expected

    scale = robust_scale(residuals)
    if scale == 0:
        return ()

    band = (critical_value(len(values)) if threshold is None else threshold) * scale
    anomalies = [
        Anomaly(
            period=pd.Timestamp(periods[position]),
            value=float(values[position]),
            expected=float(expected[position]),
            expected_low=float(expected[position] - band),
            expected_high=float(expected[position] + band),
            direction="above" if residuals[position] > 0 else "below",
            severity=round(abs(float(residuals[position])) / scale, 2),
        )
        for position in range(len(values))
        if abs(float(residuals[position])) > band
    ]
    anomalies.sort(key=lambda anomaly: anomaly.severity, reverse=True)
    return tuple(anomalies[:limit])
