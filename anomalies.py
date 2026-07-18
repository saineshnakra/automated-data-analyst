"""Robust, explainable anomaly detection over period aggregates.

Periods are compared against a median-slope trendline: the slope is the
median of consecutive period-over-period differences and the intercept is
the median offset, so a single wild period cannot bend the baseline. Any
residual beyond ``threshold`` scaled median absolute deviations is flagged,
with the expected range reported alongside the observed value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_PERIODS = 8
THRESHOLD = 3.0
MAD_SCALE = 1.4826  # MAD → standard-deviation equivalent for normal data


@dataclass(frozen=True)
class Anomaly:
    period: pd.Timestamp
    value: float
    expected_low: float
    expected_high: float
    direction: str  # "above" or "below"
    severity: float  # residual in scaled-MAD units


def format_period(period: pd.Timestamp, grain: str) -> str:
    if grain == "Q":
        return f"Q{period.quarter} {period.year}"
    if grain in ("W", "D"):
        return period.strftime("%d %b %Y")
    if grain == "Y":
        return period.strftime("%Y")
    return period.strftime("%b %Y")


def detect_anomalies(
    trend: pd.DataFrame,
    *,
    min_periods: int = MIN_PERIODS,
    threshold: float = THRESHOLD,
    limit: int = 5,
) -> tuple[Anomaly, ...]:
    """Flag periods whose value escapes the trendline's expected range."""
    if trend.empty or not {"Period", "Value"}.issubset(trend.columns) or len(trend) < min_periods:
        return ()

    values = trend["Value"].to_numpy(dtype=float)
    index = np.arange(len(values))
    slope = float(np.median(np.diff(values)))
    intercept = float(np.median(values - slope * index))
    expected = intercept + slope * index
    residuals = values - expected

    mad = float(np.median(np.abs(residuals))) * MAD_SCALE
    if mad == 0:
        mad = float(np.mean(np.abs(residuals)))
    if mad == 0:
        return ()

    band = threshold * mad
    anomalies = [
        Anomaly(
            period=pd.Timestamp(trend.iloc[position]["Period"]),
            value=float(values[position]),
            expected_low=float(expected[position] - band),
            expected_high=float(expected[position] + band),
            direction="above" if residuals[position] > 0 else "below",
            severity=round(abs(float(residuals[position])) / mad, 2),
        )
        for position in range(len(values))
        if abs(float(residuals[position])) > band
    ]
    anomalies.sort(key=lambda anomaly: anomaly.severity, reverse=True)
    return tuple(anomalies[:limit])
