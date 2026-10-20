"""Robust trendline primitives shared by anomaly detection and forecasting.

Slopes are Theil-Sen (median of pairwise slopes, so one wild period can't tilt
the line). Periods are positioned on the calendar rather than by row order, so a
missing month leaves a gap in the fit instead of shortening the timeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MAD_SCALE = 1.4826  # converts MAD to a standard deviation for normal data
# 1/sqrt(2/pi): puts the mean absolute deviation on the same scale as MAD_SCALE,
# so falling back to it doesn't shift z-scores.
MEAN_ABS_DEV_SCALE = 1.2533
MAX_PAIRWISE_POINTS = 800  # keeps the O(n^2) slope search bounded


@dataclass(frozen=True)
class TrendLine:
    """A robust straight line fitted against calendar positions."""

    slope: float
    intercept: float
    positions: tuple[float, ...]

    def at(self, positions: np.ndarray | list[float]) -> np.ndarray:
        """Evaluate the line at any calendar positions."""
        return self.intercept + self.slope * np.asarray(positions, dtype=float)

    def fitted(self) -> np.ndarray:
        """Evaluate the line at the positions it was fitted on."""
        return self.at(np.asarray(self.positions, dtype=float))

    def residuals(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) - self.fitted()

    def future_positions(self, horizon: int) -> np.ndarray:
        """Calendar positions of the next ``horizon`` periods."""
        last = self.positions[-1] if self.positions else 0.0
        return np.arange(last + 1.0, last + 1.0 + horizon)


def period_grain(periods: pd.DatetimeIndex) -> str:
    """Classify the median spacing of ``periods`` as "D", "W", "M", "Q" or "Y".

    The median keeps one gap in a regular series from changing the answer.
    Returns "D" for fewer than two periods.
    """
    index = pd.DatetimeIndex(periods)
    if len(index) < 2:
        return "D"
    spacing = np.diff(index.to_numpy()).astype("timedelta64[s]").astype(float) / 86_400.0
    positive = spacing[spacing > 0]
    typical = float(np.median(positive)) if positive.size else 0.0
    if typical <= 1.5:
        return "D"
    if typical <= 10.0:
        return "W"
    if typical <= 45.0:
        return "M"
    if typical <= 130.0:
        return "Q"
    return "Y"


def period_positions(periods: pd.DatetimeIndex) -> np.ndarray:
    """Number periods by the calendar, so gaps stay visible to the fit.

    Consecutive periods are one apart and a skipped period leaves a hole.
    Counting calendar periods instead of dividing elapsed days avoids drift
    from unequal month and quarter lengths.
    """
    index = pd.DatetimeIndex(periods)
    if len(index) == 0:
        return np.zeros(0)
    if len(index) == 1:
        return np.zeros(1)

    ordinals = pd.PeriodIndex(index, freq=period_grain(index)).asi8.astype(float)
    return ordinals - ordinals[0]


def _slope_sample(size: int) -> np.ndarray:
    """Indices used for the pairwise slope search, thinned when very long."""
    if size <= MAX_PAIRWISE_POINTS:
        return np.arange(size)
    return np.unique(np.linspace(0, size - 1, MAX_PAIRWISE_POINTS).round().astype(int))


def theil_sen(positions: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """Fit ``value = intercept + slope x position`` by median pairwise slope."""
    positions = np.asarray(positions, dtype=float)
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0, 0.0
    if values.size == 1:
        return 0.0, float(values[0])

    sample = _slope_sample(values.size)
    first, second = np.triu_indices(sample.size, k=1)
    spans = positions[sample][second] - positions[sample][first]
    usable = spans != 0
    if not usable.any():
        return 0.0, float(np.median(values))

    rises = values[sample][second][usable] - values[sample][first][usable]
    slope = float(np.median(rises / spans[usable]))
    intercept = float(np.median(values - slope * positions))
    return slope, intercept


def fit_trendline(periods: pd.DatetimeIndex, values: np.ndarray) -> TrendLine:
    """Fit a robust trendline against the calendar positions of ``periods``."""
    positions = period_positions(periods)
    slope, intercept = theil_sen(positions, values)
    return TrendLine(slope=slope, intercept=intercept, positions=tuple(float(p) for p in positions))


def robust_scale(residuals: np.ndarray) -> float:
    """Estimate the residuals' standard deviation robustly, from the scaled MAD.

    Falls back to the scaled mean absolute deviation when more than half the
    residuals are identical, since MAD is then zero and every other period would
    look infinitely unusual. Returns 0.0 when residuals are empty or all equal.
    """
    values = np.asarray(residuals, dtype=float)
    if values.size == 0:
        return 0.0

    centre = float(np.median(values))
    deviations = np.abs(values - centre)
    scale = float(np.median(deviations)) * MAD_SCALE
    if scale > 0:
        return scale

    fallback = float(np.mean(deviations)) * MEAN_ABS_DEV_SCALE
    return fallback if fallback > 0 else 0.0


def observed_periods(trend: pd.DataFrame) -> pd.DataFrame:
    """Drop periods with no observation before fitting.

    The trend frame keeps empty periods as NaN for the chart, but one NaN makes
    the Theil-Sen slope NaN. Returns the frame unchanged when nothing is missing.
    """
    if trend.empty or "Value" not in trend.columns:
        return trend
    missing = trend["Value"].isna()
    if not missing.any():
        return trend
    return trend.loc[~missing]
