"""Robust trendline primitives shared by anomaly detection and forecasting.

Slopes are Theil-Sen (median of pairwise slopes, so one wild period can't tilt
the line). Periods are positioned on the calendar rather than by row order, so a
missing month leaves a gap in the fit instead of shortening the timeline.
"""

from __future__ import annotations

import math
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
    """Classify the median spacing of ``periods`` as "s", "min", "h", "D", "W", "M", "Q" or "Y".

    The median keeps one gap in a regular series from changing the answer.
    Returns "D" for fewer than two periods.
    """
    index = pd.DatetimeIndex(periods)
    if len(index) < 2:
        return "D"
    spacing = np.diff(index.to_numpy()).astype("timedelta64[ms]").astype(float) / 86_400_000.0
    positive = spacing[spacing > 0]
    typical = float(np.median(positive)) if positive.size else 0.0
    if 0 < typical * 86_400 <= 1.5:
        return "s"
    if 0 < typical * 1_440 <= 1.5:
        return "min"
    if 0 < typical * 24 <= 1.5:
        return "h"
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


# Grains with a yearly cycle, and how many periods make one year.
PERIODS_PER_YEAR = {"M": 12, "Q": 4}
# Chance of calling noise a yearly pattern: the F test of the season of the
# year against the residuals from the trendline must reach this p-value. Missing
# a real cycle flags every winter; mistaking noise for one only compares each
# period with the same months of other years, so the bar is not set higher.
SEASONAL_SIGNIFICANCE = 0.05
# Degrees of freedom left after fitting each season, so the check needs three
# years of quarters or two of months.
MIN_SEASONAL_DF = 8


def _incomplete_beta(a: float, b: float, x: float) -> float:
    """The regularized incomplete beta function, by its continued fraction."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _incomplete_beta(b, a, 1.0 - x)
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    ) / a
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    result = d
    for step in range(1, 300):
        for numerator in (
            step * (b - step) * x / ((a + 2 * step - 1) * (a + 2 * step)),
            -(a + step) * (a + b + step) * x / ((a + 2 * step) * (a + 2 * step + 1)),
        ):
            d = 1.0 + numerator * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + numerator / c
            c = c if abs(c) > tiny else tiny
            result *= c * d
        if abs(c * d - 1.0) < 1e-12:
            break
    return front * result


def f_test_p_value(statistic: float, numerator_df: int, denominator_df: int) -> float:
    """Chance of an F statistic at least this large when the groups do not differ."""
    if statistic <= 0:
        return 1.0
    x = denominator_df / (denominator_df + numerator_df * statistic)
    return _incomplete_beta(denominator_df / 2, numerator_df / 2, x)


def season_numbers(periods: pd.DatetimeIndex, grain: str) -> np.ndarray:
    """The month (1-12) or quarter (1-4) of each period."""
    index = pd.DatetimeIndex(periods)
    return (index.month if grain == "M" else index.quarter).to_numpy()


def yearly_pattern(
    periods: pd.DatetimeIndex, residuals: np.ndarray, *, min_cycles: int = 2
) -> np.ndarray | None:
    """Each period's usual offset from the trendline for its time of year, or None.

    A period's offset is the median residual of the same month (or quarter) in
    the other years, so an unusual period does not set its own expectation.
    The offsets are centred so they cannot shift the level. Returns None for a
    grain with no yearly cycle, fewer than ``min_cycles`` sightings of every
    season, or a season of the year that explains little of the variation.
    """
    index = pd.DatetimeIndex(periods)
    grain = period_grain(index)
    length = PERIODS_PER_YEAR.get(grain)
    residuals = np.asarray(residuals, dtype=float)
    if length is None or len(residuals) - length < MIN_SEASONAL_DF:
        return None
    seasons = season_numbers(index, grain)
    counts = np.bincount(seasons, minlength=length + 1)[1:]
    if counts.min() < min_cycles:
        return None

    means = np.array([residuals[seasons == season].mean() for season in range(1, length + 1)])
    total = float(np.sum((residuals - residuals.mean()) ** 2))
    within = float(np.sum((residuals - means[seasons - 1]) ** 2))
    size = len(residuals)
    if total == 0:
        return None
    if within > 0:
        statistic = ((total - within) / (length - 1)) / (within / (size - length))
        if f_test_p_value(statistic, length - 1, size - length) > SEASONAL_SIGNIFICANCE:
            return None

    medians = np.array([np.median(residuals[seasons == season]) for season in range(1, length + 1)])
    centre = float(np.median(medians))
    offsets = np.empty(size)
    for position, season in enumerate(seasons):
        others = (seasons == season) & (np.arange(size) != position)
        offsets[position] = float(np.median(residuals[others])) - centre
    return offsets


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
