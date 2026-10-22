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
import ruptures
from scipy import stats

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


def period_positions(periods: pd.DatetimeIndex, grain: str | None = None) -> np.ndarray:
    """Number periods by the calendar, so gaps stay visible to the fit.

    Consecutive periods are one apart and a skipped period leaves a hole.
    Counting calendar periods instead of dividing elapsed days avoids drift
    from unequal month and quarter lengths. Pass the ``grain`` the periods
    were built at: inferred from the periods that hold values, months with
    gaps between them can read as quarters, and the slope would then be per
    quarter while labelled per month.
    """
    index = pd.DatetimeIndex(periods)
    if len(index) == 0:
        return np.zeros(0)
    if len(index) == 1:
        return np.zeros(1)

    ordinals = pd.PeriodIndex(index, freq=grain or period_grain(index)).asi8.astype(float)
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


def fit_trendline(periods: pd.DatetimeIndex, values: np.ndarray, grain: str | None = None) -> TrendLine:
    """Fit a robust trendline against the calendar positions of ``periods`` at ``grain``."""
    positions = period_positions(periods, grain)
    slope, intercept = theil_sen(positions, values)
    return TrendLine(slope=slope, intercept=intercept, positions=tuple(float(p) for p in positions))


# Confidence of the slope's interval, and the most points the test and interval
# use: beyond it the series is thinned evenly, as for the trendline itself.
TREND_CONFIDENCE = 0.95
MAX_TEST_POINTS = 2_000


@dataclass(frozen=True)
class TrendTest:
    """A Theil-Sen slope with its confidence interval and a Mann-Kendall test.

    The test and the interval allow for lag-1 autocorrelation in the detrended
    series, which makes a wandering series look trending when ignored.
    """

    slope: float
    intercept: float
    low: float
    high: float
    #: Kendall's tau between position and value.
    tau: float
    #: Two-sided Mann-Kendall p-value, after the autocorrelation allowance.
    p_value: float
    points: int
    #: Lag-1 autocorrelation of the detrended values (0 when negative or unknown).
    autocorrelation: float = 0.0

    @property
    def clear(self) -> bool:
        """Whether the interval leaves out zero, so the series has a trend one way."""
        return self.low > 0 or self.high < 0


def autocorrelation_inflation(residuals: np.ndarray) -> tuple[float, float]:
    """How much positive lag-1 autocorrelation inflates the variance of Kendall's S.

    Yue and Wang's (2004) effective-sample-size factor
    ``1 + 2 sum (1 - k/n) r^k`` for an AR(1) series with lag-1 autocorrelation
    ``r`` of the detrended values. Returns ``(factor, r)``; negative
    autocorrelation is not credited, so the factor is never below 1.
    """
    values = np.asarray(residuals, dtype=float)
    size = values.size
    centred = values - values.mean()
    total = float(np.sum(centred**2))
    if size < 4 or total == 0:
        return 1.0, 0.0
    rho = float(np.sum(centred[:-1] * centred[1:]) / total)
    if rho <= 0:
        return 1.0, 0.0
    lags = np.arange(1, size)
    return float(1 + 2 * np.sum((1 - lags / size) * rho**lags)), rho


def trend_test(positions: np.ndarray, values: np.ndarray) -> TrendTest | None:
    """Test a series for a monotonic trend and size its slope, or None with fewer than 4 points.

    The slope and its interval come from ``scipy.stats.theilslopes`` and the
    Mann-Kendall test from ``scipy.stats.kendalltau`` (its asymptotic form is
    the Mann-Kendall statistic). Both are widened by
    ``autocorrelation_inflation``: the test's z is divided by the square root
    of the factor, and the interval is taken at the matching higher confidence.
    """
    positions = np.asarray(positions, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(positions) & np.isfinite(values)
    positions, values = positions[finite], values[finite]
    if values.size > MAX_TEST_POINTS:
        keep = np.unique(np.linspace(0, values.size - 1, MAX_TEST_POINTS).round().astype(int))
        positions, values = positions[keep], values[keep]
    if values.size < 4 or np.ptp(positions) == 0:
        return None
    if np.ptp(values) == 0:
        return TrendTest(0.0, float(values[0]), 0.0, 0.0, 0.0, 1.0, int(values.size))
    slope, intercept = theil_sen(positions, values)
    factor, rho = autocorrelation_inflation(values - (intercept + slope * positions))
    tau, p_value = stats.kendalltau(positions, values, method="asymptotic")
    tau, p_value = float(tau), float(p_value)
    if np.isfinite(p_value) and factor > 1:
        z = stats.norm.isf(p_value / 2) / np.sqrt(factor)
        p_value = float(2 * stats.norm.sf(z))
    critical = stats.norm.isf((1 - TREND_CONFIDENCE) / 2) * np.sqrt(factor)
    confidence = min(float(1 - 2 * stats.norm.sf(critical)), 0.999999)
    fitted = stats.theilslopes(values, positions, alpha=confidence)
    return TrendTest(
        slope=slope,
        intercept=intercept,
        low=float(fitted.low_slope),
        high=float(fitted.high_slope),
        tau=tau,
        p_value=p_value if np.isfinite(p_value) else 1.0,
        points=int(values.size),
        autocorrelation=rho,
    )


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
    periods: pd.DatetimeIndex, residuals: np.ndarray, *, min_cycles: int = 2, grain: str | None = None
) -> np.ndarray | None:
    """Each period's usual offset from the trendline for its time of year, or None.

    A period's offset is the median residual of the same month (or quarter) in
    the other years, so an unusual period does not set its own expectation.
    The offsets are centred so they cannot shift the level. Returns None for a
    grain with no yearly cycle, fewer than ``min_cycles`` sightings of every
    season, or a season of the year that explains little of the variation.
    """
    index = pd.DatetimeIndex(periods)
    grain = grain or period_grain(index)
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


# A level shift is looked for in a series of at least this many periods, and
# each of its two stretches holds at least MIN_STRETCH periods and STRETCH_SHARE
# of the series: a couple of odd periods at either end are anomalies, not a new level.
MIN_SHIFT_PERIODS = 8
MIN_STRETCH = 2
STRETCH_SHARE = 0.15
# A shift must be at least this many within-stretch standard deviations: a
# stable series wanders by about one.
MIN_SHIFT_EFFECT = 1.5
# (history length, t statistic) from tools/calibrate_level_shifts.py, above
# which only 1 stable series in 20 shows a shift that also passes the other rules.
SHIFT_CRITICAL_VALUES: tuple[tuple[int, float], ...] = (
    (8, 3.19),
    (12, 2.75),
    (20, 2.00),
    (40, 2.00),
    (60, 2.00),
    (120, 2.00),
    (240, 2.00),
)


@dataclass(frozen=True)
class LevelShift:
    """The split of a series into two stretches at different levels that fits it best."""

    #: Position of the first value at the new level.
    position: int
    before: float
    after: float
    #: The shift in within-stretch standard deviations.
    effect: float
    #: The two-sample t statistic, discounted for autocorrelation within the stretches.
    statistic: float
    #: Squared error of the two-level fit as a share of a straight line's.
    against_line: float
    #: Each stretch's median sits on its own side of the midpoint between the two levels,
    #: so neither level rests on one or two extreme values.
    steady: bool = True


def shift_critical_value(periods: int) -> float:
    """The calibrated t statistic for a series of this length, interpolated on a log scale."""
    lengths = np.array([length for length, _ in SHIFT_CRITICAL_VALUES], dtype=float)
    values = np.array([value for _, value in SHIFT_CRITICAL_VALUES], dtype=float)
    return float(np.interp(np.log(max(periods, 1)), np.log(lengths), values))


def best_split(values: np.ndarray) -> LevelShift | None:
    """The best two-level fit of ``values`` by binary segmentation, with no test applied.

    The split comes from ``ruptures.Binseg`` with the l2 cost, which minimises
    the squared error around each stretch's mean. Its t statistic is
    discounted by the lag-1 autocorrelation left within the stretches (with
    Kendall's small-sample correction), since a slowly wandering series makes
    any split look significant. Returns None for a series too short or flat.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    size = len(values)
    if size < MIN_SHIFT_PERIODS or np.ptp(values) == 0:
        return None
    stretch = max(MIN_STRETCH, round(STRETCH_SHARE * size))
    position = ruptures.Binseg(model="l2", min_size=stretch, jump=1).fit(values).predict(n_bkps=1)[0]
    first, second = values[:position], values[position:]
    residual = np.r_[first - first.mean(), second - second.mean()]
    error = float((residual**2).sum())
    shift = float(second.mean() - first.mean())
    spread = np.sqrt(error / (size - 2))
    if spread == 0:
        return LevelShift(position, float(first.mean()), float(second.mean()), np.inf, np.inf, 0.0)
    midpoint = (first.mean() + second.mean()) / 2
    side = np.sign(second.mean() - first.mean())
    steady = bool(
        np.sign(np.median(first) - midpoint) == -side and np.sign(np.median(second) - midpoint) == side
    )
    lagged = np.r_[
        residual[: position - 1] * residual[1:position], residual[position:-1] * residual[position + 1 :]
    ]
    rho = float(lagged.sum()) / error
    rho = min(0.9, max(0.0, rho + (1 + 3 * rho) / size))
    error_of_shift = spread * np.sqrt(1 / len(first) + 1 / len(second))
    statistic = abs(shift) / error_of_shift * np.sqrt((1 - rho) / (1 + rho))
    steps = np.arange(size, dtype=float)
    line = np.polyval(np.polyfit(steps, values, 1), steps)
    line_error = float(((values - line) ** 2).sum())
    return LevelShift(
        position=position,
        before=float(first.mean()),
        after=float(second.mean()),
        effect=float(abs(shift) / spread),
        statistic=float(statistic),
        against_line=error / line_error if line_error > 0 else 1.0,
        steady=steady,
    )


def passes_shift_rules(found: LevelShift, size: int) -> bool:
    """Whether a split is large and steady enough to report, short of the calibrated t statistic.

    The two-level fit must beat a straight line by more than its extra
    parameters would by chance (a BIC-like margin of n^(-2/n) on the squared
    error), so a steady rise is not called a shift, and each level must hold
    for most of its stretch, so one extreme period is not a new level.
    """
    return (
        found.steady and found.effect >= MIN_SHIFT_EFFECT and found.against_line < size ** (-2 / size)
    )


def level_shift(values: np.ndarray) -> LevelShift | None:
    """The largest lasting change in a series' level, or None when it has none.

    Binary segmentation finds the best split; it counts as a shift when its
    t statistic passes ``shift_critical_value`` for the series' length and
    ``passes_shift_rules`` holds. The threshold is calibrated so that only 1
    stable series in 20 (normal noise) shows a shift; series whose noise
    wanders slowly show one more often.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    found = best_split(values)
    if found is None:
        return None
    if found.statistic < shift_critical_value(len(values)) or not passes_shift_rules(found, len(values)):
        return None
    return found
