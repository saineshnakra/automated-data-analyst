"""Guarded baseline forecasting with a visible backtest.

The model is deliberately simple and fully explainable: a median-slope
trendline (robust to single wild periods) plus an optional month-of-year
seasonal adjustment learned from residual medians. A forecast is only
produced when history is long enough, the horizon never exceeds half the
observed history, and the honest backtested error ships with the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_PERIODS = 8
MIN_SEASONAL_PERIODS = 18
MAD_SCALE = 1.4826


@dataclass(frozen=True)
class Forecast:
    periods: tuple[pd.Timestamp, ...]
    values: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    backtest_mape: float | None  # mean absolute % error on the holdout
    holdout_periods: int
    method: str


def _fit(values: np.ndarray) -> tuple[float, float]:
    index = np.arange(len(values))
    slope = float(np.median(np.diff(values))) if len(values) > 1 else 0.0
    intercept = float(np.median(values - slope * index))
    return slope, intercept


def _monthly_seasonality(
    periods: pd.DatetimeIndex, residuals: np.ndarray
) -> dict[int, float]:
    seasonal: dict[int, float] = {}
    months = periods.month
    for month in range(1, 13):
        month_residuals = residuals[months == month]
        if len(month_residuals) >= 2:
            seasonal[month] = float(np.median(month_residuals))
    return seasonal


def _is_monthly(periods: pd.DatetimeIndex) -> bool:
    if len(periods) < 3:
        return False
    diffs = np.diff(periods.to_numpy()).astype("timedelta64[D]").astype(int)
    return bool(np.all((diffs >= 28) & (diffs <= 31)))


def _predict(
    positions: np.ndarray,
    months: np.ndarray,
    slope: float,
    intercept: float,
    seasonal: dict[int, float],
) -> np.ndarray:
    baseline = intercept + slope * positions
    adjustment = np.array([seasonal.get(int(month), 0.0) for month in months])
    return baseline + adjustment


def build_forecast(
    trend: pd.DataFrame,
    *,
    horizon: int = 6,
    min_periods: int = MIN_PERIODS,
) -> Forecast | None:
    """Forecast the next periods, or return None when history is too thin."""
    if trend.empty or not {"Period", "Value"}.issubset(trend.columns) or len(trend) < min_periods:
        return None

    periods = pd.DatetimeIndex(pd.to_datetime(trend["Period"]))
    values = trend["Value"].to_numpy(dtype=float)
    count = len(values)
    horizon = max(1, min(horizon, count // 2))

    monthly = _is_monthly(periods)
    slope, intercept = _fit(values)
    residuals = values - (intercept + slope * np.arange(count))
    seasonal = (
        _monthly_seasonality(periods, residuals)
        if monthly and count >= MIN_SEASONAL_PERIODS
        else {}
    )
    if seasonal:
        residuals = values - _predict(
            np.arange(count), periods.month.to_numpy(), slope, intercept, seasonal
        )

    spread = float(np.median(np.abs(residuals))) * MAD_SCALE
    if spread == 0:
        spread = float(np.mean(np.abs(residuals)))
    band = 2 * spread

    inferred = pd.infer_freq(periods)
    if inferred:
        future_periods = pd.date_range(periods[-1], periods=horizon + 1, freq=inferred)[1:]
    else:
        step = pd.Timedelta(np.median(np.diff(periods.to_numpy())))
        future_periods = pd.DatetimeIndex([periods[-1] + step * (offset + 1) for offset in range(horizon)])

    future_positions = np.arange(count, count + horizon)
    predictions = _predict(
        future_positions, future_periods.month.to_numpy(), slope, intercept, seasonal
    )
    if float(values.min()) >= 0:
        predictions = np.maximum(predictions, 0.0)
    lower = np.maximum(predictions - band, 0.0) if float(values.min()) >= 0 else predictions - band
    upper = predictions + band

    backtest_mape, holdout = _backtest(periods, values, monthly)

    method = "Median-slope trendline"
    if seasonal:
        method += " + month-of-year seasonality"
    method += " · band = ±2 robust deviations"

    return Forecast(
        periods=tuple(pd.Timestamp(period) for period in future_periods),
        values=tuple(float(value) for value in predictions),
        lower=tuple(float(value) for value in lower),
        upper=tuple(float(value) for value in upper),
        backtest_mape=backtest_mape,
        holdout_periods=holdout,
        method=method,
    )


def _backtest(
    periods: pd.DatetimeIndex, values: np.ndarray, monthly: bool
) -> tuple[float | None, int]:
    """Refit on a training split and score the held-out tail honestly."""
    count = len(values)
    holdout = min(max(3, count // 5), count - MIN_PERIODS + 3)
    if count - holdout < 5:
        return None, 0

    train_values = values[:-holdout]
    slope, intercept = _fit(train_values)
    train_residuals = train_values - (intercept + slope * np.arange(len(train_values)))
    seasonal = (
        _monthly_seasonality(periods[:-holdout], train_residuals)
        if monthly and len(train_values) >= MIN_SEASONAL_PERIODS
        else {}
    )
    positions = np.arange(len(train_values), count)
    predicted = _predict(positions, periods[-holdout:].month.to_numpy(), slope, intercept, seasonal)
    actual = values[-holdout:]
    nonzero = np.abs(actual) > 1e-9
    if not nonzero.any():
        return None, holdout
    mape = float(np.mean(np.abs(predicted[nonzero] - actual[nonzero]) / np.abs(actual[nonzero])) * 100)
    return round(mape, 1), holdout
