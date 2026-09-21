"""Guarded baseline forecasting with a visible backtest.

The model is deliberately simple and fully explainable: the shared Theil-Sen
trendline plus an optional month-of-year seasonal adjustment learned from
residual medians. A forecast is only produced when history is long enough,
the horizon never exceeds half the observed history, and the honest
backtested error ships with the numbers.

Two properties matter more than the model itself.

The band widens with the horizon. Six periods out is not as knowable as one,
and a constant band claims otherwise. The width uses the least-squares
prediction-interval shape with a robust scale, which is an approximation
for a median-based fit -- close enough to keep the shape honest, which is
why the backtest, not the band, is the number to trust.

The backtest reports a scaled error next to the percentage one. A MAPE of
12% means nothing on its own; a MASE below 1 means the forecast beat simply
assuming next period looks like this one, and above 1 means it did not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from timeseries import fit_trendline, observed_periods, period_grain, robust_scale

MIN_PERIODS = 8
MIN_SEASONAL_PERIODS = 18
BAND_DEVIATIONS = 2.0


@dataclass(frozen=True)
class Backtest:
    """Honest error, measured on periods the fit never saw."""

    mape: float | None  # mean absolute % error on the holdout
    mase: float | None  # holdout error ÷ error of a no-change forecast
    holdout_periods: int
    periods_without_mape: int  # holdout periods too near zero for a percentage

    @property
    def beats_no_change(self) -> bool | None:
        return None if self.mase is None else self.mase < 1.0


@dataclass(frozen=True)
class Forecast:
    periods: tuple[pd.Timestamp, ...]
    values: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    backtest: Backtest
    method: str


def describe_backtest(backtest: Backtest) -> str:
    """Plain-language error note, kept free of any presentation framework."""
    if not backtest.holdout_periods:
        return "history is too thin for a backtest"

    parts = [f"held out the last {backtest.holdout_periods} periods"]
    if backtest.mape is not None:
        note = f"±{backtest.mape:.1f}% average error"
        if backtest.periods_without_mape:
            measurable = backtest.holdout_periods - backtest.periods_without_mape
            note += f" across the {measurable} of them above zero"
        parts.append(note)
    elif backtest.periods_without_mape:
        parts.append("every held-out period was zero, so a percentage error says nothing")

    if backtest.mase is not None:
        verdict = "better" if backtest.beats_no_change else "no better"
        parts.append(f"MASE {backtest.mase:.2f}, {verdict} than assuming no change")

    return ", ".join(parts)


def _monthly_seasonality(periods: pd.DatetimeIndex, residuals: np.ndarray) -> dict[int, float]:
    """Median residual per calendar month, centred so it cannot shift the level."""
    seasonal: dict[int, float] = {}
    months = periods.month
    for month in range(1, 13):
        month_residuals = residuals[months == month]
        if len(month_residuals) >= 2:
            seasonal[month] = float(np.median(month_residuals))

    if not seasonal:
        return {}

    centre = float(np.median(list(seasonal.values())))
    return {month: value - centre for month, value in seasonal.items()}


def _seasonal_adjustment(months: np.ndarray, seasonal: dict[int, float]) -> np.ndarray:
    return np.array([seasonal.get(int(month), 0.0) for month in months])


def _future_periods(periods: pd.DatetimeIndex, horizon: int) -> pd.DatetimeIndex:
    """The next ``horizon`` periods on the same calendar grain."""
    last = pd.Period(periods[-1], freq=period_grain(periods))
    return pd.DatetimeIndex([(last + offset).to_timestamp() for offset in range(1, horizon + 1)])


def _prediction_spread(positions: np.ndarray, future: np.ndarray, scale: float) -> np.ndarray:
    """Uncertainty at each future position, widening with distance from the data.

    The least-squares prediction-interval shape: irreducible noise, plus the
    uncertainty in the level, plus the uncertainty in the slope -- the last
    of which is what grows as the forecast reaches further out.
    """
    count = len(positions)
    centre = float(np.mean(positions))
    spread_of_positions = float(np.sum((positions - centre) ** 2))
    if count < 3 or spread_of_positions == 0:
        return np.full(len(future), scale)

    leverage = 1.0 + 1.0 / count + (future - centre) ** 2 / spread_of_positions
    return scale * np.sqrt(leverage)


def build_forecast(
    trend: pd.DataFrame,
    *,
    horizon: int = 6,
    min_periods: int = MIN_PERIODS,
) -> Forecast | None:
    """Forecast the next periods, or return None when history is too thin."""
    if trend.empty or not {"Period", "Value"}.issubset(trend.columns):
        return None
    # Unobserved periods are NaN by design (see aggregation's calendar rules).
    # Dropped BEFORE the length check, so "enough history" counts periods that
    # were actually measured rather than blanks the fit cannot use anyway.
    trend = observed_periods(trend)
    if len(trend) < min_periods:
        return None

    periods = pd.DatetimeIndex(pd.to_datetime(trend["Period"]))
    values = trend["Value"].to_numpy(dtype=float)
    count = len(values)
    horizon = max(1, min(horizon, count // 2))

    monthly = period_grain(periods) == "M"
    line = fit_trendline(periods, values)
    positions = np.asarray(line.positions, dtype=float)
    residuals = line.residuals(values)

    seasonal = (
        _monthly_seasonality(periods, residuals)
        if monthly and count >= MIN_SEASONAL_PERIODS
        else {}
    )
    if seasonal:
        residuals = residuals - _seasonal_adjustment(periods.month.to_numpy(), seasonal)

    scale = robust_scale(residuals)
    future_periods = _future_periods(periods, horizon)
    future_positions = line.future_positions(horizon)

    predictions = line.at(future_positions) + _seasonal_adjustment(
        future_periods.month.to_numpy(), seasonal
    )
    band = BAND_DEVIATIONS * _prediction_spread(positions, future_positions, scale)

    non_negative = float(values.min()) >= 0
    if non_negative:
        predictions = np.maximum(predictions, 0.0)
    lower = predictions - band
    upper = predictions + band
    if non_negative:
        lower = np.maximum(lower, 0.0)

    method = "Theil–Sen trendline"
    if seasonal:
        method += " + month-of-year seasonality"
    if scale > 0:
        method += f" · band = ±{BAND_DEVIATIONS:g} robust deviations, widening with horizon"
    else:
        # A history with no variation gives a zero-width band. Drawing that as
        # a line and captioning it as a widening interval claims a certainty
        # nothing here supports.
        method += " · history shows no variation, so no uncertainty band could be estimated"

    return Forecast(
        periods=tuple(pd.Timestamp(period) for period in future_periods),
        values=tuple(float(value) for value in predictions),
        lower=tuple(float(value) for value in lower),
        upper=tuple(float(value) for value in upper),
        backtest=_backtest(periods, values, monthly=monthly),
        method=method,
    )


def _backtest(periods: pd.DatetimeIndex, values: np.ndarray, *, monthly: bool) -> Backtest:
    """Refit on a training split and score the held-out tail honestly."""
    count = len(values)
    holdout = min(max(3, count // 5), count - MIN_PERIODS + 3)
    # periods[:-0] is the whole array, not "everything but nothing", so a
    # holdout that resolves to zero or less trained on an empty split.
    if holdout < 1 or count - holdout < 5:
        return Backtest(mape=None, mase=None, holdout_periods=0, periods_without_mape=0)

    train_periods, train_values = periods[:-holdout], values[:-holdout]
    line = fit_trendline(train_periods, train_values)
    train_residuals = line.residuals(train_values)

    seasonal = (
        _monthly_seasonality(train_periods, train_residuals)
        if monthly and len(train_values) >= MIN_SEASONAL_PERIODS
        else {}
    )

    holdout_periods = periods[-holdout:]
    horizon_positions = line.future_positions(holdout)
    predicted = line.at(horizon_positions) + _seasonal_adjustment(
        holdout_periods.month.to_numpy(), seasonal
    )
    if float(train_values.min()) >= 0:
        # Score the forecast that is actually shown: production clips a
        # non-negative series at zero, so the backtest must too.
        predicted = np.maximum(predicted, 0.0)
    actual = values[-holdout:]
    errors = np.abs(predicted - actual)

    measurable = np.abs(actual) > 1e-9
    mape = (
        round(float(np.mean(errors[measurable] / np.abs(actual[measurable])) * 100), 1)
        if measurable.any()
        else None
    )

    # A no-change forecast is the bar any baseline has to clear.
    no_change_error = float(np.mean(np.abs(np.diff(train_values))))
    mase = round(float(np.mean(errors) / no_change_error), 2) if no_change_error > 0 else None

    return Backtest(
        mape=mape,
        mase=mase,
        holdout_periods=holdout,
        periods_without_mape=int((~measurable).sum()),
    )
