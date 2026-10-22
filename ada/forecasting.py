"""Baseline forecasting: Theil-Sen trendline plus the usual offset for the time of year.

Forecasts need ``MIN_PERIODS`` observed periods, the horizon is capped at half
the history, and each forecast carries a holdout backtest against a no-change
forecast and, when the time of year is modelled, against repeating the same
period a year earlier. The band widens with the horizon using the
least-squares prediction-interval shape with a robust scale; that is an
approximation for a median fit, so the backtest is the better guide to accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .timeseries import fit_trendline, observed_periods, period_grain, period_positions, robust_scale

MIN_PERIODS = 8
BAND_DEVIATIONS = 2.0
# Grains below a year whose periods follow a yearly cycle, with the periods in
# a year. A day is offset by its month: a separate offset for each of 365 days
# would rest on one or two values.
PERIODS_PER_YEAR = {"D": 365, "W": 52, "M": 12, "Q": 4}
# What each grain's period is called, for the benchmark that repeats it a year later.
YEAR_AGO_NAMES = {"D": "day", "W": "week", "M": "month", "Q": "quarter"}
# History before the time of year is modelled: a year and a half, so most
# seasons have been seen twice (18 months, 6 quarters).
MIN_SEASONAL_YEARS = 1.5


@dataclass(frozen=True)
class Backtest:
    """Forecast error measured on held-out periods the fit never saw.

    ``mase`` divides the holdout error by the in-sample error of a one-step
    no-change forecast, so a value below 1 beats that baseline.
    ``seasonal_mase`` does the same against repeating the same period a year
    earlier, when the forecast models the time of year.
    """

    mape: float | None  # mean absolute % error on the holdout
    mase: float | None  # holdout MAE ÷ in-sample MAE of a no-change forecast
    holdout_periods: int
    periods_without_mape: int  # holdout periods too near zero for a percentage
    #: Holdout MAE ÷ in-sample MAE of repeating the same period a year earlier.
    seasonal_mase: float | None = None
    #: The period that benchmark repeats ("quarter" for the same quarter a year earlier).
    season: str = ""

    @property
    def beats_no_change(self) -> bool | None:
        return None if self.mase is None else self.mase < 1.0

    @property
    def worse_than_no_change(self) -> bool:
        """True when assuming next period equals this one did better on the held-out periods."""
        return self.mase is not None and self.mase > 1.0

    @property
    def worse_than_last_year(self) -> bool:
        """True when repeating the same period a year earlier would have done better."""
        return self.seasonal_mase is not None and self.seasonal_mase > 1.0

    @property
    def shortfall(self) -> str:
        """The simpler forecast that did better on the held-out periods, or "" when neither did."""
        if self.worse_than_no_change:
            return f"simply assuming no change (MASE {self.mase:.2f})"
        if self.worse_than_last_year:
            return f"repeating the same {self.season} a year earlier (seasonal MASE {self.seasonal_mase:.2f})"
        return ""


@dataclass(frozen=True)
class Forecast:
    periods: tuple[pd.Timestamp, ...]
    values: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    backtest: Backtest
    method: str

    @property
    def reliable(self) -> bool:
        """False when a simpler forecast did better on the backtest, so the line misleads."""
        return not self.backtest.shortfall

    @property
    def summary(self) -> str:
        """The projection and its backtest in one plain sentence."""
        start, end = self.values[0], self.values[-1]
        projection = (
            f"Projects {len(self.values)} periods ahead, from {start:,.4g} to {end:,.4g} "
            f"({self.method})"
        )
        return f"{projection}; backtest: {describe_backtest(self.backtest)}."


def describe_backtest(backtest: Backtest) -> str:
    """Describe the backtest in one plain-language line, independent of the UI."""
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

    if backtest.worse_than_no_change:
        parts.append(
            f"MASE {backtest.mase:.2f}: worse than simply assuming no change, so do not rely "
            "on this forecast"
        )
    elif backtest.mase is not None:
        verdict = "better" if backtest.beats_no_change else "no better"
        parts.append(f"MASE {backtest.mase:.2f}, {verdict} than assuming no change")

    if backtest.seasonal_mase is not None and not backtest.worse_than_no_change:
        last_year = f"repeating the same {backtest.season} a year earlier"
        if backtest.worse_than_last_year:
            parts.append(
                f"seasonal MASE {backtest.seasonal_mase:.2f}: worse than {last_year}, so do not "
                "rely on this forecast"
            )
        else:
            verdict = "better" if backtest.seasonal_mase < 1.0 else "no better"
            parts.append(f"seasonal MASE {backtest.seasonal_mase:.2f}, {verdict} than {last_year}")

    return ", ".join(parts)


def season_of(periods: pd.DatetimeIndex, grain: str) -> np.ndarray:
    """Each period's place in the year: its quarter, week (a 53rd counts as the 52nd) or month."""
    index = pd.DatetimeIndex(periods)
    if grain == "Q":
        return index.quarter.to_numpy()
    if grain == "W":
        return np.minimum(index.isocalendar().week.to_numpy(dtype=int), 52)
    return index.month.to_numpy()


def _year_ago(periods: pd.DatetimeIndex, grain: str) -> pd.DatetimeIndex:
    """The same period a year before each one."""
    if grain == "W":
        return periods - pd.Timedelta(weeks=52)
    return periods - pd.DateOffset(years=1)


def _yearly_seasonality(periods: pd.DatetimeIndex, residuals: np.ndarray, grain: str) -> dict[int, float]:
    """Median residual per season of the year, centred so it cannot shift the level.

    A season gets an offset once it has been seen in two different years.
    Returns {} for a grain with no yearly cycle or a history shorter than
    MIN_SEASONAL_YEARS.
    """
    per_year = PERIODS_PER_YEAR.get(grain)
    if per_year is None or len(residuals) < MIN_SEASONAL_YEARS * per_year:
        return {}
    seasons = season_of(periods, grain)
    years = pd.DatetimeIndex(periods).year.to_numpy()
    seasonal: dict[int, float] = {}
    for season in np.unique(seasons):
        chosen = seasons == season
        if len(np.unique(years[chosen])) >= 2:
            seasonal[int(season)] = float(np.median(residuals[chosen]))

    if not seasonal:
        return {}

    centre = float(np.median(list(seasonal.values())))
    return {season: value - centre for season, value in seasonal.items()}


def _seasonal_adjustment(seasons: np.ndarray, seasonal: dict[int, float]) -> np.ndarray:
    return np.array([seasonal.get(int(season), 0.0) for season in seasons])


def _last_year_scale(periods: pd.DatetimeIndex, values: np.ndarray, grain: str) -> float | None:
    """In-sample mean absolute error of repeating the same period a year earlier, or None."""
    earlier = pd.Series(values, index=periods).reindex(_year_ago(periods, grain)).to_numpy(dtype=float)
    usable = np.isfinite(earlier)
    if usable.sum() < 2:
        return None
    scale = float(np.mean(np.abs(values[usable] - earlier[usable])))
    return scale if scale > 0 else None


def _future_periods(periods: pd.DatetimeIndex, horizon: int, grain: str) -> pd.DatetimeIndex:
    """The next ``horizon`` periods on the same calendar grain."""
    last = pd.Period(periods[-1], freq=grain)
    return pd.DatetimeIndex([(last + offset).to_timestamp() for offset in range(1, horizon + 1)])


def _prediction_spread(positions: np.ndarray, future: np.ndarray, scale: float) -> np.ndarray:
    """Uncertainty at each future position, widening with distance from the data.

    Least-squares prediction-interval shape: noise, plus level uncertainty, plus
    slope uncertainty, which grows with distance. Flat at ``scale`` for fewer
    than three positions.
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
    floor_at_zero: bool = True,
    grain: str | None = None,
) -> Forecast | None:
    """Forecast the next periods, or return None when history is too thin.

    With ``floor_at_zero`` and a history that never goes below zero, the
    forecast and its band stop at zero, as a count or an amount must. Pass
    False for a level (``MetricSpec.floors_at_zero``): a temperature can keep
    falling past 0 °C. ``grain`` is the grain the trend was built at
    (``TrendSeries.frequency``); without it, it is read from the spacing of
    the periods. At a grain below a year, the usual offset of each month,
    quarter or week of the year is added once the history allows.
    """
    if trend.empty or not {"Period", "Value"}.issubset(trend.columns):
        return None
    # Read the grain before empty periods are dropped: from the periods with
    # values alone, months with gaps between them can read as quarters.
    grain = grain or period_grain(pd.DatetimeIndex(pd.to_datetime(trend["Period"])))
    # Drop unobserved (NaN) periods before the length check so only measured
    # periods count as history.
    trend = observed_periods(trend)
    if len(trend) < min_periods:
        return None

    periods = pd.DatetimeIndex(pd.to_datetime(trend["Period"]))
    values = trend["Value"].to_numpy(dtype=float)
    count = len(values)
    horizon = max(1, min(horizon, count // 2))

    line = fit_trendline(periods, values, grain)
    positions = np.asarray(line.positions, dtype=float)
    residuals = line.residuals(values)

    seasonal = _yearly_seasonality(periods, residuals, grain)
    if seasonal:
        residuals = residuals - _seasonal_adjustment(season_of(periods, grain), seasonal)

    scale = robust_scale(residuals)
    future_periods = _future_periods(periods, horizon, grain)
    future_positions = line.future_positions(horizon)

    predictions = line.at(future_positions) + _seasonal_adjustment(
        season_of(future_periods, grain), seasonal
    )
    band = BAND_DEVIATIONS * _prediction_spread(positions, future_positions, scale)

    floored = floor_at_zero and float(values.min()) >= 0
    if floored:
        predictions = np.maximum(predictions, 0.0)
    lower = predictions - band
    upper = predictions + band
    if floored:
        lower = np.maximum(lower, 0.0)

    method = "Theil–Sen trendline"
    if seasonal:
        method += f" + {'month' if grain == 'D' else YEAR_AGO_NAMES[grain]}-of-year seasonality"
    if scale > 0:
        method += f" · band = ±{BAND_DEVIATIONS:g} robust deviations, widening with horizon"
    else:
        # Zero variation gives a zero-width band; say so instead of calling it a
        # widening interval.
        method += " · history shows no variation, so no uncertainty band could be estimated"

    return Forecast(
        periods=tuple(pd.Timestamp(period) for period in future_periods),
        values=tuple(float(value) for value in predictions),
        lower=tuple(float(value) for value in lower),
        upper=tuple(float(value) for value in upper),
        backtest=_backtest(periods, values, grain=grain, floor_at_zero=floor_at_zero),
        method=method,
    )


def _backtest(
    periods: pd.DatetimeIndex, values: np.ndarray, *, grain: str, floor_at_zero: bool = True
) -> Backtest:
    """Refit on a training split and score the held-out tail.

    When the time of year is modelled, the score is also set against
    repeating the same period a year earlier, the forecast a seasonal series
    must beat.
    """
    count = len(values)
    holdout = min(max(3, count // 5), count - MIN_PERIODS + 3)
    # Guard the slicing below: periods[:-0] is empty, and a negative holdout
    # would slice from the front.
    if holdout < 1 or count - holdout < 5:
        return Backtest(mape=None, mase=None, holdout_periods=0, periods_without_mape=0)

    train_periods, train_values = periods[:-holdout], values[:-holdout]
    line = fit_trendline(train_periods, train_values, grain)
    train_residuals = line.residuals(train_values)

    seasonal = _yearly_seasonality(train_periods, train_residuals, grain)

    holdout_periods = periods[-holdout:]
    # The held-out periods' own calendar positions, so a gap among them is kept.
    horizon_positions = period_positions(periods, grain)[-holdout:]
    predicted = line.at(horizon_positions) + _seasonal_adjustment(
        season_of(holdout_periods, grain), seasonal
    )
    if floor_at_zero and float(train_values.min()) >= 0:
        # Clip at zero like build_forecast, so the score matches what is shown.
        predicted = np.maximum(predicted, 0.0)
    actual = values[-holdout:]
    errors = np.abs(predicted - actual)

    measurable = np.abs(actual) > 1e-9
    mape = (
        round(float(np.mean(errors[measurable] / np.abs(actual[measurable])) * 100), 1)
        if measurable.any()
        else None
    )

    # MASE denominator: in-sample MAE of a one-step no-change forecast.
    no_change_error = float(np.mean(np.abs(np.diff(train_values))))
    mase = round(float(np.mean(errors) / no_change_error), 2) if no_change_error > 0 else None
    last_year_error = _last_year_scale(train_periods, train_values, grain) if seasonal else None
    seasonal_mase = round(float(np.mean(errors) / last_year_error), 2) if last_year_error else None

    return Backtest(
        mape=mape,
        mase=mase,
        holdout_periods=holdout,
        periods_without_mape=int((~measurable).sum()),
        seasonal_mase=seasonal_mase,
        season=YEAR_AGO_NAMES.get(grain, "") if seasonal_mase is not None else "",
    )
