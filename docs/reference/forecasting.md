# Forecasting

**Code:** `ada/forecasting.py` · **Tests:** `tests/test_forecasting.py`

Projects the measure forward and reports how well the same method did on
recent history. The model is a simple, explainable baseline.

## The model

Theil-Sen trendline plus optional month-of-year seasonality.

1. Drop periods with no observation, then fit the shared trendline over the
   period values.
2. If the grain is monthly and there are at least 18 periods
   (`MIN_SEASONAL_PERIODS`), learn a seasonal adjustment: the median residual of
   each calendar month that has at least two observations, centred so it does
   not shift the level.
3. Project the line forward and add the seasonal adjustment back.
4. Put a band around the projection at ±2 robust deviations
   (`BAND_DEVIATIONS`), widened with distance using the least-squares
   prediction-interval shape (noise, plus level and slope uncertainty). This is
   an approximation for a median fit, so the backtest is the better guide to
   accuracy.
5. If the history has no negative values, clamp the forecast and the lower
   band at zero.

The `method` string on the result names what was used and is shown next to the
chart. When the history has no variation, the method says that no band could
be estimated.

## Guards

| Guard | Value | Reason |
|---|---|---|
| Minimum history | 8 observed periods (`MIN_PERIODS`) | Below that, `build_forecast` returns `None` |
| Maximum horizon | Half the history | 12 periods of history allow at most 6 forecast periods |
| Default horizon | 6 periods | Capped by the rule above |
| Seasonality | 18 or more monthly periods | A yearly pattern needs more than one year |
| Non-negative clamp | History has no negatives | A count or amount cannot go below zero |

## The backtest

Each forecast includes one. The last periods are held out (a fifth of the
history, at least 3, leaving at least 5 to fit on), the model is refitted on
the rest, and the held-out periods are scored.

| Metric | Meaning |
|---|---|
| MAPE | Mean absolute percentage error on the held-out periods |
| MASE | Mean absolute error on the held-out periods ÷ the in-sample mean absolute error of a one-step no-change forecast |

A MASE below 1.0 means the forecast beat "next period equals this period". At
1.0 or above it did not, and `describe_backtest` says so in the interface, for
example: "held out the last 3 periods, ±14.9% average error, MASE 2.33, no
better than assuming no change".

`periods_without_mape` counts held-out periods whose actual value is zero.
They are excluded from MAPE, since a percentage error against zero is
undefined.

## Edge cases

| Situation | Behavior |
|---|---|
| Fewer than 8 observed periods | `None`; no forecast shown |
| 10 periods, horizon 6 requested | Horizon capped at 5 |
| Weekly, quarterly or yearly grain | No seasonal component |
| Monthly, under 18 periods | No seasonal component |
| History never negative | Forecast and lower band clamped at 0 |
| Held-out values of zero | Excluded from MAPE, counted separately |
| No variation in history | Zero-width band, stated in `method` |

`ada.Analysis.forecast()` returns a `Refusal` with code `thin_history` where
`build_forecast` returns `None`.

## Changing this

Any replacement model must keep two properties:

1. Return `None` on thin history rather than a guess.
2. Report a backtest that can say "no better than assuming no change".

Rolling-origin backtesting and a seasonal-naive comparison are on the
[roadmap](../../ROADMAP.md); both use short histories better than a single
holdout split.
