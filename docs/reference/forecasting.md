# Forecasting

**Code:** `ada/forecasting.py` · **Tests:** `tests/test_forecasting.py`

Projects the measure forward and reports how well the same method did on
recent history. The model is a simple, explainable baseline.

## The model

Theil-Sen trendline plus the usual offset for the time of year.

1. Drop periods with no observation, then fit the shared trendline over the
   period values. The calendar positions use the grain the trend was built at
   (`grain=TrendSeries.frequency`), read from the period spacing before empty
   periods are dropped, so months with gaps between them stay months.
2. At any grain below a year (day, week, month, quarter) with a year and a
   half of history (`MIN_SEASONAL_YEARS`: 18 months, 6 quarters, 78 weeks),
   learn a seasonal adjustment: the median residual of each month, quarter or
   week of the year seen in at least two different years, centred so it does
   not shift the level. A daily series is offset by its month, since an offset
   for each of 365 days would rest on one or two values.
3. Project the line forward and add the seasonal adjustment back.
4. Put a band around the projection at ±2 robust deviations
   (`BAND_DEVIATIONS`), widened with distance using the least-squares
   prediction-interval shape (noise, plus level and slope uncertainty). This is
   an approximation for a median fit, so the backtest is the better guide to
   accuracy.
5. If the history has no negative values, clamp the forecast and the lower
   band at zero, unless the measure is a reading whose scale goes below zero
   (`SIGNED_LEVEL_WORDS`: a temperature, an elevation, or a name with °). A
   temperature falling past 0 °C keeps falling, band included; rainfall or a
   concentration stops at zero. Callers pass
   `floor_at_zero=MetricSpec.floors_at_zero`.

The `method` string on the result names what was used and is shown next to the
chart. When the history has no variation, the method says that no band could
be estimated.

## Guards

| Guard | Value | Reason |
|---|---|---|
| Minimum history | 8 observed periods (`MIN_PERIODS`) | Below that, `build_forecast` returns `None` |
| Maximum horizon | Half the history | 12 periods of history allow at most 6 forecast periods |
| Default horizon | 6 periods | Capped by the rule above |
| Seasonality | A year and a half of days, weeks, months or quarters | A yearly pattern needs each season seen twice |
| Non-negative clamp | A measure whose history has no negatives, other than a temperature or elevation | It cannot go below zero; a temperature can |

## The backtest

Each forecast includes one. The last periods are held out (a fifth of the
history, at least 3, leaving at least 5 to fit on), the model is refitted on
the rest, and the held-out periods are scored.

| Metric | Meaning |
|---|---|
| MAPE | Mean absolute percentage error on the held-out periods |
| MASE | Mean absolute error on the held-out periods ÷ the in-sample mean absolute error of a one-step no-change forecast |
| Seasonal MASE | The same error ÷ the in-sample mean absolute error of repeating the same period a year earlier; only when the forecast models the time of year |

A MASE below 1.0 means the forecast beat "next period equals this period". At
1.0 it did not, and above 1.0 it did worse; `describe_backtest` says so
plainly, for example: "held out the last 3 periods, ±14.9% average error,
MASE 2.33: worse than simply assuming no change, so do not rely on this
forecast". A seasonal forecast must also beat repeating the same month or
quarter of the previous year, which a strong yearly cycle makes a hard
benchmark: "seasonal MASE 0.81, better than repeating the same quarter a year
earlier", or "worse than ..., so do not rely on this forecast".
`Backtest.worse_than_no_change`, `Backtest.worse_than_last_year` and
`Forecast.reliable` carry the verdicts, `Backtest.shortfall` names the simpler
forecast that won, and `Forecast.summary` puts the projection and the backtest
in one sentence.

The dashboard does not draw a forecast whose backtest is worse than no change
or than last year's same period.
It says why under the chart, with a toggle to draw it anyway, since a dashed
line that lost to "no change" on held-out periods would still look like a
prediction.

`periods_without_mape` counts held-out periods whose actual value is zero.
They are excluded from MAPE, since a percentage error against zero is
undefined.

## Edge cases

| Situation | Behavior |
|---|---|
| Fewer than 8 observed periods | `None`; no forecast shown |
| 10 periods, horizon 6 requested | Horizon capped at 5 |
| Yearly grain | No seasonal component |
| Under a year and a half of history | No seasonal component |
| Monthly with gaps between the recorded months | Fitted and projected by the month, not re-read as quarters |
| History never negative | Forecast and lower band clamped at 0, unless the measure is a level |
| Held-out values of zero | Excluded from MAPE, counted separately |
| No variation in history | Zero-width band, stated in `method` |
| Backtest worse than no change (MASE above 1) or than last year's same period (seasonal MASE above 1) | Stated plainly; not drawn on the dashboard unless asked for |
| Grain finer than a day | Forecast in seconds, minutes or hours, labelled by the clock |

`ada.Analysis.forecast()` returns a `Refusal` with code `thin_history` where
`build_forecast` returns `None`, and one with code `no_measure_column` when the
table has no measure and its rows per period are fixed by its design (each
plot once a year; see [Fixed designs](evidence.md#fixed-designs)), or
`not_enough_periods` when the dates cluster into a few sampling trips with
most periods between them empty ([Sampling trips](evidence.md#sampling-trips)).

## Changing this

Any replacement model must keep two properties:

1. Return `None` on thin history rather than a guess.
2. Report a backtest that can say "no better than assuming no change".

Rolling-origin backtesting is on the [roadmap](../../ROADMAP.md); it uses
short histories better than a single holdout split.
