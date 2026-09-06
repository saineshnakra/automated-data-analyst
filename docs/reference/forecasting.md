# Forecasting

**Code:** `forecasting.py` · **Tests:** `tests/test_forecasting.py`

Projects the measure forward, and says plainly when the projection is not worth
much.

The model is deliberately simple and fully explainable. The point is not
accuracy at any cost — it is a baseline whose limits are visible.

## The model

**Theil–Sen trendline + optional month-of-year seasonality.**

1. Fit the shared trendline over the period totals.
2. If the grain is monthly **and** there are at least **18 periods**, learn a
   seasonal adjustment from the median residual of each calendar month.
3. Project the line forward, adding the seasonal adjustment back.
4. Put a band around it at **±2 robust deviations**, widening with the horizon.
5. If the history never goes negative, clamp both the forecast and the lower
   band at zero.

The `method` string on the result spells out exactly what was used, and it is
shown next to the chart.

## The guards

| Guard | Value | Why |
|---|---|---|
| Minimum history | 8 periods | Below that, returns `None` rather than guessing |
| Maximum horizon | Half the history | 12 periods of history means at most 6 forecast |
| Default horizon | 6 periods | Capped by the rule above |
| Seasonality | 18+ monthly periods | You need more than one year to learn a yearly pattern |
| Non-negative clamp | If history has no negatives | Stops a projected count going below zero |

The band widens with distance because uncertainty does. A flat band would claim
month 6 is as knowable as month 1.

## The backtest

Every forecast ships with one. The last stretch of history is held out, the
model is refitted on what is left, and the held-out periods are scored — so the
error is measured on periods the fit never saw.

Two numbers come back:

| Metric | Meaning |
|---|---|
| **MAPE** | Mean absolute percentage error on the holdout |
| **MASE** | Holdout error ÷ the error of assuming no change |

**MASE is the one that matters.** Below 1.0, the forecast beat "assume next
period equals this period". At or above 1.0, it did not — and `describe_backtest`
says so in the interface instead of hiding it.

A forecast that cannot beat no-change is not automatically useless, but the user
deserves to know before planning around it.

`periods_without_mape` counts holdout periods too near zero for a percentage
error to be meaningful. Those are excluded from MAPE rather than allowed to
produce a huge meaningless number.

## Edge cases

| Situation | Behavior |
|---|---|
| Fewer than 8 periods | Returns `None`, no forecast shown |
| 10 periods, horizon 6 requested | Horizon capped to 5 |
| Weekly or quarterly grain | No seasonal component |
| Monthly but under 18 periods | No seasonal component |
| History always ≥ 0 | Forecast and lower band clamped at 0 |
| Holdout values near zero | Excluded from MAPE, counted separately |

## Changing this

Two rules survive any model change:

1. **Refuse rather than guess.** Thin history returns `None`.
2. **Ship the honest error.** Whatever replaces the current backtest must still
   be able to say "this was no better than assuming no change."

Rolling-origin backtesting and a seasonal-naive comparison are on the
[roadmap](../../ROADMAP.md) — both make better use of short histories than the
current single holdout split.
