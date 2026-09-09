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

The held-out periods are scored at their own calendar positions, the same
positions the trendline is fitted on. A month missing from inside the holdout
leaves a hole rather than shifting every later prediction one period early.
And the scored forecast is the one the page shows: a series that never goes
negative is clipped at zero in the backtest exactly as it is on the chart.

Three numbers come back:

| Metric | Meaning |
|---|---|
| **MAPE** | Mean absolute percentage error of the model on the holdout |
| **holdout_naive_mape** | The same percentage error for a no-change forecast (the last training value carried across the holdout) |
| **MASE** | Holdout error ÷ the mean one-step movement of the training history |

**MASE and the no-change contest answer different questions.** MASE is the
textbook mean absolute scaled error: its denominator is how far the series
moved from one period to the next over the training history, so a MASE of 0.5
means the holdout misses were half the size of a typical period-to-period
change. That says how large the error is in the series' own units. It does
*not* say whether assuming no change would have done better on the held-out
periods, because no-change was never scored there.

So the backtest also runs that contest. The last training value is carried
across the holdout and scored on exactly the same periods as the model, and
`beats_naive_on_holdout` says who won. A zig-zag history whose holdout sits at
the last training value gets a MASE of 0.5 and still loses: the earlier note,
which read the verdict off MASE alone, would have called that "better than
assuming no change".

`describe_backtest` reports both, in words that say what each measured:

> average error of 1.7% on the last 4 held-out periods, assuming no change
> would have erred 4.9% on the same periods, so the model beat it, MASE 0.72
> (error relative to a one-step no-change baseline on the training history)

The percentage is an average of past misses, so it is never written as
"±1.7%". That sign reads as an uncertainty interval, which MAPE is not. The
band on the chart has its own description in the `method` string, kept
separate from the error note.

A forecast that cannot beat no-change is not automatically useless, but the user
deserves to know before planning around it.

`periods_without_mape` counts holdout periods too near zero for a percentage
error to be meaningful. Those are excluded from both percentage errors rather
than allowed to produce a huge meaningless number; when every held-out period
is zero, the note says so and only MASE is reported.

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
   score a no-change forecast on the same held-out periods and be able to say
   "the model did not beat it." MASE alone cannot say that.

Rolling-origin backtesting and a seasonal-naive comparison are on the
[roadmap](../../ROADMAP.md) — both make better use of short histories than the
current single holdout split.
