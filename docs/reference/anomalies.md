# Anomalies

**Code:** `ada/anomalies.py`, calibration script `tools/calibrate_anomalies.py`
**Tests:** `tests/test_anomalies.py`

Flags periods that sit further from the trendline than expected.

## How a period is flagged

1. Drop periods whose value is NaN, such as an empty period of an averaged
   measure.
2. Fit the shared Theil-Sen trendline over the period values
   ([Periods](periods.md#the-trendline)).
3. Take each period's residual: the actual value minus the line's value.
4. If the series is monthly or quarterly, covers at least three years
   (`MIN_SEASONAL_CYCLES`) and has a yearly pattern
   ([Periods](periods.md#yearly-patterns)), add each period's usual offset for
   its month or quarter to the expected value, and take the residual from
   that. Each winter is then compared with other winters, so a snowy first
   quarter every year is not flagged, but an unusually snowy one still is.
5. Measure the typical residual size with `robust_scale`.
6. Flag each period whose residual exceeds `critical_value(n) × scale`, where
   `n` is the number of observed periods.
7. Sort by severity and return the five most severe.

For a measure that cannot be negative, the expected range stops at zero when
the history never goes below it, as the forecast does: callers pass
`floor_at_zero=MetricSpec.floors_at_zero`, so a temperature keeps its full
range and rainfall does not get a band reaching -1.0M. The floor changes only
the range shown, not which periods are flagged.

Each `Anomaly` has the period, the observed value, the expected value, the
expected range, the direction (`above` or `below`), the severity in scale
units, and `seasonal`, which says whether the expected value allows for the
time of year.

Re-running the calibration simulation with this step in place still gives
about 5% of stable series a false flag at 24 to 60 periods, and a strongly
seasonal quarterly series no more often.

At least 8 observed periods are needed (`MIN_PERIODS`). When the chosen grain
leaves fewer, `finer_trend` tries the finer grains in turn and checks the
first one with 8 recorded periods: 147 daily readings roll up to 5 months, too
few, but give 21 weeks. A grain finer than the data was recorded at only adds
empty periods, so one with over a fifth of its periods empty
(`MAX_EMPTY_SHARE`) is skipped. The brief's card then says which grain it
checked ("Periods are checked by week, since by month there are only 5"), and
`Analysis.anomalies` uses the same fallback. With no grain that has enough
periods, nothing is flagged.

## Why the threshold is calibrated

A fixed multiplier such as 3 flags stable series too often, for two reasons:

- Every period is tested, not one. Testing 24 periods against a 1-in-100 rule
  gives about one false flag per series.
- The robust scale is unstable on short histories. With 10 periods it varies
  enough to push ordinary points past a fixed line.

A fixed 3.0 raises at least one false flag in about a quarter of stable
series.

`tools/calibrate_anomalies.py` instead simulates stable series (a straight line
plus normal noise) at each history length and records the multiplier that
keeps the share of series with any false flag at 5% (`FALSE_ALARM_RATE`). One
stable series in twenty raises a flag.

## The calibrated table

`CRITICAL_VALUES` holds `(history length, multiplier)` pairs:

| Periods | Multiplier | | Periods | Multiplier |
|---|---|---|---|---|
| 8 | 7.01 | | 60 | 3.79 |
| 10 | 5.82 | | 80 | 3.79 |
| 12 | 5.21 | | 110 | 3.77 |
| 16 | 4.63 | | 150 | 3.79 |
| 20 | 4.33 | | 220 | 3.88 |
| 26 | 4.07 | | 320 | 3.88 |
| 34 | 3.94 | | | |
| 45 | 3.88 | | | |

The multiplier is 7.01 at 8 periods and settles near 3.8 by 60. Short
histories need a much wider band.

`critical_value(periods)` interpolates across the table on a log scale and is
flat beyond either end.

## Regenerating the table

```bash
python tools/calibrate_anomalies.py
```

Run it after changing the fit, the scale estimator or the target false-alarm
rate, and paste the output into `CRITICAL_VALUES`. The script uses a fixed
seed, so the table is reproducible. It is not imported by the app or the
package.

## Known limitation: heavy tails

The 5% rate assumes roughly normal noise. A spiky measure, such as one where a
few large deals dominate some months, exceeds the band more often than one
series in twenty. Calibration for heavy-tailed measures is on the
[roadmap](../../ROADMAP.md).

## Overriding the multiplier

`detect_anomalies(trend, threshold=3.0)` replaces the calibrated multiplier
with a fixed one. The 5% false-alarm rate does not hold for a fixed multiplier.

## Edge cases

| Situation | Behavior |
|---|---|
| Fewer than 8 observed periods | Returns an empty tuple |
| Perfectly flat series (scale 0) | Returns an empty tuple |
| More than 5 anomalies | The 5 most severe |
| A yearly pattern over 3 or more years | Each period is judged against the same month or quarter in other years |
| A measure that cannot be negative | The expected range stops at zero |
| Missing `Period` or `Value` column | Returns an empty tuple |

`ada.Analysis.anomalies` returns a `Refusal` (`no_date_column` or
`not_enough_periods`) when it cannot check, and an empty tuple when it checked
and found nothing.
