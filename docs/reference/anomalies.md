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
4. Measure the typical residual size with `robust_scale`.
5. Flag each period whose residual exceeds `critical_value(n) × scale`, where
   `n` is the number of observed periods.
6. Sort by severity and return the five most severe.

Each `Anomaly` has the period, the observed value, the expected value, the
expected range, the direction (`above` or `below`) and the severity in scale
units.

At least 8 observed periods are needed (`MIN_PERIODS`). With fewer, nothing is
flagged.

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
| Missing `Period` or `Value` column | Returns an empty tuple |

`ada.Analysis.anomalies` returns a `Refusal` (`no_date_column` or
`not_enough_periods`) when it cannot check, and an empty tuple when it checked
and found nothing.
