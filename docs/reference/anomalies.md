# Anomalies

**Code:** `anomalies.py` · calibration script `tools/calibrate_anomalies.py`
**Tests:** `tests/test_anomalies.py`

Flags periods that sit further from the trendline than they plausibly should.

## How a period gets flagged

1. Fit the shared Theil–Sen trendline over the period totals
   ([Periods](periods.md#the-trendline)).
2. Take each period's **residual** — the gap between what happened and what the
   line expected.
3. Measure the typical residual size with `robust_scale`.
4. Flag any period whose residual exceeds `critical × scale`.
5. Sort by severity, return the worst 5.

Each `Anomaly` carries the period, the observed value, the expected value, the
expected range, the direction (`above` / `below`), and the severity in
scale units.

Needs at least **8 periods** (`MIN_PERIODS`). Below that, nothing is flagged —
there is not enough history to know what normal looks like.

## Why the threshold is measured, not chosen

The obvious approach is "flag anything beyond 3 standard deviations". It gives a
detector that cries wolf.

Two reasons:

- The decision is being made over **every period at once**, not one. Check 24
  periods against a 1-in-100 rule and you expect roughly one false flag per
  series just from volume.
- The robust scale is itself **unstable on short histories**. With 10 periods it
  wobbles enough to push ordinary points past a fixed line.

A fixed 3.0 raises at least one false flag in roughly **a quarter** of perfectly
stable series. That is a detector nobody trusts by the third dashboard.

So ADA measures the multiplier instead. `tools/calibrate_anomalies.py` simulates
stable series — a straight line plus normal noise, containing nothing to find —
at each history length, and records the multiplier that keeps the false-alarm
rate at **5%** (`FALSE_ALARM_RATE`). One stable series in twenty raises a flag.

## The calibrated table

`CRITICAL_VALUES` holds the result, as `(history length, multiplier)` pairs:

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

Note the shape: **7.01 at 8 periods, settling near 3.8 by 60**. Short histories
need a much wider band, which is precisely what a hand-picked 3.0 gets wrong.

`critical_value(periods)` interpolates across this table on a log scale, and
holds flat beyond either end.

## Regenerating the table

```bash
python tools/calibrate_anomalies.py
```

Run this after changing the fit, the scale estimator, or the target false-alarm
rate. Paste the output back into `CRITICAL_VALUES`. The script is not imported
by the app — it exists so the numbers in the table are reproducible rather than
folklore.

## Known limitation: heavy tails

The 5% guarantee assumes roughly normal noise. A spiky measure — one where a
couple of large deals genuinely dominate a month — will exceed the band more
often than one series in twenty, because for that measure those periods are
ordinary rather than anomalous.

Calibration for heavy-tailed measures is on the [roadmap](../../ROADMAP.md).

## Trading the guarantee for sensitivity

`detect_anomalies(trend, threshold=3.0)` overrides the calibrated multiplier.
That is occasionally the right call, but it gives up the false-alarm guarantee,
so it should be a deliberate decision rather than a default.

## Edge cases

| Situation | Behavior |
|---|---|
| Fewer than 8 periods | Returns empty |
| Perfectly flat series (scale 0) | Returns empty — not "everything is an anomaly" |
| More than 5 anomalies | The 5 most severe |
| Missing `Period`/`Value` columns | Returns empty |
