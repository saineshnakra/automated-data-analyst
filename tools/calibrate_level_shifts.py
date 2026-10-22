"""Regenerate the level-shift check's critical-value table.

For each history length, simulate stable series (normal noise around a fixed
level), find each one's best two-level split with ``timeseries.best_split``,
and report the percentile of its t statistic, among splits that also pass
``passes_shift_rules``, above which only ``FALSE_ALARM_RATE`` of the series
fall. A threshold never drops below 2.0, about the 5% point of a single t
test. Rerun after changing the split, the statistic or the rules, and paste
the output into ``timeseries.SHIFT_CRITICAL_VALUES``:

    python tools/calibrate_level_shifts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ada.timeseries import best_split, passes_shift_rules  # noqa: E402

FALSE_ALARM_RATE = 0.05
LENGTHS = (8, 12, 20, 40, 60, 120, 240)
FLOOR = 2.0
SEED = 2026


def calibrate(length: int, trials: int, rng: np.random.Generator) -> float:
    statistics = np.zeros(trials)
    for trial, values in enumerate(rng.normal(0.0, 1.0, (trials, length))):
        found = best_split(values)
        if found is not None and passes_shift_rules(found, length):
            statistics[trial] = found.statistic
    return max(float(np.percentile(statistics, (1.0 - FALSE_ALARM_RATE) * 100.0)), FLOOR)


def main() -> None:
    rng = np.random.default_rng(SEED)
    print(f"False-alarm rate: {FALSE_ALARM_RATE:.0%}")
    print("SHIFT_CRITICAL_VALUES = (")
    for length in LENGTHS:
        trials = 4_000 if length <= 60 else 2_000
        print(f"    ({length}, {calibrate(length, trials, rng):.2f}),")
    print(")")


if __name__ == "__main__":
    main()
