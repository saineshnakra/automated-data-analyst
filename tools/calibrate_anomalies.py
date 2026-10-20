"""Regenerate the anomaly detector's critical-value table.

For each history length, simulate stable series (a line plus normal noise), fit
them with ``timeseries``, and report the percentile of ``max|residual| / scale``
above which only ``FALSE_ALARM_RATE`` of them fall. Rerun after changing the
trendline fit or the scale estimator and paste the output into
``anomalies.CRITICAL_VALUES``:

    python tools/calibrate_anomalies.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from timeseries import fit_trendline, robust_scale  # noqa: E402

FALSE_ALARM_RATE = 0.05
LENGTHS = (8, 10, 12, 16, 20, 26, 34, 45, 60, 80, 110, 150, 220, 320)
SEED = 2024


def _extreme_ratios(length: int, trials: int, rng: np.random.Generator) -> np.ndarray:
    """Largest scaled residual seen in each simulated stable series."""
    periods = pd.date_range("2000-01-01", periods=length, freq="MS")
    baseline = 100.0 + 2.0 * np.arange(length)
    ratios = np.empty(trials)

    for trial, noise in enumerate(rng.normal(0.0, 1.0, (trials, length))):
        values = baseline + noise
        residuals = values - fit_trendline(periods, values).fitted()
        scale = robust_scale(residuals)
        ratios[trial] = float(np.abs(residuals).max() / scale) if scale else 0.0

    return ratios


def calibrate(length: int, trials: int, rng: np.random.Generator) -> float:
    ratios = _extreme_ratios(length, trials, rng)
    return float(np.percentile(ratios, (1.0 - FALSE_ALARM_RATE) * 100.0))


def main() -> None:
    rng = np.random.default_rng(SEED)
    print(f"Family-wise false-alarm rate: {FALSE_ALARM_RATE:.0%}")
    print("CRITICAL_VALUES = (")
    for length in LENGTHS:
        trials = 12_000 if length <= 40 else 5_000
        print(f"    ({length}, {calibrate(length, trials, rng):.2f}),")
    print(")")


if __name__ == "__main__":
    main()
