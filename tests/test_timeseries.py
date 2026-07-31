import unittest

import numpy as np
import pandas as pd

from timeseries import (
    fit_trendline,
    period_grain,
    period_positions,
    robust_scale,
    theil_sen,
)


class PeriodPositionTests(unittest.TestCase):
    def test_regular_months_are_numbered_consecutively(self):
        periods = pd.date_range("2024-01-01", periods=6, freq="MS")

        np.testing.assert_array_equal(period_positions(periods), np.arange(6, dtype=float))

    def test_a_missing_month_leaves_a_hole(self):
        periods = pd.DatetimeIndex(["2024-01-01", "2024-02-01", "2024-04-01", "2024-05-01"])

        np.testing.assert_array_equal(period_positions(periods), np.array([0.0, 1.0, 3.0, 4.0]))

    def test_long_monthly_spans_do_not_drift(self):
        periods = pd.date_range("2010-01-01", periods=121, freq="MS")

        positions = period_positions(periods)

        self.assertEqual(positions[-1], 120.0)

    def test_quarters_and_weeks_are_numbered_by_their_own_grain(self):
        quarters = pd.date_range("2020-01-01", periods=9, freq="QS")
        weeks = pd.date_range("2024-01-07", periods=10, freq="W")

        np.testing.assert_array_equal(period_positions(quarters), np.arange(9, dtype=float))
        np.testing.assert_array_equal(period_positions(weeks), np.arange(10, dtype=float))

    def test_grain_classification_follows_median_spacing(self):
        self.assertEqual(period_grain(pd.date_range("2024-01-01", periods=5, freq="D")), "D")
        self.assertEqual(period_grain(pd.date_range("2024-01-01", periods=5, freq="W")), "W")
        self.assertEqual(period_grain(pd.date_range("2024-01-01", periods=5, freq="MS")), "M")
        self.assertEqual(period_grain(pd.date_range("2024-01-01", periods=5, freq="QS")), "Q")
        self.assertEqual(period_grain(pd.date_range("2024-01-01", periods=5, freq="YS")), "Y")

    def test_single_and_empty_inputs_are_safe(self):
        self.assertEqual(len(period_positions(pd.DatetimeIndex([]))), 0)
        np.testing.assert_array_equal(
            period_positions(pd.DatetimeIndex(["2024-01-01"])), np.zeros(1)
        )


class TheilSenTests(unittest.TestCase):
    def test_recovers_an_exact_line(self):
        positions = np.arange(10, dtype=float)
        values = 5.0 + 3.0 * positions

        slope, intercept = theil_sen(positions, values)

        self.assertAlmostEqual(slope, 3.0)
        self.assertAlmostEqual(intercept, 5.0)

    def test_one_wild_period_cannot_bend_the_slope(self):
        positions = np.arange(12, dtype=float)
        values = 100.0 + 10.0 * positions
        values[6] = 5_000.0

        slope, _ = theil_sen(positions, values)

        self.assertAlmostEqual(slope, 10.0)

    def test_beats_consecutive_differences_on_a_noisy_history(self):
        rng = np.random.default_rng(12)
        positions = np.arange(14, dtype=float)
        values = 200.0 + 8.0 * positions + rng.normal(0, 30, 14)

        pairwise, _ = theil_sen(positions, values)
        consecutive = float(np.median(np.diff(values)))

        self.assertLess(abs(pairwise - 8.0), abs(consecutive - 8.0))

    def test_a_gap_does_not_compress_the_timeline(self):
        positions = np.array([0.0, 1.0, 2.0, 3.0, 8.0, 9.0])
        values = 100.0 + 10.0 * positions

        slope, _ = theil_sen(positions, values)

        self.assertAlmostEqual(slope, 10.0)

    def test_degenerate_inputs_return_a_flat_line(self):
        self.assertEqual(theil_sen(np.zeros(0), np.zeros(0)), (0.0, 0.0))
        self.assertEqual(theil_sen(np.array([4.0]), np.array([9.0])), (0.0, 9.0))
        self.assertEqual(theil_sen(np.zeros(3), np.array([1.0, 2.0, 3.0])), (0.0, 2.0))

    def test_very_long_histories_stay_bounded_and_accurate(self):
        positions = np.arange(2_000, dtype=float)
        values = 3.0 + 0.5 * positions

        slope, intercept = theil_sen(positions, values)

        self.assertAlmostEqual(slope, 0.5)
        self.assertAlmostEqual(intercept, 3.0)


class TrendLineTests(unittest.TestCase):
    def test_fitted_values_and_future_positions_line_up(self):
        periods = pd.date_range("2024-01-01", periods=8, freq="MS")
        values = np.array([100.0 + 20.0 * index for index in range(8)])

        line = fit_trendline(periods, values)

        np.testing.assert_allclose(line.fitted(), values, atol=1e-9)
        np.testing.assert_allclose(line.residuals(values), np.zeros(8), atol=1e-9)
        np.testing.assert_array_equal(line.future_positions(3), np.array([8.0, 9.0, 10.0]))
        np.testing.assert_allclose(line.at(line.future_positions(1)), np.array([260.0]))

    def test_a_gapped_history_predicts_the_true_next_value(self):
        periods = pd.DatetimeIndex(["2024-01-01", "2024-02-01", "2024-03-01", "2024-07-01"])
        values = np.array([100.0, 110.0, 120.0, 160.0])

        line = fit_trendline(periods, values)

        self.assertAlmostEqual(line.slope, 10.0)
        self.assertAlmostEqual(float(line.at(line.future_positions(1))[0]), 170.0)


class RobustScaleTests(unittest.TestCase):
    def test_matches_the_standard_deviation_of_normal_noise(self):
        rng = np.random.default_rng(21)

        scale = robust_scale(rng.normal(0, 10, 4_000))

        self.assertAlmostEqual(scale, 10.0, delta=0.6)

    def test_ignores_a_handful_of_extreme_residuals(self):
        rng = np.random.default_rng(31)
        clean = rng.normal(0, 5, 400)
        contaminated = np.concatenate([clean, np.array([9_000.0, -7_500.0])])

        self.assertAlmostEqual(robust_scale(contaminated), robust_scale(clean), delta=0.2)
        self.assertLess(robust_scale(contaminated), float(np.std(contaminated)) / 10)

    def test_falls_back_when_most_residuals_are_identical(self):
        residuals = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 12.0])

        self.assertGreater(robust_scale(residuals), 0.0)

    def test_perfectly_fitted_residuals_have_no_scale(self):
        self.assertEqual(robust_scale(np.zeros(6)), 0.0)
        self.assertEqual(robust_scale(np.zeros(0)), 0.0)


if __name__ == "__main__":
    unittest.main()
