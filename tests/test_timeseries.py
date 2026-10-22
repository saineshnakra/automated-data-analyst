import unittest

import numpy as np
import pandas as pd
from scipy import stats

from ada.timeseries import (
    MAD_SCALE,
    MEAN_ABS_DEV_SCALE,
    best_split,
    f_test_p_value,
    fit_trendline,
    level_shift,
    observed_periods,
    period_grain,
    period_positions,
    robust_scale,
    shift_critical_value,
    theil_sen,
    trend_test,
    yearly_pattern,
)


def month_end_trend(values):
    return pd.DataFrame(
        {"Period": pd.date_range("2022-01-31", periods=len(values), freq="ME"), "Value": values}
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

    def test_sparse_months_keep_the_grain_they_were_built_at(self):
        periods = pd.DatetimeIndex(["2003-06-01", "2003-09-01", "2003-11-01", "2004-04-01", "2004-05-01"])

        self.assertEqual(period_positions(periods, "M").tolist(), [0.0, 3.0, 5.0, 10.0, 11.0])
        self.assertNotEqual(period_positions(periods).tolist(), [0.0, 3.0, 5.0, 10.0, 11.0])

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


class ScaleEstimatorAgreementTests(unittest.TestCase):
    """MAD_SCALE and MEAN_ABS_DEV_SCALE each turn their estimator into sigma for normal noise."""

    def test_the_fallback_recovers_the_same_sigma_as_the_median_path(self):
        sample = np.random.default_rng(3).normal(0.0, 1.0, 200_000)
        deviations = np.abs(sample - np.median(sample))

        median_estimate = float(np.median(deviations)) * MAD_SCALE
        fallback_estimate = float(np.mean(deviations)) * MEAN_ABS_DEV_SCALE

        self.assertAlmostEqual(median_estimate, 1.0, places=2)
        self.assertAlmostEqual(fallback_estimate, 1.0, places=2)

    def test_the_fallback_is_used_when_most_residuals_are_identical(self):
        residuals = np.array([0.0] * 12 + [1.0, -1.0, 2.0, -2.0, 1.5, -1.5, 0.5, -0.5])

        # The median absolute deviation is zero here, so the mean path runs.
        self.assertEqual(float(np.median(np.abs(residuals - np.median(residuals)))), 0.0)
        self.assertAlmostEqual(
            robust_scale(residuals), float(np.mean(np.abs(residuals))) * MEAN_ABS_DEV_SCALE
        )

    def test_a_series_with_no_variation_has_no_scale(self):
        self.assertEqual(robust_scale(np.zeros(10)), 0.0)


class ObservedPeriodsTests(unittest.TestCase):
    def test_a_frame_with_no_gaps_is_returned_untouched(self):
        frame = month_end_trend([1.0, 2.0, 3.0])
        self.assertIs(observed_periods(frame), frame)

    def test_gaps_are_dropped_and_the_rest_kept_in_order(self):
        frame = month_end_trend([1.0, np.nan, 3.0])
        kept = observed_periods(frame)
        self.assertEqual(kept["Value"].tolist(), [1.0, 3.0])
        self.assertTrue(kept["Period"].is_monotonic_increasing)

    def test_an_empty_or_shapeless_frame_is_survivable(self):
        self.assertTrue(observed_periods(pd.DataFrame()).empty)
        odd = pd.DataFrame({"Period": []})
        self.assertIs(observed_periods(odd), odd)


class SubDayGrainTests(unittest.TestCase):
    def test_periods_finer_than_a_day_are_named(self):
        for freq, grain in (("s", "s"), ("min", "min"), ("h", "h")):
            periods = pd.date_range("2024-01-01", periods=10, freq=freq)
            self.assertEqual(period_grain(periods), grain)

    def test_minutes_are_numbered_by_the_minute(self):
        periods = pd.date_range("2024-01-01 15:00", periods=4, freq="min").delete(2)

        self.assertEqual(period_positions(periods).tolist(), [0.0, 1.0, 3.0])


class YearlyPatternTests(unittest.TestCase):
    def test_the_f_test_matches_its_table(self):
        self.assertAlmostEqual(f_test_p_value(4.94, 3, 20), 0.01, places=3)
        self.assertAlmostEqual(f_test_p_value(4.96, 1, 10), 0.05, places=3)

    def test_a_cycle_that_repeats_every_year_is_found(self):
        quarters = pd.date_range("2018-01-01", periods=24, freq="QS")
        rng = np.random.default_rng(4)
        residuals = np.tile([20.0, 0.0, -15.0, -5.0], 6) + rng.normal(0, 2, 24)

        offsets = yearly_pattern(quarters, residuals)

        self.assertIsNotNone(offsets)
        self.assertGreater(offsets[0], 15)
        self.assertLess(offsets[2], -10)

    def test_noise_has_no_yearly_pattern(self):
        months = pd.date_range("2018-01-01", periods=36, freq="MS")
        residuals = np.random.default_rng(9).normal(0, 1, 36)

        self.assertIsNone(yearly_pattern(months, residuals))

    def test_too_short_a_history_is_not_judged(self):
        quarters = pd.date_range("2022-01-01", periods=8, freq="QS")

        self.assertIsNone(yearly_pattern(quarters, np.tile([5.0, 0.0, -5.0, 0.0], 2)))

    def test_an_unusual_period_does_not_set_its_own_expectation(self):
        quarters = pd.date_range("2018-01-01", periods=24, freq="QS")
        residuals = np.tile([20.0, 0.0, -15.0, -5.0], 6)
        residuals[8] = 80.0

        offsets = yearly_pattern(quarters, residuals, min_cycles=3)

        self.assertLess(offsets[8], 25)


class TrendTestTests(unittest.TestCase):
    def test_a_steady_rise_has_an_interval_above_zero(self):
        rng = np.random.default_rng(70)
        years = np.arange(1910, 2010, dtype=float)
        values = 26 + 0.014 * (years - 1910) + rng.normal(0, 0.3, years.size)

        found = trend_test(years, values)

        assert found is not None
        self.assertTrue(found.clear)
        self.assertLess(found.low, 0.014)
        self.assertGreater(found.high, 0.014)
        self.assertLess(found.p_value, 0.001)

    def test_five_noisy_points_have_no_clear_trend(self):
        found = trend_test(np.arange(5.0), np.array([60.0, 70.0, 58.0, 65.0, 68.0]))

        assert found is not None
        self.assertFalse(found.clear)
        self.assertLess(found.low, 0)
        self.assertGreater(found.high, 0)

    def test_autocorrelation_widens_the_interval_and_weakens_the_test(self):
        rng = np.random.default_rng(71)
        noise = np.zeros(60)
        for step in range(1, 60):
            noise[step] = 0.8 * noise[step - 1] + rng.normal()
        positions = np.arange(60.0)
        values = 0.03 * positions + noise

        found = trend_test(positions, values)
        assert found is not None
        tau, naive_p = stats.kendalltau(positions, values, method="asymptotic")

        self.assertGreater(found.autocorrelation, 0.3)
        self.assertGreater(found.p_value, naive_p)
        naive = stats.theilslopes(values, positions)
        self.assertGreater(found.high - found.low, naive.high_slope - naive.low_slope)

    def test_too_few_points_or_a_flat_series(self):
        self.assertIsNone(trend_test(np.arange(3.0), np.array([1.0, 2.0, 3.0])))
        flat = trend_test(np.arange(6.0), np.full(6, 4.0))
        assert flat is not None
        self.assertFalse(flat.clear)


class LevelShiftTests(unittest.TestCase):
    def test_a_step_is_found_with_the_levels_on_either_side(self):
        values = np.r_[5.5, 103.5, 805.5, 795.0, 791.6, 840.3, 924.7, 958.9]

        found = level_shift(values)

        assert found is not None
        self.assertEqual(found.position, 2)
        self.assertAlmostEqual(found.before, 54.5)
        self.assertAlmostEqual(found.after, values[2:].mean())

    def test_the_split_matches_the_least_squared_error(self):
        rng = np.random.default_rng(61)
        values = np.r_[rng.normal(90, 10, 40), rng.normal(35, 10, 20)]

        found = best_split(values)

        assert found is not None
        errors = [
            ((values[:k] - values[:k].mean()) ** 2).sum() + ((values[k:] - values[k:].mean()) ** 2).sum()
            for k in range(9, 52)
        ]
        self.assertEqual(found.position, 9 + int(np.argmin(errors)))
        self.assertIsNotNone(level_shift(values))

    def test_stable_series_rarely_show_a_shift(self):
        rng = np.random.default_rng(62)
        flagged = sum(level_shift(rng.normal(0, 1, 40)) is not None for _ in range(300))
        self.assertLessEqual(flagged / 300, 0.09)

    def test_a_steady_rise_is_not_a_shift(self):
        rng = np.random.default_rng(63)
        flagged = sum(level_shift(np.arange(30) + rng.normal(0, 1, 30)) is not None for _ in range(100))
        self.assertLessEqual(flagged, 5)

    def test_one_extreme_period_at_an_end_is_not_a_new_level(self):
        rng = np.random.default_rng(64)
        values = rng.normal(500, 30, 20)
        values[-1] = 40_000

        self.assertIsNone(level_shift(values))

    def test_short_or_flat_series_have_none(self):
        self.assertIsNone(level_shift(np.array([1.0, 5.0, 9.0])))
        self.assertIsNone(level_shift(np.full(20, 3.0)))

    def test_the_critical_values_rise_for_short_series(self):
        self.assertGreater(shift_critical_value(8), shift_critical_value(60))


if __name__ == "__main__":
    unittest.main()
