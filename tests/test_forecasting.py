import unittest

import numpy as np
import pandas as pd

from forecasting import Backtest, build_forecast, describe_backtest


def make_trend(values, freq="MS", start="2024-01-01"):
    periods = pd.date_range(start, periods=len(values), freq=freq)
    return pd.DataFrame({"Period": periods, "Value": [float(value) for value in values]})


class ForecastTests(unittest.TestCase):
    def test_linear_history_extrapolates_with_tiny_error(self):
        trend = make_trend([100 + 10 * index for index in range(16)])

        forecast = build_forecast(trend)

        self.assertIsNotNone(forecast)
        assert forecast is not None
        self.assertEqual(len(forecast.periods), 6)
        self.assertAlmostEqual(forecast.values[0], 260.0, delta=1e-6)
        self.assertAlmostEqual(forecast.backtest.mape or 0.0, 0.0, places=1)
        for lower, value, upper in zip(forecast.lower, forecast.values, forecast.upper, strict=True):
            self.assertLessEqual(lower, value)
            self.assertLessEqual(value, upper)

    def test_short_history_is_refused(self):
        self.assertIsNone(build_forecast(make_trend([100, 110, 120, 130, 140, 150, 160])))

    def test_horizon_never_exceeds_half_the_history(self):
        forecast = build_forecast(make_trend([100 + index for index in range(8)]))

        self.assertIsNotNone(forecast)
        assert forecast is not None
        self.assertEqual(len(forecast.periods), 4)

    def test_monthly_seasonality_is_learned_when_history_allows(self):
        rng = np.random.default_rng(9)
        index = np.arange(30)
        values = 1_000 + 12 * index + 90 * np.sin(2 * np.pi * index / 12) + rng.normal(0, 6, 30)

        forecast = build_forecast(make_trend(values))

        self.assertIsNotNone(forecast)
        assert forecast is not None
        self.assertIn("seasonality", forecast.method)
        self.assertIsNotNone(forecast.backtest.mape)
        self.assertLess(forecast.backtest.mape, 15)

    def test_noisy_history_produces_a_real_uncertainty_band(self):
        rng = np.random.default_rng(4)
        values = 500 + rng.normal(0, 40, 14)

        forecast = build_forecast(make_trend(values))

        self.assertIsNotNone(forecast)
        assert forecast is not None
        for lower, upper in zip(forecast.lower, forecast.upper, strict=True):
            self.assertLess(lower, upper)

    def test_non_negative_history_is_never_forecast_below_zero(self):
        forecast = build_forecast(make_trend([800, 700, 600, 500, 400, 300, 200, 100, 50, 10]))

        self.assertIsNotNone(forecast)
        assert forecast is not None
        self.assertGreaterEqual(min(forecast.values), 0.0)
        self.assertGreaterEqual(min(forecast.lower), 0.0)

    def test_the_band_widens_with_the_horizon(self):
        rng = np.random.default_rng(8)
        trend = make_trend(600 + 5 * np.arange(20) + rng.normal(0, 30, 20))

        forecast = build_forecast(trend)

        assert forecast is not None
        widths = [upper - lower for lower, upper in zip(forecast.lower, forecast.upper, strict=True)]
        self.assertTrue(all(later > earlier for earlier, later in zip(widths, widths[1:], strict=False)))
        self.assertGreater(widths[-1], widths[0] * 1.05)

    def test_a_forecastable_series_beats_a_no_change_forecast(self):
        rng = np.random.default_rng(15)
        trend = make_trend(1_000 + 40 * np.arange(24) + rng.normal(0, 20, 24))

        forecast = build_forecast(trend)

        assert forecast is not None
        self.assertIsNotNone(forecast.backtest.mase)
        self.assertLess(forecast.backtest.mase, 1.0)
        self.assertIs(forecast.backtest.beats_no_change, True)

    def test_an_unforecastable_series_admits_it_did_not_beat_no_change(self):
        """A random walk has no trend to extrapolate; the scaled error says so."""
        rng = np.random.default_rng(6)
        trend = make_trend(500 + np.cumsum(rng.normal(0, 60, 26)))

        forecast = build_forecast(trend)

        assert forecast is not None
        self.assertIsNotNone(forecast.backtest.mase)
        self.assertGreater(forecast.backtest.mase, 1.0)
        self.assertIs(forecast.backtest.beats_no_change, False)

    def test_seasonal_adjustments_do_not_shift_the_level(self):
        rng = np.random.default_rng(31)
        index = np.arange(36)
        values = 2_000 + 100 * np.sin(2 * np.pi * index / 12) + rng.normal(0, 5, 36)

        forecast = build_forecast(make_trend(values))

        assert forecast is not None
        self.assertIn("seasonality", forecast.method)
        self.assertAlmostEqual(float(np.mean(forecast.values)), 2_000, delta=90)

    def test_zero_valued_holdout_periods_are_reported_not_hidden(self):
        forecast = build_forecast(make_trend([40, 44, 39, 41, 45, 38, 42, 40, 43, 0, 0, 0]))

        assert forecast is not None
        self.assertEqual(forecast.backtest.periods_without_mape, 3)
        self.assertIsNone(forecast.backtest.mape)
        self.assertIsNotNone(forecast.backtest.mase)

    def test_the_error_note_states_the_no_change_comparison(self):
        rng = np.random.default_rng(15)
        trend = make_trend(1_000 + 40 * np.arange(24) + rng.normal(0, 20, 24))

        forecast = build_forecast(trend)

        assert forecast is not None
        note = describe_backtest(forecast.backtest)
        self.assertIn("held out the last", note)
        self.assertIn("better than assuming no change", note)
        self.assertNotIn("no better", note)

    def test_the_error_note_is_honest_when_there_is_no_backtest(self):
        self.assertEqual(
            describe_backtest(Backtest(mape=None, mase=None, holdout_periods=0, periods_without_mape=0)),
            "history is too thin for a backtest",
        )

    def test_gapped_periods_still_forecast_forward(self):
        trend = make_trend([100 + 5 * index for index in range(14)])
        trend = trend.drop(index=6).reset_index(drop=True)

        forecast = build_forecast(trend)

        self.assertIsNotNone(forecast)
        assert forecast is not None
        self.assertEqual(len(forecast.periods), 6)
        pairs = zip(forecast.periods, forecast.periods[1:], strict=False)
        self.assertTrue(all(later > earlier for earlier, later in pairs))
        self.assertGreater(forecast.periods[0], pd.Timestamp(trend.iloc[-1]["Period"]))


if __name__ == "__main__":
    unittest.main()
