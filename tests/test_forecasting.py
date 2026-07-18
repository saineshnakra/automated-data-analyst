import unittest

import numpy as np
import pandas as pd

from forecasting import build_forecast


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
        self.assertAlmostEqual(forecast.backtest_mape or 0.0, 0.0, places=1)
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
        self.assertIsNotNone(forecast.backtest_mape)
        self.assertLess(forecast.backtest_mape, 15)

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
