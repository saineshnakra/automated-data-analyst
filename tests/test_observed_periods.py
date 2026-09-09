"""A period nobody measured is not a number the arithmetic can carry.

The trend frame holds an unobserved period as NaN, which is the honest value
and the right one for the chart. But `to_numpy(dtype=float)` hands that NaN to
Theil-Sen, one NaN makes the slope NaN, and every forecast value, band edge and
residual downstream is NaN as well. The forecast card printed "nan", the
backtest caption claimed the model "did not beat" a naive one it had never been
scored against, and anomaly detection returned nothing at all for a series with
an obvious spike in it.

The frames here are built by hand rather than through build_trend, because the
change that makes build_trend produce gaps is a separate pull request. What is
under test is what the two consumers do when a gap arrives.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from anomalies import detect_anomalies
from forecasting import build_forecast
from timeseries import observed_periods


def trend(values):
    return pd.DataFrame(
        {"Period": pd.date_range("2022-01-31", periods=len(values), freq="ME"), "Value": values}
    )


class ObservedPeriodsTests(unittest.TestCase):
    def test_a_frame_with_no_gaps_is_returned_untouched(self):
        frame = trend([1.0, 2.0, 3.0])
        self.assertIs(observed_periods(frame), frame)

    def test_gaps_are_dropped_and_the_rest_kept_in_order(self):
        frame = trend([1.0, np.nan, 3.0])
        kept = observed_periods(frame)
        self.assertEqual(kept["Value"].tolist(), [1.0, 3.0])
        self.assertTrue(kept["Period"].is_monotonic_increasing)

    def test_an_empty_or_shapeless_frame_is_survivable(self):
        self.assertTrue(observed_periods(pd.DataFrame()).empty)
        odd = pd.DataFrame({"Period": []})
        self.assertIs(observed_periods(odd), odd)


class ForecastTests(unittest.TestCase):
    def test_one_blank_month_does_not_make_the_whole_forecast_nan(self):
        values = [100.0 + 5 * i for i in range(24)]
        values[7] = np.nan
        forecast = build_forecast(trend(values))
        self.assertIsNotNone(forecast)
        for name in ("values", "lower", "upper"):
            numbers = np.asarray(getattr(forecast, name), dtype=float)
            self.assertFalse(np.isnan(numbers).any(), f"{name} carries NaN")

    def test_history_is_counted_in_periods_that_were_measured(self):
        # Eight rows, one of them blank, is seven observations - below the
        # minimum, and refusing is the honest answer rather than fitting a
        # line through a NaN.
        values = [100.0, np.nan] + [100.0 + i for i in range(6)]
        self.assertIsNone(build_forecast(trend(values), min_periods=8))


class AnomalyTests(unittest.TestCase):
    def test_a_spike_is_still_found_when_a_month_is_blank(self):
        values = [100.0] * 24
        values[5] = np.nan
        values[18] = 1000.0
        found = detect_anomalies(trend(values))
        self.assertTrue(found, "an obvious spike went unreported")
        self.assertEqual(
            pd.Timestamp(found[0].period).to_period("M"),
            pd.Timestamp("2023-07-31").to_period("M"),
        )

    def test_a_series_with_no_gaps_is_unaffected(self):
        values = [100.0] * 24
        values[18] = 1000.0
        self.assertTrue(detect_anomalies(trend(values)))


if __name__ == "__main__":
    unittest.main()
