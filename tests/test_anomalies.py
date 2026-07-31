import unittest

import numpy as np
import pandas as pd

from anomalies import CRITICAL_VALUES, FALSE_ALARM_RATE, critical_value, detect_anomalies
from business_insights import analyze_business
from formatting import format_period
from schema import detect_roles


def make_trend(values):
    periods = pd.date_range("2024-01-01", periods=len(values), freq="MS")
    return pd.DataFrame({"Period": periods, "Value": [float(value) for value in values]})


def make_gapped_trend(values, *, skip):
    """A monthly trend with the given period positions missing entirely."""
    trend = make_trend(values)
    return trend.drop(index=list(skip)).reset_index(drop=True)


class AnomalyDetectionTests(unittest.TestCase):
    def test_flags_an_injected_spike_with_expected_range(self):
        rng = np.random.default_rng(7)
        values = 100 + rng.normal(0, 3, 18)
        values[9] = 200

        anomalies = detect_anomalies(make_trend(values))

        self.assertEqual(len(anomalies), 1)
        spike = anomalies[0]
        self.assertEqual(spike.period, pd.Timestamp("2024-10-01"))
        self.assertEqual(spike.direction, "above")
        self.assertGreater(spike.severity, 3)
        self.assertGreater(spike.value, spike.expected_high)

    def test_flags_a_dip_below_the_band(self):
        rng = np.random.default_rng(11)
        values = 500 + rng.normal(0, 10, 16)
        values[12] = 180

        anomalies = detect_anomalies(make_trend(values))

        self.assertTrue(anomalies)
        self.assertEqual(anomalies[0].direction, "below")
        self.assertLess(anomalies[0].value, anomalies[0].expected_low)

    def test_steady_trend_produces_no_false_positives(self):
        values = [100 + 10 * index for index in range(14)]
        self.assertEqual(detect_anomalies(make_trend(values)), ())

    def test_trending_series_with_noise_keeps_edges_clean(self):
        """The largest draw here is 3.3 sigma: ordinary for 15 periods, not news."""
        rng = np.random.default_rng(3)
        values = np.arange(15) * 25 + 400 + rng.normal(0, 5, 15)
        self.assertEqual(detect_anomalies(make_trend(values)), ())

    def test_short_history_is_never_flagged(self):
        self.assertEqual(detect_anomalies(make_trend([1, 2, 300, 4, 5])), ())

    def test_severity_orders_multiple_anomalies(self):
        rng = np.random.default_rng(5)
        values = 100 + rng.normal(0, 2, 20)
        values[4] = 160
        values[15] = 240

        anomalies = detect_anomalies(make_trend(values))

        self.assertEqual(len(anomalies), 2)
        self.assertGreater(anomalies[0].severity, anomalies[1].severity)
        self.assertEqual(anomalies[0].period, pd.Timestamp("2025-04-01"))

    def test_missing_periods_do_not_manufacture_anomalies(self):
        values = [100 + 10 * index for index in range(20)]

        self.assertEqual(detect_anomalies(make_gapped_trend(values, skip=(5, 6, 7, 8))), ())

    def test_a_gapped_history_still_finds_the_real_spike(self):
        rng = np.random.default_rng(19)
        values = 400 + 15 * np.arange(20) + rng.normal(0, 4, 20)
        values[17] = 1_500

        anomalies = detect_anomalies(make_gapped_trend(values, skip=(4, 5, 6)))

        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0].period, pd.Timestamp("2025-06-01"))
        self.assertEqual(anomalies[0].direction, "above")

    def test_expected_value_sits_inside_the_reported_range(self):
        rng = np.random.default_rng(23)
        values = 100 + rng.normal(0, 3, 18)
        values[9] = 200

        spike = detect_anomalies(make_trend(values))[0]

        self.assertLess(spike.expected_low, spike.expected)
        self.assertLess(spike.expected, spike.expected_high)
        self.assertAlmostEqual(spike.expected, (spike.expected_low + spike.expected_high) / 2)

    def test_an_explicit_threshold_overrides_the_calibration(self):
        rng = np.random.default_rng(3)
        values = np.arange(15) * 25 + 400 + rng.normal(0, 5, 15)

        self.assertTrue(detect_anomalies(make_trend(values), threshold=3.0))

    def test_period_formatting_matches_grain(self):
        period = pd.Timestamp("2025-03-08")
        self.assertEqual(format_period(period, "M"), "Mar 2025")
        self.assertEqual(format_period(period, "Q"), "Q1 2025")
        self.assertEqual(format_period(period, "W"), "08 Mar 2025")
        self.assertEqual(format_period(period, "Y"), "2025")


class CalibrationTests(unittest.TestCase):
    def test_short_histories_demand_a_wider_band_than_long_ones(self):
        self.assertGreater(critical_value(8), critical_value(20))
        self.assertGreater(critical_value(20), critical_value(60))

    def test_the_multiplier_is_interpolated_and_clamped(self):
        self.assertAlmostEqual(critical_value(8), CRITICAL_VALUES[0][1], places=2)
        self.assertLess(critical_value(13), critical_value(12))
        self.assertGreater(critical_value(13), critical_value(16))
        self.assertEqual(critical_value(2), critical_value(8))
        self.assertEqual(critical_value(10_000), critical_value(320))

    def test_stable_series_stay_below_the_promised_false_alarm_rate(self):
        """The calibration is a promise; re-measure it rather than trust it."""
        rng = np.random.default_rng(101)
        trials = 400
        flagged = 0
        for _ in range(trials):
            values = 1_000 + 8 * np.arange(24) + rng.normal(0, 45, 24)
            if detect_anomalies(make_trend(values)):
                flagged += 1

        observed = flagged / trials
        self.assertLess(observed, FALSE_ALARM_RATE * 2.5)

    def test_a_real_shock_is_still_caught_after_calibration(self):
        rng = np.random.default_rng(77)
        caught = 0
        for _ in range(60):
            values = 1_000 + rng.normal(0, 40, 24)
            values[14] += 400  # a ten-sigma operational event
            anomalies = detect_anomalies(make_trend(values))
            caught += any(item.period == pd.Timestamp("2025-03-01") for item in anomalies)

        self.assertGreater(caught, 57)


class AnomalyEvidenceTests(unittest.TestCase):
    def test_brief_surfaces_anomaly_evidence_and_action(self):
        rng = np.random.default_rng(2)
        dates = pd.date_range("2024-01-01", periods=18, freq="MS")
        revenue = 1_000 + rng.normal(0, 25, 18)
        revenue[8] = 5_000
        dataframe = pd.DataFrame(
            {
                "Date": dates,
                "Revenue": revenue,
                "Region": (["West", "East"] * 9),
            }
        )

        brief = analyze_business(dataframe, detect_roles(dataframe))

        anomaly = next((item for item in brief.evidence if item.kind == "anomaly"), None)
        self.assertIsNotNone(anomaly)
        assert anomaly is not None
        self.assertEqual(anomaly.tone, "warning")
        self.assertIn("Sep 2024", anomaly.statement)
        self.assertIn("trendline", anomaly.calculation)
        self.assertTrue(
            any("anomalous periods" in item.title.lower() for item in brief.recommendations)
        )


if __name__ == "__main__":
    unittest.main()
