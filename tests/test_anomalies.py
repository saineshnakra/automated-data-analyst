import unittest

import numpy as np
import pandas as pd

from anomalies import detect_anomalies, format_period
from business_insights import analyze_business, detect_roles


def make_trend(values):
    periods = pd.date_range("2024-01-01", periods=len(values), freq="MS")
    return pd.DataFrame({"Period": periods, "Value": [float(value) for value in values]})


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

    def test_period_formatting_matches_grain(self):
        period = pd.Timestamp("2025-03-08")
        self.assertEqual(format_period(period, "M"), "Mar 2025")
        self.assertEqual(format_period(period, "Q"), "Q1 2025")
        self.assertEqual(format_period(period, "W"), "08 Mar 2025")
        self.assertEqual(format_period(period, "Y"), "2025")


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
