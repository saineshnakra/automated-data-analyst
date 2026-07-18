import unittest

import pandas as pd

from business_insights import (
    analyze_business,
    build_business_report,
    detect_roles,
    driver_frame,
    format_number,
    heatmap_frame,
    segment_frame,
    trend_frame,
)
from demo_data import make_demo_data


class RoleDetectionTests(unittest.TestCase):
    def test_detects_demo_business_schema(self):
        dataframe = make_demo_data(rows=400)

        roles = detect_roles(dataframe)

        self.assertEqual(roles.date, "Order Date")
        self.assertEqual(roles.measure, "Revenue")
        self.assertEqual(roles.dimension, "Product")
        self.assertEqual(roles.identifier, "Order ID")

    def test_prefers_business_metric_over_numeric_identifier(self):
        dataframe = pd.DataFrame(
            {
                "Customer ID": range(10_000, 10_020),
                "Revenue": range(100, 120),
                "Region": ["West", "East"] * 10,
            }
        )

        roles = detect_roles(dataframe)

        self.assertEqual(roles.measure, "Revenue")
        self.assertEqual(roles.identifier, "Customer ID")


class BusinessAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.dataframe = make_demo_data(rows=1_200)
        self.roles = detect_roles(self.dataframe)

    def test_builds_decision_ready_brief(self):
        brief = analyze_business(self.dataframe, self.roles)
        evidence_kinds = {item.kind for item in brief.evidence}

        self.assertEqual(len(brief.kpis), 4)
        self.assertIn("trend", evidence_kinds)
        self.assertIn("driver", evidence_kinds)
        self.assertIn("leader", evidence_kinds)
        self.assertIn("concentration", evidence_kinds)
        self.assertTrue(brief.recommendations)
        self.assertTrue(brief.headline)

    def test_trend_and_segment_frames_are_chart_ready(self):
        trend = trend_frame(self.dataframe, self.roles)
        segments = segment_frame(self.dataframe, self.roles)

        self.assertEqual(trend.columns.tolist(), ["Period", "Value"])
        self.assertGreater(len(trend), 2)
        self.assertEqual(segments.columns.tolist(), ["Segment", "Value"])
        self.assertEqual(segments.iloc[0]["Segment"], "Enterprise")

    def test_driver_frame_reconciles_with_the_net_movement(self):
        trend = trend_frame(self.dataframe, self.roles)
        net_movement = float(trend.iloc[-1]["Value"] - trend.iloc[-2]["Value"])

        drivers = driver_frame(self.dataframe, self.roles)

        self.assertEqual(drivers.columns.tolist(), ["Segment", "Change"])
        self.assertTrue(len(drivers) >= 2)
        self.assertAlmostEqual(float(drivers["Change"].sum()), net_movement, places=6)

    def test_driver_frame_needs_full_schema(self):
        roles = detect_roles(self.dataframe.drop(columns=["Order Date"]))
        self.assertTrue(driver_frame(self.dataframe.drop(columns=["Order Date"]), roles).empty)

    def test_heatmap_frame_is_segment_by_period(self):
        heat = heatmap_frame(self.dataframe, self.roles)

        self.assertFalse(heat.empty)
        self.assertLessEqual(len(heat), 8)
        self.assertTrue(all(isinstance(period, pd.Timestamp) for period in heat.columns))
        row_totals = heat.sum(axis=1)
        self.assertTrue(row_totals.is_monotonic_decreasing)
        self.assertEqual(str(row_totals.index[0]), "Enterprise")

    def test_heatmap_frame_without_dimension_is_empty(self):
        frame = self.dataframe[["Order Date", "Revenue"]].copy()
        self.assertTrue(heatmap_frame(frame, detect_roles(frame)).empty)

    def test_report_separates_facts_from_recommendations(self):
        brief = analyze_business(self.dataframe, self.roles)
        report = build_business_report(
            self.dataframe,
            brief,
            source_name="demo.csv",
            context="Operating data",
        )

        self.assertIn("## What the data says", report)
        self.assertIn("## What ADA recommends", report)
        self.assertIn("Calculation:", report)
        self.assertIn("not causal proof", report)

    def test_negative_latest_period_prioritizes_diagnosis(self):
        dataframe = pd.DataFrame(
            {
                "Date": pd.to_datetime(
                    ["2025-01-15", "2025-02-15", "2025-03-15", "2025-04-15", "2025-05-15"]
                ),
                "Revenue": [1000, 1100, 1200, 1300, 600],
                "Region": ["West", "West", "East", "East", "West"],
            }
        )

        brief = analyze_business(dataframe)

        self.assertEqual(brief.recommendations[0].title, "Start with East")
        self.assertIn("largest region driver", brief.recommendations[0].rationale)

    def test_business_number_formatting(self):
        self.assertEqual(format_number(1_250_000, "Revenue"), "$1.2M")
        self.assertEqual(format_number(12_000, "Units"), "12.0K")


if __name__ == "__main__":
    unittest.main()
