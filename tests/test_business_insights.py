import unittest

import pandas as pd

from business_insights import (
    analyze_business,
    build_business_report,
    detect_roles,
    format_number,
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
