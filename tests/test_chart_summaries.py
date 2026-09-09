import unittest

import pandas as pd

from chart_summaries import (
    summarize_categories,
    summarize_distribution,
    summarize_heatmap,
    summarize_movement,
    summarize_relationship,
    summarize_trend,
)


class ChartSummaryTests(unittest.TestCase):

    def test_trend_summary(self):
        frame = pd.DataFrame({
            "Period": ["Jan", "Feb", "Mar"],
            "Value": [100, 120, 150],
        })

        summary = summarize_trend(frame, "Period", "Value")

        self.assertIn("increased", summary)
        self.assertIn("100.00", summary)
        self.assertIn("150.00", summary)
        self.assertIn("+50.0%", summary)

    def test_category_summary(self):
        frame = pd.DataFrame({
            "Segment": ["A", "B", "C"],
            "Value": [100, 300, 200],
        })

        summary = summarize_categories(frame, "Segment", "Value")

        self.assertIn("highest", summary)
        self.assertIn("B", summary)
        self.assertIn("300.00", summary)
        self.assertIn("lowest", summary)
        self.assertIn("A", summary)

    def test_distribution_summary(self):
        frame = pd.DataFrame({
            "Value": [10, 20, 30],
        })

        summary = summarize_distribution(frame, "Value")

        self.assertIn("3", summary)
        self.assertIn("20.00", summary)
        self.assertIn("10.00", summary)
        self.assertIn("30.00", summary)

    def test_relationship_summary(self):
        frame = pd.DataFrame({
            "X": [1, 2, 3],
            "Y": [2, 4, 6],
        })

        summary = summarize_relationship(frame, "X", "Y")

        self.assertIn("positive", summary)
        self.assertIn("1.00", summary)

    def test_heatmap_summary(self):
        frame = pd.DataFrame({
            "Segment": ["A", "A", "B", "B"],
            "Period": ["Jan", "Feb", "Jan", "Feb"],
            "Value": [10, 40, 20, 30],
        })

        summary = summarize_heatmap(
            frame,
            "Segment",
            "Period",
            "Value",
        )

        self.assertIn("highest", summary)
        self.assertIn("40.00", summary)
        self.assertIn("A", summary)
        self.assertIn("Feb", summary)

    def test_movement_summary(self):
        frame = pd.DataFrame({
            "Segment": ["A", "B", "C"],
            "Change": [50, -30, 10],
        })

        summary = summarize_movement(
            frame,
            "Segment",
            "Change",
        )

        self.assertIn("largest increase", summary)
        self.assertIn("A", summary)
        self.assertIn("largest decrease", summary)
        self.assertIn("B", summary)
        self.assertIn("net change", summary)
        self.assertIn("+30.00", summary)

    def test_empty_data_is_handled(self):
        frame = pd.DataFrame({
            "Period": [],
            "Value": [],
        })

        summary = summarize_trend(
            frame,
            "Period",
            "Value",
        )

        self.assertIn("No data", summary)

    def test_summary_does_not_invent_causes(self):
        frame = pd.DataFrame({
            "Period": ["Jan", "Feb"],
            "Value": [100, 150],
        })

        summary = summarize_trend(
            frame,
            "Period",
            "Value",
        )

        self.assertNotIn("because", summary.lower())
        self.assertNotIn("caused by", summary.lower())


if __name__ == "__main__":
    unittest.main()
