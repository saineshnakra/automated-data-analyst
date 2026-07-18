import unittest

import numpy as np
import pandas as pd

from analysis import build_markdown_report, clean_dataframe, column_profile, generate_insights


class CleanDataframeTests(unittest.TestCase):
    def test_cleaning_is_conservative_and_audited(self):
        raw = pd.DataFrame(
            {
                "Unnamed: 0": [0, 1, 2, 3],
                " amount ": [" 10.5 ", "20", "20", "20"],
                "Order Date": ["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-02"],
                "Region": [" West ", "East", "East", "East"],
                "Empty": [np.nan, np.nan, np.nan, np.nan],
            }
        )

        cleaned, report = clean_dataframe(raw)

        self.assertNotIn("Unnamed: 0", cleaned.columns)
        self.assertNotIn("Empty", cleaned.columns)
        self.assertEqual(cleaned.columns.tolist(), ["amount", "Order Date", "Region"])
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned["amount"]))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned["Order Date"]))
        self.assertEqual(cleaned.loc[0, "Region"], "West")
        self.assertEqual(report.index_columns_removed, 1)
        self.assertEqual(report.empty_columns_removed, 1)

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            clean_dataframe(pd.DataFrame())


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.dataframe = pd.DataFrame(
            {
                "sales": [10, 20, 30, 40, 1000],
                "units": [1, 2, 3, 4, 100],
                "region": ["West", "West", "West", "East", "West"],
            }
        )
        self.cleaned, self.report = clean_dataframe(self.dataframe)

    def test_profile_contains_quality_metrics(self):
        profile = column_profile(self.cleaned)
        self.assertEqual(profile["Column"].tolist(), ["sales", "units", "region"])
        self.assertIn("Missing %", profile.columns)

    def test_insights_are_computed_without_model_output(self):
        insights = generate_insights(self.cleaned)
        titles = {insight.title for insight in insights}
        self.assertIn("Strongest numeric relationship", titles)
        self.assertIn("Largest category share", titles)

    def test_markdown_report_states_local_processing(self):
        insights = generate_insights(self.cleaned)
        report = build_markdown_report(self.cleaned, self.report, insights, "Sales example")
        self.assertIn("# Automated Data Analysis Report", report)
        self.assertIn("No uploaded data was sent to an external AI service", report)


if __name__ == "__main__":
    unittest.main()
