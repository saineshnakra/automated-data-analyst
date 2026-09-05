import unittest

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from analysis import build_markdown_report, clean_dataframe, column_profile, generate_insights
from schema import DIMENSION_KEYWORDS, MEASURE_KEYWORDS, _keyword_score, detect_roles


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
    def test_a_date_column_is_found_whatever_it_is_called(self):
        """"Month", "Period", "FY" hold dates too, and gate the whole timeline."""
        frame = pd.DataFrame(
            {"Month": [f"1949-{month:02d}" for month in range(1, 13)], "Passengers": range(12)}
        )

        cleaned, report = clean_dataframe(frame)

        self.assertTrue(is_datetime64_any_dtype(cleaned["Month"]))
        self.assertEqual(report.datetime_columns_inferred, 1)

    def test_bare_years_stay_numbers(self):
        """1949 is a year, not the first of January 1949."""
        frame = pd.DataFrame({"Year": ["1949", "1950", "1951"], "Passengers": [1, 2, 3]})

        cleaned, _ = clean_dataframe(frame)

        self.assertFalse(is_datetime64_any_dtype(cleaned["Year"]))

    def test_text_categories_are_not_mistaken_for_dates(self):
        frame = pd.DataFrame(
            {"Container": ["Small Box", "Jumbo Drum", "Large Box"], "Sales": [1.0, 2.0, 3.0]}
        )

        cleaned, _ = clean_dataframe(frame)

        self.assertFalse(is_datetime64_any_dtype(cleaned["Container"]))



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
class KeywordScoreTests(unittest.TestCase):
    def test_the_last_word_decides_what_a_column_is(self):
        """"Product Container" is a container, not the product dimension."""
        self.assertGreater(
            _keyword_score("Product Category", DIMENSION_KEYWORDS),
            _keyword_score("Product Container", DIMENSION_KEYWORDS),
        )
        self.assertGreater(
            _keyword_score("Region", DIMENSION_KEYWORDS),
            _keyword_score("Product Container", DIMENSION_KEYWORDS),
        )

    def test_a_trailing_qualifier_does_not_erase_the_keyword(self):
        """"Revenue USD" is still revenue."""
        self.assertGreater(_keyword_score("Revenue USD", MEASURE_KEYWORDS), 0)
        self.assertEqual(
            _keyword_score("Revenue", MEASURE_KEYWORDS),
            max(MEASURE_KEYWORDS["revenue"], 0),
        )

    def test_the_real_segment_is_chosen_over_a_packaging_attribute(self):
        frame = pd.DataFrame(
            {
                "Sales": np.arange(12, dtype=float),
                "Product Container": ["Small Box", "Jumbo Drum", "Large Box"] * 4,
                "Region": ["West", "East", "North", "South"] * 3,
                "Customer Segment": ["Consumer", "Corporate"] * 6,
            }
        )

        self.assertNotEqual(detect_roles(frame).dimension, "Product Container")


if __name__ == "__main__":
    unittest.main()
