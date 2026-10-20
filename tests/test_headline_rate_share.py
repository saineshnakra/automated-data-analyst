"""The headline must not wrap a rate's level in share language."""

import unittest

import pandas as pd

from ada.business_insights import analyze_business
from ada.pipeline import prepare_analysis


def brief_for(measure: str):
    dates = pd.date_range("2024-01-01", periods=24, freq="MS")
    frame = pd.DataFrame({
        "Month": list(dates) * 2,
        "Team": ["A"] * 24 + ["B"] * 24,
        measure: [10.0] * 24 + [20.0] * 24,
    })
    prepared = prepare_analysis(frame, row_limit=5000)
    return analyze_business(prepared.dataframe, prepared.detected_roles)


class HeadlineTests(unittest.TestCase):
    """B averages 20% churn. It does not "contribute" 20% of anything."""

    def test_a_rate_headline_does_not_say_coming_from(self):
        headline = brief_for("Churn Rate").headline
        self.assertNotIn("coming from", headline, headline)

    def test_a_rate_headline_still_names_the_leader_level(self):
        headline = brief_for("Churn Rate").headline
        self.assertIn("20.0%", headline, headline)
        self.assertIn("team", headline.lower(), headline)

    def test_an_additive_headline_keeps_its_share(self):
        headline = brief_for("Revenue").headline
        self.assertIn("coming from", headline, headline)
        self.assertIn("66.7%", headline, headline)


if __name__ == "__main__":
    unittest.main()
