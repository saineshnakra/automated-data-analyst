"""Weak signals are shown, but never dressed up as findings."""

import unittest

import numpy as np
import pandas as pd

from business_insights import analyze_business
from schema import detect_roles


def monthly_frame(values, **columns):
    dates = pd.date_range("2023-01-01", periods=len(values), freq="MS")
    frame = pd.DataFrame({"Date": dates, "Revenue": [float(value) for value in values]})
    frame["Region"] = ["West", "East"] * (len(values) // 2) + ["West"] * (len(values) % 2)
    for name, series in columns.items():
        frame[name] = series
    return frame


def evidence_of(frame, kind):
    brief = analyze_business(frame, detect_roles(frame))
    return next((item for item in brief.evidence if item.kind == kind), None)


class MovementContextTests(unittest.TestCase):
    def test_a_routine_swing_is_labelled_as_routine(self):
        rng = np.random.default_rng(41)
        values = 1_000 * (1 + rng.normal(0, 0.25, 20))  # a habitually volatile metric

        trend = evidence_of(monthly_frame(values), "trend")

        assert trend is not None
        self.assertIn("within normal period-to-period variation", trend.statement)
        self.assertIn("typically moves about", trend.statement)

    def test_a_genuine_break_is_not_explained_away(self):
        values = [1_000, 1_010, 995, 1_005, 1_002, 998, 1_008, 1_001, 1_004, 300]

        trend = evidence_of(monthly_frame(values), "trend")

        assert trend is not None
        self.assertIn("unusually large swing", trend.statement)
        self.assertEqual(trend.tone, "negative")

    def test_the_qualifier_travels_into_the_recommendation(self):
        rng = np.random.default_rng(41)
        frame = monthly_frame(1_000 * (1 + rng.normal(0, 0.25, 20)))

        brief = analyze_business(frame, detect_roles(frame))
        rationales = " ".join(item.rationale for item in brief.recommendations)

        self.assertIn("within normal period-to-period variation", rationales)

    def test_too_little_history_makes_no_claim_either_way(self):
        trend = evidence_of(monthly_frame([100, 120, 90, 140]), "trend")

        assert trend is not None
        self.assertNotIn("typically moves", trend.statement)


class CorrelationEvidenceTests(unittest.TestCase):
    def test_a_solid_correlation_reports_its_interval_and_sample(self):
        rng = np.random.default_rng(3)
        revenue = rng.normal(1_000, 200, 300)
        frame = monthly_frame(revenue, Units=revenue * 0.01 + rng.normal(0, 1, 300))

        relationship = evidence_of(frame, "relationship")

        assert relationship is not None
        self.assertIn("95% CI", relationship.statement)
        self.assertIn("n = 300", relationship.statement)
        self.assertNotIn("cannot be told apart", relationship.statement)
        self.assertIn("not proof of causation", relationship.statement)

    def test_a_correlation_from_a_thin_sample_admits_it_proves_nothing(self):
        """r = 0.51 looks material until you see the interval runs from -0.06 to 0.83."""
        rng = np.random.default_rng(2)
        revenue = rng.normal(1_000, 200, 13)
        frame = monthly_frame(revenue, Units=revenue * 0.01 + rng.normal(0, 4.0, 13))

        relationship = evidence_of(frame, "relationship")

        assert relationship is not None
        self.assertIn("95% CI", relationship.statement)
        self.assertIn("cannot be told apart from no relationship", relationship.statement)

    def test_too_few_pairs_produces_no_relationship_card_at_all(self):
        rng = np.random.default_rng(2)
        revenue = rng.normal(1_000, 200, 8)
        frame = monthly_frame(revenue, Units=revenue * 0.01)

        self.assertIsNone(evidence_of(frame, "relationship"))

    def test_a_correlation_carried_by_outliers_says_so(self):
        rng = np.random.default_rng(5)
        revenue = rng.normal(500, 30, 60)
        units = rng.normal(50, 10, 60)
        revenue[0], units[0] = 40_000, 4_000  # one enormous deal invents a straight line
        revenue[1], units[1] = 35_000, 3_600
        frame = monthly_frame(revenue, Units=units)

        relationship = evidence_of(frame, "relationship")

        assert relationship is not None
        self.assertIn("rank correlation", relationship.statement)
        self.assertIn("few extreme records", relationship.statement)

    def test_a_clean_relationship_is_not_accused_of_being_outlier_driven(self):
        rng = np.random.default_rng(3)
        revenue = rng.normal(1_000, 200, 300)
        frame = monthly_frame(revenue, Units=revenue * 0.01 + rng.normal(0, 1, 300))

        relationship = evidence_of(frame, "relationship")

        assert relationship is not None
        self.assertNotIn("few extreme records", relationship.statement)


if __name__ == "__main__":
    unittest.main()
