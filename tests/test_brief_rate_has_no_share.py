"""A rate has no total, so no segment contributes a share of one."""

import unittest

import pandas as pd

from business_insights import analyze_business
from pipeline import prepare_analysis


def brief_for(measure: str):
    dates = pd.date_range("2024-01-01", periods=24, freq="MS")
    frame = pd.DataFrame({
        "Month": list(dates) * 2,
        "Team": ["A"] * 24 + ["B"] * 24,
        measure: [10.0] * 24 + [20.0] * 24,
    })
    prepared = prepare_analysis(frame, row_limit=5000)
    return analyze_business(prepared.dataframe, prepared.detected_roles)


class RateHasNoShareTests(unittest.TestCase):
    """A averages 10%, B averages 20%. "B contributes 66.7% of churn rate"
    is 20/(10+20): a share of a total that does not exist."""

    def setUp(self):
        self.brief = brief_for("Churn Rate")

    def leader(self):
        return next(e for e in self.brief.evidence if e.kind == "leader")

    def test_the_leader_card_quotes_no_share(self):
        leader = self.leader()
        self.assertNotIn("66.7%", leader.statement, leader.statement)
        self.assertNotIn("contributing", leader.statement, leader.statement)

    def test_the_leader_calculation_does_not_divide_by_a_total(self):
        """Saying "a rate has no total" is right; dividing by one is not."""
        self.assertNotIn("÷ total", self.leader().calculation, self.leader().calculation)

    def test_no_concentration_card_for_a_rate(self):
        """A Herfindahl index over shares needs shares."""
        kinds = [e.kind for e in self.brief.evidence]
        self.assertNotIn("concentration", kinds, kinds)

    def test_the_kpi_is_not_labelled_a_share(self):
        tops = [k for k in self.brief.kpis if k.label.startswith("Top ")]
        for kpi in tops:
            self.assertNotIn("Share", kpi.context, f"{kpi.label}: {kpi.context}")

    def test_the_leader_is_still_named(self):
        """Refusing the share must not lose the finding."""
        self.assertIn("B", self.leader().statement)
        self.assertIn("20", self.leader().statement)


class AdditiveMeasureKeepsItsShareTests(unittest.TestCase):

    def setUp(self):
        self.brief = brief_for("Revenue")

    def test_an_additive_leader_still_quotes_a_share(self):
        leader = next(e for e in self.brief.evidence if e.kind == "leader")
        self.assertIn("contributing", leader.statement, leader.statement)
        self.assertIn("66.7%", leader.statement, leader.statement)

    def test_an_additive_measure_keeps_its_concentration_card(self):
        self.assertIn("concentration", [e.kind for e in self.brief.evidence])


if __name__ == "__main__":
    unittest.main()
