"""Defects an adversarial review of this branch confirmed, each with its repro."""

import unittest

import numpy as np
import pandas as pd

from nlq import answer_question, parse_question
from schema import ColumnRoles


def roles(**kw):
    base = dict(date=None, measure=None, dimension=None, identifier=None, numeric=(), dimensions=())
    base.update(kw)
    return ColumnRoles(**base)


class GrowthDirectionTests(unittest.TestCase):
    """TOP_PATTERN claims "bottom 3" whole, so blanked() hides the word from
    ASCENDING_PATTERN. Only the growth branch lacked the top_match fallback."""

    def setUp(self):
        self.frame = pd.DataFrame({
            "Date": pd.to_datetime(["2025-01-05", "2025-02-05"] * 3),
            "Region": ["East", "East", "West", "West", "North", "North"],
            "Revenue": [500.0, 0.0, 100.0, 200.0, 300.0, 330.0],
        })
        self.roles = roles(date="Date", measure="Revenue", dimension="Region",
                           numeric=("Revenue",), dimensions=("Region",))

    def test_bottom_n_growth_ranks_ascending(self):
        plan = parse_question("bottom 3 regions by revenue growth", self.frame, self.roles)
        self.assertTrue(plan.ascending, "bottom asked for the slowest movers")

    def test_top_and_bottom_growth_are_not_the_same_plan(self):
        top = parse_question("top 3 regions by revenue growth", self.frame, self.roles)
        bottom = parse_question("bottom 3 regions by revenue growth", self.frame, self.roles)
        self.assertNotEqual(top.ascending, bottom.ascending)


class RateScaleTests(unittest.TestCase):
    """_shown says the column settles the scale once. _execute_growth was
    handing it the filtered frame, so a scoped question read 100x off."""

    def setUp(self):
        self.frame = pd.DataFrame({
            "Date": pd.to_datetime(["2025-01-05", "2025-02-05"] * 2),
            "Region": ["East", "East", "West", "West"],
            "Churn Rate": [0.5, 0.8, 2.0, 3.0],
        })
        self.roles = roles(date="Date", measure="Churn Rate", dimension="Region",
                           numeric=("Churn Rate",), dimensions=("Region",))

    def test_a_filter_does_not_change_the_rate_scale(self):
        whole = answer_question("how did churn rate change", self.frame, self.roles)
        scoped = answer_question("how did churn rate change in east", self.frame, self.roles)
        self.assertIn("0.5%", scoped.answer, f"East moved 0.5 -> 0.8, got: {scoped.answer}")
        self.assertIn("percentage points", whole.answer)

    def test_the_scoped_move_is_reported_in_points(self):
        scoped = answer_question("how did churn rate change in east", self.frame, self.roles)
        self.assertIn("0.3 percentage points", scoped.answer, scoped.answer)


class TrendFinitenessTests(unittest.TestCase):
    """A period whose every value is missing reached the arithmetic as NaN."""

    def test_a_missing_first_period_does_not_print_nan(self):
        frame = pd.DataFrame({
            "Date": pd.to_datetime(["2025-01-05", "2025-02-05", "2025-03-05"]),
            "Revenue": [np.nan, 100.0, 200.0],
        })
        answer = answer_question("revenue per month", frame,
                                 roles(date="Date", measure="Revenue", numeric=("Revenue",)))
        self.assertIsNotNone(answer)
        self.assertNotIn("nan", answer.answer.lower(), answer.answer)


class OverallIsNotASumTests(unittest.TestCase):
    """"overall" is a filler word, not an instruction to add a rate up."""

    def setUp(self):
        self.frame = pd.DataFrame({
            "Date": pd.to_datetime(["2025-01-05", "2025-02-05"] * 2),
            "Region": ["East", "East", "West", "West"],
            "Churn Rate": [0.5, 0.8, 2.0, 3.0],
        })
        self.roles = roles(date="Date", measure="Churn Rate", dimension="Region",
                           numeric=("Churn Rate",), dimensions=("Region",))

    def test_overall_does_not_sum_a_rate(self):
        answer = answer_question("overall churn rate", self.frame, self.roles)
        self.assertIsNotNone(answer)
        self.assertNotIn("sum(", answer.calculation, answer.calculation)

    def test_an_explicit_average_still_wins(self):
        answer = answer_question("average churn rate", self.frame, self.roles)
        self.assertIn("mean(", answer.calculation)

    def test_overall_still_totals_an_additive_measure(self):
        frame = self.frame.rename(columns={"Churn Rate": "Revenue"})
        answer = answer_question("overall revenue", frame,
                                 roles(date="Date", measure="Revenue", numeric=("Revenue",)))
        self.assertIn("sum(", answer.calculation)


if __name__ == "__main__":
    unittest.main()
