"""A question that names a dimension must not be answered as a grand total."""

import unittest

import pandas as pd

from demo_data import make_demo_data
from nlq import answer_question, parse_question
from pipeline import prepare_analysis


class MostNamesTheDimensionTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        prepared = prepare_analysis(make_demo_data(rows=900), row_limit=900)
        cls.frame, cls.roles = prepared.dataframe, prepared.detected_roles

    def test_most_ranks_the_named_dimension(self):
        """"least" already does this. "most" was answered as the whole file."""
        plan = parse_question("which region made the most revenue", self.frame, self.roles)
        self.assertIsNotNone(plan, "the question was refused outright")
        self.assertEqual(plan.intent, "rank")
        self.assertEqual(plan.dimension, "Region")
        self.assertFalse(plan.ascending)

    def test_most_and_least_are_symmetric(self):
        most = parse_question("which region made the most revenue", self.frame, self.roles)
        least = parse_question("which region made the least revenue", self.frame, self.roles)
        self.assertEqual(most.intent, least.intent)
        self.assertEqual(most.dimension, least.dimension)
        self.assertNotEqual(most.ascending, least.ascending)

    def test_has_is_an_ordinary_function_word(self):
        """"which X has the highest Y" is the most ordinary phrasing there is."""
        answer = answer_question("which region has the highest revenue", self.frame, self.roles)
        self.assertIsNotNone(answer)
        self.assertIn("Region", answer.calculation)

    def test_the_grand_total_is_still_the_grand_total(self):
        plan = parse_question("total revenue", self.frame, self.roles)
        self.assertEqual(plan.intent, "aggregate")
        self.assertIsNone(plan.dimension)


class RelativeMonthIsRefusedTests(unittest.TestCase):

    def setUp(self):
        frame = pd.DataFrame({
            "Order Date": pd.to_datetime(["2023-12-05", "2024-12-05", "2025-12-05"]),
            "Region": ["East"] * 3,
            "Revenue": [100.0, 100.0, 100.0],
        })
        prepared = prepare_analysis(frame, row_limit=900)
        self.frame, self.roles = prepared.dataframe, prepared.detected_roles

    def test_last_december_is_refused_rather_than_summed_across_years(self):
        """Three Decembers summed to 300 and the sentence said "in December"."""
        self.assertIsNone(answer_question("revenue last december", self.frame, self.roles))

    def test_next_december_is_refused(self):
        """A future period cannot be answered from past rows."""
        self.assertIsNone(answer_question("revenue next december", self.frame, self.roles))

    def test_a_plain_month_still_answers(self):
        answer = answer_question("revenue in december", self.frame, self.roles)
        self.assertIsNotNone(answer)


if __name__ == "__main__":
    unittest.main()
