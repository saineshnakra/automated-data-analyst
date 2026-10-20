"""formatting knows a rate moves in points; the chat sentences did not ask it."""

import unittest

import pandas as pd

from ada.nlq import answer_question
from ada.schema import ColumnRoles


def rate_frame():
    frame = pd.DataFrame({
        "Date": pd.to_datetime(["2025-01-05", "2025-02-05"] * 2),
        "Region": ["East", "East", "West", "West"],
        "Churn Rate": [2.0, 3.0, 4.0, 5.0],
    })
    roles = ColumnRoles(date="Date", measure="Churn Rate", dimension="Region", identifier=None,
                        numeric=("Churn Rate",), dimensions=("Region",))
    return frame, roles


class RateMovesInPointsTests(unittest.TestCase):

    def test_a_rate_growth_reports_points(self):
        """3% to 4% is one point, not "+33.3%"."""
        frame, roles = rate_frame()
        answer = answer_question("how did churn rate change", frame, roles)
        self.assertIn("percentage points", answer.answer, answer.answer)
        self.assertIn("1.0", answer.answer, answer.answer)

    def test_a_rate_trend_reports_points(self):
        frame, roles = rate_frame()
        answer = answer_question("churn rate per month", frame, roles)
        self.assertIn("percentage points", answer.answer, answer.answer)

    def test_the_two_surfaces_agree(self):
        frame, roles = rate_frame()
        for question in ("how did churn rate change", "churn rate per month"):
            with self.subTest(question=question):
                self.assertIn("percentage points", answer_question(question, frame, roles).answer)

    def test_an_additive_measure_still_reports_a_percentage(self):
        frame, roles = rate_frame()
        frame = frame.rename(columns={"Churn Rate": "Revenue"})
        roles = ColumnRoles(date="Date", measure="Revenue", dimension="Region", identifier=None,
                            numeric=("Revenue",), dimensions=("Region",))
        answer = answer_question("how did revenue change", frame, roles)
        self.assertIn("%", answer.answer)
        self.assertNotIn("percentage points", answer.answer)


if __name__ == "__main__":
    unittest.main()
