"""A column's own name must not ask a different question from the reader's."""

import unittest

import pandas as pd

from ada.nlq import answer_question, parse_question
from ada.schema import ColumnRoles


def frame_with(measure: str) -> tuple[pd.DataFrame, ColumnRoles]:
    frame = pd.DataFrame({
        "Date": pd.to_datetime(["2025-01-05", "2025-02-05"] * 2),
        "Team": ["A", "A", "B", "B"],
        measure: [100.0, 200.0, 300.0, 400.0],
    })
    roles = ColumnRoles(date="Date", measure=measure, dimension="Team", identifier=None,
                        numeric=(measure,), dimensions=("Team",))
    return frame, roles


class ColumnNameDoesNotSetTheIntentTests(unittest.TestCase):

    def test_a_count_column_is_summed_not_counted(self):
        """"total cell count" was answered with "4 rows match"."""
        frame, roles = frame_with("Cell Count")
        answer = answer_question("total cell count", frame, roles)
        self.assertIsNotNone(answer)
        self.assertIn("sum(Cell Count)", answer.calculation, answer.calculation)
        self.assertIn("1.0K", answer.answer, answer.answer)

    def test_a_count_column_breaks_down_by_its_own_values(self):
        frame, roles = frame_with("Cell Count")
        answer = answer_question("cell count by team", frame, roles)
        self.assertIn("Cell Count", answer.calculation, answer.calculation)
        self.assertNotIn("row count", answer.calculation, answer.calculation)

    def test_an_explicit_average_beats_the_word_total_in_the_column_name(self):
        """"average total sales" was answered with sum(Total Sales)."""
        frame, roles = frame_with("Total Sales")
        answer = answer_question("average total sales", frame, roles)
        self.assertIn("mean(Total Sales)", answer.calculation, answer.calculation)

    def test_a_growth_column_still_totals_when_asked_for_a_total(self):
        """"total revenue growth" was answered as a period-over-period move."""
        frame, roles = frame_with("Revenue Growth")
        plan = parse_question("total revenue growth", frame, roles)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.intent, "aggregate", f"got {plan.intent}")

    def test_a_change_column_still_totals(self):
        frame, roles = frame_with("Change Score")
        plan = parse_question("total change score", frame, roles)
        self.assertEqual(plan.intent, "aggregate", f"got {plan.intent}")

    def test_the_reader_can_still_ask_for_each_intent(self):
        """Blanking the column name must not disarm the words the reader typed."""
        frame, roles = frame_with("Cell Count")
        self.assertEqual(parse_question("how many teams", frame, roles).intent, "count")
        self.assertEqual(parse_question("cell count per month", frame, roles).intent, "trend")
        self.assertEqual(
            parse_question("how did cell count change", frame, roles).intent, "growth"
        )
        self.assertEqual(parse_question("cell count by team", frame, roles).intent, "breakdown")

    def test_an_ordinary_measure_is_unaffected(self):
        frame, roles = frame_with("Revenue")
        self.assertIn("sum(Revenue)", answer_question("total revenue", frame, roles).calculation)
        self.assertIn("mean(Revenue)", answer_question("average revenue", frame, roles).calculation)


if __name__ == "__main__":
    unittest.main()
