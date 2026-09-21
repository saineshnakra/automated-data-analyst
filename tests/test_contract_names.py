"""Nothing ADA computes may depend on what the user called their columns.

Every intermediate column is private, and every display column is chosen
against the names actually present, so a dimension called "Latest", a
measure called "__period" or a metric called "None" is just a column.
"""

from __future__ import annotations

import unittest

import pandas as pd

from autovis import ChartSpec
from nlq import answer_question
from pipeline import NO_SELECTION, apply_role_selection
from schema import detect_roles
from ui import _explore_frame

ADVERSARIAL_NAMES = (
    "Latest",
    "Previous",
    "Change %",
    "Change (pp)",
    "Share %",
    "Rows",
    "Records",
    "Value",
    "Segment",
    "Period",
    "__period",
    "__segment",
    "__measure",
    "__value",
    "__0",
    "None",
)


def base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2024-01-01", "2024-02-01", "2024-01-01", "2024-02-01", "2024-01-01", "2024-02-01"]
            ),
            "Region": ["Alpha", "Alpha", "Beta", "Beta", "Gamma", "Gamma"],
            "Revenue": [100.0, 200.0, 100.0, 150.0, 50.0, 40.0],
        }
    )


class RenameInvarianceTests(unittest.TestCase):
    """The same data under adversarial headers gives the same answers."""

    def answers(self, frame: pd.DataFrame, dimension: str, measure: str) -> dict[str, tuple]:
        roles = detect_roles(frame)
        # Chat questions name the columns as the user did.
        growth = answer_question(f"which {dimension} grew fastest", frame, roles)
        breakdown = answer_question(f"{measure} by {dimension}", frame, roles)
        ranked = answer_question(f"top 2 {dimension} by {measure}", frame, roles)
        counted = answer_question(f"how many rows by {dimension}", frame, roles)
        return {
            "growth": (
                str(growth.table.iloc[0][dimension]),
                float(
                    growth.table.iloc[0][
                        [c for c in growth.table.columns if c.startswith("Change") and c != dimension][0]
                    ]
                ),
                growth.answer.split(" moved")[0],
            ),
            "breakdown": tuple(breakdown.table[dimension].tolist()),
            "ranked": (tuple(ranked.table[dimension].tolist()), ranked.answer.split(" is the")[0]),
            "counted": tuple(counted.table[dimension].tolist()),
        }

    def test_a_dimension_may_be_called_anything(self):
        expected = self.answers(base_frame(), "Region", "Revenue")
        self.assertEqual(expected["growth"][:2], ("Alpha", 100.0))
        for name in ADVERSARIAL_NAMES:
            with self.subTest(dimension=name):
                frame = base_frame().rename(columns={"Region": name})
                self.assertEqual(self.answers(frame, name, "Revenue"), expected)

    def test_a_measure_may_be_called_anything(self):
        expected = self.answers(base_frame(), "Region", "Revenue")
        for name in ("Period", "__period", "__measure", "Value", "Latest", "Rows", "None"):
            with self.subTest(measure=name):
                frame = base_frame().rename(columns={"Revenue": name})
                self.assertEqual(self.answers(frame, "Region", name), expected)

    def test_the_growth_table_never_overwrites_the_dimension(self):
        for name in ("Latest", "Previous", "Change %"):
            with self.subTest(dimension=name):
                frame = base_frame().rename(columns={"Region": name})
                result = answer_question(f"which {name} grew fastest", frame, detect_roles(frame))
                self.assertIn("Alpha moved fastest", result.answer)
                self.assertEqual(result.table[name].tolist()[0], "Alpha")
                self.assertEqual(len(set(result.table.columns)), len(result.table.columns))

    def test_a_share_column_never_overwrites_the_dimension(self):
        frame = base_frame().rename(columns={"Region": "Share %"})
        result = answer_question("revenue by share %", frame, detect_roles(frame))
        self.assertIn("Alpha is the leading", result.answer)
        self.assertEqual(result.table["Share %"].tolist()[0], "Alpha")

    def test_a_dimension_called_rows_survives_a_count(self):
        frame = base_frame().rename(columns={"Region": "Rows"})
        result = answer_question("how many rows by rows", frame, detect_roles(frame))
        self.assertIsNotNone(result)
        self.assertEqual(sorted(result.table["Rows"].tolist()), ["Alpha", "Beta", "Gamma"])


class ExploreNameTests(unittest.TestCase):
    def test_a_count_chart_finds_a_free_name_however_many_are_taken(self):
        # D-14: Records, Record count and Rows all present exhausted the
        # three-name generator and raised StopIteration.
        frame = pd.DataFrame(
            {
                "Records": ["a", "b", "a", "c"],
                "Record count": [1, 2, 3, 4],
                "Rows": [5, 6, 7, 8],
                "Records (2)": [9, 9, 9, 9],
            }
        )
        spec = ChartSpec(form="bar", rationale="", x="Records", aggregation="count")
        counted = _explore_frame(frame, spec)
        value = [column for column in counted.columns if column != "Records"][0]
        self.assertNotIn(value, frame.columns)
        self.assertEqual(dict(zip(counted["Records"], counted[value], strict=True)), {"a": 2, "b": 1, "c": 1})


class SelectionSentinelTests(unittest.TestCase):
    def test_a_column_called_none_can_be_the_metric(self):
        frame = base_frame().rename(columns={"Revenue": "None"})
        detected = detect_roles(frame)
        self.assertEqual(detected.measure, "None")
        chosen = apply_role_selection(detected, date="Date", measure="None", dimension="Region")
        self.assertEqual(chosen.measure, "None")
        cleared = apply_role_selection(detected, date="Date", measure=NO_SELECTION, dimension="Region")
        self.assertIsNone(cleared.measure)


if __name__ == "__main__":
    unittest.main()
