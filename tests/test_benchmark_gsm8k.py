"""How the GSM8K benchmark combines the model's own answer with ADA's (benchmarks/gsm8k.py)."""

import sys
import unittest
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from gsm8k import agree, combined, large_numbers  # noqa: E402


def made(own, ada, slip=False, checked=None):
    """The records of none, check and solve for one problem."""
    return {"none": {"given": own, "reply": f"Answer: {own}"},
            "check": {"given": checked or own, "reply": f"Answer: {checked or own}",
                      "checks": [{"correct": not slip}]},
            "solve": {"given": ada, "reply": "" if ada is None else f"Answer: {ada}",
                      "ada": {"equations": ["a = 2 + 3"], "target": "a", "value": ada}}}


class CombinedTests(unittest.TestCase):
    def decided(self, records, question="Tom has 3 apples.", tolerance=Fraction(0)):
        found = combined("", "", question, records, tolerance, ask=False)
        return found["reply"], found["decided_by"]

    def test_the_answer_both_give(self):
        self.assertEqual(self.decided(made("5", "5")), ("Answer: 5", "both agree"))
        # Within the data set's tolerance is the same answer.
        self.assertEqual(self.decided(made("5", "5.0004"), tolerance=Fraction(1, 1000))[1], "both agree")

    def test_no_answer_from_ada_leaves_the_checked_answer(self):
        self.assertEqual(self.decided(made("6", None, slip=True, checked="5")),
                         ("Answer: 5", "no answer from ADA"))

    def test_a_wrong_equation_in_the_models_answer_takes_adas(self):
        self.assertEqual(self.decided(made("6", "5", slip=True)),
                         ("Answer: 5", "a wrong equation in its answer"))

    def test_large_numbers_take_adas(self):
        self.assertEqual(self.decided(made("6", "5"), "Tom has 12,345 apples."),
                         ("Answer: 5", "large numbers"))

    def test_otherwise_the_models_own(self):
        self.assertEqual(self.decided(made("6", "5"), "Tom has 1,234 apples."),
                         ("Answer: 6", "the model's answer"))


class PartsTests(unittest.TestCase):
    def test_large_numbers(self):
        self.assertTrue(large_numbers("A farm sells 48,000 eggs."))
        self.assertTrue(large_numbers("It costs 12345.5 dollars."))
        self.assertFalse(large_numbers("It costs 1234.56789 dollars."))
        self.assertFalse(large_numbers("No numbers here."))
        self.assertFalse(large_numbers(""))

    def test_agree(self):
        self.assertTrue(agree("18", "18", Fraction(0)))
        self.assertFalse(agree("18", "19", Fraction(0)))
        self.assertFalse(agree(None, "18", Fraction(0)))


if __name__ == "__main__":
    unittest.main()
