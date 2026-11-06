"""How the GSM8K benchmark combines the model's own answer with ADA's (benchmarks/gsm8k.py)."""

import sys
import unittest
import unittest.mock
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import gsm8k  # noqa: E402
from gsm8k import agree, combined, formalized, large_numbers, programmed  # noqa: E402


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


class RetryTests(unittest.TestCase):
    def asked(self, *replies):
        """formalized under retry, with the model's replies in turn; the result and how often it was asked."""
        with unittest.mock.patch.object(gsm8k, "chat", side_effect=[{"content": r} for r in replies]) as chat:
            return formalized("", "", "Tom has 3 apples.", retry=True), chat.call_count

    def test_equations_with_a_value_are_not_asked_again(self):
        found, calls = self.asked("apples = 3 + 2\nAnswer: apples")
        self.assertEqual((found["reply"], calls), ("Answer: 5", 1))
        self.assertNotIn("first_ada", found)

    def test_a_number_instead_of_equations_is_asked_again(self):
        found, calls = self.asked("Answer: 6", "apples = 3 + 2\nAnswer: apples")
        self.assertEqual((found["reply"], calls), ("Answer: 5", 2))
        self.assertEqual(found["first_ada"]["status"], "number given")
        self.assertIn("a number you worked out", found["feedback"])

    def test_the_first_number_stands_when_the_second_reply_has_no_value(self):
        found, _ = self.asked("Answer: 6", "apples = 3 + pears\nAnswer: apples")
        self.assertEqual(found["reply"], "Answer: 6")
        self.assertTrue(found["kept_first"])
        self.assertEqual(found["second_ada"]["status"], "no value")

    def test_no_value_is_asked_again_as_before(self):
        found, _ = self.asked("apples = 3 + pears\nAnswer: apples", "apples = 3 + 2\nAnswer: apples")
        self.assertEqual(found["reply"], "Answer: 5")
        self.assertNotIn("kept_first", found)

    def test_thin_input(self):
        found, calls = self.asked("", "")
        self.assertEqual((found["reply"], calls), ("", 2))


class ProgramTests(unittest.TestCase):
    def test_the_program_answers(self):
        reply = "```python\ndef solution():\n    apples = 3 + 2\n    return apples\n```"
        with unittest.mock.patch.object(gsm8k, "chat", return_value={"content": reply}):
            found = programmed("", "", "Tom has 3 apples.")
        self.assertEqual(found["reply"], "Answer: 5")
        self.assertEqual(found["program"]["status"], "ran")
        with unittest.mock.patch.object(gsm8k, "chat", return_value={"content": "It is 5."}):
            self.assertEqual(programmed("", "", "Tom has 3 apples.")["reply"], "")


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
