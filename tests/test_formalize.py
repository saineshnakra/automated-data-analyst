"""The benchmarks' solve condition: the model writes equations, ADA solves them (benchmarks/formalize.py)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from formalize import latex_value, lettered, plain_number, solved_by_ada, written  # noqa: E402


class WrittenTests(unittest.TestCase):
    def test_equations_and_the_answer_line(self):
        equations, target = written("- boxes = 3 * 12\n- total = boxes + 5\n**Answer:** total")
        self.assertEqual((equations, target), (["boxes = 3 * 12", "total = boxes + 5"], "total"))

    def test_what_is_not_an_equation_is_left_out(self):
        equations, _ = written("Let x be the cost.\nx > 2\nx == 3\nx = 5 (dollars).\ny <= 4")
        self.assertEqual(equations, ["x = 5"])

    def test_answer_lines_written_other_ways(self):
        self.assertEqual(written("Answer: \\boxed{x}")[1], "x")
        self.assertEqual(written("Answer: cost = price * count")[1], "cost")
        self.assertEqual(written("Answer: 2 * 7 = 14")[1], "14")
        self.assertIsNone(written("x = 2")[1])

    def test_thin_input(self):
        self.assertEqual(written(""), ([], None))
        self.assertEqual(written("Answer:"), ([], None))


class LetteredTests(unittest.TestCase):
    def test_names_get_letters_of_their_own(self):
        equations, target, mapping = lettered(["eggs_sold = 16 - 7", "money = eggs_sold * 2"], "money")
        self.assertEqual(mapping, {"eggs_sold": "a", "money": "b"})
        self.assertEqual((equations, target), (["a = 16 - 7", "b = a * 2"], "b"))

    def test_letters_already_used_and_functions_stay(self):
        equations, _, mapping = lettered(["a = sqrt(16) + pi", "total = a * comb(5, 2)"], "total")
        self.assertEqual(mapping, {"total": "b"})
        self.assertEqual(equations, ["a = sqrt(16) + pi", "b = a * comb(5, 2)"])

    def test_e_is_a_quantity_not_the_constant(self):
        equations, target, mapping = lettered(["e = 16 - 7"], "e")
        self.assertEqual((equations, target), (["a = 16 - 7"], "a"))


class SolvedTests(unittest.TestCase):
    def test_a_chain_of_steps(self):
        found = solved_by_ada("eggs_laid = 16\neggs_used = 3 + 4\neggs_sold = eggs_laid - eggs_used\n"
                              "money = eggs_sold * 2\nAnswer: money")
        self.assertEqual((found["status"], found["value"]), ("solved", "18"))

    def test_a_system_and_an_expression_to_report(self):
        found = solved_by_ada("x + y = 10\nx - y = 2\nAnswer: x * y")
        self.assertEqual((found["status"], found["value"]), ("solved", "24"))

    def test_an_expression_with_no_unknowns(self):
        self.assertEqual(solved_by_ada("Answer: (16 - 3 - 4) * 2")["value"], "18")
        # A bare number is the model's own answer, and is marked as such.
        self.assertEqual(solved_by_ada("Answer: 18")["status"], "number given")

    def test_several_values(self):
        self.assertEqual(solved_by_ada("x^2 = 4\nAnswer: x")["status"], "several values")
        # Counts and amounts are above zero: the one positive value is taken when asked.
        found = solved_by_ada("n * (n + 1) / 2 = 55\nAnswer: n", positive=True)
        self.assertEqual(found["value"], "10")

    def test_roots_and_pi_stay_exact(self):
        self.assertEqual(solved_by_ada("$$y = 2 \\sqrt{2}$$\n$$z = y^2$$\nAnswer: \\boxed{z}")["value"], "8")
        self.assertEqual(solved_by_ada("a = 3\nb = a*sqrt(3)\nAnswer: b")["value"], "3*sqrt(3)")
        self.assertEqual(solved_by_ada("area = pi * 3^2\nAnswer: area")["value"], "9*pi")

    def test_what_has_no_value(self):
        self.assertEqual(solved_by_ada("x = 2*y\nAnswer: x")["status"], "no value")
        self.assertEqual(solved_by_ada("x = 5\ny = x/0\nAnswer: y")["status"], "refused")
        self.assertEqual(solved_by_ada("x = 5")["status"], "no answer line")
        self.assertEqual(solved_by_ada("")["value"], None)


class ValueTests(unittest.TestCase):
    def test_plain_numbers_and_latex(self):
        self.assertEqual([plain_number(value) for value in ("18", "-2.25", "16/57", "3*sqrt(3)")],
                         ["18", "-2.25", "0.280701754386", "5.19615242271"])
        self.assertIsNone(plain_number("I"))
        self.assertEqual(latex_value("3*sqrt(3)"), "3 \\sqrt{3}")
        self.assertEqual(latex_value("16/57"), "\\frac{16}{57}")


if __name__ == "__main__":
    unittest.main()
